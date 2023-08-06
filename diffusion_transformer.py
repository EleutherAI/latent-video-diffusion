import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import optax
import functools

N_ITER = 10000
LR = 0.001
B1 = 0.001

N_LAYERS = 5
D_IO = 4096
D_L = 8192
D_MLP = 256
N_Q = 4
D_QK = 128
D_DV = 128

class LatentVideoTransformer(eqx.Module):
    layers: list
    x_embedding: jnp.array
    y_embedding: jnp.array
    gamma_embedding: jnp.array
    unembedding: jnp.array

    def __init__(self, key, n_layers, d_io, d_l, d_mlp, n_q, d_qk, d_dv):
        
        t_key, x_emb_key, y_emb_key, gamma_emb_key, unemb_key = jax.random.split(key, 5)
        t_keys = jax.random.split(t_key, n_layers)

        self.x_embedding = jax.nn.initializers.glorot_normal()(
                x_emb_key, (d_io, d_l), dtype=jnp.float32)
        
        self.y_embedding = jax.nn.initializers.glorot_normal()(
                y_emb_key, (d_io, d_l), dtype=jnp.float32)
        
        self.gamma_embedding = jax.nn.initializers.glorot_normal()(
                gamma_emb_key, (d_io, d_l), dtype=jnp.float32)
        
        self.unembedding = jax.nn.initializers.glorot_normal()(
                unemb_key, (d_l, d_io), dtype=jnp.float32)
        
        self.layers = []
        for i in range(n_layers):
            tb = TransformerBlock(t_keys[i], d_l, d_mlp, n_q, d_qk, d_dv, 1/n_layers)
            self.layers.append(tb)
    
    def __call__(x, y, gamma):
        x_emb = jax.vmap(self.x_embedding)(x)
        y_emb = jax.vmap(self.y_embedding)(y)
        gamma_emb = jax.vmap(self.gamma_embedding)(gamma.reshape((1,)))

        y_len = y.shape(0)

        h = jnp.concatenate((x_emb, y_emb + gamma_emb[:,jnp.newaxis]))
        for layer in self.layers:
            h = layer(h)

        output = self.unembedding(h[-y_len:]) #TODO: Add Analytic correction
        return output

class CrossAttentionBlock(eqx.Module):
    q: jnp.array
    k: jnp.array
    v: jnp.array
    o: jnp.array
    theta: jnp.array
    
    def __init__(self, key, d_dx, d_dy, n_q, d_qk, d_dv, freq_base=10000):
        q_key, k_key, v_key, o_key = jax.random.split(key, 4)
        self.q = jax.nn.initializers.glorot_normal(
                in_axis=(1,), out_axis=(2,), batch_axis=(0,))(
                q_key, (n_q, d_dy, d_qk), dtype=jnp.float32)
        self.k = jax.nn.initializers.glorot_normal(
                in_axis=(0), out_axis=(1,), batch_axis=())(
                k_key, (d_dx, d_qk), dtype=jnp.float32)
        self.v = jax.nn.initializers.glorot_normal(
                in_axis=(0), out_axis=(1,), batch_axis=())(
                v_key, (d_dx, d_dv), dtype=jnp.float32)
        self.o = jax.nn.initializers.glorot_normal(
                in_axis=(0,1), out_axis=(2,), batch_axis=())(
                o_key, (n_q, d_dv, d_dy), dtype=jnp.float32)
        self.theta = freq_base**(jnp.arange(0,d_qk//2, 2)/d_qk)

    def __call__(x,y):
        return multi_query_cross_attention(x, y, self.q, self.k, self.theta)

class MLP(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, key, d_io, d_int):
        key_l1, key_l2 = jax.random.split(key)
        self.l1 = eqx.nn.Linear(d_io, d_int, key=key_l1)
        self.l2 = eqx.nn.Linear(d_int, d_io, key=key_l2)
    
    def __call__(self, x):
        h1 = self.l1(x)
        h2 = jax.nn.leaky_relu(h1)
        y = self.l2(h2)

class TransformerBlock(eqx.Module):
    mlp: MLP
    attention: CrossAttentionBlock
    rescale: jnp.float32
    
    def __init__(self, key, d, d_mlp, n_q, d_qk, d_dv, layer_rescale):
        mlp_key, cross_key = jax.random.split(key, 2)
        self.mlp = MLP(mlp_key, d, d_mlp)
        self.attention = CrossAttentionBlock(cross_key, d, d, n_q, d_qk, d_dv)
        self.rescale = layer_rescale

    def __call__():
        x_norm = self._layer_norm(x)
        x_mlp = self.mlp(x)
        x_attn = self.attention(x,x)
        y = x + (x_mlp+x_attn)/self.rescale
        return y

    def _layer_norm(self, x, eps=1e-7):
        f = lambda x: (x-jnp.mean(x))/jnp.sqrt(jnp.var(x)+eps)
        x = jax.vmap(f)(x)


def multi_query_cross_attention(x, y, q, k, v, o, theta):
    """
        x: lx x d_dx
        y: ly x d_dy

        q: n_q x d_dy x d_qk
        k: d_dx x d_qk
        v: d_dx x d_dv
        o: n_q x d_dv x d_dy
        
        theta: (d_qk//2)
    """
    #[n_q x ly x d_dv]
    pre_output = jax.vmap(lambda q: single_head_attention(q, k, v, theta, x, y))(q)

    #[ly x d_dy]
    output = jax.einsum("ijk,ikl->jl", pre_output, o)

    return output

def single_head_attention(q, k, v, theta, x, y):
    #[ly x d_dy] x [d_dy x d_qk] -> [ly x d_qk]
    qs = jnp.einsum("ij,jk->ik", y, q)
    
    #[lx x d_dx] x [d_dx x d_qk] -> [lx x d_qk]
    ks = jnp.einsum("ij,jk->ik", x, k)
    
    #[lx x d_dx] x [d_dx x d_dv] -> [lx x d_dv]
    vs = jnp.einsum("ij,jk->ik", x, y)

    #[ly x lx]
    attention_matrix = rotary_attention(qs, ks, theta)

    #[ly x lx] x [lx x d_dv] -> [ly x d_dv]
    output = jnp.einsum("ij,jk->ik", attention_matrix, vs)

    return output

def rotary_attention(qs, ks, theta):
    """
        qs: [ly x d_qk] 
        ks: [lx x d_qk] 
        theta: (d_kv//2)
    """

    m_sin = jax.vmap(jax.vmap(jnp.sin))
    m_cos = jax.vmap(jax.vmap(jnp.sin))

    def complex_mul(x, y):
        a,b = x
        c,d = y

        r = ac-bd
        i = ad+bc

        z = r,i
        return  z

    def complex_inner_product(x,y):
        a,b = x
        c,d = y
        ac,ad,bd,bc = map(lambda x: jnp.einsum("ij,jk->ik",x[0],x[1]),[(a,c),(a,d),(b,d),(b,c)])
        
        r = ac+bd
        i = bc-ad 

        return r,i

    ly, lx, d_kv = qs.shape[0], ks.shape[0], ks.shape[1]

    #[ly], [lx] -> [ly], [lx]
    idx_y, idx_x = jnp.arange(ly), jnp.arange(lx)
    # [ly] x [(d_kv//2)] -> [ly x (d_kv//2)]
    q_theta = jnp.outer(idy, theta)
    
    # [lx] x [(d_kv//2)] -> [lx x (d_kv//2)]
    k_theta = jnp.outer(idx, theta)
    
    # [ly x (d_kv//2)],[ly x (d_kv//2)]
    qs_r, qs_i = qs[0:d_kv//2], qs[d_kv//2:]

    # [lx x (d_kv//2)],[lx x (d_kv//2)]
    ks_r, ks_i = ks[0:d_kv//2], ks[d_kv//2:]
    
    #[ly x (d_kv//2)],[ly x (d_kv//2)]
    a =  complex_mul((qs_r, qs_i),(m_cos(q_theta),m_sin(q_theta)))
    
    #[lx x (d_kv//2)],[lx x (d_kv//2)]
    b =  complex_mul((ks_r, ks_i),(m_cos(q_theta),m_sin(q_theta)))
    
    # ([ly x lx],[ly x lx])
    c = complex_inner_product(a,b)
    
    #[ly x lx]
    logits = (c[0]+c[1])/jnp.sqrt(d_kv)

    #[ly x lx]
    attention_matrix = jax.nn.softmax(logits, axis=1)

    return attention_matrix

#Linear SNR Schedule
def f_neg_gamma(t, min_snr= -10, max_snr = 10):
    #equivalent to log SNR
    return max_snr - t*(max_snr - min_snr)

def sigma_squared(neg_gamma):
    return jax.nn.sigmoid(-neg_gamma)

def alpha_squared(neg_gamma):
    return jax.nn.sigmoid(neg_gamma)

def diffusion_loss(model, data, f_neg_gamma,  key):
    #As defined in https://arxiv.org/abs/2107.00630 eq. #17 
    
    batch_size = data.shape[0]
    
    keys = jax.random.split(key, batch_size)

    def _diffusion_loss(model, f_neg_gamma, data, key):
        t_key, noise_key = jax.random.split(key,2)
        
        t = jax.random.uniform(t_key)
        
        neg_gamma, neg_gamma_prime = jax.value_and_grad(f_neg_gamma)(t)

        alpha, sigma = jnp.sqrt(alpha_squared(neg_gamma)), jnp.sqrt(sigma_squared(neg_gamma))

        epsilon = jax.random.normal(key, shape = data.shape)

        z = data*alpha + sigma*epsilon

        epsilon_hat = model(z, neg_gamma)

        loss = -1/2*neg_gamma_prime*(epsilon_hat-epsilon)**2

        return jnp.sum(loss)

    losses = jax.vmap(lambda x, y: _diffusion_loss(model, f_neg_gamma, x, y))(data, keys)
    mean_loss = jnp.sum(losses)/data.size

    return mean_loss

def sample_diffusion(model, f_neg_gamma, key, n_steps, shape, n_samples):
    #Following https://arxiv.org/abs/2202.00512 eq. #8
    time_steps = jnp.linspace(0, 1, num=n_steps+1)

    z = jax.random.normal(key, (n_samples,) + shape)
    for i in range(n_steps):
        # t_s < t_t
        t_s, t_t = time_steps[n_steps-i-1], time_steps[n_steps-i]

        neg_gamma_s, neg_gamma_t = f_neg_gamma(t_s), f_neg_gamma(t_t)
        
        alpha_s = jnp.sqrt(alpha_squared(neg_gamma_s))
        alpha_t, sigma_t = jnp.sqrt(alpha_squared(neg_gamma_t)), jnp.sqrt(sigma_squared(neg_gamma_t))

        epsilon_hat = jax.vmap(lambda x: model(x, neg_gamma_t))(z)

        k = jnp.exp((neg_gamma_t-neg_gamma_s)/2)
        z = (alpha_s/alpha_t)*(z + sigma_t*epsilon_hat*(k-1))

    return z

@functools.partial(jax.jit, static_argnums=(2, 3))
def update_state(state, data, optimizer, loss_fn):
    model, opt_state, key = state
    new_key, subkey = jax.random.split(key)
    
    loss, grads = jax.value_and_grad(loss_fn)(model, data, subkey)

    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    new_state = new_model, new_opt_state, new_key
    
    return loss,new_state

def main():
    key = jax.random.PRNGKey(42)
    init_key, state_key = jax.random.split(key,2)
    
    model = LatentVideoTransformer(init_key, N_LAYERS, D_IO, D_L, D_MLP, N_Q, D_QK, D_DV)

    optimizer = optax.adam(LR, b1=B1)
    opt_state = optimizer.init(model)

    state = model, opt_state, state_key

    loss_fn = lambda x,y,z: diffusion_loss(x, y, f_neg_gamma, z)

    for i in range(N_ITER):
        loss, state = update_state(state, data, optimizer, loss_fn)
        if i % 1000 == 0:
            print(i,loss)

    trained_model = state[0]


    """
    t = jnp.array(0.35)
    gamma = f_neg_gamma(t)
    x,y = jnp.meshgrid(jnp.linspace(-6,6,15),jnp.linspace(-6,6,15))

    v_in = jnp.dstack([x,y])
    v_out = jax.vmap(jax.vmap(lambda x: trained_model(x, gamma)))(v_in)
    print(v_out)

    print(jax.vmap(jax.vmap(lambda x: model.idx(x, gamma)))(v_in))
    plt_val = v_out


    plt.quiver(x, y, plt_val[:,:,0], plt_val[:,:,1])
    plt.show()
    """


    """
    n_samples = 500
    n_steps = 100
    shape = (2,)
    samples = sample_diffusion(trained_model, f_neg_gamma, sample_key, n_steps, shape, n_samples)
    plt.scatter(samples[:,0],samples[:,1])
    plt.scatter(data[:,0],data[:,1])
    plt.show()
    """

if __name__ == "__main__":
    main()
