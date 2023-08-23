import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import optax
import functools

N_ITER = 1000
LR = 0.0003
B1 = 0.9

N_LAYERS = 3
D_IO = 2 
D_L = 128 
D_MLP = 256
N_Q = 4
D_QK = 32
D_DV = 64

class LatentVideoTransformer(eqx.Module):
    layers: list
    x_embedding: eqx.nn.Linear
    y_embedding: eqx.nn.Linear
    gamma_embedding: eqx.nn.Linear
    unembedding: eqx.nn.Linear

    def __init__(self, key, n_layers, d_io, d_l, d_mlp, n_q, d_qk, d_dv):
        t_key, x_emb_key, y_emb_key, gamma_emb_key, unemb_key = jax.random.split(key, 5)
        t_keys = jax.random.split(t_key, n_layers)

        self.x_embedding = eqx.nn.Linear(d_io, d_l, key=x_emb_key)
        
        self.y_embedding = eqx.nn.Linear(d_io, d_l, key=y_emb_key)
        
        self.gamma_embedding = eqx.nn.Linear(1, d_l, key=gamma_emb_key)
        
        self.unembedding = eqx.nn.Linear(d_l, d_io, key=unemb_key)
        
        self.layers = []
        for i in range(n_layers):
            tb = TransformerBlock(t_keys[i], d_l, d_mlp, n_q, d_qk, d_dv, float(n_layers))
            self.layers.append(tb)
    
    def __call__(self, x, y, neg_gamma):
        """
            x: lx x d_io
            y: ly x d_io
            neg_gamma: ()
        """
        x_emb = jax.vmap(self.x_embedding)(x)
        y_emb = jax.vmap(self.y_embedding)(y)
        gamma_emb = self.gamma_embedding(neg_gamma.reshape((1,)))

        y_len = y.shape[0]

        h = jnp.concatenate((x_emb, y_emb + gamma_emb))
        for layer in self.layers:
            h = layer(h)

        output = jax.vmap(self.unembedding)(h[-y_len:]) #TODO: Add Analytic correction for unit gaussian
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
        self.theta = freq_base**(jnp.arange(0,d_qk//2)*2/d_qk)

    def __call__(self, x, y):
        return multi_query_cross_attention(x, y, self.q, self.k, self.v, self.o, self.theta)

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
        return y

class TransformerBlock(eqx.Module):
    mlp: MLP
    attention: CrossAttentionBlock
    rescale: jnp.float32
    
    def __init__(self, key, d, d_mlp, n_q, d_qk, d_dv, layer_rescale):
        mlp_key, cross_key = jax.random.split(key, 2)
        self.mlp = MLP(mlp_key, d, d_mlp)
        self.attention = CrossAttentionBlock(cross_key, d, d, n_q, d_qk, d_dv)
        self.rescale = layer_rescale

    def __call__(self, x):
        x_norm = self._layer_norm(x)
        x_mlp = jax.vmap(self.mlp)(x)
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
    output = jnp.einsum("ijk,ikl->jl", pre_output, o)

    return output

def single_head_attention(q, k, v, theta, x, y):

    #[ly x d_dy] x [d_dy x d_qk] -> [ly x d_qk]
    qs = jnp.einsum("ij,jk->ik", y, q)
    
    #[lx x d_dx] x [d_dx x d_qk] -> [lx x d_qk]
    ks = jnp.einsum("ij,jk->ik", x, k)
    
    #[lx x d_dx] x [d_dx x d_dv] -> [lx x d_dv]
    vs = jnp.einsum("ij,jk->ik", x, v)

    #[ly x lx]
    attention_matrix = rotary_attention(qs, ks, theta)

    #[ly x lx] x [lx x d_dv] -> [ly x d_dv]
    output = jnp.einsum("ij,jk->ik", attention_matrix, vs)

    return output

def rotary_attention(qs, ks, theta):
    """
        qs: [ly x d_qk] 
        ks: [lx x d_qk] 
        theta: (d_qk//2)
    """
    
    m_sin = jax.vmap(jax.vmap(jnp.sin))
    m_cos = jax.vmap(jax.vmap(jnp.sin))

    def complex_mul(x, y):
        a,b = x
        c,d = y

        r = a*c-b*d
        i = a*d+b*c

        z = r,i
        return  z

    def complex_inner_product(x,y):
        a,b = x
        c,d = y
        
        ac,ad,bd,bc = map(lambda x: jnp.einsum("ik,jk->ij",x[0],x[1]),[(a,c),(a,d),(b,d),(b,c)])
        
        r = ac+bd
        i = bc-ad 

        return r,i

    ly, lx, d_qk = qs.shape[0], ks.shape[0], ks.shape[1]

    #[ly], [lx]
    idx_y, idx_x = jnp.arange(ly), jnp.arange(lx)
    
    # [ly] x [(d_qk//2)] -> [ly x (d_qk//2)]
    q_theta = jnp.outer(idx_y, theta)
    
    # [lx] x [(d_qk//2)] -> [lx x (d_qk//2)]
    k_theta = jnp.outer(idx_x, theta)
    
    # [ly x (d_qk//2)],[ly x (d_qk//2)]
    qs_r, qs_i = qs[:,0:d_qk//2], qs[:,d_qk//2:]

    # [lx x (d_qk//2)],[lx x (d_qk//2)]
    ks_r, ks_i = ks[:,0:d_qk//2], ks[:,d_qk//2:]
    
    #[ly x (d_qk//2)],[ly x (d_qk//2)]
    a =  complex_mul((qs_r, qs_i),(m_cos(q_theta),m_sin(q_theta)))
    
    #[lx x (d_qk//2)],[lx x (d_qk//2)]
    b =  complex_mul((ks_r, ks_i),(m_cos(q_theta),m_sin(q_theta)))
    
    # ([ly x lx],[ly x lx])
    c = complex_inner_product(a,b)
    
    #[ly x lx]
    logits = (c[0]+c[1])/jnp.sqrt(d_qk)

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
    x_data, y_data = data
    assert x_data.shape[0] == y_data.shape[0]
    
    batch_size = y_data.shape[0]
    
    keys = jax.random.split(key, batch_size)

    def _diffusion_loss(model, f_neg_gamma, x_data, y_data, key):

        t_key, noise_key = jax.random.split(key,2)
        
        t = jax.random.uniform(t_key)
        
        neg_gamma, neg_gamma_prime = jax.value_and_grad(f_neg_gamma)(t)

        alpha, sigma = jnp.sqrt(alpha_squared(neg_gamma)), jnp.sqrt(sigma_squared(neg_gamma))

        epsilon = jax.random.normal(key, shape = y_data.shape)

        z = y_data*alpha + sigma*epsilon

        epsilon_hat = model(x_data, z, neg_gamma)

        loss = -1/2*neg_gamma_prime*(epsilon_hat-epsilon)**2

        return jnp.sum(loss)

    losses = jax.vmap(lambda x, y, z: _diffusion_loss(model, f_neg_gamma, x, y, z))(x_data, y_data, keys)
    mean_loss = jnp.sum(losses)/y_data.size

    return mean_loss

def sample_diffusion(inputs, model, f_neg_gamma, key, n_steps, shape):
    #Following https://arxiv.org/abs/2202.00512 eq. #8
    time_steps = jnp.linspace(0, 1, num=n_steps+1)

    n_samples = inputs.shape[0]

    z = jax.random.normal(key, (n_samples,) + shape)
    for i in range(n_steps):
        # t_s < t_t
        t_s, t_t = time_steps[n_steps-i-1], time_steps[n_steps-i]

        neg_gamma_s, neg_gamma_t = f_neg_gamma(t_s), f_neg_gamma(t_t)
        
        alpha_s = jnp.sqrt(alpha_squared(neg_gamma_s))
        alpha_t, sigma_t = jnp.sqrt(alpha_squared(neg_gamma_t)), jnp.sqrt(sigma_squared(neg_gamma_t))

        epsilon_hat = jax.vmap(lambda x, y: model(x, y, neg_gamma_t))(inputs, z)

        k = jnp.exp((neg_gamma_t-neg_gamma_s)/2)
        z = (alpha_s/alpha_t)*(z + sigma_t*epsilon_hat*(k-1))

    outputs = z

    return outputs

#@functools.partial(jax.jit, static_argnums=(2, 3))
def update_state(state, data, optimizer, loss_fn):
    model, opt_state, key = state
    new_key, subkey = jax.random.split(key)
    
    loss, grads = jax.value_and_grad(loss_fn)(model, data, subkey)

    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    new_state = new_model, new_opt_state, new_key
    
    return loss,new_state

def toy_data():
    key = jax.random.PRNGKey(42)
    x_noise = 0.05*jax.random.normal(key,shape=(128,50,2))
    y_noise = 0.05*jax.random.normal(key,shape=(128,50,2))
    x = jnp.array([[1,1],[-1,-1]]*25)+x_noise
    y = jnp.array([[1,1],[-1,-1]]*25)+y_noise
    return (x,y)

def main():
    data = toy_data()
    
    key = jax.random.PRNGKey(42)
    init_key, state_key, sample_key = jax.random.split(key,3)
    
    model = LatentVideoTransformer(init_key, N_LAYERS, D_IO, D_L, D_MLP, N_Q, D_QK, D_DV)

    optimizer = optax.adam(LR, b1=B1)
    opt_state = optimizer.init(model)

    state = model, opt_state, state_key
    
    #print(model(data[0][0],data[1][0],jnp.array(1)))

    loss_fn = lambda x,y,z: diffusion_loss(x, y, f_neg_gamma, z)

    for i in range(N_ITER):
        loss, state = update_state(state, data, optimizer, loss_fn)
        if i % 1 == 0:
            print(i,loss)

    trained_model = state[0]


    n_samples = 128 
    n_steps = 20 
    shape = (50,2)
    samples = sample_diffusion(data[0], trained_model, f_neg_gamma, sample_key, n_steps, shape)
    for i in range(2):
        plt.scatter(samples[:,i,0],samples[:,i,1])
        plt.scatter(data[1][:,i,0],data[1][:,i,1])
        plt.show()
    

if __name__ == "__main__":
    main()
