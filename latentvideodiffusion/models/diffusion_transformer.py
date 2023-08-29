import jax
import jax.numpy as jnp
import equinox as eqx

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
