import jax
import jax.numpy as jnp
import equinox as eqx

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
