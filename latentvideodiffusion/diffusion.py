import jax
import jax.numpy as jnp
import equinox as eqx

import latentvideodiffusion as lvd
import latentvideodiffusion.frame_transcode

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

def sample(args, cfg):
    n_samples = cfg["dt"]["sample"]["n_sample"]
    n_latent = cfg["lvm"]["n_latent"]

    vae_state = lvd.utils.load_checkpoint(args.vae_checkpoint)
    trained_vae = vae_state[0]
    m_encoder, m_decoder = map(lambda x: jax.vmap(jax.vmap(x)), trained_vae)
    
    vae_state = lvd.utils.load_checkpoint(args.diffusion_checkpoint)
    trained_dt = vae_state[0]

    key = jax.random.PRNGKey(cfg["seed"])

    dt_sample_key, encode_sample_key, decode_sample_key = jax.random.split(key, 3)

    prompt_frames = get_prompt_frames(args.vid_prompts)
    prompt_latents = m_encoder(prompt_frames)
    prompt_samples = lvd.vae.sample_gaussian(prompt_latents, encode_sample_key)

    latent_continuations = sample_diffusion(prompt_samples, trained_dt, f_neg_gamma, dt_sample_key, n_steps, shape)

    continuation_frames = m_decoder(latent_continuations)
    
    lvd.utils.show_samples(samples)

def train(args, cfg):
    key = jax.random.PRNGKey(cfg["seed"])
    ckpt_dir = cfg["dt"]["train"]["ckpt_dir"]
    lr = cfg["dt"]["train"]["lr"]
    ckpt_interval = cfg["dt"]["train"]["ckpt_interval"]
    latent_paths = cfg["dt"]["train"]["data_dir"]
    batch_size = cfg["dt"]["train"]["bs"]
    clip_norm = cfg["dt"]["train"]["clip_norm"]
    metrics_path = cfg["dt"]["train"]["metrics_path"]

    n_layers = cfg["dt"]["n_layers"]
    d_io = cfg["dt"]["d_io"]
    d_l = cfg["dt"]["d_mlp"]
    n_q = cfg["dt"]["n_q"]
    d_qk = cfg["dt"]["d_qk"]
    d_dv = cfg["dt"]["d_dv"]
    
    adam_optimizer = optax.adam(lr)
    optimizer = optax.chain(adam_optimizer, optax.zero_nans(), optax.clip_by_global_norm(clip_norm))
    
    if args.checkpoint is None:
        key = jax.random.PRNGKey(cfg["seed"])
        init_key, state_key = jax.random.split(key)
        model = LatentVideoTransformer(init_key, n_layers, d_io, d_l, d_mlp, n_q, d_qk, d_dv)
        opt_state = optimizer.init(model)
        i = 0
        state = vae, opt_state, state_key, i
    else:
        checkpoint_path = "dt"
        state = lvd.utils.load_checkpoint(checkpoint_path)
    
    with open(metrics_path,"w") as f:
        #TODO: Fix Frame extractor rng
        with lvd.frame_transcode.LatentDataset(video_paths, batch_size, state[2]) as ld:
            while lvd.utils.tqdm_inf:
                data = jnp.array(next(ld),dtype=jnp.float32)
                loss, state = lvd.utils.update_state(state, data, optimizer, vae_loss)
                f.write(f"{loss}\n")
                f.flush()
                iteration = state[3]
                if (iteration % ckpt_interval) == (ckpt_interval - 1):
                    ckpt_path = lvd.utils.ckpt_path(ckpt_dir, iteration+1, "vae")
                    lvd.utils.save_checkpoint(state, ckpt_path)

