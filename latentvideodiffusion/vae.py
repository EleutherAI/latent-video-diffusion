import jax
import jax.numpy as jnp
import optax

import latentvideodiffusion as lvd
import latentvideodiffusion.models.frame_vae as frame_vae
import latentvideodiffusion.frame_extractor

#Gaussian VAE primitives
def gaussian_kl_divergence(p, q):
    p_mean, p_log_var = p
    q_mean, q_log_var = q

    kl_div = (q_log_var-p_log_var + (jnp.exp(p_log_var)+(p_mean-q_mean)**2)/jnp.exp(q_log_var)-1)/2
    return kl_div

def gaussian_log_probabilty(p, x):
    p_mean, p_log_var = p
    log_p = (-1/2)*((x-p_mean)**2/jnp.exp(p_log_var))-p_log_var/2-jnp.log(jnp.sqrt(2*jnp.pi))
    return log_p

def sample_gaussian(p, key):
    p_mean, p_log_var = p
    samples = jax.random.normal(key,shape=p_mean.shape)*jnp.exp(p_log_var/2)+p_mean
    return samples

def concat_probabilties(p_a, p_b):
    mean = jnp.concatenate([p_a[0],p_b[0]], axis=1)
    log_var = jnp.concatenate([p_a[1],p_b[1]], axis=1)
    return (mean, log_var)

@jax.jit
def vae_loss(vae, data, key):

    encoder, decoder = vae

    #Generate latent q distributions in z space
    q = jax.vmap(encoder)(data)

    #Sample Z values
    z = sample_gaussian(q, key)

    #Compute kl_loss terms
    z_prior = (0,0)
    kl = gaussian_kl_divergence(q, z_prior)

    #Ground truth predictions
    p = jax.vmap(decoder)(z)

    #Compute the probablity of the data given the latent sample
    log_p = gaussian_log_probabilty(p, data)

    #Maximise p assigned to data, minimize KL div
    loss = sum(map(jnp.sum,[-log_p, kl]))/(data.size)

    return loss

def make_vae(n_latent, size_multipier, key):
    enc_key, dec_key = jax.random.split(key)
    e = frame_vae.VAEEncoder(n_latent, size_multipier, enc_key)
    d = frame_vae.VAEDecoder(n_latent, size_multipier, dec_key)
    
    vae = e,d
    return vae

def sample_vae(n_latent, n_samples, vae, key):
    z_key, x_key = jax.random.split(key)
    decoder = vae[1]
    p_z = (jnp.zeros((n_samples,n_latent)),)*2
    z = sample_gaussian(p_z, z_key)
    p_x = jax.vmap(decoder)(z)
    x = sample_gaussian(p_x, x_key)
    return x

def show_samples(samples):
    y = jax.lax.clamp(0., x ,255.)
    frame = np.array(y.transpose(2,1,0),dtype=np.uint8)
    cv2.imshow('Random Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE model.')
    subparsers = parser.add_subparsers()
    
    #Training arguments
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('--checkpoint', type=int, default=None,
                        help='Checkpoint iteration to load state from.')
    
    #Sampling arguments
    sample_parser = subparsers.add_parser('sample')
    sample_parser.set_defaults(func=sample)
    sample_parser.add_argument('--checkpoint', type=int,
                        help='Checkpoint iteration to load state from.')
    
    sample_parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory')
    
    args = parser.parse_args()
    return args

def sample(args, cfg):
    n_samples = cfg["vae"]["sample"]["n_sample"]
    n_latent = cfg["lvm"]["n_latent"]

    state = lvd.utils.load_checkpoint(args.checkpoint)
    trained_vae = state[0]

    key = jax.random.PRNGKey(cfg["seed"])
    samples = sample_vae(n_latent, n_samples, trained_vae, key)
    lvd.utils.show_samples(samples)

def train(args, cfg):
    ckpt_dir = cfg["vae"]["train"]["ckpt_dir"]
    lr = cfg["vae"]["train"]["lr"]
    ckpt_interval = cfg["vae"]["train"]["ckpt_interval"]
    video_paths = cfg["vae"]["train"]["data_dir"]
    batch_size = cfg["vae"]["train"]["bs"]
    clip_norm = cfg["vae"]["train"]["clip_norm"]
    metrics_path = cfg["vae"]["train"]["metrics_path"]
    
    adam_optimizer = optax.adam(lr)
    optimizer = optax.chain(adam_optimizer, optax.zero_nans(), optax.clip_by_global_norm(clip_norm))
    
    if args.checkpoint is None:
        key = jax.random.PRNGKey(cfg["seed"])
        init_key, state_key = jax.random.split(key)
        vae = make_vae(cfg["lvm"]["n_latent"], cfg["vae"]["size_multiplier"], init_key)
        opt_state = optimizer.init(vae)
        i = 0
        state = vae, opt_state, state_key, i
    else:
        checkpoint_path = "vae"
        state = lvd.utils.load_checkpoint(checkpoint_path)
    
    with open(metrics_path,"w") as f:
        #TODO: Fix Frame extractor rng
        with lvd.frame_extractor.FrameExtractor(video_paths, batch_size, state[2]) as fe:
            while lvd.utils.tqdm_inf:
                data = jnp.array(next(fe),dtype=jnp.float32)
                loss, state = lvd.utils.update_state(state, data, optimizer, vae_loss)
                f.write(f"{loss}\n")
                f.flush()
                iteration = state[3]
                if (iteration % ckpt_interval) == (ckpt_interval - 1):
                    ckpt_path = lvd.utils.ckpt_path(ckpt_dir, iteration+1, "vae")
                    lvd.utils.save_checkpoint(state, ckpt_path)
