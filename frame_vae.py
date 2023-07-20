import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import cv2
import tqdm

import functools

import frame_extractor

class ConvResBlock(eqx.Module):
    inp_layer: eqx.nn.Conv
    outp_layer: eqx.nn.Conv
    layer_norm: eqx.nn.LayerNorm
    def __init__(self, n_latent, key):
        a,b = jax.random.split(key, 2) 
        padding = [(1,1),(1,1)]
        self.inp_layer = eqx.nn.Conv(num_spatial_dims=2, in_channels=n_latent, out_channels=n_latent*2, kernel_size=(3,3), stride=1,padding=padding,key=a)
        self.outp_layer = eqx.nn.Conv(num_spatial_dims=2, in_channels=n_latent*2, out_channels=n_latent, kernel_size=(3,3), stride=1,padding=padding,key=b)
        self.layer_norm = eqx.nn.LayerNorm(shape=None, elementwise_affine=False)
    
    def __call__(self, x):
        a = self.inp_layer(x)
        b = jax.nn.leaky_relu(a)
        c = self.outp_layer(b)
        d = c + x
        y = self.layer_norm(d)
        return y

class VAEEncoder(eqx.Module): 
    #Maps from image to latents
    conv_layers: list
    mean_output: eqx.nn.Linear 
    def __init__(self, n_latent, key):
        keys = jax.random.split(key, 12) 
        self.conv_layers = [
            eqx.nn.Conv(num_spatial_dims=2, in_channels=3, out_channels=8, kernel_size=(2,2), stride=2,key=keys[0]),
            ConvResBlock(8,key=keys[1]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=8, out_channels=16, kernel_size=(2,2), stride=2,key=keys[2], padding=[(0,0),(1,1)]),
            ConvResBlock(16,key=keys[3]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=16, out_channels=32, kernel_size=(2,2), stride=2,key=keys[4]),
            ConvResBlock(32,key=keys[5]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=32, out_channels=64, kernel_size=(2,2), stride=2,key=keys[6], padding=[(0,0),(1,1)]),
            ConvResBlock(64,key=keys[7]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=64, out_channels=128, kernel_size=(2,2), stride=2,key=keys[8]),
            ConvResBlock(128,key=keys[8]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=128, out_channels=256, kernel_size=(2,2), stride=2,key=keys[9]),
            ConvResBlock(256,key=keys[10])
        ]
        self.mean_output = eqx.nn.Linear(10240, n_latent, key=keys[11])

    def __call__(self,x):
        h = (x/256)-0.5
        for layer in self.conv_layers:
            h = layer(h)
        mean = self.mean_output(h.reshape(-1))
        log_var = jnp.zeros_like(mean)-3
        return mean, log_var

class VAEDecoder(eqx.Module): 
    #Maps from latents to images
    input_layer: eqx.nn.Linear
    conv_layers: list
    mean_output: eqx.nn.Conv
    log_var_output: eqx.nn.Conv
    def __init__(self, n_latent, key):
        keys = jax.random.split(key, 14) 
        self.input_layer = eqx.nn.Linear(n_latent, 10240,key=keys[11])
        self.conv_layers = [
            ConvResBlock(256,key=keys[10]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=256, out_channels=128, kernel_size=(2,2), stride=2,key=keys[9]),
            ConvResBlock(128,key=keys[8]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=128, out_channels=64, kernel_size=(2,2), stride=2,key=keys[8]),
            ConvResBlock(64,key=keys[7]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=64, out_channels=32, kernel_size=(2,2), stride=2,key=keys[6], padding=[(0,0),(1,1)]),
            ConvResBlock(32,key=keys[5]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=32, out_channels=16, kernel_size=(2,2), stride=2,key=keys[4]),
            ConvResBlock(16,key=keys[3]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=16, out_channels=8, kernel_size=(2,2), stride=2,key=keys[2], padding=[(0,0),(1,1)]),
            ConvResBlock(8,key=keys[1]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=8, out_channels=8, kernel_size=(2,2), stride=2,key=keys[0]),
        ]
        self.mean_output = eqx.nn.Conv(num_spatial_dims=2, in_channels=8, out_channels=3, kernel_size=(1,1), key=keys[12])
        self.log_var_output = eqx.nn.Conv(num_spatial_dims=2, in_channels=8, out_channels=3, kernel_size=(1,1), key=keys[13])

    def __call__(self,x):
        h = self.input_layer(x).reshape(256,8,5)
        for layer in self.conv_layers:
            h = layer(h)
        mean = (self.mean_output(h)+0.5)*128
        log_var = self.log_var_output(h)+2
        return mean, log_var

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
    return (mean,log_var)

@jax.jit
def vae_loss(vae, data, key):

    encoder, decoder = vae

    #Generate latent q distributions in z space
    q = jax.vmap(encoder)(data)

    #Sample Z values
    z = sample_gaussian(q, key)

    #Compute kl_loss terms
    z_prior = (0,0)
    kl = gaussian_kl_divergence(q,z_prior)

    #Ground truth predictions
    p = jax.vmap(decoder)(z)

    #Compute the probablity of the data given the latent sample
    log_p = gaussian_log_probabilty(p, data)

    #Maximise p assigned to data, minimize KL div
    loss = sum(map(jnp.sum,[-log_p, kl]))/(data.size)

    return loss

def make_vae(n_latent, key):
    enc_key, dec_key = jax.random.split(key)
    e = VAEEncoder(2048, enc_key)
    d = VAEDecoder(2048, dec_key)
    
    vae = e,d
    return vae

@functools.partial(jax.jit, static_argnums=(2, 3))
def update_state(state, data, optimizer, loss_fn):
    model, opt_state, key = state
    new_key, subkey = jax.random.split(key)
    
    loss, grads = jax.value_and_grad(loss_fn)(model, data, subkey)

    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    new_state = new_model, new_opt_state, new_key
    
    return loss,new_state

def sample_vae(n_latent, n_samples, vae, key):
    z_key, x_key = jax.random.split(key)
    decoder = vae[1]
    p_z = (jnp.zeros((n_samples,n_latent)),)*2
    z = sample_gaussian(p_z, z_key)
    p_x = jax.vmap(decoder)(z)
    x = sample_gaussian(p_x, x_key)
    return x

def show_sample(x):
    y = jax.lax.clamp(0., x ,255.)
    frame = np.array(y.transpose(2,1,0),dtype=np.uint8)
    cv2.imshow('Random Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def data_generator(key):
    while(True):
        yield jax.random.normal(key,(1,3,512,300))


def main():
    LR = 0.001
    N_LATENT = 2048
    VIDEO_PATH = "recording/video.avi"
    BS = 16

    key = jax.random.PRNGKey(42)
    data_key, init_key, state_key, sample_key = jax.random.split(key,4)
    
    vae = make_vae(N_LATENT, init_key)
    
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(vae)
    
    state = vae, opt_state, state_key

    dg = data_generator(data_key)
    
    with open("loss.txt","w") as f:
        with frame_extractor.FrameExtractor(VIDEO_PATH, BS, key) as fe:
            for i in tqdm.tqdm(range(5000)):
                data = jnp.array(next(fe),dtype=jnp.float32)
                loss,state = update_state(state, data, optimizer, vae_loss)
                f.write(f"{loss}\n")
                f.flush()
            
    
    trained_vae = state[0]

    samples = sample_vae(N_LATENT, 10, trained_vae, sample_key)
    for sample in samples:
        show_sample(sample)


if __name__ == "__main__":
    main()
