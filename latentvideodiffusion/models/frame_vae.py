import equinox as eqx
import jax
import jax.numpy as jnp

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
    def __init__(self, n_latent, k, key):
        keys = jax.random.split(key, 12) 
        self.conv_layers = [
            eqx.nn.Conv(num_spatial_dims=2, in_channels=3, out_channels=8*k, kernel_size=(2,2), stride=2,key=keys[0]),
            ConvResBlock(8*k,key=keys[1]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=8*k, out_channels=16*k, kernel_size=(2,2), stride=2,key=keys[2], padding=[(0,0),(1,1)]),
            ConvResBlock(16*k,key=keys[3]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=16*k, out_channels=32*k, kernel_size=(2,2), stride=2,key=keys[4]),
            ConvResBlock(32*k,key=keys[5]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=32*k, out_channels=64*k, kernel_size=(2,2), stride=2,key=keys[6], padding=[(0,0),(1,1)]),
            ConvResBlock(64*k,key=keys[7]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=64*k, out_channels=128*k, kernel_size=(2,2), stride=2,key=keys[8]),
            ConvResBlock(128*k,key=keys[8]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=128*k, out_channels=256*k, kernel_size=(2,2), stride=2,key=keys[9]),
            ConvResBlock(256*k,key=keys[10])
        ]
        self.mean_output = eqx.nn.Linear(10240*k, n_latent, key=keys[11])

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
    def __init__(self, n_latent, k, key):
        keys = jax.random.split(key, 14) 
        self.input_layer = eqx.nn.Linear(n_latent, 10240*k, key=keys[11])
        self.conv_layers = [
            ConvResBlock(256*k,key=keys[10]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=256*k, out_channels=128*k, kernel_size=(2,2), stride=2,key=keys[9]),
            ConvResBlock(128*k,key=keys[8]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=128*k, out_channels=64*k, kernel_size=(2,2), stride=2,key=keys[8]),
            ConvResBlock(64*k,key=keys[7]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=64*k, out_channels=32*k, kernel_size=(2,2), stride=2,key=keys[6], padding=[(0,0),(1,1)]),
            ConvResBlock(32*k,key=keys[5]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=32*k, out_channels=16*k, kernel_size=(2,2), stride=2,key=keys[4]),
            ConvResBlock(16*k,key=keys[3]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=16*k, out_channels=8*k, kernel_size=(2,2), stride=2,key=keys[2], padding=[(0,0),(1,1)]),
            ConvResBlock(8*k,key=keys[1]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=8*k, out_channels=8*k, kernel_size=(2,2), stride=2,key=keys[0]),
        ]
        self.mean_output = eqx.nn.Conv(num_spatial_dims=2, in_channels=8*k, out_channels=3, kernel_size=(1,1), key=keys[12])
        self.log_var_output = eqx.nn.Conv(num_spatial_dims=2, in_channels=8*k, out_channels=3, kernel_size=(1,1), key=keys[13])

    def __call__(self,x):
        h = self.input_layer(x).reshape(-1,8,5)
        for layer in self.conv_layers:
            h = layer(h)
        mean = (self.mean_output(h)+0.5)*128
        log_var = self.log_var_output(h)+3
        return mean, log_var
