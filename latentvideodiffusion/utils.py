import tqdm
import os
import pickle
import functools
import numpy
import cv2

def ckpt_path(ckpt_dir,iteration, ckpt_type):
    filename = f'checkpoint_{ckpt_type}_{iteration}.pkl'
    ckpt_path = os.path.join(ckpt_dir, filename)
    return ckpt_path 

def save_checkpoint(state, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

def save_checkpoint(state, filepath):
    directory = os.path.dirname(filepath)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

def show_samples(samples):
    y = jax.lax.clamp(0., x ,255.)
    frame = np.array(y.transpose(2,1,0),dtype=np.uint8)
    cv2.imshow('Random Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@functools.partial(jax.jit, static_argnums=(2, 3))
def update_state(state, data, optimizer, loss_fn):
    model, opt_state, key, i = state
    new_key, subkey = jax.random.split(key)
    
    loss, grads = jax.value_and_grad(loss_fn)(model, data, subkey)

    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    i = i+1
    
    new_state = new_model, new_opt_state, new_key, i
    
    return loss,new_state

@tqdm.tqdm
def tqdm_inf():
  while True:
    yield

def encode_frames(args, cfg):
    pass

def generate(args, cfg):
    pass

