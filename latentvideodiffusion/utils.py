import json
import tqdm
import os
import pickle
import functools
import numpy
import cv2
import jax
import equinox as eqx
import numpy as np

def ckpt_path(ckpt_dir,iteration, ckpt_type):
    filename = f'checkpoint_{ckpt_type}_{iteration}.pkl'
    ckpt_path = os.path.join(ckpt_dir, filename)
    return ckpt_path 


def save_checkpoint(state, filepath):
    directory = os.path.dirname(filepath)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

def load_checkpoint(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' does not exist.")

    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    return state

def show_samples(samples):
    for x in samples:
        y = jax.lax.clamp(0., x ,255.)
        frame = np.array(y.transpose(2,1,0),dtype=np.uint8)
        cv2.imshow('Random Frame', frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_config(config_file_path):
    try:
        with open(config_file_path, 'r') as config_file:
            config_data = json.load(config_file)
        return config_data
    except FileNotFoundError:
        raise Exception("Config file not found.")
    except json.JSONDecodeError:
        raise Exception("Error decoding JSON in the config file.")

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

