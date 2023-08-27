import os

import numpy as np
import jax
import jax.numpy as jnp
import cv2

def encode(video_path, vae_encoder, batch_size, key):
    encoded_frames = []
    with frame_extractor.FrameExtractor(video_path, batch_size, key) as fe:
        for batch in fe:
            data = jnp.array(batch, dtype=jnp.float32)
            mean, _ = jax.vmap(vae_encoder)(data)
            encoded_frames.extend(mean)
    return jnp.array(encoded_frames)

def decode(encoded_frames, vae_decoder, key):
    decoded_frames = []
    for encoded_frame in encoded_frames:
        mean, _ = vae_decoder(encoded_frame)
        frame = jax.lax.clamp(0., mean, 255.)
        frame = np.array(frame.transpose(2, 1, 0), dtype=np.uint8)
        decoded_frames.append(frame)
    return decoded_frames

def save_video(frames, output_path, fps):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def load_video(video_path, batch_size=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if batch_size and len(frames) == batch_size:
            yield jnp.array(frames, dtype=jnp.float32)
            frames = []
    if frames:
        yield jnp.array(frames, dtype=jnp.float32)
    cap.release()

def save_latents(encoded_frames, file_path):
    with open(file_path, 'wb') as f:
        np.save(f, np.array(encoded_frames))

def load_latents(file_path):
    with open(file_path, 'rb') as f:
        return np.load(f)

class LatentDataset:
    def __init__(self, directory_path, batch_size, key):
        self.directory_path = directory_path
        self.latent_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]  # Assuming latents are saved as .npy files
        self.batch_size = batch_size
        self.key = key
        self.total_latents = sum(np.load(os.path.join(directory_path, f)).shape[0] for f in self.latent_files)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # No cleanup necessary for now

    def __iter__(self):
        return self

    def __next__(self):
        self.key, idx_key = jax.random.split(self.key)
        idx_array = jax.random.randint(idx_key, (self.batch_size,), 0, self.total_latents)
        latents = []

        for global_idx in idx_array:
            local_idx = int(global_idx)
            latent_idx = 0

            while local_idx >= np.load(os.path.join(self.directory_path, self.latent_files[latent_idx])).shape[0]:
                local_idx -= np.load(os.path.join(self.directory_path, self.latent_files[latent_idx])).shape[0]
                latent_idx += 1

            latent_file_path = os.path.join(self.directory_path, self.latent_files[latent_idx])
            all_latents = np.load(latent_file_path)
            latents.append(all_latents[local_idx])

        return jnp.array(latents)
 
