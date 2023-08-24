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

def save_frames(encoded_frames, file_path):
    with open(file_path, 'wb') as f:
        np.save(f, np.array(encoded_frames))

def load_frames(file_path):
    with open(file_path, 'rb') as f:
        return np.load(f)
