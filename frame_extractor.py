import cv2
import jax
import numpy as np

class FrameExtractor:
    def __init__(self, video_path, batch_size, key):
        self.video_path = video_path
        self.batch_size = batch_size
        self.key = key

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self

    def __iter__(self):
        return self

    def __next__(self):
        self.key, idx_key = jax.random.split(self.key)
        idx_array = jax.random.randint(idx_key, (self.batch_size,), 0, self.total_frames)
        frames = []
        for idx in idx_array:
            # Set the frame position to the randomly selected index
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

            # Read the frame
            ret, frame = self.cap.read()

            if ret:
                # Convert the frame to RGB color space
                frame_rgb = frame#cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Append the frame to the list
                frames.append(frame_rgb)
        array = jax.numpy.array(frames)
        return array.transpose(0,3,2,1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

