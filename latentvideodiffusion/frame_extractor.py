import cv2
import jax
import numpy as np
import os

class FrameExtractor:
    def __init__(self, directory_path, batch_size, key):
        self.directory_path = directory_path
        self.video_files = [f for f in os.listdir(directory_path) if f.endswith(('.mp4', '.avi'))] # Adjust as needed
        self.batch_size = batch_size
        self.key = key
        self.total_frames = sum(int(cv2.VideoCapture(os.path.join(directory_path, f)).get(cv2.CAP_PROP_FRAME_COUNT)) for f in self.video_files)
        self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        self.key, idx_key = jax.random.split(self.key)
        idx_array = jax.random.randint(idx_key, (self.batch_size,), 0, self.total_frames)
        frames = []
        for global_idx in idx_array:
            local_idx = int(global_idx)
            video_idx = 0
            while local_idx >= int(cv2.VideoCapture(os.path.join(self.directory_path, self.video_files[video_idx])).get(cv2.CAP_PROP_FRAME_COUNT)):
                local_idx -= int(cv2.VideoCapture(os.path.join(self.directory_path, self.video_files[video_idx])).get(cv2.CAP_PROP_FRAME_COUNT))
                video_idx += 1
            self.cap = cv2.VideoCapture(os.path.join(self.directory_path, self.video_files[video_idx]))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
            ret, frame = self.cap.read()
            self.cap.release()

            if ret:
                frames.append(frame)

        array = jax.numpy.array(frames)
        return array.transpose(0,3,2,1)
