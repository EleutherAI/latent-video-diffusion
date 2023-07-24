import cv2
import jax
import numpy as np

class FrameExtractor:
    def __init__(self, video_path, segment_length, batch_size, key):
        self.video_path = video_path
        self.batch_size = batch_size
        self.segment_length = segment_length
        self.key = key

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self

    def __iter__(self):
        return self

    def __next__(self):
        self.key, idx_key = jax.random.split(self.key)
        idx_array = jax.random.randint(idx_key, (self.batch_size,), 0, self.total_frames-self.segment_length)
        segments = []
        for idx in idx_array:
            frames = []
            for frame_id in range(idx, idx+self.segment_length):
                # Set the frame position to the randomly selected index
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

                # Read the frame
                ret, frame = self.cap.read()

                if ret:
                    # Convert the frame to RGB color space
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Append the frame to the list
                    frames.append(frame_rgb)
            segment = jax.numpy.array(frames)
            segments.append(segment)
        array = jax.numpy.array(segments)
        return array.transpose(1,4,0,3,2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

