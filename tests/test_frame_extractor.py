import pytest
from latentvideodiffusion.frame_extractor import extract_frames

def test_frame_extractor(directory_path, batch_size, key_seed):
    key = jax.random.PRNGKey(key_seed)
    with FrameExtractor(directory_path, batch_size, key) as extractor:
        # Iterate over the frame extractor and display the frames
        for batch in extractor:
            for i, frame in enumerate(batch):
                # Convert the frame to a format suitable for displaying with OpenCV
                frame_disp = np.array(frame.transpose(2, 1, 0))
                cv2.imshow(f'Frame {i}', frame_disp)

            # Wait for a key press and then close the windows
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break # Remove this line if you want to iterate over multiple batches

