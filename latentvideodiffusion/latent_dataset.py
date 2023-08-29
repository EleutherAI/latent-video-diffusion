import os
import random
import pickle
import numpy as np
import jax.numpy as jnp


class LatentDataset:
    def __init__(self, data_directory, batch_size, prompt_length, completion_length):
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.prompt_length = prompt_length
        self.completion_length = completion_length
        self.segment_length = self.prompt_length + self.completion_length

    def __enter__(self):
        # Initialization code that will run when entering the "with" block
        self.file_list = [f for f in os.listdir(self.data_directory) if f.endswith('.pkl')]
        self.file_lengths = []
        for file in self.file_list:
            with open(os.path.join(self.data_directory, file), 'rb') as f:
                data_tuple = pickle.load(f)
            self.file_lengths.append(len(data_tuple[0]))
        total_length = sum(self.file_lengths)
        self.file_probabilities = [length / total_length for length in self.file_lengths]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleanup code that will run when exiting the "with" block
        # Currently, there's nothing specific to clean up in this example
        pass

    def load_random_file(self):
        # Randomly select a file based on weighted probabilities and load its data
        random_file = np.random.choice(self.file_list, p=self.file_probabilities)
        with open(os.path.join(self.data_directory, random_file), 'rb') as f:
            data_tuple = pickle.load(f)
        return data_tuple

    def get_random_segment(self, data_tuple):
        # Randomly select a start index for the segment
        array_length = len(data_tuple[0])
        start_idx = random.randint(0, array_length - self.segment_length)
        
        # Extract the segment from both arrays in the tuple
        segment_mean = data_tuple[0][start_idx:start_idx + self.segment_length]
        segment_log_var = data_tuple[1][start_idx:start_idx + self.segment_length]
        
        return segment_mean, segment_log_var

    def split_into_prompt_completion(self, segment_mean, segment_log_var):
        # Split segments into "prompt" and "completion"
        prompt_mean = segment_mean[:self.prompt_length]
        completion_mean = segment_mean[self.prompt_length:self.prompt_length + self.completion_length]
        
        prompt_log_var = segment_log_var[:self.prompt_length]
        completion_log_var = segment_log_var[self.prompt_length:self.prompt_length + self.completion_length]
        
        return (prompt_mean, prompt_log_var), (completion_mean, completion_log_var)

    def __iter__(self):
        return self

    def __next__(self):
        batch_prompts_mean, batch_completions_mean = [], []
        batch_prompts_log_var, batch_completions_log_var = [], []
        
        for _ in range(self.batch_size):
            data_tuple = self.load_random_file()
            segment_mean, segment_log_var = self.get_random_segment(data_tuple)
            (prompt_mean, prompt_log_var), (completion_mean, completion_log_var) = self.split_into_prompt_completion(segment_mean, segment_log_var)
            
            batch_prompts_mean.append(prompt_mean)
            batch_completions_mean.append(completion_mean)
            batch_prompts_log_var.append(prompt_log_var)
            batch_completions_log_var.append(completion_log_var)
        
        return (jnp.array(batch_prompts_mean), jnp.array(batch_prompts_log_var)), (jnp.array(batch_completions_mean), jnp.array(batch_completions_log_var))
