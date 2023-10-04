import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def plot_loss(args, cfg):
    if args.type == "vae":
        metrics_path = cfg["vae"]["train"]["metrics_path"]
    elif args.type == "dt":
        metrics_path = cfg["dt"]["train"]["metrics_path"]
    else:
        print("Invalid type specified")
    plot_data_and_filtered(metrics_path)

def plot_data_and_filtered(file_path):
    # Read the data from the file
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Create a Butterworth low pass filter
    b, a = butter(3, 0.02, btype='low', analog=False)
    y = filtfilt(b, a, data)

    # Plot the original data and the filtered data on the same plot
    plt.figure(figsize=(14, 8))
    plt.loglog(data, 'b-', label='loss')
    plt.loglog(y, 'orange', linewidth=2, label='filtered loss')
    plt.title('Loss vs interation')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (nats/dim)')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
