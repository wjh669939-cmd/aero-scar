import torch
import random
import numpy as np
from scipy.signal import butter, lfilter


def filter_frequencies(data, lowpass_cutoff=40.0, highpass_cutoff=0.5):
    # Handle CUDA tensor: move to CPU and convert to numpy
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu().numpy()
        else:
            data = data.numpy()

    # Apply FFT on each sample in the data tensor
    fft_results = np.stack([apply_fft(sample) for sample in data])
    if random.random() < 0.5:
        # Apply low-pass filter to each sample
        filtered_results = np.stack(
            [lowpass_filter(sample, lowpass_cutoff, sampling_rate=128) for sample in fft_results]
        )
    else:
        # Apply high-pass filter to each sample
        filtered_results = np.stack(
            [highpass_filter(sample, highpass_cutoff, sampling_rate=128) for sample in fft_results]
        )
    return torch.from_numpy(filtered_results).float()


def apply_fft(sample):
    fft_result = np.fft.fft(sample)
    return fft_result


# Function to apply low-pass filter
def lowpass_filter(data, cutoff_frequency, sampling_rate):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(N=6, Wn=normal_cutoff, btype='low', analog=False)

    filtered_data = lfilter(b, a, data)

    return np.array(filtered_data, dtype=np.float32)


# Function to apply high-pass filter
def highpass_filter(data, cutoff_frequency, sampling_rate):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(N=6, Wn=normal_cutoff, btype='high', analog=False)

    filtered_data = lfilter(b, a, data)

    return np.array(filtered_data, dtype=np.float32)