import math
import numpy as np
import scipy.signal as signal

import matplotlib.pyplot as plt
import IPython.display as ipd


def plot_wave_spectrograms(waveforms, sample_rate, class_names, num_cols=2, specs=None):
    """
    Plots the spectrograms and waveforms of given audio waveforms using a shared sample rate.
    
    Args:
        waveforms (list of np.ndarray): List of audio waveforms (NumPy arrays of shape [samples, channels]).
        sample_rate (int): Common sample rate for all waveforms.
        class_names (list of str): List of corresponding class names.
        num_cols (int): Number of columns in the plot layout. Default is 2.
        specs (list of np.ndarray, optional): List of precomputed spectrograms (2D tensors, only `sxx` values).
                                              If None, spectrograms will be computed automatically.
    """
    num_files = len(waveforms)
    num_rows = math.ceil(num_files / num_cols) * 2  # Each audio takes 2 rows (spectrogram + waveform)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.6, num_rows * 2))

    if num_rows == 2:
        axs = np.reshape(axs, (num_rows, num_cols))  # Ensure correct indexing for small cases

    for idx, (waveform, class_name) in enumerate(zip(waveforms, class_names)):

        # Determine row and column indices
        i, j = (idx // num_cols) * 2, idx % num_cols  # Spectrogram in row i, waveform in i+1
        
        # Compute spectrogram if not provided
        if specs is None:
            sampleFreqs, segmentTimes, sxx = signal.spectrogram(waveform, sample_rate)

            # Plot spectrogram
            axs[i][j].pcolormesh(segmentTimes, sampleFreqs, 10 * np.log10(sxx + 1e-15))
            axs[i][j].set_title(f"{class_name}", fontsize=10)
            axs[i][j].set_axis_off()
        else:
            # Plot spectrogram
            axs[i][j].imshow(specs[idx])
            axs[i][j].set_title(f"{class_name}", fontsize=10)
            axs[i][j].set_axis_off()

        # Plot waveform
        axs[i + 1][j].plot(waveform)
        axs[i + 1][j].set_axis_off()

    plt.tight_layout()
    plt.show()

    # Play audio
    for waveform, class_name in zip(waveforms, class_names):
        print(f"Playing: {class_name}")
        ipd.display(ipd.Audio(waveform, rate=sample_rate))