#!/usr/bin/env python
# Test script for time_domain.py

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from time_domain.time_domain_features import (
    extract_time_domain_features,
    extract_autoregressive_features,
    extract_hjorth_features,
    extract_inter_peak_features,
    extract_all_time_domain_features,
    extract_multichannel_time_domain_features
)


def generate_sample_eeg(fs=250, duration=5, num_channels=4):
    """
    Generate sample EEG data for testing.

    Parameters:
    -----------
    fs : float
        Sampling frequency in Hz
    duration : float
        Duration of the signal in seconds
    num_channels : int
        Number of EEG channels to generate

    Returns:
    --------
    numpy.ndarray
        Multi-channel EEG signals with shape (num_channels, n_samples)
    """
    t = np.arange(0, duration, 1 / fs)
    n_samples = len(t)

    # Initialize multi-channel EEG array
    eeg_signals = np.zeros((num_channels, n_samples))

    # Generate different signals for each channel
    for ch in range(num_channels):
        # Base signal: mixture of sine waves at different frequencies
        alpha = 3 * np.sin(2 * np.pi * 10 * t)  # 10 Hz - Alpha rhythm
        beta = 1.5 * np.sin(2 * np.pi * 20 * t)  # 20 Hz - Beta rhythm
        theta = 2 * np.sin(2 * np.pi * 5 * t)  # 5 Hz - Theta rhythm
        delta = 4 * np.sin(2 * np.pi * 2 * t)  # 2 Hz - Delta rhythm

        # Add noise
        noise = 1.5 * np.random.normal(0, 1, n_samples)

        # Mix components differently for each channel
        if ch == 0:
            eeg_signals[ch] = alpha + 0.5 * beta + 0.3 * theta + 0.2 * delta + noise
        elif ch == 1:
            eeg_signals[ch] = 0.3 * alpha + beta + 0.5 * theta + 0.1 * delta + noise
        elif ch == 2:
            eeg_signals[ch] = 0.2 * alpha + 0.3 * beta + theta + 0.5 * delta + noise
        else:
            eeg_signals[ch] = 0.1 * alpha + 0.2 * beta + 0.4 * theta + delta + noise

        # Add some artifacts to make it more realistic
        if ch == 0:
            # Add blink artifact
            blink_idx = np.random.randint(fs, n_samples - fs)
            blink = 40 * signal.gaussian(fs // 2, std=fs // 10)
            eeg_signals[ch, blink_idx:blink_idx + len(blink)] += blink

    return eeg_signals


def try_loading_real_eeg(directory='.'):
    """
    Try to load real EEG data from files in the specified directory.

    Returns:
    --------
    tuple
        (eeg_data, fs) if files found, else (None, None)
    """
    # Look for common EEG file extensions
    eeg_extensions = ['.edf', '.bdf', '.gdf', '.csv', '.txt', '.npy']

    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in eeg_extensions:
            filepath = os.path.join(directory, filename)

            try:
                # Try to determine the file type and load accordingly
                if ext == '.npy':
                    # Numpy array
                    eeg_data = np.load(filepath)
                    fs = 250  # Assume default fs
                    print(f"Loaded NumPy file: {filename}")
                    return eeg_data, fs

                elif ext == '.csv' or ext == '.txt':
                    # CSV/TXT file - assume simple format with one column per channel
                    eeg_data = np.loadtxt(filepath, delimiter=',').T
                    fs = 250  # Assume default fs
                    print(f"Loaded CSV/TXT file: {filename}")
                    return eeg_data, fs

                elif ext in ['.edf', '.bdf', '.gdf']:
                    # Try to use MNE-Python if available
                    try:
                        import mne
                        raw = mne.io.read_raw(filepath, preload=True)
                        eeg_data = raw.get_data()
                        fs = raw.info['sfreq']
                        print(f"Loaded {ext} file: {filename} using MNE")
                        return eeg_data, fs
                    except ImportError:
                        print("MNE-Python not found, skipping EDF/BDF/GDF file.")
                        continue

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    print("No valid EEG files found, using generated data.")
    return None, None


def test_individual_functions(eeg_signal, fs):
    """Test each function individually with a single EEG channel"""

    print("\n1. Testing extract_time_domain_features:")
    time_features = extract_time_domain_features(eeg_signal, fs)
    print("Number of time domain features:", len(time_features))
    for key, value in list(time_features.items())[:5]:  # Print first 5 features
        print(f"  {key}: {value:.4f}")

    print("\n2. Testing extract_autoregressive_features:")
    ar_features = extract_autoregressive_features(eeg_signal, order=6)
    print("Number of AR features:", len(ar_features))
    for key, value in ar_features.items():
        print(f"  {key}: {value:.4f}")

    print("\n3. Testing extract_hjorth_features:")
    hjorth_features = extract_hjorth_features(eeg_signal)
    print("Number of Hjorth features:", len(hjorth_features))
    for key, value in hjorth_features.items():
        print(f"  {key}: {value:.4f}")

    print("\n4. Testing extract_inter_peak_features:")
    peak_features = extract_inter_peak_features(eeg_signal, fs)
    print("Number of peak features:", len(peak_features))
    for key, value in list(peak_features.items())[:5]:  # Print first 5 features
        print(f"  {key}: {value:.4f}")

    print("\n5. Testing extract_all_time_domain_features:")
    all_features = extract_all_time_domain_features(eeg_signal, fs, ar_order=6)
    print("Number of all features combined:", len(all_features))
    print("Feature names:", ", ".join(list(all_features.keys())[:5]) + "...")


def test_multichannel_function(eeg_signals, fs):
    """Test the multichannel function with multiple EEG channels"""

    print("\n6. Testing extract_multichannel_time_domain_features:")
    features_df = extract_multichannel_time_domain_features(eeg_signals, fs, ar_order=6)
    print("DataFrame shape:", features_df.shape)
    print("Columns:", ", ".join(features_df.columns[:5]) + "...")
    print("\nSample of the DataFrame:")
    print(features_df.head(2))

    # Create a simple visualization
    plt.figure(figsize=(15, 8))

    # Plot a few key features across channels
    key_features = ['mean', 'std', 'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']
    for i, feature in enumerate(key_features):
        if feature in features_df.columns:
            plt.subplot(len(key_features), 1, i + 1)
            plt.bar(features_df['channel'], features_df[feature])
            plt.title(feature)
            plt.ylabel('Value')
            if i == len(key_features) - 1:
                plt.xlabel('Channel')

    plt.tight_layout()
    plt.savefig('eeg_features_comparison.png')
    print("Saved visualization to 'eeg_features_comparison.png'")


def main():
    """Main test function"""
    print("EEG Time Domain Features - Test Script")
    print("======================================")

    # Try to load real EEG data from files
    eeg_data, fs = try_loading_real_eeg()

    # If no real data found, generate synthetic data
    if eeg_data is None:
        fs = 250  # Hz
        duration = 5  # seconds
        num_channels = 4
        print(f"Generating synthetic {num_channels}-channel EEG data ({duration} seconds at {fs} Hz)")
        eeg_data = generate_sample_eeg(fs, duration, num_channels)

    # Plot the data
    plt.figure(figsize=(15, 8))
    num_channels = eeg_data.shape[0]
    time = np.arange(eeg_data.shape[1]) / fs

    for ch in range(min(num_channels, 4)):  # Plot up to 4 channels
        plt.subplot(min(num_channels, 4), 1, ch + 1)
        plt.plot(time, eeg_data[ch])
        plt.title(f'Channel {ch}')
        plt.ylabel('Amplitude (μV)')
        if ch == min(num_channels, 4) - 1:  # Last subplot
            plt.xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig('eeg_signals.png')
    print("Saved EEG plot to 'eeg_signals.png'")

    # Test individual functions with the first channel
    test_individual_functions(eeg_data[0], fs)

    # Test multichannel function
    test_multichannel_function(eeg_data, fs)

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
