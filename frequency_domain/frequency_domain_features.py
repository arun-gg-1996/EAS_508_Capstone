import numpy as np
from scipy import signal

from const import DATA_FOLDER, SAMPLE_RATE
from data_reader import EEGDataReader


class EEGFrequencyExtractor:
    """Class for extracting frequency-domain features from EEG signals"""

    # Define EEG frequency bands
    FREQ_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    def extract_features(self, eeg_data, fs=SAMPLE_RATE):
        """
        Extract frequency-domain features from EEG data

        Parameters:
            eeg_data (numpy.ndarray): EEG data with shape (samples, channels)
            fs (int): Sampling frequency in Hz

        Returns:
            dict: Dictionary of extracted features
        """
        # Initialize features dictionary
        features = {}

        # Get dimensions
        n_samples, n_channels = eeg_data.shape

        # Compute FFT parameters
        nfft = 2 ** int(np.ceil(np.log2(n_samples)))
        nperseg = min(256, n_samples)

        # Loop through each channel
        for ch in range(n_channels):
            channel_data = eeg_data[:, ch]
            channel_name = f"channel_{ch}"

            # Apply window to reduce spectral leakage
            windowed_data = channel_data * np.hamming(len(channel_data))

            # Compute Power Spectral Density
            freqs, psd = signal.welch(windowed_data, fs=fs, nperseg=nperseg,
                                      nfft=nfft, scaling='density')

            # Calculate total power
            total_power = np.sum(psd)
            features[f"{channel_name}_total_power"] = total_power

            # Extract band powers and relative band powers
            for band_name, (low_freq, high_freq) in self.FREQ_BANDS.items():
                # Find indices corresponding to the frequency band
                band_idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)

                # Calculate absolute band power
                band_power = np.sum(psd[band_idx])
                features[f"{channel_name}_{band_name}_power"] = band_power

                # Calculate relative band power
                rel_band_power = band_power / total_power if total_power > 0 else 0
                features[f"{channel_name}_{band_name}_relative_power"] = rel_band_power

            # Find peak frequency
            if len(psd) > 0:
                peak_idx = np.argmax(psd)
                peak_freq = freqs[peak_idx]
                features[f"{channel_name}_peak_frequency"] = peak_freq
            else:
                features[f"{channel_name}_peak_frequency"] = 0

        return features

    def process_example(self, eeg_id, sub_id, data_reader=None, fs=SAMPLE_RATE):
        """Process a single example and return features"""
        # Create data reader if not provided
        if data_reader is None:
            data_reader = EEGDataReader(DATA_FOLDER)

        # Get the subsample as numpy array
        eeg_subsample, channels, meta = data_reader.get_eeg_subsample(eeg_id, sub_id)

        if eeg_subsample is not None:
            # Extract features
            return self.extract_features(eeg_subsample, fs=fs), meta
        return None, None


def test_extract_example():
    """Test frequency feature extraction on a single example"""
    extractor = EEGFrequencyExtractor()
    reader = EEGDataReader()

    # Extract a specific subsample
    eeg_id = "1628180742"
    sub_id = 3

    features, meta = extractor.process_example(eeg_id, sub_id, reader)

    if features:
        # Display first few features
        print("Example frequency features for", eeg_id, "sub_id", sub_id, ":")
        for i, (key, value) in enumerate(list(features.items())[:5]):
            print(f"{key}: {value:.6f}")
        print("...")  # Indicate more features available


if __name__ == '__main__':
    test_extract_example()