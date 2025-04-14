import numpy as np
from scipy import signal
from scipy import stats

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
        'gamma': (30, 100)  # Upper limit typically depends on sampling rate
    }

    def extract_features(self, eeg_data, fs=SAMPLE_RATE):
        """
        Extract frequency-domain features from EEG data

        Parameters:
            eeg_data (numpy.ndarray): EEG data with shape (samples, channels)
            fs (int): Sampling frequency in Hz (default 250 Hz)

        Returns:
            dict: Dictionary of extracted features
        """
        # Initialize features dictionary
        features = {}

        # Get dimensions
        n_samples, n_channels = eeg_data.shape

        # Compute FFT parameters
        # Use next power of 2 for better FFT performance
        nfft = 2 ** int(np.ceil(np.log2(n_samples)))

        # Loop through each channel
        for ch in range(n_channels):
            channel_data = eeg_data[:, ch]
            channel_name = f"channel_{ch}"

            # Apply Hamming window to reduce spectral leakage
            windowed_data = channel_data * np.hamming(len(channel_data))

            # Compute Power Spectral Density
            freqs, psd = signal.welch(windowed_data, fs=fs, nperseg=min(256, n_samples),
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

                # Calculate relative band power (normalized by total power)
                rel_band_power = band_power / total_power if total_power > 0 else 0
                features[f"{channel_name}_{band_name}_relative_power"] = rel_band_power

            # Find peak frequency (frequency with maximum power)
            if len(psd) > 0:
                peak_idx = np.argmax(psd)
                peak_freq = freqs[peak_idx]
                features[f"{channel_name}_peak_frequency"] = peak_freq
            else:
                features[f"{channel_name}_peak_frequency"] = 0

            # Calculate Spectral Entropy
            psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else np.zeros_like(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-16))  # Add small value to avoid log(0)
            features[f"{channel_name}_spectral_entropy"] = spectral_entropy

            # Calculate spectral edge frequency (95% of power)
            if total_power > 0:
                cumulative_power = np.cumsum(psd) / total_power
                # Find frequency where cumulative power reaches 95%
                idx_95 = np.argmax(cumulative_power >= 0.95)
                spectral_edge_95 = freqs[idx_95] if idx_95 < len(freqs) else freqs[-1]
                features[f"{channel_name}_spectral_edge_95"] = spectral_edge_95
            else:
                features[f"{channel_name}_spectral_edge_95"] = 0

            # Calculate spectral moments
            # First moment (mean frequency)
            mean_freq = np.sum(freqs * psd) / total_power if total_power > 0 else 0
            features[f"{channel_name}_mean_frequency"] = mean_freq

            # Second moment (variance of frequency)
            freq_var = np.sum(((freqs - mean_freq) ** 2) * psd) / total_power if total_power > 0 else 0
            features[f"{channel_name}_freq_variance"] = freq_var

            # Calculate spectral skewness and kurtosis
            if total_power > 0 and freq_var > 0:
                norm_psd = psd / total_power
                freq_skewness = np.sum(((freqs - mean_freq) ** 3) * norm_psd) / (freq_var ** 1.5)
                freq_kurtosis = np.sum(((freqs - mean_freq) ** 4) * norm_psd) / (freq_var ** 2) - 3

                features[f"{channel_name}_freq_skewness"] = freq_skewness
                features[f"{channel_name}_freq_kurtosis"] = freq_kurtosis
            else:
                features[f"{channel_name}_freq_skewness"] = 0
                features[f"{channel_name}_freq_kurtosis"] = 0

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
        for i, (key, value) in enumerate(list(features.items())[:10]):
            print(f"{key}: {value:.6f}")
        print("...")  # Indicate more features available


if __name__ == '__main__':
    test_extract_example()