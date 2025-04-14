import numpy as np
import pywt
from scipy import stats

from const import DATA_FOLDER, SAMPLE_RATE
from data_reader import EEGDataReader


class EEGWaveletExtractor:
    """Class for extracting time-frequency domain features from EEG signals using wavelet transforms"""

    # Define EEG frequency bands (approximate mapping to wavelet scales)
    FREQ_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)  # Upper limit typically depends on sampling rate
    }

    def __init__(self, wavelet='db4', level=5):
        """
        Initialize the wavelet extractor with specific wavelet parameters

        Parameters:
            wavelet (str): Wavelet type to use (default: 'db4' - Daubechies 4)
            level (int): Decomposition level for the wavelet transform (default: 5)
        """
        self.wavelet = wavelet
        self.level = level

    def extract_features(self, eeg_data, fs=SAMPLE_RATE):
        """
        Extract time-frequency domain features from EEG data using wavelet transform

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

        # Loop through each channel
        for ch in range(n_channels):
            channel_data = eeg_data[:, ch]
            channel_name = f"channel_{ch}"

            # Perform wavelet decomposition
            coeffs = pywt.wavedec(channel_data, self.wavelet, level=self.level)

            # The first element is the approximation coefficients, the rest are detail coefficients
            approx_coeffs = coeffs[0]
            detail_coeffs = coeffs[1:]

            # Store length of coefficients for reference when mapping to frequency bands
            coeff_lengths = [len(c) for c in coeffs]

            # Calculate relative power in each wavelet level (roughly corresponding to frequency bands)
            total_power = np.sum([np.sum(c ** 2) for c in coeffs])

            # Add approximate mapping of wavelet levels to frequency bands
            # This is an approximation as wavelet scales don't directly map to specific frequency bands
            nyquist = fs / 2
            for i, detail in enumerate(detail_coeffs):
                # Calculate the approximate frequency range for this decomposition level
                # The highest frequency band is at i=0, lowest at i=level-1
                level_idx = i + 1  # +1 because detail_coeffs doesn't include the approximation

                # Calculate frequency range for this level
                upper_freq = nyquist / (2 ** (level_idx - 1))
                lower_freq = nyquist / (2 ** level_idx)

                # Find the closest traditional frequency band
                band_name = self._get_closest_freq_band(lower_freq, upper_freq)

                # Calculate energy (power) for this level
                energy = np.sum(detail ** 2)
                relative_energy = energy / total_power if total_power > 0 else 0

                # Store the feature with both level number and approximate band name
                features[f"{channel_name}_wavelet_level_{level_idx}_energy"] = energy
                features[f"{channel_name}_wavelet_level_{level_idx}_rel_energy"] = relative_energy
                features[f"{channel_name}_wavelet_{band_name}_energy"] = energy
                features[f"{channel_name}_wavelet_{band_name}_rel_energy"] = relative_energy

                # Statistical features of wavelet coefficients for this level
                features[f"{channel_name}_wavelet_level_{level_idx}_mean"] = np.mean(detail)
                features[f"{channel_name}_wavelet_level_{level_idx}_std"] = np.std(detail)
                features[f"{channel_name}_wavelet_level_{level_idx}_kurt"] = stats.kurtosis(detail)
                features[f"{channel_name}_wavelet_level_{level_idx}_skew"] = stats.skew(detail)

                # Calculate temporal complexity using entropy measures
                # Wavelet Energy Entropy
                if len(detail) > 0:
                    coef_norm = detail ** 2 / np.sum(detail ** 2) if np.sum(detail ** 2) > 0 else np.zeros_like(detail)
                    wavelet_entropy = -np.sum(coef_norm * np.log2(coef_norm + 1e-16))
                    features[f"{channel_name}_wavelet_level_{level_idx}_entropy"] = wavelet_entropy
                    features[f"{channel_name}_wavelet_{band_name}_entropy"] = wavelet_entropy
                else:
                    features[f"{channel_name}_wavelet_level_{level_idx}_entropy"] = 0
                    features[f"{channel_name}_wavelet_{band_name}_entropy"] = 0

            # Also calculate features for approximation coefficients (lowest frequency band)
            energy_approx = np.sum(approx_coeffs ** 2)
            relative_energy_approx = energy_approx / total_power if total_power > 0 else 0

            features[f"{channel_name}_wavelet_approx_energy"] = energy_approx
            features[f"{channel_name}_wavelet_approx_rel_energy"] = relative_energy_approx
            features[f"{channel_name}_wavelet_approx_mean"] = np.mean(approx_coeffs)
            features[f"{channel_name}_wavelet_approx_std"] = np.std(approx_coeffs)

            # Additional wavelet-specific features

            # 1. Total Wavelet Entropy - overall complexity across all scales
            total_entropy = 0
            for i, detail in enumerate(detail_coeffs):
                level_idx = i + 1
                if len(detail) > 0:
                    p_i = np.sum(detail ** 2) / total_power if total_power > 0 else 0
                    if p_i > 0:
                        total_entropy -= p_i * np.log2(p_i)

            features[f"{channel_name}_total_wavelet_entropy"] = total_entropy

            # 2. Wavelet Energy Ratio - ratio of energy in different decomposition levels
            # Useful for detecting shifts in energy distribution across frequency bands
            for i in range(len(detail_coeffs) - 1):
                level_idx1 = i + 1
                level_idx2 = i + 2

                energy1 = np.sum(detail_coeffs[i] ** 2)
                energy2 = np.sum(detail_coeffs[i + 1] ** 2)

                ratio = energy1 / energy2 if energy2 > 0 else 0
                features[f"{channel_name}_wavelet_ratio_L{level_idx1}_L{level_idx2}"] = ratio

        return features

    def _get_closest_freq_band(self, lower_freq, upper_freq):
        """
        Find the closest traditional frequency band for the given wavelet level frequency range

        Parameters:
            lower_freq (float): Lower frequency bound of wavelet level
            upper_freq (float): Upper frequency bound of wavelet level

        Returns:
            str: Name of the closest traditional frequency band
        """
        # Calculate midpoint of the wavelet level frequency range
        mid_freq = (lower_freq + upper_freq) / 2

        # Find the traditional band this frequency falls into
        for band_name, (band_low, band_high) in self.FREQ_BANDS.items():
            if band_low <= mid_freq <= band_high:
                return band_name

        # If not in any band, return the closest band
        if mid_freq < self.FREQ_BANDS['delta'][0]:
            return 'sub_delta'
        elif mid_freq > self.FREQ_BANDS['gamma'][1]:
            return 'high_gamma'

        # Should not reach here if frequency bands cover the entire spectrum
        return 'unclassified'

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
    """Test wavelet feature extraction on a single example"""
    extractor = EEGWaveletExtractor()
    reader = EEGDataReader()

    # Extract a specific subsample
    eeg_id = "1628180742"
    sub_id = 3

    features, meta = extractor.process_example(eeg_id, sub_id, reader)

    if features:
        # Display first few features
        print("Example wavelet features for", eeg_id, "sub_id", sub_id, ":")
        for i, (key, value) in enumerate(list(features.items())[:10]):
            print(f"{key}: {value:.6f}")
        print("...")  # Indicate more features available


if __name__ == '__main__':
    test_extract_example()
