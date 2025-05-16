import numpy as np
from scipy import stats

from const import DATA_FOLDER
from data_reader import EEGDataReader


class EEGFeatureExtractor:
    """Class for extracting time-domain features from EEG signals"""

    def extract_features(self, eeg_data):
        """
        Extract time-domain features from EEG data
        """
        # Initialize features dictionary
        features = {}

        # Get dimensions
        n_samples, n_channels = eeg_data.shape

        # Pre-calculate frequently used metrics with vectorized operations
        means = np.mean(eeg_data, axis=0)
        variances = np.var(eeg_data, axis=0)
        stds = np.std(eeg_data, axis=0)
        rms_values = np.sqrt(np.mean(np.square(eeg_data), axis=0))

        # Loop through each channel
        for ch in range(n_channels):
            channel_data = eeg_data[:, ch]
            channel_name = f"channel_{ch}"

            # Mean, Variance, Standard Deviation, RMS
            features[f"{channel_name}_mean"] = means[ch]
            features[f"{channel_name}_var"] = variances[ch]
            features[f"{channel_name}_std"] = stds[ch]
            features[f"{channel_name}_rms"] = rms_values[ch]

            # Zero Crossing Rate (ZCR)
            zero_crossings = np.where(np.diff(np.signbit(channel_data)))[0]
            features[f"{channel_name}_zcr"] = len(zero_crossings) / n_samples

            # Skewness & Kurtosis
            features[f"{channel_name}_skewness"] = stats.skew(channel_data)
            features[f"{channel_name}_kurtosis"] = stats.kurtosis(channel_data)

            # Hjorth Parameters
            activity = variances[ch]

            # Mobility
            first_deriv = np.diff(channel_data)
            first_deriv_var = np.var(first_deriv)
            mobility = np.sqrt(first_deriv_var / activity) if activity > 0 else 0

            # Complexity
            second_deriv = np.diff(first_deriv)
            second_deriv_var = np.var(second_deriv)
            complexity = (np.sqrt(second_deriv_var / first_deriv_var) / mobility
                          if first_deriv_var > 0 and mobility > 0 else 0)

            features[f"{channel_name}_hjorth_activity"] = activity
            features[f"{channel_name}_hjorth_mobility"] = mobility
            features[f"{channel_name}_hjorth_complexity"] = complexity

        return features

    def process_example(self, eeg_id, sub_id, data_reader=None):
        """Process a single example and return features"""
        # Create data reader if not provided
        if data_reader is None:
            data_reader = EEGDataReader(DATA_FOLDER)

        # Get the subsample as numpy array
        eeg_subsample, channels, meta = data_reader.get_eeg_subsample(eeg_id, sub_id)

        if eeg_subsample is not None:
            # Extract features
            return self.extract_features(eeg_subsample), meta
        return None, None


def test_extract_example():
    """Test feature extraction on a single example"""
    extractor = EEGFeatureExtractor()
    reader = EEGDataReader()

    # Extract a specific subsample
    eeg_id = "1628180742"
    sub_id = 3

    features, meta = extractor.process_example(eeg_id, sub_id, reader)

    if features:
        # Display first few features
        print("Example features for", eeg_id, "sub_id", sub_id, ":")
        for i, (key, value) in enumerate(list(features.items())[:10]):
            print(f"{key}: {value:.6f}")
        print("...")  # Indicate more features available


if __name__ == '__main__':
    test_extract_example()