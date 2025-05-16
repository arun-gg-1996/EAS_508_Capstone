import numpy as np
from scipy import stats
import antropy as ant

from const import DATA_FOLDER, SAMPLE_RATE
from data_reader import EEGDataReader


class EEGNonlinearExtractor:
    """Class for extracting non-linear features from EEG signals"""

    def __init__(self):
        # Initialize dictionary to store timing information
        self.timing_stats = {}

    def extract_features(self, eeg_data, fs=SAMPLE_RATE):
        """
        Extract non-linear features from EEG data

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

            # Remove NaN values if any
            channel_data = channel_data[~np.isnan(channel_data)]

            if len(channel_data) <= 3:
                # Skip empty or very short channels
                continue

            # 1. Fractal Dimensions
            try:
                # Petrosian Fractal Dimension
                # Count sign changes
                diff = np.diff(channel_data)
                sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
                n = len(channel_data)

                if n > 1 and sign_changes > 0:
                    petrosian_fd = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * sign_changes)))
                else:
                    petrosian_fd = 0

                features[f"{channel_name}_petrosian_fd"] = petrosian_fd

                # Katz Fractal Dimension - optimized version
                dists = np.abs(np.diff(channel_data))
                d = np.sum(dists)
                L = np.max(np.abs(channel_data - channel_data[0]))

                if L > 0 and d > 0:
                    katz_fd = np.log10(n) / (np.log10(d / L) + np.log10(n))
                else:
                    katz_fd = 0

                features[f"{channel_name}_katz_fd"] = katz_fd

                # Higuchi Fractal Dimension - use antropy with limited k
                features[f"{channel_name}_higuchi_fd"] = ant.higuchi_fd(channel_data, kmax=5)  # Lower kmax for speed
            except Exception as e:
                features[f"{channel_name}_petrosian_fd"] = 0
                features[f"{channel_name}_katz_fd"] = 0
                features[f"{channel_name}_higuchi_fd"] = 0

            # 2. Detrended Fluctuation Analysis (simplified)
            try:
                # Use a simplified version for speed
                if len(channel_data) >= 100:  # Only compute for reasonably sized signals
                    # Just use a few scales for speed
                    scales = np.array([4, 8, 16, 32, 64]).astype(int)
                    scales = scales[scales < len(channel_data) // 4]

                    if len(scales) >= 2:
                        # Cumulative sum
                        y = np.cumsum(channel_data - np.mean(channel_data))

                        # Calculate fluctuation for limited scales
                        fluctuations = np.zeros(len(scales))

                        for i, scale in enumerate(scales):
                            # Number of segments
                            num_segments = len(channel_data) // scale

                            if num_segments > 0:
                                # Calculate local trend only for a few segments
                                max_segments = min(num_segments, 10)
                                segments = np.random.choice(num_segments, max_segments, replace=False)

                                local_rms = np.zeros(max_segments)

                                for j, seg_idx in enumerate(segments):
                                    segment = y[seg_idx * scale:(seg_idx + 1) * scale]

                                    # Simple linear detrending
                                    x_values = np.arange(len(segment))
                                    if len(segment) > 1:
                                        coeffs = np.polyfit(x_values, segment, 1)
                                        trend = np.polyval(coeffs, x_values)
                                        local_rms[j] = np.sqrt(np.mean((segment - trend) ** 2))

                                fluctuations[i] = np.mean(local_rms)

                        # Get non-zero fluctuations
                        valid_idx = fluctuations > 0
                        valid_scales = scales[valid_idx]
                        valid_fluct = fluctuations[valid_idx]

                        if len(valid_scales) >= 2:
                            # Linear regression
                            log_scales = np.log10(valid_scales)
                            log_fluct = np.log10(valid_fluct)

                            slope, _, _, _, _ = stats.linregress(log_scales, log_fluct)
                            features[f"{channel_name}_dfa"] = slope
                        else:
                            features[f"{channel_name}_dfa"] = 0
                    else:
                        features[f"{channel_name}_dfa"] = 0
                else:
                    features[f"{channel_name}_dfa"] = 0
            except Exception as e:
                features[f"{channel_name}_dfa"] = 0

            # 3. Entropy (use faster simplified versions)
            try:
                # Simplified sample entropy (faster)
                if len(channel_data) >= 100:
                    # Normalize data for numerical stability
                    data_std = np.std(channel_data)
                    if data_std > 0:
                        norm_data = (channel_data - np.mean(channel_data)) / data_std
                    else:
                        norm_data = channel_data - np.mean(channel_data)

                    # Use reduced sample size by subsampling
                    sample_size = min(len(norm_data), 1000)  # limit sample size
                    step = max(1, len(norm_data) // sample_size)
                    sample_data = norm_data[::step][:sample_size]

                    # Use chebyshev distance for speed
                    sample_entropy = ant.sample_entropy(sample_data, order=2, metric='chebyshev')
                    features[f"{channel_name}_sample_entropy"] = sample_entropy

                    # Faster permutation entropy with smaller order
                    perm_entropy = ant.perm_entropy(sample_data, order=3, normalize=True)
                    features[f"{channel_name}_perm_entropy"] = perm_entropy

                    # Faster spectral entropy
                    spectral_entropy = ant.spectral_entropy(sample_data, sf=fs, method='fft', normalize=True)
                    features[f"{channel_name}_spectral_entropy"] = spectral_entropy
                else:
                    features[f"{channel_name}_sample_entropy"] = 0
                    features[f"{channel_name}_perm_entropy"] = 0
                    features[f"{channel_name}_spectral_entropy"] = 0
            except Exception as e:
                features[f"{channel_name}_sample_entropy"] = 0
                features[f"{channel_name}_perm_entropy"] = 0
                features[f"{channel_name}_spectral_entropy"] = 0

            # 4. Simple Binary Complexity (very fast version)
            try:
                # Only use a small segment of the data
                max_points = min(len(channel_data), 1000)
                step = max(1, len(channel_data) // max_points)
                binary_data = (channel_data[::step][:max_points] > np.median(channel_data)).astype(int)

                # Count transitions as a simple complexity measure
                transitions = np.sum(np.abs(np.diff(binary_data)))
                norm_complexity = transitions / (len(binary_data) - 1) if len(binary_data) > 1 else 0
                features[f"{channel_name}_binary_complexity"] = norm_complexity
            except Exception as e:
                features[f"{channel_name}_binary_complexity"] = 0

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
            features = self.extract_features(eeg_subsample, fs=fs)
            return features, meta
        return None, None


def test_extract_example():
    """Test non-linear feature extraction on a single example"""
    extractor = EEGNonlinearExtractor()
    reader = EEGDataReader()

    # Extract a specific subsample
    eeg_id = "1628180742"
    sub_id = 3

    import time
    start_time = time.time()
    features, meta = extractor.process_example(eeg_id, sub_id, reader)
    elapsed = time.time() - start_time

    if features:
        # Display first few features
        print("Example non-linear features:")
        for i, (key, value) in enumerate(list(features.items())[:5]):
            print(f"{key}: {value:.6f}")
        print("...")  # Indicate more features available
        print(f"Total time: {elapsed:.4f} seconds")


if __name__ == '__main__':
    test_extract_example()


    def process_example(self, eeg_id, sub_id, data_reader=None, fs=SAMPLE_RATE):
        """Process a single example and return features"""
        # Create data reader if not provided
        if data_reader is None:
            data_reader = EEGDataReader(DATA_FOLDER)

        # Get the subsample as numpy array
        eeg_subsample, channels, meta = data_reader.get_eeg_subsample(eeg_id, sub_id)

        if eeg_subsample is not None:
            # Extract features
            features = self.extract_features(eeg_subsample, fs=fs)
            return features, meta
        return None, None


def test_extract_example():
    """Test non-linear feature extraction on a single example"""
    extractor = EEGNonlinearExtractor()
    reader = EEGDataReader()

    # Extract a specific subsample
    eeg_id = "1628180742"
    sub_id = 3

    features, meta = extractor.process_example(eeg_id, sub_id, reader)

    if features:
        # Display first few features
        print("Example non-linear features:")
        for i, (key, value) in enumerate(list(features.items())[:5]):
            print(f"{key}: {value:.6f}")
        print("...")  # Indicate more features available


if __name__ == '__main__':
    test_extract_example()