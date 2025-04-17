import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import antropy as ant  # Import antropy for optimized calculations
import time  # Import time module for timing

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

        # Initialize timing dictionary for this run
        timing = {}

        # Reset timing stats for this run
        self.timing_stats = {}

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

            # 1. Fractal Dimensions using antropy
            try:
                # Petrosian Fractal Dimension
                petrosian_fd = ant.petrosian_fd(channel_data)
                features[f"{channel_name}_petrosian_fd"] = petrosian_fd

                # Katz Fractal Dimension
                katz_fd = ant.katz_fd(channel_data)
                features[f"{channel_name}_katz_fd"] = katz_fd

                # Higuchi Fractal Dimension
                higuchi_fd = ant.higuchi_fd(channel_data)
                features[f"{channel_name}_higuchi_fd"] = higuchi_fd

            except Exception as e:
                # Fall back to custom implementations
                try:
                    # Higuchi Fractal Dimension
                    hfd = self._higuchi_fd(channel_data, k_max=10)
                    features[f"{channel_name}_higuchi_fd"] = hfd

                    # Katz Fractal Dimension
                    katz_fd = self._katz_fd(channel_data)
                    features[f"{channel_name}_katz_fd"] = katz_fd

                    # Petrosian Fractal Dimension
                    petrosian_fd = self._petrosian_fd(channel_data)
                    features[f"{channel_name}_petrosian_fd"] = petrosian_fd
                except Exception as e2:
                    features[f"{channel_name}_higuchi_fd"] = 0
                    features[f"{channel_name}_katz_fd"] = 0
                    features[f"{channel_name}_petrosian_fd"] = 0
                    print(f"Error computing custom fractal dimensions for {channel_name}: {e2}")

                print(f"Error computing antropy fractal dimensions for {channel_name}: {e}")

            # 2. Detrended Fluctuation Analysis using antropy
            try:
                dfa = ant.detrended_fluctuation(channel_data)
                features[f"{channel_name}_dfa"] = dfa
            except Exception as e:
                # Fall back to custom implementation
                try:
                    dfa = self._dfa(channel_data)
                    features[f"{channel_name}_dfa"] = dfa
                except Exception as e2:
                    features[f"{channel_name}_dfa"] = 0
                    print(f"Error computing custom DFA for {channel_name}: {e2}")

                print(f"Error computing antropy DFA for {channel_name}: {e}")

            # 3. Lempel-Ziv Complexity (simplified version)
            try:
                lzc = self._lempel_ziv_complexity(channel_data)
                features[f"{channel_name}_lempel_ziv_complexity"] = lzc
            except Exception as e:
                features[f"{channel_name}_lempel_ziv_complexity"] = 0
                print(f"Error computing Lempel-Ziv Complexity for {channel_name}: {e}")

        return features

    def _katz_fd(self, x):
        """Calculate Katz Fractal Dimension"""
        x = np.array(x)
        n = len(x)
        if n < 2:
            return 0

        # Calculate the "distance" between consecutive points
        dists = np.abs(np.diff(x))
        d = np.sum(dists)

        # Find the maximum distance from first point
        L = np.max(np.abs(x - x[0]))

        if L == 0 or d == 0:
            return 0

        # Calculate Katz FD
        return np.log10(n) / (np.log10(d / L) + np.log10(n))

    def _petrosian_fd(self, x):
        """Calculate Petrosian Fractal Dimension"""
        x = np.array(x)
        n = len(x)
        if n < 2:
            return 0

        # Calculate differences
        diff = np.diff(x)

        # Count sign changes (zero crossings)
        sign_changes = np.sum(diff[:-1] * diff[1:] < 0)

        # Calculate Petrosian FD
        if n > 1 and sign_changes > 0:
            return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * sign_changes)))
        return 0

    def _higuchi_fd(self, x, k_max=10):
        """Calculate Higuchi Fractal Dimension"""
        x = np.array(x)
        n = len(x)

        if n < k_max:
            k_max = n - 1

        if n < 2 or k_max < 1:
            return 0

        # Initialize length array
        L = np.zeros(k_max)

        # Calculate length for different time scales
        for k in range(1, k_max + 1):
            Lk = 0

            # For each starting point
            for m in range(k):
                # Build the subsequence
                indices = np.arange(m, n, k)
                sub_x = x[indices]

                # Calculate the length of the curve
                if len(sub_x) > 1:
                    Lm = np.sum(np.abs(np.diff(sub_x))) * (n - 1) / (((n - m) // k) * k)
                    Lk += Lm

            L[k - 1] = Lk / k

        # Fit a line to log(L) vs log(1/k)
        k_values = np.arange(1, k_max + 1)
        log_k = np.log(1.0 / k_values)
        log_L = np.log(L)

        # Linear regression to find slope
        try:
            slope, _, _, _, _ = stats.linregress(log_k, log_L)
            return slope
        except Exception:
            return 0

    def _dfa(self, x, scales=None):
        """Calculate Detrended Fluctuation Analysis exponent"""
        x = np.array(x)
        n = len(x)

        # Define scales to use (box sizes)
        if scales is None:
            scales = np.logspace(np.log10(4), np.log10(n // 4), 10).astype(int)
            scales = np.unique(scales)

        # Calculate cumulative sum (integration)
        y = np.cumsum(x - np.mean(x))

        # Calculate fluctuation for each scale
        fluctuations = np.zeros(len(scales))

        for i, scale in enumerate(scales):
            if scale < 4 or scale > n // 4:
                continue

            # Number of non-overlapping segments
            num_segments = n // scale

            # Calculate local RMS for each segment
            local_rms = np.zeros(num_segments)

            for j in range(num_segments):
                # Get segment
                segment = y[j * scale:(j + 1) * scale]

                # Fit polynomial (linear detrending)
                x_values = np.arange(scale)
                coeffs = np.polyfit(x_values, segment, 1)
                trend = np.polyval(coeffs, x_values)

                # Calculate RMS of detrended segment
                local_rms[j] = np.sqrt(np.mean((segment - trend) ** 2))

            # Calculate mean RMS across all segments
            fluctuations[i] = np.mean(local_rms)

        # Remove zeros (from invalid scales)
        valid_scales = scales[fluctuations > 0]
        valid_fluct = fluctuations[fluctuations > 0]

        if len(valid_scales) < 2:
            return 0

        # Fit power law: F(n) ~ n^alpha
        log_scales = np.log10(valid_scales)
        log_fluct = np.log10(valid_fluct)

        # Linear regression
        alpha, _, _, _, _ = stats.linregress(log_scales, log_fluct)

        return alpha

    def _lempel_ziv_complexity(self, x):
        """Calculate Lempel-Ziv complexity (simplified binary version)"""
        # Binarize the signal (above/below median)
        x = np.array(x)
        binary_seq = (x > np.median(x)).astype(int)

        # Convert to string representation
        binary_str = ''.join(binary_seq.astype(str))

        # Count distinct substrings using dictionary
        substrings = {}
        complexity = 1  # Start with 1 for the first character

        # Current substring
        current = binary_str[0]

        # Analyze rest of string
        for i in range(1, len(binary_str)):
            current += binary_str[i]

            # If current substring is new, increment complexity
            if current not in substrings:
                substrings[current] = 1
                complexity += 1
                current = ""

        # Normalize by log(n)
        n = len(binary_str)
        if n > 1:
            return complexity / (n / np.log2(n))
        return 0

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

    # Measure total processing time
    start_time = time.time()
    features, meta = extractor.process_example(eeg_id, sub_id, reader)
    total_time = time.time() - start_time

    if features:
        # Display first few features
        print("Example non-linear features for", eeg_id, "sub_id", sub_id, ":")
        print("=" * 50)
        print("Feature values (first 10):")
        for i, (key, value) in enumerate(list(features.items())[:10]):
            print(f"{key}: {value:.6f}")
        print("...")  # Indicate more features available

        print("\nTotal processing time:", total_time, "seconds")


if __name__ == '__main__':
    test_extract_example()