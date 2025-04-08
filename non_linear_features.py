import numpy as np
import pandas as pd
from scipy.stats import entropy
import nolds


def extract_sample_entropy(eeg_signal, m=2, r=0.2):
    """
    Extract Sample Entropy (SampEn) from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    m : int
        Embedding dimension
    r : float
        Tolerance (typically 0.1 to 0.25 times signal std)

    Returns:
    --------
    dict
        Dictionary with sample entropy feature
    """
    features = {}

    # Calculate Sample Entropy
    try:
        # If signal is too short or constant, the calculation might fail
        if len(eeg_signal) < (m + 2) or np.std(eeg_signal) < 1e-10:
            features['sample_entropy'] = 0
        else:
            # Normalize r by signal standard deviation
            r_normalized = r * np.std(eeg_signal)
            features['sample_entropy'] = nolds.sampen(eeg_signal, dim=m, r=r_normalized)
    except Exception as e:
        print(f"Error calculating Sample Entropy: {e}")
        features['sample_entropy'] = 0

    return features


def extract_approximate_entropy(eeg_signal, m=2, r=0.2):
    """
    Extract Approximate Entropy (ApEn) from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    m : int
        Embedding dimension
    r : float
        Tolerance (typically 0.1 to 0.25 times signal std)

    Returns:
    --------
    dict
        Dictionary with approximate entropy feature
    """
    features = {}

    # Calculate Approximate Entropy
    try:
        # If signal is too short or constant, the calculation might fail
        if len(eeg_signal) < (m + 2) or np.std(eeg_signal) < 1e-10:
            features['approximate_entropy'] = 0
        else:
            # Normalize r by signal standard deviation
            r_normalized = r * np.std(eeg_signal)
            features['approximate_entropy'] = nolds.apen(eeg_signal, dim=m, r=r_normalized)
    except Exception as e:
        print(f"Error calculating Approximate Entropy: {e}")
        features['approximate_entropy'] = 0

    return features


def extract_fractal_dimension(eeg_signal):
    """
    Extract Fractal Dimension (Higuchi and Katz) from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal

    Returns:
    --------
    dict
        Dictionary with fractal dimension features
    """
    features = {}

    # Calculate Higuchi Fractal Dimension
    try:
        features['higuchi_fd'] = nolds.hfd(eeg_signal, 6)
    except Exception as e:
        print(f"Error calculating Higuchi Fractal Dimension: {e}")
        features['higuchi_fd'] = 0

    # Calculate Katz Fractal Dimension
    try:
        # Katz Fractal Dimension implementation
        n = len(eeg_signal)
        if n < 3:
            features['katz_fd'] = 0
            return features

        # Calculate the total length of the curve
        L = np.sum(np.abs(np.diff(eeg_signal)))
        if L == 0:
            features['katz_fd'] = 0
            return features

        # Calculate the maximum distance between the first point and any other point
        d = np.max(np.abs(eeg_signal - eeg_signal[0]))
        if d == 0:
            features['katz_fd'] = 0
            return features

        # Calculate Katz Fractal Dimension
        katz_fd = np.log10(n) / (np.log10(d / L) + np.log10(n))
        features['katz_fd'] = katz_fd
    except Exception as e:
        print(f"Error calculating Katz Fractal Dimension: {e}")
        features['katz_fd'] = 0

    return features


def extract_permutation_entropy(eeg_signal, order=3, delay=1):
    """
    Extract Permutation Entropy from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    order : int
        Order of permutation entropy
    delay : int
        Delay between points in the embedding

    Returns:
    --------
    dict
        Dictionary with permutation entropy feature
    """
    from itertools import permutations

    features = {}

    try:
        # Check if the signal is long enough
        if len(eeg_signal) < order * delay:
            features['permutation_entropy'] = 0
            features['normalized_permutation_entropy'] = 0
            return features

        # Create the embedding
        time_series = np.array(
            [eeg_signal[i:i + (order * delay):delay] for i in range(len(eeg_signal) - (order - 1) * delay)])

        # Calculate the permutations
        permutation_ids = np.empty(len(time_series), dtype=int)

        # Generate all possible permutations for the given order
        all_possible_permutations = list(permutations(range(order)))
        mapping = dict((p, i) for i, p in enumerate(all_possible_permutations))

        # Count occurrences of each permutation
        for i, series in enumerate(time_series):
            sort_index = np.argsort(series)
            perm = tuple(np.take(np.arange(len(series)), sort_index))
            permutation_ids[i] = mapping[perm]

        # Calculate the probability of each permutation
        counts = np.bincount(permutation_ids, minlength=len(all_possible_permutations))
        probs = counts / float(counts.sum())

        # Calculate permutation entropy
        pe = entropy(probs, base=2)

        # Normalize by dividing by log2(factorial(order))
        import math
        max_entropy = np.log2(math.factorial(order))
        normalized_pe = pe / max_entropy if max_entropy > 0 else 0

        features['permutation_entropy'] = pe
        features['normalized_permutation_entropy'] = normalized_pe

    except Exception as e:
        print(f"Error calculating Permutation Entropy: {e}")
        features['permutation_entropy'] = 0
        features['normalized_permutation_entropy'] = 0

    return features


def extract_multiscale_entropy(eeg_signal, m=2, r=0.2, scale=5):
    """
    Extract Multiscale Entropy from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    m : int
        Embedding dimension
    r : float
        Tolerance
    scale : int
        Maximum scale factor

    Returns:
    --------
    dict
        Dictionary with multiscale entropy features
    """
    features = {}

    try:
        # Function to compute coarse-grained time series
        def coarse_grain(signal, scale_factor):
            # Truncate signal length to be divisible by scale_factor
            new_length = len(signal) // scale_factor * scale_factor
            signal = signal[:new_length]

            # Reshape and average
            return np.mean(signal.reshape(-1, scale_factor), axis=1)

        # Compute sample entropy at different scales
        for i in range(1, scale + 1):
            # Create coarse-grained time series
            coarse_signal = coarse_grain(eeg_signal, i)

            # If resulting signal is too short, skip this scale
            if len(coarse_signal) < 2 * (m + 1):
                features[f'mse_scale_{i}'] = 0
                continue

            # Compute sample entropy on coarse-grained time series
            r_normalized = r * np.std(coarse_signal)
            if r_normalized == 0:
                features[f'mse_scale_{i}'] = 0
                continue

            se = nolds.sampen(coarse_signal, dim=m, r=r_normalized)
            features[f'mse_scale_{i}'] = se

        # Calculate mean MSE
        valid_mse_values = [v for k, v in features.items() if k.startswith('mse_scale_') and v > 0]
        features['mse_mean'] = np.mean(valid_mse_values) if valid_mse_values else 0

        # Calculate slope of MSE
        scale_values = np.arange(1, scale + 1)
        mse_values = [features[f'mse_scale_{i}'] for i in scale_values]

        if len(scale_values) > 1 and np.any(mse_values):
            # Fit line to non-zero values
            valid_indices = [i for i, mse in enumerate(mse_values) if mse > 0]
            if len(valid_indices) > 1:
                valid_scales = scale_values[valid_indices]
                valid_mse = [mse_values[i] for i in valid_indices]

                # Fit line
                slope, _ = np.polyfit(valid_scales, valid_mse, 1)
                features['mse_slope'] = slope
            else:
                features['mse_slope'] = 0
        else:
            features['mse_slope'] = 0

    except Exception as e:
        print(f"Error calculating Multiscale Entropy: {e}")
        for i in range(1, scale + 1):
            features[f'mse_scale_{i}'] = 0
        features['mse_mean'] = 0
        features['mse_slope'] = 0

    return features


def extract_lempel_ziv_complexity(eeg_signal, threshold=None):
    """
    Extract Lempel-Ziv Complexity from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    threshold : float or None
        Threshold for binarization. If None, median is used.

    Returns:
    --------
    dict
        Dictionary with Lempel-Ziv complexity feature
    """
    features = {}

    try:
        # Binarize the signal
        if threshold is None:
            threshold = np.median(eeg_signal)

        binary_signal = np.where(eeg_signal > threshold, 1, 0)

        # Convert to string
        binary_string = ''.join(binary_signal.astype(str))

        # Calculate Lempel-Ziv complexity
        # Initialize
        c = 1
        i = 0
        substrings = []

        # Iterate through binary string
        while i + c <= len(binary_string):
            current = binary_string[i:i + c]
            if current in substrings:
                c += 1
            else:
                substrings.append(current)
                i += c
                c = 1

        # Add the last substring if it's not in the list
        if i < len(binary_string):
            substrings.append(binary_string[i:])

        # Calculate normalized complexity
        n = len(binary_string)
        b = 2  # binary alphabet size

        # Normalize by the upper bound of complexity for binary sequence
        if n > 0:
            upper_bound = n / np.log2(n) if n > b else b
            normalized_lzc = len(substrings) / upper_bound
        else:
            normalized_lzc = 0

        features['lempel_ziv_complexity'] = len(substrings)
        features['normalized_lzc'] = normalized_lzc

    except Exception as e:
        print(f"Error calculating Lempel-Ziv Complexity: {e}")
        features['lempel_ziv_complexity'] = 0
        features['normalized_lzc'] = 0

    return features


def extract_detrended_fluctuation_analysis(eeg_signal):
    """
    Extract Detrended Fluctuation Analysis (DFA) from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal

    Returns:
    --------
    dict
        Dictionary with DFA features
    """
    features = {}

    try:
        # Calculate DFA exponent
        features['dfa_alpha'] = nolds.dfa(eeg_signal)
    except Exception as e:
        print(f"Error calculating DFA: {e}")
        features['dfa_alpha'] = 0

    return features


def extract_hurst_exponent(eeg_signal):
    """
    Extract Hurst Exponent from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal

    Returns:
    --------
    dict
        Dictionary with Hurst exponent feature
    """
    features = {}

    try:
        # Calculate Hurst exponent
        features['hurst_exponent'] = nolds.hurst_rs(eeg_signal)
    except Exception as e:
        print(f"Error calculating Hurst Exponent: {e}")
        features['hurst_exponent'] = 0

    return features


def extract_lyapunov_exponent(eeg_signal):
    """
    Extract Largest Lyapunov Exponent from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal

    Returns:
    --------
    dict
        Dictionary with Lyapunov exponent feature
    """
    features = {}

    try:
        # Calculate Largest Lyapunov Exponent
        features['lyapunov_exp'] = nolds.lyap_r(eeg_signal, emb_dim=10, lag=2)
    except Exception as e:
        print(f"Error calculating Lyapunov Exponent: {e}")
        features['lyapunov_exp'] = 0

    return features


def extract_correlation_dimension(eeg_signal, emb_dim=10, lag=2):
    """
    Extract Correlation Dimension from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    emb_dim : int
        Embedding dimension
    lag : int
        Lag for time-delay embedding

    Returns:
    --------
    dict
        Dictionary with correlation dimension feature
    """
    features = {}

    try:
        # Calculate Correlation Dimension
        features['corr_dim'] = nolds.corr_dim(eeg_signal, emb_dim=emb_dim, lag=lag)
    except Exception as e:
        print(f"Error calculating Correlation Dimension: {e}")
        features['corr_dim'] = 0

    return features


def extract_recurrence_quantification_analysis(eeg_signal, emb_dim=3, tau=1, epsilon=None):
    """
    Extract Recurrence Quantification Analysis (RQA) features from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    emb_dim : int
        Embedding dimension
    tau : int
        Time delay
    epsilon : float or None
        Threshold distance. If None, 0.1 * signal std is used.

    Returns:
    --------
    dict
        Dictionary with RQA features
    """
    try:
        from pyts.image import RecurrencePlot
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("Warning: pyts package is required for RQA")
        return {}

    features = {}

    try:
        # Normalize signal to [0, 1]
        scaler = MinMaxScaler()
        eeg_scaled = scaler.fit_transform(eeg_signal.reshape(-1, 1)).flatten()

        # Create recurrence plot
        if epsilon is None:
            epsilon = 0.1 * np.std(eeg_signal)

        rp = RecurrencePlot(threshold=epsilon, dimension=emb_dim, time_delay=tau)
        recurrence_matrix = rp.fit_transform(eeg_scaled.reshape(1, -1))[0]

        # Calculate RQA measures

        # Recurrence Rate (RR): Percentage of recurrence points in the RP
        features['rqa_recurrence_rate'] = np.sum(recurrence_matrix) / (len(eeg_signal) ** 2)

        # Determinism (DET): Percentage of recurrence points forming diagonal lines
        diag_lengths = []
        min_diag_len = 2

        for i in range(-len(eeg_signal) + min_diag_len, len(eeg_signal) - min_diag_len + 1):
            diag = np.diag(recurrence_matrix, k=i)

            # Find consecutive 1's in the diagonal
            if len(diag) >= min_diag_len:
                # Convert to string for easier processing
                diag_str = ''.join(diag.astype(int).astype(str))

                # Find all diagonal lines
                import re
                lines = re.findall('1+', diag_str)

                # Add lengths of lines with length >= min_diag_len
                diag_lengths.extend([len(line) for line in lines if len(line) >= min_diag_len])

        if len(diag_lengths) > 0 and np.sum(recurrence_matrix) > 0:
            features['rqa_determinism'] = sum(diag_lengths) / np.sum(recurrence_matrix)
            features['rqa_avg_diag_length'] = np.mean(diag_lengths)
            features['rqa_max_diag_length'] = np.max(diag_lengths)
        else:
            features['rqa_determinism'] = 0
            features['rqa_avg_diag_length'] = 0
            features['rqa_max_diag_length'] = 0

        # Laminarity (LAM): Percentage of recurrence points forming vertical lines
        vert_lengths = []
        min_vert_len = 2

        # Iterate through columns
        for j in range(len(eeg_signal)):
            col = recurrence_matrix[:, j]

            # Convert to string for easier processing
            col_str = ''.join(col.astype(int).astype(str))

            # Find all vertical lines
            import re
            lines = re.findall('1+', col_str)

            # Add lengths of lines with length >= min_vert_len
            vert_lengths.extend([len(line) for line in lines if len(line) >= min_vert_len])

        if len(vert_lengths) > 0 and np.sum(recurrence_matrix) > 0:
            features['rqa_laminarity'] = sum(vert_lengths) / np.sum(recurrence_matrix)
            features['rqa_avg_vert_length'] = np.mean(vert_lengths)
            features['rqa_max_vert_length'] = np.max(vert_lengths)
        else:
            features['rqa_laminarity'] = 0
            features['rqa_avg_vert_length'] = 0
            features['rqa_max_vert_length'] = 0

    except Exception as e:
        print(f"Error calculating RQA: {e}")
        features['rqa_recurrence_rate'] = 0
        features['rqa_determinism'] = 0
        features['rqa_avg_diag_length'] = 0
        features['rqa_max_diag_length'] = 0
        features['rqa_laminarity'] = 0
        features['rqa_avg_vert_length'] = 0
        features['rqa_max_vert_length'] = 0

    return features


def extract_phase_amplitude_coupling(eeg_signal, fs=250):
    """
    Extract Phase-Amplitude Coupling (PAC) features from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary with PAC features
    """
    from scipy.signal import hilbert, butter, filtfilt

    features = {}

    try:
        # Define frequency bands
        phase_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13)
        }

        amplitude_bands = {
            'beta': (13, 30),
            'gamma': (30, 100)
        }

        # Function to bandpass filter the signal
        def bandpass_filter(signal, lowcut, highcut, fs, order=4):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, signal)

        # Calculate PAC for all combinations of phase and amplitude bands
        for phase_band_name, (phase_low, phase_high) in phase_bands.items():
            # Filter signal for phase band
            phase_filtered = bandpass_filter(eeg_signal, phase_low, phase_high, fs)

            # Extract phase using Hilbert transform
            phase_analytic = hilbert(phase_filtered)
            phase = np.angle(phase_analytic)

            for amp_band_name, (amp_low, amp_high) in amplitude_bands.items():
                # Filter signal for amplitude band
                amp_filtered = bandpass_filter(eeg_signal, amp_low, amp_high, fs)

                # Extract amplitude envelope using Hilbert transform
                amp_analytic = hilbert(amp_filtered)
                amplitude = np.abs(amp_analytic)

                # Calculate Modulation Index (MI)
                # Bin phase into 18 bins (20 degrees each)
                n_bins = 18
                phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

                # Calculate mean amplitude in each phase bin
                mean_amp_per_bin = []
                for i in range(n_bins):
                    bin_mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
                    if np.sum(bin_mask) > 0:
                        mean_amp_per_bin.append(np.mean(amplitude[bin_mask]))
                    else:
                        mean_amp_per_bin.append(0)

                # Normalize mean amplitudes to get probability distribution
                total_amp = np.sum(mean_amp_per_bin)
                if total_amp > 0:
                    mean_amp_norm = np.array(mean_amp_per_bin) / total_amp

                    # Calculate entropy
                    uniform = np.ones(n_bins) / n_bins  # Uniform distribution
                    kl_divergence = entropy(mean_amp_norm, uniform)

                    # Modulation Index (MI) is normalized KL divergence
                    max_entropy = np.log(n_bins)
                    mi = kl_divergence / max_entropy
                else:
                    mi = 0

                # Store feature
                features[f'pac_mi_{phase_band_name}_{amp_band_name}'] = mi

    except Exception as e:
        print(f"Error calculating Phase-Amplitude Coupling: {e}")
        for phase_band_name in phase_bands:
            for amp_band_name in amplitude_bands:
                features[f'pac_mi_{phase_band_name}_{amp_band_name}'] = 0

    return features


def extract_all_nonlinear_features(eeg_signal, fs=250):
    """
    Extract all nonlinear features for a single EEG channel.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of all nonlinear features
    """
    # Extract entropy features
    sampen_features = extract_sample_entropy(eeg_signal)
    apen_features = extract_approximate_entropy(eeg_signal)
    perm_features = extract_permutation_entropy(eeg_signal)

    # Extract complexity features
    fd_features = extract_fractal_dimension(eeg_signal)
    lzc_features = extract_lempel_ziv_complexity(eeg_signal)

    # Extract scaling exponent features
    dfa_features = extract_detrended_fluctuation_analysis(eeg_signal)
    hurst_features = extract_hurst_exponent(eeg_signal)

    # Try to extract advanced features
    try:
        mse_features = extract_multiscale_entropy(eeg_signal)
    except Exception as e:
        print(f"Error in multiscale entropy extraction: {e}")
        mse_features = {}

    try:
        lyap_features = extract_lyapunov_exponent(eeg_signal)
    except Exception as e:
        print(f"Error in Lyapunov exponent extraction: {e}")
        lyap_features = {}

    try:
        cd_features = extract_correlation_dimension(eeg_signal)
    except Exception as e:
        print(f"Error in correlation dimension extraction: {e}")
        cd_features = {}

    try:
        rqa_features = extract_recurrence_quantification_analysis(eeg_signal)
    except Exception as e:
        print(f"Error in RQA extraction: {e}")
        rqa_features = {}

    try:
        pac_features = extract_phase_amplitude_coupling(eeg_signal, fs)
    except Exception as e:
        print(f"Error in PAC extraction: {e}")
        pac_features = {}

    # Combine all features
    all_features = {}
    all_features.update(sampen_features)
    all_features.update(apen_features)
    all_features.update(perm_features)
    all_features.update(fd_features)
    all_features.update(lzc_features)
    all_features.update(dfa_features)
    all_features.update(hurst_features)
    all_features.update(mse_features)
    all_features.update(lyap_features)
    all_features.update(cd_features)
    all_features.update(rqa_features)
    all_features.update(pac_features)

    return all_features


def extract_multichannel_nonlinear_features(eeg_signals, fs=250):
    """
    Extract nonlinear features for multiple EEG channels.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    pandas.DataFrame
        DataFrame with all nonlinear features for all channels
    """
    all_channel_features = []

    for ch_idx, eeg_signal in enumerate(eeg_signals):
        # Extract features for this channel
        channel_features = extract_all_nonlinear_features(eeg_signal, fs)

        # Add channel index
        channel_features['channel'] = ch_idx

        all_channel_features.append(channel_features)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_channel_features)

    return features_df