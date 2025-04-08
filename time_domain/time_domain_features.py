import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.ar_model import AutoReg


def extract_statistical_features(eeg_signal):
    """
    Extract basic statistical features from a signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal

    Returns:
    --------
    dict
        Dictionary of statistical features
    """
    features = {}
    features['mean'] = np.mean(eeg_signal)
    features['variance'] = np.var(eeg_signal)
    features['std'] = np.std(eeg_signal)
    features['skewness'] = stats.skew(eeg_signal)
    features['kurtosis'] = stats.kurtosis(eeg_signal)
    return features


def extract_amplitude_features(eeg_signal):
    """
    Extract amplitude-based features from a signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal

    Returns:
    --------
    dict
        Dictionary of amplitude features
    """
    features = {}
    features['rms'] = np.sqrt(np.mean(np.square(eeg_signal)))
    features['range'] = np.max(eeg_signal) - np.min(eeg_signal)
    features['max_amplitude'] = np.max(np.abs(eeg_signal))
    features['min_amplitude'] = np.min(eeg_signal)
    features['max_min_ratio'] = np.max(eeg_signal) / (np.min(eeg_signal) + 1e-10)  # Avoid division by zero
    return features


def extract_activity_measures(eeg_signal):
    """
    Extract activity-based measures from a signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal

    Returns:
    --------
    dict
        Dictionary of activity measures
    """
    features = {}
    features['zcr'] = np.sum(np.abs(np.diff(np.signbit(eeg_signal)))) / len(eeg_signal)
    features['line_length'] = np.sum(np.abs(np.diff(eeg_signal)))
    features['area'] = np.sum(np.abs(eeg_signal))
    return features


def extract_hjorth_parameters(eeg_signal):
    """
    Extract Hjorth parameters from a signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal

    Returns:
    --------
    dict
        Dictionary of Hjorth parameters
    """
    features = {}
    diff1 = np.diff(eeg_signal)
    diff2 = np.diff(diff1)

    features['hjorth_activity'] = np.var(eeg_signal)
    features['hjorth_mobility'] = np.sqrt(np.var(diff1) / (np.var(eeg_signal) + 1e-10))
    features['hjorth_complexity'] = np.sqrt(np.var(diff2) * np.var(eeg_signal) / (np.var(diff1) ** 2 + 1e-10))
    return features


def extract_energy_features(eeg_signal):
    """
    Extract energy-based features from a signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal

    Returns:
    --------
    dict
        Dictionary of energy features
    """
    features = {}
    teager = eeg_signal[1:-1] ** 2 - eeg_signal[:-2] * eeg_signal[2:]
    features['teager_energy'] = np.mean(teager)
    return features


def extract_derivative_features(eeg_signal):
    """
    Extract derivative-based features from a signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal

    Returns:
    --------
    dict
        Dictionary of derivative features
    """
    features = {}
    first_deriv = np.diff(eeg_signal)
    features['first_deriv_mean'] = np.mean(first_deriv)
    features['first_deriv_max'] = np.max(np.abs(first_deriv))

    second_deriv = np.diff(first_deriv)
    features['second_deriv_mean'] = np.mean(second_deriv)
    features['second_deriv_max'] = np.max(np.abs(second_deriv))
    return features


def extract_autocorrelation_features(eeg_signal, fs=250):
    """
    Extract autocorrelation-based features from a signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of autocorrelation features
    """
    features = {}
    try:
        autocorr = np.correlate(eeg_signal, eeg_signal, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]

        decay_point = np.where(autocorr < 1 / np.e)[0]
        if len(decay_point) > 0:
            features['norm_decay'] = decay_point[0] / fs
        else:
            features['norm_decay'] = len(autocorr) / fs
    except Exception as e:
        print(f"Error in autocorrelation calculation: {e}")
        features['norm_decay'] = 0

    return features


def extract_autoregressive_features(eeg_signal, order=6):
    """
    Extract autoregressive model coefficients as features.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    order : int
        Order of the AR model

    Returns:
    --------
    dict
        Dictionary of AR coefficients
    """
    features = {}

    try:
        model = AutoReg(eeg_signal, lags=order)
        results = model.fit()

        for i, coef in enumerate(results.params[1:]):
            features[f'ar_coef_{i + 1}'] = coef
    except Exception as e:
        print(f"Error in AR model fitting: {e}")
        for i in range(order):
            features[f'ar_coef_{i + 1}'] = 0

    return features


def extract_peak_features(eeg_signal, fs=250):
    """
    Extract features related to peaks and their intervals.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of peak-related features
    """
    features = {}

    # Find peaks
    pos_peaks, _ = find_peaks(eeg_signal, height=0)
    neg_peaks, _ = find_peaks(-eeg_signal, height=0)

    # Process positive peaks
    if len(pos_peaks) > 1:
        pos_peak_heights = eeg_signal[pos_peaks]
        pos_intervals = np.diff(pos_peaks) / fs

        features['pos_peak_count'] = len(pos_peaks)
        features['pos_peak_mean_height'] = np.mean(pos_peak_heights)
        features['pos_peak_max_height'] = np.max(pos_peak_heights)
        features['pos_inter_peak_mean'] = np.mean(pos_intervals)
        features['pos_inter_peak_std'] = np.std(pos_intervals)
    else:
        features['pos_peak_count'] = 0
        features['pos_peak_mean_height'] = 0
        features['pos_peak_max_height'] = 0
        features['pos_inter_peak_mean'] = 0
        features['pos_inter_peak_std'] = 0

    # Process negative peaks
    if len(neg_peaks) > 1:
        neg_peak_heights = eeg_signal[neg_peaks]
        neg_intervals = np.diff(neg_peaks) / fs

        features['neg_peak_count'] = len(neg_peaks)
        features['neg_peak_mean_height'] = np.mean(neg_peak_heights)
        features['neg_peak_max_height'] = np.min(neg_peak_heights)
        features['neg_inter_peak_mean'] = np.mean(neg_intervals)
        features['neg_inter_peak_std'] = np.std(neg_intervals)
    else:
        features['neg_peak_count'] = 0
        features['neg_peak_mean_height'] = 0
        features['neg_peak_max_height'] = 0
        features['neg_inter_peak_mean'] = 0
        features['neg_inter_peak_std'] = 0

    # Process all peaks combined
    if len(pos_peaks) > 0 and len(neg_peaks) > 0:
        all_peaks = np.sort(np.concatenate([pos_peaks, neg_peaks]))
        all_intervals = np.diff(all_peaks) / fs

        features['all_peak_count'] = len(all_peaks)
        features['all_inter_peak_mean'] = np.mean(all_intervals)
        features['all_inter_peak_std'] = np.std(all_intervals)
    else:
        features['all_peak_count'] = 0
        features['all_inter_peak_mean'] = 0
        features['all_inter_peak_std'] = 0

    return features


def extract_time_domain_features(eeg_signal, fs=250, ar_order=6):
    """
    Extract all time domain features for a single EEG channel.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    ar_order : int
        Order of the autoregressive model

    Returns:
    --------
    dict
        Dictionary of all time domain features
    """
    # Initialize features dictionary
    features = {}

    # Add statistical features
    features.update(extract_statistical_features(eeg_signal))

    # Add amplitude features
    features.update(extract_amplitude_features(eeg_signal))

    # Add activity measures
    features.update(extract_activity_measures(eeg_signal))

    # Add Hjorth parameters
    features.update(extract_hjorth_parameters(eeg_signal))

    # Add energy features
    features.update(extract_energy_features(eeg_signal))

    # Add derivative features
    features.update(extract_derivative_features(eeg_signal))

    # Add autocorrelation features
    features.update(extract_autocorrelation_features(eeg_signal, fs))

    # Add autoregressive features
    features.update(extract_autoregressive_features(eeg_signal, ar_order))

    # Add peak-related features
    features.update(extract_peak_features(eeg_signal, fs))

    return features


def extract_ekg_specific_features(ekg_signal, fs=250):
    """
    Extract features specific to EKG signals.

    Parameters:
    -----------
    ekg_signal : numpy.ndarray
        EKG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of EKG-specific features
    """
    features = {}

    # Find R peaks in the EKG signal
    try:
        r_peaks, properties = find_peaks(ekg_signal, height=5, distance=fs / 2, prominence=5)

        if len(r_peaks) > 1:
            # Calculate heart rate from R-R intervals
            rr_intervals = np.diff(r_peaks) / fs  # Convert to seconds
            hr_values = 60 / rr_intervals  # Convert to BPM

            features['heart_rate_mean'] = np.mean(hr_values)
            features['heart_rate_std'] = np.std(hr_values)
            features['heart_rate_min'] = np.min(hr_values)
            features['heart_rate_max'] = np.max(hr_values)

            # Heart rate variability measures
            features['hrv_sdnn'] = np.std(rr_intervals) * 1000  # SDNN in ms
            features['hrv_rmssd'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) * 1000  # RMSSD in ms

            # QRS amplitude
            features['qrs_amplitude_mean'] = np.mean(properties['peak_heights'])
            features['qrs_amplitude_std'] = np.std(properties['peak_heights'])
        else:
            # Default values if not enough peaks found
            for feature in ['heart_rate_mean', 'heart_rate_std', 'heart_rate_min', 'heart_rate_max',
                            'hrv_sdnn', 'hrv_rmssd', 'qrs_amplitude_mean', 'qrs_amplitude_std']:
                features[feature] = 0
    except Exception as e:
        print(f"Error in EKG feature extraction: {e}")
        for feature in ['heart_rate_mean', 'heart_rate_std', 'heart_rate_min', 'heart_rate_max',
                        'hrv_sdnn', 'hrv_rmssd', 'qrs_amplitude_mean', 'qrs_amplitude_std']:
            features[feature] = 0

    return features


def calculate_correlation_features(eeg_data, channel_map=None):
    """
    Calculate correlation-based features between all channel pairs.

    Parameters:
    -----------
    eeg_data : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    channel_map : dict, optional
        Dictionary mapping channel indices to channel names

    Returns:
    --------
    dict
        Dictionary of correlation features
    """
    n_channels = eeg_data.shape[0]
    features = {}

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            # Create feature name based on channel names if available
            if channel_map:
                feature_name = f"corr_{channel_map[i]}_{channel_map[j]}"
            else:
                feature_name = f"corr_ch{i}_ch{j}"

            # Calculate correlation coefficient
            features[feature_name] = np.corrcoef(eeg_data[i], eeg_data[j])[0, 1]

    return features


def extract_all_time_domain_features(eeg_data, channel_map=None, fs=250, ar_order=6, include_correlations=True):
    """
    Extract all time domain features for multiple EEG channels.

    Parameters:
    -----------
    eeg_data : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    channel_map : dict, optional
        Dictionary mapping channel indices to channel names
    fs : float
        Sampling frequency in Hz
    ar_order : int
        Order of the autoregressive model
    include_correlations : bool
        Whether to include channel correlations

    Returns:
    --------
    dict
        Dictionary with all time domain features
    """
    # Initialize features dictionary
    all_features = {}

    # Loop through each channel
    for ch_idx in range(eeg_data.shape[0]):
        # Get channel name if available
        if channel_map and ch_idx in channel_map:
            ch_name = channel_map[ch_idx]
        else:
            ch_name = f"channel_{ch_idx}"

        # Check if this is an EKG channel
        is_ekg = (ch_name == 'EKG') if channel_map else False

        # Extract single-channel features
        channel_features = extract_time_domain_features(eeg_data[ch_idx], fs, ar_order)

        # Add EKG-specific features if this is the EKG channel
        if is_ekg:
            ekg_features = extract_ekg_specific_features(eeg_data[ch_idx], fs)
            channel_features.update(ekg_features)

        # Add features to all_features with channel prefix
        for feature_name, feature_value in channel_features.items():
            all_features[f"{ch_name}_{feature_name}"] = feature_value

    # Add cross-channel correlation features if requested
    if include_correlations:
        correlation_features = calculate_correlation_features(eeg_data, channel_map)
        all_features.update(correlation_features)

    return all_features


def extract_symmetry_features(eeg_data, channel_map):
    """
    Calculate symmetry features between homologous channel pairs.

    Parameters:
    -----------
    eeg_data : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    channel_map : dict
        Dictionary mapping channel indices to channel names

    Returns:
    --------
    dict
        Dictionary of symmetry features
    """
    # Define homologous channel pairs (left-right)
    homologous_pairs = [
        ('Fp1', 'Fp2'),
        ('F3', 'F4'),
        ('C3', 'C4'),
        ('P3', 'P4'),
        ('O1', 'O2'),
        ('F7', 'F8'),
        ('T3', 'T4'),
        ('T5', 'T6')
    ]

    # Create reverse mapping from channel names to indices
    name_to_idx = {name: idx for idx, name in channel_map.items()}

    features = {}

    # Calculate features for each pair
    for left_ch, right_ch in homologous_pairs:
        # Check if both channels exist in the mapping
        if left_ch in name_to_idx and right_ch in name_to_idx:
            left_idx = name_to_idx[left_ch]
            right_idx = name_to_idx[right_ch]

            # Calculate asymmetry ratio for various features
            left_features = extract_time_domain_features(eeg_data[left_idx])
            right_features = extract_time_domain_features(eeg_data[right_idx])

            # Calculate asymmetry for key features
            for feature in ['hjorth_activity', 'hjorth_mobility', 'teager_energy', 'rms']:
                if feature in left_features and feature in right_features:
                    # Asymmetry ratio (L-R)/(L+R)
                    numer = left_features[feature] - right_features[feature]
                    denom = left_features[feature] + right_features[feature]
                    if abs(denom) > 1e-10:  # Avoid division by zero
                        asym = numer / denom
                    else:
                        asym = 0
                    features[f"asym_{left_ch}_{right_ch}_{feature}"] = asym

    return features


def generate_time_domain_feature_df(eeg_data, channel_map=None, fs=250, ar_order=6,
                                    include_correlations=True, include_symmetry=True):
    """
    Generate a DataFrame with all time domain features for multiple EEG channels.

    Parameters:
    -----------
    eeg_data : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    channel_map : dict, optional
        Dictionary mapping channel indices to channel names
    fs : float
        Sampling frequency in Hz
    ar_order : int
        Order of the autoregressive model
    include_correlations : bool
        Whether to include channel correlations
    include_symmetry : bool
        Whether to include symmetry features

    Returns:
    --------
    pandas.DataFrame
        DataFrame with a single row containing all time domain features
    """
    # Extract all features
    features = extract_all_time_domain_features(eeg_data, channel_map, fs, ar_order, include_correlations)

    # Add symmetry features if requested and channel map is available
    if include_symmetry and channel_map:
        symmetry_features = extract_symmetry_features(eeg_data, channel_map)
        features.update(symmetry_features)

    # Convert to DataFrame
    features_df = pd.DataFrame([features])

    return features_df


def calculate_regional_features(eeg_data, channel_map):
    """
    Calculate aggregated features by brain region.

    Parameters:
    -----------
    eeg_data : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    channel_map : dict
        Dictionary mapping channel indices to channel names

    Returns:
    --------
    dict
        Dictionary of regional features
    """
    # Define regions and their corresponding channels
    regions = {
        'frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
        'central': ['C3', 'C4', 'Cz'],
        'temporal': ['T3', 'T4', 'T5', 'T6'],
        'parietal': ['P3', 'P4', 'Pz'],
        'occipital': ['O1', 'O2']
    }

    # Create reverse mapping from channel names to indices
    name_to_idx = {name: idx for idx, name in channel_map.items()}

    features = {}

    # Extract features for key measures by region
    key_features = ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
                    'teager_energy', 'line_length', 'rms']

    for region_name, channels in regions.items():
        # Get indices of channels in this region
        region_indices = [name_to_idx[ch] for ch in channels if ch in name_to_idx]

        if region_indices:
            # Extract features for each channel in the region
            region_channel_features = []
            for idx in region_indices:
                region_channel_features.append(extract_time_domain_features(eeg_data[idx]))

            # Calculate aggregate statistics for each feature across the region
            for feature in key_features:
                feature_values = [ch_feat[feature] for ch_feat in region_channel_features
                                  if feature in ch_feat]

                if feature_values:
                    features[f"region_{region_name}_{feature}_mean"] = np.mean(feature_values)
                    features[f"region_{region_name}_{feature}_std"] = np.std(feature_values)

    return features


if __name__ == "__main__":
    pass