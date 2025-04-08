import numpy as np
import pandas as pd
import librosa
from scipy import signal
from scipy.stats import entropy, skew, kurtosis


def extract_mel_spectrogram(eeg_signal, fs=250, n_mels=64, n_fft=512, hop_length=None):
    """
    Extract Mel spectrogram from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    n_mels : int
        Number of Mel bands to generate
    n_fft : int
        Length of the FFT window
    hop_length : int or None
        Number of samples between successive frames.
        If None, hop_length = n_fft // 4

    Returns:
    --------
    tuple
        (mel_spectrogram, frequencies, times)
    """
    # Set default hop length if not provided
    if hop_length is None:
        hop_length = n_fft // 4

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=eeg_signal,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0.5,  # Lowest EEG frequency of interest
        fmax=min(fs / 2, 100)  # Highest EEG frequency of interest, capped at 100 Hz
    )

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Get frequency and time axes
    frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0.5, fmax=min(fs / 2, 100))
    times = librosa.times_like(mel_spec, sr=fs, hop_length=hop_length)

    return mel_spec_db, frequencies, times


def extract_mel_band_statistics(mel_spec, frequencies):
    """
    Extract statistical features from each Mel band.

    Parameters:
    -----------
    mel_spec : numpy.ndarray
        Mel spectrogram with shape (n_mels, n_frames)
    frequencies : numpy.ndarray
        Frequency values for each Mel band

    Returns:
    --------
    dict
        Dictionary of Mel band statistics
    """
    features = {}
    n_mels, n_frames = mel_spec.shape

    # Define EEG frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    # Calculate statistics for each Mel band
    for i in range(n_mels):
        mel_band = mel_spec[i, :]
        freq = frequencies[i]

        # Basic statistics
        features[f'mel_band_{i}_mean'] = np.mean(mel_band)
        features[f'mel_band_{i}_std'] = np.std(mel_band)
        features[f'mel_band_{i}_min'] = np.min(mel_band)
        features[f'mel_band_{i}_max'] = np.max(mel_band)
        features[f'mel_band_{i}_range'] = np.max(mel_band) - np.min(mel_band)
        features[f'mel_band_{i}_energy'] = np.sum(mel_band ** 2)

        # Higher-order statistics
        features[f'mel_band_{i}_skewness'] = skew(mel_band)
        features[f'mel_band_{i}_kurtosis'] = kurtosis(mel_band)

        # Entropy
        if np.sum(mel_band ** 2) > 0:
            normalized_band = mel_band ** 2 / np.sum(mel_band ** 2)
            features[f'mel_band_{i}_entropy'] = entropy(normalized_band)
        else:
            features[f'mel_band_{i}_entropy'] = 0

        # Flatness
        if np.any(mel_band > 0):
            features[f'mel_band_{i}_flatness'] = np.exp(np.mean(np.log(mel_band + 1e-10))) / (np.mean(mel_band) + 1e-10)
        else:
            features[f'mel_band_{i}_flatness'] = 0

    # Calculate statistics for each EEG frequency band by grouping Mel bands
    for band_name, (fmin, fmax) in bands.items():
        band_indices = [i for i, freq in enumerate(frequencies) if fmin <= freq <= fmax]

        if band_indices:
            band_data = mel_spec[band_indices, :]

            # Average power in the band
            band_power = np.mean(band_data ** 2)
            features[f'mel_{band_name}_power'] = band_power

            # Temporal variation of band power
            temporal_std = np.std(np.mean(band_data, axis=0))
            features[f'mel_{band_name}_temporal_std'] = temporal_std

            # Entropy of band power distribution
            if band_power > 0:
                band_power_normalized = band_data ** 2 / np.sum(band_data ** 2)
                features[f'mel_{band_name}_entropy'] = entropy(np.mean(band_power_normalized, axis=1))
            else:
                features[f'mel_{band_name}_entropy'] = 0
        else:
            features[f'mel_{band_name}_power'] = 0
            features[f'mel_{band_name}_temporal_std'] = 0
            features[f'mel_{band_name}_entropy'] = 0

    return features


def extract_mel_temporal_features(mel_spec):
    """
    Extract temporal evolution features from Mel spectrogram.

    Parameters:
    -----------
    mel_spec : numpy.ndarray
        Mel spectrogram with shape (n_mels, n_frames)

    Returns:
    --------
    dict
        Dictionary of temporal features
    """
    features = {}
    n_mels, n_frames = mel_spec.shape

    if n_frames < 2:
        print("Not enough frames for temporal analysis")
        return {}

    # Calculate temporal centroid
    time_axis = np.arange(n_frames)
    spectral_energy = np.sum(mel_spec ** 2, axis=0)

    if np.sum(spectral_energy) > 0:
        # Temporal centroid
        temporal_centroid = np.sum(time_axis * spectral_energy) / np.sum(spectral_energy)
        features['temporal_centroid'] = temporal_centroid / n_frames  # Normalize to [0, 1]

        # Temporal spread
        temporal_spread = np.sqrt(
            np.sum(((time_axis - temporal_centroid) ** 2) * spectral_energy) / np.sum(spectral_energy))
        features['temporal_spread'] = temporal_spread / n_frames  # Normalize to [0, 1]

        # Temporal skewness
        temporal_skewness = np.sum(((time_axis - temporal_centroid) ** 3) * spectral_energy) / (
                    np.sum(spectral_energy) * (temporal_spread ** 3))
        features['temporal_skewness'] = temporal_skewness

        # Temporal kurtosis
        temporal_kurtosis = np.sum(((time_axis - temporal_centroid) ** 4) * spectral_energy) / (
                    np.sum(spectral_energy) * (temporal_spread ** 4)) - 3
        features['temporal_kurtosis'] = temporal_kurtosis

        # Temporal flatness
        if np.any(spectral_energy > 0):
            temporal_flatness = np.exp(np.mean(np.log(spectral_energy + 1e-10))) / (np.mean(spectral_energy) + 1e-10)
            features['temporal_flatness'] = temporal_flatness
        else:
            features['temporal_flatness'] = 0

        # Temporal entropy
        temporal_entropy = entropy(spectral_energy / np.sum(spectral_energy))
        features['temporal_entropy'] = temporal_entropy
    else:
        features['temporal_centroid'] = 0.5  # Middle of the signal
        features['temporal_spread'] = 0
        features['temporal_skewness'] = 0
        features['temporal_kurtosis'] = 0
        features['temporal_flatness'] = 0
        features['temporal_entropy'] = 0

    # Calculate temporal evolution by dividing the spectrogram into segments
    n_segments = min(4, n_frames // 5)  # At least 5 frames per segment

    if n_segments >= 2:
        segment_length = n_frames // n_segments
        segment_energies = []

        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length if i < n_segments - 1 else n_frames
            segment = mel_spec[:, start_idx:end_idx]
            segment_energies.append(np.sum(segment ** 2))

        # Calculate evolution metrics
        if np.sum(segment_energies) > 0:
            # Normalized segment energies
            norm_energies = np.array(segment_energies) / np.sum(segment_energies)

            # Energy evolution (trend)
            evolution_trend = np.polyfit(np.arange(n_segments), norm_energies, 1)[0] * n_segments
            features['energy_evolution_trend'] = evolution_trend

            # Energy ratio max/min
            if min(segment_energies) > 0:
                features['energy_max_min_ratio'] = max(segment_energies) / min(segment_energies)
            else:
                features['energy_max_min_ratio'] = 1

            # Energy variance over time
            features['energy_temporal_variance'] = np.var(norm_energies)
        else:
            features['energy_evolution_trend'] = 0
            features['energy_max_min_ratio'] = 1
            features['energy_temporal_variance'] = 0
    else:
        features['energy_evolution_trend'] = 0
        features['energy_max_min_ratio'] = 1
        features['energy_temporal_variance'] = 0

    return features


def extract_mel_contrast_features(mel_spec):
    """
    Extract contrast features from Mel spectrogram.

    Parameters:
    -----------
    mel_spec : numpy.ndarray
        Mel spectrogram with shape (n_mels, n_frames)

    Returns:
    --------
    dict
        Dictionary of contrast features
    """
    features = {}
    n_mels, n_frames = mel_spec.shape

    if n_mels < 2:
        print("Not enough Mel bands for contrast analysis")
        return {}

    # Calculate spectral contrast
    # Difference between peak and valley
    peak_spectrum = np.max(mel_spec, axis=1)
    valley_spectrum = np.min(mel_spec, axis=1)
    contrast = peak_spectrum - valley_spectrum

    # Calculate contrast features
    features['mel_contrast_mean'] = np.mean(contrast)
    features['mel_contrast_std'] = np.std(contrast)
    features['mel_contrast_max'] = np.max(contrast)

    # Calculate contrast in different frequency regions
    # Divide Mel bands into regions
    n_regions = min(4, n_mels // 4)  # At least 4 Mel bands per region

    if n_regions >= 2:
        region_length = n_mels // n_regions
        for i in range(n_regions):
            start_idx = i * region_length
            end_idx = start_idx + region_length if i < n_regions - 1 else n_mels
            region_contrast = contrast[start_idx:end_idx]

            features[f'mel_contrast_region_{i}_mean'] = np.mean(region_contrast)
            features[f'mel_contrast_region_{i}_std'] = np.std(region_contrast)

    # Calculate temporal contrast (how contrast changes over time)
    if n_frames >= 2:
        # For each frame, calculate the contrast across frequency bands
        frame_contrasts = []
        for i in range(n_frames):
            frame = mel_spec[:, i]
            frame_contrast = np.max(frame) - np.min(frame)
            frame_contrasts.append(frame_contrast)

        features['mel_temporal_contrast_mean'] = np.mean(frame_contrasts)
        features['mel_temporal_contrast_std'] = np.std(frame_contrasts)
        features['mel_temporal_contrast_max'] = np.max(frame_contrasts)

        if n_frames >= 4:
            # Calculate how contrast evolves over time
            contrast_trend = np.polyfit(np.arange(n_frames), frame_contrasts, 1)[0] * n_frames
            features['mel_contrast_evolution'] = contrast_trend
        else:
            features['mel_contrast_evolution'] = 0
    else:
        features['mel_temporal_contrast_mean'] = 0
        features['mel_temporal_contrast_std'] = 0
        features['mel_temporal_contrast_max'] = 0
        features['mel_contrast_evolution'] = 0

    return features


def extract_mel_band_correlations(mel_spec, frequencies):
    """
    Extract correlation features between different Mel bands.

    Parameters:
    -----------
    mel_spec : numpy.ndarray
        Mel spectrogram with shape (n_mels, n_frames)
    frequencies : numpy.ndarray
        Frequency values for each Mel band

    Returns:
    --------
    dict
        Dictionary of correlation features
    """
    features = {}
    n_mels, n_frames = mel_spec.shape

    if n_mels < 2 or n_frames < 2:
        print("Not enough data for correlation analysis")
        return {}

    # Define EEG frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    # Extract band indices
    band_indices = {}
    for band_name, (fmin, fmax) in bands.items():
        band_indices[band_name] = [i for i, freq in enumerate(frequencies) if fmin <= freq <= fmax]

    # Calculate mean activity in each band over time
    band_activities = {}
    for band_name, indices in band_indices.items():
        if indices:
            band_activities[band_name] = np.mean(mel_spec[indices, :], axis=0)

    # Calculate correlations between band activities
    for band1 in band_activities.keys():
        for band2 in band_activities.keys():
            if band1 < band2:  # To avoid duplicates
                corr = np.corrcoef(band_activities[band1], band_activities[band2])[0, 1]
                features[f'mel_corr_{band1}_{band2}'] = corr

    # Calculate overall correlation structure
    all_correlations = []
    for i in range(n_mels):
        for j in range(i + 1, n_mels):
            corr = np.corrcoef(mel_spec[i, :], mel_spec[j, :])[0, 1]
            all_correlations.append(corr)

    if all_correlations:
        features['mel_mean_correlation'] = np.mean(all_correlations)
        features['mel_std_correlation'] = np.std(all_correlations)
        features['mel_max_correlation'] = np.max(all_correlations)
        features['mel_min_correlation'] = np.min(all_correlations)
    else:
        features['mel_mean_correlation'] = 0
        features['mel_std_correlation'] = 0
        features['mel_max_correlation'] = 0
        features['mel_min_correlation'] = 0

    return features


def extract_all_mel_spectrogram_features(eeg_signal, fs=250, n_mels=64, n_fft=512, hop_length=None):
    """
    Extract all Mel spectrogram features from an EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    n_mels : int
        Number of Mel bands to generate
    n_fft : int
        Length of the FFT window
    hop_length : int or None
        Number of samples between successive frames.
        If None, hop_length = n_fft // 4

    Returns:
    --------
    dict
        Dictionary of all Mel spectrogram features
    """
    # Extract Mel spectrogram
    mel_spec, frequencies, times = extract_mel_spectrogram(eeg_signal, fs, n_mels, n_fft, hop_length)

    # Get different types of features
    band_features = extract_mel_band_statistics(mel_spec, frequencies)
    temporal_features = extract_mel_temporal_features(mel_spec)
    contrast_features = extract_mel_contrast_features(mel_spec)
    correlation_features = extract_mel_band_correlations(mel_spec, frequencies)

    # Combine all features
    all_features = {}
    all_features.update(band_features)
    all_features.update(temporal_features)
    all_features.update(contrast_features)
    all_features.update(correlation_features)

    # Add a few extra overall features

    # Global spectral centroid
    if np.sum(mel_spec ** 2) > 0:
        all_features['mel_spectral_centroid'] = np.sum(frequencies.reshape(-1, 1) * mel_spec ** 2) / np.sum(
            mel_spec ** 2)
    else:
        all_features['mel_spectral_centroid'] = np.mean(frequencies)

    # Global spectral bandwidth
    if 'mel_spectral_centroid' in all_features and np.sum(mel_spec ** 2) > 0:
        all_features['mel_spectral_bandwidth'] = np.sqrt(np.sum(
            ((frequencies.reshape(-1, 1) - all_features['mel_spectral_centroid']) ** 2) * mel_spec ** 2) / np.sum(
            mel_spec ** 2))
    else:
        all_features['mel_spectral_bandwidth'] = 0

    # Global spectral flatness
    log_geometric_mean = np.mean(np.log(np.mean(mel_spec, axis=1) + 1e-10))
    arithmetic_mean = np.mean(np.mean(mel_spec, axis=1))
    if arithmetic_mean > 0:
        all_features['mel_spectral_flatness'] = np.exp(log_geometric_mean) / arithmetic_mean
    else:
        all_features['mel_spectral_flatness'] = 0

    # Global spectral entropy
    if np.sum(mel_spec ** 2) > 0:
        spectral_distribution = np.sum(mel_spec ** 2, axis=1) / np.sum(mel_spec ** 2)
        all_features['mel_spectral_entropy'] = entropy(spectral_distribution)
    else:
        all_features['mel_spectral_entropy'] = 0

    return all_features, mel_spec, frequencies, times


def extract_multichannel_mel_spectrogram_features(eeg_signals, fs=250, n_mels=64, n_fft=512, hop_length=None):
    """
    Extract Mel spectrogram features for multiple EEG channels.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    n_mels : int
        Number of Mel bands to generate
    n_fft : int
        Length of the FFT window
    hop_length : int or None
        Number of samples between successive frames.
        If None, hop_length = n_fft // 4

    Returns:
    --------
    pandas.DataFrame
        DataFrame with Mel spectrogram features for all channels
    """
    all_channel_features = []

    for ch_idx, eeg_signal in enumerate(eeg_signals):
        # Extract features for this channel
        channel_features, _, _, _ = extract_all_mel_spectrogram_features(eeg_signal, fs, n_mels, n_fft, hop_length)

        # Add channel index
        channel_features['channel'] = ch_idx

        all_channel_features.append(channel_features)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_channel_features)

    # Add cross-channel summary statistics
    if len(all_channel_features) > 1:
        # Calculate mean, std for each feature across channels
        features_cols = features_df.columns[features_df.columns != 'channel']

        summary_features = {}
        for col in features_cols:
            # Only process numeric features
            if pd.api.types.is_numeric_dtype(features_df[col]):
                summary_features[f'mean_{col}'] = features_df[col].mean()
                summary_features[f'std_{col}'] = features_df[col].std()

        # Add summary features to each row
        for col, value in summary_features.items():
            features_df[col] = value

    return features_df