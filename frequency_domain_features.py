# 2nd set of features
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy


def extract_band_powers(eeg_signal, fs=250, bands=None):
    """
    Extract power in different frequency bands from EEG signal.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    bands : dict
        Dictionary of frequency bands in format {name: (min_freq, max_freq)}
        If None, default EEG bands will be used

    Returns:
    --------
    dict
        Dictionary of band powers
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }

    features = {}

    # Compute Power Spectral Density
    freqs, psd = signal.welch(eeg_signal, fs=fs, nperseg=min(256, len(eeg_signal)),
                              window='hann', average='mean')

    # Total power
    total_power = np.sum(psd)
    features['total_power'] = total_power

    # Extract band powers
    for band_name, (min_freq, max_freq) in bands.items():
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)

        # Absolute band power
        band_power = np.sum(psd[freq_mask])
        features[f'abs_{band_name}_power'] = band_power

        # Relative band power
        if total_power > 0:
            rel_band_power = band_power / total_power
        else:
            rel_band_power = 0
        features[f'rel_{band_name}_power'] = rel_band_power

    # Band power ratios (all possible combinations)
    band_names = list(bands.keys())
    for i, band1 in enumerate(band_names):
        for band2 in band_names[i + 1:]:
            abs_band1 = features[f'abs_{band1}_power']
            abs_band2 = features[f'abs_{band2}_power']

            # Avoid division by zero
            if abs_band2 > 0:
                ratio = abs_band1 / abs_band2
            else:
                ratio = 0

            features[f'ratio_{band1}_{band2}'] = ratio

    return features, freqs, psd


def extract_spectral_edge_frequency(eeg_signal, fs=250, percentages=None):
    """
    Extract spectral edge frequencies (frequencies below which X% of signal power is contained).

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    percentages : list
        List of percentages (0-100) for which to calculate SEF

    Returns:
    --------
    dict
        Dictionary of spectral edge frequencies
    """
    if percentages is None:
        percentages = [50, 75, 90, 95]

    features = {}

    # Compute Power Spectral Density
    freqs, psd = signal.welch(eeg_signal, fs=fs, nperseg=min(256, len(eeg_signal)))

    # Calculate cumulative power distribution
    cum_psd = np.cumsum(psd)
    if cum_psd[-1] > 0:  # Avoid division by zero
        cum_psd_norm = cum_psd / cum_psd[-1]
    else:
        # If no power in the signal, set all SEFs to 0
        for p in percentages:
            features[f'sef_{p}'] = 0
        return features

    # Extract spectral edge frequencies
    for p in percentages:
        p_norm = p / 100.0
        idx = np.argmax(cum_psd_norm >= p_norm)
        sef = freqs[idx]
        features[f'sef_{p}'] = sef

    return features


def extract_spectral_moments(eeg_signal, fs=250):
    """
    Extract spectral moments and related features.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of spectral moment features
    """
    features = {}

    # Compute Power Spectral Density
    freqs, psd = signal.welch(eeg_signal, fs=fs, nperseg=min(256, len(eeg_signal)))

    # Normalize PSD
    if np.sum(psd) > 0:
        psd_norm = psd / np.sum(psd)
    else:
        psd_norm = psd

    # Spectral centroid (1st spectral moment)
    if np.sum(psd) > 0:
        features['spectral_centroid'] = np.sum(freqs * psd_norm)
    else:
        features['spectral_centroid'] = 0

    # Spectral variance (2nd spectral moment)
    if np.sum(psd) > 0:
        features['spectral_variance'] = np.sum((freqs - features['spectral_centroid']) ** 2 * psd_norm)
    else:
        features['spectral_variance'] = 0

    # Spectral skewness (3rd spectral moment)
    if features['spectral_variance'] > 0:
        features['spectral_skewness'] = np.sum(((freqs - features['spectral_centroid']) ** 3 * psd_norm) /
                                               (features['spectral_variance'] ** (3 / 2)))
    else:
        features['spectral_skewness'] = 0

    # Spectral kurtosis (4th spectral moment)
    if features['spectral_variance'] > 0:
        features['spectral_kurtosis'] = np.sum(((freqs - features['spectral_centroid']) ** 4 * psd_norm) /
                                               (features['spectral_variance'] ** 2)) - 3
    else:
        features['spectral_kurtosis'] = 0

    # Spectral entropy
    features['spectral_entropy'] = entropy(psd_norm)

    # Spectral edge frequencies
    sef_features = extract_spectral_edge_frequency(eeg_signal, fs)
    features.update(sef_features)

    # Spectral peak
    if len(psd) > 0:
        peak_idx = np.argmax(psd)
        features['peak_frequency'] = freqs[peak_idx]
        features['peak_power'] = psd[peak_idx]
    else:
        features['peak_frequency'] = 0
        features['peak_power'] = 0

    # Spectral slope (in log-log space)
    if len(freqs) > 1 and np.sum(psd) > 0:
        # Only use frequencies > 1 Hz to avoid DC component
        mask = freqs > 1
        if np.sum(mask) > 1:
            log_freqs = np.log10(freqs[mask])
            log_psd = np.log10(psd[mask] + 1e-10)  # Add small constant to avoid log(0)

            # Linear regression
            slope, intercept = np.polyfit(log_freqs, log_psd, 1)
            features['spectral_slope'] = slope
        else:
            features['spectral_slope'] = 0
    else:
        features['spectral_slope'] = 0

    return features


def extract_phase_coherence(eeg_signals, fs=250, bands=None):
    """
    Extract phase coherence between all pairs of EEG channels.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    bands : dict
        Dictionary of frequency bands in format {name: (min_freq, max_freq)}

    Returns:
    --------
    dict
        Dictionary of phase coherence features
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)  # Limit to avoid line noise
        }

    features = {}
    n_channels = len(eeg_signals)

    # For each channel pair
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            ch1 = eeg_signals[i]
            ch2 = eeg_signals[j]

            # Calculate coherence
            f, Cxy = signal.coherence(ch1, ch2, fs=fs, nperseg=min(256, len(ch1)))

            # For each frequency band
            for band_name, (fmin, fmax) in bands.items():
                # Find frequencies in the band
                band_mask = (f >= fmin) & (f <= fmax)

                # Calculate mean coherence in the band
                if np.sum(band_mask) > 0:
                    band_coherence = np.mean(Cxy[band_mask])
                else:
                    band_coherence = 0

                # Store feature
                features[f'coherence_{i}_{j}_{band_name}'] = band_coherence

    return features


def extract_weighted_phase_lag_index(eeg_signals, fs=250, bands=None):
    """
    Extract weighted phase lag index (wPLI) between all pairs of EEG channels.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    bands : dict
        Dictionary of frequency bands in format {name: (min_freq, max_freq)}

    Returns:
    --------
    dict
        Dictionary of wPLI features
    """
    from scipy.signal import hilbert

    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

    features = {}
    n_channels = len(eeg_signals)

    # For each frequency band
    for band_name, (fmin, fmax) in bands.items():
        # Filter signals to band
        filtered_signals = []
        for ch in eeg_signals:
            b, a = signal.butter(4, [fmin, fmax], btype='bandpass', fs=fs)
            filtered_ch = signal.filtfilt(b, a, ch)
            filtered_signals.append(filtered_ch)

        # For each channel pair
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                ch1 = filtered_signals[i]
                ch2 = filtered_signals[j]

                # Compute analytic signal (hilbert transform)
                ch1_analytic = hilbert(ch1)
                ch2_analytic = hilbert(ch2)

                # Compute instantaneous phase
                ch1_phase = np.angle(ch1_analytic)
                ch2_phase = np.angle(ch2_analytic)

                # Phase difference
                phase_diff = ch1_phase - ch2_phase

                # Imaginary part of cross-spectrum
                cross_spectrum = np.exp(1j * phase_diff)
                imag_cs = np.imag(cross_spectrum)

                # Weighted Phase Lag Index
                if np.sum(np.abs(imag_cs)) > 0:
                    wpli = np.abs(np.mean(np.abs(imag_cs) * np.sign(imag_cs))) / np.mean(np.abs(imag_cs))
                else:
                    wpli = 0

                # Store feature
                features[f'wpli_{i}_{j}_{band_name}'] = wpli

    return features


def extract_all_frequency_domain_features(eeg_signal, fs=250):
    """
    Extract all frequency domain features for a single EEG channel.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of all frequency domain features
    """
    # Get band powers
    band_power_features, _, _ = extract_band_powers(eeg_signal, fs)

    # Get spectral moments and related features
    spectral_features = extract_spectral_moments(eeg_signal, fs)

    # Combine all features
    all_features = {}
    all_features.update(band_power_features)
    all_features.update(spectral_features)

    return all_features


def extract_multichannel_frequency_domain_features(eeg_signals, fs=250):
    """
    Extract frequency domain features for multiple EEG channels and between channels.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    pandas.DataFrame
        DataFrame with all frequency domain features
    """
    # Extract features for individual channels
    channel_features = []

    for ch_idx, eeg_signal in enumerate(eeg_signals):
        # Get individual channel features
        features = extract_all_frequency_domain_features(eeg_signal, fs)
        features['channel'] = ch_idx
        channel_features.append(features)

    # Extract cross-channel features
    coherence_features = extract_phase_coherence(eeg_signals, fs)
    wpli_features = extract_weighted_phase_lag_index(eeg_signals, fs)

    # Create cross-channel features dataframe with a single row
    cross_features_df = pd.DataFrame([{**coherence_features, **wpli_features}])

    # Convert individual channel features to dataframe
    channel_features_df = pd.DataFrame(channel_features)

    # Duplicating cross-channel features for each channel
    cross_features_expanded = pd.concat([cross_features_df] * len(channel_features), ignore_index=True)

    # Combine individual and cross-channel features
    all_features_df = pd.concat([channel_features_df, cross_features_expanded], axis=1)

    return all_features_df