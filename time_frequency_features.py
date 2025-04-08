import numpy as np
import pandas as pd
import pywt
from scipy import signal
from scipy.stats import entropy


def extract_wavelet_coefficients(eeg_signal, wavelet='db4', levels=5):
    """
    Extract wavelet coefficients using Discrete Wavelet Transform.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    wavelet : str
        Wavelet type (e.g., 'db4', 'sym8', 'coif3')
    levels : int
        Number of decomposition levels

    Returns:
    --------
    dict
        Dictionary of wavelet coefficient features
    """
    features = {}

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(eeg_signal, wavelet=wavelet, level=levels)

    # Extract features from each level
    for i, coef in enumerate(coeffs):
        if i == 0:
            name = 'approx'  # Approximation coefficients
        else:
            name = f'detail_{levels + 1 - i}'  # Detail coefficients

        # Basic statistics of coefficients
        features[f'wavelet_{name}_mean'] = np.mean(coef)
        features[f'wavelet_{name}_std'] = np.std(coef)
        features[f'wavelet_{name}_energy'] = np.sum(coef ** 2)
        features[f'wavelet_{name}_abs_mean'] = np.mean(np.abs(coef))
        features[f'wavelet_{name}_max'] = np.max(np.abs(coef))

    # Calculate relative energies
    total_energy = sum([features[f'wavelet_{lvl}_energy'] for lvl in
                        ['approx'] + [f'detail_{j}' for j in range(1, levels + 1)]])

    if total_energy > 0:
        features[f'wavelet_approx_rel_energy'] = features[f'wavelet_approx_energy'] / total_energy

        for j in range(1, levels + 1):
            features[f'wavelet_detail_{j}_rel_energy'] = features[f'wavelet_detail_{j}_energy'] / total_energy
    else:
        features[f'wavelet_approx_rel_energy'] = 0
        for j in range(1, levels + 1):
            features[f'wavelet_detail_{j}_rel_energy'] = 0

    return features


def extract_wavelet_entropy(eeg_signal, wavelet='db4', levels=5):
    """
    Extract entropy measures from wavelet decomposition.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    wavelet : str
        Wavelet type (e.g., 'db4', 'sym8', 'coif3')
    levels : int
        Number of decomposition levels

    Returns:
    --------
    dict
        Dictionary of wavelet entropy features
    """
    features = {}

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(eeg_signal, wavelet=wavelet, level=levels)

    # Calculate energies at each level
    energies = [np.sum(coef ** 2) for coef in coeffs]
    total_energy = sum(energies)

    # Calculate wavelet entropy
    if total_energy > 0:
        # Normalize energies to get probability distribution
        probs = np.array(energies) / total_energy

        # Avoid log(0) issues
        probs = probs[probs > 0]

        # Shannon entropy
        features['wavelet_shannon_entropy'] = -np.sum(probs * np.log2(probs))

        # Log-energy entropy
        features['wavelet_log_energy'] = np.sum(np.log2(energies + 1e-10))
    else:
        features['wavelet_shannon_entropy'] = 0
        features['wavelet_log_energy'] = 0

    return features


def extract_spectrogram_features(eeg_signal, fs=250, nperseg=64, noverlap=None):
    """
    Extract features from the spectrogram.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    nperseg : int
        Length of each segment in samples
    noverlap : int or None
        Number of samples to overlap between segments. If None, noverlap = nperseg // 2

    Returns:
    --------
    dict
        Dictionary of spectrogram features
    """
    features = {}

    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(eeg_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Energy in spectrogram
    features['spec_total_energy'] = np.sum(Sxx)

    # Contrast: difference between max and min values
    features['spec_contrast'] = np.max(Sxx) - np.min(Sxx)

    # Spectral entropy over time
    entropy_over_time = []
    for i in range(Sxx.shape[1]):
        if np.sum(Sxx[:, i]) > 0:
            p = Sxx[:, i] / np.sum(Sxx[:, i])
            entropy_over_time.append(entropy(p))

    if entropy_over_time:
        features['spec_entropy_mean'] = np.mean(entropy_over_time)
        features['spec_entropy_std'] = np.std(entropy_over_time)
        features['spec_entropy_max'] = np.max(entropy_over_time)
        features['spec_entropy_min'] = np.min(entropy_over_time)
    else:
        features['spec_entropy_mean'] = 0
        features['spec_entropy_std'] = 0
        features['spec_entropy_max'] = 0
        features['spec_entropy_min'] = 0

    # Energy concentration
    if np.sum(Sxx) > 0:
        # Sort spectrogram values in descending order
        sorted_values = np.sort(Sxx.flatten())[::-1]
        # Calculate cumulative sum
        cum_energy = np.cumsum(sorted_values)
        # Normalize
        cum_energy_norm = cum_energy / cum_energy[-1]

        # Percentage of coefficients needed to capture X% of energy
        for percent in [50, 75, 90, 95]:
            idx = np.argmax(cum_energy_norm >= percent / 100)
            features[f'spec_concentration_{percent}'] = idx / len(sorted_values)
    else:
        for percent in [50, 75, 90, 95]:
            features[f'spec_concentration_{percent}'] = 0

    # Compute frequency-band energies over time
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, min(100, fs / 2))
    }

    # For each band
    for band_name, (fmin, fmax) in bands.items():
        # Find frequencies in the band
        band_mask = (f >= fmin) & (f <= fmax)

        if np.sum(band_mask) > 0:
            # Energy in this band over time
            band_energy = np.sum(Sxx[band_mask, :], axis=0)

            # Calculate statistics
            features[f'spec_{band_name}_energy_mean'] = np.mean(band_energy)
            features[f'spec_{band_name}_energy_std'] = np.std(band_energy)
            features[f'spec_{band_name}_energy_max'] = np.max(band_energy)

            # Calculate temporal variation (changes over time)
            features[f'spec_{band_name}_temporal_var'] = np.var(band_energy)
        else:
            features[f'spec_{band_name}_energy_mean'] = 0
            features[f'spec_{band_name}_energy_std'] = 0
            features[f'spec_{band_name}_energy_max'] = 0
            features[f'spec_{band_name}_temporal_var'] = 0

    return features


def extract_morlet_wavelet_features(eeg_signal, fs=250, freqs=None):
    """
    Extract features using Morlet wavelet transform.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    freqs : numpy.ndarray or None
        Frequencies to use for the wavelet transform

    Returns:
    --------
    dict
        Dictionary of Morlet wavelet features
    """
    from mne.time_frequency import tfr_array_morlet

    features = {}

    # Define frequencies to analyze if not provided
    if freqs is None:
        freqs = np.arange(1, min(100, fs / 2), 1)  # 1 Hz to Nyquist frequency

    # Set signal dimension (1 channel, rest is time)
    signal = eeg_signal.reshape(1, 1, -1)

    # Compute time-frequency representation
    tfr = tfr_array_morlet(signal, sfreq=fs, freqs=freqs,
                           n_cycles=freqs / 2, output='power')

    # Remove singleton dimensions
    tfr = np.squeeze(tfr)  # Shape: (n_freqs, n_times)

    # EEG band definitions
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, min(100, fs / 2))
    }

    # Extract features for each frequency band
    for band_name, (fmin, fmax) in bands.items():
        # Find indices of frequencies in the band
        band_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]

        if len(band_idx) > 0:
            # Calculate average power in the band over time
            band_power = np.mean(tfr[band_idx, :], axis=0)

            # Statistical features
            features[f'morlet_{band_name}_mean'] = np.mean(band_power)
            features[f'morlet_{band_name}_std'] = np.std(band_power)
            features[f'morlet_{band_name}_max'] = np.max(band_power)

            # Temporal dynamics
            features[f'morlet_{band_name}_temporal_var'] = np.var(band_power)
        else:
            features[f'morlet_{band_name}_mean'] = 0
            features[f'morlet_{band_name}_std'] = 0
            features[f'morlet_{band_name}_max'] = 0
            features[f'morlet_{band_name}_temporal_var'] = 0

    return features


def extract_hilbert_huang_transform(eeg_signal, fs=250):
    """
    Extract features using Hilbert-Huang Transform (Empirical Mode Decomposition).

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of HHT features
    """
    try:
        from PyEMD import EMD
    except ImportError:
        print("Warning: PyEMD package is required for Hilbert-Huang Transform")
        return {}

    features = {}

    # Initialize EMD
    emd = EMD()

    # Perform decomposition
    imfs = emd(eeg_signal)

    # Calculate IMF energies
    imf_energies = [np.sum(imf ** 2) for imf in imfs]
    total_energy = sum(imf_energies)

    # Extract features for each IMF
    for i, imf in enumerate(imfs):
        # Calculate basic statistics
        features[f'hht_imf{i + 1}_mean'] = np.mean(imf)
        features[f'hht_imf{i + 1}_std'] = np.std(imf)
        features[f'hht_imf{i + 1}_abs_mean'] = np.mean(np.abs(imf))
        features[f'hht_imf{i + 1}_energy'] = imf_energies[i]

        # Calculate relative energy
        if total_energy > 0:
            features[f'hht_imf{i + 1}_rel_energy'] = imf_energies[i] / total_energy
        else:
            features[f'hht_imf{i + 1}_rel_energy'] = 0

        # Calculate instantaneous frequency (using Hilbert transform)
        analytic_signal = signal.hilbert(imf)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

        if len(instantaneous_frequency) > 0:
            features[f'hht_imf{i + 1}_inst_freq_mean'] = np.mean(instantaneous_frequency)
            features[f'hht_imf{i + 1}_inst_freq_std'] = np.std(instantaneous_frequency)
        else:
            features[f'hht_imf{i + 1}_inst_freq_mean'] = 0
            features[f'hht_imf{i + 1}_inst_freq_std'] = 0

        # Limit to first 5 IMFs to keep feature count reasonable
        if i >= 4:
            break

    return features


def extract_all_time_frequency_features(eeg_signal, fs=250, wavelet='db4', levels=5):
    """
    Extract all time-frequency features for a single EEG channel.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    wavelet : str
        Wavelet type for DWT
    levels : int
        Number of decomposition levels for DWT

    Returns:
    --------
    dict
        Dictionary of all time-frequency features
    """
    # Get wavelet features
    wavelet_coef_features = extract_wavelet_coefficients(eeg_signal, wavelet, levels)
    wavelet_entropy_features = extract_wavelet_entropy(eeg_signal, wavelet, levels)

    # Get spectrogram features
    spectrogram_features = extract_spectrogram_features(eeg_signal, fs)

    # Try to get Morlet wavelet features
    try:
        morlet_features = extract_morlet_wavelet_features(eeg_signal, fs)
    except ImportError:
        print("Warning: MNE package is required for Morlet wavelet analysis")
        morlet_features = {}

    # Try to get Hilbert-Huang Transform features
    try:
        hht_features = extract_hilbert_huang_transform(eeg_signal, fs)
    except Exception as e:
        print(f"Warning: Could not extract HHT features: {e}")
        hht_features = {}

    # Combine all features
    all_features = {}
    all_features.update(wavelet_coef_features)
    all_features.update(wavelet_entropy_features)
    all_features.update(spectrogram_features)
    all_features.update(morlet_features)
    all_features.update(hht_features)

    return all_features


def extract_multichannel_time_frequency_features(eeg_signals, fs=250, wavelet='db4', levels=5):
    """
    Extract time-frequency features for multiple EEG channels.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    wavelet : str
        Wavelet type for DWT
    levels : int
        Number of decomposition levels for DWT

    Returns:
    --------
    pandas.DataFrame
        DataFrame with all time-frequency features for all channels
    """
    all_channel_features = []

    for ch_idx, eeg_signal in enumerate(eeg_signals):
        # Extract features for this channel
        channel_features = extract_all_time_frequency_features(eeg_signal, fs, wavelet, levels)

        # Add channel index
        channel_features['channel'] = ch_idx

        all_channel_features.append(channel_features)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_channel_features)

    return features_df