import numpy as np
import pandas as pd


def extract_periodicity_features(eeg_signal, fs=250):
    """
    Extract features specific to detecting periodic discharges (LPDs, GPDs).

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of periodicity features
    """
    features = {}

    try:
        # Find peaks in the signal
        peaks, _ = signal.find_peaks(eeg_signal, height=np.mean(eeg_signal) + 0.5 * np.std(eeg_signal))

        if len(peaks) >= 2:
            # Calculate inter-peak intervals
            intervals = np.diff(peaks) / fs  # Convert to seconds

            # Basic statistics of intervals
            features['periodicity_mean_interval'] = np.mean(intervals)
            features['periodicity_std_interval'] = np.std(intervals)
            features['periodicity_cv_interval'] = np.std(intervals) / np.mean(intervals) if np.mean(
                intervals) > 0 else 0

            # Calculate periodicity index based on interval regularity
            # Lower coefficient of variation (CV) means more regularity
            features['periodicity_index'] = 1 / (1 + features['periodicity_cv_interval'])

            # Estimate frequency of periodic discharges
            if np.mean(intervals) > 0:
                features['discharge_frequency'] = 1 / np.mean(intervals)  # Hz
            else:
                features['discharge_frequency'] = 0

            # Check if discharge frequency is in the typical range for GPDs/LPDs (0.5-3 Hz)
            is_gpd_lpd_freq = 0.5 <= features['discharge_frequency'] <= 3
            features['is_periodic_discharge_freq'] = 1 if is_gpd_lpd_freq else 0

            # Calculate peak heights
            peak_heights = eeg_signal[peaks]
            features['peak_height_mean'] = np.mean(peak_heights)
            features['peak_height_std'] = np.std(peak_heights)
            features['peak_height_max'] = np.max(peak_heights)
            features['peak_height_cv'] = np.std(peak_heights) / np.mean(peak_heights) if np.mean(
                peak_heights) > 0 else 0

            # Calculate peak evolution (trend in peak heights)
            if len(peak_heights) >= 3:
                # Linear regression on peak heights
                x = np.arange(len(peak_heights))
                slope, _ = np.polyfit(x, peak_heights, 1)
                features['peak_height_trend'] = slope
            else:
                features['peak_height_trend'] = 0
        else:
            # Not enough peaks found
            features['periodicity_mean_interval'] = 0
            features['periodicity_std_interval'] = 0
            features['periodicity_cv_interval'] = 0
            features['periodicity_index'] = 0
            features['discharge_frequency'] = 0
            features['is_periodic_discharge_freq'] = 0
            features['peak_height_mean'] = 0
            features['peak_height_std'] = 0
            features['peak_height_max'] = 0
            features['peak_height_cv'] = 0
            features['peak_height_trend'] = 0

        # Calculate power spectral density for periodicity detection
        f, Pxx = signal.welch(eeg_signal, fs=fs, nperseg=min(256, len(eeg_signal)))

        # Spectral peak detection for periodic discharge frequency
        # Look for peaks in 0.5-3 Hz range (typical for LPDs/GPDs)
        mask = (f >= 0.5) & (f <= 3)
        if np.sum(mask) > 0:
            discharge_freqs = f[mask]
            discharge_powers = Pxx[mask]

            if len(discharge_powers) > 0:
                # Find spectral peak in the discharge frequency range
                max_idx = np.argmax(discharge_powers)
                features['spectral_peak_freq'] = discharge_freqs[max_idx]
                features['spectral_peak_power'] = discharge_powers[max_idx]

                # Calculate spectral peak prominence
                if np.mean(Pxx) > 0:
                    features['spectral_peak_prominence'] = discharge_powers[max_idx] / np.mean(Pxx)
                else:
                    features['spectral_peak_prominence'] = 0
            else:
                features['spectral_peak_freq'] = 0
                features['spectral_peak_power'] = 0
                features['spectral_peak_prominence'] = 0
        else:
            features['spectral_peak_freq'] = 0
            features['spectral_peak_power'] = 0
            features['spectral_peak_prominence'] = 0

    except Exception as e:
        print(f"Error extracting periodicity features: {e}")
        # Initialize with zeros in case of error
        features = {
            'periodicity_mean_interval': 0,
            'periodicity_std_interval': 0,
            'periodicity_cv_interval': 0,
            'periodicity_index': 0,
            'discharge_frequency': 0,
            'is_periodic_discharge_freq': 0,
            'peak_height_mean': 0,
            'peak_height_std': 0,
            'peak_height_max': 0,
            'peak_height_cv': 0,
            'peak_height_trend': 0,
            'spectral_peak_freq': 0,
            'spectral_peak_power': 0,
            'spectral_peak_prominence': 0
        }

    return features


def extract_discharge_sharpness(eeg_signal, fs=250):
    """
    Extract features related to the sharpness of discharges (helps distinguish epileptiform discharges).

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of discharge sharpness features
    """
    features = {}

    try:
        # Calculate first derivative (slope)
        derivative = np.diff(eeg_signal) * fs  # Scale by sampling frequency

        # Find peaks in the derivative (points of maximum slope)
        pos_peaks, _ = signal.find_peaks(derivative, height=np.mean(derivative) + 2 * np.std(derivative))
        neg_peaks, _ = signal.find_peaks(-derivative, height=np.mean(-derivative) + 2 * np.std(derivative))

        # Calculate statistics of peak slopes (higher slopes = sharper waveforms)
        if len(pos_peaks) > 0:
            features['max_positive_slope'] = np.max(derivative[pos_peaks])
            features['mean_positive_slope'] = np.mean(derivative[pos_peaks])
        else:
            features['max_positive_slope'] = 0
            features['mean_positive_slope'] = 0

        if len(neg_peaks) > 0:
            features['max_negative_slope'] = np.max(-derivative[neg_peaks])
            features['mean_negative_slope'] = np.mean(-derivative[neg_peaks])
        else:
            features['max_negative_slope'] = 0
            features['mean_negative_slope'] = 0

        # Calculate asymmetry between positive and negative slopes
        # (epileptiform discharges often have asymmetric rising and falling phases)
        if len(pos_peaks) > 0 and len(neg_peaks) > 0:
            features['slope_asymmetry'] = features['mean_positive_slope'] / features['mean_negative_slope'] if features[
                                                                                                                   'mean_negative_slope'] > 0 else 0
        else:
            features['slope_asymmetry'] = 0

        # Calculate half-wave durations
        zero_crossings = np.where(np.diff(np.signbit(eeg_signal)))[0]

        if len(zero_crossings) >= 2:
            # Calculate intervals between zero crossings
            half_wave_durations = np.diff(zero_crossings) / fs  # in seconds

            features['half_wave_mean_duration'] = np.mean(half_wave_durations)
            features['half_wave_min_duration'] = np.min(half_wave_durations)

            # Shorter half-wave durations indicate sharper waveforms
            features['sharpness_index'] = 1 / features['half_wave_mean_duration'] if features[
                                                                                         'half_wave_mean_duration'] > 0 else 0

            # Calculate variability of half-wave durations
            features['half_wave_cv'] = np.std(half_wave_durations) / np.mean(half_wave_durations) if np.mean(
                half_wave_durations) > 0 else 0
        else:
            features['half_wave_mean_duration'] = 0
            features['half_wave_min_duration'] = 0
            features['sharpness_index'] = 0
            features['half_wave_cv'] = 0

        # Calculate second derivative features (curvature)
        second_derivative = np.diff(derivative)
        features['max_curvature'] = np.max(np.abs(second_derivative))
        features['mean_curvature'] = np.mean(np.abs(second_derivative))

    except Exception as e:
        print(f"Error extracting discharge sharpness features: {e}")
        # Initialize with zeros in case of error
        features = {
            'max_positive_slope': 0,
            'mean_positive_slope': 0,
            'max_negative_slope': 0,
            'mean_negative_slope': 0,
            'slope_asymmetry': 0,
            'half_wave_mean_duration': 0,
            'half_wave_min_duration': 0,
            'sharpness_index': 0,
            'half_wave_cv': 0,
            'max_curvature': 0,
            'mean_curvature': 0
        }

    return features


def extract_rhythmicity_features(eeg_signal, fs=250):
    """
    Extract features to detect rhythmic delta activity (LRDA, GRDA).

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of rhythmicity features
    """
    features = {}

    try:
        # Apply a delta band filter (0.5-4 Hz)
        b, a = signal.butter(4, [0.5, 4], btype='bandpass', fs=fs)
        delta_signal = signal.filtfilt(b, a, eeg_signal)

        # Calculate envelope of the delta activity
        analytic_signal = signal.hilbert(delta_signal)
        envelope = np.abs(analytic_signal)

        # Calculate basic statistics of the envelope
        features['delta_envelope_mean'] = np.mean(envelope)
        features['delta_envelope_std'] = np.std(envelope)
        features['delta_envelope_cv'] = features['delta_envelope_std'] / features['delta_envelope_mean'] if features[
                                                                                                                'delta_envelope_mean'] > 0 else 0

        # Find peaks in the envelope to assess rhythmicity
        peaks, _ = signal.find_peaks(envelope)

        if len(peaks) >= 2:
            # Calculate intervals between peaks (should be regular for rhythmic activity)
            intervals = np.diff(peaks) / fs  # in seconds

            features['rhythm_mean_interval'] = np.mean(intervals)
            features['rhythm_std_interval'] = np.std(intervals)
            features['rhythm_cv_interval'] = features['rhythm_std_interval'] / features['rhythm_mean_interval'] if \
                features['rhythm_mean_interval'] > 0 else 0

            # Calculate rhythmicity score (1 = perfect rhythm, 0 = no rhythm)
            # Lower CV = more rhythmic
            features['rhythmicity_score'] = 1 / (1 + 5 * features['rhythm_cv_interval'])

            # Estimate dominant frequency of the rhythm
            if features['rhythm_mean_interval'] > 0:
                features['rhythm_frequency'] = 1 / features['rhythm_mean_interval']  # Hz
            else:
                features['rhythm_frequency'] = 0

            # Check if frequency is in the delta range (0.5-4 Hz, typical for LRDA/GRDA)
            is_delta_freq = 0.5 <= features['rhythm_frequency'] <= 4
            features['is_delta_rhythm_freq'] = 1 if is_delta_freq else 0
        else:
            # Not enough peaks found in the envelope
            features['rhythm_mean_interval'] = 0
            features['rhythm_std_interval'] = 0
            features['rhythm_cv_interval'] = 0
            features['rhythmicity_score'] = 0
            features['rhythm_frequency'] = 0
            features['is_delta_rhythm_freq'] = 0

        # Perform spectral analysis focused on the delta band
        f, Pxx = signal.welch(eeg_signal, fs=fs, nperseg=min(512, len(eeg_signal)))

        # Calculate relative delta power
        delta_mask = (f >= 0.5) & (f <= 4)
        if np.sum(Pxx) > 0 and np.sum(delta_mask) > 0:
            features['relative_delta_power'] = np.sum(Pxx[delta_mask]) / np.sum(Pxx)
        else:
            features['relative_delta_power'] = 0

        # Analyze delta sub-bands (slow vs fast delta)
        slow_delta_mask = (f >= 0.5) & (f < 2)
        fast_delta_mask = (f >= 2) & (f <= 4)

        if np.sum(delta_mask) > 0 and np.sum(slow_delta_mask) > 0 and np.sum(fast_delta_mask) > 0:
            slow_delta_power = np.sum(Pxx[slow_delta_mask])
            fast_delta_power = np.sum(Pxx[fast_delta_mask])
            total_delta_power = np.sum(Pxx[delta_mask])

            if total_delta_power > 0:
                features['slow_delta_ratio'] = slow_delta_power / total_delta_power
                features['fast_delta_ratio'] = fast_delta_power / total_delta_power
            else:
                features['slow_delta_ratio'] = 0
                features['fast_delta_ratio'] = 0
        else:
            features['slow_delta_ratio'] = 0
            features['fast_delta_ratio'] = 0

        # Find dominant frequency in delta band
        if np.sum(delta_mask) > 0:
            delta_freqs = f[delta_mask]
            delta_powers = Pxx[delta_mask]

            if len(delta_powers) > 0:
                max_idx = np.argmax(delta_powers)
                features['delta_peak_frequency'] = delta_freqs[max_idx]
                features['delta_peak_power'] = delta_powers[max_idx]
            else:
                features['delta_peak_frequency'] = 0
                features['delta_peak_power'] = 0
        else:
            features['delta_peak_frequency'] = 0
            features['delta_peak_power'] = 0

    except Exception as e:
        print(f"Error extracting rhythmicity features: {e}")
        # Initialize with zeros in case of error
        features = {
            'delta_envelope_mean': 0,
            'delta_envelope_std': 0,
            'delta_envelope_cv': 0,
            'rhythm_mean_interval': 0,
            'rhythm_std_interval': 0,
            'rhythm_cv_interval': 0,
            'rhythmicity_score': 0,
            'rhythm_frequency': 0,
            'is_delta_rhythm_freq': 0,
            'relative_delta_power': 0,
            'slow_delta_ratio': 0,
            'fast_delta_ratio': 0,
            'delta_peak_frequency': 0,
            'delta_peak_power': 0
        }

    return features


def extract_evolution_features(eeg_signal, fs=250):
    """
    Extract features related to the evolution of EEG patterns over time (important for seizure detection).

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of evolution features
    """
    features = {}

    try:
        # Split the signal into segments to analyze evolution
        n_segments = 5  # Divide the signal into 5 segments
        segment_len = len(eeg_signal) // n_segments

        if segment_len < 10:  # If segments are too short
            n_segments = 2
            segment_len = len(eeg_signal) // n_segments

        # Extract segment features
        segment_features = []

        for i in range(n_segments):
            start_idx = i * segment_len
            end_idx = start_idx + segment_len if i < n_segments - 1 else len(eeg_signal)

            segment = eeg_signal[start_idx:end_idx]

            # Calculate basic stats for each segment
            segment_mean = np.mean(segment)
            segment_std = np.std(segment)
            segment_power = np.sum(segment ** 2)

            # Calculate frequency content for each segment
            f, Pxx = signal.welch(segment, fs=fs, nperseg=min(256, len(segment)))

            # Extract band powers
            delta_mask = (f >= 0.5) & (f <= 4)
            theta_mask = (f >= 4) & (f <= 8)
            alpha_mask = (f >= 8) & (f <= 13)
            beta_mask = (f >= 13) & (f <= 30)

            # Calculate band powers if masks are not empty
            delta_power = np.sum(Pxx[delta_mask]) if np.sum(delta_mask) > 0 else 0
            theta_power = np.sum(Pxx[theta_mask]) if np.sum(theta_mask) > 0 else 0
            alpha_power = np.sum(Pxx[alpha_mask]) if np.sum(alpha_mask) > 0 else 0
            beta_power = np.sum(Pxx[beta_mask]) if np.sum(beta_mask) > 0 else 0

            total_power = np.sum(Pxx)

            # Calculate relative band powers
            rel_delta = delta_power / total_power if total_power > 0 else 0
            rel_theta = theta_power / total_power if total_power > 0 else 0
            rel_alpha = alpha_power / total_power if total_power > 0 else 0
            rel_beta = beta_power / total_power if total_power > 0 else 0

            # Store segment features
            segment_features.append({
                'mean': segment_mean,
                'std': segment_std,
                'power': segment_power,
                'rel_delta': rel_delta,
                'rel_theta': rel_theta,
                'rel_alpha': rel_alpha,
                'rel_beta': rel_beta
            })

        # Calculate evolution metrics

        # Amplitude evolution
        means = [f['mean'] for f in segment_features]
        stds = [f['std'] for f in segment_features]
        powers = [f['power'] for f in segment_features]

        # Apply linear regression to detect trends
        x = np.arange(len(segment_features))

        # Mean trend
        mean_slope, _ = np.polyfit(x, means, 1)
        features['mean_evolution'] = mean_slope

        # Standard deviation trend
        std_slope, _ = np.polyfit(x, stds, 1)
        features['std_evolution'] = std_slope

        # Power trend
        power_slope, _ = np.polyfit(x, powers, 1)
        features['power_evolution'] = power_slope

        # Calculate max-to-min ratios
        if min(means) != 0:
            features['mean_max_min_ratio'] = max(means) / min(means)
        else:
            features['mean_max_min_ratio'] = 1

        if min(powers) != 0:
            features['power_max_min_ratio'] = max(powers) / min(powers)
        else:
            features['power_max_min_ratio'] = 1

        # Frequency evolution
        rel_deltas = [f['rel_delta'] for f in segment_features]
        rel_thetas = [f['rel_theta'] for f in segment_features]
        rel_alphas = [f['rel_alpha'] for f in segment_features]
        rel_betas = [f['rel_beta'] for f in segment_features]

        # Calculate trends in frequency bands
        delta_slope, _ = np.polyfit(x, rel_deltas, 1)
        theta_slope, _ = np.polyfit(x, rel_thetas, 1)
        alpha_slope, _ = np.polyfit(x, rel_alphas, 1)
        beta_slope, _ = np.polyfit(x, rel_betas, 1)

        features['delta_evolution'] = delta_slope
        features['theta_evolution'] = theta_slope
        features['alpha_evolution'] = alpha_slope
        features['beta_evolution'] = beta_slope

        # Calculate spectral edge frequency evolution (SEF 95%)
        sef_values = []

        for i in range(n_segments):
            start_idx = i * segment_len
            end_idx = start_idx + segment_len if i < n_segments - 1 else len(eeg_signal)

            segment = eeg_signal[start_idx:end_idx]

            # Calculate PSD
            f, Pxx = signal.welch(segment, fs=fs, nperseg=min(256, len(segment)))

            # Calculate cumulative PSD
            cum_psd = np.cumsum(Pxx)
            cum_psd_norm = cum_psd / cum_psd[-1] if cum_psd[-1] > 0 else cum_psd

            # Find SEF 95%
            idx_95 = np.argmax(cum_psd_norm >= 0.95)
            sef_values.append(f[idx_95])

        # Calculate SEF evolution
        sef_slope, _ = np.polyfit(x, sef_values, 1)
        features['sef_evolution'] = sef_slope

        # Calculate SEF max/min ratio
        if min(sef_values) > 0:
            features['sef_max_min_ratio'] = max(sef_values) / min(sef_values)
        else:
            features['sef_max_min_ratio'] = 1

        # Overall evolution score for seizure detection
        # Combine multiple evolution metrics
        # Higher score = more evolution (typical of seizures)
        features['overall_evolution_score'] = (
                abs(features['mean_evolution']) +
                abs(features['power_evolution']) +
                abs(features['delta_evolution']) +
                abs(features['beta_evolution']) +
                abs(features['sef_evolution'])
        )

    except Exception as e:
        print(f"Error extracting evolution features: {e}")
        # Initialize with zeros in case of error
        features = {
            'mean_evolution': 0,
            'std_evolution': 0,
            'power_evolution': 0,
            'mean_max_min_ratio': 1,
            'power_max_min_ratio': 1,
            'delta_evolution': 0,
            'theta_evolution': 0,
            'alpha_evolution': 0,
            'beta_evolution': 0,
            'sef_evolution': 0,
            'sef_max_min_ratio': 1,
            'overall_evolution_score': 0
        }

    return features


def extract_spatial_distribution_features(eeg_signals):
    """
    Extract features related to the spatial distribution of EEG activity
    (helps distinguish between generalized and lateralized patterns).

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)

    Returns:
    --------
    dict
        Dictionary of spatial distribution features
    """
    features = {}

    try:
        # Check if we have enough channels
        if len(eeg_signals) < 2:
            print("Not enough channels for spatial analysis")
            return {'lateralization_index': 0, 'generalization_index': 0}

        # Calculate RMS amplitude for each channel
        channel_rms = np.sqrt(np.mean(eeg_signals ** 2, axis=1))

        # For proper lateralization analysis, we need to know channel positions
        # Here we make a simplification - we assume first half of channels are left and second half are right
        # In a real implementation, this would be replaced with actual channel positions
        n_channels = len(eeg_signals)
        n_left = n_channels // 2

        left_channels = channel_rms[:n_left]
        right_channels = channel_rms[n_left:]

        # If left and right have different numbers of channels, use the smaller count
        min_side_count = min(len(left_channels), len(right_channels))

        if min_side_count > 0:
            # Recalculate using the same number of channels on each side
            left_channels = left_channels[:min_side_count]
            right_channels = right_channels[:min_side_count]

            # Calculate mean amplitude on each side
            left_mean = np.mean(left_channels)
            right_mean = np.mean(right_channels)

            # Lateralization index: 0 = symmetric, 1 = completely lateralized
            total_amp = left_mean + right_mean
            if total_amp > 0:
                features['lateralization_index'] = abs(left_mean - right_mean) / total_amp
            else:
                features['lateralization_index'] = 0

            # Determine lateralization direction (left or right)
            if left_mean > right_mean:
                features['lateralization_direction'] = 'left'
            else:
                features['lateralization_direction'] = 'right'

            # Calculate channel-wise coefficient of variation as a measure of focal vs. generalized activity
            # Lower CV = more uniform/generalized activity
            cv = np.std(channel_rms) / np.mean(channel_rms) if np.mean(channel_rms) > 0 else 0

            # Generalization index: 1 = fully generalized, 0 = focal
            features['generalization_index'] = 1 / (1 + 2 * cv)
        else:
            features['lateralization_index'] = 0
            features['lateralization_direction'] = 'none'
            features['generalization_index'] = 0

        # Determine pattern type based on spatial indices
        # This is a simplification and would be more sophisticated in practice
        if features['lateralization_index'] > 0.3:
            if features['generalization_index'] < 0.5:
                features['spatial_pattern_type'] = 'lateralized_focal'  # Could be LPD
            else:
                features['spatial_pattern_type'] = 'lateralized_diffuse'  # Could be LRDA
        else:
            if features['generalization_index'] > 0.7:
                features['spatial_pattern_type'] = 'generalized'  # Could be GPD or GRDA
            else:
                features['spatial_pattern_type'] = 'mixed'

    except Exception as e:
        print(f"Error extracting spatial distribution features: {e}")
        # Initialize with zeros in case of error
        features = {
            'lateralization_index': 0,
            'lateralization_direction': 'none',
            'generalization_index': 0,
            'spatial_pattern_type': 'unknown'
        }

    return features


def extract_seizure_specific_features(eeg_signal, fs=250):
    """
    Extract features specifically aimed at seizure detection.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of seizure-specific features
    """
    features = {}

    try:
        # Calculate evolution features (seizures typically show evolution)
        evolution_features = extract_evolution_features(eeg_signal, fs)

        # Extract key evolution metrics that are relevant for seizures
        features['seizure_evolution_score'] = evolution_features['overall_evolution_score']
        features['frequency_increase'] = evolution_features['beta_evolution'] > 0
        features['amplitude_increase'] = evolution_features['power_evolution'] > 0

        # Calculate line length (coastline) - effective for seizure detection
        line_length = np.sum(np.abs(np.diff(eeg_signal)))
        features['line_length'] = line_length

        # Analyze high-frequency content (ictal events often have increased HF activity)
        f, Pxx = signal.welch(eeg_signal, fs=fs, nperseg=min(512, len(eeg_signal)))

        # Calculate high-frequency power ratio (beta+gamma power / total power)
        high_freq_mask = f >= 13  # Beta and above
        total_power = np.sum(Pxx)

        if total_power > 0 and np.sum(high_freq_mask) > 0:
            high_freq_power = np.sum(Pxx[high_freq_mask])
            features['high_freq_ratio'] = high_freq_power / total_power
        else:
            features['high_freq_ratio'] = 0

        # Calculate post-ictal suppression index
        # Divide the signal into 4 quarters and check if the last quarter has lower power
        quarter_len = len(eeg_signal) // 4
        if quarter_len > 0:
            first_half_power = np.mean(eeg_signal[:2 * quarter_len] ** 2)
            last_quarter_power = np.mean(eeg_signal[3 * quarter_len:] ** 2)

            if first_half_power > 0:
                features['post_suppression_index'] = 1 - (last_quarter_power / first_half_power)
            else:
                features['post_suppression_index'] = 0
        else:
            features['post_suppression_index'] = 0

        # Calculate rhythmicity features (seizures often have rhythmic qualities)
        rhythm_features = extract_rhythmicity_features(eeg_signal, fs)

        # Extract key rhythmicity metrics relevant for seizures
        features['rhythmicity_score'] = rhythm_features['rhythmicity_score']
        features['rhythm_frequency'] = rhythm_features['rhythm_frequency']

        # Spectral entropy (lower during seizures due to more organized activity)
        # Calculate power spectral density
        f, Pxx = signal.welch(eeg_signal, fs=fs, nperseg=min(512, len(eeg_signal)))

        # Calculate normalized PSD (probability distribution)
        if np.sum(Pxx) > 0:
            psd_norm = Pxx / np.sum(Pxx)
            # Calculate spectral entropy
            features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        else:
            features['spectral_entropy'] = 0

        # Analyze phase synchrony using Hilbert transform
        # (seizures often show increased phase synchrony)
        analytic_signal = signal.hilbert(eeg_signal)
        instantaneous_phase = np.angle(analytic_signal)

        # Calculate phase difference variability (lower = more synchronized)
        phase_diff = np.diff(instantaneous_phase)
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi  # Wrap to [-pi, pi]
        features['phase_sync_index'] = 1 / (1 + np.std(phase_diff))

        # Calculate power in specific frequency bands associated with seizures
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, min(100, fs / 2))
        }

        # Specific band power ratios relevant for seizure detection
        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            mask = (f >= fmin) & (f <= fmax)
            if np.sum(mask) > 0:
                band_powers[band_name] = np.sum(Pxx[mask])
            else:
                band_powers[band_name] = 0

        # Beta/Alpha ratio (often increases during seizures)
        if band_powers['alpha'] > 0:
            features['beta_alpha_ratio'] = band_powers['beta'] / band_powers['alpha']
        else:
            features['beta_alpha_ratio'] = 0

        # Gamma/Theta ratio (often increases during seizures)
        if band_powers['theta'] > 0:
            features['gamma_theta_ratio'] = band_powers['gamma'] / band_powers['theta']
        else:
            features['gamma_theta_ratio'] = 0

        # Combine features to create seizure probability score
        # This is a simplified approach; real systems use more sophisticated models
        features['seizure_probability_score'] = (
                features['seizure_evolution_score'] * 0.3 +
                features['line_length'] / 1000 * 0.1 +  # Normalized line length
                features['high_freq_ratio'] * 0.2 +
                features['rhythmicity_score'] * 0.2 +
                features['phase_sync_index'] * 0.1 +
                (1 - features['spectral_entropy'] / 5) * 0.1  # Normalized entropy
        )

    except Exception as e:
        print(f"Error extracting seizure-specific features: {e}")
        # Initialize with zeros in case of error
        features = {
            'seizure_evolution_score': 0,
            'frequency_increase': False,
            'amplitude_increase': False,
            'line_length': 0,
            'high_freq_ratio': 0,
            'post_suppression_index': 0,
            'rhythmicity_score': 0,
            'rhythm_frequency': 0,
            'spectral_entropy': 0,
            'phase_sync_index': 0,
            'beta_alpha_ratio': 0,
            'gamma_theta_ratio': 0,
            'seizure_probability_score': 0
        }

    return features


def extract_all_pattern_specific_features(eeg_signal, fs=250):
    """
    Extract all pattern-specific features for EEG harmful brain activity detection.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    dict
        Dictionary of all pattern-specific features
    """
    # Extract periodicity features (for GPDs and LPDs)
    periodicity_features = extract_periodicity_features(eeg_signal, fs)

    # Extract discharge sharpness features (helps distinguish epileptiform discharges)
    sharpness_features = extract_discharge_sharpness(eeg_signal, fs)

    # Extract rhythmicity features (for GRDA and LRDA)
    rhythmicity_features = extract_rhythmicity_features(eeg_signal, fs)

    # Extract evolution features (helps identify pattern evolution)
    evolution_features = extract_evolution_features(eeg_signal, fs)

    # Extract seizure-specific features
    seizure_features = extract_seizure_specific_features(eeg_signal, fs)

    # Combine all features
    all_features = {}
    all_features.update(periodicity_features)
    all_features.update(sharpness_features)
    all_features.update(rhythmicity_features)
    all_features.update(evolution_features)
    all_features.update(seizure_features)

    # Calculate pattern-specific scores to help classification
    # GPD score (generalized periodic discharges)
    all_features['gpd_score'] = (
            periodicity_features['periodicity_index'] * 0.4 +
            (1 - periodicity_features['periodicity_cv_interval']) * 0.2 +
            sharpness_features['sharpness_index'] / 100 * 0.2 +  # Normalized
            (1 - evolution_features['overall_evolution_score']) * 0.2  # GPDs typically don't evolve much
    )

    # LPD score (lateralized periodic discharges) - Note: this doesn't include lateralization info
    all_features['lpd_score'] = (
            periodicity_features['periodicity_index'] * 0.4 +
            (1 - periodicity_features['periodicity_cv_interval']) * 0.2 +
            sharpness_features['sharpness_index'] / 100 * 0.2 +  # Normalized
            evolution_features['overall_evolution_score'] * 0.2  # LPDs may evolve more than GPDs
    )

    # GRDA score (generalized rhythmic delta activity)
    all_features['grda_score'] = (
            rhythmicity_features['rhythmicity_score'] * 0.5 +
            rhythmicity_features['relative_delta_power'] * 0.3 +
            (1 - sharpness_features['sharpness_index'] / 100) * 0.2  # Less sharp than periodic discharges
    )

    # LRDA score (lateralized rhythmic delta activity) - Note: doesn't include lateralization info
    all_features['lrda_score'] = (
            rhythmicity_features['rhythmicity_score'] * 0.5 +
            rhythmicity_features['relative_delta_power'] * 0.3 +
            (1 - sharpness_features['sharpness_index'] / 100) * 0.2  # Less sharp than periodic discharges
    )

    # Seizure score
    all_features['seizure_score'] = seizure_features['seizure_probability_score']

    return all_features


def extract_multichannel_pattern_specific_features(eeg_signals, fs=250):
    """
    Extract pattern-specific features for multiple EEG channels, including spatial features.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    pandas.DataFrame
        DataFrame with pattern-specific features for all channels
    """
    all_channel_features = []

    # Extract individual channel features
    for ch_idx, eeg_signal in enumerate(eeg_signals):
        # Extract features for this channel
        channel_features = extract_all_pattern_specific_features(eeg_signal, fs)

        # Add channel index
        channel_features['channel'] = ch_idx

        all_channel_features.append(channel_features)

    # Extract spatial distribution features
    spatial_features = extract_spatial_distribution_features(eeg_signals)

    # Create DataFrame
    features_df = pd.DataFrame(all_channel_features)

    # Add spatial features to all rows
    for key, value in spatial_features.items():
        features_df[key] = value

    # Adjust pattern scores based on spatial features
    if 'lateralization_index' in features_df.columns:
        # Adjust LPD score - higher if lateralized
        features_df['lpd_score'] = features_df['lpd_score'] * (0.5 + 0.5 * features_df['lateralization_index'])

        # Adjust LRDA score - higher if lateralized
        features_df['lrda_score'] = features_df['lrda_score'] * (0.5 + 0.5 * features_df['lateralization_index'])

        # Adjust GPD score - higher if generalized (low lateralization)
        features_df['gpd_score'] = features_df['gpd_score'] * (1.5 - features_df['lateralization_index'])

        # Adjust GRDA score - higher if generalized (low lateralization)
        features_df['grda_score'] = features_df['grda_score'] * (1.5 - features_df['lateralization_index'])

    return features_df
