import numpy as np


def generate_eeg_sample(duration=5, fs=250):
    """
    Generate a sample EEG dataset with standard 10-20 system channels plus EKG.

    Parameters:
    -----------
    duration : float
        Duration of the signal in seconds
    fs : int
        Sampling frequency in Hz

    Returns:
    --------
    numpy.ndarray
        EEG signal with shape (num_channels, n_samples)
    dict
        Channel names corresponding to each row in the array
    """
    # Define the channels
    channels = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
                'Fz', 'Cz', 'Pz',
                'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
                'EKG']

    num_channels = len(channels)

    # Create time array
    t = np.arange(0, duration, 1 / fs)
    n_samples = len(t)

    # Initialize EEG array
    eeg = np.zeros((num_channels, n_samples))

    # Define brain wave frequencies
    frequencies = {
        'delta': 2,  # 0.5-4 Hz
        'theta': 6,  # 4-8 Hz
        'alpha': 10,  # 8-13 Hz
        'beta': 20  # 13-30 Hz
    }

    # Generate base signal components
    components = {}
    for wave, freq in frequencies.items():
        components[wave] = np.sin(2 * np.pi * freq * t)

    # Create EKG signal (simulated heartbeat at ~70 BPM)
    ekg_freq = 70 / 60  # Heartbeats per second
    ekg_base = np.zeros(n_samples)

    # Add QRS complexes to the EKG signal
    for i in range(int(duration * ekg_freq)):
        peak_pos = int((i / ekg_freq) * fs)
        if peak_pos < n_samples - 10:
            # Create a simplified QRS complex
            ekg_base[peak_pos:peak_pos + 3] = [5, 15, 5]  # R spike
            ekg_base[peak_pos + 3:peak_pos + 7] = -2  # S wave
            ekg_base[peak_pos - 3:peak_pos] = 2  # Q wave

    # Add noise to EKG
    ekg = ekg_base + 1.0 * np.random.normal(0, 1, n_samples)

    # Process each channel
    for ch_idx, ch_name in enumerate(channels):
        if ch_name == 'EKG':
            eeg[ch_idx] = ekg
            continue

        # Different brain wave mixtures based on channel location
        alpha_scale = 0.5
        beta_scale = 0.5
        theta_scale = 0.5
        delta_scale = 0.5

        # Adjust scaling based on channel type
        if ch_name in ['O1', 'O2', 'P3', 'P4', 'Pz', 'T5', 'T6']:
            # Posterior regions have stronger alpha
            alpha_scale = 3.0 + 1.0 * np.random.rand()
            beta_scale = 1.0 + 0.5 * np.random.rand()
        elif ch_name in ['F3', 'F4', 'Fz', 'F7', 'F8', 'Fp1', 'Fp2']:
            # Frontal regions have stronger beta and theta
            beta_scale = 2.0 + 1.0 * np.random.rand()
            theta_scale = 2.5 + 0.5 * np.random.rand()
        elif ch_name in ['C3', 'C4', 'Cz', 'T3', 'T4']:
            # Central regions have mixed activity
            alpha_scale = 2.0 + 0.5 * np.random.rand()
            beta_scale = 2.0 + 0.5 * np.random.rand()
            theta_scale = 1.5 + 0.5 * np.random.rand()

        # Combine components with scaling factors
        signal = (
                alpha_scale * components['alpha'] +
                beta_scale * components['beta'] +
                theta_scale * components['theta'] +
                delta_scale * components['delta']
        )

        # Add random noise
        noise = 1.0 * np.random.normal(0, 1, n_samples)

        # Store in EEG array
        eeg[ch_idx] = signal + noise

        # Add eye blink artifacts to frontal channels
        if ch_name in ['Fp1', 'Fp2', 'F7', 'F8']:
            # Add 2-3 eye blinks
            num_blinks = np.random.randint(2, 4)
            for _ in range(num_blinks):
                blink_start = np.random.randint(0, n_samples - fs // 2)
                blink_length = fs // 4  # 250 ms blink
                # Create blink shape (roughly gaussian)
                blink = 20 * np.exp(-0.5 * ((np.arange(blink_length) - blink_length / 2) / (blink_length / 6)) ** 2)
                if blink_start + blink_length <= n_samples:
                    eeg[ch_idx, blink_start:blink_start + blink_length] += blink

    return eeg, {i: ch for i, ch in enumerate(channels)}
