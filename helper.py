import numpy as np


class EEGSimulator:
    """Class for generating synthetic EEG data for testing"""

    def __init__(self):
        # Define the standard channels
        self.all_channels = [
            'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
            'Fz', 'Cz', 'Pz',
            'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
            'EKG'
        ]

        # Define channel location groups for easier reference
        self.posterior_channels = ['O1', 'O2', 'P3', 'P4', 'Pz', 'T5', 'T6']
        self.frontal_channels = ['F3', 'F4', 'Fz', 'F7', 'F8', 'Fp1', 'Fp2']
        self.central_channels = ['C3', 'C4', 'Cz', 'T3', 'T4']
        self.blink_channels = ['Fp1', 'Fp2', 'F7', 'F8']

    def generate_sample(self, duration=5, fs=250, num_channels=None):
        """
        Generate a sample EEG dataset

        Parameters:
            duration (float): Duration of the signal in seconds
            fs (int): Sampling frequency in Hz
            num_channels (int, optional): Number of channels to generate

        Returns:
            tuple: (numpy.ndarray of shape (n_samples, num_channels), list of channel names)
        """
        # Select channels
        if num_channels is not None and num_channels < len(self.all_channels):
            channels = self.all_channels[:num_channels]
        else:
            channels = self.all_channels.copy()

        num_channels = len(channels)

        # Create time array
        t = np.arange(0, duration, 1 / fs)
        n_samples = len(t)

        # Initialize EEG array (n_samples, num_channels)
        eeg = np.zeros((n_samples, num_channels))

        # Generate base brain wave components
        delta = np.sin(2 * np.pi * 2 * t)  # 0.5-4 Hz
        theta = np.sin(2 * np.pi * 6 * t)  # 4-8 Hz
        alpha = np.sin(2 * np.pi * 10 * t)  # 8-13 Hz
        beta = np.sin(2 * np.pi * 20 * t)  # 13-30 Hz

        # Generate EKG signal
        ekg = self._generate_ekg(t, fs)

        # Pre-generate random noise for all channels
        noise = np.random.normal(0, 1, (n_samples, num_channels))

        # Process each channel
        for ch_idx, ch_name in enumerate(channels):
            if ch_name == 'EKG':
                eeg[:, ch_idx] = ekg
                continue

            # Generate signal for this channel
            eeg[:, ch_idx] = self._generate_channel_signal(
                ch_name, t, delta, theta, alpha, beta, noise[:, ch_idx]
            )

            # Add eye blink artifacts to frontal channels
            if ch_name in self.blink_channels:
                eeg[:, ch_idx] = self._add_eye_blinks(eeg[:, ch_idx], n_samples, fs)

        return eeg, channels

    def _generate_ekg(self, t, fs):
        """Generate synthetic EKG signal (~70 BPM)"""
        n_samples = len(t)
        ekg_freq = 70 / 60  # Heartbeats per second
        ekg_base = np.zeros(n_samples)

        # Add QRS complexes
        num_beats = int(len(t) / fs * ekg_freq)
        beat_positions = [int((i / ekg_freq) * fs) for i in range(num_beats)]

        for peak_pos in beat_positions:
            if peak_pos < n_samples - 10:
                # Create a simplified QRS complex
                ekg_base[peak_pos:peak_pos + 3] = [5, 15, 5]  # R spike
                ekg_base[peak_pos + 3:peak_pos + 7] = -2  # S wave
                ekg_base[peak_pos - 3:peak_pos] = 2  # Q wave

        # Add noise
        return ekg_base + np.random.normal(0, 1, n_samples)

    def _generate_channel_signal(self, ch_name, t, delta, theta, alpha, beta, noise):
        """Generate signal for a specific EEG channel"""
        # Default scaling factors
        alpha_scale = 0.5
        beta_scale = 0.5
        theta_scale = 0.5
        delta_scale = 0.5

        # Adjust scaling based on channel location
        if ch_name in self.posterior_channels:
            # Posterior regions have stronger alpha
            alpha_scale = 3.0 + np.random.rand()
            beta_scale = 1.0 + 0.5 * np.random.rand()
        elif ch_name in self.frontal_channels:
            # Frontal regions have stronger beta and theta
            beta_scale = 2.0 + np.random.rand()
            theta_scale = 2.5 + 0.5 * np.random.rand()
        elif ch_name in self.central_channels:
            # Central regions have mixed activity
            alpha_scale = 2.0 + 0.5 * np.random.rand()
            beta_scale = 2.0 + 0.5 * np.random.rand()
            theta_scale = 1.5 + 0.5 * np.random.rand()

        # Combine components
        signal = (
                alpha_scale * alpha +
                beta_scale * beta +
                theta_scale * theta +
                delta_scale * delta
        )

        return signal + noise

    def _add_eye_blinks(self, signal, n_samples, fs):
        """Add eye blink artifacts to a signal"""
        # Add 2-3 random eye blinks
        num_blinks = np.random.randint(2, 4)

        for _ in range(num_blinks):
            blink_start = np.random.randint(0, n_samples - fs // 2)
            blink_length = fs // 4  # 250 ms blink

            # Create blink shape (roughly gaussian)
            blink = 20 * np.exp(-0.5 * ((np.arange(blink_length) - blink_length / 2) / (blink_length / 6)) ** 2)

            if blink_start + blink_length <= n_samples:
                signal[blink_start:blink_start + blink_length] += blink

        return signal


# Example usage
if __name__ == "__main__":
    simulator = EEGSimulator()
    eeg_data, channels = simulator.generate_sample(duration=2, fs=250)
    print(f"Generated EEG data with shape {eeg_data.shape}")
    print(f"Channels: {channels}")