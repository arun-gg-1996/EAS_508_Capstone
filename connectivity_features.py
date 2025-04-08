def extract_coherence_features(eeg_signals, fs=250, bands=None):
    """
    Extract coherence features between EEG channels.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    bands : dict or None
        Dictionary of frequency bands in format {name: (min_freq, max_freq)}

    Returns:
    --------
    dict
        Dictionary of coherence features
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)  # Upper limit to avoid line noise
        }

    features = {}
    n_channels = len(eeg_signals)

    if n_channels < 2:
        print("At least 2 channels are required for coherence analysis")
        return {}

    # Calculate coherence for all channel pairs
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            # Get signals for channel pair
            signal_i = eeg_signals[i]
            signal_j = eeg_signals[j]

            # Calculate coherence using Welch's method
            f, Cxy = signal.coherence(signal_i, signal_j, fs=fs, nperseg=min(256, len(signal_i)))

            # Calculate mean coherence within each frequency band
            for band_name, (fmin, fmax) in bands.items():
                band_mask = (f >= fmin) & (f <= fmax)

                if np.sum(band_mask) > 0:
                    band_coherence = np.mean(Cxy[band_mask])
                else:
                    band_coherence = 0

                features[f'coherence_{i}_{j}_{band_name}'] = band_coherence

    # Calculate global coherence metrics (average across all channel pairs)
    for band_name in bands.keys():
        band_coherences = [v for k, v in features.items() if f'_{band_name}' in k]
        if band_coherences:
            features[f'global_coherence_{band_name}'] = np.mean(band_coherences)
            features[f'std_coherence_{band_name}'] = np.std(band_coherences)

    return features


def extract_phase_lag_index(eeg_signals, fs=250, bands=None):
    """
    Extract Phase Lag Index (PLI) features between EEG channels.
    PLI measures the asymmetry of the phase difference distribution.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    bands : dict or None
        Dictionary of frequency bands in format {name: (min_freq, max_freq)}

    Returns:
    --------
    dict
        Dictionary of PLI features
    """
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

    if n_channels < 2:
        print("At least 2 channels are required for PLI analysis")
        return {}

    # For each frequency band
    for band_name, (fmin, fmax) in bands.items():
        # Filter signals in the frequency band
        filtered_signals = []
        for i in range(n_channels):
            b, a = signal.butter(4, [fmin, fmax], btype='bandpass', fs=fs)
            filtered_signals.append(signal.filtfilt(b, a, eeg_signals[i]))

        # Calculate PLI for all channel pairs
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Compute Hilbert transform to get analytic signal
                analytic_i = signal.hilbert(filtered_signals[i])
                analytic_j = signal.hilbert(filtered_signals[j])

                # Extract instantaneous phase
                phase_i = np.angle(analytic_i)
                phase_j = np.angle(analytic_j)

                # Calculate phase difference
                phase_diff = phase_i - phase_j

                # Calculate PLI (mean of sign of phase difference)
                pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))

                features[f'pli_{i}_{j}_{band_name}'] = pli

        # Calculate global PLI metrics
        band_plis = [v for k, v in features.items() if f'pli_' in k and f'_{band_name}' in k]
        if band_plis:
            features[f'global_pli_{band_name}'] = np.mean(band_plis)
            features[f'std_pli_{band_name}'] = np.std(band_plis)

    return features


def extract_weighted_phase_lag_index(eeg_signals, fs=250, bands=None):
    """
    Extract Weighted Phase Lag Index (wPLI) features between EEG channels.
    wPLI is less sensitive to volume conduction effects than PLI.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    bands : dict or None
        Dictionary of frequency bands in format {name: (min_freq, max_freq)}

    Returns:
    --------
    dict
        Dictionary of wPLI features
    """
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

    if n_channels < 2:
        print("At least 2 channels are required for wPLI analysis")
        return {}

    # For each frequency band
    for band_name, (fmin, fmax) in bands.items():
        # Filter signals in the frequency band
        filtered_signals = []
        for i in range(n_channels):
            b, a = signal.butter(4, [fmin, fmax], btype='bandpass', fs=fs)
            filtered_signals.append(signal.filtfilt(b, a, eeg_signals[i]))

        # Calculate wPLI for all channel pairs
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Compute Hilbert transform to get analytic signal
                analytic_i = signal.hilbert(filtered_signals[i])
                analytic_j = signal.hilbert(filtered_signals[j])

                # Cross-spectrum
                cross_spec = analytic_i * np.conj(analytic_j)

                # Imaginary part of cross-spectrum
                imag_cs = np.imag(cross_spec)

                # Weighted Phase Lag Index
                if np.sum(np.abs(imag_cs)) > 0:
                    wpli = np.abs(np.mean(np.abs(imag_cs) * np.sign(imag_cs))) / np.mean(np.abs(imag_cs))
                else:
                    wpli = 0

                features[f'wpli_{i}_{j}_{band_name}'] = wpli

        # Calculate global wPLI metrics
        band_wplis = [v for k, v in features.items() if f'wpli_' in k and f'_{band_name}' in k]
        if band_wplis:
            features[f'global_wpli_{band_name}'] = np.mean(band_wplis)
            features[f'std_wpli_{band_name}'] = np.std(band_wplis)

    return features


def extract_phase_amplitude_coupling(eeg_signal, fs=250, phase_bands=None, amp_bands=None):
    """
    Extract Phase-Amplitude Coupling (PAC) features from a single EEG channel.

    Parameters:
    -----------
    eeg_signal : numpy.ndarray
        Single channel EEG signal
    fs : float
        Sampling frequency in Hz
    phase_bands : dict or None
        Dictionary of phase frequency bands in format {name: (min_freq, max_freq)}
    amp_bands : dict or None
        Dictionary of amplitude frequency bands in format {name: (min_freq, max_freq)}

    Returns:
    --------
    dict
        Dictionary of PAC features
    """
    if phase_bands is None:
        phase_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13)
        }

    if amp_bands is None:
        amp_bands = {
            'beta': (13, 30),
            'gamma': (30, 80)
        }

    features = {}

    try:
        # For each phase band
        for phase_band_name, (phase_low, phase_high) in phase_bands.items():
            # Filter signal for phase band
            b, a = signal.butter(4, [phase_low, phase_high], btype='bandpass', fs=fs)
            phase_filtered = signal.filtfilt(b, a, eeg_signal)

            # Extract phase using Hilbert transform
            phase_analytic = signal.hilbert(phase_filtered)
            phase = np.angle(phase_analytic)

            # For each amplitude band
            for amp_band_name, (amp_low, amp_high) in amp_bands.items():
                # Filter signal for amplitude band
                b, a = signal.butter(4, [amp_low, amp_high], btype='bandpass', fs=fs)
                amp_filtered = signal.filtfilt(b, a, eeg_signal)

                # Extract amplitude envelope using Hilbert transform
                amp_analytic = signal.hilbert(amp_filtered)
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
        print(f"Error extracting PAC features: {e}")
        for phase_band_name in phase_bands:
            for amp_band_name in amp_bands:
                features[f'pac_mi_{phase_band_name}_{amp_band_name}'] = 0

    return features


def extract_granger_causality(eeg_signals, fs=250, max_lag=10, bands=None):
    """
    Extract Granger Causality features between EEG channels.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    max_lag : int
        Maximum lag for the Granger causality test
    bands : dict or None
        Dictionary of frequency bands in format {name: (min_freq, max_freq)}

    Returns:
    --------
    dict
        Dictionary of Granger causality features
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        print("statsmodels package is required for Granger causality analysis")
        return {}

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

    if n_channels < 2:
        print("At least 2 channels are required for Granger causality analysis")
        return {}

    # For each frequency band
    for band_name, (fmin, fmax) in bands.items():
        # Filter signals in the frequency band
        filtered_signals = []
        for i in range(n_channels):
            b, a = signal.butter(4, [fmin, fmax], btype='bandpass', fs=fs)
            filtered_signals.append(signal.filtfilt(b, a, eeg_signals[i]))

        # Calculate Granger causality for all channel pairs
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:  # Skip self-connections
                    try:
                        # Prepare data
                        data = np.column_stack([filtered_signals[j], filtered_signals[i]])

                        # Run Granger causality test
                        gc_res = grangercausalitytests(data, max_lag, verbose=False)

                        # Extract p-values from F-test
                        p_values = [res[0]['ssr_ftest'][1] for res in gc_res.values()]

                        # Calculate Granger causality index (1 - min p-value)
                        # Smaller p-value = stronger evidence of causality
                        gc_index = 1 - min(p_values)

                        features[f'granger_{i}_to_{j}_{band_name}'] = gc_index
                    except Exception as e:
                        print(f"Error in Granger causality test for channels {i}->{j}: {e}")
                        features[f'granger_{i}_to_{j}_{band_name}'] = 0

        # Calculate global directional metrics
        band_gc_values = [v for k, v in features.items() if f'granger_' in k and f'_{band_name}' in k]
        if band_gc_values:
            features[f'global_granger_{band_name}'] = np.mean(band_gc_values)
            features[f'std_granger_{band_name}'] = np.std(band_gc_values)

    return features


def extract_directed_transfer_function(eeg_signals, fs=250, bands=None):
    """
    Extract Directed Transfer Function (DTF) features between EEG channels.

    Note: This is a simplified implementation of DTF.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    bands : dict or None
        Dictionary of frequency bands in format {name: (min_freq, max_freq)}

    Returns:
    --------
    dict
        Dictionary of DTF features
    """
    try:
        from statsmodels.tsa.api import VAR
    except ImportError:
        print("statsmodels package is required for DTF analysis")
        return {}

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

    if n_channels < 2:
        print("At least 2 channels are required for DTF analysis")
        return {}

    # For each frequency band
    for band_name, (fmin, fmax) in bands.items():
        # Filter signals in the frequency band
        filtered_signals = []
        for i in range(n_channels):
            b, a = signal.butter(4, [fmin, fmax], btype='bandpass', fs=fs)
            filtered_signals.append(signal.filtfilt(b, a, eeg_signals[i]))

        try:
            # Prepare data for VAR model
            data = np.column_stack(filtered_signals)

            # Fit VAR model
            model = VAR(data)
            results = model.fit(maxlags=10, ic='aic')

            # Get VAR coefficients
            coefs = results.coefs

            # Calculate DTF for each frequency
            n_freqs = 10  # Number of frequency points to evaluate
            freqs = np.linspace(fmin, fmax, n_freqs)

            # Initialize DTF matrix: channels x channels
            dtf_matrix = np.zeros((n_channels, n_channels))

            # For each frequency point
            for freq in freqs:
                # Calculate transfer function H(f)
                # This is a simplified approach
                # In practice, you would compute the full spectral transfer function

                # Approximate using only the first lag coefficient
                # H(f) ≈ I - A₁
                transfer_matrix = np.eye(n_channels) - coefs[0]

                # Sum up the magnitudes across frequencies
                dtf_matrix += np.abs(transfer_matrix) ** 2

            # Normalize by the number of frequencies
            dtf_matrix /= n_freqs

            # Store DTF values
            for i in range(n_channels):
                for j in range(n_channels):
                    if i != j:  # Skip self-connections
                        features[f'dtf_{i}_to_{j}_{band_name}'] = dtf_matrix[i, j]

            # Calculate global DTF metrics
            dtf_values = [v for k, v in features.items() if f'dtf_' in k and f'_{band_name}' in k]
            if dtf_values:
                features[f'global_dtf_{band_name}'] = np.mean(dtf_values)
                features[f'std_dtf_{band_name}'] = np.std(dtf_values)

        except Exception as e:
            print(f"Error in DTF analysis for band {band_name}: {e}")
            # Initialize with zeros in case of error
            for i in range(n_channels):
                for j in range(n_channels):
                    if i != j:
                        features[f'dtf_{i}_to_{j}_{band_name}'] = 0
            features[f'global_dtf_{band_name}'] = 0
            features[f'std_dtf_{band_name}'] = 0

    return features


def extract_graph_features(eeg_signals, fs=250, bands=None, threshold=0.5):
    """
    Extract graph theory features from EEG connectivity networks.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    bands : dict or None
        Dictionary of frequency bands in format {name: (min_freq, max_freq)}
    threshold : float
        Threshold for binarizing connectivity matrices

    Returns:
    --------
    dict
        Dictionary of graph features
    """
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

    if n_channels < 3:
        print("At least 3 channels are required for meaningful graph analysis")
        return {}

    # For each frequency band
    for band_name, (fmin, fmax) in bands.items():
        # Calculate coherence matrix
        coherence_matrix = np.zeros((n_channels, n_channels))

        # Filter signals in the frequency band
        filtered_signals = []
        for i in range(n_channels):
            b, a = signal.butter(4, [fmin, fmax], btype='bandpass', fs=fs)
            filtered_signals.append(signal.filtfilt(b, a, eeg_signals[i]))

        # Calculate coherence for all channel pairs
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    coherence_matrix[i, j] = 1.0  # Self-coherence
                elif i < j:
                    # Calculate coherence
                    f, Cxy = signal.coherence(filtered_signals[i], filtered_signals[j], fs=fs)

                    # Get frequency range of interest
                    mask = (f >= fmin) & (f <= fmax)
                    band_coherence = np.mean(Cxy[mask]) if np.sum(mask) > 0 else 0

                    coherence_matrix[i, j] = band_coherence
                    coherence_matrix[j, i] = band_coherence  # Symmetrical

        # Create binary adjacency matrix
        adjacency_matrix = (coherence_matrix > threshold).astype(int)

        # Remove self-connections
        np.fill_diagonal(adjacency_matrix, 0)

        # Create graph
        G = nx.from_numpy_array(adjacency_matrix)

        try:
            # Calculate global graph measures

            # Density (fraction of possible edges)
            features[f'graph_density_{band_name}'] = nx.density(G)

            # Average clustering coefficient
            features[f'graph_clustering_{band_name}'] = nx.average_clustering(G)

            # Average shortest path length
            if nx.is_connected(G):
                features[f'graph_path_length_{band_name}'] = nx.average_shortest_path_length(G)
            else:
                # For disconnected graphs, calculate average over connected components
                components = list(nx.connected_components(G))
                path_lengths = []
                for component in components:
                    if len(component) > 1:  # Need at least 2 nodes for paths
                        subgraph = G.subgraph(component)
                        path_lengths.append(nx.average_shortest_path_length(subgraph))
                features[f'graph_path_length_{band_name}'] = np.mean(path_lengths) if path_lengths else 0

            # Global efficiency
            features[f'graph_efficiency_{band_name}'] = nx.global_efficiency(G)

            # Small-worldness
            # (high clustering coefficient and low path length relative to random network)

            # Random network properties (Erdős–Rényi model)
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            p_random = (2 * n_edges) / (n_nodes * (n_nodes - 1))  # Edge probability in random graph

            # Create random graph with same number of nodes and edge probability
            random_G = nx.erdos_renyi_graph(n_nodes, p_random)

            # Calculate small-worldness
            if nx.is_connected(random_G) and nx.average_clustering(random_G) > 0:
                random_clustering = nx.average_clustering(random_G)
                random_path_length = nx.average_shortest_path_length(random_G)

                clustering_ratio = features[f'graph_clustering_{band_name}'] / random_clustering
                path_length_ratio = features[f'graph_path_length_{band_name}'] / random_path_length

                features[f'graph_small_worldness_{band_name}'] = clustering_ratio / path_length_ratio
            else:
                features[f'graph_small_worldness_{band_name}'] = 0

            # Calculate node-level measures

            # Degree centrality
            degree_centrality = nx.degree_centrality(G)
            features[f'graph_degree_centrality_{band_name}'] = np.mean(list(degree_centrality.values()))

            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(G)
            features[f'graph_betweenness_centrality_{band_name}'] = np.mean(list(betweenness_centrality.values()))

            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(G)
            features[f'graph_closeness_centrality_{band_name}'] = np.mean(list(closeness_centrality.values()))

            # Eigenvector centrality
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
                features[f'graph_eigenvector_centrality_{band_name}'] = np.mean(list(eigenvector_centrality.values()))
            except nx.PowerIterationFailedConvergence:
                features[f'graph_eigenvector_centrality_{band_name}'] = 0

            # Modularity
            try:
                communities = nx.community.greedy_modularity_communities(G)
                modularity = nx.community.modularity(G, communities)
                features[f'graph_modularity_{band_name}'] = modularity
                features[f'graph_num_communities_{band_name}'] = len(communities)
            except Exception as e:
                print(f"Error calculating modularity: {e}")
                features[f'graph_modularity_{band_name}'] = 0
                features[f'graph_num_communities_{band_name}'] = 0

        except Exception as e:
            print(f"Error in graph analysis for band {band_name}: {e}")
            # Initialize with zeros in case of error
            features[f'graph_density_{band_name}'] = 0
            features[f'graph_clustering_{band_name}'] = 0
            features[f'graph_path_length_{band_name}'] = 0
            features[f'graph_efficiency_{band_name}'] = 0
            features[f'graph_small_worldness_{band_name}'] = 0
            features[f'graph_degree_centrality_{band_name}'] = 0
            features[f'graph_betweenness_centrality_{band_name}'] = 0
            features[f'graph_closeness_centrality_{band_name}'] = 0
            features[f'graph_eigenvector_centrality_{band_name}'] = 0
            features[f'graph_modularity_{band_name}'] = 0
            features[f'graph_num_communities_{band_name}'] = 0

    return features


def extract_all_connectivity_features(eeg_signals, fs=250, bands=None):
    """
    Extract all connectivity features for multi-channel EEG signals.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    bands : dict or None
        Dictionary of frequency bands in format {name: (min_freq, max_freq)}

    Returns:
    --------
    dict
        Dictionary of all connectivity features
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

    # Extract different types of connectivity features
    coherence_features = extract_coherence_features(eeg_signals, fs, bands)

    try:
        pli_features = extract_phase_lag_index(eeg_signals, fs, bands)
    except Exception as e:
        print(f"Error extracting Phase Lag Index features: {e}")
        pli_features = {}

    try:
        wpli_features = extract_weighted_phase_lag_index(eeg_signals, fs, bands)
    except Exception as e:
        print(f"Error extracting Weighted Phase Lag Index features: {e}")
        wpli_features = {}

    # PAC features for each channel individually
    pac_features = {}
    for i, eeg_signal in enumerate(eeg_signals):
        try:
            channel_pac = extract_phase_amplitude_coupling(eeg_signal, fs)
            for k, v in channel_pac.items():
                pac_features[f'{k}_ch{i}'] = v
        except Exception as e:
            print(f"Error extracting PAC features for channel {i}: {e}")

    try:
        graph_features = extract_graph_features(eeg_signals, fs, bands)
    except Exception as e:
        print(f"Error extracting graph features: {e}")
        graph_features = {}

    # Combine all features
    all_features = {}
    all_features.update(coherence_features)
    all_features.update(pli_features)
    all_features.update(wpli_features)
    all_features.update(pac_features)
    all_features.update(graph_features)

    # Try to extract more advanced features if possible
    try:
        granger_features = extract_granger_causality(eeg_signals, fs, bands=bands)
        all_features.update(granger_features)
    except Exception as e:
        print(f"Error extracting Granger causality features: {e}")

    try:
        dtf_features = extract_directed_transfer_function(eeg_signals, fs, bands)
        all_features.update(dtf_features)
    except Exception as e:
        print(f"Error extracting DTF features: {e}")

    return all_features


def extract_multichannel_connectivity_features(eeg_signals, fs=250, bands=None):
    """
    Extract connectivity features for multiple EEG channels and convert to DataFrame.

    Parameters:
    -----------
    eeg_signals : numpy.ndarray
        Multi-channel EEG signals with shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    bands : dict or None
        Dictionary of frequency bands in format {name: (min_freq, max_freq)}

    Returns:
    --------
    pandas.DataFrame
        DataFrame with connectivity features
    """
    # Extract connectivity features
    connectivity_features = extract_all_connectivity_features(eeg_signals, fs, bands)

    # Convert to DataFrame (single row)
    features_df = pd.DataFrame([connectivity_features])

    # Add summary statistics
    # Calculate averages across frequency bands
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma'] if bands is None else list(bands.keys())

    # Coherence summary
    coherence_values = []
    for band in band_names:
        band_coherences = [v for k, v in connectivity_features.items()
                           if f'coherence_' in k and f'_{band}' in k and not k.startswith('global')]
        if band_coherences:
            coherence_values.extend(band_coherences)

    if coherence_values:
        features_df['mean_coherence'] = np.mean(coherence_values)
        features_df['max_coherence'] = np.max(coherence_values)
    else:
        features_df['mean_coherence'] = 0
        features_df['max_coherence'] = 0

    # PLI summary
    pli_values = []
    for band in band_names:
        band_plis = [v for k, v in connectivity_features.items()
                     if f'pli_' in k and f'_{band}' in k and not k.startswith('global')]
        if band_plis:
            pli_values.extend(band_plis)

    if pli_values:
        features_df['mean_pli'] = np.mean(pli_values)
        features_df['max_pli'] = np.max(pli_values)
    else:
        features_df['mean_pli'] = 0
        features_df['max_pli'] = 0

    # wPLI summary
    wpli_values = []
    for band in band_names:
        band_wplis = [v for k, v in connectivity_features.items()
                      if f'wpli_' in k and f'_{band}' in k and not k.startswith('global')]
        if band_wplis:
            wpli_values.extend(band_wplis)

    if wpli_values:
        features_df['mean_wpli'] = np.mean(wpli_values)
        features_df['max_wpli'] = np.max(wpli_values)
    else:
        features_df['mean_wpli'] = 0
        features_df['max_wpli'] = 0

    # Graph metrics summary
    for metric in ['density', 'clustering', 'efficiency', 'small_worldness']:
        metric_values = [v for k, v in connectivity_features.items() if f'graph_{metric}_' in k]
        if metric_values:
            features_df[f'mean_graph_{metric}'] = np.mean(metric_values)
        else:
            features_df[f'mean_graph_{metric}'] = 0

    return features_df

