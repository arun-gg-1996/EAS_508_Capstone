import torch
import numpy as np
from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGNetv4
from collections import OrderedDict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def create_dummy_eeg_input(n_channels=20, n_times=1000, batch_size=1):
    """Create a dummy EEG input tensor for testing"""
    return torch.randn(batch_size, n_channels, n_times).to(device)


def get_model(model_name, n_channels=20, input_window_samples=1000):
    """Get a pre-trained model for feature extraction"""
    if model_name == 'shallow':
        model = ShallowFBCSPNet(
            n_channels,
            n_classes=2,  # Binary classification by default
            input_window_samples=input_window_samples,
            final_conv_length='auto'
        )
    elif model_name == 'deep4':
        model = Deep4Net(
            n_channels,
            n_classes=2,  # Binary classification by default
            input_window_samples=input_window_samples,
            final_conv_length='auto'
        )
    elif model_name == 'eegnet':
        model = EEGNetv4(
            n_channels,
            n_classes=2,  # Binary classification by default
            input_window_samples=input_window_samples,
            final_conv_length='auto'
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Set to eval mode for feature extraction
    model.eval()
    return model.to(device)


def inspect_model_layers(model_name):
    """Inspect the layers of a model and their output shapes"""
    print(f"\n{'=' * 80}")
    print(f"INSPECTING MODEL: {model_name}")
    print(f"{'=' * 80}")

    # Create model
    model = get_model(model_name)

    # Create dummy input
    dummy_input = create_dummy_eeg_input()

    # Get all activations and their shapes
    activations = {}
    layer_order = []

    # Define hook function to capture activations
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.shape
                if name not in layer_order:
                    layer_order.append(name)

        return hook

    # Register hooks for all modules
    hooks = []
    for name, module in model.named_modules():
        if name and not any(x in name for x in ['bias', 'weight', '_modules']):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Run a forward pass
    with torch.no_grad():
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print model architecture
    print("\nMODEL ARCHITECTURE:")
    print("-" * 80)
    print(model)

    # Print layer shapes in order of execution
    print("\nLAYER OUTPUT SHAPES (in execution order):")
    print("-" * 80)
    print(f"{'Layer Name':<40} {'Output Shape':<30}")
    print("-" * 80)

    for name in layer_order:
        print(f"{name:<40} {str(activations[name]):<30}")

    # Identify potential penultimate layers
    print("\nPOTENTIAL PENULTIMATE LAYERS:")
    print("-" * 80)

    # Look for the classifier layer
    classifier_layer = None
    for name in reversed(layer_order):
        if 'classifier' in name or name.endswith('softmax') or 'conv_classifier' in name or name.endswith('logits'):
            classifier_layer = name
            break

    # If we found the classifier, look for the layer right before it
    if classifier_layer:
        idx = layer_order.index(classifier_layer)
        if idx > 0:
            # Get the layer right before the classifier
            penultimate_layer = layer_order[idx - 1]
            print(f"Found classifier layer: {classifier_layer}")
            print(f"Penultimate layer: {penultimate_layer}")
            print(f"Penultimate layer shape: {activations[penultimate_layer]}")

            # Calculate the flattened size
            shape = activations[penultimate_layer]
            if len(shape) > 2:
                flat_size = shape[0] * np.prod(shape[1:]).astype(int)
                print(f"Flattened size: {flat_size}")
            else:
                print(f"Already flat with size: {shape[1]}")
    else:
        print("No clear classifier layer found.")

        # Suggest potential feature-rich layers
        print("\nSuggested feature-rich layers:")
        for name, shape in activations.items():
            if len(shape) > 2 or (len(shape) == 2 and shape[1] > 10):
                flat_size = shape[0] * np.prod(shape[1:]).astype(int) if len(shape) > 2 else shape[1]
                print(f"Layer: {name}, Shape: {shape}, Flattened size: {flat_size}")

    return activations, layer_order


def examine_all_models():
    """Examine all three models and compare them"""
    models = ['shallow', 'deep4', 'eegnet']

    model_info = {}

    for model_name in models:
        activations, layer_order = inspect_model_layers(model_name)

        # Store model info
        model_info[model_name] = {
            'activations': activations,
            'layer_order': layer_order
        }

    # Print a comparison of the models
    print("\n" + "=" * 100)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 100)

    print(f"{'Model':<10} {'Total Layers':<15} {'Last Layer Shape':<20} {'Potential Embedding Size':<25}")
    print("-" * 100)

    for model_name, info in model_info.items():
        activations = info['activations']
        layer_order = info['layer_order']

        # Find potential embedding layer
        embedding_layer = None
        embedding_size = 0

        # Look for classifier
        classifier_layer = None
        for name in reversed(layer_order):
            if 'classifier' in name or name.endswith('softmax') or 'conv_classifier' in name:
                classifier_layer = name
                break

        # If found, get penultimate layer
        if classifier_layer:
            idx = layer_order.index(classifier_layer)
            if idx > 0:
                embedding_layer = layer_order[idx - 1]
                shape = activations[embedding_layer]
                if len(shape) > 2:
                    embedding_size = np.prod(shape[1:]).astype(int)
                else:
                    embedding_size = shape[1]

        # If no clear penultimate layer, try to find a good feature layer
        if not embedding_layer:
            # Find the layer with the largest feature dimension
            for name, shape in activations.items():
                if len(shape) > 2:  # Conv layer
                    size = np.prod(shape[1:]).astype(int)
                    if size > embedding_size:
                        embedding_size = size
                        embedding_layer = name
                elif len(shape) == 2 and shape[1] > embedding_size:  # Linear layer
                    embedding_size = shape[1]
                    embedding_layer = name

        last_layer = layer_order[-1]
        last_shape = activations[last_layer]

        print(f"{model_name:<10} {len(layer_order):<15} {str(last_shape):<20} {embedding_size:<25}")

    print("\nRECOMMENDATIONS:")
    print("-" * 100)
    for model_name, info in model_info.items():
        activations = info['activations']
        layer_order = info['layer_order']

        # First try to find true penultimate layer
        classifier_layer = None
        for name in reversed(layer_order):
            if 'classifier' in name or name.endswith('softmax') or 'conv_classifier' in name:
                classifier_layer = name
                break

        penultimate_layer = None
        if classifier_layer:
            idx = layer_order.index(classifier_layer)
            if idx > 0:
                penultimate_layer = layer_order[idx - 1]

        # If found, recommend it
        if penultimate_layer:
            shape = activations[penultimate_layer]
            if len(shape) > 2:
                flat_size = np.prod(shape[1:]).astype(int)
                print(f"{model_name}: Use layer '{penultimate_layer}' with flattened size {flat_size}")
            else:
                print(f"{model_name}: Use layer '{penultimate_layer}' with size {shape[1]}")
        else:
            # Otherwise find a good feature layer
            best_layer = None
            best_size = 0

            for name, shape in activations.items():
                if len(shape) > 2:  # Conv layer
                    size = np.prod(shape[1:]).astype(int)
                    if size > best_size:
                        best_size = size
                        best_layer = name
                elif len(shape) == 2 and shape[1] > best_size:  # Linear layer
                    best_size = shape[1]
                    best_layer = name

            if best_layer:
                shape = activations[best_layer]
                if len(shape) > 2:
                    flat_size = np.prod(shape[1:]).astype(int)
                    print(f"{model_name}: Use layer '{best_layer}' with flattened size {flat_size}")
                else:
                    print(f"{model_name}: Use layer '{best_layer}' with size {shape[1]}")


def find_good_embedding_layers():
    """Print specific recommendations for each model"""
    models = ['shallow', 'deep4', 'eegnet']

    print("\n" + "=" * 100)
    print("RECOMMENDED EMBEDDING LAYERS")
    print("=" * 100)

    for model_name in models:
        # Create model
        model = get_model(model_name)

        # Create dummy input
        dummy_input = create_dummy_eeg_input()

        # Create a summary of all layers with their shapes
        layer_info = OrderedDict()

        def hook_fn(name):
            def hook(module, input, output):
                layer_info[name] = {
                    'output_shape': output.shape if isinstance(output, torch.Tensor) else None,
                    'module_type': type(module).__name__,
                    'module': module
                }

            return hook

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name and not any(x in name for x in ['bias', 'weight', '_modules']):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Run forward pass
        with torch.no_grad():
            model(dummy_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Find the classifier and its input layer
        classifier_layer = None
        for name, info in reversed(list(layer_info.items())):
            if ('classifier' in name or 'softmax' in name or 'conv_classifier' in name or
                    'pred' in name or 'conv_class' in name):
                classifier_layer = name
                break

        # Print findings
        print(f"\nModel: {model_name}")
        print("-" * 50)

        if classifier_layer:
            # Find the penultimate layer
            penultimate_layer = None
            all_layers = list(layer_info.keys())

            if classifier_layer in all_layers:
                idx = all_layers.index(classifier_layer)
                if idx > 0:
                    penultimate_layer = all_layers[idx - 1]

            if penultimate_layer:
                shape = layer_info[penultimate_layer]['output_shape']
                flat_size = np.prod(shape[1:]).astype(int) if len(shape) > 2 else shape[1]
                print(f"Classifier identified: {classifier_layer}")
                print(f"Best embedding layer: {penultimate_layer}")
                print(f"Layer shape: {shape}")
                print(f"Flattened size: {flat_size}")

                # Print code to extract this specific layer
                print("\nCode to extract this layer:")
                print("```python")
                if len(shape) > 2:
                    print(f"# For model {model_name}, extract '{penultimate_layer}' and flatten")
                    print("def extract_embeddings(model, eeg_data):")
                    print("    activations = {}")
                    print("    def hook_fn(module, input, output):")
                    print(f"        activations['{penultimate_layer}'] = output")
                    print(f"    # Register hook for {penultimate_layer}")
                    print(f"    hook = model.{penultimate_layer}.register_forward_hook(hook_fn)")
                    print("    # Forward pass")
                    print("    with torch.no_grad():")
                    print("        model(eeg_data)")
                    print("    # Remove hook")
                    print("    hook.remove()")
                    print("    # Get activations and flatten")
                    print(f"    features = activations['{penultimate_layer}']")
                    print("    features_flat = features.reshape(features.shape[0], -1)")
                    print("    return features_flat.cpu().numpy()")
                else:
                    print(f"# For model {model_name}, extract '{penultimate_layer}'")
                    print("def extract_embeddings(model, eeg_data):")
                    print("    activations = {}")
                    print("    def hook_fn(module, input, output):")
                    print(f"        activations['{penultimate_layer}'] = output")
                    print(f"    # Register hook for {penultimate_layer}")
                    print(f"    hook = model.{penultimate_layer}.register_forward_hook(hook_fn)")
                    print("    # Forward pass")
                    print("    with torch.no_grad():")
                    print("        model(eeg_data)")
                    print("    # Remove hook")
                    print("    hook.remove()")
                    print(f"    # Get activations (already flat)")
                    print(f"    features = activations['{penultimate_layer}']")
                    print("    return features.cpu().numpy()")
                print("```")
            else:
                print("Could not identify penultimate layer clearly. Suggesting alternatives:")

                # Find alternative feature-rich layers
                best_layers = []
                for name, info in layer_info.items():
                    if info['output_shape'] is None:
                        continue

                    shape = info['output_shape']
                    flat_size = np.prod(shape[1:]).astype(int) if len(shape) > 2 else shape[1]

                    if flat_size > 10:  # Only consider layers with decent number of features
                        best_layers.append((name, shape, flat_size))

                # Sort by feature size
                best_layers.sort(key=lambda x: x[2], reverse=True)

                # Print top 3 alternatives
                for i, (name, shape, flat_size) in enumerate(best_layers[:3]):
                    print(f"Alternative {i + 1}: Layer '{name}' with shape {shape} (flattened: {flat_size})")
        else:
            print("Could not identify classifier layer. Suggesting best feature layers:")

            # Find layers with rich features
            best_layers = []
            for name, info in layer_info.items():
                if info['output_shape'] is None:
                    continue

                shape = info['output_shape']
                flat_size = np.prod(shape[1:]).astype(int) if len(shape) > 2 else shape[1]

                if flat_size > 10:  # Only consider layers with decent number of features
                    best_layers.append((name, shape, flat_size))

            # Sort by feature size
            best_layers.sort(key=lambda x: x[2], reverse=True)

            # Print top 3 alternatives
            for i, (name, shape, flat_size) in enumerate(best_layers[:3]):
                print(f"Alternative {i + 1}: Layer '{name}' with shape {shape} (flattened: {flat_size})")


if __name__ == "__main__":
    # Examine all models
    examine_all_models()

    # Get specific recommendations
    find_good_embedding_layers()