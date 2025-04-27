import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Hardcoded file paths
csv_path = "/Users/arun-ghontale/UB/Sem 1/EAS 508/EAS_508_Capstone/out/main/eeg_time_domain_features.csv"
output_path = "/Users/arun-ghontale/UB/Sem 1/EAS 508/EAS_508_Capstone/out/main/tsne/tsne_visualization_time_domain.html"

# Parameter for percentage of data to include from each class
sample_percentage = 5  # Adjust this value to control the amount of data (e.g., 5 means 5% of data)

# Load the dataset
print("Loading data...")
df = pd.read_csv(csv_path)

# Extract features and target
feature_cols = [col for col in df.columns if col.startswith('channel_')]
X = df[feature_cols]
y = df['expert_consensus']  # Target variable

# Check for missing values
print(f"Missing values in features: {X.isnull().sum().sum()}")

# Remove rows with any missing values
print("Removing rows with missing values...")
complete_rows = X.dropna().index
X_clean = X.loc[complete_rows]
y_clean = y.loc[complete_rows]

print(f"Data shape after removing missing values: {X_clean.shape}")

# Sample equal number of points from each class
print("Sampling equal number of points from each class...")
classes = y_clean.unique()
min_class_size = min([sum(y_clean == c) for c in classes])
samples_per_class = int(min_class_size * (sample_percentage / 100))

# Ensure we have at least some minimum number of samples
min_samples = max(50, samples_per_class)
samples_per_class = min(min_class_size, max(min_samples, samples_per_class))

print(f"Sampling {samples_per_class} points from each class")

# Sample data
sampled_indices = []
for c in classes:
    class_indices = (y_clean == c).to_numpy().nonzero()[0]
    sampled_indices.extend(np.random.choice(class_indices, size=samples_per_class, replace=False))

X_sampled = X_clean.iloc[sampled_indices]
y_sampled = y_clean.iloc[sampled_indices]

print(f"Final sampled data shape: {X_sampled.shape}")

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sampled)

# Apply t-SNE
print("Applying t-SNE (this may take a while)...")
tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, X_sampled.shape[0] // 4), max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# Create DataFrame for plotting
tsne_df = pd.DataFrame({
    'tsne_1': X_tsne[:, 0],
    'tsne_2': X_tsne[:, 1],
    'tsne_3': X_tsne[:, 2],
    'class': y_sampled.values
})

# Create 3D interactive plot
print("Creating visualization...")
fig = px.scatter_3d(
    tsne_df,
    x='tsne_1',
    y='tsne_2',
    z='tsne_3',
    color='class',
    title=f'3D t-SNE Visualization of EEG Features ({sample_percentage}% of data per class)',
    labels={'class': 'Brain Activity Class'},
    opacity=0.7
)

# Improve layout
fig.update_layout(
    scene=dict(
        xaxis_title='t-SNE Component 1',
        yaxis_title='t-SNE Component 2',
        zaxis_title='t-SNE Component 3'
    ),
    legend_title_text='Class',
    width=1000,
    height=800
)

# Save as HTML
print(f"Saving visualization to {output_path}...")
fig.write_html(output_path)
print("Visualization saved successfully!")