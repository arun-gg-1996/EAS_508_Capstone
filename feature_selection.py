import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

def shap_func(X_train, y_train, X, feature_names):
    # Train a model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Explain predictions using SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Visualize feature importance
    shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names)

    # Get global feature importance
    feature_importance = np.abs(shap_values).mean(0)
    feature_importance_normalized = feature_importance / np.sum(feature_importance)
    important_features = [feature_names[i] for i in np.argsort(feature_importance_normalized)[::-1]]

    # Select top K features
    top_k = 50
    selected_features_shap = important_features[:top_k]


def rfe(X_train, y_train, feature_names):
    # Create the RFE model
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=50, step=1)
    selector = selector.fit(X_train, y_train)

    # Get selected features
    selected_features_rfe = [feature_names[i] for i in range(len(feature_names)) if selector.support_[i]]

    # Get feature ranking
    feature_ranking = {feature_names[i]: selector.ranking_[i] for i in range(len(feature_names))}


def pca(X, feature_names):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep components that explain 95% of variance
    X_pca = pca.fit_transform(X_scaled)

    # Examine explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Components')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.grid()
    plt.show()

    # Get feature importance in PCA
    feature_importance_pca = np.abs(pca.components_).sum(axis=0)
    feature_importance_pca = feature_importance_pca / np.sum(feature_importance_pca)
    important_features_pca = [feature_names[i] for i in np.argsort(feature_importance_pca)[::-1]]


# 4. Removing Redundant Features (Correlation Analysis)
# Remove highly correlated features to reduce redundancy.
def correlation(X, feature_names):
    # Calculate correlation matrix
    X_df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = X_df.corr().abs()

    # Visualize correlation matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, cmap='viridis', annot=False)
    plt.title('Feature Correlation Matrix')
    plt.show()

    # Find highly correlated feature pairs (above threshold)
    threshold = 0.8
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant_features = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    print(f"Number of redundant features: {len(redundant_features)}")
    print("Redundant features:", redundant_features)

    # Create a list of features to keep
    features_to_keep = [f for f in feature_names if f not in redundant_features]


# 5. Combined Feature Selection Approach
# Combine multiple feature selection methods for robust results.
def combine_feature_selection_methods(X, y, feature_names, n_features=50):
    """Combine multiple feature selection methods."""
    # Train a model for SHAP
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    feature_importance_shap = np.abs(shap_values).mean(0)

    # RFE
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector = selector.fit(X_train, y_train)
    rfe_support = selector.support_

    # Correlation analysis
    X_df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = X_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
    correlation_support = [f not in redundant_features for f in feature_names]

    # Combine methods (more weight to SHAP)
    feature_scores = {}
    for i, feature in enumerate(feature_names):
        # Normalize SHAP importance
        shap_score = feature_importance_shap[i] / np.sum(feature_importance_shap)

        # Combine scores (you can adjust weights)
        combined_score = (
                0.5 * shap_score +
                0.3 * (1 if rfe_support[i] else 0) +
                0.2 * (1 if correlation_support[i] else 0)
        )

        feature_scores[feature] = combined_score

    # Sort features by combined score
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top features
    selected_features = [f[0] for f in sorted_features[:n_features]]

    return selected_features, feature_scores


# 6. Visualization and Interpretation
# Visualize feature importance for better interpretation.
def visualize_feature_importance(feature_scores, top_n=30):
    """Visualize feature importance scores."""
    # Sort features by importance
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top N features
    top_features = sorted_features[:top_n]

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.barh([f[0] for f in top_features][::-1], [f[1] for f in top_features][::-1])
    plt.xlabel('Feature Importance Score')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.show()

    # Group features by category
    feature_categories = {
        'Time-Domain': [f for f in top_features if f[0].startswith(('time_', 'hjorth_', 'statistical_'))],
        'Frequency-Domain': [f for f in top_features if f[0].startswith(('freq_', 'spectral_', 'band_'))],
        'Time-Frequency': [f for f in top_features if f[0].startswith(('wavelet_', 'tf_'))],
        'Non-Linear': [f for f in top_features if f[0].startswith(('nonlin_', 'entropy_'))],
        'Pattern-Specific': [f for f in top_features if f[0].startswith(('pattern_', 'seizure_'))],
        'Connectivity': [f for f in top_features if f[0].startswith(('conn_', 'graph_'))],
        'Mel-Spectrogram': [f for f in top_features if f[0].startswith('mel_')],
        'Image Embedding': [f for f in top_features if f[0].startswith(('efficientnet_', 'resnet_'))]
    }

    # Print summary by category
    print("Feature importance by category:")
    for category, features in feature_categories.items():
        if features:
            print(f"\n{category}:")
            for f in features:
                print(f"  - {f[0]}: {f[1]:.4f}")


