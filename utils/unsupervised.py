#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import pickle
import os
import gc

# Sequence validation function
def filter_sequence(seq):
    # Define standard amino acids
    standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWYst")

    
    # Replace non-standard amino acids with 'A'
    filtered_seq = ''.join(c if c in standard_amino_acids else 'A' for c in seq)
    
    return filtered_seq

def protein_onehot_embedding(sequence, sequence_name):
    # Define standard 20 amino acids order
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    length = len(sequence)
    
    # Initialize counts
    counts = np.zeros(len(amino_acids), dtype=float)
    
    # Count each amino acid occurrence
    for aa in sequence:
        if aa in amino_acids:
            idx = amino_acids.index(aa)
            counts[idx] += 1
    
    # Normalize counts to frequencies
    embedding = counts / length
    
    # Convert to DataFrames with dynamic column names
    protein_embedding_df = pd.DataFrame(embedding.reshape(1, -1),
                                        columns=[f"{sequence_name}_embedding_{aa}" for aa in amino_acids])
    
    return protein_embedding_df


def build_plm_features_df(sequence, sequence_name):

    wt_seq = filter_sequence(sequence) # sequence only have "ACDEFGHIKLMNPQRSTVWYst"
    # print("filter_sequence: ", wt_seq)
    seq = wt_seq.upper()

    protein_embedding_df = protein_onehot_embedding(sequence, sequence_name)

    gc.collect() 
    
    return protein_embedding_df


# PCA (Principal Component Analysis)
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def pca_reduction(embeddings, n_components=10, labels=None, plot_variance=False, plot_2d=True):
    """
    Perform PCA dimensionality reduction and plot explained variance and optional 2D scatter.

    Parameters:
    -----------
    embeddings : array-like
        High-dimensional data to reduce.
    n_components : int, default=10
        Number of PCA components to keep.
    plot_variance : bool, default=True
        Whether to plot cumulative explained variance.
    plot_2d : bool, default=True
        Whether to plot 2D scatter of first two PC components.
    labels : array-like, optional
        Labels or categories for coloring points in 2D scatter plot.

    Returns:
    --------
    embeddings_2d : numpy.ndarray
        Data projected to first two principal components.
    pca_result : numpy.ndarray
        Data projected to all chosen principal components.
    pca : PCA object
        Fitted PCA object for further analysis.
    pca_cutoff_dim : int or None
        Dimension where cumulative explained variance >= 95%, or None if not reached.
    """
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(embeddings_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find dimension where cumulative explained variance > 95%
    pca_cutoff_dim = None
    above_95 = np.where(cumulative_variance >= 0.95)[0]
    if len(above_95) > 0:
        pca_cutoff_dim = above_95[0] + 1  # Components are 1-indexed for reporting
        print(f"PCA dims to 95% variance: {pca_cutoff_dim}")

    # Plot cumulative explained variance
    if plot_variance:
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_variance, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        if pca_cutoff_dim is not None:
            plt.axvline(x=pca_cutoff_dim - 1, color='b', linestyle='--', label=f'{pca_cutoff_dim} Components')
            plt.legend()
        plt.show()
    
    # Plot 2D PCA scatter
    embeddings_2d = pca_result[:, :2]
    if plot_2d:
        plt.figure(figsize=(10, 8))
        if labels is not None:
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', s=1, alpha=0.2)
            plt.colorbar(scatter, label='Labels')
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=1, color='green')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title('PCA 2D Projection')
        plt.grid(True)
        plt.show()
    
    return embeddings_2d, pca_result, pca, pca_cutoff_dim

def optimize_umap_n_components(
    embeddings, 
    label=None, 
    n_components_range=range(2, 11),
    **umap_kwargs
):
    """
    Find the best number of UMAP components by evaluating quality for a range of dimensions.
    Also plots scores (silhouette or variance) over n_components.

    Parameters:
    -----------
    embeddings: array-like
        Original high-dimensional data.
    label: array-like or None
        Labels for silhouette scoring; if None, variance used as fallback.
    n_components_range: iterable of ints
        List or range of UMAP embedding dimensions to try.
    umap_kwargs: dict
        Other UMAP hyperparameters to keep fixed during this optimization.

    Returns:
    --------
    best_n_components: int
        The number of components that yielded the best score.
    score_dict: dict
        Mapping from n_components to scores.
    """

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    best_score = -np.inf
    best_n_components = None
    score_dict = {}

    for n in n_components_range:
        reducer = umap.UMAP(n_components=n, random_state=42, **umap_kwargs)
        embedding = reducer.fit_transform(embeddings_scaled)
        if label is not None:
            score = silhouette_score(embedding, label)
        else:
            score = np.var(embedding)
        score_dict[n] = score

        print(f"n_components={n}, score={score:.4f}")

        if score > best_score:
            best_score = score
            best_n_components = n

    print(f"Best number of components: {best_n_components} with score {best_score:.4f}")
    
    # Plotting
    plt.figure(figsize=(8, 5))
    x = list(score_dict.keys())
    y = list(score_dict.values())
    plt.plot(x, y, marker='o')
    plt.xlabel('Number of UMAP Components')
    ylabel = 'Silhouette Score' if label is not None else 'Total Variance of Embedding'
    plt.ylabel(ylabel)
    plt.title(f'UMAP Optimization: {ylabel} vs Number of Components')
    plt.grid(True)
    plt.show()

    return best_n_components, score_dict


# UMAP (Uniform Manifold Approximation and Projection)
import umap
import pandas as pd

def umap_reduction(
    embeddings, 
    n_components=2, 
    labels=None, 
    plot=True, 
    grid_search=True,
    **kwargs
):
    """
    Perform UMAP dimensionality reduction with optional hyperparameter tuning.

    Parameters:
    -----------
    embeddings : array-like
        High-dimensional data to reduce.
    n_components : int, default=2
        Number of UMAP dimensions to reduce to.
    labels : array-like, optional
        Labels or categories for coloring points in the plot.
    plot : bool, default=True
        Whether to generate a scatter plot of the UMAP embedding.
    grid_search : bool, default=True
        If True, performs grid search over n_neighbors and min_dist.
    **kwargs :
        Additional keyword arguments for UMAP.

    Returns:
    --------
    best_umap_result : numpy.ndarray
        Reduced embedding coordinates for the best parameter combination.
    best_reducer : umap.UMAP
        The UMAP reducer object for the best configuration.
    best_params : dict
        Dictionary of the best hyperparameter values.
    """

    # Standardize input embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Default parameters
    base_params = {
        "n_components": n_components,
        "metric": "euclidean",
        "random_state": 42
    }
    base_params.update(kwargs)

    param_grid = {
        "n_neighbors": [5, 15, 30],
        "min_dist": [0.1, 0.5]
    }

    best_score = -np.inf
    best_params = None
    best_umap_result = None
    best_reducer = None

    # Hyperparameter tuning loop
    if grid_search:
        print("Performing UMAP hyperparameter grid search...")
        for n in param_grid["n_neighbors"]:
            for d in param_grid["min_dist"]:
                params = base_params.copy()
                params.update({"n_neighbors": n, "min_dist": d})
                reducer = umap.UMAP(**params)
                umap_result = reducer.fit_transform(embeddings_scaled)

                # Evaluate embedding quality
                if labels is not None:
                    score = silhouette_score(umap_result, labels)
                else:
                    score = np.var(umap_result)  # fallback heuristic

                print(f"n_neighbors={n}, min_dist={d}, score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_umap_result = umap_result
                    best_reducer = reducer
        
        print(f"Best UMAP parameters: {best_params}, score={best_score:.4f}")
    else:
        best_params = base_params
        best_reducer = umap.UMAP(**base_params)
        best_umap_result = best_reducer.fit_transform(embeddings_scaled)

    # Plot 2D embedding
    if plot:
        plt.figure(figsize=(10, 8))
        if labels is not None:
            scatter = plt.scatter(
                best_umap_result[:, 0], best_umap_result[:, 1],
                c=labels, cmap='Spectral', s=1, alpha=0.2
            )
            plt.colorbar(scatter, label='Labels')
        else:
            plt.scatter(best_umap_result[:, 0], best_umap_result[:, 1], s=1, color='blue')
        plt.title(f"UMAP Projection (n_neighbors={best_params['n_neighbors']}, min_dist={best_params['min_dist']})")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(True)
        plt.show()

    return best_umap_result, best_reducer, best_params
    

# K-Means Clustering

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_optimal_kmeans_clusters(embeddings, max_k=15):
    """
    Find optimal number of clusters using Elbow Method and Silhouette Score.

    Parameters:
    -----------
    embeddings : array-like
        Data to be clustered.
    max_k : int, default=15
        Maximum number of clusters to try.

    Returns:
    --------
    best_k : int
        Optimal number of clusters (k) based on max Silhouette Score.
    k_range : range
        Range of cluster numbers evaluated.
    inertias : list
        Within-cluster sum of squares for each k.
    silhouette_scores : list
        Silhouette scores for each k.
    """
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, labels))



    # Determine the best k with highest silhouette score
    best_idx = silhouette_scores.index(max(silhouette_scores))
    best_k = k_range[best_idx]

    print(f"Best number of clusters by Silhouette Score: {best_k}")

    # Plot the results side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(k_range, inertias, 'bo-', label='Inertia')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Sum of Squared Distances)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(k_range, silhouette_scores, 'ro-', label='Silhouette Score')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis for Optimal k')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return best_k, k_range, inertias, silhouette_scores



# Visualization and Evaluation
def visualize_clusters(embeddings_2d, labels, title):
    """
    Visualize clustering results in 2D space
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='Spectral', s=1, alpha=0.2)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


def evaluate_clustering(embeddings, labels):
    """
    Evaluate clustering performance
    """
    if len(np.unique(labels)) > 1:  # Need at least 2 clusters for these metrics
        silhouette = silhouette_score(embeddings, labels)
        n_clusters = len(np.unique(labels[labels != -1]))  # Exclude noise points
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Silhouette Score: {silhouette:.3f}")
        
        # Count points in each cluster
        unique, counts = np.unique(labels, return_counts=True)
        print("Cluster distribution:")
        for cluster, count in zip(unique, counts):
            print(f"  Cluster {cluster}: {count} points")
    
    return n_clusters, silhouette



import umap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def sensitivity_analysis(data, labels=None):
    """
    Perform sensitivity analysis on K-Means clustering of UMAP embeddings.
    
    Parameters:
    -----------
    data : array-like
        Original high-dimensional data.
        
    Returns:
    --------
    None
    """

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    umap_n_neighbors_values = [15, 30, 45]
    kmeans_n_clusters_values = [12, 22, 32]

    print("Sensitivity to UMAP n_neighbors:")
    for n_neighbors in umap_n_neighbors_values:
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42, min_dist = 0.1, metric ='euclidean', n_components = 8)
        embedding = reducer.fit_transform(data_scaled)

        kmeans = KMeans(n_clusters=22, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding)

        if labels is not None:
            cluster_labels = labels
                    
        score = silhouette_score(embedding, cluster_labels)
        print(f"  n_neighbors={n_neighbors}: Silhouette Score = {score:.2f}")

    print("\nSensitivity to K-Means n_clusters:")
    reducer_optimal = umap.UMAP(n_neighbors=30, random_state=42, min_dist = 0.1, metric ='euclidean', n_components = 8)
    embedding_optimal = reducer_optimal.fit_transform(data_scaled)

    for n_clusters in kmeans_n_clusters_values:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding_optimal)
        
        if labels is not None:
            cluster_labels = labels       
            
        score = silhouette_score(embedding_optimal, cluster_labels)
        print(f"  n_clusters={n_clusters}: Silhouette Score = {score:.2f}")

    # Plot UMAP embeddings for visual inspection
    fig, axes = plt.subplots(1, len(umap_n_neighbors_values), figsize=(15, 5))
    for ax, n_neighbors in zip(axes, umap_n_neighbors_values):
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42, min_dist = 0.1, metric ='euclidean', n_components = 8)
        embedding = reducer.fit_transform(data_scaled)
        kmeans = KMeans(n_clusters=22, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding)

        if labels is not None:
            cluster_labels = labels 
            
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='Spectral', s=1)
        ax.set_title(f'UMAP n_neighbors={n_neighbors}\nSilhouette={silhouette_score(embedding, cluster_labels):.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def sensitivity_analysis_kmeans(data, labels=None):
    """
    Perform sensitivity analysis on K-Means clustering of UMAP embeddings.
    
    Parameters:
    -----------
    data : array-like
        Original high-dimensional data.
        
    Returns:
    --------
    None
    """

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    umap_n_neighbors_values = [15, 30, 45]
    kmeans_n_clusters_values = [12, 22, 32]

    print("Sensitivity to UMAP n_neighbors:")
    for n_neighbors in umap_n_neighbors_values:
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42, min_dist = 0.1, metric ='euclidean', n_components = 8)
        embedding = reducer.fit_transform(data_scaled)

        kmeans = KMeans(n_clusters=22, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding)

        if labels is not None:
            cluster_labels = labels
                    
        score = silhouette_score(embedding, cluster_labels)
        print(f"  n_neighbors={n_neighbors}: Silhouette Score = {score:.2f}")

    print("\nSensitivity to K-Means n_clusters:")
    reducer_optimal = umap.UMAP(n_neighbors=30, random_state=42, min_dist = 0.1, metric ='euclidean', n_components = 8)
    embedding_optimal = reducer_optimal.fit_transform(data_scaled)

    for n_clusters in kmeans_n_clusters_values:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding_optimal)
        
        if labels is not None:
            cluster_labels = labels       
            
        score = silhouette_score(embedding_optimal, cluster_labels)
        print(f"  n_clusters={n_clusters}: Silhouette Score = {score:.2f}")

    
    fig, axes = plt.subplots(1, len(kmeans_n_clusters_values), figsize=(15, 5))
    for ax, n_clusters in zip(axes, kmeans_n_clusters_values):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding_optimal)
        if labels is not None:
            cluster_labels = labels
        scatter = ax.scatter(embedding_optimal[:, 0], embedding_optimal[:, 1], c=cluster_labels, cmap='Spectral', s=1)
        ax.set_title(f'KMeans n_clusters={n_clusters}\nSilhouette={silhouette_score(embedding_optimal, cluster_labels):.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_unsupervised_features(embeddings, n_pca_components=88, n_umap_components=8, embedding_type="", training=False):
    """
    Create comprehensive unsupervised features from embeddings.

    Parameters:
    -----------
    embeddings : array-like
        Input data embeddings.
    n_pca_components : int
        Number of PCA components.
    n_umap_components : int
        Number of UMAP components.
    training : bool
        If True, fit and save models; if False, load models and transform.

    Returns:
    --------
    unsupervised_features_df : pd.DataFrame
        DataFrame combining PCA, UMAP, and clustering features.
    pca_model, umap_model, kmeans_model : fitted models
        Only returned in training mode; otherwise None.
    """
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    
    if training:
        # # PCA reduction and model fit
        # _, pca_features, pca_model, _ = pca_reduction(embeddings, n_pca_components, plot_variance=False, plot_2d=False)
        pca_model = PCA(n_components=n_pca_components)
        pca_features = pca_model.fit_transform(embeddings)        
        
        # # UMAP reduction and model fit
        # umap_features, umap_model, _ = umap_reduction(embeddings, n_umap_components, plot=False)
        umap_model = umap.UMAP(n_neighbors=30, random_state=42, min_dist = 0.1, metric ='euclidean', n_components = n_umap_components)
        umap_features = umap_model.fit_transform(embeddings)
        
        # KMeans clustering and fit
        kmeans_model = KMeans(n_clusters=22, random_state=42)
        kmeans_model.fit(umap_features)
        kmeans_labels = kmeans_model.predict(umap_features)
        
        # Save models
        with open(os.path.join(model_dir, f'{embedding_type}pca_model.pkl'), 'wb') as f_pca:
            pickle.dump(pca_model, f_pca)
        with open(os.path.join(model_dir, f'{embedding_type}umap_model.pkl'), 'wb') as f_umap:
            pickle.dump(umap_model, f_umap)
        with open(os.path.join(model_dir, f'{embedding_type}kmeans_model.pkl'), 'wb') as f_kmeans:
            pickle.dump(kmeans_model, f_kmeans)
        
    else:
        # Load models
        with open(os.path.join(model_dir, f'{embedding_type}pca_model.pkl'), 'rb') as f_pca:
            pca_model = pickle.load(f_pca)
        with open(os.path.join(model_dir, f'{embedding_type}umap_model.pkl'), 'rb') as f_umap:
            umap_model = pickle.load(f_umap)
        with open(os.path.join(model_dir, f'{embedding_type}kmeans_model.pkl'), 'rb') as f_kmeans:
            kmeans_model = pickle.load(f_kmeans)
        
        # Transform embeddings by loaded models
        pca_features = pca_model.transform(embeddings)
        umap_features = umap_model.transform(embeddings)
        kmeans_labels = kmeans_model.predict(umap_features)
        
    feature_dict = {
        'pca_features': pca_features,
        'umap_features': umap_features,
        'kmeans_cluster': kmeans_labels,
    }
    
    # Convert features to DataFrames
    df_pca = pd.DataFrame(
        feature_dict['pca_features'],
        columns=[f"{embedding_type}pca_feature_dim{i+1}" for i in range(feature_dict['pca_features'].shape[1])]
    )
    
    df_umap = pd.DataFrame(
        feature_dict['umap_features'],
        columns=[f"{embedding_type}umap_feature_dim{i+1}" for i in range(feature_dict['umap_features'].shape[1])]
    )
    
    df_cluster = pd.DataFrame(
        feature_dict['kmeans_cluster'],
        columns=[f'{embedding_type}kmeans_cluster']
    )
    
    # Concatenate all features
    unsupervised_features_df = pd.concat([df_pca, df_umap, df_cluster], axis=1)
    
    # Return models only if training, else None
    if training:
        return unsupervised_features_df, pca_model, umap_model, kmeans_model
    else:
        return unsupervised_features_df, None, None, None