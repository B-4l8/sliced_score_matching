import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import argparse
import sys

def visualize_synthetic():
    parser = argparse.ArgumentParser(description='Visualize Synthetic 500D Data Generation using PCA and t-SNE')
    parser.add_argument('--doc', type=str, default='synthetic_500__exp01', help='Experiment documentation string')
    parser.add_argument('--dim', type=int, default=500, help='Dimension of the synthetic data')
    parser.add_argument('--run', type=str, default='run', help='Run directory')
    args = parser.parse_args()

    results_dir = os.path.join(args.run, 'results', args.doc)
    generated_csv_path = os.path.join(results_dir, f'generated_{args.doc}.csv')
    original_csv_path = f'data/{args.dim}d/synthetic_pos.csv'
    output_path = os.path.join(results_dir, 'pca_tsne_comparison_synthetic.png')

    if not os.path.exists(generated_csv_path):
        print(f"Error: {generated_csv_path} not found.")
        return

    print(f"Loading generated data from {generated_csv_path}")
    try:
        generated_data = np.loadtxt(generated_csv_path, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"Error loading generated data: {e}")
        return

    if not os.path.exists(original_csv_path):
        print(f"Error: {original_csv_path} not found.")
        return

    print(f"Loading original data from {original_csv_path}")
    try:
        original_data = np.loadtxt(original_csv_path, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"Error loading original data: {e}")
        return

    n_gen, dim_gen = generated_data.shape
    n_orig, dim_orig = original_data.shape

    print(f"Generated data shape: {generated_data.shape}")
    print(f"Original data shape: {original_data.shape}")

    if dim_gen != dim_orig:
        print(f"Error: Dimension mismatch. Generated: {dim_gen}, Original: {dim_orig}")
        return

    all_data = np.vstack([original_data, generated_data])

    print("Running PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_data)

    orig_pca = pca_result[:n_orig]
    gen_pca = pca_result[n_orig:]

    print("Running t-SNE...")

    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(all_data)

    orig_tsne = tsne_result[:n_orig]
    gen_tsne = tsne_result[n_orig:]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    axes[0].scatter(orig_pca[:, 0], orig_pca[:, 1], alpha=0.3, label='Original Synthetic Samples', color='blue', s=10)
    axes[0].scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.5, label='Generated Samples (DKEF)', color='red', s=10)
    axes[0].set_title(f'PCA Visualization (Dim={dim_gen})')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(orig_tsne[:, 0], orig_tsne[:, 1], alpha=0.3, label='Original Synthetic Samples', color='blue', s=10)
    axes[1].scatter(gen_tsne[:, 0], gen_tsne[:, 1], alpha=0.5, label='Generated Samples (DKEF)', color='red', s=10)
    axes[1].set_title(f't-SNE Visualization (Dim={dim_gen})')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    visualize_synthetic()
