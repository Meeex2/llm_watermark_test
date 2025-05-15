import torch
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, split_num_cat_target, recover_data


def verify_generation_quality(args):
    device = args.device

    # Load original data and model
    train_z, curr_dir, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse = (
        get_input_generate(args)
    )
    in_dim = train_z.shape[1]

    # Load model
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)
    model.load_state_dict(torch.load(f"{ckpt_dir}/model.pt"))
    model.eval()

    # Generate samples
    with torch.no_grad():
        samples = model.sample_ddim(
            args.num_samples, in_dim, num_steps=args.steps, eta=0.0
        )

        # Recover original data format
        syn_num, syn_cat, syn_target = split_num_cat_target(
            samples, info, num_inverse, cat_inverse, device
        )
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        # Load original data for comparison
        original_df = pd.read_csv(f"{dataset_dir}/train.csv")

        # 1. Statistical Distance Metrics
        numerical_metrics = {}
        for col in original_df.select_dtypes(include=[np.number]).columns:
            if col in syn_df.columns:
                wd = wasserstein_distance(original_df[col], syn_df[col])
                mse = mean_squared_error(original_df[col], syn_df[col])
                numerical_metrics[col] = {"wasserstein": wd, "mse": mse}

        # 2. Categorical Distribution Comparison
        categorical_metrics = {}
        for col in original_df.select_dtypes(include=["object", "category"]).columns:
            if col in syn_df.columns:
                orig_dist = original_df[col].value_counts(normalize=True)
                syn_dist = syn_df[col].value_counts(normalize=True)
                # Ensure same categories
                all_cats = set(orig_dist.index) | set(syn_dist.index)
                orig_dist = orig_dist.reindex(all_cats, fill_value=0)
                syn_dist = syn_dist.reindex(all_cats, fill_value=0)
                wd = wasserstein_distance(orig_dist, syn_dist)
                categorical_metrics[col] = {"wasserstein": wd}

        # 3. Verify Noise Recovery
        # Take a subset of generated samples
        test_samples = samples[:10]  # Test with 10 samples
        recovered_noise = []
        for i in range(10):
            t = torch.tensor([500], device=device)  # Start from middle of diffusion
            noise = model.recover_noise(test_samples[i : i + 1], t, num_steps=20)
            recovered_noise.append(noise)

        # 4. Visualize distributions
        plot_distributions(original_df, syn_df, numerical_metrics, categorical_metrics)

        return {
            "numerical_metrics": numerical_metrics,
            "categorical_metrics": categorical_metrics,
            "recovered_noise": recovered_noise,
        }


def plot_distributions(original_df, syn_df, numerical_metrics, categorical_metrics):
    # Plot numerical distributions
    num_cols = list(numerical_metrics.keys())
    n_cols = min(4, len(num_cols))
    n_rows = (len(num_cols) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 4 * n_rows))
    for i, col in enumerate(num_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(original_df[col], alpha=0.5, label="Original", bins=30)
        plt.hist(syn_df[col], alpha=0.5, label="Synthetic", bins=30)
        plt.title(f"{col}\nWD: {numerical_metrics[col]['wasserstein']:.3f}")
        plt.legend()
    plt.tight_layout()
    plt.savefig("numerical_distributions.png")

    # Plot categorical distributions
    cat_cols = list(categorical_metrics.keys())
    n_cols = min(4, len(cat_cols))
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 4 * n_rows))
    for i, col in enumerate(cat_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        orig_dist = original_df[col].value_counts(normalize=True)
        syn_dist = syn_df[col].value_counts(normalize=True)
        plt.bar(range(len(orig_dist)), orig_dist.values, alpha=0.5, label="Original")
        plt.bar(range(len(syn_dist)), syn_dist.values, alpha=0.5, label="Synthetic")
        plt.title(f"{col}\nWD: {categorical_metrics[col]['wasserstein']:.3f}")
        plt.legend()
    plt.tight_layout()
    plt.savefig("categorical_distributions.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify TabSyn generation quality")
    parser.add_argument(
        "--dataname", type=str, default="adult", help="Name of dataset."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to generate."
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of sampling steps."
    )

    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    results = verify_generation_quality(args)

    # Print summary statistics
    print("\nNumerical Features Quality:")
    for col, metrics in results["numerical_metrics"].items():
        print(f"{col}:")
        print(f"  Wasserstein Distance: {metrics['wasserstein']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")

    print("\nCategorical Features Quality:")
    for col, metrics in results["categorical_metrics"].items():
        print(f"{col}:")
        print(f"  Wasserstein Distance: {metrics['wasserstein']:.4f}")

    # Print noise recovery statistics
    print("\nNoise Recovery Statistics:")
    recovered_noise = torch.cat(results["recovered_noise"], dim=0)
    print(f"Mean: {recovered_noise.mean().item():.4f}")
    print(f"Std: {recovered_noise.std().item():.4f}")
    print(f"Min: {recovered_noise.min().item():.4f}")
    print(f"Max: {recovered_noise.max().item():.4f}")
