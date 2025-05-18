import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, split_num_cat_target, recover_data
from tabsyn.inverse_diffusion import InverseDiffusion


def verify_inverse_quality(args):
    """
    Verify the quality of the inverse diffusion process by:
    1. Generating synthetic data from random noise
    2. Recovering the noise using inverse diffusion
    3. Generating new synthetic data from the recovered noise
    4. Comparing the original and regenerated synthetic data
    
    Args:
        args: Command line arguments
    """
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

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Generate synthetic data
    print("Generating synthetic data...")
    with torch.no_grad():
        # Generate random noise
        original_noises = torch.randn(args.num_samples, in_dim).to(device)
        
        # Generate samples using DDIM
        samples = model.sample_ddim(
            args.num_samples, in_dim, num_steps=args.steps, eta=0.0, noise=original_noises
        )
        
        # Convert to tabular data
        syn_num, syn_cat, syn_target = split_num_cat_target(
            samples, info, num_inverse, cat_inverse, device
        )
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)
        
        print(f"Generated {len(syn_df)} synthetic samples")
        print(syn_df.head())
    
    # Recover noise using inverse diffusion
    print("Recovering noise using inverse diffusion...")
    t = torch.tensor([500], device=device)  # Start from middle of diffusion
    recovered_noises = []
    
    for i in range(0, len(samples), args.batch_size):
        batch_samples = samples[i:i+args.batch_size]
        batch_recovered = model.recover_noise(batch_samples, t, num_steps=args.inv_steps)
        recovered_noises.append(batch_recovered)
    
    recovered_noises = torch.cat(recovered_noises, dim=0)
    print(f"Recovered {len(recovered_noises)} noise vectors")
    
    # Calculate similarity between original and recovered noise
    cosine_similarities = []
    mse_values = []
    
    for i in range(len(original_noises)):
        cos_sim = torch.nn.functional.cosine_similarity(
            original_noises[i].flatten(), recovered_noises[i].flatten(), dim=0
        )
        mse = torch.nn.functional.mse_loss(original_noises[i], recovered_noises[i])
        
        cosine_similarities.append(cos_sim.item())
        mse_values.append(mse.item())
    
    avg_cos_sim = np.mean(cosine_similarities)
    avg_mse = np.mean(mse_values)
    
    print(f"Average cosine similarity: {avg_cos_sim:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    
    # Generate new samples from recovered noise
    print("Generating new samples from recovered noise...")
    with torch.no_grad():
        regenerated_samples = model.sample_ddim(
            len(recovered_noises), in_dim, num_steps=args.steps, eta=0.0, noise=recovered_noises
        )
        
        # Convert to tabular data
        regen_num, regen_cat, regen_target = split_num_cat_target(
            regenerated_samples, info, num_inverse, cat_inverse, device
        )
        regen_df = recover_data(regen_num, regen_cat, regen_target, info)
        
        print(f"Generated {len(regen_df)} regenerated samples")
        print(regen_df.head())
    
    # Compare original and regenerated samples
    sample_similarities = []
    sample_mse_values = []
    
    for i in range(len(samples)):
        sample_cos_sim = torch.nn.functional.cosine_similarity(
            samples[i].flatten(), regenerated_samples[i].flatten(), dim=0
        )
        sample_mse = torch.nn.functional.mse_loss(samples[i], regenerated_samples[i])
        
        sample_similarities.append(sample_cos_sim.item())
        sample_mse_values.append(sample_mse.item())
    
    avg_sample_cos_sim = np.mean(sample_similarities)
    avg_sample_mse = np.mean(sample_mse_values)
    
    print(f"Average sample cosine similarity: {avg_sample_cos_sim:.4f}")
    print(f"Average sample MSE: {avg_sample_mse:.4f}")
    
    # Compare tabular data distributions
    print("Comparing tabular data distributions...")
    
    # For numerical columns
    num_cols = syn_df.select_dtypes(include=[np.number]).columns
    num_metrics = {}
    
    for col in num_cols:
        # Calculate Wasserstein distance
        from scipy.stats import wasserstein_distance
        wd = wasserstein_distance(syn_df[col].dropna(), regen_df[col].dropna())
        
        # Calculate correlation
        corr = np.corrcoef(syn_df[col].dropna(), regen_df[col].dropna())[0, 1]
        
        num_metrics[col] = {"wasserstein": wd, "correlation": corr}
    
    # For categorical columns
    cat_cols = syn_df.select_dtypes(include=['object', 'category']).columns
    cat_metrics = {}
    
    for col in cat_cols:
        orig_dist = syn_df[col].value_counts(normalize=True)
        regen_dist = regen_df[col].value_counts(normalize=True)
        
        # Ensure same categories
        all_cats = set(orig_dist.index) | set(regen_dist.index)
        orig_dist = orig_dist.reindex(all_cats, fill_value=0)
        regen_dist = regen_dist.reindex(all_cats, fill_value=0)
        
        # Calculate Jensen-Shannon divergence
        from scipy.spatial.distance import jensenshannon
        js_div = jensenshannon(orig_dist, regen_dist)
        
        cat_metrics[col] = {"js_divergence": js_div}
    
    # Print metrics
    print("\nNumerical Features Comparison:")
    for col, metrics in num_metrics.items():
        print(f"{col}:")
        print(f"  Wasserstein Distance: {metrics['wasserstein']:.4f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
    
    print("\nCategorical Features Comparison:")
    for col, metrics in cat_metrics.items():
        print(f"{col}:")
        print(f"  Jensen-Shannon Divergence: {metrics['js_divergence']:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot noise similarity distribution
    plt.subplot(2, 2, 1)
    plt.hist(cosine_similarities, bins=20)
    plt.title(f'Noise Cosine Similarity (Avg: {avg_cos_sim:.4f})')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    
    # Plot sample similarity distribution
    plt.subplot(2, 2, 2)
    plt.hist(sample_similarities, bins=20)
    plt.title(f'Sample Cosine Similarity (Avg: {avg_sample_cos_sim:.4f})')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    
    # Plot numerical feature correlation
    plt.subplot(2, 2, 3)
    correlations = [metrics['correlation'] for metrics in num_metrics.values()]
    plt.bar(range(len(correlations)), correlations)
    plt.xticks(range(len(correlations)), num_metrics.keys(), rotation=45)
    plt.title('Numerical Feature Correlation')
    plt.ylabel('Correlation')
    plt.ylim(0, 1)
    
    # Plot categorical feature JS divergence
    plt.subplot(2, 2, 4)
    js_divs = [metrics['js_divergence'] for metrics in cat_metrics.values()]
    plt.bar(range(len(js_divs)), js_divs)
    plt.xticks(range(len(js_divs)), cat_metrics.keys(), rotation=45)
    plt.title('Categorical Feature JS Divergence')
    plt.ylabel('JS Divergence')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('inverse_diffusion_quality.png')
    print("Saved visualization to 'inverse_diffusion_quality.png'")
    
    # Save results
    results = {
        "noise_similarity": {
            "avg_cosine_similarity": avg_cos_sim,
            "avg_mse": avg_mse,
        },
        "sample_similarity": {
            "avg_cosine_similarity": avg_sample_cos_sim,
            "avg_mse": avg_sample_mse,
        },
        "numerical_metrics": num_metrics,
        "categorical_metrics": cat_metrics,
    }
    
    import json
    with open('inverse_diffusion_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Saved results to 'inverse_diffusion_results.json'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify inverse diffusion quality")
    parser.add_argument(
        "--dataname", type=str, default="adult", help="Name of dataset."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of sampling steps."
    )
    parser.add_argument(
        "--inv_steps", type=int, default=20, help="Number of inverse diffusion steps."
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate."
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for processing."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    verify_inverse_quality(args)
