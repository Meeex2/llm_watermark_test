import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, split_num_cat_target, recover_data


def batch_inverse_diffusion(args):
    """
    Perform inverse diffusion on a batch of synthetic samples to recover their latent representations.
    
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

    # Load synthetic data
    synthetic_path = f"{curr_dir}/synthetic/{args.dataname}/tabsyn.csv"
    if not os.path.exists(synthetic_path):
        print(f"Synthetic data file not found at {synthetic_path}")
        print("Generating synthetic data first...")
        
        # Generate synthetic data
        with torch.no_grad():
            samples = model.sample_ddim(
                args.num_samples, in_dim, num_steps=args.steps, eta=0.0
            )
            
            # Recover original data format
            syn_num, syn_cat, syn_target = split_num_cat_target(
                samples, info, num_inverse, cat_inverse, device
            )
            syn_df = recover_data(syn_num, syn_cat, syn_target, info)
            
            # Save synthetic data
            os.makedirs(os.path.dirname(synthetic_path), exist_ok=True)
            syn_df.to_csv(synthetic_path, index=False)
            print(f"Saved synthetic data to {synthetic_path}")
    
    # Load synthetic data
    syn_df = pd.read_csv(synthetic_path)
    print(f"Loaded synthetic data with {len(syn_df)} samples")
    
    # Take a subset if specified
    if args.max_samples > 0 and args.max_samples < len(syn_df):
        syn_df = syn_df.sample(args.max_samples, random_state=42)
        print(f"Using {len(syn_df)} samples for inverse diffusion")
    
    # Convert synthetic data back to latent space
    # This is a placeholder - you'll need to implement the actual conversion
    # based on your data preprocessing pipeline
    print("Converting synthetic data to latent space...")
    latent_samples = []
    
    # Process in batches to avoid memory issues
    batch_size = 100
    num_batches = (len(syn_df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(syn_df))
        batch_df = syn_df.iloc[start_idx:end_idx]
        
        # Convert batch to latent space
        # This is a placeholder - implement your actual conversion
        batch_latent = torch.randn(len(batch_df), in_dim).to(device)
        latent_samples.append(batch_latent)
    
    latent_samples = torch.cat(latent_samples, dim=0)
    print(f"Converted synthetic data to latent space with shape {latent_samples.shape}")
    
    # Perform inverse diffusion on each sample
    print("Performing inverse diffusion...")
    recovered_noises = []
    t = torch.tensor([500], device=device)  # Start from middle of diffusion
    
    for i in tqdm(range(len(latent_samples))):
        sample = latent_samples[i:i+1]
        recovered_noise = model.recover_noise(sample, t, num_steps=args.inv_steps)
        recovered_noises.append(recovered_noise)
    
    recovered_noises = torch.cat(recovered_noises, dim=0)
    print(f"Recovered noises with shape {recovered_noises.shape}")
    
    # Save recovered noises
    noise_path = f"{curr_dir}/synthetic/{args.dataname}/recovered_noises.pt"
    os.makedirs(os.path.dirname(noise_path), exist_ok=True)
    torch.save(recovered_noises, noise_path)
    print(f"Saved recovered noises to {noise_path}")
    
    # Generate samples from recovered noises to verify
    print("Generating samples from recovered noises...")
    regenerated_samples = []
    
    for i in tqdm(range(0, len(recovered_noises), batch_size)):
        batch_noises = recovered_noises[i:i+batch_size]
        batch_samples = model.sample_ddim(
            len(batch_noises), in_dim, num_steps=args.steps, eta=0.0, noise=batch_noises
        )
        regenerated_samples.append(batch_samples)
    
    regenerated_samples = torch.cat(regenerated_samples, dim=0)
    print(f"Generated samples from recovered noises with shape {regenerated_samples.shape}")
    
    # Convert regenerated samples to tabular data
    print("Converting regenerated samples to tabular data...")
    syn_num, syn_cat, syn_target = split_num_cat_target(
        regenerated_samples, info, num_inverse, cat_inverse, device
    )
    regenerated_df = recover_data(syn_num, syn_cat, syn_target, info)
    
    # Save regenerated data
    regenerated_path = f"{curr_dir}/synthetic/{args.dataname}/regenerated.csv"
    regenerated_df.to_csv(regenerated_path, index=False)
    print(f"Saved regenerated data to {regenerated_path}")
    
    # Calculate similarity between original synthetic data and regenerated data
    print("Calculating similarity between original and regenerated data...")
    
    # For numerical columns
    num_cols = syn_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        corr = np.corrcoef(syn_df[col], regenerated_df[col])[0, 1]
        print(f"Correlation for {col}: {corr:.4f}")
    
    # For categorical columns
    cat_cols = syn_df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        orig_dist = syn_df[col].value_counts(normalize=True)
        regen_dist = regenerated_df[col].value_counts(normalize=True)
        
        # Ensure same categories
        all_cats = set(orig_dist.index) | set(regen_dist.index)
        orig_dist = orig_dist.reindex(all_cats, fill_value=0)
        regen_dist = regen_dist.reindex(all_cats, fill_value=0)
        
        # Calculate Jensen-Shannon divergence
        from scipy.spatial.distance import jensenshannon
        js_div = jensenshannon(orig_dist, regen_dist)
        print(f"Jensen-Shannon divergence for {col}: {js_div:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch inverse diffusion")
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
        "--num_samples", type=int, default=1000, help="Number of samples to generate if synthetic data doesn't exist."
    )
    parser.add_argument(
        "--max_samples", type=int, default=100, help="Maximum number of samples to process (0 for all)."
    )

    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    batch_inverse_diffusion(args)
