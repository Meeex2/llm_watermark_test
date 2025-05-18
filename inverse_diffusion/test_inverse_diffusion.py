import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, split_num_cat_target, recover_data


def test_inverse_diffusion(args):
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

    # Generate a sample
    with torch.no_grad():
        # Start with random noise
        original_noise = torch.randn(1, in_dim).to(device)
        
        print(f"Original noise shape: {original_noise.shape}")
        print(f"Original noise mean: {original_noise.mean().item():.4f}, std: {original_noise.std().item():.4f}")
        
        # Generate a sample using DDIM
        print("Generating sample from original noise...")
        sample = model.sample_ddim(
            1, in_dim, num_steps=args.steps, eta=0.0, noise=original_noise
        )
        
        print(f"Generated sample shape: {sample.shape}")
        print(f"Generated sample mean: {sample.mean().item():.4f}, std: {sample.std().item():.4f}")
        
        # Try to recover the original noise
        print("Recovering noise from sample...")
        t = torch.tensor([500], device=device)  # Start from middle of diffusion
        recovered_noise = model.recover_noise(sample, t, num_steps=20)
        
        print(f"Recovered noise shape: {recovered_noise.shape}")
        print(f"Recovered noise mean: {recovered_noise.mean().item():.4f}, std: {recovered_noise.std().item():.4f}")
        
        # Calculate similarity between original and recovered noise
        cosine_similarity = torch.nn.functional.cosine_similarity(
            original_noise.flatten(), recovered_noise.flatten(), dim=0
        )
        mse = torch.nn.functional.mse_loss(original_noise, recovered_noise)
        
        print(f"Cosine similarity between original and recovered noise: {cosine_similarity.item():.4f}")
        print(f"Mean squared error between original and recovered noise: {mse.item():.4f}")
        
        # Generate samples from both noises to compare
        print("Generating sample from recovered noise...")
        sample_from_recovered = model.sample_ddim(
            1, in_dim, num_steps=args.steps, eta=0.0, noise=recovered_noise
        )
        
        # Compare the generated samples
        sample_similarity = torch.nn.functional.cosine_similarity(
            sample.flatten(), sample_from_recovered.flatten(), dim=0
        )
        sample_mse = torch.nn.functional.mse_loss(sample, sample_from_recovered)
        
        print(f"Cosine similarity between original and recovered samples: {sample_similarity.item():.4f}")
        print(f"Mean squared error between original and recovered samples: {sample_mse.item():.4f}")
        
        # Convert to tabular data for visual comparison
        syn_num_orig, syn_cat_orig, syn_target_orig = split_num_cat_target(
            sample, info, num_inverse, cat_inverse, device
        )
        syn_df_orig = recover_data(syn_num_orig, syn_cat_orig, syn_target_orig, info)
        
        syn_num_rec, syn_cat_rec, syn_target_rec = split_num_cat_target(
            sample_from_recovered, info, num_inverse, cat_inverse, device
        )
        syn_df_rec = recover_data(syn_num_rec, syn_cat_rec, syn_target_rec, info)
        
        print("\nSample from original noise:")
        print(syn_df_orig.head())
        
        print("\nSample from recovered noise:")
        print(syn_df_rec.head())
        
        # Visualize the noise distributions
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(original_noise.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Original Noise')
        plt.hist(recovered_noise.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Recovered Noise')
        plt.title('Noise Distribution Comparison')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(original_noise.cpu().numpy().flatten(), recovered_noise.cpu().numpy().flatten(), alpha=0.5)
        plt.title(f'Original vs Recovered Noise (Cosine Sim: {cosine_similarity.item():.4f})')
        plt.xlabel('Original Noise')
        plt.ylabel('Recovered Noise')
        
        plt.tight_layout()
        plt.savefig('noise_comparison.png')
        print("Saved noise comparison plot to 'noise_comparison.png'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test inverse diffusion")
    parser.add_argument(
        "--dataname", type=str, default="adult", help="Name of dataset."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of sampling steps."
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate."
    )

    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    test_inverse_diffusion(args)
