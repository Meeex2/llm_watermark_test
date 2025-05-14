import os
import torch
import argparse
import warnings
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import (
    get_input_train,
    get_input_generate,
    split_num_cat_target,
    recover_data,
)

warnings.filterwarnings("ignore")


def train(args):
    device = args.device
    train_z, _, _, ckpt_path, _ = get_input_train(args)

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1]
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2
    train_data = train_z

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    num_epochs = 10000 + 1
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=20, verbose=True
    )

    model.train()
    best_loss = float("inf")
    patience = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f"{ckpt_path}/model.pt")
        else:
            patience += 1
            if patience == 500:
                print("Early stopping")
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f"{ckpt_path}/model_{epoch}.pt")

    end_time = time.time()
    print("Training Time: ", end_time - start_time)


def generate_samples(args):
    device = args.device
    train_z, curr_dir, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse = (
        get_input_generate(args)
    )

    in_dim = train_z.shape[1]
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    # Load trained model
    model.load_state_dict(torch.load(f"{ckpt_dir}/model.pt"))
    model.eval()

    # Generate samples using DDIM
    num_samples = args.num_samples
    samples = model.sample_ddim(num_samples, in_dim, num_steps=50, eta=0.0)

    # Recover original data format
    syn_num, syn_cat, syn_target = split_num_cat_target(
        samples, info, num_inverse, cat_inverse, device
    )
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    # Save generated samples
    output_path = f"{dataset_dir}/synthetic_data.csv"
    syn_df.to_csv(output_path, index=False)
    print(f"Generated samples saved to {output_path}")


def recover_noise_from_image(args):
    device = args.device
    train_z, curr_dir, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse = (
        get_input_generate(args)
    )

    in_dim = train_z.shape[1]
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    # Load trained model
    model.load_state_dict(torch.load(f"{ckpt_dir}/model.pt"))
    model.eval()

    # Load image and convert to latent space
    image_path = args.image_path
    # Add your image loading and preprocessing code here
    # This is a placeholder for the actual image loading code
    image = torch.randn(1, in_dim, device=device)  # Replace with actual image loading

    # Recover noise using DPM-Solver
    t = torch.tensor([500], device=device)  # Start from middle of the diffusion process
    recovered_noise = model.recover_noise(image, t, num_steps=20)

    print("Noise recovered successfully")
    return recovered_noise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and Generation of TabSyn")
    parser.add_argument(
        "--dataname", type=str, default="adult", help="Name of dataset."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "generate", "recover_noise"],
        help="Mode to run the script in.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to generate."
    )
    parser.add_argument(
        "--image_path", type=str, help="Path to image for noise recovery."
    )

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        generate_samples(args)
    elif args.mode == "recover_noise":
        recover_noise_from_image(args)
