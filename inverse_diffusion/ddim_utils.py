import torch
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm


def extract(v, t, x_shape):
    """
    Extract the appropriate timestep value for each sample in a batch.
    
    Args:
        v: Tensor of shape (T,) containing values for each timestep
        t: Tensor of shape (batch_size,) containing timestep indices
        x_shape: Shape of the input tensor
        
    Returns:
        Tensor of shape (batch_size, 1, 1, ...) for broadcasting
    """
    out = torch.gather(v, index=t, dim=0)
    out = out.to(device=t.device, dtype=torch.float32)
    
    # Reshape for broadcasting
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def ddim_sample(
    model,
    x,
    t,
    t_next,
    alphas_cumprod,
    eta=0.0,
    temperature=1.0,
    noise=None,
):
    """
    DDIM sampling step.
    
    Args:
        model: Noise prediction model
        x: Current latent tensor
        t: Current timestep
        t_next: Next timestep
        alphas_cumprod: Cumulative product of alphas
        eta: Parameter controlling the stochasticity (0 = deterministic, 1 = full stochastic)
        temperature: Temperature for sampling
        noise: Optional noise tensor
        
    Returns:
        Next latent tensor
    """
    # Get alpha values
    alpha_t = alphas_cumprod[t]
    alpha_next = alphas_cumprod[t_next]
    
    # Predict noise
    noise_pred = model(x, t)
    
    # Apply temperature
    if temperature != 1.0:
        noise_pred = noise_pred / temperature
    
    # Predict x0
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    pred_x0 = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
    
    # Direction pointing to xt
    direction = torch.sqrt(1 - alpha_next) * noise_pred
    
    # Random noise for stochasticity
    if eta > 0:
        if noise is None:
            noise = torch.randn_like(x)
        
        # Calculate sigma_t
        sigma_t = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
        
        # Add noise
        direction = direction + sigma_t * noise
    
    # Update x
    x_next = torch.sqrt(alpha_next) * pred_x0 + direction
    
    return x_next


def ddim_sample_loop(
    model,
    shape,
    num_timesteps,
    alphas_cumprod,
    eta=0.0,
    temperature=1.0,
    noise=None,
    device="cuda",
    progress=True,
):
    """
    DDIM sampling loop.
    
    Args:
        model: Noise prediction model
        shape: Shape of the output tensor
        num_timesteps: Number of timesteps
        alphas_cumprod: Cumulative product of alphas
        eta: Parameter controlling the stochasticity (0 = deterministic, 1 = full stochastic)
        temperature: Temperature for sampling
        noise: Optional noise tensor
        device: Device to use
        progress: Whether to show progress bar
        
    Returns:
        Generated sample
    """
    # Start with random noise
    if noise is None:
        x = torch.randn(shape, device=device)
    else:
        x = noise
    
    # Set up timesteps
    timesteps = torch.linspace(0, num_timesteps - 1, num_timesteps, dtype=torch.long, device=device)
    timesteps = timesteps.flip(0)  # Reverse for sampling
    
    # Sampling loop
    iterator = tqdm(timesteps, desc="DDIM Sampling") if progress else timesteps
    for i, t in enumerate(iterator):
        # Get next timestep
        t_next = t - 1 if i < len(timesteps) - 1 else torch.tensor([-1], device=device)
        
        # DDIM step
        x = ddim_sample(
            model=model,
            x=x,
            t=t,
            t_next=t_next,
            alphas_cumprod=alphas_cumprod,
            eta=eta,
            temperature=temperature,
            noise=None,  # Use new noise for each step
        )
    
    return x


def ddim_inversion(
    model,
    x,
    num_timesteps,
    alphas_cumprod,
    eta=0.0,
    temperature=1.0,
    device="cuda",
    progress=True,
):
    """
    DDIM inversion process.
    
    Args:
        model: Noise prediction model
        x: Target sample
        num_timesteps: Number of timesteps
        alphas_cumprod: Cumulative product of alphas
        eta: Parameter controlling the stochasticity (0 = deterministic, 1 = full stochastic)
        temperature: Temperature for sampling
        device: Device to use
        progress: Whether to show progress bar
        
    Returns:
        Recovered latent
    """
    # Set up timesteps
    timesteps = torch.linspace(0, num_timesteps - 1, num_timesteps, dtype=torch.long, device=device)
    
    # Inversion loop
    iterator = tqdm(timesteps, desc="DDIM Inversion") if progress else timesteps
    for i, t in enumerate(iterator):
        # Get next timestep
        t_next = t + 1 if i < len(timesteps) - 1 else torch.tensor([num_timesteps], device=device)
        
        # Get alpha values
        alpha_t = alphas_cumprod[t]
        alpha_next = alphas_cumprod[t_next] if t_next < num_timesteps else torch.tensor([0.0], device=device)
        
        # Predict noise
        noise_pred = model(x, t)
        
        # Apply temperature
        if temperature != 1.0:
            noise_pred = noise_pred / temperature
        
        # Predict x0
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        pred_x0 = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # Direction pointing to xt+1
        direction = torch.sqrt(1 - alpha_next) * noise_pred
        
        # Update x
        x = torch.sqrt(alpha_next) * pred_x0 + direction
    
    return x
