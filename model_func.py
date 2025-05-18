def recover_noise(self, x, t, num_steps=20):
    """
    Recover noise from a sample using inverse diffusion.
    
    Args:
        x: Input tensor
        t: Starting timestep
        num_steps: Number of solver steps
        
    Returns:
        Recovered noise tensor
    """
    # Import InverseDiffusion here to avoid circular imports
    from tabsyn.inverse_diffusion import InverseDiffusion
    
    # Use the inverse diffusion process
    inverse_diffusion = InverseDiffusion(
        model=self.denoise_fn,
        num_timesteps=self.num_timesteps,
        beta_start=self.betas[0].item(),
        beta_end=self.betas[-1].item(),
        device=self.device,
        verbose=False,
    )
    
    # Convert t to a tensor if it's not already
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], device=self.device)
    
    # Ensure x is on the correct device
    x = x.to(self.device)
    
    # Recover the latent that would generate x at timestep t
    recovered_latent = inverse_diffusion.recover_latent(
        target_sample=x,
        num_inference_steps=num_steps,
        guidance_scale=1.0,  # No guidance for tabular data
        use_fixed_point_correction=True,
        fp_correction_steps=50,
        fp_step_size=0.05,
        fp_threshold=1e-4,
    )
    
    return recovered_latent
