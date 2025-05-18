import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Tuple, Union, Optional
from tqdm import tqdm


class InverseDiffusion:
    """
    Inverse diffusion process for tabular data.
    
    This class implements methods to recover the latent representation that would
    generate a given synthetic sample, allowing for traceability and interpretability.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """
        Initialize the inverse diffusion process.
        
        Args:
            model: The diffusion model
            num_timesteps: Number of diffusion steps
            beta_start: Starting value for noise schedule
            beta_end: Ending value for noise schedule
            device: Device to use for computations
            verbose: Whether to print verbose logs
        """
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        self.verbose = verbose

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Pre-compute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For numerical stability
        self.eps = 1e-8
        
        if self.verbose:
            print(f"Initialized Inverse Diffusion with {num_timesteps} timesteps")
            print(f"Beta range: [{beta_start}, {beta_end}]")

    def add_noise(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to the input according to the noise schedule.
        
        Args:
            x: Input tensor
            t: Timestep tensor
            
        Returns:
            Tuple of (noisy_input, noise)
        """
        # Generate random noise
        noise = torch.randn_like(x)
        
        # Extract alpha values for the given timestep
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        
        # Add noise according to the diffusion process
        noisy_input = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        
        return noisy_input, noise

    def inverse_diffusion(
        self,
        target_sample: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        use_fixed_point_correction: bool = True,
        fp_correction_steps: int = 100,
        fp_step_size: float = 0.1,
        fp_threshold: float = 1e-3,
    ) -> torch.Tensor:
        """
        Perform inverse diffusion to recover the latent that would generate the target sample.
        
        Args:
            target_sample: The sample to invert
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for classifier-free guidance
            eta: Parameter controlling the stochasticity
            use_fixed_point_correction: Whether to use fixed-point correction
            fp_correction_steps: Number of fixed-point correction steps
            fp_step_size: Step size for fixed-point correction
            fp_threshold: Threshold for early stopping in fixed-point correction
            
        Returns:
            Recovered latent tensor
        """
        # Start with random noise
        latents = torch.randn_like(target_sample).to(self.device)
        
        # Set up timesteps
        timesteps = torch.linspace(0, self.num_timesteps - 1, num_inference_steps, dtype=torch.long, device=self.device)
        timesteps = timesteps.flip(0)  # Reverse for inverse diffusion
        
        # Classifier-free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Inverse diffusion process
        for i, t in enumerate(tqdm(timesteps, desc="Inverse Diffusion") if self.verbose else timesteps):
            # Get previous timestep
            prev_t = t - 1 if i < len(timesteps) - 1 else torch.tensor([0], device=self.device)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor([1.0], device=self.device)
            
            # Prepare model input
            model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(model_input, t.expand(model_input.shape[0]))
                
                # Apply guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Calculate coefficients
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            sqrt_alpha_prev = torch.sqrt(alpha_prev)
            sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
            
            # DDIM inversion formula
            pred_x0 = (latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            # Calculate direction
            direction = (sqrt_one_minus_alpha_prev - sqrt_one_minus_alpha_t * sqrt_alpha_prev / sqrt_alpha_t) * noise_pred
            
            # Update latents
            latents = sqrt_alpha_prev * pred_x0 + direction
            
            # Apply fixed-point correction if enabled
            if use_fixed_point_correction:
                latents = self.fixed_point_correction(
                    latents, t, prev_t, target_sample, 
                    n_iter=fp_correction_steps, 
                    step_size=fp_step_size,
                    th=fp_threshold,
                    guidance_scale=guidance_scale
                )
        
        return latents

    def fixed_point_correction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prev_t: torch.Tensor,
        target: torch.Tensor,
        n_iter: int = 100,
        step_size: float = 0.1,
        th: float = 1e-3,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply fixed-point correction to improve the inverse diffusion result.
        
        Args:
            x: Current latent tensor
            t: Current timestep
            prev_t: Previous timestep
            target: Target sample
            n_iter: Number of correction iterations
            step_size: Step size for correction
            th: Threshold for early stopping
            guidance_scale: Guidance scale for classifier-free guidance
            
        Returns:
            Corrected latent tensor
        """
        # Make a copy of the input
        input_latent = x.clone()
        original_step_size = step_size
        
        # Get alpha values
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor([1.0], device=self.device)
        
        # Classifier-free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Step size scheduler
        step_scheduler = StepScheduler(current_lr=step_size, factor=0.5, patience=20)
        
        # Fixed-point iteration
        for i in range(n_iter):
            # Step size warmup
            if i < 20:  # Warmup period
                step_size = original_step_size * (i + 1) / 20
            
            # Prepare model input
            model_input = torch.cat([input_latent] * 2) if do_classifier_free_guidance else input_latent
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(model_input, t.expand(model_input.shape[0]))
                
                # Apply guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Calculate coefficients
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            sqrt_alpha_prev = torch.sqrt(alpha_prev)
            sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
            
            # Predict x0
            pred_x0 = (input_latent - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            # Forward process to get x_t
            direction = (sqrt_one_minus_alpha_prev - sqrt_one_minus_alpha_t * sqrt_alpha_prev / sqrt_alpha_t) * noise_pred
            pred_target = sqrt_alpha_prev * pred_x0 + direction
            
            # Calculate loss
            loss = F.mse_loss(pred_target, target, reduction='sum')
            
            # Early stopping
            if loss.item() < th:
                break
            
            # Update latent
            input_latent = input_latent - step_size * (pred_target - target)
            
            # Update step size
            step_size = step_scheduler.step(loss)
        
        return input_latent

    def recover_latent(
        self,
        target_sample: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        use_fixed_point_correction: bool = True,
        fp_correction_steps: int = 100,
        fp_step_size: float = 0.1,
        fp_threshold: float = 1e-3,
    ) -> torch.Tensor:
        """
        Recover the latent representation that would generate the target sample.
        
        Args:
            target_sample: The sample to recover the latent for
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for classifier-free guidance
            eta: Parameter controlling the stochasticity
            use_fixed_point_correction: Whether to use fixed-point correction
            fp_correction_steps: Number of fixed-point correction steps
            fp_step_size: Step size for fixed-point correction
            fp_threshold: Threshold for early stopping in fixed-point correction
            
        Returns:
            Recovered latent tensor
        """
        return self.inverse_diffusion(
            target_sample,
            num_inference_steps,
            guidance_scale,
            eta,
            use_fixed_point_correction,
            fp_correction_steps,
            fp_step_size,
            fp_threshold,
        )


class StepScheduler:
    """
    Step size scheduler for optimization processes.
    
    Reduces the step size when the loss doesn't improve for a certain number of iterations.
    """
    
    def __init__(
        self,
        current_lr: float = 0.1,
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-6,
        verbose: bool = False,
    ):
        """
        Initialize the step scheduler.
        
        Args:
            current_lr: Initial learning rate
            factor: Factor by which to reduce the learning rate
            patience: Number of iterations with no improvement after which to reduce the rate
            min_lr: Minimum learning rate
            verbose: Whether to print when learning rate is reduced
        """
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        if current_lr <= 0:
            raise ValueError('Learning rate should be > 0.')
        
        self.current_lr = current_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.bad_epochs = 0
        self.last_epoch = 0
    
    def step(self, loss: torch.Tensor) -> float:
        """
        Update the learning rate based on the loss.
        
        Args:
            loss: Current loss value
            
        Returns:
            Updated learning rate
        """
        current_loss = loss.item()
        self.last_epoch += 1
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        
        if self.bad_epochs > self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.bad_epochs = 0
            if self.verbose:
                print(f'Epoch {self.last_epoch}: reducing learning rate to {self.current_lr:.6f}')
        
        return self.current_lr
