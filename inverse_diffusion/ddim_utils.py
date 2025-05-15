import torch
import numpy as np
from typing import Optional, Union, Tuple


class DDIMSampler:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda",
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to the input according to the noise schedule."""
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise, noise

    def remove_noise(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, eta: float = 0.0
    ) -> torch.Tensor:
        """Remove noise using DDIM sampling."""
        # Ensure t is a tensor of the right shape
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        t = t.view(-1)  # Make sure t is 1D

        # Get alpha values with proper broadcasting
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        alpha_t_prev = torch.where(
            t > 0, self.alphas_cumprod[t - 1].view(-1, 1), torch.ones_like(alpha_t)
        )

        # Predict noise
        pred_noise = model(x, t)

        # DDIM sampling with proper broadcasting
        sigma_t = (
            eta
            * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t))
            * torch.sqrt(1 - alpha_t / alpha_t_prev)
        )

        # Mean prediction
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

        # Direction pointing to x_t (ensure broadcasting)
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * pred_noise

        # Final prediction
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        if eta > 0:
            noise = torch.randn_like(x)
            x_prev = x_prev + sigma_t * noise

        return x_prev


class DPMSolver:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda",
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def recover_noise(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        num_steps: int = 20,
    ) -> torch.Tensor:
        """Recover noise from a noisy image using DPM-Solver."""
        x_t = x.clone()
        t_steps = torch.linspace(t, 0, num_steps, device=self.device)

        for i in range(num_steps - 1):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]

            # Predict noise
            pred_noise = model(x_t, t_cur)

            # Update x_t using DPM-Solver
            alpha_t = self.alphas_cumprod[t_cur.long()]
            alpha_t_next = self.alphas_cumprod[t_next.long()]

            # DPM-Solver update
            x_t = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x_t = (
                torch.sqrt(alpha_t_next) * x_t
                + torch.sqrt(1 - alpha_t_next) * pred_noise
            )

        return x_t
