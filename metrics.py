"""
Metrics for trajectory prediction evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class MSELoss(nn.Module):
    """Custom MSE loss that handles missing values."""
    
    def __init__(self, config: dict):
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss, ignoring zero-padded values.
        
        Args:
            predictions: [batch_size, prediction_horizon, 3]
            targets: [batch_size, prediction_horizon, 6]
        
        Returns:
            Loss value
        """
        # Use only first 3 columns of targets to match predictions
        targets_subset = targets[:, :, 3:6]
        predictions = predictions.squeeze(-2)
        
        # Create mask for non-zero targets (non-padded values)
        mask = (targets_subset != 0).any(dim=-1).float()  # [batch_size, prediction_horizon]

        # Compute MSE
        mse = (predictions - targets_subset) ** 2
        
        # Apply mask and compute mean
        masked_mse = mse * mask.unsqueeze(-1)  # [batch_size, prediction_horizon, 3]
        
        # Sum over all dimensions and divide by number of non-zero elements
        total_elements = mask.sum()
        if total_elements > 0:
            return masked_mse.sum() / total_elements
        else:
            return torch.tensor(0.0, device=predictions.device)

class CosineSimilarityLoss(nn.Module):
    """Cosine similarity loss for angle predictions."""
    
    def __init__(self, config: dict, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.config = config
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity loss.
        
        Args:
            predictions: [batch_size, prediction_horizon, 3] - [speed, cos_theta, sin_theta]
            targets: [batch_size, prediction_horizon, 3] - [speed, cos_theta, sin_theta]
        
        Returns:
            Loss value
        """
        predictions = predictions.squeeze(-2)

        # Extract speed and angle components
        pred_speed = predictions[:, :, 0]
        pred_c = predictions[:, :, 1]  # cos_theta
        pred_s = predictions[:, :, 2]  # sin_theta
        
        true_speed = targets[:, :, 0]
        true_c = targets[:, :, 1]  # cos_theta
        true_s = targets[:, :, 2]  # sin_theta

        # --- normalize predicted angle vector (safe & stable) ---
        pred_vec = torch.stack([pred_c, pred_s], dim=-1)              # (..., 2)
        pred_norm = pred_vec.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        pred_unit = pred_vec / pred_norm

        # angle unit vector for ground-truth (assume already unit; renorm just in case)
        true_vec = torch.stack([true_c, true_s], dim=-1)
        true_unit = true_vec / true_vec.norm(dim=-1, keepdim=True).clamp_min(self.eps)

        # --- angle loss: 1 - dot(unit_pred, unit_true) ---
        dot = (pred_unit * true_unit).sum(dim=-1)                     # (...)
        L_angle = 1.0 - dot                                           # in [0, 2]

        # --- speed loss: Huber (smooth L1) ---
        L_speed = torch.nn.functional.smooth_l1_loss(pred_speed, true_speed, reduction='none')

        # --- optional: penalize deviation from unit norm (helps training) ---
        L_norm = (pred_norm.squeeze(-1) - 1.0).pow(2)
        
        # Combine losses (equal weighting for speed and angle, small penalty for norm)
        loss = (L_speed.mean() + L_angle.mean() + 0.1 * L_norm.mean())
        return loss

class GaussianNLLLoss(nn.Module):
    def __init__(self, config: dict, eps: float = 1e-6):
        super().__init__()
        self.dt = config["dt"]
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds: [B, T, 4] -> [mu_x, mu_y, log_sigma_x, log_sigma_y]
        # targets: [B, T, 6] -> as in your code

        # (if there *is* an extra dim, handle it explicitly)
        if preds.dim() == 4 and preds.size(-2) == 1:
            preds = preds.squeeze(-2)

        mu = preds[..., 0:2]             # [B,T,2]
        log_sigma = preds[..., 2:4]      # [B,T,2]

        # clamp std range to something reasonable (≈ [0.01, 20])
        log_sigma = torch.clamp(log_sigma, min=-5.0, max=3.0)
        sigma = torch.exp(log_sigma)

        # build true deltas
        speed = targets[..., 3]
        tan_sin = targets[..., 4]
        tan_cos = targets[..., 5]

        dx = speed * tan_cos * self.dt
        dy = speed * tan_sin * self.dt
        delta = torch.stack([dx, dy], dim=-1)  # [B,T,2]

        # NLL for diagonal Gaussian
        diff = (delta - mu) / (sigma + self.eps)
        quad = (diff ** 2).sum(dim=-1)                     # [B,T]
        logdet = 2 * log_sigma.sum(dim=-1)                 # [B,T]

        nll = 0.5 * (quad + logdet + 2 * math.log(2 * math.pi))
        return nll.mean()

class CorrelatedGaussianNLLLoss(nn.Module):
    def __init__(self, config: dict, eps: float = 1e-6):
        super().__init__()
        self.dt = config["dt"]
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds:   [B, T, 5] -> [mu_x, mu_y, log_sigma_x, log_sigma_y, rho_raw]
        targets: [B, T, 6] -> as in your code: [x, y, heading, speed, tan_sin, tan_cos] or similar
        """

        # handle the extra singleton dim if it appears
        if preds.dim() == 4 and preds.size(-2) == 1:
            preds = preds.squeeze(-2)  # [B, T, 5]

        # means
        mu = preds[..., 0:2]           # [B, T, 2]

        # log standard deviations
        log_sigma = preds[..., 2:4]    # [B, T, 2]
        # clamp std range to something reasonable (≈ [0.01, 20])
        log_sigma = torch.clamp(log_sigma, min=-5.0, max=3.0)
        sigma = torch.exp(log_sigma)   # [B, T, 2]
        sigma_x = sigma[..., 0]
        sigma_y = sigma[..., 1]

        # raw correlation -> rho in (-1, 1)
        rho_raw = preds[..., 4]        # [B, T]
        rho = torch.tanh(rho_raw)
        # extra clamp for numerical safety near ±1
        rho = torch.clamp(rho, -0.999, 0.999)

        # build true deltas (same as your original code)
        speed   = targets[..., 3]
        tan_sin = targets[..., 4]
        tan_cos = targets[..., 5]

        dx = speed * tan_cos * self.dt
        dy = speed * tan_sin * self.dt
        delta = torch.stack([dx, dy], dim=-1)  # [B, T, 2]

        # standardized differences
        diff = delta - mu            # [B, T, 2]
        zx = diff[..., 0] / (sigma_x + self.eps)  # [B, T]
        zy = diff[..., 1] / (sigma_y + self.eps)  # [B, T]

        # quadratic form for correlated bivariate Gaussian:
        # Q = (zx^2 + zy^2 - 2 * rho * zx * zy) / (1 - rho^2)
        one_minus_rho2 = 1.0 - rho * rho
        one_minus_rho2 = one_minus_rho2.clamp_min(self.eps)   # avoid divide-by-zero
        quad = (zx * zx + zy * zy - 2.0 * rho * zx * zy) / one_minus_rho2  # [B, T]

        # log determinant |Σ| for covariance:
        # Σ = [[σx^2, ρ σx σy],
        #      [ρ σx σy, σy^2]]
        # |Σ| = σx^2 σy^2 (1 - ρ^2)
        # log|Σ| = 2 log σx + 2 log σy + log(1 - ρ^2)
        logdet_diag = 2.0 * log_sigma.sum(dim=-1)             # 2 log σx + 2 log σy
        log_one_minus_rho2 = torch.log(one_minus_rho2)        # log(1 - ρ^2)
        logdet = logdet_diag + log_one_minus_rho2             # [B, T]

        # NLL = 0.5 * (Q + log((2π)^2 |Σ|))
        #     = 0.5 * (Q + log|Σ| + 2 log(2π))
        nll = 0.5 * (quad + logdet + 2.0 * math.log(2.0 * math.pi))

        return nll.mean()


class TrajectoryMetrics:
    """Class for calculating trajectory prediction metrics."""
    
    def __init__(self, config: dict):
        self.dt = config['dt']
        self.output_distribution_type = config['output_distribution_type']
        self.center_point = config['center_point']
    
    @staticmethod
    def gaussian_to_cartesian(current_state: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        Convert Gaussian predictions (mean_dx, mean_dy, log_std_dx, log_std_dy) to cartesian coordinates.
        
        Args:
            current_state: [batch_size, 2] - Current state with [x, y]
            predictions: [batch_size, prediction_horizon, 4] - Predictions with [mean_dx, mean_dy, log_std_dx, log_std_dy]
        
        Returns:
            cartesian_trajectory: [batch_size, prediction_horizon, 2] - Trajectory in cartesian coordinates (x, y)
        """
        batch_size = current_state.shape[0]
        predictions = predictions.squeeze(-2)
        prediction_horizon = predictions.shape[1]
        
        # Extract current position from polar coordinates
        current_x = current_state[:, 0]  # [batch_size]
        current_y = current_state[:, 1]  # [batch_size]
        
        # Initialize trajectory
        trajectory = torch.zeros(batch_size, prediction_horizon, 2, device=predictions.device)
        
        # Current position for tracking
        x = current_x
        y = current_y
        
        # For each prediction timestep, accumulate deltas
        for t in range(prediction_horizon):
            # Extract mean deltas at timestep t (ignore std for trajectory)
            mean_dx = predictions[:, t, 0]  # [batch_size]
            mean_dy = predictions[:, t, 1]  # [batch_size]
            
            # Update position
            x = x + mean_dx
            y = y + mean_dy
            
            # Store position
            trajectory[:, t, 0] = x
            trajectory[:, t, 1] = y
        
        return trajectory
    
    @staticmethod
    def polar_to_cartesian(current_state: torch.Tensor, predictions: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Convert polar predictions (speed, heading) to cartesian coordinates.
        
        Args:
            current_state: [batch_size, 2] - Current state with [x, y]
            predictions: [batch_size, prediction_horizon, 3] - Predictions with [speed, tangent_sin, tangent_cos]
            dt: Time step duration in seconds
        
        Returns:
            cartesian_trajectory: [batch_size, prediction_horizon, 2] - Trajectory in cartesian coordinates (x, y)
        """
        batch_size = current_state.shape[0]
        predictions = predictions.squeeze(-2)
        prediction_horizon = predictions.shape[1]

        # Extract current position from polar coordinates
        # r is the distance, sin_theta and cos_theta give us the angle
        current_x = current_state[:, 0]  # [batch_size]
        current_y = current_state[:, 1]  # [batch_size]
        
        # Initialize trajectory
        trajectory = torch.zeros(batch_size, prediction_horizon, 2, device=predictions.device)
        
        # Current position for tracking
        x = current_x
        y = current_y
        
        # For each prediction timestep, compute displacement and update position
        for t in range(prediction_horizon):
            # Extract predictions at timestep t
            speed = predictions[:, t, 0]  # [batch_size]
            tangent_sin = predictions[:, t, 1]  # [batch_size]
            tangent_cos = predictions[:, t, 2]  # [batch_size]
            
            # Compute velocity in cartesian coordinates
            velocity_x = speed * tangent_cos
            velocity_y = speed * tangent_sin
            
            # Update position
            x = x + velocity_x * dt
            y = y + velocity_y * dt
            
            # Store position
            trajectory[:, t, 0] = x
            trajectory[:, t, 1] = y
        
        return trajectory
    
    def target_to_cartesian(self, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert target tensor from polar to cartesian coordinates.
        
        Args:
            target_tensor: [batch_size, prediction_horizon, 6] - Target with [r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos]
        
        Returns:
            cartesian_positions: [batch_size, prediction_horizon, 2] - Positions in cartesian coordinates (x, y)
        """
        scene_center = self.center_point
        # Extract r, sin_theta, cos_theta from target
        r = target_tensor[:, :, 0]  # [batch_size, prediction_horizon]
        sin_theta = target_tensor[:, :, 1]  # [batch_size, prediction_horizon]
        cos_theta = target_tensor[:, :, 2]  # [batch_size, prediction_horizon]
        
        # Convert to cartesian: x = r * cos(theta), y = r * sin(theta)
        x = r * cos_theta + scene_center[0]  # [batch_size, prediction_horizon]
        y = r * sin_theta + scene_center[1]  # [batch_size, prediction_horizon]
        
        # Stack x and y
        cartesian_positions = torch.stack([x, y], dim=-1)  # [batch_size, prediction_horizon, 2]
        
        return cartesian_positions
    
    def calculate_ade_fde(self, history_tensor: torch.Tensor, target_tensor: torch.Tensor, predictions: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate Average Displacement Error (ADE) and Final Displacement Error (FDE).
        
        Args:
            history_tensor: [batch_size, prediction_horizon, 3] - History with [x, y, theta]
            target_tensor: [batch_size, prediction_horizon, 6] - Ground truth positions
            predictions: [batch_size, prediction_horizon, 3 or 4] - Predictions with [speed, tangent_sin, tangent_cos] or [mean_dx, mean_dy, log_std_dx, log_std_dy]
        
        Returns:
            ade: Average Displacement Error
            fde: Final Displacement Error
        """

        # Convert targets to cartesian
        history_cartesian = self.target_to_cartesian(history_tensor)  # [batch_size, prediction_horizon, 2]
        target_cartesian = self.target_to_cartesian(target_tensor)  # [batch_size, prediction_horizon, 2]

        current_coords = history_cartesian[:, -1, :]

        # Convert predictions to cartesian based on output type
        if self.output_distribution_type == 'gaussian':
            pred_cartesian = self.gaussian_to_cartesian(current_coords, predictions)  # [batch_size, prediction_horizon, 2]
        else:
            pred_cartesian = self.polar_to_cartesian(current_coords, predictions, self.dt)  # [batch_size, prediction_horizon, 2]
        
        # Create mask for non-zero targets (non-padded values)
        mask = (target_tensor[:, :, :3] != 0).any(dim=-1).float()  # [batch_size, prediction_horizon]
        
        # Calculate L2 displacement errors
        displacement_errors = torch.norm(pred_cartesian - target_cartesian, dim=-1)  # [batch_size, prediction_horizon]
        
        # Apply mask
        masked_errors = displacement_errors * mask
        
        # Calculate ADE (average across all timesteps)
        total_valid = mask.sum()
        if total_valid > 0:
            ade = masked_errors.sum() / total_valid
        else:
            ade = torch.tensor(0.0, device=predictions.device)
        
        # Calculate FDE (error at final timestep)
        final_mask = mask[:, -1]  # [batch_size]
        final_errors = displacement_errors[:, -1] * final_mask  # [batch_size]
        
        total_valid_final = final_mask.sum()
        if total_valid_final > 0:
            fde = final_errors.sum() / total_valid_final
        else:
            fde = torch.tensor(0.0, device=predictions.device)
        
        return ade.item(), fde.item()

