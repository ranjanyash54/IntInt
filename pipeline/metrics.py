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

# class GaussianNLLLoss(nn.Module):
#     """Gaussian Negative Log-Likelihood loss for delta x and delta y predictions."""
    
#     def __init__(self, config: dict, eps: float = 1e-6):
#         super().__init__()
#         self.dt = config['dt']
#         self.eps = eps
    
#     def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         """
#         Compute Gaussian NLL loss for delta x and delta y.
        
#         Args:
#             predictions: [batch_size, prediction_horizon, 5] - [mu_x, mu_y, a, b, c]
#             targets: [batch_size, prediction_horizon, 6] - [r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos]
        
#         Returns:
#             Loss value
#         """
#         predictions = predictions.squeeze(-2)
#         batch_size, prediction_horizon, _ = predictions.shape
#         # Extract Gaussian parameters

#         mean = predictions[:, :, :2]
#         a, b, c = predictions[:, :, 2], predictions[:, :, 3], predictions[:, :, 4]
#         l11 = F.softplus(a) + self.eps              # positive
#         l21 = b                                # free
#         l22 = F.softplus(c) + self.eps              # positive

#         # build lower-triangular L: shape (B,2,2)
#         L = torch.zeros(batch_size, prediction_horizon, 2, 2, device=predictions.device, dtype=predictions.dtype)
#         L[..., 0, 0] = l11
#         L[..., 1, 0] = l21
#         L[..., 1, 1] = l22

#         # Compute actual deltas from targets: delta = speed * tangent
#         # targets: [r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos]
#         speed = targets[:, :, 3]  # [batch_size, prediction_horizon]
#         tangent_sin = targets[:, :, 4]  # [batch_size, prediction_horizon]
#         tangent_cos = targets[:, :, 5]  # [batch_size, prediction_horizon]
        
#         actual_dx = speed * tangent_cos * self.dt  # [batch_size, prediction_horizon]
#         actual_dy = speed * tangent_sin * self.dt  # [batch_size, prediction_horizon]

#         actual_delta = torch.stack([actual_dx, actual_dy], dim=-1)  # [batch_size, prediction_horizon, 2]

#         diff = (actual_delta - mean)[..., None]  # [batch_size, prediction_horizon, 2, 1]
#         z = torch.linalg.solve_triangular(L, diff, upper=False)  # (B,T,2,1)
#         # Quadratic term: ||z||^2
#         quad = torch.sum(z.squeeze(-1)**2, dim=-1)               # (B,T)

#         # log|Σ| = 2 * (log l11 + log l22)
#         diag = torch.diagonal(L, dim1=-2, dim2=-1)               # (B,T,2)
#         logdet = 2 * torch.sum(torch.log(diag + self.eps), dim=-1)    # (B,T)

#         nll = 0.5 * (quad + logdet + 2 * math.log(2 * math.pi))  # (B,T)

#         return nll.mean()


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

