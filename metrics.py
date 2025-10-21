"""
Metrics for trajectory prediction evaluation.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class MSELoss(nn.Module):
    """Custom MSE loss that handles missing values."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
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
        
        if self.reduction == 'mean':
            # Sum over all dimensions and divide by number of non-zero elements
            total_elements = mask.sum()
            if total_elements > 0:
                return masked_mse.sum() / total_elements
            else:
                return torch.tensor(0.0, device=predictions.device)
        elif self.reduction == 'sum':
            return masked_mse.sum()
        else:
            return masked_mse


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood loss for delta x and delta y predictions."""
    
    def __init__(self, reduction: str = 'mean', dt: float = 0.1):
        super().__init__()
        self.reduction = reduction
        self.dt = dt
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                current_state: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Gaussian NLL loss for delta x and delta y.
        
        Args:
            predictions: [batch_size, prediction_horizon, 4] - [mean_dx, mean_dy, log_std_dx, log_std_dy]
            targets: [batch_size, prediction_horizon, 6] - [r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos]
            current_state: [batch_size, 6] - Not used, kept for API compatibility
        
        Returns:
            Loss value
        """
        predictions = predictions.squeeze(-2)
        
        # Extract Gaussian parameters
        mean_dx = predictions[:, :, 0]  # [batch_size, prediction_horizon]
        mean_dy = predictions[:, :, 1]  # [batch_size, prediction_horizon]
        log_std_dx = predictions[:, :, 2].clamp(min=-10.0, max=2.0)  # [batch_size, prediction_horizon]
        log_std_dy = predictions[:, :, 3].clamp(min=-10.0, max=2.0)  # [batch_size, prediction_horizon]
        
        # Convert log_std to variance for numerical stability
        var_dx = torch.exp(2 * log_std_dx)  # variance = exp(2*log_std)
        var_dy = torch.exp(2 * log_std_dy)
        
        # Compute actual deltas from targets: delta = speed * tangent
        # targets: [r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos]
        speed = targets[:, :, 3]  # [batch_size, prediction_horizon]
        tangent_sin = targets[:, :, 4]  # [batch_size, prediction_horizon]
        tangent_cos = targets[:, :, 5]  # [batch_size, prediction_horizon]
        
        actual_dx = speed * tangent_cos * self.dt  # [batch_size, prediction_horizon]
        actual_dy = speed * tangent_sin * self.dt  # [batch_size, prediction_horizon]
        
        # Create mask for non-zero targets (non-padded values)
        mask = (targets[:, :, :3] != 0).any(dim=-1).float()  # [batch_size, prediction_horizon]
        
        # Compute negative log-likelihood (numerically stable version)
        # NLL = 0.5 * (log(2*pi) + log(var) + (x - mean)^2 / var)
        nll_dx = 0.5 * (math.log(2 * math.pi) + 2 * log_std_dx + (actual_dx - mean_dx)**2 / (var_dx + 1e-6))
        nll_dy = 0.5 * (math.log(2 * math.pi) + 2 * log_std_dy + (actual_dy - mean_dy)**2 / (var_dy + 1e-6))
        
        # Total NLL per timestep
        nll = nll_dx + nll_dy  # [batch_size, prediction_horizon]
        
        # Apply mask
        masked_nll = nll * mask
        
        if self.reduction == 'mean':
            total_elements = mask.sum()
            if total_elements > 0:
                return masked_nll.sum() / total_elements
            else:
                return torch.tensor(0.0, device=predictions.device)
        elif self.reduction == 'sum':
            return masked_nll.sum()
        else:
            return masked_nll


class VonMisesSpeedNLLLoss(nn.Module):
    """Von Mises distribution for angle + Log-Normal distribution for speed."""
    
    def __init__(self, reduction: str = 'mean', dt: float = 0.1):
        super().__init__()
        self.reduction = reduction
        self.dt = dt
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                current_state: torch.Tensor = None) -> torch.Tensor:
        """
        Compute von Mises + speed NLL loss.
        
        Args:
            predictions: [batch_size, prediction_horizon, 4] - [mu_sin, mu_cos, log_kappa, log_mean_speed]
            targets: [batch_size, prediction_horizon, 6] - [r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos]
            current_state: [batch_size, 6] - Not used, kept for API compatibility
        
        Returns:
            Loss value
        """
        predictions = predictions.squeeze(-2)
        
        # Extract von Mises parameters for angle
        mu_sin = predictions[:, :, 0]  # [batch_size, prediction_horizon]
        mu_cos = predictions[:, :, 1]  # [batch_size, prediction_horizon]
        log_kappa = predictions[:, :, 2].clamp(min=-5.0, max=5.0)  # [batch_size, prediction_horizon]
        log_mean_speed = predictions[:, :, 3].clamp(min=-10.0, max=5.0)  # [batch_size, prediction_horizon]
        
        # Normalize mu to unit vector
        mu_norm = torch.sqrt(mu_sin**2 + mu_cos**2 + 1e-8)
        mu_sin_normalized = mu_sin / mu_norm
        mu_cos_normalized = mu_cos / mu_norm
        
        kappa = torch.exp(log_kappa)
        mean_speed = torch.exp(log_mean_speed)
        
        # Extract actual values from targets
        target_sin = targets[:, :, 4]  # tangent_sin
        target_cos = targets[:, :, 5]  # tangent_cos
        target_speed = targets[:, :, 3]
        
        # Von Mises NLL for angle
        # NLL = -log(exp(kappa * cos(theta - mu)) / (2*pi*I_0(kappa)))
        # where cos(theta - mu) = cos(theta)*cos(mu) + sin(theta)*sin(mu)
        cos_diff = target_cos * mu_cos_normalized + target_sin * mu_sin_normalized
        # Approximate log(I_0(kappa)) for numerical stability
        # For small kappa: I_0(kappa) ≈ 1
        # For large kappa: log(I_0(kappa)) ≈ kappa - 0.5*log(2*pi*kappa)
        log_i0_kappa = torch.where(
            kappa < 3.75,
            torch.log(1 + (kappa**2) / 4 + (kappa**4) / 64),  # Small kappa approximation
            kappa - 0.5 * torch.log(2 * math.pi * kappa)  # Large kappa approximation
        )
        von_mises_nll = -kappa * cos_diff + log_i0_kappa + math.log(2 * math.pi)
        
        # Speed NLL (using simple L2 loss for mean speed)
        # Could use log-normal, but keeping it simple
        speed_nll = 0.5 * ((target_speed - mean_speed) / (mean_speed + 0.1))**2
        
        # Total NLL
        nll = von_mises_nll + speed_nll  # [batch_size, prediction_horizon]
        
        # Create mask for non-zero targets (non-padded values)
        mask = (targets[:, :, :3] != 0).any(dim=-1).float()  # [batch_size, prediction_horizon]
        
        # Apply mask
        masked_nll = nll * mask
        
        if self.reduction == 'mean':
            total_elements = mask.sum()
            if total_elements > 0:
                return masked_nll.sum() / total_elements
            else:
                return torch.tensor(0.0, device=predictions.device)
        elif self.reduction == 'sum':
            return masked_nll.sum()
        else:
            return masked_nll


class TrajectoryMetrics:
    """Class for calculating trajectory prediction metrics."""
    
    def __init__(self, dt: float = 0.1, output_distribution_type: str = 'linear'):
        """
        Initialize trajectory metrics calculator.
        
        Args:
            dt: Time step duration in seconds
            output_distribution_type: 'linear' (speed, cos, sin) or 'gaussian' (mean_dx, mean_dy, log_std_dx, log_std_dy)
        """
        self.dt = dt
        self.output_distribution_type = output_distribution_type
    
    @staticmethod
    def vonmises_speed_to_cartesian(current_state: torch.Tensor, predictions: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Convert von Mises + speed predictions to cartesian coordinates.
        
        Args:
            current_state: [batch_size, 6] - Current state with [r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos]
            predictions: [batch_size, prediction_horizon, 4] - Predictions with [mu_sin, mu_cos, log_kappa, log_mean_speed]
            dt: Time step duration in seconds
        
        Returns:
            cartesian_trajectory: [batch_size, prediction_horizon, 2] - Trajectory in cartesian coordinates (x, y)
        """
        batch_size = current_state.shape[0]
        predictions = predictions.squeeze(-2)
        prediction_horizon = predictions.shape[1]
        
        # Extract current position from polar coordinates
        r = current_state[:, 0]  # [batch_size]
        sin_theta = current_state[:, 1]  # [batch_size]
        cos_theta = current_state[:, 2]  # [batch_size]
        
        # Convert to cartesian: x = r * cos(theta), y = r * sin(theta)
        current_x = r * cos_theta  # [batch_size]
        current_y = r * sin_theta  # [batch_size]
        
        # Initialize trajectory
        trajectory = torch.zeros(batch_size, prediction_horizon, 2, device=predictions.device)
        
        # Current position for tracking
        x = current_x
        y = current_y
        
        # For each prediction timestep
        for t in range(prediction_horizon):
            # Extract mean direction and speed
            mu_sin = predictions[:, t, 0]
            mu_cos = predictions[:, t, 1]
            log_mean_speed = predictions[:, t, 3]
            
            # Normalize direction
            mu_norm = torch.sqrt(mu_sin**2 + mu_cos**2 + 1e-8)
            mu_sin_normalized = mu_sin / mu_norm
            mu_cos_normalized = mu_cos / mu_norm
            
            mean_speed = torch.exp(log_mean_speed.clamp(min=-10.0, max=5.0))
            
            # Compute velocity
            velocity_x = mean_speed * mu_cos_normalized
            velocity_y = mean_speed * mu_sin_normalized
            
            # Update position
            x = x + velocity_x * dt
            y = y + velocity_y * dt
            
            # Store position
            trajectory[:, t, 0] = x
            trajectory[:, t, 1] = y
        
        return trajectory
    
    @staticmethod
    def gaussian_to_cartesian(current_state: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        Convert Gaussian predictions (mean_dx, mean_dy, log_std_dx, log_std_dy) to cartesian coordinates.
        
        Args:
            current_state: [batch_size, 6] - Current state with [r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos]
            predictions: [batch_size, prediction_horizon, 4] - Predictions with [mean_dx, mean_dy, log_std_dx, log_std_dy]
        
        Returns:
            cartesian_trajectory: [batch_size, prediction_horizon, 2] - Trajectory in cartesian coordinates (x, y)
        """
        batch_size = current_state.shape[0]
        predictions = predictions.squeeze(-2)
        prediction_horizon = predictions.shape[1]
        
        # Extract current position from polar coordinates
        r = current_state[:, 0]  # [batch_size]
        sin_theta = current_state[:, 1]  # [batch_size]
        cos_theta = current_state[:, 2]  # [batch_size]
        
        # Convert to cartesian: x = r * cos(theta), y = r * sin(theta)
        current_x = r * cos_theta  # [batch_size]
        current_y = r * sin_theta  # [batch_size]
        
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
            current_state: [batch_size, 6] - Current state with [r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos]
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
        r = current_state[:, 0]  # [batch_size]
        sin_theta = current_state[:, 1]  # [batch_size]
        cos_theta = current_state[:, 2]  # [batch_size]
        
        # Convert to cartesian: x = r * cos(theta), y = r * sin(theta)
        current_x = r * cos_theta  # [batch_size]
        current_y = r * sin_theta  # [batch_size]
        
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
    
    @staticmethod
    def target_to_cartesian(target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert target tensor from polar to cartesian coordinates.
        
        Args:
            target_tensor: [batch_size, prediction_horizon, 6] - Target with [r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos]
        
        Returns:
            cartesian_positions: [batch_size, prediction_horizon, 2] - Positions in cartesian coordinates (x, y)
        """
        # Extract r, sin_theta, cos_theta from target
        r = target_tensor[:, :, 0]  # [batch_size, prediction_horizon]
        sin_theta = target_tensor[:, :, 1]  # [batch_size, prediction_horizon]
        cos_theta = target_tensor[:, :, 2]  # [batch_size, prediction_horizon]
        
        # Convert to cartesian: x = r * cos(theta), y = r * sin(theta)
        x = r * cos_theta  # [batch_size, prediction_horizon]
        y = r * sin_theta  # [batch_size, prediction_horizon]
        
        # Stack x and y
        cartesian_positions = torch.stack([x, y], dim=-1)  # [batch_size, prediction_horizon, 2]
        
        return cartesian_positions
    
    def calculate_ade_fde(self, current_state: torch.Tensor, predictions: torch.Tensor, 
                          target_tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate Average Displacement Error (ADE) and Final Displacement Error (FDE).
        
        Args:
            current_state: [batch_size, 6] - Current state with [r, sin_theta, cos_theta, speed, tangent_sin, tangent_cos]
            predictions: [batch_size, prediction_horizon, 3 or 4] - Predictions with [speed, tangent_sin, tangent_cos] or [mean_dx, mean_dy, log_std_dx, log_std_dy]
            target_tensor: [batch_size, prediction_horizon, 6] - Ground truth positions
        
        Returns:
            ade: Average Displacement Error
            fde: Final Displacement Error
        """
        # Convert predictions to cartesian based on output type
        if self.output_distribution_type == 'gaussian':
            pred_cartesian = self.gaussian_to_cartesian(current_state, predictions)  # [batch_size, prediction_horizon, 2]
        elif self.output_distribution_type == 'vonmises_speed':
            pred_cartesian = self.vonmises_speed_to_cartesian(current_state, predictions, self.dt)  # [batch_size, prediction_horizon, 2]
        else:
            pred_cartesian = self.polar_to_cartesian(current_state, predictions, self.dt)  # [batch_size, prediction_horizon, 2]
        
        # Convert targets to cartesian
        target_cartesian = self.target_to_cartesian(target_tensor)  # [batch_size, prediction_horizon, 2]
        
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

