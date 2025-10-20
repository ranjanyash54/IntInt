"""
Metrics for trajectory prediction evaluation.
"""

import torch
from typing import Tuple


class TrajectoryMetrics:
    """Class for calculating trajectory prediction metrics."""
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize trajectory metrics calculator.
        
        Args:
            dt: Time step duration in seconds
        """
        self.dt = dt
    
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
            predictions: [batch_size, prediction_horizon, 3] - Predictions with [speed, tangent_sin, tangent_cos]
            target_tensor: [batch_size, prediction_horizon, 6] - Ground truth positions
        
        Returns:
            ade: Average Displacement Error
            fde: Final Displacement Error
        """
        # Convert predictions to cartesian
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

