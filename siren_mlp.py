import torch
import torch.nn as nn
import math


# ---------- Periodic Time Encoding ----------
def encode_time(t):
    """
    Encode normalized time t in [0,1] into (cos, sin) form for periodic representation.
    Args:
        t: Tensor of shape (...,) or scalar
    Returns:
        encoded tensor of shape (..., 2)
    """
    # 2π t encoding
    theta = 2 * math.pi * t
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


# ---------- Sine Activation (SIREN) ----------
class SineActivation(nn.Module):
    def __init__(self, w0=30.0):
        """
        SIREN sine activation with frequency scaling parameter w0.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# ---------- SIREN MLP ----------
class SIRENVelocityField(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=3, w0=30.0):
        """
        SIREN MLP for velocity vector field.
        Args:
            hidden_dim: number of neurons per layer
            num_layers: total number of hidden layers
            w0: frequency scaling parameter for sine activation
        """
        super().__init__()

        layers = []
        in_dim = 5  # (x, y, z) + (cos t, sin t)

        # First layer with SIREN initialization
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(SineActivation(w0=w0))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(SineActivation(w0=1.0))  # Subsequent layers use w0=1

        # Output layer (vx, vy, vz)
        layers.append(nn.Linear(hidden_dim, 3))

        self.model = nn.Sequential(*layers)

        # Initialize weights according to SIREN paper
        self.init_weights(w0)

    def init_weights(self, w0):
        # First layer init
        first_linear = self.model[0]
        with torch.no_grad():
            first_linear.weight.uniform_(-1 / in_dim, 1 / in_dim)

        # Other layers
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear) and i != 0:
                fan_in = layer.weight.size(1)
                layer.weight.uniform_(-math.sqrt(6 / fan_in) / 1.0, math.sqrt(6 / fan_in) / 1.0)

    def forward(self, coords, t):
        """
        Forward pass through SIREN MLP.
        Args:
            coords: tensor of shape (N, 3) for spatial coordinates in [-1,1]^3
            t: tensor of shape (N,) for time in [0,1]
        Returns:
            velocity vectors: (N, 3)
        """
        # Encode time to periodic (cos,sin)
        t_encoded = encode_time(t)  # (N,2)
        inputs = torch.cat([coords, t_encoded], dim=-1)  # (N,5)

        return self.model(inputs)