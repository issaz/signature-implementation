import torch


class GeometricBrownianMotion(torch.nn.Module):

    def __init__(self, mu, sigma, noise_type: str, sde_type: str):
        super().__init__()
        self.mu      = torch.nn.Parameter(torch.tensor(mu, dtype=torch.float64), requires_grad=True)
        self.sigma   = torch.nn.Parameter(torch.tensor(sigma, dtype=torch.float64), requires_grad=True)
        self.noise_type = noise_type
        self.sde_type   = sde_type

    def f(self, t, y):
        return self.mu *y

    def g(self, t, y):
        return self.sigma *y