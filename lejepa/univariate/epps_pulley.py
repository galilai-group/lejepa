import torch
from .base import UnivariateTest
from torch import distributed as dist
from torch.distributed.nn import all_reduce as functional_all_reduce
from torch.distributed.nn import ReduceOp


def all_reduce(x, op="AVG"):
    if dist.is_available() and dist.is_initialized():
        op = ReduceOp.__dict__[op.upper()]
        return functional_all_reduce(x, op)
    else:
        return x


class EppsPulley(UnivariateTest):
    """
    Fast Epps-Pulley two-sample test statistic for univariate distributions.

    This implementation uses numerical integration over the characteristic function
    to compute a goodness-of-fit test statistic. The test compares the empirical
    characteristic function against a standard normal distribution.

    The statistic is computed as:
        T = N * ∫ |φ_empirical(t) - φ_normal(t)|² w(t) dt

    where φ_empirical is the empirical characteristic function, φ_normal is the
    standard normal characteristic function, and w(t) is an integration weight.

    Args:
        t_max (float, optional): Maximum integration point. The grid is linearly
            spaced over [0, t_max]. Default: 3.
        n_points (int, optional): Number of integration points, including 0.
            Must be odd. Default: 17.
        integration (str, optional): Integration method to use. One of:
            - 'trapezoid': Trapezoidal rule over [0, t_max]
            - 'simpson': Composite Simpson's 1/3 rule over [0, t_max]
            Default: 'trapezoid'. Any other value raises ``ValueError``.

    Attributes:
        t (torch.Tensor): Integration points (positive half, including 0).
        weights (torch.Tensor): Precomputed integration weights incorporating
            symmetry and φ(t) = exp(-t²/2).
        phi (torch.Tensor): Precomputed φ(t) = exp(-t²/2) values.
        integration (str): Selected integration method.
        n_points (int): Number of integration points.

    Notes:
        - The implementation exploits symmetry: only t ≥ 0 are computed, and
          contributions from -t are implicitly added via doubled weights.
        - Supports distributed training via all_reduce operations.

    Example:
        >>> test = EppsPulley(t_max=5.0, n_points=21, integration='simpson')
        >>> samples = torch.randn(1000)  # Standard normal samples
        >>> statistic = test(samples)
        >>> print(f"Test statistic: {statistic.item():.4f}")
    """

    INTEGRATION_METHODS = ("trapezoid", "simpson")

    def __init__(
        self, t_max: float = 3, n_points: int = 17, integration: str = "trapezoid"
    ):
        super().__init__()
        if integration not in self.INTEGRATION_METHODS:
            raise ValueError(
                f"Unknown integration method {integration!r}; "
                f"expected one of {self.INTEGRATION_METHODS}"
            )
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points
        # Precompute phi

        # Linearly spaced positive points (including 0)
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        # Quadrature weights over [0, t_max], doubled because the integrand is
        # even in t, so the integral over [-t_max, t_max] is twice this one.
        if integration == "trapezoid":
            weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
            weights[[0, -1]] = dt  # Half-weight at the endpoints
        else:  # simpson
            weights = torch.full((n_points,), 2 * dt / 3, dtype=torch.float32)
            weights[1:-1:2] *= 4
            weights[2:-1:2] *= 2
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x):
        N = x.size(-2)
        # Compute cos/sin only for t >= 0
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # Mean across batch
        cos_mean = cos_vals.mean(-3)  # (*, n_points)
        sin_mean = sin_vals.mean(-3)  # (*, n_points)

        # DDP reduction
        cos_mean = all_reduce(cos_mean)
        sin_mean = all_reduce(sin_mean)

        # Compute error (symmetry already in weights)
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted integration
        return (err @ self.weights) * N * self.world_size


class DeprecatedEppsPulley(UnivariateTest):
    """
    PyTorch implementation of the Epps-Pulley test for univariate normality
    based on empirical characteristic function.
    """

    def __init__(self, t_range=(-3, 3), n_points=10, weight_type="gaussian"):
        """
        Parameters:
        -----------
        t_range : tuple
            Range of t values for integration
        n_points : int
            Number of points for numerical integration
        weight_type : str
            Type of weight function ('gaussian' or 'uniform')
        """
        super().__init__()
        self.t_range = t_range
        self.n_points = n_points
        self.weight_type = weight_type

    def empirical_cf(self, x, t):
        """
        Compute empirical characteristic function φ̂(t) = (1/n)∑ e^(itX_j)

        Parameters:
        -----------
        x : torch.Tensor, shape (n,)
            Sample data
        t : torch.Tensor, shape (m,)
            Points where to evaluate CF

        Returns:
        --------
        torch.Tensor, shape (m,)
            Empirical characteristic function values
        """
        # Reshape for broadcasting: x (n,1), t (1,m)
        x_expanded = x.unsqueeze(1)  # (n, 1, *)
        t_expanded = t.unsqueeze(0)  # (1, m)
        extra_dims = x.ndim - 1
        for _ in range(extra_dims):
            t_expanded = t_expanded.unsqueeze(-1)

        # Compute e^(itX_j) for all i,j
        # Real part: cos(tX), Imaginary part: sin(tX)
        real_part = torch.cos(t_expanded * x_expanded)  # (n, m, *)
        imag_part = torch.sin(t_expanded * x_expanded)  # (n, m, *)

        # Average over samples
        # TODO: handle DDP here for gather_all
        empirical_real = torch.mean(real_part, dim=0)  # (m, *)
        empirical_imag = torch.mean(imag_part, dim=0)  # (m, *)

        return torch.complex(empirical_real.float(), empirical_imag.float())

    def normal_cf(self, t, mu, sigma):
        """
        Theoretical characteristic function for normal distribution
        φ_N(t) = exp(iμt - σ²t²/2)

        Parameters:
        -----------
        t : torch.Tensor
            Points where to evaluate CF
        mu : float
            Mean parameter
        sigma : float
            Standard deviation parameter

        Returns:
        --------
        torch.Tensor
            Normal characteristic function values
        """
        # exp(iμt - σ²t²/2) = exp(-σ²t²/2) * exp(iμt)
        magnitude = torch.exp(-0.5 * sigma**2 * t**2)
        phase = mu * t

        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)

        return torch.complex(real_part.float(), imag_part.float())

    def weight_function(self, t):
        """
        Weight function for integration

        Parameters:
        -----------
        t : torch.Tensor
            Points where to evaluate weight

        Returns:
        --------
        torch.Tensor
            Weight values
        """
        if self.weight_type == "gaussian":
            return torch.exp(-(t**2) / 2)
        elif self.weight_type == "uniform":
            return torch.ones_like(t)
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")

    def forward(self, x):
        """
        Compute Epps-Pulley test statistic

        Parameters:
        -----------
        x : torch.Tensor, shape (n,)
            Sample data

        Returns:
        --------
        float
            Test statistic value
        """
        device = x.device

        with torch.no_grad():
            # Create integration points
            t_min, t_max = self.t_range
            t = torch.linspace(t_min, t_max, self.n_points, device=device)

            # Compute theoretical characteristic functions
            phi_normal = self.normal_cf(t, mu=0.0, sigma=1.0)  # Standard normal
            # Get weight function
            weights = self.weight_function(t)
            # unsqueeze
            for _ in range(x.ndim - 1):
                phi_normal = phi_normal.unsqueeze(-1)
                weights = weights.unsqueeze(-1)

        # Compute empirical characteristic functions
        phi_emp = self.empirical_cf(x, t)

        # Compute squared difference
        diff = phi_emp - phi_normal
        squared_diff = torch.real(diff * torch.conj(diff))  # |φ̂ - φ_N|²

        # Apply weight function and integrate (trapezoidal rule)
        integrand = squared_diff * weights
        integral = torch.trapz(integrand, t, dim=0)

        # Test statistic (without n* scaling)
        test_stat = integral

        return test_stat
