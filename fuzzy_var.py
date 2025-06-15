from typing import Callable, Dict, Union
import math
import numpy as np

# Type alias for scalar or array input
dtype = Union[float, np.ndarray]
MembershipFunction = Callable[[dtype], dtype]

# --- t-norms ---
def tnorm_min(a: dtype, b: dtype) -> dtype:
    """Gödel t-norm (minimum): T(a,b) = min(a, b)."""
    return np.minimum(a, b)


def tnorm_prod(a: dtype, b: dtype) -> dtype:
    """Probabilistic product t-norm: T(a,b) = a * b."""
    return a * b


def tnorm_lukas(a: dtype, b: dtype) -> dtype:
    """Lukasiewicz t-norm: T(a,b) = max(0, a + b - 1)."""
    return np.clip(a + b - 1, 0.0, 1.0)

# --- t-conorms (s-norms) ---

def s_norm_max(a: dtype, b: dtype) -> dtype:
    """Maximum t-conorm: S(a,b) = max(a, b)."""
    return np.maximum(a, b)


def s_norm_prob(a: dtype, b: dtype) -> dtype:
    """Probabilistic sum t-conorm: S(a,b) = a + b - a*b."""
    return a + b - a * b


def s_norm_lukas(a: dtype, b: dtype) -> dtype:
    """Lukasiewicz t-conorm: S(a,b) = min(1, a + b)."""
    return np.minimum(a + b, 1.0)

def tconorm_aggregate(df: pd.DataFrame, s_norm) -> np.ndarray:
    """
    Aggregate every row (intervention) across *all* park columns with `s_norm`.

    Returns a 1-D array of length = n_interventions.
    Safe against duplicate column labels because it operates on `.values`.
    """
    if df.shape[1] == 0:
        raise ValueError("Membership table has no columns to aggregate.")

    vals = df.values                    # shape = (n_rows, n_cols)
    agg   = vals[:, 0]                  # first column → 1-D
    for j in range(1, vals.shape[1]):
        agg = s_norm(agg, vals[:, j])   # still 1-D
    return agg

# --- fuzzy membership functions ---

def triangular_mf(a: float, m: float, b: float) -> MembershipFunction:
    """Generate a triangular membership function over ℝ.
    Supports vertical edges when a==m or m==b."""
    assert a < b, "Require a < b"
    assert a <= m <= b, "Require a <= m <= b"
    
    def mf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x_arr = np.asarray(x)
        # Below a or above b → 0
        mu = np.zeros_like(x_arr, dtype=float)
        # Rising edge
        if m != a:
            mask = (x_arr > a) & (x_arr < m)
            mu[mask] = (x_arr[mask] - a) / (m - a)
        else:
            mu[x_arr == a] = 1.0
        # Falling edge
        if m != b:
            mask = (x_arr > m) & (x_arr < b)
            mu[mask] = (b - x_arr[mask]) / (b - m)
        else:
            mu[x_arr == b] = 1.0
        # Peak
        mu[x_arr == m] = 1.0
        # Return scalar if input was scalar
        return float(mu) if np.isscalar(x) else mu
    return mf


def trapezoidal_mf(a: float, b: float, c: float, d: float) -> MembershipFunction:
    """Generate a trapezoidal membership function over ℝ.
    Supports vertical edges when a==b or c==d. Use math.inf for unbounded sides."""
    assert a < d, "Require a < d"
    assert b <= c, "Require b <= c"

    def mf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x_arr = np.asarray(x)
        mu = np.zeros_like(x_arr, dtype=float)
        # Rising edge
        if b != a:
            mask = (x_arr > a) & (x_arr < b)
            mu[mask] = (x_arr[mask] - a) / (b - a)
        else:
            mu[x_arr == b] = 1.0
        # Plateau
        mask = (x_arr >= b) & (x_arr <= c)
        mu[mask] = 1.0
        # Falling edge
        if d != c:
            mask = (x_arr > c) & (x_arr < d)
            mu[mask] = (d - x_arr[mask]) / (d - c)
        else:
            mu[x_arr == c] = 1.0
        return float(mu) if np.isscalar(x) else mu
    return mf

class FuzzyVariable:
    """Callable fuzzy variable over ℝ mapping term names to membership values. Returns a map"""
    def __init__(self):
        self._terms: Dict[str, MembershipFunction] = {}

    def add_triangular(self, name: str, a: float, m: float, b: float):
        """Add a triangular term."""
        self._terms[name] = triangular_mf(a, m, b)

    def add_trapezoidal(self, name: str, a: float, b: float, c: float, d: float):
        """Add a trapezoidal term."""
        self._terms[name] = trapezoidal_mf(a, b, c, d)

    def __call__(
        self, x: Union[float, np.ndarray]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Evaluate all membership functions at x."""
        return {name: mf(x) for name, mf in self._terms.items()}

    def __str__(self) -> str:
        lines = ["FuzzyVariable with terms:"]
        for name, mf in self._terms.items():
            lines.append(f"  - {name}")
        return "\n".join(lines)

    __repr__ = __str__


def fuzz_dist(a: float, b: float, c: float, d: float, e: float) -> FuzzyVariable:
    """Construct distance-based fuzzy variable: close, mid-close, mid, mid-far, far."""
    fv = FuzzyVariable()
    fv.add_trapezoidal("close",    0,   0,   a,  b)
    fv.add_triangular("mid-close", a,   b,   c)
    fv.add_triangular("mid",       b,   c,   d)
    fv.add_triangular("mid-far",   c,   d,   e)
    fv.add_trapezoidal("far",      d,   e,   math.inf, math.inf)
    return fv
