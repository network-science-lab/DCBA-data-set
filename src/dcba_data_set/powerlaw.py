"""Utilities for fitting power-law distributions to data, using MLE."""

import math
from typing import Callable, Tuple

import torch


def golden_section_search(
    f: Callable[[float], float], a: float, b: float, tol: float = 1e-6
) -> float:
    """Find the minimum of a unimodal function f(x) within the interval [a, b]."""
    gr: float = (math.sqrt(5) + 1) / 2

    c: float = b - (b - a) / gr
    d: float = a + (b - a) / gr

    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2


def fit_truncated_power_law(
    data: torch.Tensor, search_bounds: Tuple[float, float] = (1.01, 10.0), tol: float = 1e-6
) -> float:
    """Estimate the exponent (alpha) of a truncated power-law distribution."""
    xmin = torch.min(data).item()
    xmax = torch.max(data).item()
    n: int = data.numel()

    sum_log_x: float = torch.sum(torch.log(data)).item()

    def neg_log_likelihood(alpha: float) -> float:
        gamma: float = alpha - 1.0

        if abs(gamma) < 1e-7:
            log_C: float = -math.log(math.log(xmax / xmin))
            return -(n * log_C - alpha * sum_log_x)

        else:
            num: float = gamma
            den: float = (xmin**-gamma) - (xmax**-gamma)

            if den <= 0 or (num / den) <= 0:
                return float("inf")

            log_C: float = math.log(num / den)
            return -(n * log_C - alpha * sum_log_x)

    best_alpha: float = golden_section_search(
        f=neg_log_likelihood, a=search_bounds[0], b=search_bounds[1], tol=tol
    )

    return best_alpha
