"""Training objectives used by SIMPLE (paper §3.2–3.3) and by its ablations (Appendix A).

Each score is a latent Gaussian ``N(mu, sigma^2)``. Under Thurstone's Case III model the
probability that candidate ``i`` is preferred over ``j`` is::

    P(i > j) = Phi((mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2))

SIMPLE minimises ``L = L_Thurstone + lambda * L_NLL``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _weighted_mean(per_sample: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        return per_sample.mean()
    return (weight * per_sample).sum() / weight.sum()


def _log_ndtr(x: torch.Tensor) -> torch.Tensor:
    try:
        return torch.special.log_ndtr(x)
    except AttributeError:  # torch < 1.11
        return torch.log(0.5 * torch.erfc(-x / math.sqrt(2)).clamp(min=1e-7))


def thurstone_loss(
    mu_pos: torch.Tensor,
    mu_neg: torch.Tensor,
    sigma_pos: torch.Tensor,
    sigma_neg: torch.Tensor,
    weight: torch.Tensor | None = None,
    margin: torch.Tensor | None = None,
) -> torch.Tensor:
    """``-log Phi((mu_pos - mu_neg - margin) / sqrt(sigma_pos^2 + sigma_neg^2))`` (paper Eq. 4).

    ``margin`` (default 0) asks the score gap to reach at least the human rating gap;
    ``weight`` gives a per-pair weight, and the result is the weighted mean.
    """
    combined_std = (sigma_pos**2 + sigma_neg**2).sqrt().clamp(min=1e-4)
    diff = mu_pos - mu_neg
    if margin is not None:
        diff = diff - margin
    return _weighted_mean(-_log_ndtr(diff / combined_std), weight)


def bradley_terry_loss(
    mu_pos: torch.Tensor,
    mu_neg: torch.Tensor,
    sigma_pos: torch.Tensor | None = None,
    sigma_neg: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    margin: torch.Tensor | None = None,
) -> torch.Tensor:
    """``-log sigmoid(mu_pos - mu_neg - margin)``. Ignores sigma (Appendix A ablation)."""
    diff = mu_pos - mu_neg
    if margin is not None:
        diff = diff - margin
    return _weighted_mean(-F.logsigmoid(diff), weight)


def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of ``target`` under ``N(mu, sigma^2)`` (paper Eq. 5)."""
    return -torch.distributions.Normal(mu, sigma).log_prob(target).mean()


def mse_loss(mu: torch.Tensor, sigma: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between ``mu`` and ``target``. Ignores sigma (Appendix A ablation)."""
    return F.mse_loss(mu, target)


PAIRWISE_LOSSES = {"thurstone": thurstone_loss, "bt": bradley_terry_loss}
REGRESSION_LOSSES = {"nll": gaussian_nll_loss, "mse": mse_loss}


def preference_probability(mu_a, sigma_a, mu_b, sigma_b):
    """Model-implied probability that candidate ``a`` is better than ``b`` (paper Eq. 3).

    Works on floats, sequences or numpy arrays; returns a float or a numpy array.
    """
    t = [torch.as_tensor(x, dtype=torch.float64) for x in (mu_a, sigma_a, mu_b, sigma_b)]
    p = torch.special.ndtr((t[0] - t[2]) / (t[1] ** 2 + t[3] ** 2).sqrt()).numpy()
    return float(p) if p.ndim == 0 else p
