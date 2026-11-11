import math

import torch

from simple_metric.losses import (
    bradley_terry_loss,
    gaussian_nll_loss,
    preference_probability,
    thurstone_loss,
)


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def test_thurstone_matches_closed_form():
    mu_p, mu_n = torch.tensor([0.6]), torch.tensor([0.1])
    s_p, s_n = torch.tensor([0.3]), torch.tensor([0.4])
    expected = -math.log(_phi(0.5 / 0.5))
    assert abs(thurstone_loss(mu_p, mu_n, s_p, s_n).item() - expected) < 1e-6


def test_thurstone_margin_is_subtracted_from_the_numerator():
    mu_p, mu_n = torch.tensor([0.6]), torch.tensor([0.1])
    s = torch.tensor([0.3])
    with_margin = thurstone_loss(mu_p, mu_n, s, s, margin=torch.tensor([0.2])).item()
    expected = -math.log(_phi((0.5 - 0.2) / math.sqrt(0.18)))
    assert abs(with_margin - expected) < 1e-6


def test_weighted_mean():
    mu_p, mu_n = torch.tensor([0.5, -0.5]), torch.tensor([0.0, 0.0])
    s = torch.tensor([0.1, 0.1])
    per = [thurstone_loss(mu_p[i:i + 1], mu_n[i:i + 1], s[:1], s[:1]).item() for i in range(2)]
    w = torch.tensor([3.0, 1.0])
    got = thurstone_loss(mu_p, mu_n, s, s, weight=w).item()
    assert abs(got - (3 * per[0] + per[1]) / 4) < 1e-6


def test_bt_and_nll():
    assert abs(bradley_terry_loss(torch.tensor([1.0]), torch.tensor([0.0])).item()
               - math.log(1 + math.exp(-1))) < 1e-6
    nll = gaussian_nll_loss(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0])).item()
    assert abs(nll - (0.5 * math.log(2 * math.pi) + 0.5)) < 1e-6


def test_preference_probability():
    assert abs(preference_probability(0.5, 0.3, 0.5, 0.4) - 0.5) < 1e-12
    p = preference_probability([0.8, 0.0], [0.1, 0.1], [0.0, 0.8], [0.1, 0.1])
    assert p[0] > 0.99 and p[1] < 0.01
