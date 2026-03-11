"""Tests for synthetic datasets."""

import numpy as np
import pytest

from insurance_gas.datasets import (
    load_motor_frequency,
    load_severity_trend,
    load_loss_ratio,
    SyntheticDataset,
)


class TestMotorFrequency:
    def test_returns_dataset(self):
        d = load_motor_frequency()
        assert isinstance(d, SyntheticDataset)

    def test_length(self):
        for T in [24, 48, 60]:
            d = load_motor_frequency(T=T)
            assert len(d.y) == T

    def test_y_non_negative_integers(self):
        d = load_motor_frequency(T=60)
        assert np.all(d.y >= 0)
        np.testing.assert_array_equal(d.y, np.floor(d.y))

    def test_exposure_positive(self):
        d = load_motor_frequency(T=60)
        assert np.all(d.exposure > 0)

    def test_exposure_length(self):
        d = load_motor_frequency(T=60)
        assert len(d.exposure) == 60

    def test_filter_truth_positive(self):
        d = load_motor_frequency(T=60)
        assert np.all(d.filter_truth["mean"] > 0)

    def test_filter_truth_length(self):
        d = load_motor_frequency(T=60)
        assert len(d.filter_truth["mean"]) == 60

    def test_reproducible_with_seed(self):
        d1 = load_motor_frequency(T=40, seed=42)
        d2 = load_motor_frequency(T=40, seed=42)
        np.testing.assert_array_equal(d1.y, d2.y)

    def test_different_seed_different_data(self):
        d1 = load_motor_frequency(T=40, seed=1)
        d2 = load_motor_frequency(T=40, seed=2)
        assert not np.all(d1.y == d2.y)

    def test_trend_break_flag(self):
        d_break = load_motor_frequency(T=60, seed=1, trend_break=True)
        d_no_break = load_motor_frequency(T=60, seed=1, trend_break=False)
        # Series with a trend break should have higher claims in second half
        assert d_break.y[40:].mean() != d_no_break.y[40:].mean()

    def test_params_dict(self):
        d = load_motor_frequency()
        assert "omega_mean" in d.params
        assert "alpha_mean_1" in d.params
        assert "phi_mean_1" in d.params

    def test_description_is_string(self):
        d = load_motor_frequency()
        assert isinstance(d.description, str)
        assert len(d.description) > 0


class TestSeverityTrend:
    def test_returns_dataset(self):
        d = load_severity_trend()
        assert isinstance(d, SyntheticDataset)

    def test_length(self):
        for T in [20, 40, 80]:
            d = load_severity_trend(T=T)
            assert len(d.y) == T

    def test_y_positive(self):
        d = load_severity_trend(T=40)
        assert np.all(d.y > 0)

    def test_no_exposure(self):
        d = load_severity_trend()
        assert d.exposure is None

    def test_upward_trend(self):
        """With positive inflation rate, mean should rise over time."""
        d = load_severity_trend(T=40, inflation_rate=0.05)
        mean_first_half = d.filter_truth["mean"][:20].mean()
        mean_second_half = d.filter_truth["mean"][20:].mean()
        assert mean_second_half > mean_first_half

    def test_inflation_rate_effect(self):
        d_low = load_severity_trend(T=40, seed=1, inflation_rate=0.01)
        d_high = load_severity_trend(T=40, seed=1, inflation_rate=0.10)
        # High inflation should produce higher final mean
        assert d_high.filter_truth["mean"][-1] > d_low.filter_truth["mean"][-1]

    def test_reproducible(self):
        d1 = load_severity_trend(T=30, seed=5)
        d2 = load_severity_trend(T=30, seed=5)
        np.testing.assert_allclose(d1.y, d2.y)


class TestLossRatio:
    def test_returns_dataset(self):
        d = load_loss_ratio()
        assert isinstance(d, SyntheticDataset)

    def test_y_in_unit_interval(self):
        d = load_loss_ratio(T=48)
        assert np.all(d.y > 0)
        assert np.all(d.y < 1)

    def test_length(self):
        d = load_loss_ratio(T=48)
        assert len(d.y) == 48

    def test_filter_truth_in_unit_interval(self):
        d = load_loss_ratio()
        assert np.all(d.filter_truth["mean"] > 0)
        assert np.all(d.filter_truth["mean"] < 1)

    def test_no_exposure(self):
        d = load_loss_ratio()
        assert d.exposure is None
