"""Tests for the GAS filter recursion."""

import numpy as np
import pytest

from insurance_gas.distributions import PoissonGAS, GammaGAS, LogNormalGAS
from insurance_gas.filter import GASFilter, FilterResult


class TestGASFilterBasic:
    """Test the filter forward pass with known inputs."""

    def setup_method(self):
        self.dist = PoissonGAS()
        self.filt = GASFilter(
            distribution=self.dist,
            time_varying=["mean"],
            scaling="fisher_inv",
            p=1,
            q=1,
        )

    def _simple_params(self, omega=0.0, alpha=0.1, phi=0.9):
        return {"omega_mean": omega, "alpha_mean_1": alpha, "phi_mean_1": phi}

    def test_output_length(self):
        """Filter path length should equal input length."""
        T = 20
        y = np.ones(T) * 2.0
        gas_params = self._simple_params()
        r = self.filt.run(y, gas_params, static_params={})
        assert len(r.filter_paths["mean"]) == T

    def test_log_likelihoods_length(self):
        T = 15
        y = np.ones(T) * 2.0
        r = self.filt.run(y, self._simple_params(), {})
        assert len(r.log_likelihoods) == T

    def test_filter_stays_positive(self):
        """Mean filter path must remain positive."""
        rng = np.random.default_rng(42)
        y = rng.poisson(3.0, 50).astype(float)
        r = self.filt.run(y, self._simple_params(alpha=0.2, phi=0.8), {})
        assert np.all(r.filter_paths["mean"] > 0)

    def test_constant_input_stationary(self):
        """With constant y = mu, filter should stay near initial value."""
        mu0 = 3.0
        y = np.full(100, mu0)
        # omega / (1 - phi) = initial f
        omega = np.log(mu0) * 0.1
        phi = 0.9
        gas_params = {"omega_mean": omega, "alpha_mean_1": 0.1, "phi_mean_1": phi}
        r = self.filt.run(y, gas_params, {})
        # After burn-in, filter mean should be close to unconditional value
        assert np.all(np.abs(r.filter_paths["mean"] - mu0) < 2.0)

    def test_unconditional_mean_initial_condition(self):
        """Default f0 = omega/(1-phi)."""
        omega = 0.2
        phi = 0.8
        expected_f0 = omega / (1.0 - phi)  # = 1.0
        gas_params = {"omega_mean": omega, "alpha_mean_1": 0.1, "phi_mean_1": phi}
        y = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        r = self.filt.run(y, gas_params, {})
        # First period uses f0
        f0_path = r.f_paths["mean"][0]
        assert f0_path == pytest.approx(expected_f0, rel=0.01)

    def test_explicit_initial_condition(self):
        """Custom f0 should be respected."""
        gas_params = self._simple_params()
        y = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        custom_f0 = {"mean": 1.5}
        r = self.filt.run(y, gas_params, {}, f0=custom_f0)
        assert r.f_paths["mean"][0] == pytest.approx(1.5)

    def test_log_likelihoods_finite(self):
        rng = np.random.default_rng(99)
        y = rng.poisson(2.0, 30).astype(float)
        r = self.filt.run(y, self._simple_params(), {})
        assert np.all(np.isfinite(r.log_likelihoods))

    def test_score_residuals_length(self):
        y = np.ones(20) * 2.0
        r = self.filt.run(y, self._simple_params(), {})
        assert len(r.score_residuals["mean"]) == 20

    def test_score_residuals_zero_at_mean(self):
        """With constant data at the fitted mean and alpha→0, score residuals are 0."""
        omega = np.log(2.0) * 0.001
        phi = 0.999
        gas_params = {"omega_mean": omega, "alpha_mean_1": 0.001, "phi_mean_1": phi}
        y = np.full(30, 2.0)
        r = self.filt.run(y, gas_params, {})
        assert np.abs(np.mean(r.score_residuals["mean"])) < 0.5

    def test_exposure_weighted_filter(self):
        """Filter with exposure should use effective rate mu*E.

        When y = mu*E exactly (y=6, mu=2, E=3), the score y/(mu*E) - 1 = 0,
        so the filter should not move. We set f0 explicitly so mu starts at 2.
        """
        y = np.full(20, 6.0)
        exposure = np.full(20, 3.0)
        gas_params = {"omega_mean": 0.0, "alpha_mean_1": 0.3, "phi_mean_1": 0.9}
        # f0 = log(2) so mu starts at exactly 2
        f0 = {"mean": np.log(2.0)}
        r = self.filt.run(y, gas_params, {}, exposure=exposure, f0=f0)
        # With y/rate = 6/(2*3) = 1 at every step, score = 0
        np.testing.assert_allclose(
            r.score_residuals["mean"], np.zeros(20), atol=1e-10
        )


class TestGASFilterOrders:
    """Test GAS(p,q) with different lag orders."""

    def test_gas_p2_q1_output_length(self):
        dist = PoissonGAS()
        filt = GASFilter(dist, ["mean"], p=2, q=1)
        y = np.ones(30) * 2.0
        gas_params = {
            "omega_mean": 0.1,
            "alpha_mean_1": 0.1,
            "alpha_mean_2": 0.05,
            "phi_mean_1": 0.8,
        }
        r = filt.run(y, gas_params, {})
        assert len(r.filter_paths["mean"]) == 30

    def test_gas_p1_q2_output_length(self):
        dist = PoissonGAS()
        filt = GASFilter(dist, ["mean"], p=1, q=2)
        y = np.ones(30) * 2.0
        gas_params = {
            "omega_mean": 0.05,
            "alpha_mean_1": 0.1,
            "phi_mean_1": 0.7,
            "phi_mean_2": 0.1,
        }
        r = filt.run(y, gas_params, {})
        assert len(r.filter_paths["mean"]) == 30

    def test_filter_result_dataclass(self):
        dist = PoissonGAS()
        filt = GASFilter(dist, ["mean"])
        y = np.ones(10) * 2.0
        r = filt.run(y, {"omega_mean": 0.1, "alpha_mean_1": 0.1, "phi_mean_1": 0.9}, {})
        assert isinstance(r, FilterResult)
        assert "mean" in r.filter_paths
        assert "mean" in r.f_paths


class TestGASFilterGamma:
    """Filter with Gamma distribution."""

    def test_gamma_filter_positive(self):
        dist = GammaGAS()
        filt = GASFilter(dist, ["mean"], scaling="fisher_inv")
        rng = np.random.default_rng(1)
        y = rng.gamma(shape=3.0, scale=200.0, size=40)
        gas_params = {
            "omega_mean": 0.02,
            "alpha_mean_1": 0.1,
            "phi_mean_1": 0.85,
        }
        r = filt.run(y, gas_params, static_params={"shape": 3.0})
        assert np.all(r.filter_paths["mean"] > 0)
        assert np.all(np.isfinite(r.log_likelihoods))

    def test_gamma_filter_tracks_trend(self):
        """Filter should increase when observations are consistently above initial mean."""
        dist = GammaGAS()
        filt = GASFilter(dist, ["mean"], scaling="fisher_inv")
        # All observations at 1000, initial mu will be around 300 → filter should rise
        y = np.full(50, 1000.0)
        gas_params = {"omega_mean": 0.05, "alpha_mean_1": 0.2, "phi_mean_1": 0.8}
        r = filt.run(y, gas_params, static_params={"shape": 2.0})
        # Last mean should be higher than first
        assert r.filter_paths["mean"][-1] > r.filter_paths["mean"][0]


class TestFilterScalingOptions:
    """Check all three scaling options produce different results."""

    def _run_filter(self, scaling: str):
        dist = PoissonGAS()
        filt = GASFilter(dist, ["mean"], scaling=scaling)
        rng = np.random.default_rng(42)
        y = rng.poisson(3.0, 30).astype(float)
        gas_params = {"omega_mean": 0.05, "alpha_mean_1": 0.1, "phi_mean_1": 0.85}
        return filt.run(y, gas_params, {})

    def test_unit_runs(self):
        r = self._run_filter("unit")
        assert r.filter_paths["mean"] is not None

    def test_fisher_inv_runs(self):
        r = self._run_filter("fisher_inv")
        assert r.filter_paths["mean"] is not None

    def test_fisher_inv_sqrt_runs(self):
        r = self._run_filter("fisher_inv_sqrt")
        assert r.filter_paths["mean"] is not None

    def test_scalings_differ(self):
        r_unit = self._run_filter("unit")
        r_fi = self._run_filter("fisher_inv")
        # Different scaling → different filter paths
        assert not np.allclose(r_unit.filter_paths["mean"], r_fi.filter_paths["mean"])

    def test_invalid_scaling_raises(self):
        dist = PoissonGAS()
        filt = GASFilter(dist, ["mean"], scaling="bad_option")
        with pytest.raises(ValueError, match="Unknown scaling"):
            filt.run(
                np.array([1.0, 2.0, 3.0, 1.0]),
                {"omega_mean": 0.1, "alpha_mean_1": 0.1, "phi_mean_1": 0.8},
                {},
            )
