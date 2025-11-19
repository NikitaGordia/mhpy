import pytest

from mhpy.utils.metrics import EMA


class TestEMA:
    def test_ema_initialization_default(self):
        ema = EMA()
        assert ema.alpha == 0.1
        assert ema.values == {}

    def test_ema_initialization_custom_alpha(self):
        alpha = 0.5
        ema = EMA(alpha=alpha)
        assert ema.alpha == alpha
        assert ema.values == {}

    def test_ema_update_first_value(self):
        ema = EMA(alpha=0.1)
        metric_dict = {"loss": 1.0, "accuracy": 0.9}

        result = ema.update(metric_dict)

        assert result == {"loss": 1.0, "accuracy": 0.9}
        assert ema.values == {"loss": 1.0, "accuracy": 0.9}

    def test_ema_update_subsequent_values(self):
        ema = EMA(alpha=0.1)

        ema.update({"loss": 1.0})

        result = ema.update({"loss": 2.0})

        # Expected: 0.1 * 2.0 + 0.9 * 1.0 = 1.1
        assert result["loss"] == pytest.approx(1.1)
        assert ema.values["loss"] == pytest.approx(1.1)

    def test_ema_update_multiple_metrics(self):
        ema = EMA(alpha=0.2)

        ema.update({"loss": 1.0, "accuracy": 0.8, "f1": 0.75})

        result = ema.update({"loss": 0.5, "accuracy": 0.9, "f1": 0.85})

        # Expected values:
        # loss: 0.2 * 0.5 + 0.8 * 1.0 = 0.9
        # accuracy: 0.2 * 0.9 + 0.8 * 0.8 = 0.82
        # f1: 0.2 * 0.85 + 0.8 * 0.75 = 0.77
        assert result["loss"] == pytest.approx(0.9)
        assert result["accuracy"] == pytest.approx(0.82)
        assert result["f1"] == pytest.approx(0.77)

    def test_ema_update_new_metric_added(self):
        ema = EMA(alpha=0.1)

        ema.update({"loss": 1.0})

        result = ema.update({"loss": 2.0, "accuracy": 0.9})

        assert "accuracy" in result
        assert result["accuracy"] == 0.9  # First value, no smoothing
        assert result["loss"] == pytest.approx(1.1)  # Smoothed value

    def test_ema_get(self):
        """Test EMA get method."""
        ema = EMA(alpha=0.1)

        assert ema.get() == {}

        ema.update({"loss": 1.0, "accuracy": 0.9})
        ema.update({"loss": 0.5, "accuracy": 0.95})

        result = ema.get()
        assert "loss" in result
        assert "accuracy" in result
        assert result == ema.values

    def test_ema_alpha_zero(self):
        ema = EMA(alpha=0.0)

        ema.update({"loss": 1.0})
        result = ema.update({"loss": 2.0})

        assert result["loss"] == pytest.approx(1.0)

    def test_ema_alpha_one(self):
        ema = EMA(alpha=1.0)

        ema.update({"loss": 1.0})
        result = ema.update({"loss": 2.0})

        assert result["loss"] == pytest.approx(2.0)

    def test_ema_sequence_of_updates(self):
        ema = EMA(alpha=0.5)

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for val in values:
            ema.update({"metric": val})

        # Manually calculate expected value
        # Step 1: 1.0
        # Step 2: 0.5 * 2.0 + 0.5 * 1.0 = 1.5
        # Step 3: 0.5 * 3.0 + 0.5 * 1.5 = 2.25
        # Step 4: 0.5 * 4.0 + 0.5 * 2.25 = 3.125
        # Step 5: 0.5 * 5.0 + 0.5 * 3.125 = 4.0625

        assert ema.get()["metric"] == pytest.approx(4.0625)

    def test_ema_empty_update(self):
        ema = EMA(alpha=0.1)

        result = ema.update({})

        assert result == {}
        assert ema.values == {}
