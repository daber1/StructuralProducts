import pytest


def _base_config() -> dict:
    return {
        "meta": {
            "isin": "TEST",
            "name": "Test Phoenix",
            "type": "PhoenixAutocall",
        },
        "params": {
            "nominal": 1000.0,
            "currency": "RUB",
            "underlyings": ["ASSET1"],
            "feature_type": "standard",
            "coupon_rate": 0.12,
            "memory_effect": True,
            "coupon_frequency": 4,
            "barriers": {
                "autocall": 1.0,
                "coupon": 0.8,
                "capital_protection": 0.7,
            },
        },
    }


def test_evaluate_requires_path_of_observations(make_product):
    product = make_product(_base_config(), {"ASSET1": 100.0})

    with pytest.raises(TypeError):
        product.evaluate({"ASSET1": 100.0})


def test_evaluate_applies_memory_coupon_and_autocall(make_product):
    product = make_product(_base_config(), {"ASSET1": 100.0})

    result = product.evaluate(
        [
            {"ASSET1": 70.0},
            {"ASSET1": 90.0},
            {"ASSET1": 105.0},
            {"ASSET1": 105.0},
        ]
    )

    assert result["action"] == "autocall"
    assert result["step"] == 3
    assert result["cashflows"] == [(2, 60.0), (3, 30.0), (3, 1000.0)]
    assert result["payout"] == pytest.approx(1090.0)
    assert result["perf_worst"] == pytest.approx(1.05)
    assert result["perf_avg"] == pytest.approx(1.05)


def test_evaluate_returns_capital_loss_when_protection_is_breached(make_product):
    config = _base_config()
    config["params"]["coupon_rate"] = 0.0
    config["params"]["memory_effect"] = False
    config["params"]["barriers"] = {
        "autocall": 10.0,
        "coupon": 10.0,
        "capital_protection": 0.9,
    }
    product = make_product(config, {"ASSET1": 100.0})

    result = product.evaluate([{"ASSET1": 70.0}])

    assert result["action"] == "maturity"
    assert result["cashflows"] == [(1, 700.0)]
    assert result["payout"] == pytest.approx(700.0)


def test_evaluate_adds_upside_participation_above_strike(make_product):
    config = _base_config()
    config["params"]["coupon_rate"] = 0.0
    config["params"]["memory_effect"] = False
    config["params"]["participation_rate"] = 1.5
    config["params"]["strike"] = 1.0
    config["params"]["barriers"] = {
        "autocall": 10.0,
        "coupon": 10.0,
        "capital_protection": 1.0,
    }
    product = make_product(config, {"ASSET1": 100.0})

    result = product.evaluate([{"ASSET1": 120.0}])

    assert result["action"] == "maturity"
    assert result["cashflows"] == [(1, 1300.0)]
    assert result["payout"] == pytest.approx(1300.0)
