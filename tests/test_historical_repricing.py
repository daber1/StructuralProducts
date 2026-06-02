import pandas as pd
import pytest

from historical_repricing_sp500_barrier import _build_daily_schedule, _resolve_market_date


def test_resolve_market_date_returns_exact_match():
    index = pd.to_datetime(["2024-06-28", "2024-07-01", "2024-07-02"])

    resolved = _resolve_market_date(index, pd.Timestamp("2024-07-01"), direction="next")

    assert resolved == pd.Timestamp("2024-07-01")


def test_resolve_market_date_snaps_forward_for_launch_dates():
    index = pd.to_datetime(["2024-06-28", "2024-07-01", "2024-07-02"])

    resolved = _resolve_market_date(index, pd.Timestamp("2024-06-29"), direction="next")

    assert resolved == pd.Timestamp("2024-07-01")


def test_resolve_market_date_snaps_backward_for_maturity_dates():
    index = pd.to_datetime(["2029-06-27", "2029-06-28", "2029-06-29"])

    resolved = _resolve_market_date(
        index,
        pd.Timestamp("2029-06-30"),
        direction="previous",
    )

    assert resolved == pd.Timestamp("2029-06-29")


def test_resolve_market_date_rejects_unsupported_direction():
    index = pd.to_datetime(["2024-06-28"])

    with pytest.raises(ValueError):
        _resolve_market_date(index, pd.Timestamp("2024-06-28"), direction="nearest")


def test_build_daily_schedule_returns_all_trading_dates_in_window():
    index = pd.to_datetime(
        [
            "2024-06-28",
            "2024-07-01",
            "2024-07-02",
            "2024-07-03",
        ]
    )
    data = pd.DataFrame({"SP500": [1.0, 2.0, 3.0, 4.0]}, index=index)

    schedule = _build_daily_schedule(
        data,
        pd.Timestamp("2024-06-28"),
        pd.Timestamp("2024-07-03"),
    )

    assert list(schedule) == list(index)
