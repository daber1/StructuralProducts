from datetime import date

import pandas as pd

from structpy.data.mock_provider import MockDataProvider
from structpy.data.moex import MoexDataProvider


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_mock_provider_returns_current_prices_and_history():
    provider = MockDataProvider({"SBER": 280.0})

    prices = provider.get_current_prices(["SBER", "GAZP"])
    history = provider.get_history(["SBER"], date(2024, 1, 1), date(2024, 1, 2))

    assert prices == {"SBER": 280.0, "GAZP": 100.0}
    assert list(history["SBER"]["CLOSE"]) == [280.0, 280.0]


def test_moex_provider_collects_paginated_history(monkeypatch):
    provider = MoexDataProvider()
    payloads = [
        {
            "history": {
                "columns": ["TRADEDATE", "CLOSE"],
                "data": [["2024-01-01", "100.5"], ["2024-01-02", "101.0"]],
            }
        },
        {"history": {"columns": ["TRADEDATE", "CLOSE"], "data": []}},
    ]

    def fake_get(url, params, timeout):
        return _FakeResponse(payloads.pop(0))

    monkeypatch.setattr("structpy.data.moex.requests.get", fake_get)
    monkeypatch.setattr("structpy.data.moex.time.sleep", lambda _: None)

    data = provider.get_history(["SBER"], date(2024, 1, 1), date(2024, 1, 2))
    frame = data["SBER"]

    assert list(frame.columns) == ["CLOSE"]
    assert frame["CLOSE"].tolist() == [100.5, 101.0]
    assert isinstance(frame.index, pd.DatetimeIndex)


def test_moex_provider_uses_last_close_for_current_prices(monkeypatch):
    provider = MoexDataProvider()
    frame = pd.DataFrame(
        {"CLOSE": [100.0, 101.5]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    monkeypatch.setattr(
        provider,
        "get_history",
        lambda tickers, start_date, end_date: {"SBER": frame},
    )

    prices = provider.get_current_prices(["SBER"])

    assert prices == {"SBER": 101.5}
