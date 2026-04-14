from __future__ import annotations

import copy
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

from structpy.core.pricer import Pricer
from structpy.core.scenario import ScenarioGenerator
from structpy.parser import parse_dict_config


FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
OUTPUT_DIR = Path("artifacts")
CSV_OUTPUT = OUTPUT_DIR / "sp500_barrier_note_backtest.csv"
PNG_OUTPUT = OUTPUT_DIR / "sp500_barrier_note_backtest.png"


@dataclass(frozen=True)
class BacktestConfig:
    launch_date: str = "2023-01-03"
    maturity_date: str = "2026-01-05"
    nominal: float = 1000.0
    capital_barrier: float = 0.70
    n_scenarios: int = 25000
    pricing_seed: int = 42


def _fetch_fred_series(series_id: str) -> pd.Series:
    response = requests.get(FRED_BASE_URL, params={"id": series_id}, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    return df.set_index("observation_date")[series_id].rename(series_id)


def _load_market_data() -> pd.DataFrame:
    data = pd.concat(
        [
            _fetch_fred_series("SP500"),
            _fetch_fred_series("VIXCLS"),
            _fetch_fred_series("DGS3MO"),
        ],
        axis=1,
        sort=True,
    ).sort_index()

    data = data.ffill().dropna(subset=["SP500", "VIXCLS", "DGS3MO"])
    return data


def _product_config(config: BacktestConfig) -> dict:
    return {
        "meta": {
            "isin": "HIST-SP500-BARRIER",
            "name": "SP500 Barrier Note 70%",
            "type": "PhoenixAutocall",
        },
        "params": {
            "nominal": config.nominal,
            "currency": "USD",
            "underlyings": ["SP500"],
            "feature_type": "standard",
            "coupon_rate": 0.0,
            "memory_effect": False,
            "coupon_frequency": 1,
            "barriers": {
                "autocall": 10.0,
                "coupon": 10.0,
                "capital_protection": config.capital_barrier,
            },
        },
    }


def _build_monthly_schedule(data: pd.DataFrame, launch_date: pd.Timestamp, maturity_date: pd.Timestamp) -> pd.DatetimeIndex:
    window = data.loc[launch_date:maturity_date]
    monthly = window.groupby(pd.Grouper(freq="MS")).head(1).index
    if monthly.empty or monthly[0] != launch_date:
        monthly = monthly.insert(0, launch_date)
    if monthly[-1] != maturity_date:
        monthly = monthly.insert(len(monthly), maturity_date)
    return pd.DatetimeIndex(monthly.unique()).sort_values()


def _price_on_date(
    *,
    product_config: dict,
    launch_spot: float,
    valuation_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
    current_spot: float,
    risk_free_rate: float,
    volatility: float,
    nominal: float,
    n_scenarios: int,
    seed: int,
) -> float:
    product = parse_dict_config(copy.deepcopy(product_config))
    product.set_initial_prices({"SP500": launch_spot})

    remaining_years = max((maturity_date - valuation_date).days / 365.0, 0.0)
    if remaining_years == 0.0:
        return float(product.evaluate([{"SP500": current_spot}])["payout"])

    generator = ScenarioGenerator(
        tickers=["SP500"],
        initial_prices={"SP500": current_spot},
        volatilities={"SP500": max(volatility, 0.01)},
        risk_free_rate=risk_free_rate,
    )
    pricer = Pricer(product, generator)
    result = pricer.price(
        n_scenarios=n_scenarios,
        n_steps=1,
        T=remaining_years,
        seed=seed,
    )
    return float(result["fair_value"])


def run_backtest(config: BacktestConfig) -> pd.DataFrame:
    market_data = _load_market_data()
    launch_date = pd.Timestamp(config.launch_date)
    maturity_date = pd.Timestamp(config.maturity_date)

    launch_row = market_data.loc[launch_date]
    launch_spot = float(launch_row["SP500"])
    product_config = _product_config(config)
    valuation_dates = _build_monthly_schedule(market_data, launch_date, maturity_date)

    rows = []
    for valuation_date in valuation_dates:
        market_row = market_data.loc[valuation_date]
        current_spot = float(market_row["SP500"])
        risk_free_rate = float(market_row["DGS3MO"]) / 100.0
        volatility = float(market_row["VIXCLS"]) / 100.0
        fair_value = _price_on_date(
            product_config=product_config,
            launch_spot=launch_spot,
            valuation_date=valuation_date,
            maturity_date=maturity_date,
            current_spot=current_spot,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            nominal=config.nominal,
            n_scenarios=config.n_scenarios,
            seed=config.pricing_seed,
        )
        rows.append(
            {
                "valuation_date": valuation_date,
                "underlying_spot": current_spot,
                "underlying_normalized": config.nominal * current_spot / launch_spot,
                "fair_value": fair_value,
                "remaining_years": max((maturity_date - valuation_date).days / 365.0, 0.0),
                "risk_free_rate": risk_free_rate,
                "volatility": volatility,
            }
        )

    result = pd.DataFrame(rows).set_index("valuation_date")
    return result


def _plot_backtest(result: pd.DataFrame, config: BacktestConfig) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        result.index,
        result["fair_value"],
        label="Model Fair Value",
        color="#0b5cad",
        linewidth=2.5,
    )
    ax.plot(
        result.index,
        result["underlying_normalized"],
        label="SP500 Normalized To Nominal",
        color="#d97706",
        linewidth=2.0,
        linestyle="--",
    )
    ax.axhline(
        config.nominal * config.capital_barrier,
        color="#b91c1c",
        linewidth=1.5,
        linestyle=":",
        label="70% Barrier Level",
    )
    ax.axhline(config.nominal, color="#475569", linewidth=1.0, linestyle="-.", label="Nominal")

    ax.set_title("Historical Repricing Of A 70% Barrier Note On SP500")
    ax.set_ylabel("USD")
    ax.set_xlabel("Valuation Date")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUTPUT, dpi=160)
    plt.close(fig)


def main() -> None:
    config = BacktestConfig()
    result = run_backtest(config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(CSV_OUTPUT, index=True)
    _plot_backtest(result, config)

    print(f"Launch date: {config.launch_date}")
    print(f"Maturity date: {config.maturity_date}")
    print(f"Launch SP500 level: {result['underlying_spot'].iloc[0]:.2f}")
    print(f"Final SP500 level: {result['underlying_spot'].iloc[-1]:.2f}")
    print(f"Initial fair value: {result['fair_value'].iloc[0]:.2f}")
    print(f"Final fair value: {result['fair_value'].iloc[-1]:.2f}")
    print(f"Min fair value: {result['fair_value'].min():.2f}")
    print(f"Max fair value: {result['fair_value'].max():.2f}")
    print(f"CSV saved to: {CSV_OUTPUT}")
    print(f"Chart saved to: {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
