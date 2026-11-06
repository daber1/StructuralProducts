import json
from pathlib import Path

import numpy as np

from structpy.core.pricer import Pricer
from structpy.core.scenario import ScenarioGenerator
from structpy.parser import parse_dict_config


EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "examples" / "market_structured_notes.json"


def test_market_structured_note_examples_parse_and_price():
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))

    assert examples

    for example in examples:
        product = parse_dict_config(example["product"])
        market_params = example["market_params"]
        initial_prices = market_params["initial_prices"]
        product.set_initial_prices(initial_prices)

        generator = ScenarioGenerator(
            tickers=[asset.ticker for asset in product.underlyings],
            initial_prices=initial_prices,
            volatilities=market_params["volatilities"],
            risk_free_rate=market_params["risk_free_rate"],
            correlation_matrix=np.array(market_params["correlation_matrix"], dtype=float),
            dividend_yields=market_params["dividend_yields"],
        )

        result = Pricer(product, generator).price(
            n_scenarios=250,
            n_steps=market_params["n_steps"],
            T=market_params["T"],
            seed=42,
        )

        assert product.isin
        assert result["fair_value"] > 0.0
        assert result["std_error"] >= 0.0
