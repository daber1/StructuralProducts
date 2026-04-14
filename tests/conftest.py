import copy

import pytest

from structpy.parser import parse_dict_config


@pytest.fixture
def make_product():
    def _make_product(config: dict, initial_prices: dict):
        product = parse_dict_config(copy.deepcopy(config))
        product.set_initial_prices(initial_prices)
        return product

    return _make_product
