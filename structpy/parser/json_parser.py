import importlib
import json
from typing import Any, Dict, Type

from ..core.product import StructuralProduct


class ProductFactory:
    _registry: Dict[str, Type[StructuralProduct]] = {}

    @classmethod
    def register(cls, type_name: str, product_cls: Type[StructuralProduct]) -> None:
        cls._registry[type_name] = product_cls

    @classmethod
    def create(cls, config: Dict[str, Any]) -> StructuralProduct:
        meta = config.get("meta", {})
        product_type = meta.get("type")
        if not product_type:
            raise ValueError("Product config must include meta.type")

        if product_type not in cls._registry:
            # Try to load built-in product registrations lazily.
            importlib.import_module("structpy.products")

        if product_type not in cls._registry:
            raise ValueError(f"Unknown product type: {product_type}")

        product_class = cls._registry[product_type]
        return product_class.from_config(config)


def load_product(json_str: str) -> StructuralProduct:
    data = json.loads(json_str)
    return ProductFactory.create(data)


def parse_dict_config(config: Dict[str, Any]) -> StructuralProduct:
    return ProductFactory.create(config)
