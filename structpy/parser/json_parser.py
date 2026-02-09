import json
from typing import Type
from ..core.product import StructuralProduct
from ..core.asset import Asset

class ProductFactory:
    _registry = {}

    @classmethod
    def register(cls, type_name: str, product_cls: Type[StructuralProduct]):
        cls._registry[type_name] = product_cls

    @classmethod
    def create(cls, config: dict) -> StructuralProduct:
        p_type = config['meta']['type']
        if p_type not in cls._registry:
            raise ValueError(f"Unknown product type: {p_type}")
        
        product_class = cls._registry[p_type]
        # Здесь логика распаковки параметров специфична для каждого класса
        # Для простоты передадим config целиком в конструктор
        return product_class.from_config(config)

def load_product(json_str: str) -> StructuralProduct:
    data = json.loads(json_str)
    return ProductFactory.create(data)