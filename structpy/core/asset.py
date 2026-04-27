from dataclasses import dataclass

@dataclass(frozen=True)
class Asset:
    """Класс, представляющий базовый актив."""
    ticker: str
    currency: str = "RUB"
    exchange: str = "MOEX"