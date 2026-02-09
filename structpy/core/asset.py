from dataclasses import dataclass

@dataclass(frozen=True)
class Asset:
    ticker: str
    currency: str = "RUB"
    exchange: str = "MOEX"