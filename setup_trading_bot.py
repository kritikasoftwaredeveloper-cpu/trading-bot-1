import os

# ---------------- Folder ----------------
folder = "trading-bot"
os.makedirs(folder, exist_ok=True)

# ---------------- requirements.txt ----------------
with open(os.path.join(folder, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write("""alpaca-trade-api
pandas
numpy
pydantic
python-dotenv
requests
yfinance
""")

# ---------------- .env.example ----------------
with open(os.path.join(folder, ".env.example"), "w", encoding="utf-8") as f:
    f.write("""ALPACA_KEY_ID=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
DRY_RUN=true
""")

# ---------------- README.md ----------------
with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
    f.write("""# Trading Bot (SMA Crossover)

🚀 Simple Python trading bot for Alpaca or DRY_RUN.

## Setup (Local)
pip install -r requirements.txt
python auto_trader.py
""")

# ---------------- auto_trader.py ----------------
with open(os.path.join(folder, "auto_trader.py"), "w", encoding="utf-8") as f:
    f.write("""from __future__ import annotations
import os
import time
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from alpaca_trade_api import REST
    from alpaca_trade_api.rest import TimeFrame
except Exception:
    REST = None
    TimeFrame = None

# ---------------- Settings ----------------
class Settings(BaseModel):
    symbol: str = Field(default="AAPL")
    timeframe: str = Field(default="5Min")
    fast_sma: int = 50
    slow_sma: int = 200
    poll_seconds: int = 60
    invest_pct: float = 0.10
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.01
    max_position_qty: int = 100
    dry_run: bool = Field(default=os.getenv("DRY_RUN", "false").lower() == "true")

    alpaca_key_id: Optional[str] = Field(default=os.getenv("ALPACA_KEY_ID"))
    alpaca_secret_key: Optional[str] = Field(default=os.getenv("ALPACA_SECRET_KEY"))
    alpaca_base_url: Optional[str] = Field(default=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))

# ---------------- Logging ----------------
logger = logging.getLogger("trader")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

fh = RotatingFileHandler("trader.log", maxBytes=1_000_000, backupCount=3)
fh.setFormatter(fmt)
logger.addHandler(fh)

# ---------------- Broker ----------------
@dataclass
class Broker:
    rest: Optional[REST]
    settings: Settings

    @classmethod
    def create(cls, settings: Settings) -> "Broker":
        if settings.dry_run:
            logger.info("DRY_RUN enabled")
            return cls(rest=None, settings=settings)
        if REST is None:
            raise RuntimeError("Install alpaca-trade-api first")
        if not settings.alpaca_key_id or not settings.alpaca_secret_key:
            raise RuntimeError("Missing Alpaca API keys")
        rest = REST(key_id=settings.alpaca_key_id,
                    secret_key=settings.alpaca_secret_key,
                    base_url=settings.alpaca_base_url)
        return cls(rest=rest, settings=settings)

    def get_bars(self, symbol: str, limit: int = 400) -> pd.DataFrame:
        if self.rest is None:
            import yfinance as yf
            tf_map = {"1Min":"1m","5Min":"5m","15Min":"15m","1Hour":"60m","1Day":"1d"}
            interval = tf_map.get(self.settings.timeframe, "5m")
            df = yf.download(symbol, period="30d", interval=interval, progress=False)
            if df.empty: raise RuntimeError("No data from yfinance")
            df = df.reset_index().rename(columns={"Date":"timestamp"})
            df.columns = [c.lower() for c in df.columns]
            return df.tail(limit)
        tf = {"1Min":TimeFrame.Minute,
              "5Min":TimeFrame(5,"Min"),
              "15Min":TimeFrame(15,"Min"),
              "1Hour":TimeFrame(1,"Hour"),
              "1Day":TimeFrame.Day}.get(self.settings.timeframe, TimeFrame(5,"Min"))
        bars = self.rest.get_bars(symbol, tf, limit=limit)
        df = bars.df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        return df

    def get_equity(self) -> float:
        return 100_000.0 if self.rest is None else float(self.rest.get_account().equity)

    def get_position_qty(self, symbol: str) -> int:
        if self.rest is None: return 0
        try:
            return int(float(self.rest.get_position(symbol).qty))
        except: return 0

    def submit_bracket_order(self, symbol: str, qty: int, side: str, take_profit: float, stop_loss: float):
        if qty <= 0: return None
        if self.rest is None:
            logger.info(f"[DRY_RUN] {side.upper()} {qty} {symbol} TP={take_profit} SL={stop_loss}")
            return None
        order = self.rest.submit_order(symbol=symbol, qty=qty, side=side, type='market',
                                       time_in_force='gtc', order_class='bracket',
                                       take_profit={'limit_price': take_profit},
                                       stop_loss={'stop_price': stop_loss})
        logger.info(f"Order submitted id={order.id}")
        return order

    def close_position(self, symbol: str):
        if self.rest is None:
            logger.info(f"[DRY_RUN] CLOSE {symbol}")
            return None
        try:
            order = self.rest.close_position(symbol)
            logger.info(f"Close order id={order.id}")
            return order
        except Exception as e:
            logger.warning(f"Close position failed: {e}")
            return None

# ---------------- Strategy ----------------
@dataclass
class Signal:
    side: Optional[str]
    price: Optional[float]

def compute_signals(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    x['sma_fast'] = x['close'].rolling(fast).mean()
    x['sma_slow'] = x['close'].rolling(slow).mean()
    x['signal'] = 0
    cross_up = (x['sma_fast'] > x['sma_slow']) & (x['sma_fast'].shift(1) <= x['sma_slow'].shift(1))
    cross_dn = (x['sma_fast'] < x['sma_slow']) & (x['sma_fast'].shift(1) >= x['sma_slow'].shift(1))
    x.loc[cross_up, 'signal'] = 1
    x.loc[cross_dn, 'signal'] = -1
    return x

def size_position(equity: float, price: float, invest_pct: float, max_qty: int) -> int:
    raw = int((equity * invest_pct) // max(price, 1e-6))
    return max(0, min(raw, max_qty))

def next_action(broker: Broker, df: pd.DataFrame, settings: Settings) -> Optional[Signal]:
    if df.empty: return None
    last = df.iloc[-1]
    price = float(last['close'])
    df_sig = compute_signals(df, settings.fast_sma, settings.slow_sma)
    last_sig = int(df_sig.iloc[-1]['signal'])
    position_qty = broker.get_position_qty(settings.symbol)
    if last_sig == 1 and position_qty == 0: return Signal('buy', price)
    elif last_sig == -1 and position_qty > 0: return Signal('sell', price)
    else: return Signal(None, price)

# ---------------- Main Loop ----------------
def main():
    settings = Settings()
    logger.info(f"Starting bot | {settings.symbol} dry_run={settings.dry_run}")
    broker = Broker.create(settings)
    while True:
        try:
            bars = broker.get_bars(settings.symbol, limit=max(settings.slow_sma*3,400))
            sig = next_action(broker, bars, settings)
            if sig is None:
                logger.info("No data yet; sleeping...")
                time.sleep(settings.poll_seconds)
                continue
            equity = broker.get_equity()
            qty = size_position(equity, sig.price or 0.0, settings.invest_pct, settings.max_position_qty)
            if sig.side == 'buy' and qty > 0:
                tp = round(sig.price * (1+settings.take_profit_pct),2)
                sl = round(sig.price * (1-settings.stop_loss_pct),2)
                broker.submit_bracket_order(settings.symbol, qty, 'buy', tp, sl)
            elif sig.side == 'sell':
                broker.close_position(settings.symbol)
            else:
                logger.info("No action")
        except Exception as e:
            logger.exception(e)
        time.sleep(settings.poll_seconds)

if __name__ == "__main__":
    main()
""")

print(f"All files created in ./{folder}/")
with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
    f.write("""# Trading Bot (SMA Crossover)

Simple Python trading bot for Alpaca or DRY_RUN.
...
""")