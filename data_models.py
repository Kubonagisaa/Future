from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP = "TRAILING_STOP"

class OrderStatus(Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class MarketRegime(Enum):
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"

@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Candle':
        return cls(
            timestamp=data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume']
        )

@dataclass
class Position:
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    leverage: int
    stop_loss: float
    take_profit: float
    timestamp: int
    trailing_stop: float = 0.0
    last_funding_time: int = 0
    scale_in_level: int = 0
    scale_out_level: int = 0
    entry_reason: str = ""
    unrealized_pnl: float = 0.0
    id: str = field(default_factory=lambda: f"pos_{int(datetime.now().timestamp() * 1000)}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'entry_price': self.entry_price,
            'leverage': self.leverage,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'timestamp': self.timestamp,
            'last_funding_time': self.last_funding_time,
            'scale_in_level': self.scale_in_level,
            'scale_out_level': self.scale_out_level,
            'entry_reason': self.entry_reason,
            'unrealized_pnl': self.unrealized_pnl
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        return cls(
            id=data.get('id', f"pos_{int(datetime.now().timestamp() * 1000)}"),
            symbol=data['symbol'],
            side=PositionSide(data['side']),
            size=data['size'],
            entry_price=data['entry_price'],
            leverage=data['leverage'],
            stop_loss=data['stop_loss'],
            take_profit=data['take_profit'],
            trailing_stop=data.get('trailing_stop', 0.0),
            timestamp=data['timestamp'],
            last_funding_time=data.get('last_funding_time', 0),
            scale_in_level=data.get('scale_in_level', 0),
            scale_out_level=data.get('scale_out_level', 0),
            entry_reason=data.get('entry_reason', ""),
            unrealized_pnl=data.get('unrealized_pnl', 0.0)
        )

@dataclass
class Order:
    id: str
    symbol: str
    type: OrderType
    side: PositionSide
    amount: float
    price: float
    status: OrderStatus
    timestamp: int
    filled: float = 0.0
    remaining: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'type': self.type.value,
            'side': self.side.value,
            'amount': self.amount,
            'price': self.price,
            'status': self.status.value,
            'timestamp': self.timestamp,
            'filled': self.filled,
            'remaining': self.remaining,
            'params': self.params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        return cls(
            id=data['id'],
            symbol=data['symbol'],
            type=OrderType(data['type']),
            side=PositionSide(data['side']),
            amount=data['amount'],
            price=data['price'],
            status=OrderStatus(data['status']),
            timestamp=data['timestamp'],
            filled=data.get('filled', 0.0),
            remaining=data.get('remaining', 0.0),
            params=data.get('params', {})
        )

@dataclass
class Trade:
    id: str
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    timestamp: int
    duration: float
    fee: float = 0.0
    funding: float = 0.0
    slippage: float = 0.0
    strategy: str = ""
    exit_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'fee': self.fee,
            'funding': self.funding,
            'slippage': self.slippage,
            'strategy': self.strategy,
            'exit_reason': self.exit_reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        return cls(
            id=data['id'],
            symbol=data['symbol'],
            side=PositionSide(data['side']),
            size=data['size'],
            entry_price=data['entry_price'],
            exit_price=data['exit_price'],
            pnl=data['pnl'],
            pnl_percent=data['pnl_percent'],
            timestamp=data['timestamp'],
            duration=data['duration'],
            fee=data.get('fee', 0.0),
            funding=data.get('funding', 0.0),
            slippage=data.get('slippage', 0.0),
            strategy=data.get('strategy', ""),
            exit_reason=data.get('exit_reason', "")
        )

@dataclass
class PortfolioItem:
    symbol: str
    weight: float
    correlation: float
    volatility: float
    signal_strength: float

@dataclass
class MarketCondition:
    volatility: float
    trend_strength: float
    volume_ratio: float
    market_regime: MarketRegime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'volume_ratio': self.volume_ratio,
            'market_regime': self.market_regime.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketCondition':
        return cls(
            volatility=data['volatility'],
            trend_strength=data['trend_strength'],
            volume_ratio=data['volume_ratio'],
            market_regime=MarketRegime(data['market_regime'])
        )

@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    details: List[Dict[str, Any]] = field(default_factory=list)