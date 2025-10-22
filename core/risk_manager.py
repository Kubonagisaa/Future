import logging
import numpy as np
from typing import Dict, List
from scipy.stats import norm
from .config import Config
from .data_models import Position, Candle
from .exchange import FuturesExchange

logger = logging.getLogger('FuturesBot')

class RiskManager:
    def __init__(self, config: Config, exchange: FuturesExchange):
        self.config = config
        self.exchange = exchange
        self.daily_trade_count = 0
        self.last_equity = exchange.initial_balance
        self.max_equity_reached = exchange.initial_balance
        self.current_drawdown = 0
        self.consecutive_losses = 0
        self.var_cache = {}  # Cache for VaR calculations

    def update_drawdown(self, equity: float) -> None:
        """Update drawdown calculations"""
        self.current_drawdown = (self.max_equity_reached - equity) / self.max_equity_reached * 100
        if equity > self.max_equity_reached:
            self.max_equity_reached = equity

    def can_trade(self) -> bool:
        """Check if trading is allowed based on risk rules"""
        # Check daily drawdown
        if self.current_drawdown >= self.config.risk.max_daily_drawdown:
            logger.warning(f"Daily drawdown limit reached: {self.current_drawdown:.2f}%")
            return False
        
        # Check daily trade count
        if self.daily_trade_count >= self.config.trading.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.config.risk.max_consecutive_losses:
            logger.warning(f"Max consecutive losses reached: {self.consecutive_losses}")
            return False
        
        # Check circuit breaker
        if self.exchange.circuit_breaker_enabled:
            logger.warning("Circuit breaker active, trading paused")
            return False
        
        return True

    def calculate_position_size(self, current_price: float, atr: float, portfolio_weight: float = 1.0) -> float:
        """Calculate position size based on risk parameters"""
        risk_amount = self.exchange.equity * (self.config.trading.risk_per_trade / 100.0) * portfolio_weight
        stop_loss_distance = atr * self.config.risk.stop_loss_atr
        
        if stop_loss_distance == 0:
            return 0
            
        position_size = risk_amount / stop_loss_distance
        
        # Adjust for risk-off mode
        if self.config.general.risk_off_mode:
            position_size *= (self.config.general.risk_off_capital_pct / 100.0)
            
        return position_size

    def calculate_adaptive_leverage(self, atr: float, current_price: float) -> int:
        """Calculate adaptive leverage based on volatility"""
        if atr == 0:
            return self.config.trading.min_leverage
            
        vol_pct = atr / current_price * 100
        
        # Lower leverage in high volatility
        if vol_pct > 5:
            return self.config.trading.min_leverage
        elif vol_pct < 1:
            return self.config.trading.max_leverage
        else:
            # Linear interpolation between min and max leverage
            ratio = (5 - vol_pct) / 4
            leverage = self.config.trading.min_leverage + int(
                ratio * (self.config.trading.max_leverage - self.config.trading.min_leverage)
            )
            return max(self.config.trading.min_leverage, min(self.config.trading.max_leverage, leverage))

    def calculate_dynamic_stop_loss(self, position: Position, current_price: float, atr: float) -> float:
        """Calculate dynamic stop loss with trailing functionality"""
        if position.side == PositionSide.LONG:
            # For long positions, stop loss is below current price
            new_stop = current_price - (atr * self.config.risk.trailing_stop_atr)
            # Only trail if price has moved in our favor
            if new_stop > position.stop_loss or position.stop_loss == 0:
                return new_stop
            else:
                return position.stop_loss
        else:
            # For short positions, stop loss is above current price
            new_stop = current_price + (atr * self.config.risk.trailing_stop_atr)
            # Only trail if price has moved in our favor
            if new_stop < position.stop_loss or position.stop_loss == 0:
                return new_stop
            else:
                return position.stop_loss

    def calculate_dynamic_take_profit(self, position: Position, current_price: float, atr: float) -> float:
        """Calculate dynamic take profit"""
        if position.side == PositionSide.LONG:
            return current_price + (atr * self.config.risk.take_profit_atr)
        else:
            return current_price - (atr * self.config.risk.take_profit_atr)

    def calculate_var(self, symbol: str, candles: List[Candle], 
                     confidence_level: float = 0.95, lookback_period: int = 100) -> float:
        """Calculate Value at Risk for a symbol"""
        if len(candles) < lookback_period + 1:
            return 0
            
        # Check cache first
        cache_key = f"{symbol}_{candles[-1].timestamp}_{lookback_period}"
        if cache_key in self.var_cache:
            return self.var_cache[cache_key]
            
        # Calculate returns
        closes = [c.close for c in candles[-lookback_period:]]
        returns = np.diff(closes) / closes[:-1]
        
        # Calculate VaR using parametric method
        mean = np.mean(returns)
        std_dev = np.std(returns)
        var = norm.ppf(1 - confidence_level, mean, std_dev) * self.exchange.equity
        
        # Cache the result
        self.var_cache[cache_key] = var
        # Limit cache size
        if len(self.var_cache) > 1000:
            self.var_cache.pop(next(iter(self.var_cache)))
            
        return var

    def check_extreme_conditions(self, current_price: float, previous_price: float) -> bool:
        """Check for extreme market conditions"""
        price_change = abs(current_price - previous_price) / previous_price * 100
        return price_change >= self.config.risk.max_volatility

    def update_trade_result(self, pnl: float) -> None:
        """Update risk metrics after a trade"""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        self.daily_trade_count += 1

    def reset_daily_metrics(self) -> None:
        """Reset daily metrics (should be called at midnight)"""
        self.daily_trade_count = 0
        self.consecutive_losses = 0
        # Don't reset max_equity_reached to track overall performance