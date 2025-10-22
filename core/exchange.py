import time
import ccxt.async_support as ccxt
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from .config import Config
from .data_models import Candle, Position, Order, OrderType, PositionSide, OrderStatus
from .utils.notifier import TelegramNotifier

logger = logging.getLogger('FuturesBot')

class ExchangeError(Exception):
    """Custom exception for exchange-related errors"""
    pass

class FuturesExchange:
    def __init__(self, config: Config, notifier: 'TelegramNotifier' = None):
        self.config = config
        self.notifier = notifier
        self.exchange = None
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, Order] = {}
        self.initial_balance = config.trading.initial_balance
        self.equity = config.trading.initial_balance
        self.circuit_breaker_enabled = False
        self.circuit_breaker_trigger_time = 0
        self.last_circuit_breaker_time = 0
        
        # Initialize the exchange (Binance)
        try:
            if config.api.testnet:
                self.exchange = ccxt.binance({
                    'apiKey': config.api.key,
                    'secret': config.api.secret,
                    'options': {'defaultType': 'future'},
                    'enableRateLimit': True
                })
                self.exchange.set_sandbox_mode(True)
            else:
                self.exchange = ccxt.binance({
                    'apiKey': config.api.key,
                    'secret': config.api.secret,
                    'options': {'defaultType': 'future'},
                    'enableRateLimit': True
                })
            logger.info("Exchange initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise ExchangeError(f"Exchange initialization failed: {e}")

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[Candle]:
        """Fetch OHLCV data from exchange"""
        try:
            data = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            candles = [Candle(*ohlcv) for ohlcv in data]
            return candles
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            if self.notifier:
                await self.notifier.send_message(f"Failed to fetch market data for {symbol}: {e}")
            return []

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[Candle]:
        """Get OHLCV data with error handling"""
        return await self.fetch_ohlcv(symbol, timeframe, limit)

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        try:
            await self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage set to {leverage} for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            if self.notifier:
                await self.notifier.send_message(f"Failed to set leverage for {symbol}: {e}")
            return False

    async def apply_funding(self) -> None:
        """Apply funding payments (for paper trading)"""
        if self.config.general.paper_trading:
            current_time = time.time()
            for symbol, position in self.positions.items():
                # Simulate funding every 8 hours
                if current_time - position.last_funding_time >= 28800:
                    # Calculate funding payment
                    funding_rate = 0.0001  # Simulated funding rate
                    funding_amount = position.size * position.entry_price * funding_rate
                    
                    if position.side == PositionSide.LONG:
                        self.equity -= funding_amount
                    else:
                        self.equity += funding_amount
                    
                    position.last_funding_time = current_time
                    
                    logger.info(f"Funding applied for {symbol}: {funding_amount:.4f} USDT")
                    if self.notifier:
                        await self.notifier.notify_funding(symbol, funding_amount, position.side)

    async def update_equity(self) -> None:
        """Update equity from exchange balance"""
        try:
            balance = await self.exchange.fetch_balance()
            if 'USDT' in balance['total']:
                self.equity = balance['total']['USDT']
                logger.debug(f"Equity updated: {self.equity:.2f} USDT")
        except Exception as e:
            logger.error(f"Failed to update equity: {e}")
            if self.notifier:
                await self.notifier.send_message(f"Failed to update equity: {e}")

    async def check_open_orders(self, symbol: str) -> None:
        """Check status of open orders"""
        try:
            open_orders = await self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                order_id = order['id']
                if order_id in self.open_orders:
                    our_order = self.open_orders[order_id]
                    if order['status'] == 'closed':
                        # Order filled
                        our_order.status = OrderStatus.FILLED
                        our_order.filled = order['filled']
                        our_order.remaining = order['remaining']
                        del self.open_orders[order_id]
                        logger.info(f"Order {order_id} filled for {symbol}")
                    elif order['status'] == 'canceled':
                        # Order canceled
                        our_order.status = OrderStatus.CANCELLED
                        del self.open_orders[order_id]
                        logger.info(f"Order {order_id} canceled for {symbol}")
        except Exception as e:
            logger.error(f"Error checking open orders for {symbol}: {e}")

    async def create_order(self, symbol: str, order_type: OrderType, side: PositionSide, 
                          amount: float, price: Optional[float] = None, 
                          params: Optional[Dict[str, Any]] = None) -> Optional[Order]:
        """Create a new order"""
        try:
            order_params = {
                'symbol': symbol,
                'type': order_type.value.lower(),
                'side': 'buy' if side == PositionSide.LONG else 'sell',
                'amount': amount
            }
            
            if price is not None:
                order_params['price'] = price
            
            if params:
                order_params['params'] = params
            
            result = await self.exchange.create_order(**order_params)
            
            # Create Order object
            order = Order(
                id=result['id'],
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=result.get('price', price) if price else result['average'],
                status=OrderStatus.OPEN,
                timestamp=int(time.time() * 1000),
                params=params or {}
            )
            
            self.open_orders[order.id] = order
            logger.info(f"Order created: {order.id} for {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to create order for {symbol}: {e}")
            if self.notifier:
                await self.notifier.send_message(f"Failed to create order for {symbol}: {e}")
            return None

    async def close_position(self, symbol: str, side: PositionSide, amount: float, 
                            reason: str = "") -> bool:
        """Close a position"""
        try:
            # Create opposite order to close position
            close_side = 'sell' if side == PositionSide.LONG else 'buy'
            
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=close_side,
                amount=amount
            )
            
            if order and order['status'] == 'closed':
                # Remove position if fully closed
                if symbol in self.positions:
                    if self.positions[symbol].size <= amount:
                        del self.positions[symbol]
                    else:
                        self.positions[symbol].size -= amount
                
                logger.info(f"Position closed for {symbol}: {amount} {reason}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            if self.notifier:
                await self.notifier.send_message(f"Failed to close position for {symbol}: {e}")
            return False

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol"""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            if positions and len(positions) > 0:
                pos = positions[0]
                if pos['contracts'] > 0:
                    position = Position(
                        symbol=symbol,
                        side=PositionSide.LONG if pos['side'] == 'long' else PositionSide.SHORT,
                        size=pos['contracts'],
                        entry_price=pos['entryPrice'],
                        leverage=pos['leverage'],
                        stop_loss=pos['stopLoss'] if pos['stopLoss'] else 0,
                        take_profit=pos['takeProfit'] if pos['takeProfit'] else 0,
                        timestamp=int(time.time() * 1000)
                    )
                    self.positions[symbol] = position
                    return position
            return None
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None

    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker['last'] if ticker else 0
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return 0

    async def check_circuit_breaker(self, symbol: str, candles: List[Candle]) -> None:
        """Check for circuit breaker conditions"""
        if len(candles) < 2:
            return
            
        current_price = candles[-1].close
        previous_price = candles[-2].close
        price_change = abs(current_price - previous_price) / previous_price * 100
        
        current_time = time.time()
        
        # Hard circuit breaker
        if price_change >= self.config.risk.circuit_breaker_hard_pct:
            self.circuit_breaker_enabled = True
            self.circuit_breaker_trigger_time = current_time
            logger.warning(f"Hard circuit breaker activated for {symbol}: {price_change:.2f}%")
            if self.notifier:
                await self.notifier.notify_circuit_breaker(symbol, price_change, "HARD")
        
        # Soft circuit breaker
        elif price_change >= self.config.risk.circuit_breaker_soft_pct:
            if not self.circuit_breaker_enabled:
                self.circuit_breaker_enabled = True
                self.circuit_breaker_trigger_time = current_time
                logger.warning(f"Soft circuit breaker activated for {symbol}: {price_change:.2f}%")
                if self.notifier:
                    await self.notifier.notify_circuit_breaker(symbol, price_change, "SOFT")
        
        # Check if we can disable circuit breaker
        if self.circuit_breaker_enabled:
            breaker_duration = current_time - self.circuit_breaker_trigger_time
            required_duration = (
                self.config.risk.circuit_breaker_hard_duration 
                if price_change >= self.config.risk.circuit_breaker_hard_pct
                else self.config.risk.circuit_breaker_soft_duration
            )
            
            if breaker_duration >= required_duration:
                self.circuit_breaker_enabled = False
                logger.info(f"Circuit breaker deactivated for {symbol}")
                if self.notifier:
                    await self.notifier.send_message(f"Circuit breaker deactivated for {symbol}")

    async def close(self) -> None:
        """Close exchange connection"""
        try:
            await self.exchange.close()
            logger.info("Exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing exchange connection: {e}")