import asyncio
import time
import signal
import sys
from typing import Dict, List, Any
from .config import Config
from .core.exchange import FuturesExchange
from .core.strategy import create_strategy
from .core.risk_manager import RiskManager
from .core.portfolio import PortfolioManager
from .utils.notifier import TelegramNotifier
from .data_models import Position, Order, Trade, PositionSide, MarketCondition
from .utils.database import DatabaseManager
from .utils.dashboard import DashboardServer

class FuturesTradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.notifier = TelegramNotifier(config)
        self.exchange = FuturesExchange(config, self.notifier)
        self.strategy = create_strategy(config)
        self.risk_manager = RiskManager(config, self.exchange)
        self.portfolio_manager = PortfolioManager(config, self.exchange)
        self.sentiment_analyzer = NewsSentimentAnalyzer(config)
        self.db = DatabaseManager() if config.use_database else None
        self.performance = PerformanceTracker(self.db)
        self.dashboard = DashboardServer(config, self)
        self.running = False
        self.active_positions = {}
        self.pending_orders = {}
        self.last_stop_loss_time = 0
        self.stop_loss_cooldown = 30 * 60  # 30 minutes cooldown after stop loss

        self.last_daily_report = 0
        self.daily_report_interval = 24 * 60 * 60  # 24 hours
        self.last_sentiment_check = 0
        self.sentiment_check_interval = 30 * 60  # 30 minutes
        
    async def initialize(self):
        # Set leverage for all symbols
        for symbol in self.config.symbols:
            await self.exchange.set_leverage(symbol, self.config.max_leverage)
        
        # Set initial equity
        self.exchange.equity = self.exchange.initial_balance
        
        # Load positions from database
        if self.config.use_database:
            self.active_positions = self.db.load_positions()
        
        # Start dashboard
        self.dashboard.run()
        
        logger.info("Bot initialized")
        
    async def run(self):
        self.running = True
        logger.info("Starting futures trading bot")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        await self.initialize()
        
        try:
            while self.running:
                await self.trading_cycle()
                await asyncio.sleep(self.config.update_interval)
                
                # Send daily report if needed
                await self.send_daily_report()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await self.notifier.send_message(f"Bot stopped due to error: {str(e)}")
            await self.shutdown()
    
    async def trading_cycle(self):
        try:
            # Apply funding payments (for paper trading)
            await self.exchange.apply_funding()
            
            # Update equity
            await self.exchange.update_equity()
            
            # Update risk manager
            self.risk_manager.update_drawdown(self.exchange.equity)
            
            # Check if we can trade
            if not self.risk_manager.can_trade():
                # If max drawdown reached, shutdown bot
                equity_drawdown = (self.exchange.initial_balance - self.exchange.equity) / self.exchange.initial_balance * 100
                if equity_drawdown >= self.config.max_equity_drawdown:
                    await self.notifier.notify_shutdown(f"Max equity drawdown reached: {equity_drawdown:.2f}%", self.exchange.equity)
                    self.running = False
                return
            
            # Get market data for all symbols
            all_candles = {}
            for symbol in self.config.symbols:
                candles = await self.exchange.get_ohlcv(symbol, self.config.timeframe, 100)
                if candles:
                    all_candles[symbol] = candles
                    
                    # Check for circuit breaker conditions
                    await self.exchange.check_circuit_breaker(symbol, candles)
                    
                    # Check for extreme conditions
                    if len(candles) >= 2:
                        current_price = candles[-1].close
                        previous_price = candles[-2].close
                        if self.risk_manager.check_extreme_conditions(current_price, previous_price):
                            await self.notifier.notify_shutdown(f"Extreme volatility detected: {abs(current_price - previous_price)/previous_price*100:.2f}%", self.exchange.equity)
                            self.running = False
                            return
            
            # Update portfolio metrics
            await self.portfolio_manager.update_portfolio_metrics(all_candles)
            
            # Check sentiment if it's time
            current_time = time.time()
            if current_time - self.last_sentiment_check >= self.sentiment_check_interval:
                await self.check_sentiment(all_candles)
                self.last_sentiment_check = current_time
            
            # Generate signals for all symbols
            signals = {}
            market_conditions = {}
            
            for symbol, candles in all_candles.items():
                if candles and len(candles) > 20:
                    signal_data = self.strategy.generate_signal(candles)
                    signals[symbol] = signal_data
                    
                    # Analyze market condition
                    market_condition = self.strategy.analyze_market_condition(candles)
                    market_conditions[symbol] = market_condition
                    
                    # Save market condition to database if enabled
                    if self.config.use_database:
                        self.db.save_market_condition(int(time.time() * 1000), symbol, market_condition)
            
            # Get current positions
            current_positions = {}
            for symbol in self.config.symbols:
                position = await self.exchange.get_position(symbol)
                if position:
                    current_positions[symbol] = position
                    self.active_positions[symbol] = position
            
            # Calculate portfolio allocation
            portfolio_allocations = self.portfolio_manager.get_portfolio_allocation(signals)
            
            # Handle positions for each symbol
            for symbol in self.config.symbols:
                if symbol in all_candles and symbol in signals:
                    candles = all_candles[symbol]
                    signal_data = signals[symbol]
                    
                    # Check open orders for this symbol
                    await self.exchange.check_open_orders(symbol)
                    
                    # Handle position if we have one
                    if symbol in current_positions:
                        await self.manage_position(current_positions[symbol], candles, market_conditions[symbol])
                    else:
                        # Check if we're in cooldown after stop loss
                        current_time = time.time()
                        if current_time - self.last_stop_loss_time < self.stop_loss_cooldown:
                            logger.info(f"In cooldown period after stop loss for {symbol}")
                            continue
                        
                        # Check portfolio constraints
                        if not self.portfolio_manager.should_enter_trade(symbol, signals, current_positions):
                            continue
                        
                        # No position, check for new entry
                        if signal_data["signal"] != "HOLD" and signal_data["confidence"] > 0.3:
                            # Get portfolio weight for this symbol
                            portfolio_weight = portfolio_allocations.get(symbol, 1.0)
                            
                            await self.enter_position(signal_data, candles, portfolio_weight)
        
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def check_sentiment(self, all_candles: Dict[str, List[Candle]]):
        """Check market sentiment and adjust trading accordingly"""
        for symbol in all_candles.keys():
            sentiment_score = await self.sentiment_analyzer.get_sentiment_score(symbol)
            
            # Adjust risk-off mode based on sentiment
            if abs(sentiment_score) > self.config.sentiment_threshold:
                if sentiment_score > 0:
                    # Positive sentiment - increase risk appetite
                    self.config.risk_off_mode = False
                    logger.info(f"Positive sentiment for {symbol}, risk-off mode disabled")
                else:
                    # Negative sentiment - reduce risk appetite
                    self.config.risk_off_mode = True
                    logger.info(f"Negative sentiment for {symbol}, risk-off mode enabled")
    
    async def enter_position(self, signal_data: Dict, candles: List[Candle], portfolio_weight: float = 1.0):
        symbol = signal_data.get('symbol', self.config.symbols[0])
        side = PositionSide.LONG if signal_data["signal"] == "LONG" else PositionSide.SHORT
        current_price = signal_data["price"]
        atr = signal_data["atr"]
        
        # Calculate position size with portfolio weighting
        position_size = self.risk_manager.calculate_position_size(current_price, atr, portfolio_weight)
        
        if position_size <= 0:
            logger.warning(f"Position size is zero or negative for {symbol}")
            return
        
        # Calculate adaptive leverage
        leverage = self.risk_manager.calculate_adaptive_leverage(atr, current_price)
        
        # Calculate stop loss and take profit
        if side == PositionSide.LONG:
            stop_loss = current_price - (atr * self.config.stop_loss_atr)
            take_profit = current_price + (atr * self.config.take_profit_atr)
        else:  # SHORT
            stop_loss = current_price + (atr * self.config.stop_loss_atr)
            take_profit = current_price - (atr * self.config.take_profit_atr)
        
        # For scale-in strategy, start with partial position
        scale_in_factor = 1.0 / self.config.scale_in_levels
        initial_size = position_size * scale_in_factor
        
        # Create order
        order = await self.exchange.create_order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            amount=initial_size,
            params={
                'leverage': leverage,
                'stopLoss': stop_loss,
                'takeProfit': take_profit,
                'entry_reason': self.config.strategy_type
            }
        )
        
        if order:
            logger.info(f"Entered {side.value} position for {symbol}, size: {initial_size}, leverage: {leverage}")
            await self.notifier.notify_trade(side, symbol, initial_size, current_price, self.config.strategy_type)
            
            # Update daily trade count
            self.risk_manager.daily_trade_count += 1
            
            # Store scale-in level
            if symbol in self.exchange.positions:
                self.exchange.positions[symbol].scale_in_level = 1
    
    async def manage_position(self, position: Position, candles: List[Candle], market_condition: MarketCondition):
        symbol = position.symbol
        current_price = candles[-1].close if candles else await self.exchange.get_current_price(symbol)
        atr = TechnicalIndicators.calculate_atr(candles, self.config.atr_period)[-1] if candles else 0
        
        # Update trailing stop and take profit
        new_stop_loss = self.risk_manager.calculate_dynamic_stop_loss(position, current_price, atr)
        new_take_profit = self.risk_manager.calculate_dynamic_take_profit(position, current_price, atr)
        
        # Update position if stop loss or take profit has changed
        if new_stop_loss != position.stop_loss or new_take_profit != position.take_profit:
            position.stop_loss = new_stop_loss
            position.take_profit = new_take_profit
            
            # Save to database if enabled
            if self.config.use_database:
                self.db.save_position(position)
                
            # Notify about trailing update
            await self.notifier.notify_trailing_update(symbol, new_stop_loss, new_take_profit)
        
        # Check if we hit stop loss or take profit
        if position.side == PositionSide.LONG:
            if current_price <= position.stop_loss:
                await self.exit_position(position, current_price, "Stop Loss")
                self.last_stop_loss_time = time.time()
                
                # Auto-reverse if configured
                if self.config.auto_reverse:
                    await asyncio.sleep(1)  # Small delay
                    signal_data = {
                        "signal": "SHORT",
                        "confidence": 0.5,
                        "atr": atr,
                        "price": current_price
                    }
                    await self.enter_position(signal_data, candles, 1.0)
                return
            elif current_price >= position.take_profit:
                # Scale out partially or fully exit
                if position.scale_out_level < self.config.scale_out_levels - 1:
                    # Scale out partially
                    scale_out_size = position.size / (self.config.scale_out_levels - position.scale_out_level)
                    await self.exit_partial_position(position, scale_out_size, current_price, "Partial Take Profit")
                    position.scale_out_level += 1
                else:
                    # Fully exit
                    await self.exit_position(position, current_price, "Take Profit")
                return
        else:  # SHORT
            if current_price >= position.stop_loss:
                await self.exit_position(position, current_price, "Stop Loss")
                self.last_stop_loss_time = time.time()
                
                # Auto-reverse if configured
                if self.config.auto_reverse:
                    await asyncio.sleep(1)  # Small delay
                    signal_data = {
                        "signal": "LONG",
                        "confidence": 0.5,
                        "atr": atr,
                        "price": current_price
                    }
                    await self.enter_position(signal_data, candles, 1.0)
                return
            elif current_price <= position.take_profit:
                # Scale out partially or fully exit
                if position.scale_out_level < self.config.scale_out_levels - 1:
                    # Scale out partially
                    scale_out_size = position.size / (self.config.scale_out_levels - position.scale_out_level)
                    await self.exit_partial_position(position, scale_out_size, current_price, "Partial Take Profit")
                    position.scale_out_level += 1
                else:
                    # Fully exit
                    await self.exit_position(position, current_price, "Take Profit")
                return
        
        # Check for scale-in opportunities
        if position.scale_in_level < self.config.scale_in_levels - 1:
            # Check if price has moved in our favor for scale-in
            price_move_pct = abs(current_price - position.entry_price) / position.entry_price * 100
            
            if position.side == PositionSide.LONG:
                # For long positions, scale in on pullbacks
                if current_price < position.entry_price and price_move_pct > 0.5:
                    await self.scale_in_position(position, candles, "Pullback scale-in")
            else:
                # For short positions, scale in on bounces
                if current_price > position.entry_price and price_move_pct > 0.5:
                    await self.scale_in_position(position, candles, "Bounce scale-in")
        
        # Check if we should exit based on strategy signal
        signal_data = self.strategy.generate_signal(candles)
        
        # For LONG position, exit if signal turns bearish
        if position.side == PositionSide.LONG and signal_data["signal"] == "SHORT":
            await self.exit_position(position, current_price, "Strategy signal")
        
        # For SHORT position, exit if signal turns bullish
        elif position.side == PositionSide.SHORT and signal_data["signal"] == "LONG":
            await self.exit_position(position, current_price, "Strategy signal")
    
    async def scale_in_position(self, position: Position, candles: List[Candle], reason: str):
        symbol = position.symbol
        current_price = candles[-1].close if candles else await self.exchange.get_current_price(symbol)
        atr = TechnicalIndicators.calculate_atr(candles, self.config.atr_period)[-1] if candles else 0
        
        # Calculate additional position size
        scale_in_factor = 1.0 / self.config.scale_in_levels
        additional_size = (position.size / (position.scale_in_level + 1)) * scale_in_factor
        
        # Create additional order
        order = await self.exchange.create_order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=position.side,
            amount=additional_size,
            params={
                'leverage': position.leverage,
                'stopLoss': position.stop_loss,
                'takeProfit': position.take_profit,
                'entry_reason': f"Scale-in: {reason}"
            }
        )
        
        if order:
            # Update position size and average entry price
            total_size = position.size + additional_size
            new_avg_price = ((position.entry_price * position.size) + (current_price * additional_size)) / total_size
            
            position.size = total_size
            position.entry_price = new_avg_price
            position.scale_in_level += 1
            
            # Save to database if enabled
            if self.config.use_database:
                self.db.save_position(position)
            
            logger.info(f"Scaled in {position.side.value} position for {symbol}, new size: {position.size}, avg price: {new_avg_price}")
            await self.notifier.notify_trade(position.side, symbol, additional_size, current_price, f"Scale-in: {reason}")
    
    async def exit_partial_position(self, position: Position, amount: float, exit_price: float, reason: str):
        success = await self.exchange.close_position(
            position.symbol, position.side, amount, reason
        )
        
        if success:
            # Calculate PnL for the partial exit
            if position.side == PositionSide.LONG:
                pnl = (exit_price - position.entry_price) * amount
            else:  # SHORT
                pnl = (position.entry_price - exit_price) * amount
            
            pnl_percent = (pnl / (amount * position.entry_price)) * 100 * position.leverage
            
            # For paper trading, calculate fees and slippage
            fee = (amount * exit_price) * self.config.taker_fee
            slippage = exit_price * self.config.slippage
            
            # Update position size
            position.size -= amount
            
            # Save to database if enabled
            if self.config.use_database:
                self.db.save_position(position)
            
            # Create partial trade record
            trade = Trade(
                id=f"partial_{int(time.time() * 1000)}",
                symbol=position.symbol,
                side=position.side,
                size=amount,
                entry_price=position.entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                timestamp=int(time.time() * 1000),
                duration=(time.time() * 1000 - position.timestamp) / 1000,
                fee=fee,
                slippage=slippage,
                strategy=position.entry_reason,
                exit_reason=reason
            )
            
            self.performance.update(trade)
            
            logger.info(f"Partially exited {position.side.value} position for {position.symbol}, Amount: {amount}, PnL: {pnl:.2f} USDT, Reason: {reason}")
            await self.notifier.notify_exit(position.side, position.symbol, pnl, pnl_percent, reason)
    
    async def exit_position(self, position: Position, exit_price: float, reason: str):
        success = await self.exchange.close_position(
            position.symbol, position.side, position.size, reason
        )
        
        if success:
            # Calculate PnL
            if position.side == PositionSide.LONG:
                pnl = (exit_price - position.entry_price) * position.size
            else:  # SHORT
                pnl = (position.entry_price - exit_price) * position.size
            
            pnl_percent = (pnl / (position.size * position.entry_price)) * 100 * position.leverage
            
            # For paper trading, calculate fees and slippage
            fee = (position.size * exit_price) * self.config.taker_fee
            slippage = exit_price * self.config.slippage
            
            # Log trade
            trade = Trade(
                id=f"trade_{int(time.time() * 1000)}",
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                timestamp=int(time.time() * 1000),
                duration=(time.time() * 1000 - position.timestamp) / 1000,
                fee=fee,
                slippage=slippage,
                strategy=position.entry_reason,
                exit_reason=reason
            )
            
            self.performance.update(trade)
            
            logger.info(f"Exited {position.side.value} position for {position.symbol}, PnL: {pnl:.2f} USDT, Reason: {reason}")
            await self.notifier.notify_exit(position.side, position.symbol, pnl, pnl_percent, reason)
            
            # Reset active position
            if position.symbol in self.active_positions:
                del self.active_positions[position.symbol]
    
    async def send_daily_report(self):
        current_time = time.time()
        if current_time - self.last_daily_report >= self.daily_report_interval:
            self.last_daily_report = current_time
            
            # Get today's trades
            trades = self.db.get_trade_history(100) if self.db else []
            today = datetime.now().date()
            today_trades = [t for t in trades if datetime.fromtimestamp(t.timestamp/1000).date() == today]
            
            # Calculate daily PnL and win rate
            daily_pnl = sum(t.pnl for t in today_trades)
            winning_trades = [t for t in today_trades if t.pnl >= 0]
            win_rate = len(winning_trades) / len(today_trades) * 100 if today_trades else 0
            
            await self.notifier.notify_daily_report(len(today_trades), daily_pnl, win_rate)
    
    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down")
        self.running = False
    
    async def shutdown(self):
        logger.info("Shutting down bot")
        
        # Close any open positions if needed
        for symbol, position in self.active_positions.items():
            current_price = await self.exchange.get_current_price(symbol)
            await self.exit_position(position, current_price, "Shutdown")
        
        # Log performance stats
        stats = self.performance.get_stats()
        logger.info(f"Performance Stats: {stats}")
        
        # Send performance summary via Telegram
        if self.config.telegram_enabled:
            message = (
                f"<b>Performance Summary</b>\n"
                f"Total Trades: {stats['total_trades']}\n"
                f"Win Rate: {stats['win_rate']:.2f}%\n"
                f"Total PnL: ${stats['total_pnl']:.2f}\n"
                f"Avg Profit: ${stats['avg_profit']:.2f}\n"
                f"Profit Factor: {stats['profit_factor']:.2f}\n"
                f"Duration: {timedelta(seconds=int(stats['duration']))}"
            )
            await self.notifier.send_message(message)
        
        # Close exchange connection
        await self.exchange.exchange.close()
        
        logger.info("Bot shutdown complete")
    
    def run_backtest(self, symbol: str, candles: List[Candle]) -> Dict[str, Any]:
        """Run backtest on historical data"""
        backtester = Backtester(self.config)
        return backtester.run(symbol, candles)
    
    def run_walk_forward_analysis(self, symbol: str, candles: List[Candle], strategy_class, train_size: int, test_size: int, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run walk-forward analysis on historical data"""
        analyzer = WalkForwardAnalyzer(self.config)
        return analyzer.run(symbol, candles, strategy_class, train_size, test_size, param_space)
    
    def optimize_parameters(self, symbol: str, candles: List[Candle], strategy_class, param_space: Dict[str, Any], n_trials: int = 100) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        optimizer = ParameterOptimizer(self.config)
        return optimizer.optimize(symbol, candles, strategy_class, param_space, n_trials)