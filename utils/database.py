import sqlite3
import threading
from typing import Dict, List, Optional
from contextlib import contextmanager
from datetime import datetime
import time
from ..data_models import Position, Trade, MarketCondition, MarketRegime

class DatabaseManager:
    def __init__(self, db_path='trading_bot.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_db()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            c = conn.cursor()
            
            # Create tables if they don't exist
            c.execute('''CREATE TABLE IF NOT EXISTS positions (
                            id TEXT PRIMARY KEY,
                            symbol TEXT,
                            side TEXT,
                            size REAL,
                            entry_price REAL,
                            leverage INTEGER,
                            stop_loss REAL,
                            take_profit REAL,
                            trailing_stop REAL,
                            timestamp INTEGER,
                            last_funding_time INTEGER,
                            scale_in_level INTEGER,
                            scale_out_level INTEGER,
                            entry_reason TEXT,
                            unrealized_pnl REAL
                        )''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS trades (
                            id TEXT PRIMARY KEY,
                            symbol TEXT,
                            side TEXT,
                            size REAL,
                            entry_price REAL,
                            exit_price REAL,
                            pnl REAL,
                            pnl_percent REAL,
                            timestamp INTEGER,
                            duration REAL,
                            fee REAL,
                            funding REAL,
                            slippage REAL,
                            strategy TEXT,
                            exit_reason TEXT
                        )''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS market_conditions (
                            timestamp INTEGER,
                            symbol TEXT,
                            volatility REAL,
                            trend_strength REAL,
                            volume_ratio REAL,
                            market_regime TEXT,
                            PRIMARY KEY (timestamp, symbol)
                        )''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS equity (
                            timestamp INTEGER PRIMARY KEY,
                            equity REAL
                        )''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS performance_metrics (
                            date TEXT PRIMARY KEY,
                            total_trades INTEGER,
                            winning_trades INTEGER,
                            losing_trades INTEGER,
                            win_rate REAL,
                            total_pnl REAL,
                            max_drawdown REAL,
                            sharpe_ratio REAL,
                            profit_factor REAL
                        )''')
            
            conn.commit()

    def save_position(self, position: Position) -> bool:
        """Save position to database"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (position.id, position.symbol, position.side.value, position.size,
                           position.entry_price, position.leverage, position.stop_loss,
                           position.take_profit, position.trailing_stop, position.timestamp,
                           position.last_funding_time, position.scale_in_level,
                           position.scale_out_level, position.entry_reason, position.unrealized_pnl))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving position: {e}")
            return False

    def load_positions(self) -> Dict[str, Position]:
        """Load positions from database"""
        positions = {}
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('SELECT * FROM positions')
                rows = c.fetchall()
                
                for row in rows:
                    position = Position(
                        id=row['id'],
                        symbol=row['symbol'],
                        side=PositionSide(row['side']),
                        size=row['size'],
                        entry_price=row['entry_price'],
                        leverage=row['leverage'],
                        stop_loss=row['stop_loss'],
                        take_profit=row['take_profit'],
                        trailing_stop=row['trailing_stop'],
                        timestamp=row['timestamp'],
                        last_funding_time=row['last_funding_time'],
                        scale_in_level=row['scale_in_level'],
                        scale_out_level=row['scale_out_level'],
                        entry_reason=row['entry_reason'],
                        unrealized_pnl=row['unrealized_pnl']
                    )
                    positions[position.symbol] = position
        except Exception as e:
            print(f"Error loading positions: {e}")
        
        return positions

    def save_trade(self, trade: Trade) -> bool:
        """Save trade to database"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''REPLACE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (trade.id, trade.symbol, trade.side.value, trade.size,
                           trade.entry_price, trade.exit_price, trade.pnl, trade.pnl_percent,
                           trade.timestamp, trade.duration, trade.fee, trade.funding,
                           trade.slippage, trade.strategy, trade.exit_reason))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving trade: {e}")
            return False

    def get_trade_history(self, limit: int = 100, 
                         start_date: Optional[int] = None, 
                         end_date: Optional[int] = None) -> List[Trade]:
        """Get trade history with optional filters"""
        trades = []
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                query = 'SELECT * FROM trades'
                params = []
                
                if start_date or end_date:
                    conditions = []
                    if start_date:
                        conditions.append('timestamp >= ?')
                        params.append(start_date)
                    if end_date:
                        conditions.append('timestamp <= ?')
                        params.append(end_date)
                    
                    query += ' WHERE ' + ' AND '.join(conditions)
                
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                
                c.execute(query, params)
                rows = c.fetchall()
                
                for row in rows:
                    trade = Trade(
                        id=row['id'],
                        symbol=row['symbol'],
                        side=PositionSide(row['side']),
                        size=row['size'],
                        entry_price=row['entry_price'],
                        exit_price=row['exit_price'],
                        pnl=row['pnl'],
                        pnl_percent=row['pnl_percent'],
                        timestamp=row['timestamp'],
                        duration=row['duration'],
                        fee=row['fee'],
                        funding=row['funding'],
                        slippage=row['slippage'],
                        strategy=row['strategy'],
                        exit_reason=row['exit_reason']
                    )
                    trades.append(trade)
        except Exception as e:
            print(f"Error getting trade history: {e}")
        
        return trades

    def save_market_condition(self, timestamp: int, symbol: str, condition: MarketCondition) -> bool:
        """Save market condition to database"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''REPLACE INTO market_conditions VALUES (?, ?, ?, ?, ?, ?)''',
                          (timestamp, symbol, condition.volatility, condition.trend_strength, 
                           condition.volume_ratio, condition.market_regime.value))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving market condition: {e}")
            return False

    def get_market_conditions(self, symbol: str, limit: int = 100,
                             start_date: Optional[int] = None,
                             end_date: Optional[int] = None) -> List[MarketCondition]:
        """Get market conditions for a symbol"""
        conditions = []
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                query = 'SELECT * FROM market_conditions WHERE symbol = ?'
                params = [symbol]
                
                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)
                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)
                
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                
                c.execute(query, params)
                rows = c.fetchall()
                
                for row in rows:
                    condition = MarketCondition(
                        volatility=row['volatility'],
                        trend_strength=row['trend_strength'],
                        volume_ratio=row['volume_ratio'],
                        market_regime=MarketRegime(row['market_regime'])
                    )
                    conditions.append(condition)
        except Exception as e:
            print(f"Error getting market conditions: {e}")
        
        return conditions

    def save_equity(self, timestamp: int, equity: float) -> bool:
        """Save equity value to database"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('REPLACE INTO equity VALUES (?, ?)', (timestamp, equity))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving equity: {e}")
            return False

    def get_equity_history(self, hours: int = 24) -> List[tuple]:
        """Get equity history"""
        history = []
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                cutoff = int(time.time() * 1000) - hours * 3600 * 1000
                c.execute('SELECT * FROM equity WHERE timestamp >= ? ORDER BY timestamp', (cutoff,))
                rows = c.fetchall()
                
                for row in rows:
                    history.append((row['timestamp'], row['equity']))
        except Exception as e:
            print(f"Error getting equity history: {e}")
        
        return history

    def save_performance_metrics(self, date: str, metrics: Dict[str, Any]) -> bool:
        """Save performance metrics to database"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''REPLACE INTO performance_metrics 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (date, metrics.get('total_trades', 0),
                           metrics.get('winning_trades', 0),
                           metrics.get('losing_trades', 0),
                           metrics.get('win_rate', 0),
                           metrics.get('total_pnl', 0),
                           metrics.get('max_drawdown', 0),
                           metrics.get('sharpe_ratio', 0),
                           metrics.get('profit_factor', 0)))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving performance metrics: {e}")
            return False

    def get_performance_metrics(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get performance metrics for a date range"""
        metrics = []
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''SELECT * FROM performance_metrics 
                            WHERE date BETWEEN ? AND ? ORDER BY date''',
                          (start_date, end_date))
                rows = c.fetchall()
                
                for row in rows:
                    metrics.append({
                        'date': row['date'],
                        'total_trades': row['total_trades'],
                        'winning_trades': row['winning_trades'],
                        'losing_trades': row['losing_trades'],
                        'win_rate': row['win_rate'],
                        'total_pnl': row['total_pnl'],
                        'max_drawdown': row['max_drawdown'],
                        'sharpe_ratio': row['sharpe_ratio'],
                        'profit_factor': row['profit_factor']
                    })
        except Exception as e:
            print(f"Error getting performance metrics: {e}")
        
        return metrics