import threading
import logging
from flask import Flask, jsonify, render_template
from ..config import Config
logger = logging.getLogger('FuturesBot')

class DashboardServer:
    def __init__(self, config: Config, bot: 'FuturesTradingBot'):
        self.config = config
        self.bot = bot
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
        @self.app.route('/api/equity')
        def get_equity():
            equity_history = self.bot.db.get_equity_history(24) if self.config.use_database else []
            return jsonify([{'timestamp': e[0], 'equity': e[1]} for e in equity_history])
            
        @self.app.route('/api/trades')
        def get_trades():
            trades = self.bot.db.get_trade_history(50) if self.config.use_database else []
            return jsonify([{
                'id': t.id,
                'symbol': t.symbol,
                'side': t.side.value,
                'size': t.size,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'timestamp': t.timestamp,
                'duration': t.duration,
                'strategy': t.strategy,
                'exit_reason': t.exit_reason
            } for t in trades])
            
        @self.app.route('/api/positions')
        def get_positions():
            positions = list(self.bot.exchange.positions.values()) if hasattr(self.bot.exchange, 'positions') else []
            return jsonify([{
                'symbol': p.symbol,
                'side': p.side.value,
                'size': p.size,
                'entry_price': p.entry_price,
                'leverage': p.leverage,
                'stop_loss': p.stop_loss,
                'take_profit': p.take_profit,
                'unrealized_pnl': p.unrealized_pnl,
                'timestamp': p.timestamp
            } for p in positions])
            
        @self.app.route('/api/performance')
        def get_performance():
            stats = self.bot.performance.get_stats() if hasattr(self.bot, 'performance') else {}
            return jsonify(stats)
    
    def run(self):
        if self.config.dashboard_enabled:
            threading.Thread(target=lambda: self.app.run(
                host='0.0.0.0', port=self.config.dashboard_port, debug=False, use_reloader=False
            )).start()
            logger.info(f"Dashboard started on port {self.config.dashboard_port}")