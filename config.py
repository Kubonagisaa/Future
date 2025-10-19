import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from dataclasses import dataclass

load_dotenv()

@dataclass
class APIConfig:
    key: str = os.getenv('BINANCE_API_KEY', '')
    secret: str = os.getenv('BINANCE_API_SECRET', '')
    testnet: bool = os.getenv('TESTNET', 'True').lower() == 'true'

@dataclass
class TradingConfig:
    symbols: List[str] = os.getenv('TRADING_SYMBOLS', 'BTCUSDT,ETHUSDT,ADAUSDT').split(',')
    timeframe: str = os.getenv('TIMEFRAME', '5m')
    initial_balance: float = float(os.getenv('INITIAL_BALANCE', 1000))
    risk_per_trade: float = float(os.getenv('RISK_PER_TRADE', 1.0))
    max_daily_trades: int = int(os.getenv('MAX_DAILY_TRADES', 10))
    max_leverage: int = int(os.getenv('MAX_LEVERAGE', 3))
    min_leverage: int = int(os.getenv('MIN_LEVERAGE', 1))

@dataclass
class StrategyConfig:
    ema_fast: int = int(os.getenv('EMA_FAST', 9))
    ema_slow: int = int(os.getenv('EMA_SLOW', 21))
    rsi_period: int = int(os.getenv('RSI_PERIOD', 14))
    rsi_overbought: int = int(os.getenv('RSI_OVERBOUGHT', 70))
    rsi_oversold: int = int(os.getenv('RSI_OVERSOLD', 30))
    atr_period: int = int(os.getenv('ATR_PERIOD', 14))
    atr_multiplier: float = float(os.getenv('ATR_MULTIPLIER', 2.0))
    macd_fast: int = int(os.getenv('MACD_FAST', 12))
    macd_slow: int = int(os.getenv('MACD_SLOW', 26))
    macd_signal: int = int(os.getenv('MACD_SIGNAL', 9))
    volume_ma_period: int = int(os.getenv('VOLUME_MA_PERIOD', 20))
    strategy_type: str = os.getenv('STRATEGY_TYPE', 'EMA_RSI_ATR')

@dataclass
class RiskConfig:
    max_daily_drawdown: float = float(os.getenv('MAX_DAILY_DRAWDOWN', 5.0))
    stop_loss_atr: float = float(os.getenv('STOP_LOSS_ATR', 1.5))
    take_profit_atr: float = float(os.getenv('TAKE_PROFIT_ATR', 2.0))
    max_consecutive_losses: int = int(os.getenv('MAX_CONSECUTIVE_LOSSES', 5))
    max_equity_drawdown: float = float(os.getenv('MAX_EQUITY_DRAWDOWN', 20.0))
    trailing_stop_activation: float = float(os.getenv('TRAILING_STOP_ACTIVATION', 0.5))
    trailing_stop_atr: float = float(os.getenv('TRAILING_STOP_ATR', 1.0))
    circuit_breaker_pct: float = float(os.getenv('CIRCUIT_BREAKER_PCT', 10.0))
    volume_spike_threshold: float = float(os.getenv('VOLUME_SPIKE_THRESHOLD', 2.0))
    circuit_breaker_soft_pct: float = float(os.getenv('CIRCUIT_BREAKER_SOFT_PCT', 5.0))
    circuit_breaker_hard_pct: float = float(os.getenv('CIRCUIT_BREAKER_HARD_PCT', 10.0))
    circuit_breaker_soft_duration: int = int(os.getenv('CIRCUIT_BREAKER_SOFT_DURATION', 300))
    circuit_breaker_hard_duration: int = int(os.getenv('CIRCUIT_BREAKER_HARD_DURATION', 900))
    max_volatility: float = float(os.getenv('MAX_VOLATILITY', 15.0))
    var_confidence_level: float = float(os.getenv('VAR_CONFIDENCE_LEVEL', 0.95))
    var_lookback_period: int = int(os.getenv('VAR_LOOKBACK_PERIOD', 100))

@dataclass
class OrderConfig:
    maker_fee: float = float(os.getenv('MAKER_FEE', 0.0002))
    taker_fee: float = float(os.getenv('TAKER_FEE', 0.0004))
    slippage: float = float(os.getenv('SLIPPAGE', 0.0005))
    scale_in_levels: int = int(os.getenv('SCALE_IN_LEVELS', 3))
    scale_out_levels: int = int(os.getenv('SCALE_OUT_LEVELS', 2))
    grid_atr_multiplier: float = float(os.getenv('GRID_ATR_MULTIPLIER', 0.5))
    auto_reverse: bool = os.getenv('AUTO_REVERSE', 'False').lower() == 'true'

@dataclass
class PortfolioConfig:
    max_correlation: float = float(os.getenv('MAX_CORRELATION', 0.7))
    portfolio_risk_pct: float = float(os.getenv('PORTFOLIO_RISK_PCT', 5.0))
    min_diversification: int = int(os.getenv('MIN_DIVERSIFICATION', 3))

@dataclass
class NotificationConfig:
    telegram_enabled: bool = os.getenv('TELEGRAM_ENABLED', 'False').lower() == 'true'
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    slack_enabled: bool = os.getenv('SLACK_ENABLED', 'False').lower() == 'true'
    slack_webhook_url: str = os.getenv('SLACK_WEBHOOK_URL', '')
    email_enabled: bool = os.getenv('EMAIL_ENABLED', 'False').lower() == 'true'
    email_sender: str = os.getenv('EMAIL_SENDER', '')
    email_password: str = os.getenv('EMAIL_PASSWORD', '')
    email_recipient: str = os.getenv('EMAIL_RECIPIENT', '')

@dataclass
class SentimentConfig:
    twitter_enabled: bool = os.getenv('TWITTER_ENABLED', 'False').lower() == 'true'
    twitter_api_key: str = os.getenv('TWITTER_API_KEY', '')
    twitter_api_secret: str = os.getenv('TWITTER_API_SECRET', '')
    twitter_access_token: str = os.getenv('TWITTER_ACCESS_TOKEN', '')
    twitter_access_secret: str = os.getenv('TWITTER_ACCESS_SECRET', '')
    rss_feeds: List[str] = os.getenv('RSS_FEEDS', '').split(',')
    sentiment_threshold: float = float(os.getenv('SENTIMENT_THRESHOLD', 0.2))
    sentiment_update_interval: int = int(os.getenv('SENTIMENT_UPDATE_INTERVAL', 1800))

@dataclass
class GeneralConfig:
    paper_trading: bool = os.getenv('PAPER_TRADING', 'True').lower() == 'true'
    update_interval: int = int(os.getenv('UPDATE_INTERVAL', 60))
    use_database: bool = os.getenv('USE_DATABASE', 'True').lower() == 'true'
    dashboard_enabled: bool = os.getenv('DASHBOARD_ENABLED', 'False').lower() == 'true'
    dashboard_port: int = int(os.getenv('DASHBOARD_PORT', 5000))
    risk_off_mode: bool = os.getenv('RISK_OFF_MODE', 'False').lower() == 'true'
    risk_off_capital_pct: float = float(os.getenv('RISK_OFF_CAPITAL_PCT', 50.0))
    backtest_enabled: bool = os.getenv('BACKTEST_ENABLED', 'False').lower() == 'true'
    backtest_data_path: str = os.getenv('BACKTEST_DATA_PATH', './data/backtest')

class Config:
    def __init__(self):
        self.api = APIConfig()
        self.trading = TradingConfig()
        self.strategy = StrategyConfig()
        self.risk = RiskConfig()
        self.order = OrderConfig()
        self.portfolio = PortfolioConfig()
        self.notification = NotificationConfig()
        self.sentiment = SentimentConfig()
        self.general = GeneralConfig()