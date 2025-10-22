from typing import List, Dict, Any, Optional
import numpy as np
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from .config import Config
from .data_models import Candle, MarketCondition, MarketRegime
from .utils.indicators import TechnicalIndicators

logger = logging.getLogger('FuturesBot')

class MLSignalPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cache = {}  # Cache for calculated features
        
    def prepare_features(self, candles: List[Candle]) -> Optional[np.ndarray]:
        """Prepare features for ML model from candle data"""
        if len(candles) < 50:
            return None
            
        # Check cache first
        cache_key = f"{candles[-1].timestamp}_{len(candles)}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
            
        closes = np.array([c.close for c in candles])
        opens = np.array([c.open for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        
        # Calculate technical indicators as features
        rsi = np.array(TechnicalIndicators.calculate_rsi(closes.tolist(), 14))
        macd, signal, hist = TechnicalIndicators.calculate_macd(closes.tolist(), 12, 26, 9)
        macd = np.array(macd)
        
        # Price changes
        price_change_1 = np.diff(closes) / closes[:-1] * 100
        price_change_1 = np.concatenate(([0], price_change_1))
        
        price_change_5 = (closes[5:] - closes[:-5]) / closes[:-5] * 100
        price_change_5 = np.concatenate(([0] * 5, price_change_5))
        
        # Volume changes
        volume_change = np.diff(volumes) / (volumes[:-1] + 1e-10) * 100
        volume_change = np.concatenate(([0], volume_change))
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(closes, 20, 2)
        bb_width = (bb_upper - bb_lower) / bb_middle * 100 if bb_middle != 0 else 0
        
        # Create feature matrix
        features = np.column_stack([
            rsi[-50:],
            macd[-50:],
            price_change_1[-50:],
            price_change_5[-50:],
            volume_change[-50:],
            bb_width[-50:] if isinstance(bb_width, np.ndarray) else np.zeros(50)
        ])
        
        # Cache the result
        self.feature_cache[cache_key] = features
        # Limit cache size
        if len(self.feature_cache) > 1000:
            self.feature_cache.pop(next(iter(self.feature_cache)))
            
        return features
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the ML model"""
        if len(X) < 100:
            logger.warning("Not enough data to train ML model")
            return
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info("ML model trained successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_trained or self.model is None:
            return np.zeros(X.shape[0])
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]  # Probability of positive return
    
    def save_model(self, filename="ml_model.pkl"):
        """Save the trained model to disk"""
        if self.model is not None and self.scaler is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }, filename)
            logger.info(f"ML model saved to {filename}")
    
    def load_model(self, filename="ml_model.pkl"):
        """Load a trained model from disk"""
        try:
            data = joblib.load(filename)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            logger.info(f"ML model loaded from {filename}")
        except FileNotFoundError:
            logger.warning(f"No saved model found at {filename}")

class BaseStrategy:
    def __init__(self, config: Config):
        self.config = config
        self.indicators = TechnicalIndicators()
        self.name = self.__class__.__name__
    
    def generate_signal(self, candles: List[Candle]) -> Dict[str, Any]:
        raise NotImplementedError("Strategy must implement generate_signal method")
    
    def analyze_market_condition(self, candles: List[Candle]) -> MarketCondition:
        """Analyze current market condition"""
        if len(candles) < 50:
            return MarketCondition(
                volatility=0, 
                trend_strength=0, 
                volume_ratio=0, 
                market_regime=MarketRegime.UNKNOWN
            )
        
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        
        # Calculate volatility (ATR normalized by price)
        atr = self.indicators.calculate_atr(candles, 14)
        current_atr = atr[-1] if atr and len(atr) > 0 else 0
        volatility = (current_atr / closes[-1]) * 100 if closes[-1] > 0 else 0
        
        # Calculate trend strength (slope of linear regression)
        if len(closes) >= 20:
            x = np.arange(len(closes[-20:]))
            y = np.array(closes[-20:])
            slope, _, r_value, _, _ = linregress(x, y)
            trend_strength = abs(slope / y[0]) * 100 * 100  # Normalized percentage
            r_squared = r_value ** 2  # Coefficient of determination
        else:
            trend_strength = 0
            r_squared = 0
        
        # Calculate volume ratio (current volume vs MA)
        if len(volumes) >= 20:
            volume_ma = sum(volumes[-20:-1]) / 19 if len(volumes) >= 20 else volumes[-1]
            volume_ratio = volumes[-1] / volume_ma if volume_ma > 0 else 1
        else:
            volume_ratio = 1
        
        # Determine market regime
        if volatility > 2.0:
            market_regime = MarketRegime.HIGH_VOLATILITY
        elif trend_strength > 1.0 and r_squared > 0.6:
            market_regime = MarketRegime.TRENDING
        elif trend_strength < 0.2 and volatility < 0.5:
            market_regime = MarketRegime.LOW_VOLATILITY
        else:
            market_regime = MarketRegime.MEAN_REVERTING
        
        return MarketCondition(
            volatility=volatility,
            trend_strength=trend_strength,
            volume_ratio=volume_ratio,
            market_regime=market_regime
        )

class EmaRsiAtrStrategy(BaseStrategy):
    def generate_signal(self, candles: List[Candle]) -> Dict[str, Any]:
        if len(candles) < max(self.config.strategy.ema_slow, self.config.strategy.rsi_period, self.config.strategy.atr_period) + 1:
            return {"signal": "HOLD", "confidence": 0, "atr": 0, "price": 0, "strategy": self.name}
        
        # Extract closing prices
        closes = [candle.close for candle in candles]
        # Calculate indicators
        ema_fast = self.indicators.calculate_ema(closes, self.config.strategy.ema_fast)
        ema_slow = self.indicators.calculate_ema(closes, self.config.strategy.ema_slow)
        rsi = self.indicators.calculate_rsi(closes, self.config.strategy.rsi_period)
        atr = self.indicators.calculate_atr(candles, self.config.strategy.atr_period)
        
        current_price = closes[-1]
        current_rsi = rsi[-1] if rsi else 0
        current_atr = atr[-1] if atr else 0
        ema_fast_current = ema_fast[-1] if ema_fast else 0
        ema_slow_current = ema_slow[-1] if ema_slow else 0
        
        signal = "HOLD"
        confidence = 0
        
        if ema_fast_current > ema_slow_current and current_rsi < self.config.strategy.rsi_overbought:
            signal = "LONG"
            # Confidence based on RSI distance from oversold and EMA crossover strength
            rsi_confidence = (self.config.strategy.rsi_oversold - current_rsi) / self.config.strategy.rsi_oversold
            ema_confidence = (ema_fast_current - ema_slow_current) / ema_slow_current * 100
            confidence = min(0.9, (rsi_confidence + ema_confidence / 10) / 2)
        elif ema_fast_current < ema_slow_current and current_rsi > self.config.strategy.rsi_overbought:
            signal = "SHORT"
            rsi_confidence = (current_rsi - self.config.strategy.rsi_overbought) / (100 - self.config.strategy.rsi_overbought)
            ema_confidence = (ema_slow_current - ema_fast_current) / ema_fast_current * 100
            confidence = min(0.9, (rsi_confidence + ema_confidence / 10) / 2)
        
        return {"signal": signal, "confidence": max(confidence, 0), "atr": current_atr, "price": current_price, "strategy": self.name}

# Other strategy implementations would follow similar patterns with improvements

class AIMLStrategy(BaseStrategy):
    def __init__(self, config: Config):
        super().__init__(config)
        self.predictor = MLSignalPredictor()
        # Try to load pre-trained model
        self.predictor.load_model("ml_model.pkl")

    def generate_signal(self, candles: List[Candle]) -> Dict[str, Any]:
        if len(candles) < 60:
            return {"signal": "HOLD", "confidence": 0, "atr": 0, "price": 0, "strategy": self.name}

        features = self.predictor.prepare_features(candles)
        if features is None:
            return {"signal": "HOLD", "confidence": 0, "atr": 0, "price": 0, "strategy": self.name}

        predictions = self.predictor.predict(features)
        last_pred = predictions[-1] if len(predictions) > 0 else 0
        signal = "LONG" if last_pred > 0.5 else "SHORT"
        confidence = abs(last_pred - 0.5) * 2  # Convert to [0,1] range
        atr = self.indicators.calculate_atr(candles, self.config.strategy.atr_period)[-1]
        current_price = candles[-1].close
        return {"signal": signal, "confidence": confidence, "atr": atr, "price": current_price, "strategy": self.name}

class EnsembleStrategy(BaseStrategy):
    def __init__(self, config: Config):
        super().__init__(config)
        self.strategies = [
            EmaRsiAtrStrategy(config),
            # Add other strategies here
            AIMLStrategy(config)
        ]
        self.weights = [0.3, 0.7]  # Weight for each strategy

    def generate_signal(self, candles: List[Candle]) -> Dict[str, Any]:
        signals = [s.generate_signal(candles) for s in self.strategies]
        
        # Weighted voting based on confidence
        long_score = 0
        short_score = 0
        
        for i, sig in enumerate(signals):
            if sig["signal"] == "LONG":
                long_score += sig["confidence"] * self.weights[i]
            elif sig["signal"] == "SHORT":
                short_score += sig["confidence"] * self.weights[i]
        
        signal = "HOLD"
        confidence = 0
        
        if long_score > short_score and long_score > 0.3:
            signal = "LONG"
            confidence = long_score
        elif short_score > long_score and short_score > 0.3:
            signal = "SHORT"
            confidence = short_score
        
        return {"signal": signal, "confidence": confidence, "atr": 0, "price": candles[-1].close, "strategy": self.name}

def create_strategy(config: Config) -> BaseStrategy:
    strategy_map = {
        "EMA_RSI_ATR": EmaRsiAtrStrategy,
        "SCALPING": ScalpingStrategy,
        "MEAN_REVERSION": MeanReversionStrategy,
        "MULTI_INDICATOR": MultiIndicatorStrategy,
        "AI_ML": AIMLStrategy,
        "ENSEMBLE": EnsembleStrategy
    }
    
    strategy_class = strategy_map.get(config.strategy.strategy_type, EmaRsiAtrStrategy)
    return strategy_class(config)