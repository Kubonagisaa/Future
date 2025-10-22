import numpy as np
import talib
from typing import List, Dict, Tuple
import logging
from ..data_models import Candle

logger = logging.getLogger('FuturesBot')

class TechnicalIndicators:
    # Cache for indicator calculations
    _cache = {}
    _cache_size = 1000
    
    @classmethod
    def _get_cache_key(cls, method_name: str, *args) -> str:
        """Generate a cache key for indicator calculation"""
        key_parts = [method_name]
        for arg in args:
            if isinstance(arg, list) and len(arg) > 0:
                # Use the last value and length for lists
                key_parts.append(f"{len(arg)}_{arg[-1]}")
            else:
                key_parts.append(str(arg))
        return "_".join(key_parts)
    
    @classmethod
    def _check_cache(cls, key: str):
        """Check if result is in cache"""
        if key in cls._cache:
            # Move to end to mark as recently used
            result = cls._cache.pop(key)
            cls._cache[key] = result
            return result
        return None
    
    @classmethod
    def _add_to_cache(cls, key: str, result):
        """Add result to cache"""
        if len(cls._cache) >= cls._cache_size:
            # Remove oldest item
            cls._cache.pop(next(iter(cls._cache)))
        cls._cache[key] = result
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int, cache: bool = True) -> List[float]:
        """Calculate EMA with optional caching"""
        if cache:
            cache_key = TechnicalIndicators._get_cache_key("ema", prices, period)
            cached = TechnicalIndicators._check_cache(cache_key)
            if cached is not None:
                return cached
        
        if len(prices) < period:
            result = [0.0] * len(prices)
        else:
            prices_arr = np.array(prices)
            ema = talib.EMA(prices_arr, timeperiod=period)
            result = ema.tolist()
        
        if cache:
            TechnicalIndicators._add_to_cache(cache_key, result)
        
        return result

    @staticmethod
    def calculate_rsi(prices: List[float], period: int, cache: bool = True) -> List[float]:
        """Calculate RSI with optional caching"""
        if cache:
            cache_key = TechnicalIndicators._get_cache_key("rsi", prices, period)
            cached = TechnicalIndicators._check_cache(cache_key)
            if cached is not None:
                return cached
        
        if len(prices) < period:
            result = [0.0] * len(prices)
        else:
            prices_arr = np.array(prices)
            rsi = talib.RSI(prices_arr, timeperiod=period)
            result = rsi.tolist()
        
        if cache:
            TechnicalIndicators._add_to_cache(cache_key, result)
        
        return result

    @staticmethod
    def calculate_macd(prices: List[float], fast_period: int, slow_period: int, 
                      signal_period: int, cache: bool = True) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD with optional caching"""
        if cache:
            cache_key = TechnicalIndicators._get_cache_key("macd", prices, fast_period, slow_period, signal_period)
            cached = TechnicalIndicators._check_cache(cache_key)
            if cached is not None:
                return cached
        
        if len(prices) < slow_period + signal_period:
            result = ([0.0] * len(prices), [0.0] * len(prices), [0.0] * len(prices))
        else:
            prices_arr = np.array(prices)
            macd, signal, hist = talib.MACD(prices_arr, fastperiod=fast_period, 
                                          slowperiod=slow_period, signalperiod=signal_period)
            result = (macd.tolist(), signal.tolist(), hist.tolist())
        
        if cache:
            TechnicalIndicators._add_to_cache(cache_key, result)
        
        return result

    @staticmethod
    def calculate_atr(candles: List[Candle], period: int, cache: bool = True) -> List[float]:
        """Calculate ATR with optional caching"""
        if cache:
            cache_key = TechnicalIndicators._get_cache_key("atr", [c.to_dict() for c in candles], period)
            cached = TechnicalIndicators._check_cache(cache_key)
            if cached is not None:
                return cached
        
        if len(candles) < period + 1:
            result = [0.0] * len(candles)
        else:
            highs = np.array([c.high for c in candles])
            lows = np.array([c.low for c in candles])
            closes = np.array([c.close for c in candles])
            
            # Calculate True Range
            tr1 = highs[1:] - lows[1:]
            tr2 = abs(highs[1:] - closes[:-1])
            tr3 = abs(lows[1:] - closes[:-1])
            tr = np.maximum.reduce([tr1, tr2, tr3])
            
            # Calculate ATR
            atr = np.zeros(len(candles))
            atr[period] = np.mean(tr[:period])
            
            for i in range(period + 1, len(candles)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
            
            result = atr.tolist()
        
        if cache:
            TechnicalIndicators._add_to_cache(cache_key, result)
        
        return result

    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int, 
                                 std_dev: float = 2.0, cache: bool = True) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands with optional caching"""
        if cache:
            cache_key = TechnicalIndicators._get_cache_key("bb", prices, period, std_dev)
            cached = TechnicalIndicators._check_cache(cache_key)
            if cached is not None:
                return cached
        
        if len(prices) < period:
            result = ([0.0] * len(prices), [0.0] * len(prices), [0.0] * len(prices))
        else:
            prices_arr = np.array(prices)
            upper, middle, lower = talib.BBANDS(prices_arr, timeperiod=period, 
                                               nbdevup=std_dev, nbdevdn=std_dev)
            result = (upper.tolist(), middle.tolist(), lower.tolist())
        
        if cache:
            TechnicalIndicators._add_to_cache(cache_key, result)
        
        return result

    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 20, cache: bool = True) -> List[float]:
        """Calculate volatility (standard deviation of returns) with optional caching"""
        if cache:
            cache_key = TechnicalIndicators._get_cache_key("volatility", prices, period)
            cached = TechnicalIndicators._check_cache(cache_key)
            if cached is not None:
                return cached
        
        if len(prices) < period:
            result = [0.0] * len(prices)
        else:
            returns = np.diff(prices) / prices[:-1]
            volatility = np.zeros(len(prices))
            
            for i in range(period, len(prices)):
                volatility[i] = np.std(returns[i-period:i])
            
            result = volatility.tolist()
        
        if cache:
            TechnicalIndicators._add_to_cache(cache_key, result)
        
        return result

    @staticmethod
    def calculate_correlation(returns1: List[float], returns2: List[float]) -> float:
        """Calculate correlation between two return series"""
        if len(returns1) != len(returns2) or len(returns1) < 2:
            return 0.0
        
        returns1_arr = np.array(returns1)
        returns2_arr = np.array(returns2)
        
        correlation = np.corrcoef(returns1_arr, returns2_arr)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    @staticmethod
    def calculate_momentum(prices: List[float], period: int = 10, cache: bool = True) -> List[float]:
        """Calculate momentum indicator with optional caching"""
        if cache:
            cache_key = TechnicalIndicators._get_cache_key("momentum", prices, period)
            cached = TechnicalIndicators._check_cache(cache_key)
            if cached is not None:
                return cached
        
        if len(prices) < period:
            result = [0.0] * len(prices)
        else:
            momentum = np.zeros(len(prices))
            for i in range(period, len(prices)):
                momentum[i] = prices[i] - prices[i-period]
            result = momentum.tolist()
        
        if cache:
            TechnicalIndicators._add_to_cache(cache_key, result)
        
        return result

    @staticmethod
    def calculate_obv(prices: List[float], volumes: List[float], cache: bool = True) -> List[float]:
        """Calculate On-Balance Volume with optional caching"""
        if cache:
            cache_key = TechnicalIndicators._get_cache_key("obv", prices, volumes)
            cached = TechnicalIndicators._check_cache(cache_key)
            if cached is not None:
                return cached
        
        if len(prices) != len(volumes) or len(prices) < 2:
            result = [0.0] * len(prices)
        else:
            obv = np.zeros(len(prices))
            obv[0] = volumes[0]
            
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    obv[i] = obv[i-1] + volumes[i]
                elif prices[i] < prices[i-1]:
                    obv[i] = obv[i-1] - volumes[i]
                else:
                    obv[i] = obv[i-1]
            
            result = obv.tolist()
        
        if cache:
            TechnicalIndicators._add_to_cache(cache_key, result)
        
        return result

    @staticmethod
    def clear_cache():
        """Clear the indicator cache"""
        TechnicalIndicators._cache.clear()