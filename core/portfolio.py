import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from ..config import Config
from ..data_models import Candle, Position, PortfolioItem
from ..utils.indicators import TechnicalIndicators
from .exchange import FuturesExchange

logger = logging.getLogger('FuturesBot')

class PortfolioManager:
    def __init__(self, config: Config, exchange: FuturesExchange):
        self.config = config
        self.exchange = exchange
        self.weights: Dict[str, float] = {}
        self.correlations: Dict[Tuple[str, str], float] = {}
        self.volatilities: Dict[str, float] = {}

    async def update_portfolio_metrics(self, all_candles: Dict[str, List[Candle]]) -> None:
        """Update portfolio metrics including correlations and volatilities"""
        returns = {}
        volatilities = {}
        
        # Calculate returns and volatilities for each symbol
        for symbol, candles in all_candles.items():
            if len(candles) >= 20:  # Need enough data for meaningful calculations
                closes = [c.close for c in candles]
                returns[symbol] = np.diff(closes) / closes[:-1]
                volatilities[symbol] = np.std(returns[symbol][-20:]) if len(returns[symbol]) >= 20 else 0
        
        # Calculate correlations between all pairs
        symbols = list(returns.keys())
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:  # Avoid duplicate calculations
                    if len(returns[sym1]) == len(returns[sym2]) and len(returns[sym1]) > 0:
                        correlation = TechnicalIndicators.calculate_correlation(returns[sym1], returns[sym2])
                        self.correlations[(sym1, sym2)] = correlation
                        self.correlations[(sym2, sym1)] = correlation  # Symmetric
        
        # Update volatilities
        self.volatilities = volatilities
        
        # Calculate optimal weights using Markowitz portfolio optimization
        self.weights = self._calculate_optimal_weights(returns, volatilities)

    def _calculate_optimal_weights(self, returns: Dict[str, List[float]], 
                                 volatilities: Dict[str, float]) -> Dict[str, float]:
        """Calculate optimal portfolio weights using Markowitz optimization"""
        symbols = list(returns.keys())
        n = len(symbols)
        
        if n == 0:
            return {}
        
        # Create covariance matrix
        cov_matrix = np.zeros((n, n))
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i == j:
                    # Variance
                    cov_matrix[i, j] = volatilities[sym1] ** 2 if sym1 in volatilities else 0
                else:
                    # Covariance = correlation * std_dev1 * std_dev2
                    correlation = self.correlations.get((sym1, sym2), 0)
                    std_dev1 = volatilities[sym1] if sym1 in volatilities else 0
                    std_dev2 = volatilities[sym2] if sym2 in volatilities else 0
                    cov_matrix[i, j] = correlation * std_dev1 * std_dev2
        
        # Calculate expected returns (simple mean of recent returns)
        expected_returns = np.zeros(n)
        for i, symbol in enumerate(symbols):
            if symbol in returns and len(returns[symbol]) > 0:
                expected_returns[i] = np.mean(returns[symbol][-20:]) if len(returns[symbol]) >= 20 else 0
        
        # Simple portfolio optimization (maximize Sharpe ratio)
        try:
            # Inverse of covariance matrix
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            
            # Calculate optimal weights
            ones = np.ones(n)
            numerator = np.dot(inv_cov_matrix, expected_returns)
            denominator = np.dot(ones, np.dot(inv_cov_matrix, ones))
            
            if denominator != 0:
                weights = numerator / denominator
            else:
                weights = np.ones(n) / n  # Equal weighting if calculation fails
        except:
            # Fallback to equal weighting if optimization fails
            weights = np.ones(n) / n
        
        # Normalize weights and convert to dictionary
        weights = np.maximum(weights, 0)  # No short selling
        weights /= np.sum(weights)  # Normalize to sum to 1
        
        return {symbol: weights[i] for i, symbol in enumerate(symbols)}

    def get_portfolio_allocation(self, signals: Dict[str, Any], 
                               current_positions: Dict[str, Position]) -> Dict[str, float]:
        """Get portfolio allocation based on signals and current positions"""
        allocation = {}
        active_symbols = [s for s in signals if signals[s]["signal"] != "HOLD"]
        n_active = len(active_symbols)
        
        if n_active == 0:
            return allocation
        
        # If we have optimized weights, use them
        if self.weights:
            total_weight = sum(self.weights.get(s, 0) for s in active_symbols)
            if total_weight > 0:
                for symbol in active_symbols:
                    allocation[symbol] = self.weights.get(symbol, 0) / total_weight
                return allocation
        
        # Fallback: equal weighting with consideration for existing positions
        base_weight = 1.0 / n_active
        
        for symbol in active_symbols:
            # Reduce weight if already have a position
            if symbol in current_positions:
                allocation[symbol] = base_weight * 0.5  # Half weight for existing positions
            else:
                allocation[symbol] = base_weight
        
        # Normalize to ensure sum is 1
        total = sum(allocation.values())
        if total > 0:
            for symbol in allocation:
                allocation[symbol] /= total
        
        return allocation

    def should_enter_trade(self, symbol: str, signals: Dict[str, Any], 
                          current_positions: Dict[str, Position]) -> bool:
        """Check if we should enter a trade based on portfolio constraints"""
        # Prevent more than one position per symbol
        if symbol in current_positions:
            return False
        
        # Check if we have a valid signal
        if symbol not in signals or signals[symbol]["signal"] == "HOLD":
            return False
        
        # Check correlation with existing positions
        for existing_symbol in current_positions:
            correlation = self.correlations.get((symbol, existing_symbol), 0)
            if abs(correlation) > self.config.portfolio.max_correlation:
                logger.info(f"High correlation ({correlation:.2f}) between {symbol} and {existing_symbol}, skipping trade")
                return False
        
        # Check diversification constraint
        if len(current_positions) >= self.config.portfolio.min_diversification:
            logger.info(f"Already have {len(current_positions)} positions, skipping for diversification")
            return False
        
        return True

    def calculate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk for the entire portfolio"""
        if not self.weights or not self.volatilities:
            return 0
        
        # Calculate portfolio variance
        portfolio_variance = 0
        symbols = list(self.weights.keys())
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                weight1 = self.weights[sym1]
                weight2 = self.weights[sym2]
                
                if i == j:
                    # Variance contribution
                    portfolio_variance += weight1 ** 2 * self.volatilities[sym1] ** 2
                else:
                    # Covariance contribution
                    correlation = self.correlations.get((sym1, sym2), 0)
                    portfolio_variance += weight1 * weight2 * correlation * self.volatilities[sym1] * self.volatilities[sym2]
        
        # Calculate portfolio standard deviation
        portfolio_std_dev = np.sqrt(portfolio_variance)
        
        # Calculate VaR (parametric method)
        var = portfolio_std_dev * np.percentile(np.random.normal(0, 1, 10000), (1 - confidence_level) * 100)
        
        return var * self.exchange.equity  # Convert to dollar amount

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        status = {
            "equity": self.exchange.equity,
            "positions": len(self.exchange.positions),
            "weights": self.weights,
            "var_95": self.calculate_portfolio_var(0.95),
            "var_99": self.calculate_portfolio_var(0.99),
            "volatilities": self.volatilities,
            "correlations": self.correlations
        }
        
        return status