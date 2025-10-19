import asyncio
import sys
import logging
from config import Config
from futures_trading_bot import FuturesTradingBot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('futures_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FuturesBot')

config = Config()

async def main():
    # Check if we should run backtest
    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        # Run backtest
        symbol = sys.argv[2] if len(sys.argv) > 2 else config.symbols[0]
        
        # Load historical data (this would need to be implemented)
        # For now, we'll just print a message
        print(f"Backtest mode for {symbol} - historical data loading not implemented")
        return
    
    # Run live trading bot
    bot = FuturesTradingBot(config)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())