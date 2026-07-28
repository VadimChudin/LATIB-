import asyncio
import logging
from core.engine import BacktestEngine
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("Test.Backtest")

async def test_engine():
    logger.info("Initializing Backtest Engine for test run...")
    engine = BacktestEngine(data_path="data/test_active_config.json")
    
    # Run the cycle
    await engine.execute_optimization_cycle(top_n=2)
    
if __name__ == "__main__":
    asyncio.run(test_engine())
