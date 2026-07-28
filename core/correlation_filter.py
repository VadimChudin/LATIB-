"""
Correlation Filter
==================
Prevents multiple strategies from opening conflicting trades on the same symbol.

Rules:
- Max 2 positions on same symbol in same direction
- No LONG + SHORT on same symbol at same time
- Cross-strategy correlation check: if 2+ strategies agree → stronger signal
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CorrelationFilter:
    """Track open positions across strategies and filter conflicts"""
    
    def __init__(self, max_same_direction: int = 2):
        self.max_same_direction = max_same_direction
        self.open_positions: Dict[str, List[dict]] = {}  # symbol → [{strategy, direction}]
    
    def can_open(self, symbol: str, direction: str, strategy: str, klines: Optional[Dict[str, 'pd.DataFrame']] = None) -> bool:
        """Check if a new position can be opened, using real-time Pearson correlation."""
        positions = self.open_positions.get(symbol, [])
        
        # Rule 1: No opposite direction on same symbol
        if positions:
            opposite = "SHORT" if direction == "LONG" else "LONG"
            if any(p['direction'] == opposite for p in positions):
                logger.debug(f"❌ Blocked {strategy} {direction} on {symbol}: opposite position exists")
                return False
            
            # Rule 2: Max same direction positions on EXACT SAME symbol
            same_dir_count = sum(1 for p in positions if p['direction'] == direction)
            if same_dir_count >= self.max_same_direction:
                logger.debug(f"❌ Blocked {strategy} {direction} on {symbol}: max {self.max_same_direction} reached")
                return False
            
            # Rule 3: Don't duplicate same strategy on same symbol
            if any(p['strategy'] == strategy for p in positions):
                logger.debug(f"❌ Blocked {strategy}: already has position on {symbol}")
                return False

        # Rule 4: Advanced Pearson Correlation Check across ALL open symbols
        if klines and symbol in klines:
            import pandas as pd
            import numpy as np
            
            try:
                cand_df = klines[symbol]
                if len(cand_df) >= 50:
                    cand_returns = cand_df['close'].pct_change().tail(50).values
                    
                    for open_sym, open_pos_list in self.open_positions.items():
                        if open_sym == symbol or open_sym not in klines:
                            continue
                            
                        open_df = klines[open_sym]
                        if len(open_df) >= 50:
                            open_returns = open_df['close'].pct_change().tail(50).values
                            
                            # Align lengths just in case
                            min_len = min(len(cand_returns), len(open_returns))
                            c_ret = cand_returns[-min_len:]
                            o_ret = open_returns[-min_len:]
                            
                            # Calculate Pearson correlation coefficient
                            # Ignore NaNs
                            valid = ~np.isnan(c_ret) & ~np.isnan(o_ret)
                            if sum(valid) > 10:
                                corr = np.corrcoef(c_ret[valid], o_ret[valid])[0, 1]
                                
                                # Check if any open position on this correlated symbol is in the SAME direction
                                same_dir_exists = any(p['direction'] == direction for p in open_pos_list)
                                
                                # If highly correlated (> 0.75) and we are trying to go the same way -> BLOCK
                                if corr > 0.75 and same_dir_exists:
                                    logger.info(f"🔗 Correlation Block: {symbol} is {corr*100:.1f}% correlated with open {open_sym} ({direction}). Skipping to avoid risk-doubling.")
                                    return False
            except Exception as e:
                logger.warning(f"Correlation check failed for {symbol}: {e}")
                
        return True
    
    def open_position(self, symbol: str, direction: str, strategy: str):
        """Register a new open position"""
        if symbol not in self.open_positions:
            self.open_positions[symbol] = []
        self.open_positions[symbol].append({
            'strategy': strategy,
            'direction': direction,
        })
    
    def close_position(self, symbol: str, strategy: str):
        """Remove a closed position"""
        if symbol in self.open_positions:
            self.open_positions[symbol] = [
                p for p in self.open_positions[symbol]
                if p['strategy'] != strategy
            ]
            if not self.open_positions[symbol]:
                del self.open_positions[symbol]
    
    def get_agreement_score(self, symbol: str, direction: str) -> int:
        """How many strategies agree on this direction? (confluence signal)"""
        positions = self.open_positions.get(symbol, [])
        return sum(1 for p in positions if p['direction'] == direction)
    
    def status(self) -> dict:
        """Current open position summary"""
        return {
            'total_positions': sum(len(v) for v in self.open_positions.values()),
            'symbols_active': len(self.open_positions),
            'positions': dict(self.open_positions),
        }
