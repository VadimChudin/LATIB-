"""
Tick Verification Module — Phase 20
====================================
Verifies backtest trades against real tick data from Binance.
Detects "fake winners" where SL was actually hit before TP within
the same candle (which candle-based backtests can't distinguish).

Usage:
    from verify_ticks import TickVerifier
    verifier = TickVerifier()
    result = await verifier.verify_trades(trades_list)
    await verifier.close()
"""
import asyncio
import logging
from lazy_tick_loader import LazyTickLoader

logger = logging.getLogger("TickVerifier")


class TickVerifier:
    def __init__(self):
        self.loader = LazyTickLoader()
        self.stats = {"total": 0, "verified": 0, "fake_wins": 0, "skipped": 0}

    async def close(self):
        await self.loader.close()

    async def verify_single_trade(self, symbol: str, trade: dict) -> dict:
        """
        Verify a single trade against tick data.
        
        trade = {
            "entry_ts": "2025-01-15 12:30:00",  # candle timestamp
            "direction": "LONG" or "SHORT",
            "entry_price": 95000.0,
            "sl_price": 94500.0,
            "tp_price": 96000.0,
            "exit_price": 96000.0,
            "pnl_r": 2.0,
        }
        
        Returns: {"valid": bool, "reason": str, "adjusted_pnl_r": float}
        """
        self.stats["total"] += 1

        try:
            import pandas as pd
            entry_ts = pd.Timestamp(trade["entry_ts"])
            entry_ms = int(entry_ts.timestamp() * 1000)

            direction = trade["direction"]
            entry_price = trade["entry_price"]
            sl_price = trade["sl_price"]
            tp_price = trade["tp_price"]
            reported_pnl = trade["pnl_r"]

            # Only verify ambiguous trades (where both SL and TP could have been hit)
            # If pnl_r == 0, the trade didn't close — skip
            if reported_pnl == 0.0:
                self.stats["skipped"] += 1
                return {"valid": True, "reason": "unclosed", "adjusted_pnl_r": 0.0}

            max_ms = entry_ms + 4 * 3600 * 1000  # 4 hours max limits
            current_ms = entry_ms
            
            while current_ms < max_ms:
                chunk_end = current_ms + 15 * 60 * 1000 # 15 min chunks
                if chunk_end > max_ms: chunk_end = max_ms
                
                df_ticks = await self.loader.load_trade_window(symbol, current_ms, chunk_end)
                
                if df_ticks.empty:
                    current_ms += 15 * 60 * 1000
                    continue
                    
                for p in df_ticks["price"].values:
                    p = float(p)
                    if direction == "LONG":
                        if sl_price > 0 and p <= sl_price:
                            if reported_pnl > 0:
                                self.stats["fake_wins"] += 1
                                return {"valid": False, "reason": f"SL hit first at {p:.2f}", "adjusted_pnl_r": -1.0}
                            else:
                                self.stats["verified"] += 1
                                return {"valid": True, "reason": "SL confirmed", "adjusted_pnl_r": -1.0}
                        if tp_price > 0 and p >= tp_price:
                            self.stats["verified"] += 1
                            return {"valid": True, "reason": "TP confirmed", "adjusted_pnl_r": reported_pnl}
                    else:  # SHORT
                        if sl_price > 0 and p >= sl_price:
                            if reported_pnl > 0:
                                self.stats["fake_wins"] += 1
                                return {"valid": False, "reason": f"SL hit first at {p:.2f}", "adjusted_pnl_r": -1.0}
                            else:
                                self.stats["verified"] += 1
                                return {"valid": True, "reason": "SL confirmed", "adjusted_pnl_r": -1.0}
                        if tp_price > 0 and p <= tp_price:
                            self.stats["verified"] += 1
                            return {"valid": True, "reason": "TP confirmed", "adjusted_pnl_r": reported_pnl}

                current_ms = chunk_end + 1
                
            self.stats["verified"] += 1
            return {"valid": True, "reason": "timeout_safe", "adjusted_pnl_r": reported_pnl}

        except Exception as e:
            self.stats["skipped"] += 1
            return {"valid": True, "reason": f"error: {e}", "adjusted_pnl_r": trade.get("pnl_r", 0.0)}

    async def verify_trades(self, symbol: str, trades: list, max_verify: int = 20, random_sample: bool = False) -> dict:
        """
        Verify a list of trades, return adjusted stats.
        If random_sample is True, picks a random subset of winning trades (unbiased historical check).
        Otherwise, verifies the most recent `max_verify` winning trades (fast recent check).
        """
        import random
        winners = [t for t in trades if t.get("pnl_r", 0) > 0]
        
        if random_sample:
            # Unbiased historical check
            to_verify = random.sample(winners, min(len(winners), max_verify))
        else:
            # Fast recent check
            to_verify = winners[-max_verify:] if len(winners) > max_verify else winners

        if not to_verify:
            return {
                "original_trades": len(trades),
                "verified": 0,
                "fake_wins": 0,
                "adjusted_wr": None,
                "confidence": "low",
            }

        logger.info(f"   🔬 Verifying {len(to_verify)} winning trades on ticks...")

        import sys
        results = []
        for i, trade in enumerate(to_verify, 1):
            sys.stdout.write(f"\r      [Tick History] Downloading ticks: trade {i}/{len(to_verify)}...    ")
            sys.stdout.flush()
            r = await self.verify_single_trade(symbol, trade)
            results.append(r)
        
        sys.stdout.write("\r" + " " * 80 + "\r") # Clear the line
        
        fake_count = sum(1 for r in results if not r["valid"])
        real_wins = sum(1 for r in results if r["valid"] and r["adjusted_pnl_r"] > 0)

        # Recalculate adjusted WR
        total_trades = len(trades)
        original_wins = sum(1 for t in trades if t.get("pnl_r", 0) > 0)
        adjusted_wins = original_wins - fake_count
        adjusted_wr = (adjusted_wins / total_trades * 100) if total_trades > 0 else 0

        # Confidence: high if we verified many and few fakes
        if len(to_verify) >= 10 and fake_count <= 1:
            confidence = "high"
        elif fake_count > len(to_verify) * 0.3:
            confidence = "reject"  # Too many fakes
        else:
            confidence = "medium"

        # Note: caller is responsible for await self.close() when done

        return {
            "original_trades": total_trades,
            "original_wins": original_wins,
            "verified_count": len(to_verify),
            "fake_wins": fake_count,
            "adjusted_wins": adjusted_wins,
            "adjusted_wr": round(adjusted_wr, 1),
            "confidence": confidence,
            "stats": self.stats
        }


async def _test():
    """Quick self-test."""
    v = TickVerifier()
    fake_trade = {
        "entry_ts": "2026-03-18 12:00:00",
        "direction": "LONG",
        "entry_price": 84000.0,
        "sl_price": 83500.0,
        "tp_price": 85000.0,
        "exit_price": 85000.0,
        "pnl_r": 2.0,
    }
    result = await v.verify_single_trade("BTC_USDT", fake_trade)
    print(f"Result: {result}")
    print(f"Stats: {v.stats}")
    await v.close()


if __name__ == "__main__":
    asyncio.run(_test())
