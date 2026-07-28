"""
Risk Manager — Position Sizing & Kelly Criterion
================================================
Calculates optimal position sizes based on Account Equity, Stop Loss distance,
and the ML Model's probability of success (Kelly Criterion).
"""
import math
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        """
        config expectations:
          - max_risk_per_trade_pct: float (e.g., 0.10 for 10% max risk)
          - base_risk_pct: float (e.g., 0.02 for 2% base risk if Kelly is off)
          - use_kelly: bool (true to dynamically size based on P(win))
          - kelly_fraction: float (e.g., 0.5 for Half-Kelly, safer than Full Kelly)
          - max_leverage: float (e.g., 40.0)
        """
        self.max_risk_pct   = config.get('max_risk_per_trade_pct', 0.02)
        self.base_risk_pct  = config.get('base_risk_pct', 0.02)
        self.use_kelly      = config.get('use_kelly', True)
        self.kelly_fraction = config.get('kelly_fraction', 0.5) # Half-Kelly is standard for trading
        self.max_leverage   = config.get('max_leverage', 40.0)

    def calculate_position_size(self, 
                                account_equity: float, 
                                entry_price: float, 
                                sl_price: float, 
                                ml_prob_win: float = 0.5, 
                                avg_win_r: float = 1.0,
                                volatility_scalar: float = 1.0) -> Dict[str, Any]:
        """
        Calculates the exact position size (in base currency, e.g., BTC) to buy/sell.
        Returns a dictionary with sizing details.
        
        Math assumptions:
        - If ML probability is available, uses the Kelly Criterion:
          Kelly % (f*) = W - [(1 - W) / R]
          where W = Probability of Win, R = Risk/Reward ratio (Avg Win in our case)
        """
        if account_equity <= 0 or entry_price <= 0:
            return {'size': 0.0, 'risk_amount': 0.0, 'leverage_needed': 0.0}

        # 0. Fixed Dollar Amount Override (from Aegis UI Settings)
        trade_amount = float(os.getenv('TRADE_AMOUNT', '0'))
        max_leverage = float(os.getenv('MAX_LEVERAGE', str(self.max_leverage)))
        
        # Kelly Criterion Formula processing
        w = ml_prob_win
        r = avg_win_r  # Average Risk:Reward ratio
        
        if self.use_kelly and w > 0 and r > 0:
            full_kelly = w - ((1 - w) / r)
        else:
            full_kelly = self.base_risk_pct * 5 # Fallback flat fraction
            
        fractional_kelly = max(0.0, full_kelly * self.kelly_fraction * volatility_scalar)

        if trade_amount > 0:
            virtual_bank = trade_amount * max_leverage # This is the "Max Open Interest Notional" allowed
            
            # Treat the Kelly % directly as the "% of overall virtual bank to allocate to this position"
            allocation_pct = min(fractional_kelly, 1.0) # Cap at 100% of virtual bank
            
            if allocation_pct <= 0:
                return {'size': 0.0, 'risk_amount': 0.0, 'leverage_needed': 0.0, 'reason': 'kelly_zero'}

            notional_value = virtual_bank * allocation_pct
            size = notional_value / entry_price
            leverage_needed = max_leverage
                
            risk_amount = size * abs(entry_price - sl_price)
            logger.debug(f"Allocating {allocation_pct*100:.1f}% of Virtual Bank (${virtual_bank}) = ${notional_value} Notional")
            
            return {
                'size': size,
                'risk_amount': risk_amount,
                'leverage_needed': max_leverage  # Handled by UI bounds
            }

        # 1. Standard mode (No Fixed Amount): Calculate SL distance percentage
        sl_dist_pct = abs(entry_price - sl_price) / entry_price
        if sl_dist_pct == 0:
            logger.warning("SL price equals Entry price. Cannot calculate size.")
            return {'size': 0.0, 'risk_amount': 0.0, 'leverage_needed': 0.0}

        # 2. Determine Risk Percentage (Standard Account Equity Risk Model)
        risk_pct = min(fractional_kelly, self.max_risk_pct) if self.use_kelly else self.base_risk_pct

        # If Kelly says don't trade, return 0
        if risk_pct <= 0:
            return {'size': 0.0, 'risk_amount': 0.0, 'leverage_needed': 0.0, 'reason': 'kelly_zero'}

        # 3. Calculate absolute Dollar Risk
        dollar_risk = account_equity * risk_pct

        # 4. Calculate Position Size
        # Size = Dollar Risk / Difference in price (Entry - SL)
        # Because: (Entry - SL) * Size = Dollar Risk
        size = dollar_risk / abs(entry_price - sl_price)
        
        # 5. Calculate Notional Value & Required Leverage
        notional_value = size * entry_price
        leverage_needed = notional_value / account_equity
        
        # 6. Safety Check: Cap Leverage
        if leverage_needed > self.max_leverage:
            logger.warning(f"Calculated leverage ({leverage_needed:.1f}x) exceeds max ({self.max_leverage}x). Scaling down.")
            notional_value = account_equity * self.max_leverage
            size = notional_value / entry_price
            leverage_needed = self.max_leverage
            dollar_risk = size * abs(entry_price - sl_price) # Recalculate actual risk dollar amount
            risk_pct = dollar_risk / account_equity

        return {
            'size': round(size, 6),
            'risk_amount_usd': round(dollar_risk, 2),
            'risk_pct': round(risk_pct * 100, 2),
            'leverage_needed': round(leverage_needed, 2),
            'notional_value_usd': round(notional_value, 2)
        }
