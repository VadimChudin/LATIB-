import os
import time
import logging

logger = logging.getLogger("AutoCore.Utils")

import asyncio
import random

_COOLDOWN_UNTIL = 0

def set_global_cooldown(seconds=60):
    """Sets a project-wide timestamp to pause all API-heavy loops, with 20% jitter."""
    global _COOLDOWN_UNTIL
    # Add +/- 20% jitter to prevent synchronized retry spikes
    jitter = random.uniform(0.8, 1.2)
    actual_seconds = int(seconds * jitter)
    _COOLDOWN_UNTIL = time.time() + actual_seconds
    logger.warning(f"⚠️ GLOBAL COOLDOWN ACTIVATED. Pausing all requests for ~{actual_seconds}s.")

def is_in_cooldown():
    """Checks if the system is currently in a defensive cooling period."""
    return time.time() < _COOLDOWN_UNTIL

def get_cooldown_remaining():
    """Returns seconds remaining in the cooldown period."""
    return max(0, int(_COOLDOWN_UNTIL - time.time()))

async def wait_for_cooldown(tag="Task"):
    """Helper for loops to sleep until cooldown is over, with additional staggered jitter."""
    while is_in_cooldown():
        rem = get_cooldown_remaining()
        # Staggered wake-ups: each task waits for the full duration + its own small jitter
        stagger = random.uniform(1.0, 5.0)
        logger.info(f"[{tag}] Defensive wait: {rem}s remaining (staggered +{stagger:.1f}s)...")
        await asyncio.sleep(min(rem + stagger, 30))

def strip_proxies():
    """Definitively removes all proxy-related environment variables to prevent library interference."""
    pvars = ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']
    for pvar in pvars:
        if pvar in os.environ:
            os.environ.pop(pvar, None)
            
import re

def recursive_url_rewrite(data, target_host):
    """
    Recursively replaces any hostname in a URL or dict of URLs with the target_host,
    BUT only if the target_host is appropriate for the subdomain.
    """
    if isinstance(data, str):
        if '://' in data:
            # If the URL is already pointing to fapi, and we are giving it a host like 'fapi.binance.me', it's fine.
            # But if a URL is 'api.binance.com' (Spot) and we replace it with 'fapi.binance.me' (Futures host),
            # the path /api/v3 will 404.
            
            # Smart logic: if target_host starts with 'fapi', only replace URLs that are already fapi.
            # Otherwise, replace standard api.
            current_host = data.split('://')[1].split('/')[0]
            
            if 'fapi' in target_host and 'fapi' not in current_host:
                # Don't point Spot API to a Futures-only host
                return data
            
            return re.sub(r'(https?://)[^/]+', r'\1' + target_host, data)
        
        if '.' in data and '/' not in data and ' ' not in data:
            return target_host
        return data
    elif isinstance(data, dict):
        return {k: recursive_url_rewrite(v, target_host) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_url_rewrite(i, target_host) for i in data]
    return data

def get_browser_headers():
    """Returns high-reputation headers to bypass WAF/403 blocks."""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Referer': 'https://www.binance.com/en/futures',
    }
