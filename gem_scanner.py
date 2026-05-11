#!/usr/bin/env python3
"""
GEM SCANNER — 2-Stage Crypto Scanner
Finds coins passing ALL strict bot criteria

Stage 1: Fast pre-filter on CoinGecko top-500 (batch API)
Stage 2: Deep scan (full pipeline) on survivors only

Run: python gem_scanner.py
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime
from colorama import Fore, Style, init
init(autoreset=True)
import os as _ntfy_os

def send_ntfy(title, message, priority="default", tags="", topic_env="NTFY_TOPIC"):
    """Push notification via ntfy.sh.
    topic_env='NTFY_WATCH_TOPIC' for near-miss/watch alerts.
    topic_env='NTFY_TOPIC' for gem found alerts.
    """
    topic = _ntfy_os.environ.get(topic_env, "") or _ntfy_os.environ.get("NTFY_TOPIC", "")
    if not topic:
        print(f"[NTFY] ⚠️  NTFY_TOPIC not set — skipping: '{title}'")
        print(f"[NTFY]    Fix: export NTFY_TOPIC=your-topic-name")
        return
    try:
        import requests as _r
        resp = _r.post(f"https://ntfy.sh/{topic}",
                data=message.encode("utf-8"),
                headers={
                    "Title":    title.encode("utf-8"),
                    "Priority": priority,
                    "Tags":     tags or "",
                    "Content-Type": "text/plain; charset=utf-8",
                },
                timeout=8)
        if resp.status_code == 200:
            print(f"[NTFY] ✅ Sent: '{title}'")
        else:
            print(f"[NTFY] ❌ HTTP {resp.status_code}: '{title}'")
    except Exception as e:
        print(f"[NTFY] ❌ Error: {e}")


def get_liquidity_tier(book_usd):
    """Classify order book depth for position sizing guidance."""
    if not book_usd or book_usd <= 0:
        return "unknown",    "❓ Unknown",    None
    elif book_usd >= 1_000_000:
        return "deep",       "🟢 DEEP",       None
    elif book_usd >= 300_000:
        return "moderate",   "🟡 MODERATE",   5000
    elif book_usd >= 50_000:
        return "thin",       "🟠 THIN",       500
    else:
        return "paper_thin", "🔴 PAPER-THIN", 0

# ── IMPORT BOT FUNCTIONS ─────────────────────────────────────────
# Scanner reuses all functions from main bot
# Must be in same directory as crypto_trading_bot.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from crypto_trading_bot import (
        fetch_coin_data, compute_indicators, generate_signal_and_reasons,
        apply_volume_boost, apply_price_change_penalty, apply_zscore_boost,
        apply_exhaustion_penalty, apply_orderbook_boost, apply_chop_penalty,
        apply_breakout_boost, apply_wick_adjustment,
        train_ml_model, predict_ml_outcome, save_model, load_model,
        should_retrain_model, backtest_signals, check_obv_confirmation,
        apply_obv_ml_gate, analyze_order_book_liquidity, _is_strong_chop,
        calculate_confidence_score, predict_direction, get_btc_regime,
        fetch_fear_greed_index, detect_whale_volume, detect_smart_money_accumulation,
        detect_rsi_divergence, get_exchange_symbol,
        # Short ML model — dedicated model trained on short outcomes
        train_short_ml_model, predict_short_ml_outcome,
        save_short_model, load_short_model, should_retrain_short_model,
        # Sell score — real bearish scoring (not buy score inverted)
        calculate_sell_score,
        ACCOUNT_BALANCE, MIN_BUY_SCORE, DONCHIAN_LOOKBACK,
        VOLUME_ANALYSIS_ENABLED, ML_ENABLED, BACKTEST_ENABLED,
        LOG_ENABLED, logger, API_KEY, CG_DAYS
    )
    print(Fore.GREEN + "[SCANNER] ✅ Bot functions imported successfully")
except ImportError as e:
    print(Fore.RED + f"[SCANNER] ❌ Failed to import bot: {e}")
    print(Fore.YELLOW + "Make sure gem_scanner.py is in the same folder as crypto_trading_bot.py")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════
# SCANNER CONFIGURATION
# ══════════════════════════════════════════════════════════════════

SCANNER_CONFIG = {
    # Stage 1: Pre-filter universe
    "universe_size":        250,    # Scan top-N coins by market cap
    "min_market_cap_usd":   50e6,   # Minimum $50M market cap
    "min_price_change_24h": -3.0,   # Not crashing (> -3%)
    "max_price_change_24h": 20.0,   # Not already pumped (< +20%)
    "min_vol_ratio":        0.40,   # Reject only genuinely illiquid (< 40% of avg)
                                    # 0.52x like HYPE is thin but tradeable — scoring handles it via -1 penalty
                                    # 0.45x like MORPHO was too thin AND had bad backtest — both together = skip
                                    # Hard gate kept for trading practicality (slippage, manipulation risk)
    "max_rsi_prefilter":    78,     # Not overbought
    "min_rsi_prefilter":    35,     # Not oversold/crashing

    # Stage 2: Deep scan criteria (strict gem requirements)
    "min_tech_score":       5,      # Matches MIN_BUY_SCORE — further quality gates (ML, conf, OBV) do the filtering
    "min_ml_accuracy":      0.60,   # Reliable ML model
    "min_ml_probability":   0.55,   # ≥55% = clearly bullish (aligns with dampen tier boundary)
    "min_backtest_winrate": 40.0,   # Historically profitable
    "max_zscore":           1.8,    # Not overextended
    "max_ema26_deviation":  14.0,   # bot gives 0 penalty until 15% — scanner rejects at 14% (buffer)
    "cloud_breakout_gap":   3.0,    # allow coins within 3% of cloud top (near breakout)
    "min_confidence":       68,     # High conviction
    "max_don_overshoot":    1.0,    # Not chasing (max 1% above breakout)
    "require_obv_rising":   True,   # Real capital inflow
    "require_not_blocked":  True,   # No blocking conditions
    "max_wash_ratio":       50,     # Wash trade filter
}

# ══════════════════════════════════════════════════════════════════
# WATCHLIST MODE — relaxed criteria for "monitor" candidates
# Run when 0 gems found — finds coins approaching gem status
# ══════════════════════════════════════════════════════════════════

WATCHLIST_CONFIG = {
    "universe_size":        250,
    "min_market_cap_usd":   30e6,   # wider: $30M+
    "min_price_change_24h": -5.0,
    "max_price_change_24h": 25.0,
    "min_vol_ratio":        0.3,
    "max_rsi_prefilter":    80,
    "min_rsi_prefilter":    30,

    # Relaxed deep scan criteria
    "min_tech_score":       5,      # relaxed from 6
    "min_ml_accuracy":      0.55,   # relaxed from 0.60
    "min_ml_probability":   0.505,  # relaxed: dampening already handles accuracy penalty
    "min_backtest_winrate": 35.0,   # relaxed from 40
    "max_zscore":           2.2,    # relaxed from 1.8
    "max_ema26_deviation":  15.0,   # relaxed from 12
    "cloud_breakout_gap":   5.0,    # wider: allow coins within 5% of cloud top
    "min_confidence":       58,     # relaxed from 68
    "max_don_overshoot":    2.0,    # wider: 2% above
    "require_obv_rising":   False,  # OBV flat ok
    "require_not_blocked":  True,
    "max_wash_ratio":       80,     # wider wash filter
}

# ══════════════════════════════════════════════════════════════════
# SHORT SCANNER CONFIG — separate pre-filter for bearish setups
# Inverted from gem config: wants falling/flat coins, not pumping ones
# Gates (in order): ADX≥15 | Sell≤-5 | Long ML<40% | Short ML≥40% | OBV not rising
# ══════════════════════════════════════════════════════════════════

SHORT_SCANNER_CONFIG = {
    "universe_size":          250,
    "min_market_cap_usd":     50e6,    # $50M+ — need real liquidity to short
    "min_price_change_24h":   -25.0,   # allow crashes (those are short opportunities)
    "max_price_change_24h":   5.0,     # exclude pumping coins (bad short entries)
    "min_vol_ratio":          0.4,     # enough volume to short into

    # Short-specific deep scan thresholds (mirrors bot gates)
    "min_sell_score":         -5,      # must be at or below -5
    "min_short_ml_accuracy":  0.55,    # short model must be reliable
    "min_short_ml_proba":     0.40,    # ≥40% of similar setups, shorts won historically
    "min_long_ml_accuracy":   0.55,    # long model must be reliable
    "max_long_ml_proba":      0.40,    # long model also agrees bearish
    "min_adx":                15,      # real trend required — no shorting chop
    "prefer_below_cloud":     True,    # below cloud = downtrend confirmed (preferred, not required)
}

# ── FUTURES-ELIGIBLE COINS (module-level — used by deep_scan_coin and run_watchlist_scanner) ──
FUTURES_ELIGIBLE = {
    # Majors
    "bitcoin","ethereum","solana","binancecoin","ripple","cardano",
    "avalanche-2","polkadot","chainlink","litecoin","dogecoin",
    # Large alts
    "arbitrum","optimism","polygon","cosmos","near","injective-protocol",
    "sui","aptos","sei-network","atom","the-graph",
    # DeFi
    "aave","uniswap","curve-dao-token","maker","synthetix-network-token",
    "compound-governance-token","pendle","morpho","dydx-chain","gmx",
    # AI
    "fetch-ai","render-token","singularitynet","worldcoin-wld","bittensor",
    "ocean-protocol",
    # Meme
    "shiba-inu","pepe","dogwifhat","bonk","floki-inu",
    # Gaming/NFT
    "immutable-x","the-sandbox","decentraland","axie-infinity","gala",
    # L2/Infra
    "starknet","ondo-finance","pyth-network","wormhole",
    "jito-governance-token","layerzero","zcash","filecoin",
    "ether-fi","renzo-protocol","eigenlayer","arbitrum",
    # Already in bot
    "ethena","ordinals","aster-2","fartcoin",
}

# Distance-to-gem scoring — how close is this coin to qualifying?
def score_watchlist_proximity(coin_data, gem_config):
    """
    Score 0-100 how close a watchlist coin is to gem criteria.
    Higher = closer to becoming a gem on next scan.
    """
    score = 0
    notes = []

    # Tech score proximity (max 30 pts)
    ts = coin_data.get("score", 0)
    ts_gap = ts - gem_config["min_tech_score"]
    if ts_gap >= 1:
        score += 30
    elif ts_gap == 0:
        score += 20; notes.append("tech: 1pt below gem")
    elif ts_gap == -1:
        score += 10; notes.append("tech: 2pts below gem")

    # ML proximity (max 25 pts)
    ml = coin_data.get("ml_proba", 0)
    if ml >= 0.60:
        score += 25
    elif ml >= 0.55:
        score += 18; notes.append(f"ML: {ml:.0%} (gem needs 58%)")
    elif ml >= 0.50:
        score += 10; notes.append(f"ML: {ml:.0%} (needs work)")

    # Backtest proximity (max 20 pts)
    bwr = coin_data.get("backtest_wr", 0)
    if bwr >= 45:
        score += 20
    elif bwr >= 40:
        score += 15; notes.append(f"BT: {bwr:.0f}% (gem needs 40%)")
    elif bwr >= 35:
        score += 8; notes.append(f"BT: {bwr:.0f}%")

    # Entry quality (max 15 pts)
    z = coin_data.get("z_score", 0)
    dev = coin_data.get("ema26_dev", 0)
    if z < 1.5 and dev < 10:
        score += 15
    elif z < 1.8 and dev < 12:
        score += 10; notes.append("healthy entry zone")
    else:
        score += 5; notes.append(f"Z={z:.1f}, dev={dev:.0f}%")

    # OBV (max 10 pts)
    obv = coin_data.get("obv_trend", "")
    if obv == "rising":
        score += 10
    elif obv == "flat":
        score += 5; notes.append("OBV flat (needs rising)")

    coin_data["proximity"] = score
    coin_data["proximity_notes"] = notes
    return score


# ══════════════════════════════════════════════════════════════════
# STAGE 1: FAST PRE-FILTER
# ══════════════════════════════════════════════════════════════════

def fetch_market_overview(page=1, per_page=250):
    """Fetch batch market data from CoinGecko — no 365d history needed"""
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": False,
            "price_change_percentage": "24h",
        }
        headers = {"x-cg-demo-api-key": API_KEY}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(Fore.RED + f"[SCANNER] Market overview error: {e}")
        return []

def prefilter_coins(coins_data, config):
    """Stage 1: Apply fast filters to market data"""
    survivors = []
    rejected  = {"market_cap": 0, "price_change": 0, "volume": 0, "rsi": 0, "stablecoin": 0}

    stablecoins = {
        # USD stablecoins
        "usdt","usdc","dai","busd","tusd","frax","usdp","gusd","usde",
        "lusd","pyusd","crvusd","susd","musd","cusd","xusd","ausd","rusd",
        "gho","usdx","fdusd","usdd","eurs","ageur","eurt",
        # Gold / commodity backed
        "xaut","paxg",
        # Other pegged
        "eur","gbp","jpy","cad","aud","chf","steur"
    }
    # CoinGecko IDs of known stables/commodities to skip by ID
    stable_ids = {
        "tether","usd-coin","dai","binance-usd","true-usd","frax","paxos-standard",
        "tether-gold","pax-gold","gho","liquity-usd","paypal-usd","crvusd",
        "ethena-usde","first-digital-usd","usdd","stasis-eurs","ageur",
    }

    for coin in coins_data:
        coin_id  = coin.get("id", "")
        symbol   = coin.get("symbol", "").lower()
        mktcap   = coin.get("market_cap", 0) or 0
        chg_24h  = coin.get("price_change_percentage_24h", 0) or 0
        vol_24h  = coin.get("total_volume", 0) or 0
        price    = coin.get("current_price", 0) or 0

        # Skip stablecoins — by symbol, CoinGecko ID, or price behavior
        _ath  = coin.get("ath", price) or price
        _atl  = coin.get("atl", price) or price
        _atl  = max(_atl, 1e-9)
        _range_pct = (_ath - _atl) / _atl * 100 if _atl > 0 else 100
        _is_stable_by_price = (price > 0.80 and price < 1.20 and _range_pct < 15)
        _is_stable = (
            any(s in symbol for s in stablecoins) or
            "usd" in symbol or
            coin_id in stable_ids or
            _is_stable_by_price
        )
        if _is_stable:
            rejected["stablecoin"] += 1
            continue

        # Market cap filter
        if mktcap < config["min_market_cap_usd"]:
            rejected["market_cap"] += 1
            continue

        # Price change filters
        if chg_24h < config["min_price_change_24h"]:
            rejected["price_change"] += 1
            continue
        if chg_24h > config["max_price_change_24h"]:
            rejected["price_change"] += 1
            continue

        # Volume filter (rough: vol > 0.5% of mktcap = reasonable)
        if mktcap > 0 and vol_24h / mktcap < 0.003:
            rejected["volume"] += 1
            continue

        # Volume ratio filter — reject coins trading far below their average
        # Uses vol/mktcap as proxy since we don't have 90d avg here
        # Deep scan will apply the stricter vol_avg90 check
        _vol_ratio_proxy = vol_24h / mktcap if mktcap > 0 else 0
        if _vol_ratio_proxy < (config.get("min_vol_ratio", 0.70) * 0.003):
            rejected["volume"] += 1
            continue

        survivors.append({
            "id":       coin_id,
            "symbol":   symbol.upper(),
            "name":     coin.get("name", ""),
            "price":    price,
            "chg_24h":  chg_24h,
            "mktcap":   mktcap,
            "vol_24h":  vol_24h,
        })

    return survivors, rejected



def prefilter_coins_short(coins_data, config):
    """
    Stage 1 pre-filter tuned for SHORT candidates.
    Inverted from gem pre-filter: we want falling/flat coins, not pumping ones.
    """
    survivors = []
    stablecoins = {
        "usdt","usdc","dai","busd","tusd","frax","usdp","gusd","usde",
        "lusd","pyusd","crvusd","susd","musd","cusd","xusd","ausd","rusd",
        "gho","usdx","fdusd","usdd","eurs","ageur","eurt","xaut","paxg",
        "eur","gbp","jpy","cad","aud","chf","steur"
    }
    stable_ids = {
        "tether","usd-coin","dai","binance-usd","true-usd","frax","paxos-standard",
        "tether-gold","pax-gold","gho","liquity-usd","paypal-usd","crvusd",
        "ethena-usde","first-digital-usd","usdd","stasis-eurs","ageur",
    }

    for coin in coins_data:
        coin_id = coin.get("id", "")
        symbol  = coin.get("symbol", "").lower()
        mktcap  = coin.get("market_cap", 0) or 0
        chg_24h = coin.get("price_change_percentage_24h", 0) or 0
        vol_24h = coin.get("total_volume", 0) or 0
        price   = coin.get("current_price", 0) or 0

        # Skip stablecoins
        _ath  = coin.get("ath", price) or price
        _atl  = coin.get("atl", price) or price
        _atl  = max(_atl, 1e-9)
        _range_pct = (_ath - _atl) / _atl * 100 if _atl > 0 else 100
        _is_stable = (
            any(s in symbol for s in stablecoins) or
            "usd" in symbol or coin_id in stable_ids or
            (price > 0.80 and price < 1.20 and _range_pct < 15)
        )
        if _is_stable:
            continue

        if mktcap < config["min_market_cap_usd"]:
            continue

        # For shorts: exclude anything pumping hard (bad entry)
        if chg_24h > config["max_price_change_24h"]:
            continue
        # Allow crashes — those are short opportunities
        if chg_24h < config["min_price_change_24h"]:
            continue

        # Volume check
        if mktcap > 0 and vol_24h / mktcap < 0.003:
            continue

        survivors.append({
            "id":      coin_id,
            "symbol":  symbol.upper(),
            "name":    coin.get("name", ""),
            "price":   price,
            "chg_24h": chg_24h,
            "mktcap":  mktcap,
            "vol_24h": vol_24h,
        })

    return survivors


# ══════════════════════════════════════════════════════════════════
# STAGE 2: DEEP SCAN
# ══════════════════════════════════════════════════════════════════

def deep_scan_coin(coin_id, coin_info, btc_regime, btc_df, fng_value, config):
    """
    Full pipeline scan for a single coin.
    Returns gem dict if passes all criteria, else None with fail reason.
    """
    try:
        # Fetch full historical data
        df = fetch_coin_data(coin_id, days=365)
        if df.empty or len(df) < 80:
            return None, "insufficient_data"

        last = df.iloc[-1]
        price = last["price"]
        ema26 = last["EMA26"]
        ema12 = last["EMA12"]

        # ── FAST REJECTS (no API calls) ──────────────────────────
        # Pegged/stable asset: 365d price range < 10% = skip
        _p_range = (df["price"].max() - df["price"].min()) / df["price"].mean() * 100
        if _p_range < 10:
            return None, f"pegged_asset_{_p_range:.1f}pct_range"

        # ── Volume ratio vs 90-day average ────────────────────────
        # Reject coins trading far below their historical average volume.
        # This catches MORPHO-style situations (0.45x avg) that pass the
        # crude mktcap-based pre-filter but would be blocked by main bot.
        # Uses vol_avg90 computed by compute_indicators().
        _vol_now  = last.get("volume", 0) or 0
        _vol_avg  = df["vol_avg90"].iloc[-1] if "vol_avg90" in df.columns else 0
        if _vol_avg > 0:
            _vol_ratio = _vol_now / _vol_avg
            _min_ratio = config.get("min_vol_ratio", 0.70)
            if _vol_ratio < _min_ratio:
                return None, f"volume_low_{_vol_ratio:.2f}x_avg (needs {_min_ratio:.2f}x)"

        # Cloud position — only hard reject if BELOW cloud
        # Inside cloud: transitional zone — scoring already penalises -1
        # If ML and OBV are strong despite being in cloud, let it through
        sa = last.get("senkou_a", np.nan)
        sb = last.get("senkou_b", np.nan)
        if not (np.isnan(sa) or np.isnan(sb)):
            cloud_top = max(sa, sb)
            cloud_bot = min(sa, sb)
            if price < cloud_bot:
                # BELOW cloud — hard reject, trend is bearish
                gap_pct = (cloud_top - price) / price * 100
                return None, f"below_cloud | price=${price:.6f} cloud=${cloud_bot:.6f}→${cloud_top:.6f} | {gap_pct:.1f}% to cloud top"
            elif price < cloud_top:
                # INSIDE cloud — allow through, scoring penalises -1
                gap_pct = (cloud_top - price) / price * 100
                print(Fore.YELLOW + f"  [CLOUD] Inside cloud ({gap_pct:.1f}% to top) — allowing through, scoring applies -1 penalty")

        # Z-score check
        prices_20 = df["price"].iloc[-20:].values
        z = (prices_20[-1] - prices_20.mean()) / (prices_20.std() + 1e-9)
        # Z-score: not checked here as a hard gate — apply_zscore_boost() below
        # already penalises the score for stretched Z-scores, which then fails
        # the score threshold gate. Pre-filter handles extreme cases before fetch.
        prices_20 = df["price"].iloc[-20:].values
        z = (prices_20[-1] - prices_20.mean()) / (prices_20.std() + 1e-9)

        # EMA26 deviation — not a hard gate here since apply_exhaustion_penalty()
        # in scoring already deducts -1 for deviation > 12%.
        # Extreme deviation (>25%) is still flagged as a warning.
        dev_pct = (price - ema26) / ema26 * 100 if ema26 > 0 else 0
        if dev_pct > 25:
            print(Fore.YELLOW + f"  [EXHAUSTION] {coin_id}: extreme deviation {dev_pct:.1f}% above EMA26 — scoring will penalise")

        # Donchian chasing — not a hard gate here since entry_quality="chasing"
        # in the main bot already penalises -8 confidence and sets WAIT timing.
        # Allow through so ML/backtest/OBV can still be evaluated.
        don_upper = df["price"].iloc[-DONCHIAN_LOOKBACK:].max()
        _is_chasing = price > don_upper * (1 + config.get("max_don_overshoot", 1.0) / 100)
        if _is_chasing:
            print(Fore.YELLOW + f"  [CHASING] {coin_id}: price {((price/don_upper)-1)*100:.1f}% above Donchian — scoring will penalise -8 confidence")

        # ── ML TRAINING & GATE ────────────────────────────────────
        ml_model, ml_scaler, ml_accuracy, ml_samples = None, None, 0, 0
        # Try cached model first
        if not should_retrain_model(coin_id):
            ml_model, ml_scaler, metadata = load_model(coin_id)
            if ml_model:
                ml_accuracy = metadata.get("accuracy", 0)
                ml_samples  = metadata.get("samples", 0)

        if ml_model is None:
            ml_model, ml_scaler, ml_accuracy, ml_samples = train_ml_model(
                df, coin_id, btc_df=btc_df, fng_value=fng_value)
            if ml_model:
                save_model(coin_id, ml_model, ml_scaler, ml_accuracy, ml_samples)

        # ML accuracy gate — tiered (mirrors main bot logic)
        # ≤ 52%: at/near random baseline → reject (too unreliable)
        # 52-55%: heavily dampened → allow with heavy penalty
        # 55-60%: lightly dampened → allow with caution
        # 60%+:   trusted → use as-is
        if ml_accuracy <= 0.52:
            return None, f"ml_unreliable_{ml_accuracy:.1%}"

        # ML prediction with accuracy-based dampening
        ml_proba = 0.5
        if ml_model:
            ml_proba, _ = predict_ml_outcome(last, ml_model, ml_scaler,
                                              df=df, btc_df=btc_df, fng_value=fng_value)
            # Mirror main bot tiered dampening
            if ml_accuracy < 0.55:
                ml_proba = ml_proba * 0.30 + 0.5 * 0.70   # heavily dampened
            elif ml_accuracy < 0.60:
                ml_proba = ml_proba * 0.70 + 0.5 * 0.30   # lightly dampened
            # 60%+: use as-is

        # Regime-aware ML threshold:
        # Bull BTC + F&G > 45 = early alt recovery → slightly more lenient
        # Bear/neutral market → strict 55% minimum
        _regime_ml_min = config["min_ml_probability"]
        if btc_regime == "bull" and fng_value > 45:
            _regime_ml_min = max(0.52, config["min_ml_probability"] - 0.03)

        if ml_proba < _regime_ml_min:
            return None, f"ml_bearish_{ml_proba:.1%}"

        # OBV gate
        _, _, obv_trend, _ = check_obv_confirmation(df, coin_id)
        if config["require_obv_rising"] and obv_trend == "falling":
            return None, "obv_distribution"
        # OBV flat: apply confidence penalty later instead of hard reject
        _obv_flat_penalty = (obv_trend == "flat" and config["require_obv_rising"])

        # OBV dampening on ML
        if obv_trend == "rising":
            pass  # confirmed
        elif obv_trend == "flat":
            ml_proba *= 0.75
        else:
            ml_proba *= 0.60

        # ── SCORING PIPELINE ──────────────────────────────────────
        sig = generate_signal_and_reasons(last, df, btc_regime=btc_regime,
                                          is_btc=(coin_id == "bitcoin"),
                                          suppress_debug=True)
        if sig["blocked"] and config["require_not_blocked"]:
            return None, f"blocked_{','.join(sig.get('blockers',['?']))}"

        # CRITICAL: save technical score BEFORE post-tech adjustments
        # Cap is relative to this — without it every score caps to ±2
        sig["technical_score"] = sig["score"]

        # Apply all post-technical adjustments
        # Capture output silently — print only if an exception occurs for debugging
        import io as _sio, sys as _ssys, logging as _slog, traceback as _tb
        _old_stdout = _ssys.stdout
        _captured   = _sio.StringIO()
        _ssys.stdout = _captured
        _scan_logger = _slog.getLogger("TradingBot")
        _old_log_lvl = _scan_logger.level
        _scan_logger.setLevel(_slog.CRITICAL + 1)
        chop_detected = False
        _pipeline_error = None
        try:
            sig["score"] = apply_volume_boost(sig["score"], df, coin_id)
            sig["score"] = apply_price_change_penalty(sig["score"], df, coin_id)
            in_trend = (sig.get("cloud_state") == "above" and ema12 > ema26)
            pre_z = sig["score"]
            sig["score"] = apply_zscore_boost(sig["score"], df, coin_id, in_trend=in_trend)
            zscore_adj = sig["score"] - pre_z
            sig["score"] = apply_exhaustion_penalty(sig["score"], df, coin_id, zscore_adj=zscore_adj)
            sig["score"] = apply_wick_adjustment(sig["score"], df, coin_id)

            # Order book skipped in scanner — no historical data + 4s API call per coin

            pre_chop = sig["score"]
            sig["score"] = apply_chop_penalty(sig["score"], df, coin_id)
            if sig["score"] != pre_chop:
                chop_detected = True

            if not chop_detected:
                sig["score"] = apply_breakout_boost(sig["score"], df, coin_id)

        except Exception as _e:
            _pipeline_error = f"{type(_e).__name__}: {_e}\n{_tb.format_exc()}"
        finally:
            _ssys.stdout = _old_stdout
            _scan_logger.setLevel(_old_log_lvl)

        # Show captured output ONLY if there was an error — full trace for debugging
        if _pipeline_error:
            print(Fore.RED + f"  [PIPELINE ERROR] {coin_id}: {_pipeline_error}")
            print(Fore.YELLOW + "  [DEBUG OUTPUT]:")
            print(_captured.getvalue())

        # Cap adjustments
        tech_score = sig.get("technical_score", 0)
        cap = 1 if chop_detected else 2
        adj = sig["score"] - tech_score
        if adj < -cap:
            sig["score"] = tech_score - cap
        elif adj > cap:
            sig["score"] = tech_score + cap

        # Score threshold
        if sig["score"] < config["min_tech_score"]:
            return None, f"score_low_{sig['score']}"

        # ── BACKTEST ──────────────────────────────────────────────
        backtest_wr = 50.0  # default
        backtest_pf = 1.0   # default
        if BACKTEST_ENABLED:
            bt = backtest_signals(df, coin_id, btc_regime=btc_regime)
            total_trades = bt.winning_trades + bt.losing_trades
            if total_trades > 0:
                backtest_wr = bt.win_rate
                backtest_pf = bt.profit_factor if hasattr(bt, "profit_factor") else 1.0
                sig["backtest_win_rate"] = backtest_wr
                sig["backtest_pf"]       = backtest_pf
                if backtest_wr < config["min_backtest_winrate"]:
                    return None, f"backtest_low_{backtest_wr:.1f}pct"

        # Entry quality
        if price > don_upper * 1.01:
            sig["entry_quality"] = "chasing"
        elif dev_pct > 12:
            sig["entry_quality"] = "near_exhaustion"
        else:
            sig["entry_quality"] = "normal"

        # ── CONFIDENCE SCORE ──────────────────────────────────────
        direction, dir_conf = predict_direction(df)
        is_whale, _, _ = detect_whale_volume(df)
        sm_mode, sm_conf, _ = detect_smart_money_accumulation(df)

        # Ensure sig has all keys that calculate_confidence_score reads from sig
        sig["ml_proba"]   = ml_proba   # prevents default 0.5 override inside function

        conf_result = calculate_confidence_score(
            score=sig["score"], direction=direction, direction_conf=dir_conf,
            ml_proba=ml_proba, blocked=sig["blocked"],
            smart_money_mode=sm_mode, is_whale=is_whale, sig=sig
        )

        conf_score = conf_result["score"]
        # OBV flat penalty: -5 confidence (unconfirmed accumulation)
        if _obv_flat_penalty:
            conf_score -= 5
        if conf_score < config["min_confidence"]:
            return None, f"confidence_low_{conf_score}"

        # ── PASSES ALL CRITERIA — IT'S A GEM! ────────────────────
        is_futures = coin_id in FUTURES_ELIGIBLE
        return {
            "id":           coin_id,
            "name":         coin_info.get("name", coin_id),
            "symbol":       coin_info.get("symbol", "?").upper(),
            "price":        price,
            "is_futures":   is_futures,
            "score":        sig["score"],
            "ml_accuracy":  ml_accuracy,
            "ml_proba":     ml_proba,
            "backtest_wr":  backtest_wr,
            "z_score":      z,
            "ema26_dev":    dev_pct,
            "obv_trend":    obv_trend,
            "direction":    direction,
            "dir_conf":     dir_conf,
            "confidence":   conf_score,
            "obv_flat":     _obv_flat_penalty,
            "conf_rating":  conf_result["rating"],
            "conf_rec":     conf_result["recommendation"],
            "chg_24h":      coin_info.get("chg_24h", 0),
            "mktcap":       coin_info.get("mktcap", 0),
            "smart_money":  sm_mode,
            "is_whale":     is_whale,
            "adx":          last.get("ADX", 0) or 0,
            "scan_hour":    datetime.utcnow().hour,
        }, None

    except Exception as e:
        return None, f"error_{str(e)[:30]}"


# ══════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ══════════════════════════════════════════════════════════════════


def print_gate_summary(fail_reasons, gems_found, total_scanned):
    """Print rejection stats in gate order — shows pipeline funnel clearly."""
    # Gate definitions in check order
    gates = [
        (1,  "Data length",      ["insufficient_data"],              "insufficient_data"),
        (2,  "Pegged asset",     ["pegged_asset"],                   "pegged_asset"),
        (3,  "Volume",           ["volume_low", "volume"],           "volume_low_X"),
        (4,  "Cloud position",   ["below_cloud"],                    "below_cloud"),
        (5,  "ML accuracy",      ["ml_unreliable"],                  "ml_unreliable_X%"),
        (6,  "ML prediction",    ["ml_bearish"],                     "ml_bearish_X%"),
        (7,  "OBV direction",    ["obv_distribution", "obv_flat"],   "obv_dist / obv_flat"),
        (8,  "Blocked signal",   ["blocked_Very", "blocked_"],       "blocked_X"),
        (9,  "Score threshold",  ["score_low"],                      "score_low_X"),
        (10, "Backtest WR",      ["backtest_low"],                   "backtest_low_X%"),
        (11, "Confidence",       ["confidence_low"],                 "confidence_low_X"),
    ]

    # Count per gate using prefix matching
    gate_counts = []
    for gate_num, label, prefixes, reason_fmt in gates:
        count = sum(
            v for k, v in fail_reasons.items()
            if any(k.startswith(p) for p in prefixes)
        )
        gate_counts.append((gate_num, label, reason_fmt, count))

    total_rejected = sum(c for _, _, _, c in gate_counts)

    print(Fore.CYAN + "\n" + "═"*63)
    print(Fore.CYAN + "  FILTER FUNNEL — Gate-by-gate rejection breakdown")
    print(Fore.CYAN + "═"*63)
    print(Fore.WHITE + f"  {'#':<4} {'Gate':<20} {'Reason':<26} {'Rejected':>8}")
    print(Fore.WHITE + "─"*63)

    running = total_scanned
    for gate_num, label, reason_fmt, count in gate_counts:
        if count == 0:
            color = Fore.WHITE
            bar   = ""
        elif count <= 5:
            color = Fore.YELLOW
            bar   = "▌" * count
        elif count <= 20:
            color = Fore.YELLOW
            bar   = "█" * (count // 2) + ("▌" if count % 2 else "")
        else:
            color = Fore.RED
            bar   = "█" * min(count // 3, 15)

        count_str = f"{count:>3}" if count > 0 else "  0"
        print(color + f"  {gate_num:<4} {label:<20} {reason_fmt:<26} {count_str}  {bar}")
        running -= count

    print(Fore.WHITE + "─"*63)
    gem_color = Fore.GREEN if gems_found > 0 else Fore.YELLOW
    print(gem_color + f"  {'':4} {'PASS ALL':<20} {'✅ GEM':<26} {gems_found:>3}")
    print(Fore.CYAN + "═"*63)
    print(Fore.WHITE + f"  Scanned: {total_scanned}  Rejected: {total_rejected}  Gems: {gems_found}")


def deep_scan_short(coin_id, coin_info, btc_regime, btc_df, fng_value, config=None):
    """
    Full pipeline scan for SHORT opportunities.

    Gates (in order):
      1. Data length
      2. ADX ≥ 15          — real trend required, no shorting chop
      3. Sell score ≤ -5   — technically bearish
      4. Long ML accuracy  — must be reliable (≥ 55%)
      5. Long ML proba     — must agree bearish (< 40%)
      6. Short ML accuracy — dedicated short model must be reliable (≥ 55%)
      7. Short ML proba    — shorts historically win here (≥ 40%)
      8. OBV NOT rising    — OBV rising = capital flowing IN = blocks shorts
      9. Cloud position    — below cloud preferred, but not a hard gate

    Cloud position is context, not a gate (below cloud = bonus, above = lower priority).
    """
    if config is None:
        config = SHORT_SCANNER_CONFIG

    try:
        import io as _si, sys as _ss, logging as _sl
        _old_out = _ss.stdout; _ss.stdout = _si.StringIO()
        _log = _sl.getLogger("TradingBot"); _old_lvl = _log.level
        _log.setLevel(_sl.CRITICAL + 1)

        df = fetch_coin_data(coin_id, days=365)
        if df.empty or len(df) < 80:
            _ss.stdout = _old_out; _log.setLevel(_old_lvl)
            return {"rejected": True, "reason": "no_data",
                    "symbol": coin_info.get("symbol", "?").upper(),
                    "price": 0, "chg_24h": coin_info.get("chg_24h", 0)}

        last  = df.iloc[-1]
        price = last["price"]

        # ── Gate 1: ADX — no shorting into chop ─────────────────
        adx = last.get("ADX", 0) or 0
        if adx < config["min_adx"]:
            _ss.stdout = _old_out; _log.setLevel(_old_lvl)
            return {"rejected": True,
                    "reason": f"adx_weak_{adx:.1f}",
                    "symbol": coin_info.get("symbol", "?").upper(),
                    "price": price, "chg_24h": coin_info.get("chg_24h", 0),
                    "adx": adx}

        # ── Gate 2: Sell score ───────────────────────────────────
        # MUST use calculate_sell_score() directly — generate_signal_and_reasons()
        # never populates sell_score in the sig dict (that's done in run_once()).
        # The old fallback of -abs(buy_score) was completely wrong: a coin with
        # buy_score=8 would get sell_score=-8, making every bullish coin look like
        # a great short.
        try:
            sell_score = calculate_sell_score(df, is_btc=(coin_id == "bitcoin"))
        except Exception:
            sell_score = 0

        if sell_score > config["min_sell_score"]:
            _ss.stdout = _old_out; _log.setLevel(_old_lvl)
            return {"rejected": True,
                    "reason": f"sell_score_{sell_score:.1f}",
                    "symbol": coin_info.get("symbol", "?").upper(),
                    "price": price, "chg_24h": coin_info.get("chg_24h", 0),
                    "sell_score": sell_score}

        # ── Gate 3: Long ML — must be reliable + agree bearish ──
        long_model, long_scaler, long_accuracy, long_samples = None, None, 0, 0
        if not should_retrain_model(coin_id):
            long_model, long_scaler, long_meta = load_model(coin_id)
            if long_model:
                long_accuracy = long_meta.get("accuracy", 0) if long_meta else 0

        if long_model is None:
            long_model, long_scaler, long_accuracy, long_samples = train_ml_model(
                df, coin_id, btc_df=btc_df, fng_value=fng_value)
            if long_model:
                save_model(coin_id, long_model, long_scaler, long_accuracy, long_samples)

        long_ml_proba = 0.5
        if long_model and long_accuracy >= config["min_long_ml_accuracy"]:
            long_ml_proba, _ = predict_ml_outcome(
                last, long_model, long_scaler, df=df, btc_df=btc_df, fng_value=fng_value)
            if long_accuracy < 0.60:
                long_ml_proba = long_ml_proba * 0.70 + 0.5 * 0.30

            if long_ml_proba >= config["max_long_ml_proba"]:
                _ss.stdout = _old_out; _log.setLevel(_old_lvl)
                return {"rejected": True,
                        "reason": f"long_ml_bullish_{long_ml_proba:.0%}",
                        "symbol": coin_info.get("symbol", "?").upper(),
                        "price": price, "chg_24h": coin_info.get("chg_24h", 0),
                        "long_ml_accuracy": long_accuracy,
                        "long_ml_proba": long_ml_proba}
        elif long_accuracy < config["min_long_ml_accuracy"] and long_accuracy > 0:
            # Long model unreliable — don't gate on it, just note it
            long_ml_proba = 0.5  # treat as neutral

        # ── Gate 4: Short ML — dedicated model ──────────────────
        short_model, short_scaler, short_accuracy = None, None, 0
        if not should_retrain_short_model(coin_id):
            short_model, short_scaler, short_meta = load_short_model(coin_id)
            if short_model:
                short_accuracy = short_meta.get("accuracy", 0) if short_meta else 0

        if short_model is None:
            short_model, short_scaler, short_accuracy, _ = train_short_ml_model(
                df, coin_id, btc_df=btc_df, fng_value=fng_value)
            if short_model:
                save_short_model(coin_id, short_model, short_scaler, short_accuracy,
                                 len(df))

        if short_accuracy < config["min_short_ml_accuracy"]:
            _ss.stdout = _old_out; _log.setLevel(_old_lvl)
            return {"rejected": True,
                    "reason": f"short_ml_unreliable_{short_accuracy:.0%}",
                    "symbol": coin_info.get("symbol", "?").upper(),
                    "price": price, "chg_24h": coin_info.get("chg_24h", 0),
                    "short_ml_accuracy": short_accuracy}

        short_ml_proba = 0.5
        if short_model:
            short_ml_proba, _ = predict_short_ml_outcome(
                last, short_model, short_scaler, df=df, btc_df=btc_df, fng_value=fng_value)
            if short_accuracy < 0.60:
                short_ml_proba = short_ml_proba * 0.70 + 0.5 * 0.30
            # Accuracy gate — same as bot
            if short_accuracy < 0.50:
                short_ml_proba = 0.5

        if short_ml_proba < config["min_short_ml_proba"]:
            _ss.stdout = _old_out; _log.setLevel(_old_lvl)
            return {"rejected": True,
                    "reason": f"short_ml_low_{short_ml_proba:.0%}",
                    "symbol": coin_info.get("symbol", "?").upper(),
                    "price": price, "chg_24h": coin_info.get("chg_24h", 0),
                    "short_ml_accuracy": short_accuracy,
                    "short_ml_proba": short_ml_proba}

        # ── Gate 5: OBV — HARD BLOCKER if rising ──────────────────
        _, _, obv_trend, _ = check_obv_confirmation(df, coin_id)
        # OBV rising = smart money accumulating = capital flowing IN = bad for short
        # OBV flat   = neutral — allowed but lower priority
        # OBV falling= distribution confirmed — ideal short
        
        if obv_trend == "rising":
            _ss.stdout = _old_out; _log.setLevel(_old_lvl)
            return {"rejected": True,
                    "reason": f"obv_rising",
                    "symbol": coin_info.get("symbol", "?").upper(),
                    "price": price, "chg_24h": coin_info.get("chg_24h", 0),
                    "obv_trend": obv_trend}

        # ── Cloud position (context, not gate) ───────────────────
        sa = last.get("senkou_a", float("nan"))
        sb = last.get("senkou_b", float("nan"))
        cloud_pos = "unknown"
        if not (sa != sa or sb != sb):
            cloud_top = max(sa, sb)
            cloud_bot = min(sa, sb)
            if price < cloud_bot:
                cloud_pos = "below"      # ✅ best — downtrend confirmed
            elif price < cloud_top:
                cloud_pos = "inside"     # ⚠️ transitional
            else:
                cloud_pos = "above"      # ⬇️ counter-trend short

        # Priority: urgent if strong confluence, lower if above cloud
        if sell_score <= -7 or (short_ml_proba >= 0.65 and cloud_pos == "below"):
            priority = "urgent"
        elif cloud_pos == "above":
            priority = "default"   # counter-trend short = lower alert priority
        else:
            priority = "high"

        # ── Stop Loss Calculation ─────────────────────────────────
        # For shorts: place stop above entry where thesis breaks
        # Conservative: 1.5 × ATR above current price
        # This gives shorts room to breathe but exits before major losses
        atr_val = last.get("ATR", 0) or 0
        stop_loss = price + (atr_val * 1.5) if atr_val > 0 else price * 1.05  # fallback: 5% above
        
        # ── Profit Target ──────────────────────────────────────────
        # Aggressive: 2.5 × ATR below current price
        # This is the historical target that shorts aim for
        profit_target = price - (atr_val * 2.5) if atr_val > 0 else price * 0.90  # fallback: 10% below
        
        # Risk/Reward ratio
        risk_amount = stop_loss - price
        reward_amount = price - profit_target
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        # Try Binance order book first; fall back to 24h volume estimate
        book_usd = 0
        liq_source = "order_book"
        try:
            book_usd, _, _ = analyze_order_book_liquidity(coin_id)
        except Exception:
            pass

        if book_usd <= 0:
            # Fallback: estimate from 24h volume (conservative: assume ~0.5% sits in book)
            vol_24h = coin_info.get("vol_24h", 0) or 0
            if vol_24h > 0:
                book_usd = vol_24h * 0.005   # rough: 0.5% of daily vol in active book
                liq_source = f"vol_estimate(${vol_24h/1e6:.1f}M 24h vol)"
            else:
                liq_source = "no_data"

        liq_key, liq_label, max_pos = get_liquidity_tier(book_usd)
        if liq_source != "order_book":
            liq_label = f"{liq_label}~"   # tilde = estimated, not real order book

        _ss.stdout = _old_out; _log.setLevel(_old_lvl)
        return {
            "id":               coin_id,
            "cg_id":            coin_id,                           # CoinGecko ID for main bot
            "binance_id":       get_exchange_symbol(coin_id) or "N/A",  # Binance symbol
            "name":             coin_info.get("name", coin_id),
            "symbol":           coin_info.get("symbol", "?").upper(),
            "price":            price,
            "chg_24h":          coin_info.get("chg_24h", 0),
            "adx":              adx,
            "sell_score":       sell_score,
            "long_ml_accuracy": long_accuracy,
            "long_ml_proba":    long_ml_proba,
            "short_ml_accuracy":short_accuracy,
            "short_ml_proba":   short_ml_proba,
            "obv_trend":        obv_trend,
            "cloud_pos":        cloud_pos,
            "priority":         priority,
            "book_usd":         book_usd,
            "liq_label":        liq_label,
            "liq_source":       liq_source,
            "max_pos":          max_pos,
            "tradeable":        liq_key != "paper_thin",
            "stop_loss":        stop_loss,
            "profit_target":    profit_target,
            "rr_ratio":         rr_ratio,
        }

    except Exception as e:
        try:
            _ss.stdout = _old_out; _log.setLevel(_old_lvl)
        except Exception:
            pass
        return {"rejected": True, "reason": f"error_{str(e)[:30]}",
                "symbol": coin_info.get("symbol", "?").upper(),
                "price": 0, "chg_24h": coin_info.get("chg_24h", 0)}


def deep_scan_both(coin_id, coin_info, btc_regime, btc_df, fng_value,
                   long_config, short_config):
    """
    Single-pass scan: fetch data ONCE, evaluate BOTH long gem and short criteria.

    Returns:
        gem_result   — dict if long gem passes, None otherwise
        gem_reason   — failure reason string for long
        short_result — dict if short passes, None otherwise

    API calls: 1 per coin (down from 2 when running separately).
    The fetch is also session-cached so re-runs within 30min cost 0 calls.
    """
    import io as _si, sys as _ss, logging as _sl
    _old_out = _ss.stdout; _ss.stdout = _si.StringIO()
    _log = _sl.getLogger("TradingBot"); _old_lvl = _log.level
    _log.setLevel(_sl.CRITICAL + 1)

    try:
        # ── ONE FETCH for both scanners ───────────────────────────
        df = fetch_coin_data(coin_id, days=365)
        _ss.stdout = _old_out; _log.setLevel(_old_lvl)

        if df.empty or len(df) < 80:
            return None, "insufficient_data", None, None

        last  = df.iloc[-1]
        price = last["price"]

        # ── SHARED INDICATORS (computed once) ─────────────────────
        adx      = last.get("ADX", 0) or 0
        ema12    = last.get("EMA12", price)
        ema26    = last.get("EMA26", price)
        sa       = last.get("senkou_a", float("nan"))
        sb       = last.get("senkou_b", float("nan"))
        obv_trend_result = check_obv_confirmation(df, coin_id)
        _, _, obv_trend, _ = obv_trend_result

        # Cloud position
        cloud_pos = "unknown"
        if not (np.isnan(sa) or np.isnan(sb)):
            cloud_top = max(sa, sb)
            cloud_bot = min(sa, sb)
            if price > cloud_top:   cloud_pos = "above"
            elif price < cloud_bot: cloud_pos = "below"
            else:                   cloud_pos = "inside"

        # Z-score
        prices_20 = df["price"].iloc[-20:].values
        z = (prices_20[-1] - prices_20.mean()) / (prices_20.std() + 1e-9)

        # Sell score (used by both — confirms long is clean, gates short)
        try:
            sell_score = calculate_sell_score(df, is_btc=(coin_id == "bitcoin"))
        except Exception:
            sell_score = 0

        # ═══════════════════════════════════════════════════════════
        # LONG GEM EVALUATION
        # ═══════════════════════════════════════════════════════════
        gem_result = None
        gem_reason = None

        _suppress = _si.StringIO(); _old2 = _ss.stdout; _ss.stdout = _suppress
        try:
            # Reuse deep_scan_coin but pass pre-fetched df to avoid re-fetch
            # Gate 1: pegged asset
            _p_range = (df["price"].max() - df["price"].min()) / df["price"].mean() * 100
            if _p_range < 10:
                gem_reason = f"pegged_asset_{_p_range:.1f}pct_range"
            # Gate 2: cloud
            elif cloud_pos == "below" or (cloud_pos == "inside" and
                  (max(sa, sb) - price) / price * 100 > long_config.get("cloud_breakout_gap", 3.0)):
                gap = (max(sa, sb) - price) / price * 100 if not np.isnan(sa) else 0
                gem_reason = f"{'below_cloud' if cloud_pos=='below' else 'in_cloud'} | {gap:.1f}% to top"
            else:
                # Run remaining gates via deep_scan_coin
                # EMA deviation not a hard gate here — apply_exhaustion_penalty handles it in scoring
                # Run remaining gates via deep_scan_coin (passes df indirectly via fetch cache)
                gem_result, gem_reason = deep_scan_coin(
                    coin_id, coin_info, btc_regime, btc_df, fng_value, long_config)
        except Exception as e:
            gem_reason = f"long_error_{str(e)[:20]}"
        finally:
            _ss.stdout = _old2

        # ═══════════════════════════════════════════════════════════
        # SHORT EVALUATION (reuses same df via session cache)
        # ═══════════════════════════════════════════════════════════
        short_result = None
        short_reason = None

        _suppress2 = _si.StringIO(); _old3 = _ss.stdout; _ss.stdout = _suppress2
        try:
            _short_ok = True

            if adx < short_config["min_adx"]:
                short_reason = f"adx_weak_{adx:.1f}"
                _short_ok = False
            elif sell_score > short_config["min_sell_score"]:
                short_reason = f"sell_score_{sell_score:.1f}"
                _short_ok = False
            elif obv_trend == "rising":
                short_reason = "obv_rising"
                _short_ok = False

            if _short_ok:
                short_result_raw = deep_scan_short(
                    coin_id, coin_info, btc_regime, btc_df, fng_value, short_config)
                if short_result_raw and not short_result_raw.get("rejected"):
                    short_result = short_result_raw
                elif short_result_raw and short_result_raw.get("rejected"):
                    short_reason = short_result_raw.get("reason", "rejected")
        except Exception as e:
            short_reason = f"error_{str(e)[:20]}"
        finally:
            _ss.stdout = _old3

        return gem_result, gem_reason, short_result, short_reason

    except Exception as e:
        try:
            _ss.stdout = _old_out; _log.setLevel(_old_lvl)
        except Exception:
            pass
        return None, f"error_{str(e)[:30]}", None, None



    """
    Run dedicated short scan with its own pre-filter and dual ML display.
    Uses ADX gate + dedicated short ML model trained on short outcomes.
    """
    print(Fore.RED + "\n" + "─"*70)
    print(Fore.RED + "  🔴 SHORT SCAN — Dedicated bearish setup scanner")
    print(Fore.RED + "  Gates: ADX≥15 | Sell≤-5 | Long ML<40% | Short ML≥40%")
    print(Fore.RED + "─"*70)

    config = SHORT_SCANNER_CONFIG

    # Short-specific pre-filter (different from gem pre-filter)
    print(Fore.YELLOW + "  [SHORT SETUP] Fetching market data with short-friendly filter...")
    all_coins_raw = []
    for page in range(1, 2):
        batch = fetch_market_overview(page=page, per_page=250)
        all_coins_raw.extend(batch)
        time.sleep(1)
    all_coins_raw = all_coins_raw[:config["universe_size"]]
    short_universe = prefilter_coins_short(all_coins_raw, config)
    print(Fore.RED + f"  [SHORT SETUP] {len(short_universe)} candidates "
          f"(chg ≤ {config['max_price_change_24h']:+.0f}%, mktcap ≥ ${config['min_market_cap_usd']/1e6:.0f}M)")

    shorts      = []
    fail_counts = {}

    for i, coin_info in enumerate(short_universe, 1):
        coin_id = coin_info["id"]
        symbol  = coin_info["symbol"].upper()
        print(Fore.WHITE + f"  [{i:3d}/{len(short_universe)}] {symbol:<8} "
              f"${coin_info['price']:.4f} ({coin_info['chg_24h']:+.1f}%)  ... ", end="", flush=True)

        result = deep_scan_short(coin_id, coin_info, btc_regime, btc_df, fng_value, config)

        if result and not result.get("rejected"):
            shorts.append(result)
            cloud_tag = {"below": "☁️↓below", "inside": "⚡in-cloud",
                         "above": "⬆️above"}.get(result["cloud_pos"], "")
            obv_tag   = {"falling": "OBV↓", "flat": "OBV→",
                         "rising": "OBV↑"}.get(result.get("obv_trend", ""), "")
            liq_label = result.get("liq_label", "")
            liq_source = result.get("liq_source", "order_book")
            liq_display = liq_label if liq_source == "order_book" else f"{liq_label} (est)"
            trade_tag = "" if result.get("tradeable", True) else "  ⛔ SKIP"
            print(Fore.RED +
                  f"🔴 SHORT  sell={result['sell_score']:.0f}  ADX={result['adx']:.1f}  "
                  f"L.ML={result['long_ml_proba']:.0%}({result['long_ml_accuracy']:.0%})  "
                  f"S.ML={result['short_ml_proba']:.0%}({result['short_ml_accuracy']:.0%})  "
                  f"{cloud_tag}  {obv_tag}  {liq_display}{trade_tag}")

            _book  = result.get("book_usd", 0)
            _max_p = result.get("max_pos")
            _liq_src = result.get("liq_source", "order_book")
            _liq_note = "" if _liq_src == "order_book" else f" (estimated from {_liq_src})"
            _pos_msg = f"Max position: ${_max_p:,}" if _max_p else "No size limit"
            if not result.get("tradeable", True):
                print(Fore.YELLOW + f"  ↳ {liq_label} (${_book:,.0f}) — UNTRADEABLE")
            else:
                send_ntfy(
                    title=f"SHORT GEM: {symbol} | sell={result['sell_score']:.0f} | S.ML={result['short_ml_proba']:.0%}",
                    message=(
                        f"Sell Score:   {result['sell_score']:.0f} / -5\n"
                        f"Long ML:      {result['long_ml_proba']:.0%} (acc {result['long_ml_accuracy']:.0%})\n"
                        f"Short ML:     {result['short_ml_proba']:.0%} (acc {result['short_ml_accuracy']:.0%})\n"
                        f"ADX:          {result['adx']:.1f} ✅\n"
                        f"Cloud:        {result['cloud_pos']}\n"
                        f"OBV:          {result.get('obv_trend', '?')}\n"
                        f"Price:        ${result['price']:.6f}  ({result['chg_24h']:+.1f}% 24h)\n"
                        f"Stop Loss:    ${result['stop_loss']:.6f}  (+{(result['stop_loss']-result['price'])/result['price']*100:.2f}% above entry)\n"
                        f"Profit Tgt:   ${result['profit_target']:.6f}  (-{(result['price']-result['profit_target'])/result['price']*100:.2f}% below entry, R/R {result['rr_ratio']:.2f}:1)\n"
                        f"Liquidity:    {liq_label} (${_book:,.0f} book){_liq_note}\n"
                        f"{_pos_msg}\n"
                        f"──────────────────────\n"
                        f"CoinGecko ID: {result.get('cg_id', result.get('id','?'))}\n"
                        f"Binance:      {result.get('binance_id','N/A')}"
                    ),
                    priority=result["priority"],
                    tags="chart_decreasing,warning"
                )

        elif result and result.get("rejected"):
            reason       = result.get("reason", "?")
            _long_acc    = result.get("long_ml_accuracy", 0)
            _long_proba  = result.get("long_ml_proba", 0)
            _short_acc   = result.get("short_ml_accuracy", 0)
            _short_proba = result.get("short_ml_proba", 0)
            _sell_s      = result.get("sell_score", 0)
            _adx         = result.get("adx", 0)
            fail_counts[reason.split("_")[0]] = fail_counts.get(reason.split("_")[0], 0) + 1

            if reason.startswith("sell_score"):
                gap   = abs(-5 - _sell_s)
                color = Fore.YELLOW if gap <= 1 else Fore.WHITE
                print(color + f"✗ sell={_sell_s:.1f}  [{reason}]")
            elif reason.startswith("adx_weak"):
                print(Fore.YELLOW + f"✗ ADX={_adx:.1f}<15  [{reason}]")
            elif reason.startswith("long_ml_bullish"):
                print(Fore.WHITE + f"✗ L.ML={_long_proba:.0%}({_long_acc:.0%}) too bullish  [{reason}]")
            elif reason.startswith("short_ml_low"):
                color = Fore.YELLOW if _short_proba >= 0.30 else Fore.WHITE
                print(color + f"✗ S.ML={_short_proba:.0%}({_short_acc:.0%})<40%  [{reason}]")
            elif reason == "obv_rising":
                obv_trend = result.get("obv_trend", "?")
                print(Fore.YELLOW + f"✗ OBV {obv_trend.upper()} — smart money accumulating, shorts blocked")
            elif "unreliable" in reason:
                print(Fore.WHITE + f"✗ model unreliable  [{reason}]")
            else:
                print(Fore.WHITE + f"✗ {reason}")
        else:
            print(Fore.WHITE + "✗ no data")

        time.sleep(1.5)

    if not shorts:
        print(Fore.WHITE + "  No short gems found. " + str(fail_counts))
        return

    shorts_sorted = sorted(shorts, key=lambda x: (x["sell_score"], -x["short_ml_proba"]))

    print(Fore.RED + "\n" + "═"*90)
    print(Fore.RED + f"  🔴 {len(shorts)} SHORT GEM(S) FOUND")
    print(Fore.RED + "═"*90)
    print(Fore.WHITE + f"  {'Symbol':<10} {'Price':>10} {'24h':>7} {'Sell':>6} "
          f"{'ADX':>5} {'L.ML(acc)':>12} {'S.ML(acc)':>12} {'OBV':>7} {'Cloud':>9} {'Liq'}")
    print("─"*90)
    for s in shorts_sorted:
        cloud_color = (Fore.RED    if s["cloud_pos"] == "below"  else
                       Fore.YELLOW if s["cloud_pos"] == "inside" else Fore.WHITE)
        obv_color   = (Fore.RED    if s.get("obv_trend") == "falling" else
                       Fore.YELLOW if s.get("obv_trend") == "flat"    else Fore.WHITE)
        tradeable_flag = "" if s.get("tradeable", True) else " ⛔"
        liq_src_flag   = "" if s.get("liq_source", "order_book") == "order_book" else "~"
        long_ml_str  = f"{s['long_ml_proba']:.0%}({s['long_ml_accuracy']:.0%})"
        short_ml_str = f"{s['short_ml_proba']:.0%}({s['short_ml_accuracy']:.0%})"
        liq_display  = f"{s.get('liq_label','')}{liq_src_flag}"
        print(Fore.RED +
              f"  {s['symbol']:<10} ${s['price']:>9.4f} {s['chg_24h']:>+6.1f}%  "
              f"{s['sell_score']:>+4.0f}  {s['adx']:>4.1f}  "
              f"{long_ml_str:>11}  {short_ml_str:>11}  " +
              obv_color + f"{s.get('obv_trend','?'):>7}" +
              cloud_color + f"  {s['cloud_pos']:>7}" +
              Fore.WHITE + f"  {liq_display}{tradeable_flag}")
        
        _stop = s.get("stop_loss", 0)
        _pt = s.get("profit_target", 0)
        _rr = s.get("rr_ratio", 0)
        _risk_pct   =  ((_stop - s["price"]) / s["price"] * 100) if s["price"] > 0 else 0   # +% above entry (stop)
        _reward_pct = -((s["price"] - _pt)   / s["price"] * 100) if s["price"] > 0 else 0   # -% below entry (target)
        _cg_id  = s.get("cg_id", s.get("id", "?"))
        _bin_id = s.get("binance_id", "N/A")
        print(Fore.CYAN +
              f"    └─ Stop: ${_stop:.6f} ({_risk_pct:+.2f}%) | "
              f"Target: ${_pt:.6f} ({_reward_pct:+.2f}%) | "
              f"R/R: {_rr:.2f}:1")
        print(Fore.WHITE +
              f"       CoinGecko: {_cg_id}  |  Binance: {_bin_id}")

        # ── Bot vs Scanner conflict warning ────────────────────────
        # If this coin also has a strong BUY signal in the bot (score ≥ 5),
        # that means technicals are giving contradictory signals —
        # warn the trader rather than letting two opposing signals cancel silently.
        _sell_s = s.get("sell_score", 0)
        _lml    = s.get("long_ml_proba", 0.5)
        _lml_acc= s.get("long_ml_accuracy", 0)
        if _lml >= 0.50 and _lml_acc >= 0.55:
            print(Fore.YELLOW +
                  f"    ⚠️  BOT CONFLICT: Short scanner found -8 sell score but "
                  f"Long ML {_lml:.0%} (acc {_lml_acc:.0%}) suggests longs may also win here. "
                  f"Check main bot — if bot shows BUY this is a mixed-signal coin. "
                  f"SKIP unless both models agree.")
    print(Fore.RED + "═"*90)
    print(Fore.WHITE + "  L.ML(acc) = long model: low % = bearish confirmation")
    print(Fore.WHITE + "  S.ML(acc) = short model: high % = shorts historically win here")
    print(Fore.WHITE + "  Liq~      = estimated from 24h volume (Binance order book unavailable)")
    print(Fore.CYAN  + "  Stop/Tgt  = Conservative stop (1.5×ATR above) | Profit target (2.5×ATR below) | R/R = risk/reward ratio")


def run_gem_scanner():
    start_time = time.time()
    config = SCANNER_CONFIG

    print(Fore.CYAN + "\n" + "═"*70)
    print(Fore.CYAN + "  💎 GEM SCANNER — Strict Criteria Crypto Hunter")
    print(Fore.CYAN + f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(Fore.CYAN + "═"*70)

    # ── NTFY check at startup ─────────────────────────────────────
    _ntfy_topic = _ntfy_os.environ.get("NTFY_TOPIC", "")
    if _ntfy_topic:
        print(Fore.GREEN + f"[NTFY] ✅ Notifications enabled → ntfy.sh/{_ntfy_topic}")
    else:
        print(Fore.YELLOW + "[NTFY] ⚠️  NTFY_TOPIC not set — push notifications disabled")
        print(Fore.YELLOW + "[NTFY]    To enable: export NTFY_TOPIC=your-topic-name")

    # Fetch BTC regime
    print(Fore.YELLOW + "\n[SETUP] Fetching BTC regime...")
    btc_regime = get_btc_regime()
    print(Fore.GREEN + f"[SETUP] BTC regime: {btc_regime.upper()}")

    if btc_regime == "bear":
        print(Fore.RED + "[SCANNER] ⚠️  BTC in BEAR regime — gem count will be low")

    # Fetch BTC data for ML features
    print(Fore.YELLOW + "[SETUP] Fetching BTC historical data...")
    try:
        from crypto_trading_bot import fetch_coin_data as _fcd
        btc_df = _fcd("bitcoin", days=365)
    except:
        btc_df = None

    # Fetch Fear & Greed
    fng_value = 50
    try:
        fng_value, _ = fetch_fear_greed_index()
        print(Fore.GREEN + f"[SETUP] Fear & Greed: {fng_value}")
    except:
        pass

    # ── STAGE 1: PRE-FILTER ───────────────────────────────────────
    print(Fore.CYAN + f"\n[STAGE 1] Fetching top-{config['universe_size']} coins by market cap...")
    all_coins = []
    pages = config["universe_size"] // 250 + 1
    for page in range(1, pages + 1):
        batch = fetch_market_overview(page=page, per_page=min(250, config["universe_size"]))
        all_coins.extend(batch)
        if len(all_coins) >= config["universe_size"]:
            break
        time.sleep(1)

    all_coins = all_coins[:config["universe_size"]]
    print(Fore.GREEN + f"[STAGE 1] Fetched {len(all_coins)} coins")

    survivors, rejected = prefilter_coins(all_coins, config)
    print(Fore.GREEN + f"[STAGE 1] Pre-filter: {len(survivors)} survivors from {len(all_coins)}")
    print(Fore.YELLOW + f"          Rejected: {rejected}")

    # ── STAGE 2: DEEP SCAN ────────────────────────────────────────
    print(Fore.CYAN + f"\n[STAGE 2] Deep scanning {len(survivors)} coins (est. {len(survivors)*20//60} min)...")
    print(Fore.CYAN + "═"*70)

    # ── STAGE 2: UNIFIED DEEP SCAN (long + short in one pass) ────
    # Merge both universes — union of long pre-filter and short pre-filter
    # Each coin is fetched ONCE and evaluated for both opportunities.
    short_config = SHORT_SCANNER_CONFIG
    short_universe = prefilter_coins_short(all_coins, short_config)
    short_ids = {c["id"] for c in short_universe}

    # Unified universe: all long survivors + any short candidates not already in it
    long_ids = {c["id"] for c in survivors}
    extra_short = [c for c in short_universe if c["id"] not in long_ids]
    unified_universe = survivors + extra_short

    print(Fore.CYAN + f"\n[STAGE 2] Unified scan: {len(survivors)} long candidates + "
          f"{len(extra_short)} short-only candidates = {len(unified_universe)} total")
    print(Fore.CYAN + f"          Each coin fetched ONCE, evaluated for both long and short")
    print(Fore.CYAN + "═"*70)

    gems        = []
    shorts      = []
    fail_reasons = {}
    near_misses  = []
    short_near_misses = []   # shorts that passed all gates except the last one
    short_fail_counts = {}

    _gate_num = {
        "insufficient": 1,
        "pegged":        2,
        "volume":        3,  "volume_low": 3,
        "below":         4,
        "ml":            5,  "ml_unreliable": 5,
        "ml_bearish":    6,
        "obv":           7,  "obv_distribution": 7,
        "blocked":       8,
        "score":         9,  "score_low": 9,
        "backtest":      10, "backtest_low": 10,
        "confidence":    11, "confidence_low": 11,
    }

    for i, coin_info in enumerate(unified_universe, 1):
        coin_id  = coin_info["id"]
        symbol   = coin_info["symbol"].upper()
        is_long  = coin_id in long_ids
        is_short = coin_id in short_ids

        _tags = []
        if is_long:  _tags.append("L")
        if is_short: _tags.append("S")
        _tag_str = "/".join(_tags)

        print(Fore.WHITE + f"  [{i:3d}/{len(unified_universe)}] {symbol:<8} "
              f"${coin_info['price']:.4f} ({coin_info['chg_24h']:+.1f}%) "
              f"[{_tag_str}]  ... ", end="", flush=True)

        gem_result, gem_reason, short_result, short_reason = deep_scan_both(
            coin_id, coin_info, btc_regime, btc_df, fng_value,
            config, short_config)

        # ── LONG result ───────────────────────────────────────────
        if gem_result and is_long:
            gems.append(gem_result)
            # Show ADX context so trader understands trendlessness
            _gem_adx = gem_result.get("adx", 0) or 0
            _adx_note = (f" ⚠️ ADX={_gem_adx:.0f}(no trend)" if _gem_adx < 15
                         else f" ⚠️ ADX={_gem_adx:.0f}(weak)" if _gem_adx < 20 else "")
            print(Fore.GREEN + f"💎 GEM! Score={gem_result['score']} ML={gem_result['ml_proba']:.0%} Conf={gem_result['confidence']}{_adx_note}", end="")
            send_ntfy(
                title=f"💎 GEM FOUND: {gem_result['symbol']}",
                message=(
                    f"Score: {gem_result['score']}/{config['min_tech_score']}\n"
                    f"ML: {gem_result['ml_proba']:.0%} (acc {gem_result.get('ml_accuracy',0):.0%})\n"
                    f"Confidence: {gem_result['confidence']}/100\n"
                    f"Price: ${gem_result['price']:.6f}\n"
                    f"CoinGecko: {coin_id}"
                ),
                priority="high", tags="gem,chart_increasing",
                topic_env="NTFY_TOPIC"
            )
        elif is_long and gem_reason:
            _raw = gem_reason.split(" ")[0]
            _parts = _raw.split("_")
            _key = "_".join(_parts[:2]) if len(_parts) >= 2 else _parts[0]
            fail_reasons[_key] = fail_reasons.get(_key, 0) + 1
            _gate = _gate_num.get(_key, _gate_num.get(_parts[0], 0))
            # Near-miss: gate 9+ (score/backtest/confidence) AND score > 3
            _score_val_nm = 0
            if "score_low" in gem_reason:
                try: _score_val_nm = float(gem_reason.replace("score_low_","").split(" ")[0])
                except: pass

            _is_near_miss = (
                _gate >= 9 and
                (_gate > 9 or _score_val_nm > 3)  # gate 9 only if score > 3
            )

            if _is_near_miss:
                near_misses.append({"symbol": symbol, "reason": gem_reason,
                                    "gate": _gate, "price": coin_info["price"],
                                    "chg_24h": coin_info.get("chg_24h", 0)})
                # Notify for last-gate near-misses (gate 10 or 11)
                if _gate >= 10:
                    _skip_ntfy = False
                    if "score_low" in gem_reason and _score_val_nm < 4:
                        _skip_ntfy = True
                    if not _skip_ntfy:
                        send_ntfy(
                            title=f"⏳ LONG NEAR-MISS: {symbol} — gate {_gate}",
                            message=(
                                f"Passed {_gate-1}/11 gates — stopped at gate {_gate}\n"
                                f"Reason: {gem_reason[:80]}\n"
                                f"Price: ${coin_info['price']:.6f}  ({coin_info.get('chg_24h',0):+.1f}%)\n"
                                f"→ Watch for conditions to improve"
                            ),
                            priority="default", tags="eyes,chart_increasing",
                            topic_env="NTFY_WATCH_TOPIC"
                        )
            # Watch alert: score >= 3 but failed early gate — building up
            # Extract score from reason e.g. "score_low_3" → 3
            _score_val = 0
            if "score_low_" in gem_reason:
                try: _score_val = float(gem_reason.replace("score_low_","").split(" ")[0])
                except: pass
            if _score_val >= 3 and _gate < 10:
                send_ntfy(
                    title=f"👀 EARLY WATCH: {symbol} — score {_score_val:.0f}/5",
                    message=(
                        f"Score {_score_val:.0f}/5 — needs {5-_score_val:.0f} more points\n"
                        f"Reason: {gem_reason[:80]}\n"
                        f"Price: ${coin_info['price']:.6f}  ({coin_info.get('chg_24h',0):+.1f}%)\n"
                        f"→ Monitor for score improvement"
                    ),
                    priority="min", tags="eyes",
                    topic_env="NTFY_WATCH_TOPIC"
                )
            print(Fore.WHITE + f"✗ L:{gem_reason}", end="")
        elif is_long and not gem_result:
            fail_reasons["insufficient_data"] = fail_reasons.get("insufficient_data", 0) + 1
            print(Fore.WHITE + "✗ L:no_data", end="")

        # ── SHORT result ──────────────────────────────────────────
        if short_result and is_short:
            shorts.append(short_result)
            _cloud_tag = {"below": "☁️↓", "inside": "⚡", "above": "⬆️"}.get(
                short_result.get("cloud_pos", ""), "")
            _obv_tag   = {"falling": "OBV↓", "flat": "OBV→"}.get(
                short_result.get("obv_trend", ""), "")
            print(Fore.RED + f"  🔴 SHORT S.ML={short_result['short_ml_proba']:.0%} "
                  f"sell={short_result['sell_score']:.0f} {_cloud_tag}{_obv_tag}", end="")
            send_ntfy(
                title=f"SHORT GEM: {symbol} | S.ML={short_result['short_ml_proba']:.0%}",
                message=(
                    f"Sell Score: {short_result['sell_score']:.0f}/-5\n"
                    f"Short ML:   {short_result['short_ml_proba']:.0%} (acc {short_result['short_ml_accuracy']:.0%})\n"
                    f"ADX:        {short_result['adx']:.1f}\n"
                    f"Cloud:      {short_result['cloud_pos']}\n"
                    f"Stop:       ${short_result.get('stop_loss',0):.6f}\n"
                    f"Target:     ${short_result.get('profit_target',0):.6f}\n"
                    f"CoinGecko:  {coin_id}"
                ),
                priority=short_result.get("priority","high"),
                tags="chart_decreasing,warning",
                topic_env="NTFY_TOPIC"
            )
        elif is_short and not short_result:
            _s_reason = short_reason if short_reason else "rejected"
            _s_raw   = _s_reason.split(" ")[0]
            _s_parts = _s_raw.split("_")
            _s_key   = "_".join(_s_parts[:2]) if len(_s_parts) >= 2 else _s_parts[0]
            short_fail_counts[_s_key] = short_fail_counts.get(_s_key, 0) + 1

            # Short near-miss: passed ADX + sell score + OBV but failed ML gates (last gates)
            _is_short_near_miss = _s_key in ("short_ml", "long_ml")
            if _is_short_near_miss:
                short_near_misses.append({
                    "symbol":  symbol,
                    "reason":  _s_reason,
                    "price":   coin_info["price"],
                    "chg_24h": coin_info.get("chg_24h", 0),
                })
                send_ntfy(
                    title=f"⏳ SHORT NEAR-MISS: {symbol} — ML gate",
                    message=(
                        f"Passed ADX + sell score + OBV gates\n"
                        f"Stopped at: {_s_reason[:80]}\n"
                        f"Price: ${coin_info['price']:.6f}  ({coin_info.get('chg_24h',0):+.1f}%)\n"
                        f"→ Watch for ML to flip bearish"
                    ),
                    priority="default", tags="eyes,chart_decreasing",
                    topic_env="NTFY_WATCH_TOPIC"
                )

            if _s_reason.startswith("adx_weak"):
                _adx_val = _s_reason.replace("adx_weak_", "")
                _s_explain = f"ADX {_adx_val} < 15 (no trend to short into — same ADX as long)"
            elif _s_reason.startswith("sell_score"):
                _score_val = _s_reason.replace("sell_score_", "")
                _s_explain = f"sell={_score_val} (needs ≤ -5)"
            elif _s_reason == "obv_rising":
                _s_explain = "OBV↑ smart money buying"
            elif _s_reason.startswith("short_ml_low"):
                _s_explain = "short ML too low"
            elif _s_reason.startswith("long_ml_bullish"):
                _s_explain = "long ML still bullish"
            elif "unreliable" in _s_reason:
                _s_explain = "short model unreliable"
            else:
                _s_explain = _s_reason
            print(Fore.WHITE + f"  ✗ S:{_s_explain}", end="")

        print()  # newline after each coin's status line
        time.sleep(1.5)

    # ── RESULTS ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(Fore.CYAN + "\n" + "═"*70)
    print(Fore.GREEN + f"  💎 SCAN COMPLETE  |  {elapsed/60:.1f} min  |  {len(gems)} GEMS FOUND")
    print(Fore.CYAN + "═"*70)

    if not gems:
        print(Fore.YELLOW + "\nNo gems found today. Market conditions too tough.")

    # ── DISPLAY GEM RESULTS (if any) ────────────────────────────
    gems_sorted = sorted(gems, key=lambda x: x["confidence"], reverse=True)

    if gems_sorted:
        print(Fore.GREEN + f"\n{'Rank':<5} {'Symbol':<10} {'Price':>10} {'Score':>6} {'ML%':>6} {'MLAcc':>7} {'BT%':>5} {'Z':>5} {'Dev%':>6} {'Conf':>6}  {'Rating'}")
        print("─"*95)

        for rank, gem in enumerate(gems_sorted, 1):
            conf_color = Fore.GREEN if gem["confidence"] >= 75 else Fore.YELLOW
            print(
                conf_color +
                f"{rank:<5} {gem['symbol']:<10} ${gem['price']:>9.4f} "
                f"{gem['score']:>+5.0f}  {gem['ml_proba']:>5.0%}  {gem['ml_accuracy']:>6.0%}  "
                f"{gem['backtest_wr']:>4.0f}%  {gem['z_score']:>4.1f}  "
                f"{gem['ema26_dev']:>5.1f}%  {gem['confidence']:>5.0f}  "
                f"{gem['conf_rating']}"
            )

        print(Fore.CYAN + "\n" + "─"*95)
        print(Fore.CYAN + "\nDETAILED GEM REPORT:")
        print("─"*70)

    for rank, gem in enumerate(gems_sorted, 1):
        # Staleness check — how many hours ago was this gem found
        from datetime import timezone
        _scan_hour = gem.get("scan_hour", datetime.now(timezone.utc).hour)
        _utc_now   = datetime.now(timezone.utc).hour
        _hrs_since = (_utc_now - _scan_hour) % 24
        _candle_close_hrs = (24 - _utc_now) % 24   # hours to next 00:00 UTC

        print(Fore.GREEN + f"\n{'='*60}")
        print(Fore.GREEN + f"  #{rank}  💎 {gem['name']} ({gem['symbol']})")
        print(Fore.GREEN + f"{'='*60}")

        # Staleness warning
        if _hrs_since >= 3:
            print(Fore.YELLOW + f"  ⏰ SIGNAL AGE: Found ~{_hrs_since}h ago — conditions may have changed")
            print(Fore.YELLOW + f"     Re-run main bot on {gem['id']} before entering")
        if _candle_close_hrs > 2 and _utc_now >= 4:
            print(Fore.YELLOW + f"  ⏳ CANDLE CLOSES IN: ~{_candle_close_hrs}h — for futures wait for 00:00 UTC confirmation")

        print(f"  Price:          ${gem['price']:.6f}")
        print(f"  24h Change:     {gem['chg_24h']:+.2f}%")
        print(f"  Market Cap:     ${gem['mktcap']/1e6:.0f}M")
        print(f"  Tech Score:     {gem['score']:+.0f}/5 ✅")
        print(f"  ML Model:       {gem['ml_proba']:.0%} (accuracy: {gem['ml_accuracy']:.0%}) ✅")
        print(f"  Backtest WR:    {gem['backtest_wr']:.1f}% ✅")
        print(f"  Z-Score:        {gem['z_score']:.2f} (< 1.8 = not stretched) ✅")
        print(f"  EMA26 Dev:      {gem['ema26_dev']:.1f}% (< 12% = healthy) ✅")
        print(f"  OBV Trend:      {gem['obv_trend']} ✅")
        print(f"  Direction:      {gem['direction']} ({gem['dir_conf']:.0%} consensus)")
        print(f"  Smart Money:    {gem['smart_money'].upper()}")
        print(f"  Whale:          {'Yes 🐋' if gem['is_whale'] else 'No'}")
        obv_note = " ⚠️ OBV flat (-5 conf)" if gem.get("obv_flat") else ""
        print(Fore.GREEN + f"  CONFIDENCE:     {gem['confidence']}/100 — {gem['conf_rating']}{obv_note}")
        print(Fore.YELLOW + f"  ⚠️  Run full bot analysis on {gem['id']} before trading")

    # Failure summary
    # ── SYMBOL TABLE ──────────────────────────────────────────────
    if gems_sorted:
        print(Fore.CYAN + "\n" + "─"*70)
        print(Fore.CYAN + "  📋 GEM SYMBOL REFERENCE TABLE")
        print(Fore.CYAN + "─"*70)
        print(Fore.CYAN + f"  {'Coin':<20} {'CoinGecko ID':<25} {'Spot':<16} {'Futures':<18} {'Add to COINS'}")
        print("─"*95)
        for gem in gems_sorted:
            cg_id    = gem["id"]
            sym      = gem["symbol"].upper()
            _bsym    = get_exchange_symbol(cg_id)
            spot     = _bsym if _bsym else "Not on Binance"
            futures  = f"{sym}USDT (PERP)" if gem.get("is_futures") else "Spot only"
            fut_color = Fore.CYAN if gem.get("is_futures") else Fore.WHITE
            print(Fore.GREEN + f"  {gem['name']:<20} {cg_id:<25} {spot:<16} " + fut_color + futures.ljust(18) + Fore.GREEN + '  "' + cg_id + '"')
        print()
        print(Fore.YELLOW + "  ➕ To add to your bot, paste the CoinGecko IDs into COINS list:")
        print(Fore.WHITE  + "  COINS = [")
        for gem in gems_sorted:
            print(Fore.WHITE + '      "' + gem["id"] + '",')
        print(Fore.WHITE  + "      ... # your existing coins")
        print(Fore.WHITE  + "  ]")

    print_gate_summary(fail_reasons, len(gems_sorted), len(survivors))

    # ── SHORT REJECTION FUNNEL ────────────────────────────────────
    if short_fail_counts or shorts:
        short_gates = [
            ("ADX weak",       ["adx_weak"],        "ADX < 15 (no trend)"),
            ("Sell score",     ["sell_score"],       "sell > -5 (not bearish enough)"),
            ("OBV rising",     ["obv_rising"],       "OBV↑ capital flowing in"),
            ("Long ML bullish",["long_ml_bullish"],  "long model disagrees"),
            ("Short ML low",   ["short_ml_low"],     "short ML < 40%"),
            ("ML unreliable",  ["unreliable", "short_ml_unreliable"], "model accuracy < 55%"),
            ("Other",          ["error", "rejected"],"other rejection"),
        ]
        total_short_candidates = sum(short_fail_counts.values()) + len(shorts)
        print(Fore.RED + "\n" + "═"*63)
        print(Fore.RED + "  SHORT FILTER FUNNEL")
        print(Fore.RED + "═"*63)
        print(Fore.WHITE + f"  {'Gate':<25} {'Reason':<28} {'Rejected':>6}")
        print(Fore.WHITE + "─"*63)
        for label, prefixes, reason_fmt in short_gates:
            count = sum(v for k, v in short_fail_counts.items()
                        if any(k.startswith(p) for p in prefixes))
            if count == 0:
                continue
            color = Fore.YELLOW if count <= 10 else Fore.RED
            bar   = "█" * min(count // 2, 20)
            print(color + f"  {label:<25} {reason_fmt:<28} {count:>6}  {bar}")
        print(Fore.WHITE + "─"*63)
        print(Fore.GREEN + f"  {'PASS ALL':<25} {'✅ SHORT GEM':<28} {len(shorts):>6}")
        print(Fore.WHITE + f"  Total short candidates scanned: {total_short_candidates}")
        print(Fore.RED + "═"*63)

    # ── NEAR-MISSES (coins that passed gate 10+) ──────────────────────
    if near_misses:
        print(Fore.CYAN + "\n" + "─"*70)
        print(Fore.CYAN + "  🔍 NEAR-MISSES — Passed 10+ gates (closest to qualifying)")
        print(Fore.CYAN + "─"*70)
        print(Fore.WHITE + f"  {'Symbol':<10} {'Price':>10} {'24h':>7}  {'Died at gate':>14}  {'Reason'}")
        print("─"*70)
        for nm in sorted(near_misses, key=lambda x: x["gate"], reverse=True):
            gate_label = {8:"blocked", 9:"score", 10:"backtest", 11:"confidence"}.get(nm["gate"],"?")
            chg_color  = Fore.GREEN if nm["chg_24h"] >= 0 else Fore.RED
            print(Fore.YELLOW + f"  {nm['symbol']:<10} ${nm['price']:>9.4f} " +
                  chg_color + f"{nm['chg_24h']:>+6.1f}%" +
                  Fore.YELLOW + f"  Gate {nm['gate']:>2} ({gate_label:<10})  {nm['reason'][:45]}")
        print(Fore.CYAN + f"\n  {len(near_misses)} coin(s) made it past gate 10 — closest to being gems")
        print(Fore.WHITE + "  These are your best watchlist candidates → run watchlist mode for details")

    # ── SHORT NEAR-MISSES ─────────────────────────────────────────
    if short_near_misses:
        print(Fore.RED + "\n" + "─"*70)
        print(Fore.RED + "  🔍 SHORT NEAR-MISSES — Passed ADX+Sell+OBV gates, failed ML")
        print(Fore.RED + "─"*70)
        print(Fore.WHITE + f"  {'Symbol':<10} {'Price':>10} {'24h':>7}  {'Reason'}")
        print("─"*70)
        for nm in short_near_misses:
            chg_color = Fore.GREEN if nm["chg_24h"] >= 0 else Fore.RED
            print(Fore.YELLOW + f"  {nm['symbol']:<10} ${nm['price']:>9.4f} " +
                  chg_color + f"{nm['chg_24h']:>+6.1f}%" +
                  Fore.YELLOW + f"  {nm['reason'][:50]}")
        print(Fore.RED + f"\n  {len(short_near_misses)} coin(s) near short qualification — watch for ML to flip")

    # ── SHORT RESULTS TABLE ───────────────────────────────────────
    if shorts:
        shorts_sorted = sorted(shorts, key=lambda x: (x["sell_score"], -x["short_ml_proba"]))
        print(Fore.RED + "\n" + "═"*90)
        print(Fore.RED + f"  🔴 {len(shorts)} SHORT GEM(S) FOUND")
        print(Fore.RED + "═"*90)
        print(Fore.WHITE + f"  {'Symbol':<10} {'Price':>10} {'24h':>7} {'Sell':>6} "
              f"{'ADX':>5} {'L.ML(acc)':>12} {'S.ML(acc)':>12} {'OBV':>7} {'Cloud':>9} {'Liq'}")
        print("─"*90)
        for s in shorts_sorted:
            cloud_color = (Fore.RED    if s["cloud_pos"] == "below"  else
                           Fore.YELLOW if s["cloud_pos"] == "inside" else Fore.WHITE)
            obv_color   = (Fore.RED    if s.get("obv_trend") == "falling" else
                           Fore.YELLOW if s.get("obv_trend") == "flat"    else Fore.WHITE)
            liq_flag     = "" if s.get("tradeable", True) else " ⛔"
            liq_src_flag = "" if s.get("liq_source", "order_book") == "order_book" else "~"
            lml = f"{s['long_ml_proba']:.0%}({s['long_ml_accuracy']:.0%})"
            sml = f"{s['short_ml_proba']:.0%}({s['short_ml_accuracy']:.0%})"
            print(Fore.RED +
                  f"  {s['symbol']:<10} ${s['price']:>9.4f} {s['chg_24h']:>+6.1f}%  "
                  f"{s['sell_score']:>+4.0f}  {s['adx']:>4.1f}  "
                  f"{lml:>11}  {sml:>11}  " +
                  obv_color + f"{s.get('obv_trend','?'):>7}" +
                  cloud_color + f"  {s['cloud_pos']:>7}" +
                  Fore.WHITE + f"  {s.get('liq_label','')}{liq_src_flag}{liq_flag}")
            _stop = s.get("stop_loss", 0)
            _pt   = s.get("profit_target", 0)
            _rr   = s.get("rr_ratio", 0)
            _rsk  =  ((_stop - s["price"]) / s["price"] * 100) if s["price"] > 0 else 0
            _rwrd = -((s["price"] - _pt)   / s["price"] * 100) if s["price"] > 0 else 0
            print(Fore.CYAN + f"    └─ Stop: ${_stop:.6f} ({_rsk:+.2f}%) | Target: ${_pt:.6f} ({_rwrd:+.2f}%) | R/R: {_rr:.2f}:1")
            print(Fore.WHITE + f"       CoinGecko: {s.get('cg_id', s.get('id','?'))}  |  Binance: {s.get('binance_id','N/A')}")
            if s.get("long_ml_proba", 0.5) >= 0.50 and s.get("long_ml_accuracy", 0) >= 0.55:
                print(Fore.YELLOW + f"    ⚠️  BOT CONFLICT: Long ML {s['long_ml_proba']:.0%} — mixed signal. SKIP unless both models agree.")
        print(Fore.RED + "═"*90)
        print(Fore.WHITE + "  L.ML(acc) = long model win% | S.ML(acc) = short model win%")
        print(Fore.WHITE + "  Liq~      = estimated from 24h volume")
    else:
        print(Fore.WHITE + "\n  No short gems found this scan.")


if __name__ == "__main__":
    print(Fore.CYAN + "\n💎 CRYPTO GEM SCANNER")
    print(Fore.CYAN + "Finds coins passing ALL strict bot criteria")
    print(Fore.CYAN + "─"*50)
    print(f"Universe:   Top-{SCANNER_CONFIG['universe_size']} by market cap")
    print(f"Min score:  {SCANNER_CONFIG['min_tech_score']}+")
    print(f"Min ML acc: {SCANNER_CONFIG['min_ml_accuracy']:.0%}+")
    print(f"Min ML:     {SCANNER_CONFIG['min_ml_probability']:.0%}+ win probability")
    print(f"Min BT WR:  {SCANNER_CONFIG['min_backtest_winrate']}%+")
    print(f"Min conf:   {SCANNER_CONFIG['min_confidence']}/100")
    print()
if __name__ == "__main__":
    print(Fore.CYAN + "\n💎 CRYPTO GEM SCANNER")
    print(Fore.CYAN + "Finds coins passing ALL strict bot criteria")
    print(Fore.CYAN + "─"*50)
    print(f"Universe:   Top-{SCANNER_CONFIG['universe_size']} by market cap")
    print(f"Min score:  {SCANNER_CONFIG['min_tech_score']}+")
    print(f"Min ML acc: {SCANNER_CONFIG['min_ml_accuracy']:.0%}+")
    print(f"Min ML:     {SCANNER_CONFIG['min_ml_probability']:.0%}+ win probability")
    print(f"Min BT WR:  {SCANNER_CONFIG['min_backtest_winrate']}%+")
    print(f"Min conf:   {SCANNER_CONFIG['min_confidence']}/100")
    print()
    run_gem_scanner()
