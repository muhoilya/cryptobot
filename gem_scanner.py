#!/usr/bin/env python3
"""
GEM SCANNER — Probabilistic 2-Stage Crypto Scanner
Finds coins with HIGH Bayesian continuation probability (P ≥ 65%)

Architecture upgrade from old additive scoring:
  OLD: technical score + confidence → score ≥ 5 AND confidence ≥ 68
  NEW: Bayesian probability with latent group compression → P ≥ 0.65

Three-level signal compression (same as main bot):
  Signal → Group (geometric mean) → Latent State (geometric mean) → P

Stage 1: Fast pre-filter on CoinGecko top-500 (batch API)
Stage 2: Deep scan (probabilistic pipeline) on survivors only

Run: python gem_scanner.py
"""

import requests
import pandas as pd
import numpy as np
import time, os, sys, math
from datetime import datetime
from colorama import Fore, Style, init
init(autoreset=True)
import os as _ntfy_os

def send_ntfy(title, message, priority="default", tags="", topic_env="NTFY_TOPIC"):
    topic = _ntfy_os.environ.get(topic_env, "") or _ntfy_os.environ.get("NTFY_TOPIC", "")
    if not topic:
        print(f"[NTFY] ⚠️  NTFY_TOPIC not set — skipping: '{title}'")
        return
    try:
        import requests as _r
        resp = _r.post(f"https://ntfy.sh/{topic}",
                data=message.encode("utf-8"),
                headers={"Title": title.encode("utf-8"), "Priority": priority,
                         "Tags": tags or "", "Content-Type": "text/plain; charset=utf-8"},
                timeout=8)
        if resp.status_code == 200:
            print(f"[NTFY] ✅ Sent: '{title}'")
        else:
            print(f"[NTFY] ❌ HTTP {resp.status_code}: '{title}'")
    except Exception as e:
        print(f"[NTFY] ❌ Error: {e}")


def get_liquidity_tier(book_usd):
    if not book_usd or book_usd <= 0: return "unknown", "❓ Unknown", None
    elif book_usd >= 1_000_000:       return "deep",       "🟢 DEEP",       None
    elif book_usd >= 300_000:         return "moderate",   "🟡 MODERATE",   5000
    elif book_usd >= 50_000:          return "thin",       "🟠 THIN",       500
    else:                             return "paper_thin", "🔴 PAPER-THIN", 0


# ── IMPORT FROM MAIN BOT ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from crypto_trading_bot import (
        # Data + indicators
        fetch_coin_data, compute_indicators,
        check_obv_confirmation, analyze_order_book_liquidity,
        predict_direction, get_btc_regime, fetch_fear_greed_index,
        detect_whale_volume, detect_smart_money_accumulation,
        get_exchange_symbol,

        # ── NEW: Probabilistic scoring pipeline ──────────────────────────
        generate_signal_probabilistic,     # replaces generate_signal_and_reasons
        compute_tradeability_report,       # 5-component quality display
        compute_short_tradeability_report, # 6-component short quality display

        # ── NEW: Short scorer components ─────────────────────────────────
        detect_distribution_divergence,
        detect_break_of_participation,
        detect_failed_breakout,
        detect_bear_acceleration,
        compute_trend_fragility,       # was: detect_trend_fragility
        detect_short_prior,
        ProbabilisticScorer,
        run_short_scorer,          # actual 6-question short scorer (same as main bot)
        _logit, _sigmoid, _lr_safe,

        # ── NEW: Probability thresholds ──────────────────────────────────
        BUY_PROB_THRESHOLD,    # 0.65 — long entry gate
        STRONG_BUY_THRESHOLD,  # 0.72
        SHORT_PROB_THRESHOLD,  # 0.65 — short entry gate
        SHORT_STRONG_THRESHOLD,# 0.75
        PRIOR_CONTINUATION,    # 0.42

        # ML models (unchanged)
        train_ml_model, predict_ml_outcome, save_model, load_model,
        should_retrain_model,
        train_short_ml_model, predict_short_ml_outcome,
        save_short_model, load_short_model, should_retrain_short_model,

        # Misc
        DONCHIAN_LOOKBACK, ML_ENABLED, BACKTEST_ENABLED, LOG_ENABLED,
        ML_FEATURE_VERSION,
        logger, API_KEY, CG_DAYS,
        # Global model
        load_global_model, predict_global_ml, needs_global_retrain,
        _GLOBAL_MODEL, COIN_METADATA,
        compute_relative_strength_full,   # RS tier
        classify_trend_lifecycle,          # lifecycle stage
        # Regime calibration (shared Brier-monitor singleton — same models/brier_predictions.json
        # the main bot writes to, so scanner and bot read identical historical calibration data)
        _brier_monitor,
        # Regime entropy (shared EntropyTracker singleton — adaptive percentile
        # thresholds learned from shared history, same as the Brier monitor above)
        compute_regime_entropy, apply_regime_entropy,
    )
    print(Fore.GREEN + "[SCANNER] ✅ Bot functions imported successfully (probabilistic mode)")
except ImportError as e:
    print(Fore.RED + f"[SCANNER] ❌ Failed to import bot: {e}")
    sys.exit(1)


# ── Cache version constants ─────────────────────────────────────────────────
# Bump these when the model schema changes to force a retrain by the main bot.
# Scanner never retrains — it skips coins with no fresh/valid model.
SCANNER_FEATURE_VERSION    = "v20.0"   # 16 features, leakage-fixed (df_slice)
SCANNER_REGIME_VERSION     = "v3.0"    # latent state compression
SCANNER_MODEL_VERSION      = "v8.2"    # must match save_model "model_version"
SCANNER_MAX_MODEL_AGE_HOURS = 26       # skip coin if model older than this

# ── GLOBAL MODEL SUPPORT ────────────────────────────────────────────────────
# The scanner now uses the global model as a fallback when a per-coin model
# is missing or stale. This means ALL 250 universe coins get ML prediction,
# not just the ~35 that the main bot has trained on.
#
# Load order:
#   1. Per-coin model (most specific — if fresh and v8.2)
#   2. Global model  (56,000-sample cross-coin model — fallback)
#   3. No model      (pure Bayesian — still valid, just no ML signal)
#
# The global model is NEVER retrained by the scanner — only the main bot
# retrains it (weekly via --train-global or auto at scan end).
_SCANNER_GLOBAL_LOADED = False

def _ensure_global_model():
    """Load global model once per scanner session."""
    global _SCANNER_GLOBAL_LOADED
    if not _SCANNER_GLOBAL_LOADED:
        if load_global_model():
            sharpe = _GLOBAL_MODEL.get("sharpe", 0)
            acc    = _GLOBAL_MODEL.get("acc", 0)
            coins  = _GLOBAL_MODEL.get("meta", {}).get("n_coins", 0)
            print(Fore.GREEN + f"[SCANNER] ✅ Global model loaded  "
                  f"coins={coins}  Sharpe={sharpe:+.2f}  acc={acc:.0%}")
        else:
            print(Fore.YELLOW + "[SCANNER] ⚠️  No global model — run: "
                  "python crypto_trading_bot.py --train-global")
        _SCANNER_GLOBAL_LOADED = True


# PREDICT-ONLY MODE: scanner never calls train_ml_model.
# If a coin has no fresh model → skip it (add to "needs_training" list).
# Main bot handles retraining on its own schedule.
SCANNER_PREDICT_ONLY = True


# ══════════════════════════════════════════════════════════════════════════════
# SCANNER CONFIGURATION — probability-based gates
# Old: min_tech_score=5, min_confidence=68
# New: min_prob=0.65 (Bayesian continuation probability)
# ══════════════════════════════════════════════════════════════════════════════

SCANNER_CONFIG = {
    # Stage 1 pre-filter (unchanged — fast market cap batch check)
    "universe_size":        250,
    "min_market_cap_usd":   50e6,
    "min_price_change_24h": -3.0,
    "max_price_change_24h": 20.0,
    "min_vol_ratio":        0.40,
    "max_rsi_prefilter":    78,
    "min_rsi_prefilter":    35,

    # Stage 2 deep scan — NEW PROBABILISTIC GATES
    "min_prob":             BUY_PROB_THRESHOLD,   # 0.65 — Bayesian long P
    "strong_prob":          STRONG_BUY_THRESHOLD, # 0.72 — strong gem
    "min_ml_accuracy":      0.60,
    "min_ml_probability":   0.55,                 # secondary ML gate
    "min_backtest_winrate": 40.0,
    "max_zscore":           1.8,
    "cloud_breakout_gap":   10.0,  # raised from 3.0 — in-cloud up to 10% allowed
    "cloud_watch_gap":       8.0,  # 8-10% = WATCH zone (needs positive chg_24h)
    "max_below_cloud_gap":  15.0,  # below cloud up to 15% allowed with RS
    "require_obv_rising":   True,
    "require_not_blocked":  True,
}

WATCHLIST_CONFIG = {
    "universe_size":        250,
    "min_market_cap_usd":   30e6,
    "min_price_change_24h": -5.0,
    "max_price_change_24h": 25.0,
    "min_vol_ratio":        0.3,
    "max_rsi_prefilter":    80,
    "min_rsi_prefilter":    30,
    # Relaxed gates
    "min_prob":             0.55,
    "strong_prob":          0.65,
    "min_ml_accuracy":      0.55,
    "min_ml_probability":   0.50,
    "min_backtest_winrate": 35.0,
    "max_zscore":           2.2,
    "cloud_breakout_gap":   12.0,  # raised from 5.0
    "cloud_watch_gap":      10.0,
    "max_below_cloud_gap":  20.0,
    "require_obv_rising":   False,
    "require_not_blocked":  True,
}

SHORT_SCANNER_CONFIG = {
    "universe_size":          250,
    "min_market_cap_usd":     50e6,
    "min_price_change_24h":   -25.0,
    "max_price_change_24h":   5.0,
    "min_vol_ratio":          0.4,
    # NEW: probability-based short gates
    "min_short_prob":         max(SHORT_PROB_THRESHOLD, 0.70),  # raised: 65% too easy in bear
    "strong_short_prob":      SHORT_STRONG_THRESHOLD,  # 0.75
    "require_short_signal":   True,   # must have exhaust/distr/accel, not just bear+cloud
    "min_short_ml_accuracy":  0.55,
    "min_short_ml_proba":     0.45,
    "min_adx":                15,
}

FUTURES_ELIGIBLE = {
    "bitcoin","ethereum","solana","binancecoin","ripple","cardano",
    "avalanche-2","polkadot","chainlink","litecoin","dogecoin",
    "arbitrum","optimism","polygon","cosmos","near","injective-protocol",
    "sui","aptos","sei-network","atom","the-graph",
    "aave","uniswap","curve-dao-token","maker","synthetix-network-token",
    "compound-governance-token","pendle","morpho","dydx-chain","gmx",
    "fetch-ai","render-token","singularitynet","worldcoin-wld","bittensor",
    "ocean-protocol","shiba-inu","pepe","dogwifhat","bonk","floki-inu",
    "immutable-x","the-sandbox","decentraland","axie-infinity","gala",
    "starknet","ondo-finance","pyth-network","wormhole",
    "jito-governance-token","layerzero","zcash","filecoin",
    "ether-fi","renzo-protocol","eigenlayer","ethena","fartcoin",
}


# ══════════════════════════════════════════════════════════════════════════════
# PROXIMITY SCORING (watchlist mode)
# Updated to use probability-based distance to gem criteria
# ══════════════════════════════════════════════════════════════════════════════


def load_model_for_scanner(coin_id: str) -> tuple:
    """
    Load ML model for scanner prediction — NEVER retrains.

    Returns: (ml_model, ml_scaler, ml_accuracy, wf_sharpe, skip_reason)
      skip_reason is None on success, or a string explaining why to skip.

    Skips coin (returns None model) when:
      - No model file exists
      - Model version mismatch (feature_version changed → needs main-bot retrain)
      - Model is too old (> SCANNER_MAX_MODEL_AGE_HOURS)
    """
    try:
        ml_model, ml_scaler, meta = load_model(coin_id)
    except Exception as e:
        return None, None, 0.0, 0.0, f"load_error_{str(e)[:20]}"

    if ml_model is None or meta is None:
        # Don't hard-skip — caller will check global model
        return None, None, 0.0, 0.0, "no_per_coin_model"

    # Architecture version check
    saved_ver = meta.get("model_version", "unknown")
    if saved_ver != SCANNER_MODEL_VERSION:
        return None, None, 0.0, 0.0, f"version_mismatch_{saved_ver}_need_{SCANNER_MODEL_VERSION}"

    # Feature version check — when features change (e.g. leakage fix), old models are invalid
    saved_fv = meta.get("feature_version", "unknown")
    if saved_fv != ML_FEATURE_VERSION:
        return None, None, 0.0, 0.0, f"feature_version_{saved_fv}_need_{ML_FEATURE_VERSION}"

    # Age check
    try:
        from datetime import timezone as _tz
        trained_str = meta.get("trained_date", "")
        if trained_str:
            trained_dt  = datetime.fromisoformat(trained_str)
            if trained_dt.tzinfo is None:
                trained_dt = trained_dt.replace(tzinfo=_tz.utc)
            now_utc  = datetime.now(_tz.utc)
            age_hours = (now_utc - trained_dt).total_seconds() / 3600
            if age_hours > SCANNER_MAX_MODEL_AGE_HOURS:
                return None, None, 0.0, 0.0, f"model_stale_{age_hours:.0f}h_run_main_bot"
    except Exception:
        pass   # if we can't parse the date, accept the model

    accuracy  = float(meta.get("accuracy", 0.0))
    wf_sharpe = float(meta.get("wf_sharpe", 0.0))
    return ml_model, ml_scaler, accuracy, wf_sharpe, None


def load_short_model_for_scanner(coin_id: str) -> tuple:
    """Same as load_model_for_scanner but for the short ML model."""
    try:
        short_model, short_scaler, meta = load_short_model(coin_id)
    except Exception as e:
        return None, None, 0.0, f"load_error_{str(e)[:20]}"

    if short_model is None or meta is None:
        return None, None, 0.0, "no_short_model"

    saved_ver = meta.get("model_version", "unknown")
    if saved_ver != "v8.2":   # short model version aligned with long model
        return None, None, 0.0, f"short_version_mismatch_{saved_ver}"

    try:
        from datetime import timezone as _tz
        trained_str = meta.get("trained_date", "")
        if trained_str:
            trained_dt  = datetime.fromisoformat(trained_str)
            if trained_dt.tzinfo is None:
                trained_dt = trained_dt.replace(tzinfo=_tz.utc)
            age_hours = (datetime.now(_tz.utc) - trained_dt).total_seconds() / 3600
            if age_hours > SCANNER_MAX_MODEL_AGE_HOURS:
                return None, None, 0.0, f"short_stale_{age_hours:.0f}h"
    except Exception:
        pass

    acc = float(meta.get("accuracy", 0.0))
    return short_model, short_scaler, acc, None


def score_watchlist_proximity(coin_data, gem_config):
    score = 0
    notes = []

    prob = coin_data.get("prob", 0.5)
    min_prob = gem_config.get("min_prob", BUY_PROB_THRESHOLD)
    gap = min_prob - prob

    # Probability proximity (max 40 pts — now the primary metric)
    if prob >= min_prob:         score += 40
    elif gap <= 0.05:            score += 30; notes.append(f"P={prob:.0%} (needs {min_prob:.0%})")
    elif gap <= 0.10:            score += 20; notes.append(f"P={prob:.0%} (gap {gap:.0%})")
    elif gap <= 0.15:            score += 10; notes.append(f"P={prob:.0%} (wide gap)")

    # ML quality (max 25 pts)
    ml = coin_data.get("ml_proba", 0)
    if ml >= 0.65:   score += 25
    elif ml >= 0.55: score += 18; notes.append(f"ML={ml:.0%}")
    elif ml >= 0.50: score += 10; notes.append(f"ML={ml:.0%} weak")

    # OBV (max 15 pts)
    obv = coin_data.get("obv_trend", "")
    if obv == "rising":  score += 15
    elif obv == "flat":  score += 8;  notes.append("OBV flat")

    # Entry quality (max 10 pts)
    z = coin_data.get("z_score", 0)
    if z < 1.5:   score += 10
    elif z < 1.8: score += 7;  notes.append(f"Z={z:.1f}")

    # Backtest (max 10 pts)
    bwr = coin_data.get("backtest_wr", 0)
    if bwr >= 45:   score += 10
    elif bwr >= 40: score += 7;  notes.append(f"BT={bwr:.0f}%")

    coin_data["proximity"] = score
    coin_data["proximity_notes"] = notes
    return score


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: FAST PRE-FILTER (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_market_overview(page=1, per_page=250):
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "order": "market_cap_desc",
                  "per_page": per_page, "page": page, "sparkline": False,
                  "price_change_percentage": "24h"}
        headers = {"x-cg-demo-api-key": API_KEY}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(Fore.RED + f"[SCANNER] Market overview error: {e}")
        return []


def prefilter_coins(coins_data, config):
    survivors = []
    rejected = {"market_cap": 0, "price_change": 0, "volume": 0, "stablecoin": 0}
    stablecoins = {
        "usdt","usdc","dai","busd","tusd","frax","usdp","gusd","usde","lusd",
        "pyusd","crvusd","susd","gho","usdx","fdusd","usdd","eurs","ageur","xaut","paxg",
    }
    stable_ids = {
        "tether","usd-coin","dai","binance-usd","true-usd","frax","paxos-standard",
        "tether-gold","pax-gold","gho","liquity-usd","paypal-usd","crvusd",
        "ethena-usde","first-digital-usd","usdd","stasis-eurs","ageur",
    }
    for coin in coins_data:
        coin_id = coin.get("id",""); symbol = coin.get("symbol","").lower()
        mktcap  = coin.get("market_cap", 0) or 0
        chg_24h = coin.get("price_change_percentage_24h", 0) or 0
        vol_24h = coin.get("total_volume", 0) or 0
        price   = coin.get("current_price", 0) or 0

        _ath = coin.get("ath", price) or price
        _atl = max(coin.get("atl", price) or price, 1e-9)
        _range_pct = (_ath - _atl) / _atl * 100
        if (any(s in symbol for s in stablecoins) or "usd" in symbol or
                coin_id in stable_ids or (0.80 < price < 1.20 and _range_pct < 15)):
            rejected["stablecoin"] += 1; continue

        if mktcap < config["min_market_cap_usd"]:
            rejected["market_cap"] += 1; continue
        if not (config["min_price_change_24h"] <= chg_24h <= config["max_price_change_24h"]):
            rejected["price_change"] += 1; continue
        if mktcap > 0 and vol_24h / mktcap < 0.003:
            rejected["volume"] += 1; continue

        survivors.append({"id": coin_id, "symbol": symbol.upper(),
                          "name": coin.get("name",""), "price": price,
                          "chg_24h": chg_24h, "mktcap": mktcap, "vol_24h": vol_24h})
    return survivors, rejected


def prefilter_coins_short(coins_data, config):
    survivors = []
    stablecoins = {"usdt","usdc","dai","busd","tusd","frax","usdp","gusd","usde",
                   "lusd","pyusd","crvusd","susd","gho","fdusd","usdd","xaut","paxg",
                   "eur","eurc","eurt","steur","ageur"}  # added EUR stablecoins
    stable_ids  = {"tether","usd-coin","dai","binance-usd","true-usd","frax",
                   "tether-gold","pax-gold","gho","liquity-usd","paypal-usd","crvusd",
                   "ethena-usde","first-digital-usd","usdd","stasis-eurs",
                   "usx","figure-heloc","unibase","falcon-finance-ff"}  # known pegged
    for coin in coins_data:
        coin_id = coin.get("id",""); symbol = coin.get("symbol","").lower()
        mktcap  = coin.get("market_cap", 0) or 0
        chg_24h = coin.get("price_change_percentage_24h", 0) or 0
        vol_24h = coin.get("total_volume", 0) or 0
        price   = coin.get("current_price", 0) or 0

        _ath = coin.get("ath", price) or price
        _atl = max(coin.get("atl", price) or price, 1e-9)
        _range_pct = (_ath - _atl) / _atl * 100 if _atl > 0 else 100
        # Stablecoin/pegged asset: USD-named, EUR-named, or price range < 15%
        if (any(s in symbol for s in stablecoins) or
                "usd" in symbol or "eur" in symbol or "usx" in symbol or
                coin_id in stable_ids or
                (0.85 < price < 1.15 and _range_pct < 15)):  # tighter peg detection
            continue
        if mktcap < config["min_market_cap_usd"]: continue
        if chg_24h > config["max_price_change_24h"]: continue
        if chg_24h < config["min_price_change_24h"]: continue
        if mktcap > 0 and vol_24h / mktcap < 0.003: continue

        survivors.append({"id": coin_id, "symbol": symbol.upper(), "name": coin.get("name",""),
                          "price": price, "chg_24h": chg_24h, "mktcap": mktcap, "vol_24h": vol_24h})
    return survivors


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2A: DEEP SCAN — LONG GEMS (Probabilistic)
#
# Gate order:
#   1. Data length / pegged asset
#   2. Volume ratio
#   3. Cloud position (below cloud = hard reject)
#   4. ML accuracy gate
#   5. Probabilistic scorer → P ≥ BUY_PROB_THRESHOLD (0.65)  ← KEY CHANGE
#   6. ML probability gate (secondary — already in scorer but explicit check)
#   7. OBV direction
#   8. Backtest win rate
# ══════════════════════════════════════════════════════════════════════════════

def _sf(val, default=0.0):
    """Safe float — handles None, pd.NA, NaN. Returns default on any error."""
    try:
        if val is None: return default
        f = float(val)
        return default if (f != f) else f   # catches NaN
    except (TypeError, ValueError):
        return default

def _sfnan(val):
    """Safe float with NaN default — for Ichimoku/cloud fields."""
    return _sf(val, float("nan"))


def _apply_lr(prob: float, lr: float) -> float:
    """Apply a likelihood ratio to a probability in log-odds space."""
    import math as _m
    try:
        p = max(0.001, min(0.999, prob))
        return 1.0 / (1.0 + _m.exp(-(_m.log(p/(1-p)) + _m.log(max(1e-6, lr)))))
    except Exception:
        return prob


def deep_scan_coin(coin_id, coin_info, btc_regime, btc_df, fng_value, config):
    """
    Full probabilistic scan. Returns (gem_dict, None) on pass, (None, reason) on fail.

    Architecture: three hard gates (stablecoin, volume, data) then everything
    else is a Bayesian LR applied to the posterior from generate_signal_probabilistic.

    Hard gates (structural — coin literally cannot be traded):
        1. Insufficient data  < 80 bars
        2. Pegged/stablecoin  price range < 10%
        3. Volume             < 20% of average

    LR adjustments (evidence for/against, proportional):
        4. Cloud position     LR 0.50–1.00 based on distance
        5. OBV direction      LR 0.80–1.15
        6. Backtest WR        LR 0.70–1.30
        7. ML probability     LR 0.80–1.25
        8. ADX strength       LR 0.90–1.05

    Output tiers:
        P ≥ 65%   → GEM  (notify)
        P 55–65%  → WATCH (listed at end, no notification)
        P < 55%   → filtered
    """
    try:
        # ── 1. FETCH + DATA CHECK (hard gate) ────────────────────────────────
        df = fetch_coin_data(coin_id, days=365)
        if df.empty or len(df) < 80:
            return None, "insufficient_data"

        last    = df.iloc[-1]
        price   = float(last["price"])
        chg_24h = float(coin_info.get("chg_24h", 0) or 0)

        # ── 2. PEGGED ASSET (hard gate) ───────────────────────────────────────
        _p_range = (df["price"].max() - df["price"].min()) / df["price"].mean() * 100
        if _p_range < 10:
            return None, f"pegged_asset_{_p_range:.1f}pct"

        # ── 3. VOLUME (hard gate) ─────────────────────────────────────────────
        _vol_now = _sf(last.get("volume"), 0)
        _vol_avg = _sf(df["vol_avg90"].iloc[-1]) if "vol_avg90" in df.columns else 0
        if _vol_avg > 0 and _vol_now / _vol_avg < config.get("min_vol_ratio", 0.20):
            return None, f"volume_low_{_vol_now/_vol_avg:.2f}x"

        # ── Z-SCORE (informational) ───────────────────────────────────────────
        prices_20 = df["price"].iloc[-20:].values
        z = float((prices_20[-1] - prices_20.mean()) / (prices_20.std() + 1e-9))

        # ── ML PREDICTION ─────────────────────────────────────────────────────
        _ensure_global_model()
        ml_model, ml_scaler, ml_accuracy, wf_sharpe, _ = load_model_for_scanner(coin_id)
        _using_global = False
        _global_ok    = bool(_GLOBAL_MODEL.get("model")) and _GLOBAL_MODEL.get("sharpe", 0) > 0.3

        if ml_model is None and _global_ok:
            _using_global = True
            ml_accuracy   = _GLOBAL_MODEL.get("acc", 0.5)
            wf_sharpe     = _GLOBAL_MODEL.get("sharpe", 0.0)

        if _using_global:
            ml_proba, _, _ = predict_global_ml(last, df, btc_df, coin_id,
                                               fng_value=int(fng_value))
            if wf_sharpe < 0:
                ml_proba = ml_proba * 0.50 + 0.5 * 0.50
        elif ml_model is None:
            ml_proba = 0.5
        elif wf_sharpe < -1.0:
            ml_proba = 0.5
        elif wf_sharpe > 0.5:
            ml_proba, _ = predict_ml_outcome(last, ml_model, ml_scaler,
                                             df=df, btc_df=btc_df, fng_value=fng_value)
            if ml_accuracy < 0.40:
                ml_proba = ml_proba * 0.50 + 0.5 * 0.50
            if _global_ok:
                _g, _, _ = predict_global_ml(last, df, btc_df, coin_id,
                                             fng_value=int(fng_value))
                ml_proba = _g * 0.70 + ml_proba * 0.30
        elif ml_accuracy <= 0.50:
            ml_proba = 0.5
        else:
            ml_proba, _ = predict_ml_outcome(last, ml_model, ml_scaler,
                                             df=df, btc_df=btc_df, fng_value=fng_value)
            if ml_accuracy < 0.60:
                ml_proba = ml_proba * 0.70 + 0.5 * 0.30

        # ── OBV DIRECTION ─────────────────────────────────────────────────────
        _, _, obv_trend, _ = check_obv_confirmation(df, coin_id)

        # ── BAYESIAN SCORER ───────────────────────────────────────────────────
        import io as _io, sys as _sys, logging as _log
        _old_out = _sys.stdout; _sys.stdout = _io.StringIO()
        _old_lvl = _log.getLogger("TradingBot").level
        _log.getLogger("TradingBot").setLevel(_log.CRITICAL + 1)
        try:
            _scorer, sig = generate_signal_probabilistic(
                last, df,
                btc_regime     = btc_regime,
                is_btc         = (coin_id == "bitcoin"),
                suppress_debug = True,
                fng_value      = int(fng_value),
            )
            if sig.get("prob") is None:
                try:
                    _res        = _scorer.finalize()
                    sig["prob"] = float(_res.prob_continuation)
                except Exception:
                    sig["prob"] = 0.0
        finally:
            _sys.stdout = _old_out
            _log.getLogger("TradingBot").setLevel(_old_lvl)


        prob    = float(sig.get("prob") or 0.0)
        blocked = sig.get("blocked", False)

        # Only hard-block on EXECUTION issues — can't fill the trade at all.
        # Signal-quality blockers (BTC Bear Regime, Weekly EMA, macro headwinds)
        # are already captured as LRs — hard-blocking them double-penalises.
        # Very Low Volume is execution: 20-50% avg passes prefilter but scorer
        # catches it here. We still want to block — can't execute thin books.
        _exec_blockers  = {"Very Low Volume", "No Data", "Exchange Issue",
                           "Very Low Volume"}
        _active_blockers = set(sig.get("blockers", []) or [])
        if _active_blockers & _exec_blockers:
            _reason = next(iter(_active_blockers & _exec_blockers))
            return None, f"exec_{_reason[:25].replace(' ','_').lower()}"

        # ── 3. VOLUME LR ─────────────────────────────────────────────────────
        # < 20% average  → hard gate above already caught this
        # 20-50% average → LR=0.75 (execution is harder, spread wider)
        # 50-80% average → LR=0.90 (mild headwind)
        # > 80% average  → LR=1.00 (normal)
        _vol_now = _sf(last.get("volume"), 0)
        _vol_avg_raw = _sf(df["vol_avg90"].iloc[-1]) if "vol_avg90" in df.columns else 0
        _vol_ratio = _vol_now / max(_vol_avg_raw, 1e-9) if _vol_avg_raw > 0 else 1.0
        _vol_lr = (0.75 if _vol_ratio < 0.50 else
                   0.90 if _vol_ratio < 0.80 else 1.00)
        # No hard rejection — graduated LR based on distance from cloud.
        # Above cloud = no penalty. Deep below cloud = strong headwind.
        _cloud_state = sig.get("cloud_state", "unknown")
        prob = _apply_lr(prob, _vol_lr)
        _cloud_lr    = 1.0
        _cloud_note  = ""
        try:
            _cloud_valid = df[["senkou_a","senkou_b"]].dropna()
            if len(_cloud_valid) > 0:
                _sa = float(_cloud_valid["senkou_a"].iloc[-1])
                _sb = float(_cloud_valid["senkou_b"].iloc[-1])
                _ct = max(_sa, _sb); _cb = min(_sa, _sb)
                if price < _cb:
                    _gap = (_ct - price) / price * 100
                    _cloud_lr = (0.80 if _gap <  5 else
                                 0.70 if _gap < 10 else
                                 0.60 if _gap < 15 else
                                 0.55 if _gap < 25 else 0.45)
                    _cloud_note = f"below☁ {_gap:.1f}%"
                    coin_info["cloud_below"]   = True
                    coin_info["cloud_gap_pct"] = round(_gap, 1)
                elif price < _ct:
                    _gap = (_ct - price) / price * 100
                    _cloud_lr = (0.95 if _gap < 5 else
                                 0.85 if _gap < 10 else 0.75)
                    _cloud_note = f"in☁ {_gap:.1f}%"
                    coin_info["cloud_watch"]   = True
                    coin_info["cloud_gap_pct"] = round(_gap, 1)
        except Exception:
            pass
        prob = _apply_lr(prob, _cloud_lr)

        # ── 5. OBV LR ─────────────────────────────────────────────────────────
        _obv_lr = (1.15 if obv_trend == "rising" else
                   0.80 if obv_trend == "falling" else 1.00)
        prob    = _apply_lr(prob, _obv_lr)

        # ── 6. BACKTEST LR ────────────────────────────────────────────────────
        backtest_wr = 50.0; backtest_pf = 1.0
        _bt_lr = 1.0
        if BACKTEST_ENABLED:
            try:
                _bt_wins = 0; _bt_losses = 0
                _bt_c = df["close"].values if "close" in df.columns else df["price"].values
                _bt_a = df["ATR"].values   if "ATR"   in df.columns else None
                if _bt_a is not None and len(_bt_c) > 40:
                    for _bi in range(20, len(_bt_c) - 20):
                        _e = float(_bt_c[_bi]); _a = float(_bt_a[_bi])
                        if _a <= 0: continue
                        _tp = _e + 2.5*_a; _sl = _e - 1.5*_a
                        _hit = None
                        for _pf in _bt_c[_bi+1:_bi+21]:
                            if float(_pf) >= _tp: _hit = "win";  break
                            if float(_pf) <= _sl: _hit = "loss"; break
                        if   _hit == "win":  _bt_wins   += 1
                        elif _hit == "loss": _bt_losses += 1
                _n = _bt_wins + _bt_losses
                if _n > 0:
                    backtest_wr = _bt_wins / _n * 100
                    backtest_pf = (_bt_wins / _bt_losses) if _bt_losses > 0 else 2.0
                    _bt_lr = (1.30 if backtest_wr >= 50 else
                              1.15 if backtest_wr >= 40 else
                              1.00 if backtest_wr >= 33 else
                              0.85 if backtest_wr >= 25 else 0.70)
                    prob = _apply_lr(prob, _bt_lr)
            except Exception:
                pass

        # ── 7. ML LR ──────────────────────────────────────────────────────────
        # Strengthened for low ml_proba: when the bot's own ML_ENGINE would
        # classify this as "AVOID LONGS" (ml_proba < 35%), the LR must be
        # severe enough to actually move the needle. Old 0.80 was cosmetic —
        # a coin at P=86% with ML=30% only dropped to P=83%.
        _ml_lr = 1.0
        if ml_proba != 0.5 and ml_accuracy >= 0.50:
            _ml_lr = (1.25 if ml_proba >= 0.70 else
                      1.10 if ml_proba >= 0.60 else
                      1.00 if ml_proba >= 0.50 else
                      0.90 if ml_proba >= 0.40 else
                      0.70 if ml_proba >= 0.30 else   # was 0.80 for <40%
                      0.55)                            # NEW: <30% = AVOID LONGS territory
            prob = _apply_lr(prob, _ml_lr)

        # ── 8. ADX LR (mild) ─────────────────────────────────────────────────
        _adx = _sf(last.get("ADX"), 0)
        _adx_lr = (1.05 if _adx >= 25 else
                   1.00 if _adx >= 15 else 0.92)
        prob = _apply_lr(prob, _adx_lr)

        # ── 9. EMA26 DEVIATION — EXHAUSTION LR ──────────────────────────────────
        # Coins far above EMA26 have already made their move. ema26_stretched
        # fires inside generate_signal_probabilistic but doesn't scale enough
        # for extreme deviations (20-80%+) seen on newly-discovered/post-pump
        # coins. This LR adds graduated penalty on top.
        #   dev > 50%  → LR=0.50  severe — likely already topped (parabolic)
        #   dev > 30%  → LR=0.65  heavily extended
        #   dev > 20%  → LR=0.80  extended
        #   dev > 12%  → LR=0.92  mildly extended
        #   else       → LR=1.00  normal
        ema26   = _sf(last.get("EMA26"), price)
        dev_pct = (price - ema26) / ema26 * 100 if ema26 > 0 else 0
        _dev_lr = (0.50 if dev_pct > 50 else
                   0.65 if dev_pct > 30 else
                   0.80 if dev_pct > 20 else
                   0.92 if dev_pct > 12 else 1.00)
        prob = _apply_lr(prob, _dev_lr)

        # ── 10. ENTROPY SHRINKAGE — uncertain regime classification ────────────
        # Shrink probability toward 0.5 based on how uncertain the regime
        # classification is (soft bull/bear/range/chop/panic distribution,
        # Shannon entropy over it). Stable market -> full confidence kept.
        # Chaotic/unstable -> conviction pulled toward 50/50, direction kept.
        # Same EntropyTracker singleton the main bot writes to (adaptive
        # percentile thresholds learned from shared history). No chop-detection
        # input here (unlike the main bot) since the scanner has no chop signal
        # of its own -- defaults to False, losing one minor contributing signal
        # to the chop score while ADX/cloud/MACD/RSI/z-score still feed it normally.
        # Runs BEFORE regime cal below, same order the main bot uses.
        try:
            _re = compute_regime_entropy(
                last=last, df=df, sig=sig,
                adx=float(last.get("ADX", 20) or 20),
                chop_detected=False,
            )
            prob, _entropy_note = apply_regime_entropy(prob, _re)
            sig["regime_entropy"] = _re
            if _re.label in ("UNSTABLE", "VOLATILE", "CHAOTIC"):
                print(Fore.YELLOW + _entropy_note)
            else:
                print(_entropy_note)
        except Exception as _ee:
            print(Fore.RED + f"[ENTROPY] {coin_id}: ⚠️  compute/apply_regime_entropy "
                  f"CRASHED — {type(_ee).__name__}: {_ee}")
            _re = None
            sig["regime_entropy"] = None

        # ── 11. REGIME CALIBRATION — empirical bear/bull/chop correction ───────
        # Same Brier-monitor singleton the main bot writes to (models/brier_predictions.json),
        # so scanner and bot share identical historical calibration data. Shifts log-odds by
        # the gap between this regime's actual resolved win rate and the model's average
        # predicted confidence in that regime. No-ops until >= REGIME_MIN_N (12) resolved
        # trades exist for the regime, or if the bias is negligible (<3%). No chop-detection
        # override here (unlike the main bot) since the scanner has no chop signal of its own.
        try:
            _regime_for_cal = sig.get("market_regime", "unknown")
            _prob_before_rc = prob
            prob, _rc_note  = _brier_monitor.regime_calibrate(prob, _regime_for_cal)
            sig["regime_calibration_applied"] = abs(prob - _prob_before_rc) >= 0.01
            if sig["regime_calibration_applied"]:
                print(f"[REGIME CAL] {coin_id}: {_rc_note}")
        except Exception:
            sig["regime_calibration_applied"] = False

        # ── FINAL PROBABILITY TIERS ───────────────────────────────────────────
        _min_prob   = config.get("min_prob",    BUY_PROB_THRESHOLD)   # 0.65
        _watch_prob = config.get("watch_prob",  0.55)                  # new WATCH tier
        _strong_prob= config.get("strong_prob", STRONG_BUY_THRESHOLD)  # 0.72

        if prob < (_watch_prob - 0.001):   # -0.001 tolerance for float rounding
            return None, f"P={prob:.1%}_(needs_{_watch_prob:.0%})"

        # Supplementary metrics for display
        direction, dir_conf = predict_direction(df)
        try:
            is_whale, _, _ = detect_whale_volume(df)
            is_whale = bool(is_whale)
        except Exception:
            is_whale = False
        sm_mode, sm_conf, _ = detect_smart_money_accumulation(df)

        # ── EV — compute BEFORE the tradeability report so it sees it ────────
        # compute_tradeability_report reads sig["ev"] as a dict. In run_once()
        # this is set at line ~17163 before the report runs. Here we must do
        # the same — otherwise the report reads {} and shows EV=+0.00 (?).
        _rr      = _sf(sig.get("rr_ratio"), 2.0)
        _ev_val  = prob * _rr - (1 - prob) if _rr > 0 else 0.0
        _ev_val  = round(_ev_val, 2)
        _ev_rate = ("A+" if _ev_val >= 1.5 else "A"  if _ev_val >= 1.0 else
                    "B"  if _ev_val >= 0.5 else "C"  if _ev_val >= 0.2 else
                    "D"  if _ev_val >= 0.0 else "F")
        sig["ev"] = {
            "ev":        _ev_val,
            "rating":    _ev_rate,
            "adj_rr":    _rr,
            "kelly":     max(0.0, (_ev_val / _rr)) if _rr > 0 else 0.0,
            "breakeven": round(1 / (1 + _rr), 3)   if _rr > 0 else 0.5,
            "comment":   "scanner",
        }

        _report = None
        try:
            import io as _io2, sys as _sys2, logging as _log2
            _old2 = _sys2.stdout; _sys2.stdout = _io2.StringIO()
            try:
                _report = compute_tradeability_report(
                    coin=coin_id, sig=sig, prob=prob, last=last, df=df,
                    ml_proba=ml_proba, ml_accuracy=ml_accuracy,
                    btc_regime=btc_regime)
            finally:
                _sys2.stdout = _old2
        except Exception:
            pass

        # ── Extract display fields from sig ───────────────────────────────────
        _lc       = sig.get("lifecycle") or {}
        _lc_stage = _lc.get("stage", "UNKNOWN") if isinstance(_lc, dict) else "UNKNOWN"

        # lifecycle is set in run_once() — compute directly here
        if _lc_stage == "UNKNOWN":
            try:
                _cur_phase = sig.get("market_phase", "RANGE")
                _lifecycle = classify_trend_lifecycle(
                    last, df, phase=_cur_phase, obv_trend=obv_trend)
                _lc_stage  = (_lifecycle.get("stage", "UNKNOWN")
                              if isinstance(_lifecycle, dict) else str(_lifecycle))
            except Exception:
                pass

        # RS is set by run_once() after generate_signal_probabilistic — not in sig here.
        # Compute directly using the same function the main bot uses.
        _rs_label = "N/A"
        try:
            _rs_full  = compute_relative_strength_full(df, btc_df)
            _rs_label = _rs_full.get("label", "N/A") if isinstance(_rs_full, dict) else "N/A"
            sig["rs_full"] = _rs_full   # store so other code can use it
        except Exception:
            pass
        _entry    = _sf(sig.get("entry_price"), price)
        _stop     = _sf(sig.get("stop_price"),  price * 0.93)
        _tp       = _sf(sig.get("tp_price"),    price * 1.15)
        _rsi      = _sf(last.get("RSI"),        50.0)
        _vol_regime = sig.get("vol_regime", "normal") or "normal"

        # ── Kelly position sizing ─────────────────────────────────────────────
        # Full Kelly: f* = (P × RR - (1-P)) / RR
        # Quarter Kelly (conservative): f* × 0.25
        # Capped at 3% — never risk more than 3% on a single trade
        try:
            _kelly_full  = max(0.0, (prob * _rr - (1 - prob)) / _rr)
            _kelly_q     = _kelly_full * 0.25                    # quarter Kelly
            _risk_pct    = min(_kelly_q, 0.03)                   # cap at 3%
            _stop_dist   = abs(_entry - _stop) / _entry if _entry > 0 else 0.08
            _pos_size    = _risk_pct / _stop_dist if _stop_dist > 0 else 0.0
            _kelly_str   = (f"{_risk_pct*100:.1f}% risk  "
                           f"(¼Kelly={_kelly_q*100:.1f}%  full={_kelly_full*100:.0f}%)")
            _pos_str     = f"{_pos_size*100:.1f}% of capital"
        except Exception:
            _kelly_str = "N/A"; _pos_str = "N/A"

        # ── Whale continuity check ────────────────────────────────────────────
        # Single-day whale spike ≠ accumulation.
        # Check how many of the last 3 days had whale-level volume.
        _whale_days  = 0
        _whale_trend = "unknown"
        try:
            _w_avg = (_sf(df["vol_avg90"].iloc[-1]) if "vol_avg90" in df.columns
                      else float(df["volume"].rolling(90, min_periods=20).mean().iloc[-1]))
            _w_threshold = _w_avg * 2.0   # 200% = whale level
            _recent_vols = [_sf(df["volume"].iloc[i]) for i in [-4, -3, -2]]
            _whale_days  = sum(1 for v in _recent_vols if v > _w_threshold)
            _today_vol   = _sf(last.get("volume"), 0)
            _yest_vol    = _sf(df["volume"].iloc[-2], 0)
            _whale_trend = "rising" if _today_vol > _yest_vol * 1.1 else (
                           "falling" if _today_vol < _yest_vol * 0.9 else "flat")
        except Exception:
            pass
        _whale_str = (
            f"🐋 CONTINUOUS {_whale_days+1}d (strong)"  if _whale_days >= 2 else
            f"🐋 2d elevated ({'↑' if _whale_trend=='rising' else '↓' if _whale_trend=='falling' else '→'})" if _whale_days == 1 else
            f"⚠️ single spike ({_whale_trend}) — verify"
        ) if is_whale else "none"
        _whale_str = _whale_str or "none"   # guard against any None edge case
        # Tradeability scores from report
        _entry_score  = getattr(_report, "entry_score",        0) if _report else 0
        _trade_score  = getattr(_report, "tradeability_score", 0) if _report else 0
        _entry_label  = getattr(_report, "entry_label",        "") if _report else ""
        _trade_label  = getattr(_report, "tradeability_label", "") if _report else ""

        gem = {
            "id":          coin_id,
            "name":        coin_info.get("name", coin_id),
            "symbol":      coin_info.get("symbol", "?").upper(),
            "price":       price,
            "is_futures":  coin_id in FUTURES_ELIGIBLE,
            "prob":        round(prob, 4),
            "is_strong":   prob >= _strong_prob,
            "is_watch":    prob < _min_prob,   # True = WATCH tier (55-65%)
            "ml_accuracy": ml_accuracy,
            "ml_proba":    ml_proba,
            "backtest_wr": backtest_wr,
            "z_score":     z,
            "ema26_dev":   dev_pct,
            "obv_trend":   obv_trend,
            "direction":   direction,
            "dir_conf":    dir_conf,
            "smart_money": sm_mode,
            "is_whale":    is_whale,
            "adx":         _adx,
            "chg_24h":     coin_info.get("chg_24h", 0),
            "mktcap":      coin_info.get("mktcap", 0),
            "cloud_note":  _cloud_note,
            "cloud_below": coin_info.get("cloud_below", False),
            "cloud_watch": coin_info.get("cloud_watch", False),
            "cloud_gap":   coin_info.get("cloud_gap_pct", 0),
            "lr_cloud":    _cloud_lr,
            "lr_obv":      _obv_lr,
            "lr_bt":       _bt_lr,
            "lr_ml":       _ml_lr,
            "lr_adx":      _adx_lr,
            "lr_dev":      _dev_lr,
            "lifecycle":   _lc_stage,
            "rs_label":    _rs_label,
            "ev":          round(_ev_val, 2),
            "ev_rating":   _ev_rate,
            "entry":       _entry,
            "stop":        _stop,
            "tp":          _tp,
            "rr":          round(_rr, 2),
            "rsi":         round(_rsi, 1),
            "vol_regime":  _vol_regime,
            "btc_regime":  btc_regime,
            "entry_score": _entry_score,
            "trade_score": _trade_score,
            "entry_label": _entry_label,
            "trade_label": _trade_label,
            "kelly":       _kelly_str,
            "pos_size":    _pos_str,
            "whale_str":   _whale_str,
            "tradeability_report": _report,
            "scan_hour":   datetime.utcnow().hour,
        }
        return gem, None

    except Exception as e:
        import traceback as _tb
        _tb_str = _tb.format_exc().splitlines()
        _fail   = next((l.strip() for l in reversed(_tb_str) if "File" in l), "")
        print(f"\n[SCAN ERROR] {coin_id}: {type(e).__name__}: {str(e)}")
        if _fail: print(f"[SCAN ERROR] at: {_fail}")
        return None, f"error_{str(e)[:30]}"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2B: DEEP SCAN — SHORT GEMS (Probabilistic)
#
# Gate order:
#   1. Data length
#   2. ADX ≥ 15 (need a real trend to short)
#   3. OBV not rising (capital flowing out or flat)
#   4. Short ML accuracy + probability
#   5. SHORT SCORER → short_prob ≥ SHORT_PROB_THRESHOLD (0.65)  ← KEY CHANGE
#   6. Cloud (context: below = strong, above = counter-trend)
#
# SHORT SCORER (simplified standalone):
#   Runs the same 6-question Bayesian short scorer as the main bot.
#   Uses detect_* functions imported from the bot, same ProbabilisticScorer.
# ══════════════════════════════════════════════════════════════════════════════

def run_short_scorer_standalone(coin_id, df, last, btc_regime,
                                 short_ml_proba, short_ml_acc,
                                 fng_value=50, btc_df=None, sig=None):
    """
    Run the authoritative 6-question short scorer from the main bot.
    Uses run_short_scorer() directly — same function as run_once().

    KEY: pass sig= (from generate_signal_probabilistic) to get accurate
    market_regime and phase data. Without it, market_regime defaults to
    "bear" and phase_dur=0 which produces different priors and decay.
    """
    try:
        _sig = sig or {}
        # Use sig's detected regime — "bear" + BTC bear → btc_bear_confluence (P₀=0.62)
        # Without sig, market_regime defaults to "bear" which is still correct
        # but phase/phase_dur may differ
        _market_regime = _sig.get("market_regime", "bear") or "bear"
        _phase         = str(_sig.get("market_phase", last.get("market_phase", "RANGE")) or "RANGE")
        _phase_dur     = int(_sig.get("phase_duration", last.get("phase_duration", 0)) or 0)
        _adx           = _sf(last.get("ADX"), 20)

        short_prior, _label, freshness_lr = detect_short_prior(
            market_regime=_market_regime,
            btc_regime=btc_regime,
            fng_value=float(fng_value),
            adx=_adx,
            phase=_phase,
            phase_dur=_phase_dur,
        )

        short_scorer = ProbabilisticScorer(
            coin=coin_id,
            prior=float(short_prior),
            prior_regime=btc_regime,
        )

        # Add freshness LR if present
        if abs(float(freshness_lr) - 1.0) > 0.02:
            short_scorer.add_lr(float(freshness_lr), "phase_freshness", group="OTHER")

        # Get OBV and vol regime
        obv_trend = "unknown"
        try:
            _, _, obv_trend, _ = check_obv_confirmation(df, coin_id)
        except Exception:
            pass
        vol_regime = str(last.get("vol_regime", "neutral") or "neutral")

        # Run the real 6-question scorer (same as main bot run_once)
        import io as _io2, sys as _sys2, logging as _log2
        _old2 = _sys2.stdout; _sys2.stdout = _io2.StringIO()
        _old_lvl2 = _log2.getLogger("TradingBot").level
        _log2.getLogger("TradingBot").setLevel(_log2.CRITICAL + 1)
        try:
            run_short_scorer(
                short_scorer  = short_scorer,
                coin          = coin_id,
                df            = df,
                last          = last,
                obv_trend     = obv_trend,
                vol_regime    = vol_regime,
                phase         = _phase,
                phase_dur     = _phase_dur,
                short_ml_proba= float(short_ml_proba),
                short_ml_acc  = float(short_ml_acc),
            )
        finally:
            _sys2.stdout = _old2
            _log2.getLogger("TradingBot").setLevel(_old_lvl2)

        result = short_scorer.finalize()
        return float(result.prob_continuation)

    except Exception as e:
        # Fallback: use short ML as weak proxy (not raw probability!)
        if short_ml_acc >= 0.60 and 0 < short_ml_proba < 1:
            # Blend ML with prior toward 0.5 to avoid returning raw 98%
            return float(0.5 + (short_ml_proba - 0.5) * 0.5)
        return 0.35


def deep_scan_short(coin_id, coin_info, btc_regime, btc_df, fng_value, config=None):
    """
    Run probabilistic short scanner pipeline on one coin.

    KEY CHANGE:
      Old: sell_score ≤ -5 + short_ml_proba ≥ 0.40
      New: short_prob ≥ SHORT_PROB_THRESHOLD (Bayesian short scorer)
    """
    if config is None:
        config = SHORT_SCANNER_CONFIG

    try:
        import io as _io, sys as _sys, logging as _log
        _old_out = _sys.stdout; _sys.stdout = _io.StringIO()
        _old_lvl = _log.getLogger("TradingBot").level
        _log.getLogger("TradingBot").setLevel(_log.CRITICAL + 1)

        df = fetch_coin_data(coin_id, days=365)
        _sys.stdout = _old_out; _log.getLogger("TradingBot").setLevel(_old_lvl)

        if df.empty or len(df) < 80:
            return {"rejected": True, "reason": "no_data",
                    "symbol": coin_info.get("symbol","?").upper(), "price": 0,
                    "chg_24h": coin_info.get("chg_24h", 0)}

        last  = df.iloc[-1]
        price = float(last["price"])
        adx   = _sf(last.get("ADX"), 0)

        # Gate 1: ADX
        if adx < config.get("min_adx", 15):
            return {"rejected": True, "reason": f"adx_weak_{adx:.1f}",
                    "symbol": coin_info.get("symbol","?").upper(), "price": price,
                    "chg_24h": coin_info.get("chg_24h", 0), "adx": adx}

        # Gate 2: OBV
        _, _, obv_trend, _ = check_obv_confirmation(df, coin_id)
        if obv_trend == "rising":
            return {"rejected": True, "reason": "obv_rising",
                    "symbol": coin_info.get("symbol","?").upper(), "price": price,
                    "chg_24h": coin_info.get("chg_24h", 0), "obv_trend": obv_trend}

        # Gate 3: Short ML
        # Always retrain short model too — same reason as long model above
        short_model, short_scaler, short_acc = None, None, 0
        import io as _iosml, sys as _syssml, logging as _logsml
        _old_sml = _syssml.stdout; _syssml.stdout = _iosml.StringIO()
        _logsml.getLogger("TradingBot").setLevel(_logsml.CRITICAL + 1)
        try:
            short_model, short_scaler, short_acc, _ = train_short_ml_model(
                df, coin_id, btc_df=btc_df, fng_value=fng_value)
            if short_model:
                save_short_model(coin_id, short_model, short_scaler, short_acc, len(df))
        finally:
            _syssml.stdout = _old_sml

        if short_acc < 0.30:
            # Below 30% — model so inverse it actively misleads
            return {"rejected": True, "reason": f"short_ml_unreliable_{short_acc:.0%}",
                    "symbol": coin_info.get("symbol","?").upper(), "price": price,
                    "chg_24h": coin_info.get("chg_24h", 0),
                    "short_ml_accuracy": short_acc}

        # ── Gate 3b: Cloud position — filter "above cloud" shorts ────────────
        # Shorting above cloud = fighting structural bull trend.
        # Only allow above-cloud shorts in bear regime AND strong Bayesian case.
        sa = _sfnan(last.get("senkou_a"))
        sb = _sfnan(last.get("senkou_b"))
        cloud_pos = "unknown"
        if not (np.isnan(sa) or np.isnan(sb)):
            ct = max(sa, sb); cb = min(sa, sb)
            cloud_pos = "below" if price < cb else ("inside" if price < ct else "above")

        if cloud_pos == "above" and btc_regime != "bear":
            return {"rejected": True, "reason": "above_cloud_not_bear",
                    "symbol": coin_info.get("symbol","?").upper(), "price": price,
                    "chg_24h": coin_info.get("chg_24h", 0), "cloud_pos": cloud_pos}

        # ── Short ML prediction ───────────────────────────────────────────────
        # FIXED: line `short_ml_proba = 0.5` was hardcoded here, resetting
        # ML to neutral for ALL coins regardless of model output. Removed.
        short_ml_proba = 0.5
        if short_model and short_acc >= config.get("min_short_ml_accuracy", 0.55):
            short_ml_proba, _ = predict_short_ml_outcome(
                last, short_model, short_scaler, df=df, btc_df=btc_df, fng_value=fng_value)
            if short_acc < 0.70:
                short_ml_proba = short_ml_proba * 0.70 + 0.5 * 0.30
        elif short_model and short_acc >= 0.45:
            # Marginal accuracy — use dampened prediction
            short_ml_proba, _ = predict_short_ml_outcome(
                last, short_model, short_scaler, df=df, btc_df=btc_df, fng_value=fng_value)
            short_ml_proba = short_ml_proba * 0.40 + 0.5 * 0.60   # heavy dampen
        # else: short_ml_proba stays 0.5 (neutralized)

        # Gate: if ML actively bearish (short unlikely to work), reject
        # ── Use global model for sML when no per-coin short model ─────────────
        # MUST run before the min_short_ml_proba gate so global prediction
        # can replace the 0.5 default BEFORE that gate fires.
        if short_ml_proba == 0.5 and _GLOBAL_MODEL.get("model") and _GLOBAL_MODEL.get("sharpe", 0) > 0:
            try:
                _, _g_short_ml, _ = predict_global_ml(
                    last, df, btc_df, coin_id, fng_value=int(fng_value))
                # _g_short_ml is already rescaled relative to threshold
                # Values >0.55 = above threshold (short likely to work)
                # Values <0.50 = below threshold (short unlikely to work)
                if _g_short_ml != 0.5:
                    short_ml_proba = _g_short_ml * 0.70 + 0.5 * 0.30  # 70% global weight
            except Exception:
                pass

        # Gate: reject only if ML is actively bearish on the short (not just neutral)
        # 0.5 = neutral (no model or model uncertain)
        # < 0.40 = model says short unlikely to work
        if short_ml_proba < config.get("min_short_ml_proba", 0.40):  # lowered from 0.45
            return {"rejected": True, "reason": f"short_ml_low_{short_ml_proba:.0%}",
                    "symbol": coin_info.get("symbol","?").upper(), "price": price,
                    "chg_24h": coin_info.get("chg_24h", 0),
                    "short_ml_accuracy": short_acc, "short_ml_proba": short_ml_proba}

        # For above-cloud shorts in bear regime: require higher Bayesian threshold
        # AND ML must not be completely neutral (sML=50% + above cloud = too weak)
        _min_short_prob_adj = config.get("min_short_prob", SHORT_PROB_THRESHOLD)
        if cloud_pos == "above":
            _min_short_prob_adj = max(_min_short_prob_adj, 0.82)  # was 0.78, raised
            # Also require ML to give at least a slightly bearish reading
            # sML=50% (neutralized) + above cloud = no real short edge
            if short_ml_proba <= 0.52:
                return {"rejected": True, "reason": "above_cloud_no_ml_confirm",
                        "symbol": coin_info.get("symbol","?").upper(), "price": price,
                        "chg_24h": coin_info.get("chg_24h", 0), "cloud_pos": cloud_pos,
                        "short_ml_proba": short_ml_proba}

        # Gate 4: PROBABILISTIC SHORT SCORER ─────────────────────────────────
        import io as _io2, sys as _sys2, logging as _log2
        _old2 = _sys2.stdout; _sys2.stdout = _io2.StringIO()
        _old_lvl2 = _log2.getLogger("TradingBot").level
        _log2.getLogger("TradingBot").setLevel(_log2.CRITICAL + 1)
        try:
            short_prob = run_short_scorer_standalone(
                coin_id, df, last, btc_regime,
                short_ml_proba, short_acc,
                fng_value=fng_value, btc_df=btc_df,
                sig=None)
        finally:
            _sys2.stdout = _old2
            _log2.getLogger("TradingBot").setLevel(_old_lvl2)

        # ── ENTROPY SHRINKAGE (short side) ────────────────────────────────────
        # Same market entropy applies to both directions: a chaotic/choppy
        # regime is unreliable for shorts as much as for longs. Pull short_prob
        # toward 0.5 when the regime classification itself is uncertain.
        # Applied BEFORE the threshold gate so the corrected probability is
        # what gets compared, not the inflated raw scorer output.
        # (No long-side sig to reuse here — compute fresh from last/df.)
        try:
            _re_s = compute_regime_entropy(
                last=last, df=df, sig={},
                adx=float(last.get("ADX", 20) or 20),
                chop_detected=False,
            )
            _short_prob_pre = short_prob
            short_prob, _se_note = apply_regime_entropy(short_prob, _re_s)
            if abs(short_prob - _short_prob_pre) >= 0.01:
                print(f"[ENTROPY-S] {coin_id}: {_re_s.label}  H={_re_s.entropy_norm:.3f}  "
                      f"scale={_re_s.confidence_scale:.2f}  "
                      f"short_P: {_short_prob_pre:.3f} → {short_prob:.3f}")
        except Exception as _ese:
            print(Fore.RED + f"[ENTROPY-S] {coin_id}: ⚠️  CRASHED — "
                  f"{type(_ese).__name__}: {_ese}")

        if short_prob < _min_short_prob_adj:
            return {"rejected": True, "reason": f"short_prob_low_{short_prob:.0%}",
                    "symbol": coin_info.get("symbol","?").upper(), "price": price,
                    "chg_24h": coin_info.get("chg_24h", 0),
                    "short_prob": short_prob, "short_ml_proba": short_ml_proba}

        # ── Signal quality gate ────────────────────────────────────────────────
        # Problem: in a bear market, OBV is falling for almost EVERY coin, so
        # the old "_adx_ok OR _rsi_ok OR _obv_fall" gate was effectively
        # disabled — OBV falling alone was sufficient for any coin to pass,
        # producing 20-60 "short gems" per scan that were really just
        # "everything is down in a bear market."
        # Fix: require OBV falling AND at least one momentum signal.
        #   - OBV falling is the minimum bar (confirmed distribution)
        #   - AND either ADX≥20 (price actually trending down, not choppy)
        #        OR RSI≥65 (overbought/extended at high — rare in bear, strong signal)
        # Pure "below cloud + bear regime + OBV falling" without directional
        # momentum doesn't count.
        if config.get("require_short_signal", True):
            _adx_ok   = _sf(last.get("ADX"), 0) >= 20  # trending down (raised from 18)
            _rsi_ok   = float(last.get("RSI", 50) or 50) >= 65  # overbought/extended
            _obv_fall = obv_trend == "falling"
            _has_signal = _obv_fall and (_adx_ok or _rsi_ok)   # AND — not OR
            if not _has_signal:
                _why = ("obv_not_falling" if not _obv_fall
                        else f"obv_falling_but_adx_{_sf(last.get('ADX'),0):.0f}_and_rsi_not_ob")
                # High-confidence exception: P≥80% is too strong to drop entirely.
                # Flag as watch-only (no notification) so it appears in the watch
                # list instead of disappearing. The signal gate blocks notifications,
                # not awareness.
                if short_prob >= 0.80:
                    return {"rejected": False, "watch_only": True,
                            "watch_reason": f"signal_gate ({_why})",
                            "symbol": coin_info.get("symbol","?").upper(), "price": price,
                            "chg_24h": coin_info.get("chg_24h", 0),
                            "short_prob": short_prob, "adx": _sf(last.get("ADX"), 0),
                            "obv_trend": obv_trend, "cloud_pos": cloud_pos}
                return {"rejected": True, "reason": f"no_short_signal_{_why}",
                        "symbol": coin_info.get("symbol","?").upper(), "price": price,
                        "chg_24h": coin_info.get("chg_24h", 0),
                        "short_prob": short_prob}

        # cloud_pos already computed above (Gate 3b)

        # Stop / target
        atr_val = _sf(last.get("ATR"), 0)
        stop_loss     = price + atr_val * 1.5 if atr_val > 0 else price * 1.05
        profit_target = price - atr_val * 2.5 if atr_val > 0 else price * 0.90
        risk   = stop_loss - price
        reward = price - profit_target
        rr     = reward / risk if risk > 0 else 0

        # Sanity check: reject if ATR is so large relative to price that the
        # target would be zero or negative (physically impossible for a price).
        # BEAT showed $-1.03 target — ATR > price/2.5 means the coin is too
        # volatile for a fixed-multiple ATR stop/target to make sense.
        if profit_target <= 0:
            return {"rejected": True, "reason": f"impossible_target_{profit_target:.4f}",
                    "symbol": coin_info.get("symbol","?").upper(), "price": price,
                    "chg_24h": coin_info.get("chg_24h", 0),
                    "short_prob": short_prob}

        # Priority
        is_strong = short_prob >= config.get("strong_short_prob", SHORT_STRONG_THRESHOLD)
        priority  = "urgent" if is_strong else ("default" if cloud_pos == "above" else "high")

        # Liquidity
        book_usd = 0; liq_source = "order_book"
        try:
            book_usd, _, _ = analyze_order_book_liquidity(coin_id)
        except Exception:
            pass
        if book_usd <= 0:
            vol_24h = coin_info.get("vol_24h", 0) or 0
            if vol_24h > 0:
                book_usd = vol_24h * 0.005; liq_source = f"vol_estimate"
        liq_key, liq_label, max_pos = get_liquidity_tier(book_usd)

        return {
            "id": coin_id, "name": coin_info.get("name", coin_id),
            "symbol": coin_info.get("symbol","?").upper(),
            "price": price, "chg_24h": coin_info.get("chg_24h", 0),
            "adx": adx, "short_prob": short_prob, "is_strong": is_strong,
            "short_ml_accuracy": short_acc, "short_ml_proba": short_ml_proba,
            "obv_trend": obv_trend, "cloud_pos": cloud_pos,
            "priority": priority, "book_usd": book_usd,
            "liq_label": liq_label, "liq_source": liq_source, "max_pos": max_pos,
            "tradeable": liq_key != "paper_thin",
            "stop_loss": stop_loss, "profit_target": profit_target, "rr_ratio": rr,
            "cg_id": coin_id, "binance_id": get_exchange_symbol(coin_id) or "N/A",
        }

    except Exception as e:
        return {"rejected": True, "reason": f"error_{str(e)[:30]}",
                "symbol": coin_info.get("symbol","?").upper(), "price": 0,
                "chg_24h": coin_info.get("chg_24h", 0)}


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED DEEP SCAN (one fetch → both long + short evaluation)
# ══════════════════════════════════════════════════════════════════════════════

def deep_scan_both(coin_id, coin_info, btc_regime, btc_df, fng_value, long_config, short_config):
    gem_result, gem_reason   = deep_scan_coin(coin_id, coin_info, btc_regime, btc_df, fng_value, long_config)
    short_result = deep_scan_short(coin_id, coin_info, btc_regime, btc_df, fng_value, short_config)
    short_reason = None
    if isinstance(short_result, dict) and short_result.get("rejected"):
        short_reason = short_result.get("reason", "rejected")
        short_result = None
    return gem_result, gem_reason, short_result, short_reason


# ══════════════════════════════════════════════════════════════════════════════
# GATE SUMMARY DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def print_gate_summary(fail_reasons, gems_found, total_scanned):
    gates = [
        (1,  "Data length",    ["insufficient_data", "no_data"],        "insufficient_data"),
        (2,  "Pegged asset",   ["pegged_asset"],                         "pegged_asset"),
        (3,  "Volume",         ["volume_low"],                           "volume_low_Xx"),
        (4,  "Cloud position", ["below_cloud", "in_cloud"],              "below/in_cloud"),
        (5,  "ML accuracy",    ["ml_unreliable"],                        "ml_unreliable"),
        (6,  "ML prediction",  ["ml_bearish"],                           "ml_bearish_X%"),
        (7,  "OBV direction",  ["obv_distribution"],                     "obv_dist"),
        (8,  "Blocked signal", ["blocked"],                              "blocked_X"),
        (9,  "Long P gate",    ["prob_low"],                             "P < 65%"),      # NEW
        (10, "Backtest WR",    ["backtest_low"],                         "backtest_low"),
    ]
    gate_counts = []
    for gn, label, prefixes, fmt in gates:
        count = sum(v for k, v in fail_reasons.items() if any(k.startswith(p) for p in prefixes))
        gate_counts.append((gn, label, fmt, count))

    print(Fore.CYAN + "\n" + "═"*63)
    print(Fore.CYAN + "  FILTER FUNNEL — Probabilistic gate breakdown")
    print(Fore.CYAN + "═"*63)
    print(Fore.WHITE + f"  {'#':<4} {'Gate':<22} {'Reason':<24} {'Rej':>4}")
    print(Fore.WHITE + "─"*63)
    for gn, label, fmt, count in gate_counts:
        if count == 0: color = Fore.WHITE
        elif count <= 5: color = Fore.YELLOW
        else: color = Fore.RED
        bar = "█" * min(count // 2, 15)
        print(color + f"  {gn:<4} {label:<22} {fmt:<24} {count:>4}  {bar}")
    print(Fore.WHITE + "─"*63)
    gem_color = Fore.GREEN if gems_found > 0 else Fore.YELLOW
    print(gem_color + f"  {'':4} {'PASS ALL':<22} {'💎 GEM':<24} {gems_found:>4}")
    print(Fore.CYAN + "═"*63)
    print(Fore.WHITE + f"  Gate 9 ('P < 65%') replaces old score + confidence gates")
    print(Fore.WHITE + f"  Bayesian probability now the unified quality gate")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_rejection(reason: str, side: str = "L") -> str:
    """
    Convert internal rejection reason code to a readable string with probability.
      ml_bearish_19%         → ML=19% bearish
      prob_low_58%           → P=58% (needs 65%)
      below_cloud_15.5pct_to_top → below cloud  15.5% to top
      short_ml_low_2%        → short P=2%
      short_prob_low_41%     → short P=41% (needs 65%)
      adx_weak_9.1           → ADX=9.1 weak
      obv_distribution       → OBV falling
      ml_unreliable_48%      → ML acc=48% unreliable
      volume_low_0.38x       → vol=0.38x avg
    """
    r = str(reason)
    if r.startswith("prob_low_"):
        p = r.replace("prob_low_","").replace("%","")
        return f"P={p}% (needs {BUY_PROB_THRESHOLD:.0%})"
    if r.startswith("short_prob_low_"):
        p = r.replace("short_prob_low_","").replace("%","")
        return f"short P={p}% (needs {SHORT_PROB_THRESHOLD:.0%})"
    if r.startswith("ml_bearish_"):
        p = r.replace("ml_bearish_","").replace("%","")
        return f"ML={p}% (bearish)"
    if r.startswith("short_ml_low_"):
        p = r.replace("short_ml_low_","").replace("%","")
        return f"short P={p}%"
    if r.startswith("below_cloud_"):
        gap = r.replace("below_cloud_","").replace("pct_to_top","").replace("pct","").replace("_"," ")
        return f"below cloud  {gap.strip()}% to top"
    if r.startswith("in_cloud_"):
        gap = r.replace("in_cloud_","").replace("pct_to_top","").replace("pct","").replace("_"," ")
        return f"in cloud  {gap.strip()}% to top"
    if r.startswith("adx_weak_"):
        v = r.replace("adx_weak_","")
        return f"ADX={v} < 15"
    if r.startswith("volume_low_"):
        v = r.replace("volume_low_","")
        return f"vol={v} avg"
    if r.startswith("ml_unreliable_"):
        p = r.replace("ml_unreliable_","").replace("%","")
        return f"ML acc={p}% (neutralized — Bayesian only)"
    if r.startswith("short_ml_unreliable_"):
        p = r.replace("short_ml_unreliable_","").replace("%","")
        return f"short ML acc={p}% (< 40%, unusable)"
    if r == "obv_distribution":      return "OBV↓ distributing"
    if r == "obv_rising":            return "OBV↑ accumulating"
    if r == "no_data":               return "no data"
    if r.startswith("blocked_"):     return f"blocked ({r.replace('blocked_','')})"
    if r.startswith("backtest_low_"):return f"backtest {r.replace('backtest_low_','')}"
    if r.startswith("pegged_"):      return "pegged/stable asset"
    if r.startswith("error_"):       return f"error: {r.replace('error_','')}"
    if r.startswith("no_per_coin"):  return "🌐 using global model"  # not a skip anymore
    if r.startswith("no_model"):     return "⏳ no model — run main bot first"
    if "stale" in r:
        h = r.split("_")[-2] if "_" in r else "?"
        return f"⏳ model {h} old — run main bot to retrain"
    if "version_mismatch" in r:     return "⏳ model version outdated — run main bot"
    if "no_short_model" in r:       return "⏳ no short model"
    # fallback — clean underscores
    return r.replace("_", " ")



def train_all_models(coins_to_train=None, fng_value=50):
    """
    Standalone batch training function — run this SEPARATELY from the scanner.

    Best practice:
      cron: 0 2 * * * python gem_scanner.py --train     (2am daily)
      or:   python gem_scanner.py --train               (manual)

    Trains long + short ML models for all coins in the universe.
    Scanner then READS these cached models without retraining.

    This separates the expensive training step from the fast scanning step,
    preventing model instability and probability drift between scans.
    """
    import sys as _sys
    print(Fore.CYAN + "\n" + "="*60)
    print(Fore.CYAN + "  📦 BATCH MODEL TRAINING (scanner predict-only mode)")
    print(Fore.CYAN + f"  Feature version: {ML_FEATURE_VERSION}")
    print(Fore.CYAN + "="*60)

    btc_df = None
    try:
        btc_df = fetch_coin_data("bitcoin", days=365)
    except Exception:
        pass

    if coins_to_train is None:
        all_coins = fetch_market_overview(page=1, per_page=250)
        survivors, _ = prefilter_coins(all_coins, SCANNER_CONFIG)
        coins_to_train = [c["id"] for c in survivors[:200]]

    n = len(coins_to_train)
    trained = 0; skipped = 0; failed = 0
    needs_retrain = []

    for idx, coin_id in enumerate(coins_to_train, 1):
        print(f"  [{idx:3d}/{n}] {coin_id:<25}", end=" ", flush=True)
        try:
            # Check if retrain needed (uses main-bot MODEL_CACHE_DAYS, not scanner mode)
            if not should_retrain_model(coin_id, scanner_mode=False):
                print(f"✓ cached")
                skipped += 1
                continue
            needs_retrain.append(coin_id)
            df = fetch_coin_data(coin_id, days=365)
            if df.empty or len(df) < 80:
                print(f"✗ no_data"); failed += 1; continue

            ml_model, ml_scaler, ml_accuracy, ml_samples = train_ml_model(
                df, coin_id, btc_df=btc_df, fng_value=fng_value)
            if ml_model:
                save_model(coin_id, ml_model, ml_scaler, ml_accuracy, ml_samples)
                print(f"✅ long acc={ml_accuracy:.0%}", end=" ")
            else:
                print(f"✗ train_failed", end=" "); failed += 1

            short_model, short_scaler, short_acc, _ = train_short_ml_model(
                df, coin_id, btc_df=btc_df, fng_value=fng_value)
            if short_model:
                save_short_model(coin_id, short_model, short_scaler, short_acc, len(df))
                print(f"| short acc={short_acc:.0%}")
            else:
                print()
            trained += 1
            import time as _t; _t.sleep(1.5)

        except Exception as e:
            print(f"✗ error: {str(e)[:30]}"); failed += 1

    print(Fore.CYAN + "\n" + "="*60)
    print(Fore.GREEN + f"  Trained: {trained}  Skipped (cached): {skipped}  Failed: {failed}")
    print(Fore.WHITE + f"  Feature version: {ML_FEATURE_VERSION}")
    if needs_retrain:
        print(Fore.YELLOW + f"  Retrained {len(needs_retrain)} stale models")
    print(Fore.CYAN + "="*60)



def run_gem_scanner():
    start_time = time.time()
    config = SCANNER_CONFIG

    print(Fore.CYAN + "\n" + "═"*70)
    print(Fore.CYAN + "  💎 GEM SCANNER — Probabilistic Crypto Hunter")
    print(Fore.CYAN + f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(Fore.CYAN + "  Gate: Bayesian P ≥ 65% (replaces additive score + confidence)")
    print(Fore.CYAN + "═"*70)

    _ntfy_topic = _ntfy_os.environ.get("NTFY_TOPIC", "")
    if _ntfy_topic:
        print(Fore.GREEN + f"[NTFY] ✅ Notifications → ntfy.sh/{_ntfy_topic}")
    else:
        print(Fore.YELLOW + "[NTFY] ⚠️  NTFY_TOPIC not set — push notifications disabled")

    # Setup
    # Load global model once at startup
    _ensure_global_model()

    print(Fore.YELLOW + "\n[SETUP] Fetching BTC regime + data...")
    btc_regime = get_btc_regime()
    print(Fore.GREEN + f"[SETUP] BTC regime: {btc_regime.upper()}")
    if btc_regime == "bear":
        print(Fore.RED + "[SCANNER] ⚠️  BTC BEAR — long gems will be rare")

    btc_df = None
    try:
        btc_df = fetch_coin_data("bitcoin", days=365)
    except Exception:
        pass

    fng_value = 50
    try:
        fng_value, _ = fetch_fear_greed_index()
        print(Fore.GREEN + f"[SETUP] Fear & Greed: {fng_value}")
    except Exception:
        pass

    # Stage 1
    print(Fore.CYAN + f"\n[STAGE 1] Fetching top-{config['universe_size']} coins...")
    all_coins = []
    for page in range(1, config["universe_size"] // 250 + 2):
        batch = fetch_market_overview(page=page, per_page=min(250, config["universe_size"]))
        all_coins.extend(batch)
        if len(all_coins) >= config["universe_size"]: break
        time.sleep(1)
    all_coins = all_coins[:config["universe_size"]]

    survivors, rejected = prefilter_coins(all_coins, config)
    print(Fore.GREEN + f"[STAGE 1] {len(survivors)} survivors from {len(all_coins)} | Rejected: {rejected}")

    # Unified universe
    short_config    = SHORT_SCANNER_CONFIG
    short_universe  = prefilter_coins_short(all_coins, short_config)
    long_ids        = {c["id"] for c in survivors}
    short_ids       = {c["id"] for c in short_universe}
    extra_short     = [c for c in short_universe if c["id"] not in long_ids]
    unified         = survivors + extra_short

    print(Fore.CYAN + f"\n[STAGE 2] Unified scan: {len(survivors)} long + {len(extra_short)} short-only = {len(unified)} total")
    print(Fore.CYAN + f"  Long gate:  P ≥ {config['min_prob']:.0%}   (was: score ≥ 5 AND conf ≥ 68)")
    print(Fore.CYAN + f"  Short gate: P ≥ {short_config['min_short_prob']:.0%}  (was: sell ≤ -5 AND short_ml ≥ 40%)")
    print(Fore.CYAN + "═"*70)

    gems = []; shorts = []
    fail_reasons = {}; short_fail_counts = {}
    near_misses = []; short_near_misses = []

    # Pre-breakout watchlist: coins close to cloud with RS vs BTC
    _watchlist_candidates: list = []
    _btc_chg = btc_df["price"].iloc[-1] / btc_df["price"].iloc[-2] - 1 if btc_df is not None and len(btc_df) > 1 else 0.0
    _btc_chg_pct = _btc_chg * 100

    # Accumulation radar: quality coins far below cloud showing OBV divergence.
    # Different from swing trades — these are staged DCA candidates.
    # Position size: 20-30% of normal. Hold until cloud reclaim (weeks/months).
    _accum_candidates: list = []
    # Quality tier by market cap rank in the universe (approximated by order in unified list)
    _quality_coins = {
        "bitcoin","ethereum","binancecoin","ripple","solana","cardano","avalanche-2",
        "polkadot","chainlink","near","litecoin","filecoin","monero","hyperliquid",
        "hedera-hashgraph","bittensor","arbitrum","immutable-x","ondo-finance",
        "kucoin-shares","dydx","gnosis","ethena","story-2",
        # Privacy coins — strong niche demand, often lead bear market recoveries
        "zcash","zencash","dash",
        # DeFi / L2
        "raydium","celestia","algorand","origintrail",
    }

    for i, coin_info in enumerate(unified, 1):
        coin_id  = coin_info["id"]
        symbol   = coin_info["symbol"].upper()
        is_long  = coin_id in long_ids
        is_short = coin_id in short_ids
        _tag     = "/".join(["L"]*is_long + ["S"]*is_short)

        print(Fore.WHITE + f"  [{i:3d}/{len(unified)}] {symbol:<8} "
              f"${coin_info['price']:.4f} ({coin_info['chg_24h']:+.1f}%) [{_tag}]  ... ",
              end="", flush=True)

        gem_result, gem_reason, short_result, short_reason = deep_scan_both(
            coin_id, coin_info, btc_regime, btc_df, fng_value, config, short_config)

        # ── Long result ───────────────────────────────────────────────────────
        if gem_result and is_long:
            if gem_result.get("is_watch"):
                # WATCH tier (55-65%): collect for display, no notification
                _watchlist_candidates.append({
                    "symbol":    gem_result["symbol"],
                    "coin_id":   coin_id,
                    "price":     gem_result["price"],
                    "chg_24h":   float(gem_result.get("chg_24h", 0) or 0),
                    "rs_vs_btc": float(gem_result.get("chg_24h", 0) or 0) - _btc_chg_pct,
                    "gap_pct":   gem_result.get("cloud_gap", 0),
                    "is_below":  gem_result.get("cloud_below", False),
                    "reason":    f"P={gem_result['prob']:.0%} watch",
                    "strong_rs": float(gem_result.get("chg_24h", 0) or 0) - _btc_chg_pct >= 5.0,
                    "score":     gem_result["prob"] * 100,
                    "prob":      gem_result["prob"],
                    "obv":       gem_result.get("obv_trend", ""),
                    "cloud_note": gem_result.get("cloud_note", ""),
                })
                print(Fore.YELLOW + f"👁 WATCH P={gem_result['prob']:.0%} "
                      f"ML={gem_result['ml_proba']:.0%} "
                      f"OBV={gem_result['obv_trend']}", end="")
                send_ntfy(
                    title=f"👁 WATCH: {gem_result['symbol']} P={gem_result['prob']:.0%}",
                    message=(
                        f"Long P:     {gem_result['prob']:.0%}  (WATCH — below 65% threshold)\n"
                        f"EV:         {gem_result['ev']:+.2f} ({gem_result['ev_rating']})\n"
                        f"ML:         {gem_result['ml_proba']:.0%} (acc {gem_result['ml_accuracy']:.0%})\n"
                        f"Lifecycle:  {gem_result['lifecycle']}\n"
                        f"RS Tier:    {gem_result['rs_label']}\n"
                        f"BTC Regime: {gem_result['btc_regime'].upper()}\n"
                        f"Backtest:   {gem_result['backtest_wr']:.0f}% win rate\n"
                        f"OBV:        {gem_result['obv_trend']}\n"
                        f"RSI:        {gem_result['rsi']:.1f}\n"
                        f"ADX:        {gem_result['adx']:.1f}\n"
                        f"Vol Regime: {gem_result['vol_regime']}\n"
                        f"Entry Scr:  {gem_result['entry_score']}/100 ({gem_result['entry_label']})\n"
                        f"Trade Scr:  {gem_result['trade_score']}/100 ({gem_result['trade_label']})\n"
                        f"Kelly:      {gem_result['kelly']}\n"
                        f"Position:   {gem_result['pos_size']}\n"
                        f"Whale:      {gem_result['whale_str']}\n"
                        f"Cloud:      {gem_result.get('cloud_note','above') or 'above cloud'}\n"
                        f"Entry:      ${gem_result['entry']:.6f}\n"
                        f"Stop:       ${gem_result['stop']:.6f}\n"
                        f"TP:         ${gem_result['tp']:.6f}\n"
                        f"R:R:        {gem_result['rr']:.1f}:1\n"
                        f"CoinGecko:  {coin_id}"
                    ),
                    priority="low",
                    tags="eyes,chart_increasing",
                    topic_env="NTFY_WATCH_TOPIC"
                )
            else:
                # GEM tier (≥65%): add to gems and notify
                gems.append(gem_result)
                _cloud_watch = coin_info.get("cloud_watch", False)
                _cloud_below = coin_info.get("cloud_below", False)
                _cloud_gap   = coin_info.get("cloud_gap_pct", 0)
                _prob_tag    = ("⭐ STRONG"   if gem_result["is_strong"] else
                                "🔄 RECOVERY" if _cloud_below else "💎")
                _cloud_note  = (f" ☁️↑ {_cloud_gap:.1f}% to cloud — recovery" if _cloud_below
                                else f" ☁️ {_cloud_gap:.1f}% to cloud break" if _cloud_watch else "")
                print(Fore.GREEN + f"{_prob_tag} GEM! P={gem_result['prob']:.0%} "
                      f"ML={gem_result['ml_proba']:.0%} "
                      f"OBV={gem_result['obv_trend']}{_cloud_note}", end="")
                send_ntfy(
                    title=f"{'⭐' if gem_result['is_strong'] else ('🔄 RECOVERY' if _cloud_below else '💎')} GEM: {gem_result['symbol']} P={gem_result['prob']:.0%}",
                    message=(
                        f"Long P:     {gem_result['prob']:.0%}  {'(STRONG)' if gem_result['is_strong'] else ''}\n"
                        f"EV:         {gem_result['ev']:+.2f} ({gem_result['ev_rating']})\n"
                        f"ML:         {gem_result['ml_proba']:.0%} (acc {gem_result['ml_accuracy']:.0%})\n"
                        f"Lifecycle:  {gem_result['lifecycle']}\n"
                        f"RS Tier:    {gem_result['rs_label']}\n"
                        f"BTC Regime: {gem_result['btc_regime'].upper()}\n"
                        f"Backtest:   {gem_result['backtest_wr']:.0f}% win rate\n"
                        f"OBV:        {gem_result['obv_trend']}\n"
                        f"RSI:        {gem_result['rsi']:.1f}\n"
                        f"ADX:        {gem_result['adx']:.1f}\n"
                        f"Vol Regime: {gem_result['vol_regime']}\n"
                        f"Entry Scr:  {gem_result['entry_score']}/100 ({gem_result['entry_label']})\n"
                        f"Trade Scr:  {gem_result['trade_score']}/100 ({gem_result['trade_label']})\n"
                        f"Kelly:      {gem_result['kelly']}\n"
                        f"Position:   {gem_result['pos_size']}\n"
                        f"Whale:      {gem_result['whale_str']}\n"
                        f"Direction:  {gem_result['direction']} ({gem_result['dir_conf']:.0%})\n"
                        f"Entry:      ${gem_result['entry']:.6f}\n"
                        f"Stop:       ${gem_result['stop']:.6f}\n"
                        f"TP:         ${gem_result['tp']:.6f}\n"
                        f"R:R:        {gem_result['rr']:.1f}:1\n"
                        + (f"☁️ Cloud:    {_cloud_gap:.1f}% below cloud top — recovery\n" if _cloud_below else
                           f"☁️ Cloud:    {_cloud_gap:.1f}% to cloud break\n" if _cloud_watch else "")
                        + f"CoinGecko:  {coin_id}"
                    ),
                    priority="high" if gem_result["is_strong"] else "default",
                    tags="gem,chart_increasing", topic_env="NTFY_TOPIC"
                )
        elif is_long and gem_reason:
            _key = "_".join(gem_reason.split("_")[:2])
            fail_reasons[_key] = fail_reasons.get(_key, 0) + 1

            # ── PRE-BREAKOUT WATCHLIST COLLECTION ─────────────────────────────
            # Collect coins that failed the gem gate but show recovery setup:
            # close to cloud top + outperforming BTC + not distributing
            _coin_chg = float(coin_info.get("chg_24h", 0) or 0)
            _rs_vs_btc = _coin_chg - _btc_chg_pct   # positive = outperforming BTC

            # Check if this coin has cloud gap info (below or in cloud near top)
            _gap = coin_info.get("cloud_gap_pct", 0.0)
            _is_below = coin_info.get("cloud_below", False)

            # Also catch coins rejected by below_cloud/in_cloud gate
            if not _gap and gem_reason:
                import re as _re
                _m = _re.search(r'(?:below|in)_cloud_([\d.]+)pct', gem_reason)
                if _m:
                    _gap = float(_m.group(1))
                    _is_below = "below_cloud" in gem_reason

            if _gap > 0:
                # Standard watchlist: gap ≤ 12% + RS ≥ 1.5%
                # Strong RS exception: gap ≤ 25% + RS ≥ 5%
                # Rationale: ZEC +17% while BTC flat closes a 20% gap in 2-3 days
                _rs_threshold   = 1.5
                _gap_threshold  = 12.0
                if _rs_vs_btc >= 5.0:
                    _gap_threshold = 25.0   # exceptional RS → wider gap allowed
                    _rs_threshold  = 5.0

                if _rs_vs_btc >= _rs_threshold and _gap <= _gap_threshold:
                    _wl_entry = {
                        "symbol":    symbol,
                        "coin_id":   coin_id,
                        "price":     coin_info["price"],
                        "chg_24h":   _coin_chg,
                        "rs_vs_btc": _rs_vs_btc,
                        "gap_pct":   _gap,
                        "is_below":  _is_below,
                        "reason":    gem_reason,
                        "strong_rs": _rs_vs_btc >= 5.0,
                        # Score: RS strength × proximity to cloud
                        # Strong RS coins get 1.5× boost to rank ahead of barely-qualifying coins
                        "score":     _rs_vs_btc * (1.0 - _gap / 30.0) * (1.5 if _rs_vs_btc >= 5.0 else 1.0),
                    }
                    _watchlist_candidates.append(_wl_entry)

            # ── ACCUMULATION RADAR COLLECTION ─────────────────────────────────
            # Quality coins far below cloud but showing OBV divergence
            if (coin_id in _quality_coins and
                    "below_cloud" in gem_reason and
                    not coin_info.get("cloud_watch")):
                try:
                    import re as _re2
                    _gm = _re2.search(r'below_cloud_([\d.]+)pct', gem_reason)
                    _gap_top = float(_gm.group(1)) if _gm else 0.0

                    if 15.0 < _gap_top <= 65.0:  # far below but not hopeless
                        # Quick OBV check from prefilter / coin_info
                        # Full OBV requires df — fetch from cache if available
                        _df_cached = None
                        try:
                            from crypto_trading_bot import _session_df_cache as _sdc
                            _cached = _sdc.get(coin_id, {})
                            _df_cached = _cached.get("df") if _cached else None
                        except Exception:
                            pass

                        _obv_div = False   # OBV rising while price falling
                        _rsi_val = 50.0
                        _ath_drop = 0.0
                        _adx_val  = 0.0

                        if _df_cached is not None and len(_df_cached) > 30:
                            try:
                                _pr  = _df_cached["price"].iloc[-30:]
                                _obv = _df_cached["OBV"].iloc[-30:]
                                _pr_trend  = _pr.iloc[-1]  < _pr.iloc[0]    # price falling
                                _obv_trend2 = _obv.iloc[-1] >= _obv.iloc[0]  # OBV stable/rising
                                _obv_div   = _pr_trend and _obv_trend2
                                _rsi_val   = _sf(_df_cached["RSI"].iloc[-1], 50.0) if "RSI" in _df_cached.columns else 50.0
                                _adx_val   = _sf(_df_cached["ADX"].iloc[-1], 0.0) if "ADX" in _df_cached.columns else 0.0
                                _ath_val   = _sf(_df_cached["price"].max(), 0.0)
                                _ath_drop  = (_ath_val - _sf(_df_cached["price"].iloc[-1], 0.0)) / _ath_val * 100
                            except Exception:
                                pass

                        # Score: OBV divergence is the strongest signal
                        _accum_score = (
                            (3.0 if _obv_div else 0.0) +
                            (2.0 if _rsi_val < 35 else 1.0 if _rsi_val < 45 else 0.0) +
                            (1.0 if _ath_drop > 60 else 0.5 if _ath_drop > 40 else 0.0) +
                            (1.0 if _adx_val < 20 else 0.0)   # downtrend losing momentum
                        )

                        if _accum_score >= 1.0:
                            _accum_candidates.append({
                                "symbol":     symbol,
                                "coin_id":    coin_id,
                                "price":      coin_info["price"],
                                "chg_24h":    float(coin_info.get("chg_24h", 0) or 0),
                                "gap_top":    _gap_top,
                                "obv_div":    _obv_div,
                                "rsi":        _rsi_val,
                                "adx":        _adx_val,
                                "ath_drop":   _ath_drop,
                                "score":      _accum_score,
                            })
                except Exception:
                    pass

            # Near-miss: failed at prob_low gate but P was close
            if "prob_low" in gem_reason:
                try:
                    _p = float(gem_reason.replace("prob_low_","").replace("%","")) / 100
                    if _p >= config["min_prob"] - 0.10:
                        near_misses.append({"symbol": symbol, "reason": gem_reason,
                                            "price": coin_info["price"], "chg_24h": coin_info["chg_24h"]})
                        send_ntfy(
                            title=f"⏳ LONG NEAR-MISS: {symbol} P={_p:.0%}",
                            message=f"P={_p:.0%} (needs {config['min_prob']:.0%})\nPrice: ${coin_info['price']:.6f}",
                            priority="default", tags="eyes", topic_env="NTFY_WATCH_TOPIC"
                        )
                except Exception:
                    pass
            _needs_train = "no_model" in str(gem_reason) or "stale" in str(gem_reason) or "version_mismatch" in str(gem_reason)
            _color = Fore.CYAN if _needs_train else Fore.WHITE
            _tag   = "⏳ TRAIN" if _needs_train else "✗ L"
            print(_color + f"{_tag}: {_fmt_rejection(gem_reason)}", end="")
        elif is_long:
            fail_reasons["insufficient_data"] = fail_reasons.get("insufficient_data", 0) + 1
            print(Fore.WHITE + "✗ L: no data", end="")

        # ── Short result ──────────────────────────────────────────────────────
        if short_result and is_short:
            _sp_tag = "⭐ STRONG" if short_result.get("is_strong") else "🔴"
            print(Fore.RED + f"  {_sp_tag} SHORT P={short_result['short_prob']:.0%} "
                  f"sML={short_result['short_ml_proba']:.0%} {short_result['cloud_pos']}", end="")

            if short_result.get("is_strong"):
                # ── STRONG tier (P ≥ 75%): add to gems and notify ─────────────
                shorts.append(short_result)
                send_ntfy(
                    title=f"⭐ SHORT: {symbol} P={short_result['short_prob']:.0%}",
                    message=(
                        f"Short P:    {short_result['short_prob']:.0%}\n"
                        f"Short ML:   {short_result['short_ml_proba']:.0%} (acc {short_result['short_ml_accuracy']:.0%})\n"
                        f"ADX:        {short_result['adx']:.1f}\n"
                        f"Cloud:      {short_result['cloud_pos']}\n"
                        f"OBV:        {short_result.get('obv_trend','?')}\n"
                        f"Stop:       ${short_result['stop_loss']:.6f}\n"
                        f"Target:     ${short_result['profit_target']:.6f}\n"
                        f"R/R:        {short_result['rr_ratio']:.2f}:1\n"
                        f"CoinGecko:  {coin_id}"
                    ),
                    priority="urgent",
                    tags="chart_decreasing,warning", topic_env="NTFY_TOPIC"
                )
            else:
                # ── WATCH tier (P 70–75%): display only, no notification ───────
                # Same reasoning as long-side watch: directional read is right
                # but not enough conviction to act on. Check the next scan.
                short_near_misses.append({
                    "symbol":   symbol,
                    "reason":   f"short_watch_P={short_result['short_prob']:.0%}",
                    "price":    short_result["price"],
                    "chg_24h":  short_result.get("chg_24h", 0),
                    "short_prob": short_result["short_prob"],
                    "adx":      short_result["adx"],
                    "cloud_pos": short_result["cloud_pos"],
                })
        elif is_short and short_result and short_result.get("watch_only"):
            # High-P short (≥80%) that failed the signal quality gate —
            # too strong to discard entirely, goes to watch list
            short_near_misses.append({
                "symbol":     short_result.get("symbol", symbol),
                "reason":     f"short_watch_P={short_result['short_prob']:.0%}_gate:{short_result.get('watch_reason','')}",
                "price":      short_result.get("price", 0),
                "chg_24h":   short_result.get("chg_24h", 0),
                "short_prob": short_result["short_prob"],
                "adx":        short_result.get("adx", 0),
                "cloud_pos":  short_result.get("cloud_pos", ""),
            })
            print(Fore.YELLOW + f"  👁 S-WATCH P={short_result['short_prob']:.0%} "
                  f"({short_result.get('watch_reason','signal gate')})", end="")
        elif is_short and short_reason:
            _sk = "_".join(short_reason.split("_")[:2])
            short_fail_counts[_sk] = short_fail_counts.get(_sk, 0) + 1
            if "short_prob_low" in short_reason:
                try:
                    _sp = float(short_reason.replace("short_prob_low_","").replace("%","")) / 100
                    if _sp >= short_config["min_short_prob"] - 0.15:
                        short_near_misses.append({"symbol": symbol, "reason": short_reason,
                                                   "price": coin_info["price"], "chg_24h": coin_info["chg_24h"]})
                except Exception:
                    pass
            # Show the actual rejection reason, not just the probability threshold
            _short_p_str = ""
            if short_result and short_result.get("short_prob"):
                _short_p_str = f" P={short_result['short_prob']:.0%}"
            print(Fore.WHITE + f"  ✗ S{_short_p_str}: {_fmt_rejection(short_reason, 'S')}", end="")

        print()
        time.sleep(1.5)

    # ── WRITE DASHBOARD JSON ──────────────────────────────────────────────────
    # Persists scan results for the Streamlit dashboard to read.
    try:
        import json as _json, os as _os
        _os.makedirs("models", exist_ok=True)

        def _gem_row(g):
            price = g.get("price", 0) or 0
            stop  = g.get("stop",  0) or 0
            tp    = g.get("tp",    0) or 0
            return {
                "symbol":     g.get("symbol", ""),
                "price":      price,
                "chg_24h":   float(g.get("chg_24h", 0) or 0),
                "prob":       round(g.get("prob", 0), 4),
                "ml_proba":   round(g.get("ml_proba", 0), 4),
                "ml_accuracy":round(g.get("ml_accuracy", 0), 4),
                "is_strong":  bool(g.get("is_strong")),
                "adx":        round(float(g.get("adx", 0) or 0), 1),
                "obv_trend":  g.get("obv_trend", ""),
                "cloud_note": g.get("cloud_note", ""),
                "lifecycle":  g.get("lifecycle", ""),
                "rs_label":   g.get("rs_label", ""),
                "stop":       round(stop, 6),
                "tp":         round(tp, 6),
                "rr":         round(float(g.get("rr", 0) or 0), 2),
                "stop_pct":   round((stop - price) / price * 100, 2) if price else 0,
                "tgt_pct":    round((price - tp)   / price * 100, 2) if price else 0,
                "ev":         round(float(g.get("ev", 0) or 0), 3),
                "ev_rating":  g.get("ev_rating", ""),
                "liq_label":  g.get("liq_label", ""),
                "backtest_wr":round(float(g.get("backtest_wr", 0) or 0), 1),
            }

        def _short_row(s):
            price = s.get("price", 0) or 0
            sl    = s.get("stop_loss", 0) or 0
            tgt   = s.get("profit_target", 0) or 0
            return {
                "symbol":          s.get("symbol", ""),
                "price":           price,
                "chg_24h":        float(s.get("chg_24h", 0) or 0),
                "short_prob":      round(s.get("short_prob", 0), 4),
                "short_ml_proba":  round(s.get("short_ml_proba", 0), 4),
                "short_ml_accuracy":round(s.get("short_ml_accuracy", 0), 4),
                "is_strong":       bool(s.get("is_strong")),
                "adx":             round(float(s.get("adx", 0) or 0), 1),
                "obv_trend":       s.get("obv_trend", ""),
                "cloud_pos":       s.get("cloud_pos", ""),
                "stop_loss":       round(sl, 6),
                "profit_target":   round(tgt, 6),
                "rr_ratio":        round(float(s.get("rr_ratio", 0) or 0), 2),
                "stop_pct":        round((sl - price) / price * 100, 2) if price else 0,
                "tgt_pct":         round((price - tgt) / price * 100, 2) if price else 0,
                "liq_label":       s.get("liq_label", ""),
                "tradeable":       bool(s.get("tradeable")),
                "binance_id":      s.get("binance_id", ""),
            }

        _short_watch_list = [
            {
                "symbol":    w.get("symbol", ""),
                "short_prob":round(w.get("short_prob", 0), 4),
                "adx":       round(float(w.get("adx", 0) or 0), 1),
                "cloud_pos": w.get("cloud_pos", ""),
                "price":     w.get("price", 0),
                "chg_24h":  float(w.get("chg_24h", 0) or 0),
            }
            for w in short_near_misses if "short_watch" in w.get("reason", "")
        ]

        _long_watch_list = [
            {
                "symbol":    w.get("symbol", ""),
                "prob":      round(float(w.get("prob", 0) or 0), 4),
                "obv":       w.get("obv", ""),
                "cloud_note":w.get("cloud_note", ""),
                "price":     w.get("price", 0),
                "chg_24h":  float(w.get("chg_24h", 0) or 0),
                "rs_vs_btc": round(float(w.get("rs_vs_btc", 0) or 0), 2),
            }
            for w in _watchlist_candidates
        ]

        _scan_output = {
            "scan_time":      datetime.now().isoformat(),
            "btc_regime":     btc_regime,
            "fng_value":      fng_value,
            "fng_label":      ("Extreme Fear" if fng_value < 25 else "Fear" if fng_value < 45 else "Neutral" if fng_value < 55 else "Greed" if fng_value < 75 else "Extreme Greed") if fng_value else "N/A",
            "coins_scanned":  len(unified),
            "gems":           [_gem_row(g) for g in sorted(gems,   key=lambda x: x["prob"],       reverse=True)],
            "shorts":         [_short_row(s) for s in sorted(shorts, key=lambda x: x["short_prob"], reverse=True)],
            "watch":          sorted(_long_watch_list,  key=lambda x: x["prob"],       reverse=True),
            "short_watch":    sorted(_short_watch_list, key=lambda x: x["short_prob"], reverse=True),
        }
        with open("models/latest_scan.json", "w") as _f:
            _json.dump(_scan_output, _f, indent=2, default=str)
        print(Fore.CYAN + "  [DASHBOARD] ✅ models/latest_scan.json written")
    except Exception as _je:
        print(Fore.YELLOW + f"  [DASHBOARD] ⚠️  JSON write failed: {_je}")

    # ── RESULTS ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(Fore.CYAN + "\n" + "═"*70)
    print(Fore.GREEN + f"  💎 SCAN COMPLETE  |  {elapsed/60:.1f} min  |  {len(gems)} LONG GEMS  |  {len(shorts)} SHORT GEMS")
    print(Fore.CYAN + "═"*70)

    # ── LONG GEMS ─────────────────────────────────────────────────────────────
    gems_sorted = sorted(gems, key=lambda x: x["prob"], reverse=True)
    if gems_sorted:
        print(Fore.GREEN + f"\n{'Rank':<5} {'Symbol':<10} {'Price':>10} {'P':>6} {'ML%':>6} {'BT%':>5} {'Z':>5} {'Dev%':>6}  {'Direction'}")
        print("─"*75)
        for rank, gem in enumerate(gems_sorted, 1):
            prob_color = Fore.GREEN if gem["prob"] >= STRONG_BUY_THRESHOLD else Fore.YELLOW
            strong_tag = " ⭐" if gem["is_strong"] else ""
            print(prob_color +
                  f"{rank:<5} {gem['symbol']:<10} ${gem['price']:>9.4f} "
                  f"{gem['prob']:>5.0%}  {gem['ml_proba']:>5.0%}  {gem['backtest_wr']:>4.0f}%  "
                  f"{gem['z_score']:>4.1f}  {gem['ema26_dev']:>5.1f}%  "
                  f"{gem['direction']} ({gem['dir_conf']:.0%}){strong_tag}")

        print(Fore.CYAN + "\nDETAILED GEM REPORT:")
        for rank, gem in enumerate(gems_sorted, 1):
            print(Fore.GREEN + f"\n{'='*60}")
            print(Fore.GREEN + f"  #{rank}  {'⭐' if gem['is_strong'] else '💎'} {gem['name']} ({gem['symbol']})")
            print(Fore.GREEN + f"{'='*60}")
            print(f"  Bayesian P:     {gem['prob']:.0%}  {'← STRONG ⭐' if gem['is_strong'] else ''}")
            print(f"  ML Model:       {gem['ml_proba']:.0%} (accuracy: {gem['ml_accuracy']:.0%})")
            print(f"  Backtest WR:    {gem['backtest_wr']:.1f}%")
            print(f"  OBV Trend:      {gem['obv_trend']}")
            print(f"  Z-Score:        {gem['z_score']:.2f}")
            print(f"  EMA26 Dev:      {gem['ema26_dev']:.1f}%")
            print(f"  Direction:      {gem['direction']} ({gem['dir_conf']:.0%})")
            print(f"  Smart Money:    {gem['smart_money'].upper()}")
            print(f"  Whale:          {'Yes 🐋' if gem['is_whale'] else 'No'}")
            print(f"  ADX:            {gem['adx']:.1f}")
            print(f"  Price:          ${gem['price']:.6f}")
            print(f"  24h Change:     {gem['chg_24h']:+.2f}%")
            if gem.get("tradeability_report"):
                print(gem["tradeability_report"].display())
            print(Fore.YELLOW + f"  ⚠️  Run full bot analysis on {gem['id']} before trading")

        # Symbol table
        print(Fore.CYAN + "\n" + "─"*70)
        print(Fore.CYAN + "  📋 GEM SYMBOL REFERENCE")
        print(Fore.CYAN + "─"*70)
        for gem in gems_sorted:
            _bsym    = get_exchange_symbol(gem["id"])
            futures  = f"{gem['symbol']}USDT (PERP)" if gem.get("is_futures") else "Spot only"
            print(Fore.GREEN + f"  {gem['name']:<20} {gem['id']:<25} {_bsym or 'N/A':<16} {futures}")
        print()
        print(Fore.YELLOW + "  ➕ COINS = [")
        for gem in gems_sorted:
            print(Fore.WHITE + f'      "{gem["id"]}",')
        print(Fore.WHITE + "  ]")

    print_gate_summary(fail_reasons, len(gems_sorted), len(survivors))

    # ── WATCH LIST + PRE-BREAKOUT ────────────────────────────────────────────
    if _watchlist_candidates:
        _prob_watch = sorted([w for w in _watchlist_candidates if "watch" in w.get("reason","")],
                             key=lambda x: -x.get("prob", 0))[:10]
        _rs_watch   = sorted([w for w in _watchlist_candidates if "watch" not in w.get("reason","")],
                             key=lambda x: -x.get("score", 0))[:10]

        if _prob_watch:
            print(Fore.YELLOW + f"\n{'═'*72}")
            print(Fore.YELLOW + f"  👁  WATCH — Bayesian P=55-65%  (real signal, below BUY threshold)")
            print(Fore.YELLOW + f"{'═'*72}")
            print(Fore.WHITE + f"  {'Coin':<10} {'Price':>10}  {'P':>5}  {'OBV':>8}  {'Cloud':>12}  Today")
            print("  " + "─"*65)
            for _w in _prob_watch:
                _p    = _w.get("prob", 0)
                _obv  = _w.get("obv","")
                _cn   = _w.get("cloud_note","above cloud") or "above cloud"
                _chg  = float(_w.get("chg_24h",0) or 0)
                _pc   = Fore.YELLOW if _p >= 0.60 else Fore.WHITE
                _oc   = Fore.GREEN if _obv=="rising" else Fore.RED if _obv=="falling" else Fore.WHITE
                print(Fore.WHITE + f"  {_w['symbol']:<10} ${_w['price']:>9.4f}  "
                      + _pc + f"{_p:>5.0%}  "
                      + _oc + f"{_obv:>8}  "
                      + Fore.WHITE + f"{_cn:>12}  {_chg:>+6.1f}%")
            print(Fore.RESET)

        if _rs_watch and btc_regime in ("bear", "range"):
            print(Fore.CYAN + f"\n{'═'*72}")
            print(Fore.CYAN + f"  📋 PRE-BREAKOUT — RS outperforming BTC, close to cloud")
            print(Fore.CYAN + f"  BTC today: {_btc_chg_pct:+.1f}%")
            print(Fore.CYAN + f"{'═'*72}")
            print(Fore.WHITE + f"  {'Coin':<10} {'Price':>10}  {'Today':>7}  {'vs BTC':>7}  {'Gap':>6}  Status")
            print("  " + "─"*65)
            for _w in _rs_watch:
                _rc  = Fore.GREEN if _w.get("rs_vs_btc",0)>=5 else Fore.YELLOW
                _st  = "🔄 below" if _w.get("is_below") else "👁 in cloud"
                _tag = "⚡" if _w.get("strong_rs") else ""
                print(Fore.WHITE + f"  {_w['symbol']:<10} ${_w['price']:>9.4f}  "
                      f"{float(_w.get('chg_24h',0) or 0):>+7.1f}%  "
                      + _rc + f"{_w.get('rs_vs_btc',0):>+7.1f}%  "
                      + Fore.WHITE + f"{_w.get('gap_pct',0):>4.1f}%  {_st} {_tag}")
            print(Fore.RESET)

    # ── ACCUMULATION RADAR ────────────────────────────────────────────────────
    # Quality coins far below cloud showing OBV divergence / oversold signals
    # NOT swing trades — staged DCA candidates for patient buyers
    if _accum_candidates and btc_regime in ("bear", "range"):
        _accum_sorted = sorted(_accum_candidates, key=lambda x: -x["score"])[:8]
        print(Fore.MAGENTA + f"\n{'═'*72}")
        print(Fore.MAGENTA + f"  🎯 ACCUMULATION RADAR — Quality Coins at Bear Market Discount")
        print(Fore.MAGENTA + f"  ⚠️  NOT swing trades. Staged DCA only. 20-30% normal size.")
        print(Fore.MAGENTA + f"  ✅ Exit signal: coin reclaims Ichimoku cloud (weeks/months)")
        print(Fore.MAGENTA + f"{'═'*72}")
        print(Fore.WHITE + f"  {'Coin':<10} {'Price':>10}  {'vs ATH':>7}  "
              f"{'Cloud Gap':>9}  {'RSI':>5}  {'OBV Div':>7}  {'ADX':>5}  Score")
        print("  " + "─"*72)
        for _a in _accum_sorted:
            _obv_tag  = Fore.GREEN + "✅ YES" if _a["obv_div"] else Fore.RED + "❌ NO "
            _rsi_col  = Fore.GREEN if _a["rsi"] < 35 else Fore.YELLOW if _a["rsi"] < 45 else Fore.WHITE
            _stars    = "★" * int(_a["score"]) + "☆" * max(0, 5 - int(_a["score"]))
            print(Fore.WHITE + f"  {_a['symbol']:<10} "
                  f"${_a['price']:>9.4f}  "
                  f"{-_a['ath_drop']:>+7.1f}%  "
                  f"{_a['gap_top']:>7.1f}%☁   "
                  + _rsi_col + f"{_a['rsi']:>5.1f}  "
                  + _obv_tag
                  + Fore.WHITE + f"  {_a['adx']:>5.1f}  {_stars} {_a['score']:.1f}")
        print(Fore.MAGENTA +
              f"\n  Strategy: Buy 10% now → 10% if -10% → 10% if -20% → hold to cloud"
              f"\n  Stop: close below {chr(8805)} 30% recent low (structural break)")
        print(Fore.RESET)

    # ── SHORT RESULTS ──────────────────────────────────────────────────────────
    if shorts:
        shorts_sorted = sorted(shorts, key=lambda x: -x["short_prob"])
        print(Fore.RED + "\n" + "═"*90)
        print(Fore.RED + f"  🔴 {len(shorts)} SHORT GEM(S) FOUND")
        print(Fore.RED + "═"*90)
        print(Fore.WHITE + f"  {'Symbol':<10} {'P':>6} {'sML(acc)':>12} {'ADX':>5} {'OBV':>7} {'Cloud':>9} {'Stop%':>7} {'Tgt%':>7} {'R/R':>5}  Liq")
        print("─"*90)
        for s in shorts_sorted:
            _rsk  = (s["stop_loss"]    - s["price"]) / s["price"] * 100 if s["price"] else 0
            _rwrd = (s["price"] - s["profit_target"]) / s["price"] * 100 if s["price"] else 0
            cloud_c = Fore.RED if s["cloud_pos"]=="below" else (Fore.YELLOW if s["cloud_pos"]=="inside" else Fore.WHITE)
            obv_c   = Fore.RED if s.get("obv_trend")=="falling" else (Fore.YELLOW if s.get("obv_trend")=="flat" else Fore.WHITE)
            strong_tag = " ⭐" if s.get("is_strong") else ""
            liq_src = "~" if s.get("liq_source","order_book") != "order_book" else ""
            print(Fore.RED +
                  f"  {s['symbol']:<10} {s['short_prob']:>5.0%}  "
                  f"{s['short_ml_proba']:.0%}({s['short_ml_accuracy']:.0%}):>11  "
                  f"{s['adx']:>4.1f}  " +
                  obv_c + f"{s.get('obv_trend','?'):>7}" +
                  cloud_c + f"  {s['cloud_pos']:>7}" +
                  Fore.WHITE + f"  {_rsk:>+6.1f}%  {-_rwrd:>+6.1f}%  {s['rr_ratio']:>4.1f}  "
                  f"{s.get('liq_label','')}{liq_src}{strong_tag}")
            print(Fore.CYAN + f"    └─ Stop: ${s['stop_loss']:.6f} | Target: ${s['profit_target']:.6f} | R/R: {s['rr_ratio']:.2f}:1")
        print(Fore.RED + "═"*90)

    # ── NEAR MISSES ────────────────────────────────────────────────────────────
    if near_misses:
        print(Fore.CYAN + "\n" + "─"*70)
        print(Fore.CYAN + "  🔍 LONG NEAR-MISSES (P within 10% of threshold)")
        print(Fore.CYAN + "─"*70)
        for nm in near_misses:
            print(Fore.YELLOW + f"  {nm['symbol']:<10} ${nm['price']:.4f} ({nm['chg_24h']:+.1f}%)  {nm['reason']}")

    if short_near_misses:
        print(Fore.RED + "\n" + "─"*70)
        print(Fore.RED + "  🔍 SHORT NEAR-MISSES (short_prob within 15% of threshold)")
        print(Fore.RED + "─"*70)
        for nm in short_near_misses:
            print(Fore.YELLOW + f"  {nm['symbol']:<10} ${nm['price']:.4f} ({nm['chg_24h']:+.1f}%)  {nm['reason']}")

    # Show coins that need training (run main bot first)
    _needs_training = [coin_info["symbol"] for coin_info in unified
                       if any(s in str(fail_reasons) for s in ["no_model","stale","version"])]

    if not gems and not shorts:
        print(Fore.YELLOW + "\nNo gems found. Market conditions too weak.")

    # Coins skipped due to missing/stale models
    _skipped = [r for r in fail_reasons if "no_model" in r or "stale" in r or "version" in r]
    if _skipped:
        n_skipped = sum(fail_reasons.get(k,0) for k in _skipped)
        print(Fore.CYAN + f"\n⏳ {n_skipped} coins skipped — no fresh ML model.")
        print(Fore.CYAN + "   Run main bot (python crypto_trading_bot.py) to train models first.")
        print(Fore.CYAN + "   Scanner uses PREDICT-ONLY mode — models trained by main bot only.")

    print(Fore.WHITE + f"\n  Scan time: {elapsed/60:.1f} min  |  "
          f"{len(unified)} coins scanned  |  BTC: {btc_regime.upper()}")


if __name__ == "__main__":
    import sys as _sys
    if "--train" in _sys.argv:
        print(Fore.CYAN + "\n📦 BATCH TRAINING MODE — building model cache for scanner")
        train_all_models()
        _sys.exit(0)
    print(Fore.CYAN + "\n💎 CRYPTO GEM SCANNER — Probabilistic Mode")
    print(Fore.CYAN + "─"*50)
    print(f"Universe:    Top-{SCANNER_CONFIG['universe_size']} by market cap")
    print(f"Long gate:   P ≥ {SCANNER_CONFIG['min_prob']:.0%}  (Bayesian probability)")
    print(f"Short gate:  P ≥ {SHORT_SCANNER_CONFIG['min_short_prob']:.0%}  (short scorer)")
    print(f"ML gate:     acc ≥ {SCANNER_CONFIG['min_ml_accuracy']:.0%}")
    print(f"Backtest:    WR ≥ {SCANNER_CONFIG['min_backtest_winrate']:.0f}%")
    print()
    run_gem_scanner()
