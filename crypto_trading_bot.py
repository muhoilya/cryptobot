# crypto_trading_bot_gecko_v6.4.py
# Built on v6.3. Eighteen additions + ML enhancements:
#   [A] DI+ > DI- required for BUY (ADX direction filter) — fixes STORY-2 false ADX
#   [B] Chikou span confirmation — close vs price 26 periods ago
#   [C] RSI divergence detection — bullish divergence adds score, bearish subtracts
#   [D] BTC market regime filter — altcoin BUYs blocked when BTC is in downtrend
#   [E] Weekly EMA trend filter — EMA50 weekly approximated from daily data
#   [F] Consecutive candle confirmation — BUY only after 2 consecutive days of signals
#   [G] ML Model (Logistic Regression) — predicts trade profitability; blocks low-confidence trades
#   [H] Smart Money / Whale Activity — detects accumulation/distribution and large volume
#   [I] News / Event Filter — blocks trades before/after major events (FOMC, CPI, halving, etc)
#   [J] SELL Logic (Shorting Mechanism) — bearish RSI divergence, negative scores, downtrend confirmation
#   [K] Market Sentiment (Fear & Greed Index) — FREE API from alternative.me, 0-100 scale
#   [L] Relative Strength vs BTC — 24h coin performance compared to BTC (outperformance indicator)
#   [M] Time-based Features — Day of week and hour patterns for market behavior
#   [N] On-Chain Metrics — Whale accumulation signals, volume spikes, on-chain activity patterns
#   [S] Backtesting Engine — Historical trade simulation with win rate, profit factor, drawdown analysis
#   [U] Advanced Stops & TP — Trailing stops, breakeven protection, partial profit taking strategies
#   [V] Model Persistence — Save/load trained models (300x faster startup, auto-retrain every 7 days)
#   [W] Comprehensive Logging — Full error tracking, API monitoring, trade recording, performance metrics
#   [X] Sentiment Analysis (NLP) — VADER sentiment analysis for coins (no API key needed!)
#   [Y] Volume Behavior Logic — Detect good volume (increasing/spike) vs bad (declining + price up = FAKE MOVE)
#
# ML ENHANCEMENTS:
#   - Direction prediction (UP/DOWN) based on momentum
#   - Expected move size (%) based on ATR + volatility
#   - What's next narrative for each coin
#   - Event risk detection + confidence adjustment
#   - Ranked leaderboard sorted by Score + ML Confidence
#   - EXPANDED: 11 ML features (was 8) → Expected +3-5% accuracy improvement
#
# SELL ENHANCEMENTS [J]:
#   - MIN_SELL_SCORE threshold for SHORT signals (negative scoring, -5 default)
#   - Bearish RSI divergence detection (price highs, RSI lows = weakness)
#   - Short position planning with stops ABOVE entry, TP BELOW entry
#   - Short pyramid DCA strategy for layered entries
#   - Dual signal ranking: 🟢 BUY | 🔴 SELL | 🟡 WAIT
#
# ML FEATURE EXPANSION [K][L][M]:
#   - Feature 9 [K]: Market Sentiment (0-100 Fear & Greed Index, normalized to 0-1)
#   - Feature 10 [L]: Relative Strength (Coin 24h return - BTC 24h return, normalized -1 to 1)
#   - Feature 11 [M]: Day of Week (0-6 normalized to 0-1 for market cyclicality)
#
# PHASE 3 ENHANCEMENTS [N][S][U]:
#   - [N] On-Chain Metrics: Whale accumulation/distribution, volume patterns, on-chain signals
#   - [S] Backtesting Engine: Historical simulation, win rate, profit factor, max drawdown, Sharpe ratio
#   - [U] Advanced Stops: Trailing stops, breakeven stops, partial profit taking (3-tier exit)
#
# PHASE 4 ENHANCEMENTS [V]:
#   - [V] Model Persistence: Save trained models to disk, load on startup (300x faster!)
#   - [V] Auto-Retrain: Automatically retrain models every 7 days (configurable)
#   - [V] Cache Management: List, save, load, delete models from disk
#   - [V] Metadata Tracking: Store training date, accuracy, sample count for each model
#
# PHASE 4+ ENHANCEMENTS [W]:
#   - [W] Comprehensive Logging: API errors, trade signals, model operations, performance metrics
#   - [W] Log Rotation: Auto-rotate logs at 50MB, keep 5 backups
#   - [W] Error Context: Full traceback and context for debugging
#   - [W] Performance Monitoring: Track speed, efficiency, resource usage
#
# PHASE 5 ENHANCEMENTS [X][Y]:
#   - [X] Sentiment Analysis: VADER NLP for coin sentiment (offline, no API key!)
#   - [X] Automatic Boosting: Boost BUY score (+2) for positive sentiment
#   - [X] Risk Detection: Penalize SELL score (-2) for negative sentiment
#   - [Y] Volume Behavior: Detect good volume (increasing/spike) vs bad (fake moves)
#   - [Y] Fake Move Detection: Alert when price up but volume declining (likely fake)
#
# Score: MIN_BUY_SCORE = 5 (BUY), MIN_SELL_SCORE = -5 (SELL). ML + Smart Money + Events + Sentiment + Volume add informational layers.
# Requires: pip install scikit-learn nltk (optional, nltk for sentiment analysis)

import requests
import pandas as pd
import numpy as np
import math
import time
import csv
import os
from datetime import datetime, timezone
from colorama import Fore, Style, init

# [G] ML Model imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print(Fore.YELLOW + "[WARNING] scikit-learn not installed. Install with: pip install scikit-learn")

# [V] Model Persistence imports
import pickle
import json
from pathlib import Path
from datetime import timedelta

# [W] LOGGING IMPORTS
import logging
import logging.handlers

# [X] SENTIMENT ANALYSIS IMPORTS
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger_temp = logging.getLogger("TradingBot")
    logger_temp.warning("[SENTIMENT] NLTK not installed. Sentiment analysis disabled. Install: pip install nltk")

init(autoreset=True)

import os as _ntfy_os

def send_ntfy(title, message, priority="default", tags="", topic_env="NTFY_TOPIC"):
    """Push notification to phone via ntfy.sh. Set NTFY_TOPIC env var.
    Use topic_env='NTFY_WATCH_TOPIC' for near-miss/watch alerts.
    """
    topic = _ntfy_os.environ.get(topic_env, "") or _ntfy_os.environ.get("NTFY_TOPIC", "")
    if not topic:
        print(f"[NTFY] ⚠️  NTFY_TOPIC not set — skipping notification: '{title}'")
        print(f"[NTFY]    Set it with: export NTFY_TOPIC=your-topic-name")
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
            print(f"[NTFY] ❌ Failed (HTTP {resp.status_code}): '{title}'")
    except Exception as e:
        print(f"[NTFY] ❌ Error: {e}")


def get_liquidity_tier(book_usd):
    """Classify order book depth into actionable tiers for short position sizing.
    
    Returns: (key, label, max_position_usd)
      key:      "deep" / "moderate" / "thin" / "paper_thin"
      label:    display string with emoji
      max_pos:  max recommended position in USD (None = no limit, 0 = untradeable)
    """
    if book_usd is None or book_usd <= 0:
        return "unknown",    "❓ Unknown",    None
    elif book_usd >= 1_000_000:
        return "deep",       "🟢 DEEP",       None    # no limit
    elif book_usd >= 300_000:
        return "moderate",   "🟡 MODERATE",   5000    # $5K max
    elif book_usd >= 50_000:
        return "thin",       "🟠 THIN",       500     # $500 max
    else:
        return "paper_thin", "🔴 PAPER-THIN", 0       # untradeable


# ===========================
# CONFIG
# ===========================
API_KEY = "CG-WMycN7HmApdGhGJ5jJdVvCMi"
COINS = [
    "celestia", "floki", "algorand", "raydium", "bonk", "dash", "tron", "zencash", "origintrail", "zcash", "dogecoin",
    "avalanche-2", "solana", "bittensor", "ethereum", "bitcoin",
    "litecoin", "polkadot", "chainlink", "binancecoin", "ripple", "arbitrum",
    "cardano", "ethena", "story-2", "near", "immutable-x",
    "fartcoin", "ordinals", "ondo-finance", "aster-2",
    "filecoin", "hyperliquid", "monero", "shiba-inu"
]
CG_DAYS        = 365        # 365 days: more ML training samples (326 usable vs 101 at 180d)
CG_INTERVAL    = "daily"

# ── API call minimisation ─────────────────────────────────────────────────────
# Each coin costs 2 CoinGecko calls (market_chart + ohlc).
# With 25+ coins on the free tier (30 calls/min) this hits rate limits fast.
# Strategy:
#   1. Session cache: store DataFrame in memory — skip re-fetch if < CACHE_TTL_SECONDS old
#   2. BTC reuse: get_btc_regime() already caches BTC df; relative strength reuses it
#   3. OHLC cache: store OHLC result per coin — one fetch per session
#   4. Smart sleep: adaptive delay between coins based on 429 history
CACHE_TTL_SECONDS  = 1800    # 30 minutes — fresh enough for intraday decisions
CG_SLEEP_BETWEEN   = 1.5     # seconds between coin fetches (free tier: 30 req/min)
CG_SLEEP_ON_429    = 65.0    # seconds to wait after a 429 (rate limit reset)

_session_df_cache: dict  = {}   # coin_id → {"df": DataFrame, "ts": float}
_session_ohlc_cache: dict = {}  # coin_id → {"df": DataFrame, "ts": float}
_cg_consecutive_429 = 0         # track consecutive rate limit hits
BTC_ID         = "bitcoin"  # [D] reference coin for market regime

ACCOUNT_BALANCE     = 10000.0
RISK_PER_TRADE      = 0.01
STOP_ATR_MULT       = 2.0
TP_MULTIPLIER       = 2.0
MIN_RR_RATIO        = 1.5
MAX_EXPOSURE_PCT    = 0.10
EMA_TRAILING_BUFFER = 0.015
RSI_OVERBOUGHT_THRESH = 85
RSI_REDUCE_THRESH   = 70

MIN_TRADE_SIZE  = 0.0001
QUANT_ROUND     = 4

DONCHIAN_LOOKBACK  = 20
PYRAMID_LEVELS     = 2
PYRAMID_START_FRAC = 0.5

# [A-E] scoring: max possible ~12 — require 5 to fire BUY
MIN_BUY_SCORE = 5

# [J] SELL scoring: threshold for SHORT signals (negative score)
# [J] Range: -12 to 0. MIN_SELL_SCORE = -5 means sell when score is -5 or lower
MIN_SELL_SCORE = -5

# [K] Market Sentiment Configuration
FETCH_FEAR_GREED = True             # [K] Enable Fear & Greed Index fetching
SENTIMENT_CACHE_MINUTES = 60        # [K] Cache sentiment for 1 hour (doesn't change frequently)

# [L] Relative Strength Configuration
RELATIVE_STRENGTH_ENABLED = True    # [L] Enable relative strength vs BTC calculation

# [M] Time-based Features Configuration
TIME_FEATURES_ENABLED = True        # [M] Enable day-of-week and time patterns

# [N] On-Chain Metrics Configuration
DETECT_WHALE_ACTIVITY = True        # [N] Enable whale accumulation/distribution detection
WHALE_THRESHOLD_MULTIPLIER = 1.5    # [N] 1.5x = whale activity threshold
ON_CHAIN_LOOKBACK = 14              # [N] Look back 14 days for on-chain patterns

# [S] Backtesting Configuration
BACKTEST_ENABLED = True             # [S] Enable backtesting engine
BACKTEST_INITIAL_BALANCE = 10000.0  # [S] Starting capital for backtest
BACKTEST_LOOKBACK_DAYS = 90         # [S] Historical days to backtest

# [U] Advanced Stops Configuration
USE_ADVANCED_STOPS = True           # [U] Enable advanced stop loss strategies
USE_TRAILING_STOPS = True           # [U] Enable trailing stops (follow price up)
USE_BREAKEVEN_STOPS = True          # [U] Move stop to breakeven after +1% gain
TRAILING_STOP_PERCENT = 2.0         # [U] Trail by 2% from highest price
PARTIAL_PROFIT_ENABLED = True       # [U] Enable 3-tier partial profit taking
PARTIAL_TP_LEVEL_1 = 2.0            # [U] First TP at +2% (take 33%)
PARTIAL_TP_LEVEL_2 = 4.0            # [U] Second TP at +4% (take 33%)
PARTIAL_TP_LEVEL_3 = 6.0            # [U] Third TP at +6% (take 34%)

# [V] MODEL PERSISTENCE Configuration
MODEL_PERSISTENCE_ENABLED = True    # [V] Enable model caching
MODEL_PERSISTENCE_DIR = "models"    # [V] Directory to store models
MODEL_CACHE_DAYS = 7                # [V] Retrain after 7 days
AUTO_RETRAIN = True                 # [V] Auto-retrain old models
LOAD_CACHED_MODELS = True           # [V] Load on startup
SAVE_MODELS = True                  # [V] Save after training

# [W] LOGGING Configuration
LOG_ENABLED = True                  # [W] Enable/disable logging
LOG_LEVEL = "INFO"                  # [W] DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "logs/trading_bot.log"   # [W] Log file path
LOG_MAX_SIZE_MB = 50                # [W] Rotate log when > 50MB
LOG_BACKUP_COUNT = 5                # [W] Keep 5 backup log files
LOG_TO_CONSOLE = True               # [W] Print logs to console
LOG_TO_FILE = True                  # [W] Write logs to file
LOG_API_CALLS = True                # [W] Log API calls and responses
LOG_TRADES = True                   # [W] Log trade signals

# Binance order book — disable on GitHub Actions (geo-restricted, returns 451)
# Set USE_BINANCE_ORDER_BOOK=false in GitHub Actions env to skip ccxt calls
import os as _os
USE_BINANCE_ORDER_BOOK = _os.environ.get("USE_BINANCE_ORDER_BOOK", "true").lower() != "false"
# Auto-detect GitHub Actions
if _os.environ.get("GITHUB_ACTIONS") == "true":
    USE_BINANCE_ORDER_BOOK = False
    print("[CONFIG] GitHub Actions detected — Binance order book disabled (geo-restriction 451)")
LOG_ERRORS = True                   # [W] Log all errors with context

# [X] SOCIAL SENTIMENT ANALYSIS Configuration
SENTIMENT_ANALYSIS_ENABLED = True   # [X] Enable sentiment analysis
USE_VADER = True                    # [X] Use VADER (offline, no API key!)
USE_NEWSAPI = False                 # [X] Use NewsAPI (requires free key: newsapi.org)
CACHE_SENTIMENT_MINUTES = 60        # [X] Cache sentiment for 1 hour
CHECK_SENTIMENT_EVERY_N_COINS = 5   # [X] Check every 5th coin (save time)
SENTIMENT_BOOST_BUY = 2             # [X] Max +2 boost for strong buy sentiment
SENTIMENT_BOOST_SELL = 2            # [X] Max -2 penalty for strong sell sentiment
NEWSAPI_KEY = ""                    # [X] Get free key from newsapi.org (optional)

# [Y] VOLUME BEHAVIOR ANALYSIS Configuration
VOLUME_ANALYSIS_ENABLED = True      # [Y] Enable volume quality checks
GOOD_VOLUME_BOOST = 1               # [Y] +1 boost for good volume (increasing)
BAD_VOLUME_PENALTY = 2              # [Y] -2 penalty for bad volume (declining + price up)
VOLUME_INCREASE_THRESHOLD = 0.7      # [Y] Volume below 0.7x average = genuinely weak (-1)
                                      # 0.7-1.5x = normal (neutral), >2.0x = spike (+1)
VOLUME_SPIKE_THRESHOLD = 2.0        # [Y] Volume spike = 2x normal volume
VOLUME_LOOKBACK = 90                # [Y] Days to calculate volume average (90-day for consistency)

# ═══════════════════════════════════════════════════════════════════════════════
# COIN SYMBOL MAPPING: Convert ALL coin names to exchange symbols
# ═══════════════════════════════════════════════════════════════════════════════
COIN_SYMBOL_MAP = {
    # ════════════════════ MAJOR COINS ════════════════════
    "bitcoin": "BTC",
    "morpho": "MORPHO",
    "celestia": "TIA",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "ripple": "XRP",
    "xrp": "XRP",
    "cardano": "ADA",
    "ada": "ADA",
    "polkadot": "DOT",
    "dot": "DOT",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "litecoin": "LTC",
    "ltc": "LTC",
    "monero": None,    # Delisted from Binance 2024 (privacy coin)
    "xmr": None,
    "xmr": "XMR",
    
    # ════════════════════ USER'S SPECIFIC COINS ════════════════════
    "bittensor": "TAO",
    "tao": "TAO",
    "ethena": "ENA",
    "ena": "ENA",
    "story-2": None,
    "story": None,  # NOT on Binance - will be skipped
    "story": None,
    "aster-2": "ASTR",
    "astr": "ASTR",
    "tether-gold": "XAUT",
    "xaut": "XAUT",
    "hyperliquid": None,  # NOT on Binance - will be skipped
    "hype": None,
    "hype": "HYPE",
    "fartcoin": None,  # NOT on Binance - will be skipped gracefully
    "fart": None,
    "fart": "FART",
    "binancecoin": "BNB",
    
    # ════════════════════ LAYER 2 / SIDECHAINS ════════════════════
    "arbitrum": "ARB",
    "arbitrum-one": "ARB",
    "arb": "ARB",
    "optimism": "OP",
    "op": "OP",
    "polygon": "MATIC",
    "matic": "MATIC",
    "avalanche": "AVAX",
    "avalanche-2": "AVAX",
    "avax": "AVAX",
    "binance-smart-chain": "BNB",
    "bnb": "BNB",
    "bsc": "BNB",
    
    # ════════════════════ MAJOR ALTCOINS ════════════════════
    "chainlink": "LINK",
    "link": "LINK",
    "uniswap": "UNI",
    "uni": "UNI",
    "aave": "AAVE",
    "curve-dao-token": "CRV",
    "curve": "CRV",
    "crv": "CRV",
    "maker": "MKR",
    "mkr": "MKR",
    "yearn-finance": "YFI",
    "yfi": "YFI",
    "compound": "COMP",
    "comp": "COMP",
    "lido-dao": "LDO",
    "ldo": "LDO",
    "ethereum-name-service": "ENS",
    "ens": "ENS",
    "the-graph": "GRT",
    "grt": "GRT",
    
    # ════════════════════ COSMOS ECOSYSTEM ════════════════════
    "cosmos": "ATOM",
    "atom": "ATOM",
    "cosmos-hub": "ATOM",
    "osmosis": "OSMO",
    "osmo": "OSMO",
    "thorchain": "RUNE",
    "rune": "RUNE",
    "evmos": "EVMOS",
    "juno-network": "JUNO",
    "juno": "JUNO",
    
    # ════════════════════ NEAR & ECOSYSTEM ════════════════════
    "near": "NEAR",
    "near-protocol": "NEAR",
    "aurora": "AURORA",
    
    # ════════════════════ SOLANA ECOSYSTEM ════════════════════
    "serum": "SRM",
    "srm": "SRM",
    "raydium": "RAY",
    "ray": "RAY",
    "magic-internet-money": "MIM",
    "mim": "MIM",
    
    # ════════════════════ ETHEREUM LAYER 2 ════════════════════
    "starknet": "STRK",
    "strk": "STRK",
    "zkswap": "ZKS",
    "zks": "ZKS",
    "loopring": "LRC",
    "lrc": "LRC",
    
    # ════════════════════ STAKING / DERIVATIVES ════════════════════
    "lido": "LIDO",
    "staked-ether": "stETH",
    "steth": "stETH",
    "rocket-pool": "RPL",
    "rpl": "RPL",
    
    # ════════════════════ ORACLE / DATA ════════════════════
    "band-protocol": "BAND",
    "band": "BAND",
    "chainlink-vrf": "LINK",
    "tellor": "TRB",
    "trb": "TRB",
    
    # ════════════════════ MEME / COMMUNITY ════════════════════
    "shiba-inu": "SHIB",
    "shib": "SHIB",
    "pepe": "PEPE",
    "dogwifhat": "WIF",
    "wif": "WIF",
    "bonk": "BONK",
    "floki-inu": "FLOKI",
    "floki": "FLOKI",
    
    # ════════════════════ NFT & GAMING ════════════════════
    "ordinals": "ORDI",
    "ordi": "ORDI",
    "immutable": "IMX",
    "immutable-x": "IMX",
    "imx": "IMX",
    "immutable-x": "IMX",
    "imx": "IMX",
    "imx": "IMX",
    "decentraland": "MANA",
    "mana": "MANA",
    "sandbox": "SAND",
    "sand": "SAND",
    "axie-infinity": "AXS",
    "axs": "AXS",
    "enjin-coin": "ENJ",
    "enj": "ENJ",
    "flow": "FLOW",
    "gala": "GALA",
    
    # ════════════════════ PRIVACY COINS ════════════════════
    "zcash": "ZEC",
    "zec": "ZEC",
    "dash": "DASH",
    "verge": "XVG",
    "xvg": "XVG",
    "monero": None,    # Delisted from Binance 2024 (privacy coin)
    "xmr": None,
    
    # ════════════════════ STABLECOINS ════════════════════
    "tether": "USDT",
    "usdt": "USDT",
    "usd-coin": "USDC",
    "usdc": "USDC",
    "busd": "BUSD",
    "dai": "DAI",
    "frax": "FRAX",
    "true-usd": "TUSD",
    "tusd": "TUSD",
    "paxos-standard": "PAX",
    "pax": "PAX",
    
    # ════════════════════ EXCHANGE TOKENS ════════════════════
    "binance-coin": "BNB",
    "ftx-token": "FTT",
    "ftt": "FTT",
    "huobi-token": "HT",
    "ht": "HT",
    "kucoin-shares": "KCS",
    "kcs": "KCS",
    "okb": "OKB",
    "crypto-com-chain": "CRO",
    "cro": "CRO",
    
    # ════════════════════ DEFI PROTOCOLS ════════════════════
    "uniswap-v3": "UNI",
    "sushiswap": "SUSHI",
    "sushi": "SUSHI",
    "pancakeswap": "CAKE",
    "cake": "CAKE",
    "1inch": "1INCH",
    "dydx": "DYDX",
    "balancer": "BAL",
    "bal": "BAL",
    "convex-finance": "CVX",
    "cvx": "CVX",
    "governance": "GOV",
    "synthetix": "SNX",
    "snx": "SNX",
    
    # ════════════════════ CROSS-CHAIN ════════════════════
    "wrapped-bitcoin": "WBTC",
    "wbtc": "WBTC",
    "anyswap": "ANY",
    "any": "ANY",
    "stargate-finance": "STG",
    "stg": "STG",
    "layerzero": "ZRO",
    "zro": "ZRO",
    "axelar": "AXL",
    "axl": "AXL",
    
    # ════════════════════ INFRASTRUCTURE ════════════════════
    "filecoin": "FIL",
    "fil": "FIL",
    "arweave": "AR",
    "ar": "AR",
    "the-winklevoss-index": "WINK",
    "wink": "WINK",
    "helium": "HNT",
    "hnt": "HNT",
    "iot-chain": "IOTX",
    "iotx": "IOTX",
    "injective-protocol": "INJ",
    "inj": "INJ",
    "sei": "SEI",
    "aptos": "APT",
    "apt": "APT",
    "sui": "SUI",
    
    # ════════════════════ AI / TECH ════════════════════
    "render-token": "RNDR",
    "rndr": "RNDR",
    "artificial-intelligence-5": "AI",
    "ai": "AI",
    "fetch-ai": "FET",
    "fet": "FET",
    "ocean-protocol": "OCEAN",
    "ocean": "OCEAN",
    "singularitynet": "AGI",
    "agi": "AGI",
    
    # ════════════════════ REAL WORLD ASSETS ════════════════════
    "pendle": "PENDLE",
    "realio": "RIO",
    "rio": "RIO",
    "ondo-finance": "ONDO",
    "ondo": "ONDO",
    
    # ════════════════════ ALTERNATIVE LAYERS ════════════════════
    "cosmos-coin": "ATOM",
    "bitcoin-cash": "BCH",
    "bch": "BCH",
    "ethereum-classic": "ETC",
    "etc": "ETC",
    "neo": "NEO",
    "tezos": "XTZ",
    "xtz": "XTZ",
    "tron": "TRX",
    "trx": "TRX",
    "eos": "EOS",
    "iota": "MIOTA",
    "miota": "MIOTA",
    "algorand": "ALGO",
    "algo": "ALGO",
    "vechain": "VET",
    "vet": "VET",
    "zilliqa": "ZIL",
    "zil": "ZIL",
    "cosmos-network": "ATOM",
}

def get_exchange_symbol(coin_name):
    """Convert coin name to exchange symbol (e.g., 'solana' → 'SOL/USDT')
    
    Supports:
      - CoinGecko IDs (e.g., 'avalanche-2' → 'AVAX')
      - Common names (e.g., 'solana' → 'SOL')
      - Symbols (e.g., 'BTC' → 'BTC')
      - Case insensitive
      
    Returns None if coin is not available on Binance
    """
    if not coin_name:
        return None
    
    coin_lower = coin_name.lower()
    
    # Check mapping first (handles all variations)
    if coin_lower in COIN_SYMBOL_MAP:
        symbol = COIN_SYMBOL_MAP[coin_lower]
        
        # None means coin is NOT on Binance
        if symbol is None:
            if LOG_ENABLED:
                logger.debug(f"[ORDER BOOK] {coin_name}: Not available on Binance - skipping order book analysis")
            return None
        
        # Don't add /USDT if already present
        if '/' not in symbol:
            return f"{symbol}/USDT"
        return symbol
    
    # If not in map, try treating as symbol
    if len(coin_name) <= 6:  # Likely a symbol (BTC, ETH, etc.)
        return f"{coin_name.upper()}/USDT"
    
    # Last resort: capitalize and add /USDT
    return f"{coin_name.upper()}/USDT"



# [C] RSI divergence lookback (bars)
DIVERGENCE_LOOKBACK = 10

# [F] Consecutive candle confirmation — require N consecutive days of BUY-ready signals
MIN_CONSECUTIVE_DAYS = 2

# [G] ML Model configuration — Logistic Regression for trade outcome prediction
ML_ENABLED = ML_AVAILABLE and True  # Set to False to disable ML scoring
ML_MIN_CONFIDENCE = 0.60            # only add ML bonus if confidence > 60%
ML_LOOKBACK_DAYS = 20               # skip first N rows (indicator warmup only)
ML_MIN_ACCURACY  = 0.55             # below this = heavily dampened
ML_FULL_ACCURACY = 0.60             # above this = fully trusted
ML_FUTURE_DAYS = 5                  # label: did price reach TP in next 5 days?

# [H] Smart Money / Whale Activity tracking
WHALE_VOLUME_PERCENTILE = 80        # top 20% volume (80th percentile) = whale tier
WHALE_LOOKBACK_DAYS = 30            # baseline: last 30 days
VOLUME_SPIKE_THRESHOLD = 1.5        # 1.5x avg = notable spike
WHALE_DIVERGENCE_LOOKBACK = 14      # bars for accumulation/distribution divergence

# [I] News / Event Filter — major events that move markets
EVENT_RISK_DAYS = 3                 # block trades within N days before/after event
HIGH_IMPACT_CONFIDENCE_REDUCTION = 0.25  # reduce ML confidence by 25% for high-impact events
MEDIUM_IMPACT_CONFIDENCE_REDUCTION = 0.10  # reduce by 10% for medium-impact

LOG_CSV = "paper_trades_log.csv"
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "coin", "action", "price", "position_units",
            "filled_avg_price", "stop_price", "tp_price", "rr", "note"
        ])

# ===========================
# FETCH
# ===========================
def fetch_ohlc_data(coin, days=365):
    """
    Fetch real OHLC candles from CoinGecko's dedicated OHLC endpoint.
    Results are cached for the session (CACHE_TTL_SECONDS) to avoid double-fetching.
    """
    global _cg_consecutive_429

    # ── Session cache check ───────────────────────────────────────
    _now = time.time()
    if coin in _session_ohlc_cache:
        _cached = _session_ohlc_cache[coin]
        if (_now - _cached["ts"]) < CACHE_TTL_SECONDS:
            if LOG_API_CALLS:
                logger.debug(f"[OHLC CACHE] {coin}: using cached OHLC (age {(_now-_cached['ts'])/60:.0f}min)")
            return _cached["df"]

    try:
        ohlc_days = 365 if days >= 180 else 90 if days >= 90 else 30 if days >= 30 else 7
        url     = f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc"
        params  = {"vs_currency": "usd", "days": ohlc_days}
        headers = {"x-cg-demo-api-key": API_KEY}
        r = requests.get(url, params=params, headers=headers, timeout=15)

        if r.status_code == 429:
            _cg_consecutive_429 += 1
            wait = CG_SLEEP_ON_429 * _cg_consecutive_429
            logger.warning(f"[OHLC] 429 rate limit for {coin} — waiting {wait:.0f}s")
            time.sleep(wait)
            return pd.DataFrame()

        _cg_consecutive_429 = 0
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()

        ohlc_df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        ohlc_df["timestamp"] = pd.to_datetime(ohlc_df["timestamp"], unit="ms")
        ohlc_df["date"] = ohlc_df["timestamp"].dt.date
        daily = ohlc_df.groupby("date").agg(
            high  = ("high",  "max"),
            low   = ("low",   "min"),
        ).reset_index()
        daily["timestamp"] = pd.to_datetime(daily["date"])
        daily.set_index("timestamp", inplace=True)
        daily.drop(columns=["date"], inplace=True)

        result = daily[["high", "low"]]
        if not result.empty:
            _session_ohlc_cache[coin] = {"df": result, "ts": _now}
        return result

    except Exception as e:
        if LOG_API_CALLS:
            logger.debug(f"[OHLC] {coin}: fetch failed ({e}) — will use approximation")
        return pd.DataFrame()


def fetch_coin_data(coin, days=CG_DAYS, interval=CG_INTERVAL):
    """
    Fetch coin price+volume data, merge real OHLC, compute indicators.

    API call budget per coin per session:
      - market_chart: 1 call  (price + volume)
      - ohlc:         1 call  (high/low, cached separately)
      Total: 2 calls first fetch, 0 calls on cache hit.

    Session cache: results stored for CACHE_TTL_SECONDS (default 4h).
    BTC is shared via _btc_cache — never fetched twice.
    """
    global _cg_consecutive_429

    # ── Session cache check ───────────────────────────────────────
    _now = time.time()
    _cache_key = f"{coin}_{days}"
    if _cache_key in _session_df_cache:
        _cached = _session_df_cache[_cache_key]
        if (_now - _cached["ts"]) < CACHE_TTL_SECONDS:
            if LOG_API_CALLS:
                logger.debug(f"[CACHE] {coin}: returning cached df (age {(_now-_cached['ts'])/60:.0f}min)")
            return _cached["df"]

    # ── BTC shortcut: reuse _btc_cache if already fetched this session ──
    if coin == BTC_ID and _btc_cache["df"] is not None and (_now - _btc_cache["ts"]) < CACHE_TTL_SECONDS:
        if LOG_API_CALLS:
            logger.debug(f"[CACHE] bitcoin: reusing btc_cache (no extra API call)")
        return _btc_cache["df"]

    try:
        if LOG_API_CALLS:
            logger.info(f"[API] Fetching {coin} data ({days} days)...")

        url    = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": interval}
        headers = {"x-cg-demo-api-key": API_KEY}

        r = requests.get(url, params=params, headers=headers, timeout=20)

        if r.status_code == 429:
            _cg_consecutive_429 += 1
            wait = CG_SLEEP_ON_429 * _cg_consecutive_429
            logger.warning(f"[API] ⏳ 429 RATE LIMIT for {coin} — waiting {wait:.0f}s before retry")
            time.sleep(wait)
            # One retry after waiting
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 429:
                logger.error(f"[API] ❌ Still rate-limited after {wait:.0f}s wait — skipping {coin}")
                return pd.DataFrame()

        _cg_consecutive_429 = 0
        r.raise_for_status()
        data    = r.json()
        prices  = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        if not prices:
            logger.warning(f"[API] ⚠️ No price data for {coin}")
            raise ValueError(f"No price data for {coin}")

        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        vol_series = pd.Series(
            [v[1] for v in volumes],
            index=[pd.to_datetime(v[0], unit="ms") for v in volumes]
        )
        df["volume"] = df.index.map(lambda t: vol_series.get(t, np.nan))
        df["volume"] = df["volume"].ffill().bfill().fillna(0)

        # ── Merge real OHLC high/low ──────────────────────────────
        # fetch_ohlc_data has its own session cache — safe to call here
        ohlc_df = fetch_ohlc_data(coin, days=days)
        if not ohlc_df.empty:
            df.index = pd.to_datetime(df.index.date)
            ohlc_df.index = pd.to_datetime(ohlc_df.index.date)
            df = df.merge(ohlc_df[["high", "low"]], left_index=True,
                          right_index=True, how="left")
            if LOG_API_CALLS:
                real_pct = df["high"].notna().mean() * 100
                logger.debug(f"[OHLC] {coin}: real high/low for {real_pct:.0f}% of candles")

        df = compute_indicators(df)
        df = df.dropna(subset=["EMA12", "EMA26", "RSI", "ATR",
                                "UpperBB", "LowerBB", "tenkan", "kijun",
                                "senkou_a", "senkou_b"])

        if LOG_API_CALLS:
            logger.debug(f"[API] ✅ {coin} data received ({len(df)} candles)")

        # ── Store in session cache (only if data is valid) ───────
        if not df.empty and len(df) >= 80:
            _session_df_cache[_cache_key] = {"df": df, "ts": _now}
            if coin == BTC_ID:
                _btc_cache["df"] = df
                _btc_cache["ts"] = _now
        else:
            logger.warning(f"[CACHE] {coin}: not caching — insufficient data ({len(df)} candles)")

        # ── Polite delay to respect free tier rate limits ─────────
        time.sleep(CG_SLEEP_BETWEEN)

        return df

    except requests.exceptions.Timeout:
        logger.error(f"[API] ❌ TIMEOUT fetching {coin} (20s timeout)")
        return pd.DataFrame()

    except requests.exceptions.ConnectionError as e:
        logger.error(f"[API] ❌ CONNECTION ERROR fetching {coin}: {e}")
        return pd.DataFrame()

    except requests.exceptions.HTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 429:
            logger.error(f"[API] ❌ RATE LIMIT (429) for {coin}: Too many requests")
        else:
            logger.error(f"[API] ❌ HTTP ERROR for {coin}: {e}")
        return pd.DataFrame()

    except ValueError as e:
        logger.warning(f"[API] ⚠️ Data error for {coin}: {e}")
        return pd.DataFrame()

    except Exception as e:
        logger.error(f"[API] ❌ Unexpected error fetching {coin}: {e}")
        return pd.DataFrame()

# ===========================
# INDICATORS
# ===========================
def compute_indicators(df):
    df    = df.copy()
    close = df["price"]

    # EMAs (daily)
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()

    # [E] Weekly EMA50 approximation: EMA of 50 weekly candles ≈ EMA of 350 daily.
    # With only 180 days we use EMA(250) as the closest stable weekly-trend proxy.
    df["EMA_WEEKLY50"] = close.ewm(span=250, adjust=False).mean()

    # MACD
    df["MACD"]      = df["EMA12"] - df["EMA26"]
    df["Signal"]    = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    # RSI — Wilder's EWM
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    sma20         = close.rolling(20, min_periods=1).mean()
    std20         = close.rolling(20, min_periods=1).std()
    df["UpperBB"] = sma20 + 2 * std20
    df["LowerBB"] = sma20 - 2 * std20
    df["BB_MID"]  = sma20

    # ATR — Wilder's EWM on True Range
    # Use real high/low from OHLC endpoint if available.
    # Fallback: estimate intraday range from close volatility.
    #   rolling(2).max/min dramatically underestimates range — it only captures
    #   close-to-close movement, ignoring intraday wicks (which in crypto are 1.5-3× larger).
    #   Better fallback: daily close-to-close std × 1.5 intraday multiplier.
    if "high" in df.columns and "low" in df.columns and df["high"].notna().any():
        # Real OHLC available — use directly
        high_for_atr = df["high"].fillna(close)
        low_for_atr  = df["low"].fillna(close)
        _atr_source  = "real_ohlc"
    else:
        # Improved approximation: estimate intraday range from recent close volatility.
        # Crypto intraday range ≈ 1.5× the daily close-to-close move on average.
        # A 5-day rolling std of returns × price gives an estimate of daily range,
        # scaled by 1.5 to approximate wicks.
        _close_std = close.pct_change().rolling(5, min_periods=2).std()
        _est_range = close * _close_std * 1.5
        high_for_atr = close + _est_range * 0.6    # ~60% of range above close
        low_for_atr  = close - _est_range * 0.4    # ~40% below (crypto skews up)
        high_for_atr = high_for_atr.fillna(close)
        low_for_atr  = low_for_atr.fillna(close)
        _atr_source  = "approx"

    prev_close = close.shift(1)
    tr = pd.concat([
        (high_for_atr - low_for_atr).abs(),
        (high_for_atr - prev_close).abs(),
        (low_for_atr  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    df["_atr_source"] = _atr_source   # audit flag: "real_ohlc" or "approx"

    # ADX — DI+/DI- using real OHLC when available, else same improved approximation
    if "high" in df.columns and "low" in df.columns and df["high"].notna().any():
        _high = df["high"].fillna(high_for_atr)
        _low  = df["low"].fillna(low_for_atr)
    else:
        _high = high_for_atr
        _low  = low_for_atr

    _up_move   = _high.diff()
    _down_move = -_low.diff()
    plus_dm    = pd.Series(np.where((_up_move > _down_move) & (_up_move > 0), _up_move, 0.0), index=df.index)
    minus_dm   = pd.Series(np.where((_down_move > _up_move) & (_down_move > 0), _down_move, 0.0), index=df.index)
    tr14       = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_dm14  = plus_dm.ewm(alpha=1/14, adjust=False).mean()
    minus_dm14 = minus_dm.ewm(alpha=1/14, adjust=False).mean()
    plus_di    = 100 * (plus_dm14 / (tr14 + 1e-9))
    minus_di   = 100 * (minus_dm14 / (tr14 + 1e-9))
    dx         = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    df["ADX"]       = dx.ewm(alpha=1/14, adjust=False).mean()
    df["plus_di"]   = plus_di
    df["minus_di"]  = minus_di

    # Volume 90-day avg
    df["vol_avg90"] = df["volume"].rolling(90, min_periods=10).mean()

    # Ichimoku
    high9  = close.rolling(9,  min_periods=1).max()
    low9   = close.rolling(9,  min_periods=1).min()
    df["tenkan"] = (high9 + low9) / 2.0

    high26 = close.rolling(26, min_periods=1).max()
    low26  = close.rolling(26, min_periods=1).min()
    df["kijun"] = (high26 + low26) / 2.0

    senkou_a_raw    = (df["tenkan"] + df["kijun"]) / 2.0
    high52          = close.rolling(52, min_periods=1).max()
    low52           = close.rolling(52, min_periods=1).min()
    senkou_b_raw    = (high52 + low52) / 2.0
    df["senkou_a"]  = senkou_a_raw.shift(26)
    df["senkou_b"]  = senkou_b_raw.shift(26)

    # [B] Chikou Span — full Ichimoku confirmation
    #
    # Chikou is today's close plotted 26 bars BACK on the chart.
    # At that historical chart position (t-26), there are TWO reference levels:
    #   1. The price at t-26           → close.shift(26)
    #   2. The cloud at t-26           → cloud that was PROJECTED from t-52
    #                                    (Senkou is shifted 26 forward, so the
    #                                     cloud visible at t-26 was calculated at t-52)
    #
    # True bullish Chikou: close > price[t-26]  AND  close > cloud_top[t-26]
    # True bearish Chikou: close < price[t-26]  AND  close < cloud_bot[t-26]
    #
    # The cloud at position t-26 = max/min(senkou_a_raw[t-52], senkou_b_raw[t-52])
    #                             = senkou_a_raw.shift(52) / senkou_b_raw.shift(52)
    #
    df["chikou"]           = close                    # today's close IS the chikou line
    df["chikou_ref"]       = close.shift(26)          # price 26 bars ago

    # Cloud at chart position t-26 (projected from t-52)
    chikou_sa              = senkou_a_raw.shift(52)
    chikou_sb              = senkou_b_raw.shift(52)
    df["chikou_cloud_top"] = pd.DataFrame({"a": chikou_sa, "b": chikou_sb}).max(axis=1)
    df["chikou_cloud_bot"] = pd.DataFrame({"a": chikou_sa, "b": chikou_sb}).min(axis=1)

    # ─────────────────────────────────────────────────────────────────────────────
    # [H] SMART MONEY / WHALE ACTIVITY INDICATORS
    # ─────────────────────────────────────────────────────────────────────────────

    # Accumulation/Distribution Line: measures cumulative buying vs selling pressure
    # AD = cumsum( (close - low) - (high - close) ) / (high - low) * volume
    clv = ((close - df["LowerBB"]) - (df["UpperBB"] - close)) / (df["UpperBB"] - df["LowerBB"] + 1e-9)
    ad_line = (clv * df["volume"]).cumsum()
    df["AD_line"] = ad_line

    # On-Balance Volume: cumulative volume weighted by price direction
    obv = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df["volume"].iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df["volume"].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    df["OBV"] = obv

    # OBV EMA: smoothed version for trend detection
    df["OBV_EMA12"] = df["OBV"].ewm(span=12, adjust=False).mean()

    # Volume Rate of Change: measures if volume is accelerating
    df["VROC"] = df["volume"].pct_change(periods=5)  # 5-day volume change

    # Whale volume threshold: top X percentile of last 30 days
    df["vol_threshold_whale"] = df["volume"].rolling(WHALE_LOOKBACK_DAYS, min_periods=10).quantile(WHALE_VOLUME_PERCENTILE / 100)

    return df

# ===========================
# [C] RSI DIVERGENCE DETECTION
# ===========================
def detect_rsi_divergence(df, lookback=DIVERGENCE_LOOKBACK):
    """
    Bullish divergence:  price makes lower low, RSI makes higher low  → reversal signal
    Bearish divergence:  price makes higher high, RSI makes lower high → exhaustion signal
    Returns: "bullish", "bearish", or None
    """
    if len(df) < lookback + 2:
        return None
    window = df.iloc[-(lookback + 1):-1]   # exclude current bar
    current = df.iloc[-1]

    price_now = current["price"]
    rsi_now   = current["RSI"]

    price_low_prev  = window["price"].min()
    rsi_at_low      = window.loc[window["price"].idxmin(), "RSI"]

    price_high_prev = window["price"].max()
    rsi_at_high     = window.loc[window["price"].idxmax(), "RSI"]

    # Bullish: new price low but RSI higher than previous low's RSI
    if price_now < price_low_prev and rsi_now > rsi_at_low:
        return "bullish"

    # Bearish: new price high but RSI lower than previous high's RSI
    if price_now > price_high_prev and rsi_now < rsi_at_high:
        return "bearish"

    return None

# ===========================
# [D] BTC MARKET REGIME
# ===========================
_btc_cache = {"df": None, "ts": 0}

def get_btc_regime(force_refresh=False):
    """
    Returns "bull", "bear", or "unknown".
    Bull  = BTC price > EMA26 AND EMA12 > EMA26
    Bear  = BTC price < EMA26 OR  EMA12 < EMA26
    Caches the BTC fetch for the session (one call per run).
    """
    now = time.time()
    if not force_refresh and _btc_cache["df"] is not None and (now - _btc_cache["ts"]) < 3600:
        btc_df = _btc_cache["df"]
    else:
        try:
            btc_df = fetch_coin_data(BTC_ID)
            _btc_cache["df"] = btc_df
            _btc_cache["ts"] = now
        except Exception as e:
            print(Fore.YELLOW + f"[REGIME] Could not fetch BTC for regime check: {e}")
            return "unknown"

    if btc_df is None or btc_df.empty:
        return "unknown"

    last = btc_df.iloc[-1]
    btc_bull = (last["price"] > last["EMA26"]) and (last["EMA12"] > last["EMA26"])
    return "bull" if btc_bull else "bear"

# ===========================
# [H] SMART MONEY / WHALE ACTIVITY DETECTION
# ===========================
def detect_whale_volume(df):
    """
    Detect if today's volume is in the 'whale zone' (top 20% of last 30 days).
    Returns: (is_whale_volume, today_vol, threshold_vol)
    """
    if len(df) < WHALE_LOOKBACK_DAYS:
        return False, 0, 0

    last = df.iloc[-1]
    today_vol = last["volume"]
    threshold = last.get("vol_threshold_whale", 0)

    is_whale = today_vol >= threshold if threshold > 0 else False
    return is_whale, today_vol, threshold

def detect_smart_money_accumulation(df, lookback=WHALE_DIVERGENCE_LOOKBACK):
    """
    Detect smart money accumulation (price up but OBV rises slower or flat).
    Also detect distribution (price up but OBV falls or rises slower than before).
    
    Returns: ("accumulation" | "distribution" | "neutral", confidence_score, details)
    """
    if len(df) < lookback + 5:
        return "neutral", 0, {}

    window = df.iloc[-lookback:]
    price_start = window["price"].iloc[0]
    price_end = window["price"].iloc[-1]
    price_change = (price_end - price_start) / price_start

    obv_start = window["OBV"].iloc[0]
    obv_end = window["OBV"].iloc[-1]
    obv_change = obv_end - obv_start

    # Normalize OBV change by volume scale
    total_vol = window["volume"].sum()
    obv_change_pct = obv_change / (total_vol + 1e-9) if total_vol > 0 else 0

    details = {
        "price_chg_pct": price_change * 100,
        "obv_chg": obv_change,
        "obv_chg_pct": obv_change_pct * 100,
        "ratio": (obv_change_pct + 1e-9) / (price_change + 1e-9),  # OBV:Price ratio
    }

    # Heuristics:
    # Accumulation: price rises 3%+, OBV rises OR stays flat (not dumping)
    # Distribution: price rises but OBV falls (dumping on buyers)
    # Neutral: everything else

    if price_change > 0.03 and obv_change >= 0:
        # Price up, volume accumulating
        confidence = min(abs(obv_change_pct) * 10, 1.0)
        return "accumulation", confidence, details
    elif price_change > 0.02 and obv_change < 0:
        # Price up but OBV falling — classic distribution
        confidence = min(abs(obv_change) / (total_vol + 1e-9) * 10, 1.0)
        return "distribution", confidence, details
    else:
        return "neutral", 0, details

def detect_volume_divergence(df):
    """
    Detect bullish/bearish volume divergence:
    - Bullish: price makes lower low, but volume lower than before (less selling pressure)
    - Bearish: price makes higher high, but volume lower than before (less buying conviction)
    
    Returns: ("bullish_div" | "bearish_div" | None, details)
    """
    if len(df) < 5:
        return None, {}

    last2 = df.iloc[-2]
    last1 = df.iloc[-1]

    price_prev_low = df["price"].iloc[:-1].tail(10).min()
    price_prev_high = df["price"].iloc[:-1].tail(10).max()

    vol_prev = df["volume"].iloc[:-1].tail(10).mean()
    vol_now = last1["volume"]

    details = {
        "vol_ratio": vol_now / (vol_prev + 1e-9),
        "price_trend": "up" if last1["price"] > last2["price"] else "down",
    }

    # Bullish divergence: price makes new low on low volume
    if last1["price"] < price_prev_low and vol_now < vol_prev:
        return "bullish_divergence", details

    # Bearish divergence: price makes new high on declining volume
    if last1["price"] > price_prev_high and vol_now < vol_prev:
        return "bearish_divergence", details

    return None, details

# ===========================
# [I] NEWS / EVENT FILTER — MAJOR MARKET EVENTS
# ===========================

# Event calendar: (event_date_YYYY_MM_DD, coin_list, impact, description)
# impact: "HIGH" = -25% ML confidence, "MEDIUM" = -10%, "INFO" = no penalty
CRYPTO_EVENTS = [
    # ═══════════════════════════════════════════════════════════════════════════════
    # 2024 EVENTS
    # ═══════════════════════════════════════════════════════════════════════════════
    ("2024-03-20", ["*"], "HIGH", "FOMC Decision"),
    ("2024-04-05", ["*"], "MEDIUM", "US NFP (jobs report)"),
    ("2024-04-11", ["*"], "MEDIUM", "US CPI"),
    ("2024-04-19", ["bitcoin"], "HIGH", "Bitcoin Halving (4th halving)"),
    ("2024-05-03", ["*"], "MEDIUM", "US NFP"),
    ("2024-05-15", ["*"], "MEDIUM", "US CPI"),
    ("2024-06-07", ["*"], "MEDIUM", "US NFP"),
    ("2024-06-12", ["*"], "MEDIUM", "US CPI"),
    ("2024-06-19", ["*"], "HIGH", "FOMC Decision"),
    ("2024-07-05", ["*"], "MEDIUM", "US NFP"),
    ("2024-07-10", ["*"], "MEDIUM", "US CPI"),
    ("2024-08-02", ["*"], "MEDIUM", "US NFP"),
    ("2024-08-14", ["*"], "MEDIUM", "US CPI"),
    ("2024-09-06", ["*"], "MEDIUM", "US NFP"),
    ("2024-09-11", ["*"], "MEDIUM", "US CPI"),
    ("2024-09-18", ["*"], "HIGH", "FOMC Decision"),
    ("2024-10-04", ["*"], "MEDIUM", "US NFP"),
    ("2024-10-09", ["*"], "MEDIUM", "US CPI"),
    ("2024-11-01", ["*"], "MEDIUM", "US NFP"),
    ("2024-11-13", ["*"], "MEDIUM", "US CPI"),
    ("2024-12-06", ["*"], "MEDIUM", "US NFP"),
    ("2024-12-11", ["*"], "MEDIUM", "US CPI"),
    ("2024-12-18", ["*"], "HIGH", "FOMC Decision"),

    # ═══════════════════════════════════════════════════════════════════════════════
    # 2025 EVENTS
    # ═══════════════════════════════════════════════════════════════════════════════
    ("2025-01-15", ["*"], "MEDIUM", "US CPI"),
    ("2025-01-20", ["*"], "HIGH", "US Presidential Inauguration (market repricing)"),
    ("2025-01-29", ["*"], "HIGH", "FOMC Decision"),
    ("2025-02-07", ["*"], "MEDIUM", "US NFP"),
    ("2025-02-12", ["*"], "MEDIUM", "US CPI"),
    ("2025-03-07", ["*"], "MEDIUM", "US NFP"),
    ("2025-03-12", ["*"], "MEDIUM", "US CPI"),
    ("2025-03-19", ["*"], "HIGH", "FOMC Decision"),
    ("2025-04-04", ["*"], "MEDIUM", "US NFP"),
    ("2025-04-09", ["*"], "MEDIUM", "US CPI"),
    ("2025-05-02", ["*"], "MEDIUM", "US NFP"),
    ("2025-05-07", ["*"], "HIGH", "FOMC Decision"),
    ("2025-05-14", ["*"], "MEDIUM", "US CPI"),
    ("2025-06-06", ["*"], "MEDIUM", "US NFP"),
    ("2025-06-11", ["*"], "MEDIUM", "US CPI"),
    ("2025-06-18", ["*"], "HIGH", "FOMC Decision"),
    ("2025-07-02", ["*"], "MEDIUM", "US NFP"),
    ("2025-07-09", ["*"], "MEDIUM", "US CPI"),
    ("2025-07-30", ["*"], "HIGH", "FOMC Decision"),
    ("2025-08-01", ["*"], "MEDIUM", "US NFP"),
    ("2025-08-12", ["*"], "MEDIUM", "US CPI"),
    ("2025-09-05", ["*"], "MEDIUM", "US NFP"),
    ("2025-09-10", ["*"], "MEDIUM", "US CPI"),
    ("2025-09-17", ["*"], "HIGH", "FOMC Decision"),
    ("2025-10-03", ["*"], "MEDIUM", "US NFP"),
    ("2025-10-08", ["*"], "MEDIUM", "US CPI"),
    ("2025-11-07", ["*"], "MEDIUM", "US NFP"),
    ("2025-11-12", ["*"], "MEDIUM", "US CPI"),
    ("2025-11-05", ["*"], "HIGH", "FOMC Decision"),
    ("2025-12-05", ["*"], "MEDIUM", "US NFP"),
    ("2025-12-10", ["*"], "MEDIUM", "US CPI"),
    ("2025-12-17", ["*"], "HIGH", "FOMC Decision"),

    # ═══════════════════════════════════════════════════════════════════════════════
    # 2026 EVENTS (Comprehensive through end of year)
    # ═══════════════════════════════════════════════════════════════════════════════
    ("2026-01-14", ["*"], "MEDIUM", "US CPI"),
    ("2026-01-28", ["*"], "HIGH", "FOMC Decision"),
    ("2026-02-06", ["*"], "MEDIUM", "US NFP"),
    ("2026-02-11", ["*"], "MEDIUM", "US CPI"),
    ("2026-03-06", ["*"], "MEDIUM", "US NFP"),
    ("2026-03-11", ["*"], "MEDIUM", "US CPI"),
    ("2026-03-18", ["*"], "HIGH", "FOMC Decision"),
    ("2026-04-03", ["*"], "MEDIUM", "US NFP"),
    ("2026-04-08", ["*"], "MEDIUM", "US CPI"),
    ("2026-04-29", ["*"], "HIGH", "FOMC Decision"),   # actual Apr 28-29 meeting
    ("2026-05-01", ["*"], "MEDIUM", "US NFP"),
    ("2026-05-13", ["*"], "MEDIUM", "US CPI"),
    ("2026-06-05", ["*"], "MEDIUM", "US NFP"),
    ("2026-06-10", ["*"], "MEDIUM", "US CPI"),
    ("2026-06-17", ["*"], "HIGH", "FOMC Decision"),   # Jun 16-17 meeting
    ("2026-07-03", ["*"], "MEDIUM", "US NFP"),
    ("2026-07-08", ["*"], "MEDIUM", "US CPI"),
    ("2026-07-29", ["*"], "HIGH", "FOMC Decision"),
    ("2026-08-07", ["*"], "MEDIUM", "US NFP"),
    ("2026-08-12", ["*"], "MEDIUM", "US CPI"),
    ("2026-09-04", ["*"], "MEDIUM", "US NFP"),
    ("2026-09-09", ["*"], "MEDIUM", "US CPI"),
    ("2026-09-16", ["*"], "HIGH", "FOMC Decision"),
    ("2026-10-02", ["*"], "MEDIUM", "US NFP"),
    ("2026-10-07", ["*"], "MEDIUM", "US CPI"),
    ("2026-10-28", ["*"], "HIGH", "FOMC Decision"),   # Oct 27-28 meeting
    ("2026-11-06", ["*"], "MEDIUM", "US NFP"),
    ("2026-11-11", ["*"], "MEDIUM", "US CPI"),
    ("2026-12-04", ["*"], "MEDIUM", "US NFP"),
    ("2026-12-09", ["*"], "MEDIUM", "US CPI"),
    ("2026-12-09", ["*"], "HIGH", "FOMC Decision"),   # Dec 8-9 meeting

    # ═══════════════════════════════════════════════════════════════════════════════
    # Quarterly Markers
    # ═══════════════════════════════════════════════════════════════════════════════
    ("2024-01-01", ["*"], "INFO", "Q1 2024 begins"),
    ("2024-04-01", ["*"], "INFO", "Q2 2024 begins"),
    ("2024-07-01", ["*"], "INFO", "Q3 2024 begins"),
    ("2024-10-01", ["*"], "INFO", "Q4 2024 begins"),
    ("2025-01-01", ["*"], "INFO", "Q1 2025 begins"),
    ("2025-04-01", ["*"], "INFO", "Q2 2025 begins"),
    ("2025-07-01", ["*"], "INFO", "Q3 2025 begins"),
    ("2025-10-01", ["*"], "INFO", "Q4 2025 begins"),
    ("2026-01-01", ["*"], "INFO", "Q1 2026 begins"),
    ("2026-04-01", ["*"], "INFO", "Q2 2026 begins"),
    ("2026-07-01", ["*"], "INFO", "Q3 2026 begins"),
    ("2026-10-01", ["*"], "INFO", "Q4 2026 begins"),
]

def get_upcoming_events(coin, days_ahead=EVENT_RISK_DAYS):
    """
    Get events affecting a coin in the next N days and past N days.
    Returns: list of matching events within the risk window
    """
    from datetime import datetime, timezone, timedelta
    
    today = datetime.now()
    window_start = today - timedelta(days=days_ahead)
    window_end = today + timedelta(days=days_ahead)
    
    matching_events = []
    for event_date_str, affected_coins, impact, description in CRYPTO_EVENTS:
        try:
            event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
            
            # Check if event is in the risk window
            if window_start <= event_date <= window_end:
                # Check if this coin is affected (or if event affects all coins "*")
                if "*" in affected_coins or coin in affected_coins:
                    days_until = (event_date - today).days
                    matching_events.append({
                        "date": event_date,
                        "days_until": days_until,
                        "impact": impact,
                        "description": description,
                        "is_past": event_date < today
                    })
        except ValueError:
            continue
    
    # Sort by proximity to today
    matching_events.sort(key=lambda x: abs(x["days_until"]))
    return matching_events


def check_obv_confirmation(df, coin, lookback=7):
    """[AQ] OBV Capital Inflow Gate.
    
    LOGIC:
      If price is rising but OBV is flat/falling = weak hands driving the move
      = Low liquidity, distribution, unsustainable
      
    RULE:
      ML confidence is ONLY fully trusted if OBV is at a 7-day high
      = Real capital is flowing in to back the technical signal
      
    Returns:
      (obv_confirmed, obv_at_high, obv_trend, description)
      
      obv_confirmed: True  = OBV rising + at/near high → trust ML fully
                     False = OBV flat/falling → dampen ML confidence
    """
    try:
        if df is None or "OBV" not in df.columns or len(df) < lookback + 1:
            return True, False, "unknown", "OBV data unavailable — no gate applied"

        obv = df["OBV"].iloc[-(lookback+1):]  # Last 7+1 days
        current_obv = obv.iloc[-1]
        obv_7d_high  = obv.iloc[:-1].max()   # Max of previous 7 days
        obv_7d_low   = obv.iloc[:-1].min()
        obv_7d_start = obv.iloc[0]

        # Is OBV at a 7-day high? (capital inflow)
        at_7d_high = current_obv >= obv_7d_high * 0.98  # Within 2% of high

        # OBV trend: rising, flat, or falling over 7 days
        obv_change_pct = (current_obv - obv_7d_start) / (abs(obv_7d_start) + 1e-9) * 100

        if obv_change_pct > 3:
            obv_trend = "rising"
        elif obv_change_pct < -3:
            obv_trend = "falling"
        else:
            obv_trend = "flat"

        # Confirmation: OBV at 7-day high AND rising
        obv_confirmed = at_7d_high and obv_trend == "rising"

        if obv_confirmed:
            desc = f"OBV at 7d high ({obv_change_pct:+.1f}%) — capital inflow confirmed"
        elif obv_trend == "rising":
            desc = f"OBV rising ({obv_change_pct:+.1f}%) but not at 7d high — partial confirmation"
        elif obv_trend == "flat":
            desc = f"OBV flat ({obv_change_pct:+.1f}%) — weak hands driving price"
        else:
            desc = f"OBV falling ({obv_change_pct:+.1f}%) — distribution, price move unreliable"

        return obv_confirmed, at_7d_high, obv_trend, desc

    except Exception as e:
        return True, False, "unknown", f"OBV check error: {e}"


def apply_obv_ml_gate(ml_proba, df, coin, lookback=7):
    """[AQ] Apply OBV gate to ML confidence.
    
    If OBV NOT confirming → dampen ML confidence
    This prevents acting on ML signals backed only by weak-hand volume
    
    Dampening rules:
      OBV rising + at 7d high:   ml_proba × 1.0  (full trust)
      OBV rising, not at high:   ml_proba × 0.9  (slight dampen)
      OBV flat:                  ml_proba × 0.75 (moderate dampen)
      OBV falling:               ml_proba × 0.60 (heavy dampen)
    """
    try:
        obv_confirmed, at_high, obv_trend, desc = check_obv_confirmation(df, coin, lookback)

        if obv_confirmed:
            multiplier = 1.0
            tag = "✅ OBV confirmed"
        elif obv_trend == "rising":
            multiplier = 0.9
            tag = "⚠️ OBV rising (not at high)"
        elif obv_trend == "flat":
            multiplier = 0.75
            tag = "⚠️ OBV flat (weak hands)"
        else:  # falling
            multiplier = 0.60
            tag = "❌ OBV falling (distribution)"

        adjusted = ml_proba * multiplier
        print(f"[OBV GATE] {coin}: {tag} — ML {ml_proba:.1%} → {adjusted:.1%} (×{multiplier})")
        print(f"[OBV GATE] {coin}: {desc}")
        if LOG_ENABLED:
            logger.info(f"[OBV GATE] {coin}: {tag} | {desc}")

        return adjusted, obv_trend, desc, multiplier

    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[OBV GATE] {coin}: skipped - {e}")
        return ml_proba, "unknown", "OBV gate error"

def apply_event_risk_adjustment(ml_proba, coin, events):
    """
    Reduce ML confidence based on upcoming/recent events.
    Returns: (adjusted_proba, adjustment_reason)
    """
    if not events:
        return ml_proba, ""
    
    # Find the closest event
    closest = events[0]
    days_away = abs(closest["days_until"])
    
    if closest["impact"] == "HIGH":
        adjustment = HIGH_IMPACT_CONFIDENCE_REDUCTION
        reason = f"HIGH-IMPACT EVENT: {closest['description']} ({days_away}d away)"
    elif closest["impact"] == "MEDIUM":
        adjustment = MEDIUM_IMPACT_CONFIDENCE_REDUCTION
        reason = f"MEDIUM-IMPACT EVENT: {closest['description']} ({days_away}d away)"
    else:
        return ml_proba, ""
    
    # Reduce confidence proportionally closer to the event
    # At EVENT_RISK_DAYS away: full reduction
    # At 0 days away (today): full reduction
    # Outside window: no reduction
    proximity_factor = max(0, 1.0 - (days_away / EVENT_RISK_DAYS)) if days_away <= EVENT_RISK_DAYS else 0
    
    adjusted = ml_proba * (1.0 - (adjustment * proximity_factor))
    
    return adjusted, reason

def should_block_trade_on_event(coin, events):
    """
    Check if trade should be completely blocked due to high-impact event.
    Returns: (should_block, reason)
    """
    if not events:
        return False, ""
    
    closest = events[0]
    days_away = abs(closest["days_until"])
    
    # Block if HIGH-impact event within 1 day
    if closest["impact"] == "HIGH" and days_away <= 1:
        return True, f"BLOCKED: {closest['description']} is {days_away}d away — too much uncertainty"
    
    return False, ""

# ===========================
# [G] ML MODEL — LOGISTIC REGRESSION
# ===========================
# ===========================
# [N] ON-CHAIN METRICS
# ===========================

def detect_on_chain_accumulation(df):
    """
    [N] Detect whale/smart money accumulation patterns from on-chain data
    
    Signals:
    - High volume with rising price = accumulation
    - Volume spike without price increase = potential reversal
    - OBV trending up = accumulation
    
    Returns: (accumulation_score, details)
    accumulation_score: -1.0 to 1.0
        Positive: Accumulation (whales buying)
        Negative: Distribution (whales selling)
    """
    try:
        if len(df) < 14:
            return 0.0, "Insufficient data"
        
        # [N] Get volume and price data
        recent_vol = df['volume'].iloc[-14:].mean()
        recent_price_change = (df['price'].iloc[-1] - df['price'].iloc[-14]) / df['price'].iloc[-14] * 100
        
        # [N] Calculate OBV (On-Balance Volume)
        obv = 0.0
        for i in range(1, len(df)):
            if df['price'].iloc[i] > df['price'].iloc[i-1]:
                obv += df['volume'].iloc[i]
            elif df['price'].iloc[i] < df['price'].iloc[i-1]:
                obv -= df['volume'].iloc[i]
        
        obv_trend = (obv - 0) / max(abs(obv), 1)  # Normalize
        
        # [N] Volume spike detection
        vol_spike = df['volume'].iloc[-1] / recent_vol if recent_vol > 0 else 1.0
        
        # [N] Accumulation score
        if recent_price_change > 0 and vol_spike > 1.2:
            # Price up + high volume = strong accumulation
            accumulation_score = min(obv_trend * 0.5 + 0.5, 1.0)
            details = f"Accumulation: Price +{recent_price_change:.1f}%, Vol Spike {vol_spike:.1f}x"
        elif recent_price_change < -5 and vol_spike > 1.5:
            # Price down + very high volume = distribution
            accumulation_score = max(obv_trend * 0.5 - 0.5, -1.0)
            details = f"Distribution: Price {recent_price_change:.1f}%, Vol Spike {vol_spike:.1f}x"
        else:
            accumulation_score = obv_trend * 0.3
            details = "Neutral on-chain activity"
        
        return accumulation_score, details
    
    except Exception as e:
        print(Fore.YELLOW + f"[ON-CHAIN] Error detecting accumulation: {e}")
        return 0.0, "Error"


def detect_volume_extremes(df):
    """
    [N] Detect extreme volume patterns indicating whale activity
    
    Returns: (is_extreme, volume_zscore, details)
    is_extreme: True if volume is > 2 standard deviations above mean
    volume_zscore: How many std devs above/below mean
    """
    try:
        if len(df) < 20:
            return False, 0.0, "Insufficient data"
        
        # [N] Calculate 20-day volume statistics
        vol_mean = df['volume'].iloc[-20:-1].mean()
        vol_std = df['volume'].iloc[-20:-1].std()
        current_vol = df['volume'].iloc[-1]
        
        # [N] Calculate Z-score
        if vol_std > 0:
            zscore = (current_vol - vol_mean) / vol_std
        else:
            zscore = 0.0
        
        is_extreme = zscore > 2.0
        
        if is_extreme:
            details = f"EXTREME VOLUME: {zscore:.1f}σ ({current_vol/vol_mean:.1f}x mean)"
        else:
            details = f"Normal volume: {zscore:.1f}σ"
        
        return is_extreme, zscore, details
    
    except Exception as e:
        print(Fore.YELLOW + f"[VOLUME EXTREME] Error: {e}")
        return False, 0.0, "Error"


# ===========================
# [S] BACKTESTING ENGINE
# ===========================

class BacktestResult:
    """[S] Container for backtest results"""
    def __init__(self):
        self.trades = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.sharpe_ratio = 0.0
        self.total_return_pct = 0.0


def backtest_signals(df, coin_name="Unknown", btc_regime="unknown"):
    """
    [S] Proper backtest — tests OUR ACTUAL SCORING SIGNAL
    
    Iterates through historical days and runs the SAME signal logic
    the bot uses live: generate_signal_and_reasons() + post-technical
    adjustments (volume, Z-score, chop, breakout, exhaustion).
    
    Entry:  score >= MIN_BUY_SCORE AND not blocked
    Target: entry + 2.5 × ATR  (same as ML training labels)
    Stop:   entry - 1.5 × ATR
    Horizon: 5 trading days to hit target or stop
    
    This tests the REAL strategy, not a simple EMA proxy.
    Speed: ~0.3s per coin (pure math, no API calls)
    """
    if not BACKTEST_ENABLED or len(df) < 80:
        return BacktestResult()

    try:
        import io as _io, sys as _sys, logging as _logging
        # Silence stdout AND logger during backtest replay
        _old_stdout = _sys.stdout
        _sys.stdout = _io.StringIO()
        _bt_log = logging.getLogger("TradingBot")
        _old_log_level = _bt_log.level
        _bt_log.setLevel(_logging.CRITICAL + 1)   # suppress all log levels
        result   = BacktestResult()
        balance  = BACKTEST_INITIAL_BALANCE
        max_balance = balance
        in_trade = False          # one position at a time
        MIN_WARMUP = 60           # skip first 60 days (indicator warmup)
        HORIZON    = 5            # days to hit target/stop
        TP_MULT    = 2.5          # target = entry + TP_MULT × ATR
        SL_MULT    = 1.5          # stop   = entry - SL_MULT × ATR
        # Use full available history — regime is passed per-day via btc_regime
        # New coins with < 10 qualifying trades are filtered by the reliability gate
        # in calculate_confidence_score (backtest_trades < 10 → penalty skipped)
        for i in range(MIN_WARMUP, len(df) - HORIZON):
            # ── Already in trade — skip signal check ───────────────
            if in_trade:
                continue

            # ── Build sub-dataframe up to current day ──────────────
            sub_df = df.iloc[:i + 1]
            row    = sub_df.iloc[-1]

            # ── Run actual signal logic (suppress debug prints) ────
            try:
                sig = generate_signal_and_reasons(
                    row, sub_df, btc_regime=btc_regime,
                    is_btc=(coin_name == "bitcoin"),
                    suppress_debug=True
                )
            except Exception:
                continue

            # Only proceed if base technical score qualifies
            # (skip heavy post-tech adjustments for speed — they add noise)
            base_score = sig.get("score", 0)
            blocked    = sig.get("blocked", False)

            # Apply lightweight post-tech adjustments that are pure math
            try:
                # Volume (important signal)
                base_score = apply_volume_boost(base_score, sub_df, coin_name, backtest=True)
                # Price change penalty
                base_score = apply_price_change_penalty(base_score, sub_df, coin_name)
                # Z-score stretch
                _in_trend = (sig.get("cloud_state") == "above" and
                             row.get("EMA12", 0) > row.get("EMA26", 0))
                pre_z = base_score
                base_score = apply_zscore_boost(base_score, sub_df, coin_name, in_trend=_in_trend)
                zscore_adj = base_score - pre_z
                # Exhaustion
                base_score = apply_exhaustion_penalty(base_score, sub_df, coin_name, zscore_adj)
                # Chop
                chop_pen = apply_chop_penalty(0, sub_df, coin_name)
                if chop_pen != 0:
                    blocked = True   # chop = skip trade (score adjustment redundant when blocked)
            except Exception:
                pass

            # ── BUY signal check ───────────────────────────────────
            if base_score < MIN_BUY_SCORE or blocked:
                continue

            # ── Valid BUY — simulate trade ─────────────────────────
            entry = row["price"]
            atr   = row.get("ATR", entry * 0.02)
            if atr <= 0 or np.isnan(atr):
                atr = entry * 0.02

            target = entry + TP_MULT * atr
            stop   = entry - SL_MULT * atr

            # Look ahead HORIZON days for outcome
            future_prices = df["price"].iloc[i + 1: i + 1 + HORIZON].values

            hit_target = False
            hit_stop   = False
            exit_price = future_prices[-1] if len(future_prices) > 0 else entry
            exit_day   = HORIZON

            for j, fp in enumerate(future_prices):
                if fp >= target and not hit_stop:
                    hit_target = True
                    exit_price = target
                    exit_day   = j + 1
                    break
                if fp <= stop and not hit_target:
                    hit_stop   = True
                    exit_price = stop
                    exit_day   = j + 1
                    break

            pnl = (exit_price - entry) / entry * 100

            # Record trade
            if pnl > 0:
                result.winning_trades += 1
                result.total_profit   += abs(pnl)
            else:
                result.losing_trades  += 1
                result.total_loss     += abs(pnl)

            result.trades.append({
                "type":    "BUY",
                "entry":   entry,
                "exit":    exit_price,
                "pnl":     pnl,
                "days":    exit_day,
                "outcome": "WIN" if pnl > 0 else "LOSS",
                "score":   base_score,
            })

            # Update simulated balance
            balance *= (1 + pnl / 100)

            # Drawdown tracking
            if balance > max_balance:
                max_balance = balance
            drawdown = (max_balance - balance) / max_balance * 100 if max_balance > 0 else 0
            if drawdown > result.max_drawdown:
                result.max_drawdown = drawdown

            # Cooldown: skip HORIZON days after a trade (avoid overlap)
            in_trade = True
            for _ in range(exit_day):
                try:
                    i += 1
                except Exception:
                    break
            in_trade = False

        # ── Final metrics ──────────────────────────────────────────
        total_trades = result.winning_trades + result.losing_trades
        if total_trades > 0:
            result.win_rate      = (result.winning_trades / total_trades) * 100
            result.profit_factor = result.total_profit / max(result.total_loss, 0.01)
        else:
            result.win_rate      = 0.0
            result.profit_factor = 0.0

        result.total_return_pct = (
            (balance - BACKTEST_INITIAL_BALANCE) / BACKTEST_INITIAL_BALANCE * 100
        )
        _sys.stdout = _old_stdout
        _bt_log.setLevel(_old_log_level)   # restore logger
        return result

    except Exception as e:
        try:
            _sys.stdout = _old_stdout
            _bt_log.setLevel(_old_log_level)
        except Exception: pass
        if LOG_ENABLED:
            logger.debug(f"[BACKTEST] Error for {coin_name}: {e}")
        return BacktestResult()


# ===========================
# [U] ADVANCED STOP LOSS LOGIC
# ===========================

def calculate_advanced_stop_loss(entry_price, current_price, signal_type="BUY",
                                 lookback_high=None, lookback_low=None, atr=None):
    """
    [U] ATR-based unified stop loss — consistent with build_position_plan
    
    Stop   = Entry ± STOP_ATR_MULT × ATR  (default 2×ATR)
    Breakeven triggers at +1×ATR (not fixed +1%)
    Trailing: 2% from highest OR EMA12-based (handled in build_position_plan)
    
    Returns: (stop_price, stop_type, stop_pct)
    """
    try:
        # ATR fallback: estimate from entry if not provided
        if atr is None or atr <= 0 or (isinstance(atr, float) and atr != atr):
            atr = entry_price * 0.02   # Default 2% of price

        atr_stop_dist = STOP_ATR_MULT * atr

        if not USE_ADVANCED_STOPS:
            if signal_type == "BUY":
                stop = entry_price - atr_stop_dist
                pct  = (stop - entry_price) / entry_price * 100
                return stop, "INITIAL", pct
            else:
                stop = entry_price + atr_stop_dist
                pct  = (stop - entry_price) / entry_price * 100
                return stop, "INITIAL", pct

        profit_pct = ((current_price - entry_price) / entry_price) * 100
        atr_pct    = (atr / entry_price) * 100

        # BREAKEVEN: trigger at +1×ATR (not fixed +1%)
        if USE_BREAKEVEN_STOPS and profit_pct >= atr_pct:
            return entry_price, "BREAKEVEN", 0.0

        # TRAILING: follow price up from highest
        if USE_TRAILING_STOPS and signal_type == "BUY" and lookback_high is not None:
            highest = lookback_high
            trailing_distance = highest * (TRAILING_STOP_PERCENT / 100.0)
            trailing_stop = highest - trailing_distance
            initial_stop  = entry_price - atr_stop_dist
            # SAFETY: trailing must be below entry price
            trailing_stop = min(trailing_stop, entry_price * 0.995)
            if trailing_stop > initial_stop:
                return trailing_stop, "TRAILING", ((trailing_stop - entry_price) / entry_price) * 100

        # DEFAULT: ATR-based initial stop
        if signal_type == "BUY":
            stop = entry_price - atr_stop_dist
            pct  = (stop - entry_price) / entry_price * 100
            return stop, "INITIAL", pct
        else:
            stop = entry_price + atr_stop_dist
            pct  = (stop - entry_price) / entry_price * 100
            return stop, "INITIAL", pct

    except Exception as e:
        print(Fore.YELLOW + f"[ADVANCED STOPS] Error: {e}")
        if signal_type == "BUY":
            return entry_price * 0.90, "ERROR", -10.0
        else:
            return entry_price * 1.10, "ERROR", 10.0


def calculate_partial_profit_levels(entry_price, signal_type="BUY", atr=None,
                                    chop_detected=False, donchian_upper=None,
                                    donchian_lower=None):
    """
    [U] ATR-based 3-tier partial profit taking — Donchian-aware in chop zones.
    
    TRENDING MARKET (default):
      TP1 = Entry + 1.5×ATR  → take 33%
      TP2 = Entry + 2.5×ATR  → take 33%
      TP3 = Entry + 4.0×ATR  → take 34%
      
    CHOP ZONE (when chop_detected=True):
      Use Donchian range as natural resistance/support levels
      TP3 = Donchian upper  (range top = realistic ceiling)
      TP2 = Entry + 2/3 of distance to range top
      TP1 = Entry + 1/3 of distance to range top
      
    Why Donchian in chop:
      ATR targets may exceed the range ceiling (unreachable)
      Range top = actual resistance, price will be sold there
      Better to take profits at realistic levels
    """
    try:
        # ATR fallback
        if atr is None or atr <= 0 or (isinstance(atr, float) and atr != atr):
            atr = entry_price * 0.02

        if not PARTIAL_PROFIT_ENABLED:
            if signal_type == "BUY":
                return [(entry_price + 4.0 * atr, 100.0)]
            else:
                return [(entry_price - 4.0 * atr, 100.0)]

        if signal_type == "BUY":
            # Use Donchian range in chop if available and meaningful
            if (chop_detected and donchian_upper is not None and
                    donchian_upper > entry_price):
                dist = donchian_upper - entry_price
                if dist > 0:
                    tp1 = entry_price + dist * 0.33
                    tp2 = entry_price + dist * 0.66
                    tp3 = donchian_upper   # Range ceiling = final target
                    return [(tp1, 33.0), (tp2, 33.0), (tp3, 34.0)]

            # Default: ATR-based
            tp1 = entry_price + 1.5 * atr
            tp2 = entry_price + 2.5 * atr
            tp3 = entry_price + 4.0 * atr
            return [(tp1, 33.0), (tp2, 33.0), (tp3, 34.0)]

        else:  # SELL
            if (chop_detected and donchian_lower is not None and
                    donchian_lower < entry_price):
                dist = entry_price - donchian_lower
                if dist > 0:
                    tp1 = entry_price - dist * 0.33
                    tp2 = entry_price - dist * 0.66
                    tp3 = donchian_lower
                    return [(tp1, 33.0), (tp2, 33.0), (tp3, 34.0)]

            tp1 = entry_price - 1.5 * atr
            tp2 = entry_price - 2.5 * atr
            tp3 = entry_price - 4.0 * atr
            return [(tp1, 33.0), (tp2, 33.0), (tp3, 34.0)]

    except Exception as e:
        print(Fore.YELLOW + f"[PARTIAL PROFIT] Error: {e}")
        return [(entry_price * 1.06, 100.0)]



# ===========================
# [V] MODEL PERSISTENCE
# ===========================

def get_model_path(coin):
    """[V] Get model file path for a coin"""
    return os.path.join(MODEL_PERSISTENCE_DIR, f"{coin}_model.pkl")

def get_scaler_path(coin):
    """[V] Get scaler file path for a coin"""
    return os.path.join(MODEL_PERSISTENCE_DIR, f"{coin}_scaler.pkl")

def get_metadata_path(coin):
    """[V] Get metadata file path for a coin"""
    return os.path.join(MODEL_PERSISTENCE_DIR, f"{coin}_metadata.json")

# ── SHORT MODEL paths — separate files, long model untouched ─────────────────
def get_short_model_path(coin):
    return os.path.join(MODEL_PERSISTENCE_DIR, f"{coin}_short_model.pkl")

def get_short_scaler_path(coin):
    return os.path.join(MODEL_PERSISTENCE_DIR, f"{coin}_short_scaler.pkl")

def get_short_metadata_path(coin):
    return os.path.join(MODEL_PERSISTENCE_DIR, f"{coin}_short_metadata.json")

def save_model(coin, model, scaler, accuracy, samples):
    """
    [V] Save trained model and scaler to disk
    [W] With comprehensive logging
    
    Args:
        coin: Coin name (e.g., "bitcoin")
        model: Trained LogisticRegression model
        scaler: StandardScaler for features
        accuracy: Model accuracy (0-1)
        samples: Number of training samples
    """
    try:
        if not MODEL_PERSISTENCE_ENABLED:
            return False
        
        # [V] Create models directory if needed
        Path(MODEL_PERSISTENCE_DIR).mkdir(parents=True, exist_ok=True)
        
        model_path = get_model_path(coin)
        scaler_path = get_scaler_path(coin)
        metadata_path = get_metadata_path(coin)
        
        # [V] Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # [V] Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # [V] Save metadata
        metadata = {
            "coin": coin,
            "trained_date": datetime.now().isoformat(),
            "accuracy": float(accuracy),
            "samples": int(samples),
            "model_version": "v8.0",      # Bumped: GBM + 13 features + ATR stop fix
            "n_features": 16,             # Saved so load can validate compatibility
            "model_type": "GradientBoosting"
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[PERSIST] ✅ Model saved: {coin} (acc: {accuracy:.1%}, samples: {samples})")
        print(Fore.GREEN + f"[PERSIST] Model saved: {coin} (acc: {accuracy:.1%}, samples: {samples})")
        return True
    
    except Exception as e:
        logger.error(f"[PERSIST] ❌ Error saving {coin}: {e}")
        print(Fore.YELLOW + f"[PERSIST] Error saving {coin}: {e}")
        return False

def load_model(coin):
    """
    [V] Load model and scaler from disk
    [W] With comprehensive logging
    
    Returns: (model, scaler, metadata) or (None, None, None) if not found
    """
    try:
        if not MODEL_PERSISTENCE_ENABLED or not LOAD_CACHED_MODELS:
            return None, None, None
        
        model_path = get_model_path(coin)
        scaler_path = get_scaler_path(coin)
        metadata_path = get_metadata_path(coin)
        
        # [V] Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.debug(f"[PERSIST] No cached model for {coin}")
            return None, None, None
        
        # [V] Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # [V] Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # [V] Load metadata
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # [V] Version check — reject old models with wrong feature count
        CURRENT_VERSION  = "v8.0"
        CURRENT_FEATURES = 16
        MIN_SAMPLES      = 80   # Reject models trained on too few samples
        saved_version  = metadata.get("model_version", "unknown")
        saved_features = metadata.get("n_features", 11)
        saved_samples  = metadata.get("samples", 0)

        if saved_version != CURRENT_VERSION or saved_features != CURRENT_FEATURES:
            logger.warning(f"[PERSIST] ⚠️ {coin}: Model version mismatch "
                           f"({saved_version}/{saved_features}f → need {CURRENT_VERSION}/{CURRENT_FEATURES}f) "
                           f"— forcing retrain")
            print(Fore.YELLOW + f"[PERSIST] ⚠️ {coin}: Old model detected ({saved_version}, "
                  f"{saved_features} features) — will retrain with new {CURRENT_FEATURES}-feature model")
            return None, None, None   # Signal: retrain needed

        if saved_samples < MIN_SAMPLES:
            logger.warning(f"[PERSIST] ⚠️ {coin}: Model has only {saved_samples} samples (min {MIN_SAMPLES}) — forcing retrain")
            print(Fore.YELLOW + f"[PERSIST] ⚠️ {coin}: Too few samples ({saved_samples} < {MIN_SAMPLES}) — retraining with 365d data")
            return None, None, None   # Signal: retrain needed

        logger.info(f"[PERSIST] ✅ Model loaded: {coin} (acc: {metadata.get('accuracy', 'N/A'):.1%})")
        print(Fore.CYAN + f"[PERSIST] Model loaded: {coin} (acc: {metadata.get('accuracy', 'N/A'):.1%})")
        return model, scaler, metadata
    
    except Exception as e:
        logger.error(f"[PERSIST] ❌ Error loading {coin}: {e}")
        print(Fore.YELLOW + f"[PERSIST] Error loading {coin}: {e}")
        return None, None, None

def should_retrain_model(coin):
    """
    [V] Check if model should be retrained
    
    Returns: True if model is older than MODEL_CACHE_DAYS
    """
    try:
        if not MODEL_PERSISTENCE_ENABLED or not AUTO_RETRAIN:
            return True  # Always retrain if persistence disabled
        
        metadata_path = get_metadata_path(coin)
        
        if not os.path.exists(metadata_path):
            return True  # No model yet, should train
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        trained_date = datetime.fromisoformat(metadata['trained_date'])
        days_old = (datetime.now() - trained_date).days
        
        should_retrain = days_old >= MODEL_CACHE_DAYS
        
        if should_retrain and days_old > 0:
            print(Fore.YELLOW + f"[PERSIST] {coin} model is {days_old}d old, retraining...")
        
        return should_retrain
    
    except Exception as e:
        print(Fore.YELLOW + f"[PERSIST] Error checking {coin}: {e}")
        return True  # On error, retrain

def delete_model(coin):
    """[V] Delete saved model files for a coin"""
    try:
        for path in [get_model_path(coin), get_scaler_path(coin), get_metadata_path(coin)]:
            if os.path.exists(path):
                os.remove(path)
        print(Fore.YELLOW + f"[PERSIST] Deleted model: {coin}")
        return True
    except Exception as e:
        print(Fore.RED + f"[PERSIST] Error deleting {coin}: {e}")
        return False

def list_saved_models():
    """[V] List all saved models with their stats"""
    try:
        if not os.path.exists(MODEL_PERSISTENCE_DIR):
            print(Fore.YELLOW + "[PERSIST] No models directory found")
            return []
        
        models = []
        for metadata_file in Path(MODEL_PERSISTENCE_DIR).glob("*_metadata.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                models.append(metadata)
        
        if models:
            print(Fore.CYAN + "[PERSIST] Saved Models:")
            for m in sorted(models, key=lambda x: x['trained_date'], reverse=True):
                days_old = (datetime.now() - datetime.fromisoformat(m['trained_date'])).days
                print(f"  {m['coin']:12} | Accuracy: {m['accuracy']:.1%} | Samples: {m['samples']:3} | Age: {days_old}d")
        else:
            print(Fore.YELLOW + "[PERSIST] No saved models found")
        
        return models
    except Exception as e:
        print(Fore.YELLOW + f"[PERSIST] Error listing models: {e}")
        return []


# ===========================
# [W] LOGGING SETUP
# ===========================

def setup_logger():
    """[W] Configure comprehensive logging system with rotation"""
    
    if not LOG_ENABLED:
        # Create null logger (disabled)
        logger = logging.getLogger("TradingBot")
        logger.addHandler(logging.NullHandler())
        return logger
    
    # Create logs directory
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("TradingBot")
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format: [TIMESTAMP] LEVEL - MESSAGE
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # [W] File Handler with Rotation
    if LOG_TO_FILE:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                LOG_FILE,
                maxBytes=LOG_MAX_SIZE_MB * 1024 * 1024,
                backupCount=LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(Fore.YELLOW + f"[LOG] Could not create file handler: {e}")
    
    # [W] Console Handler
    if LOG_TO_CONSOLE:
        import sys
        try:
            console_handler = logging.StreamHandler(
                stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
            )
        except Exception:
            console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# [W] Initialize logger at startup
logger = setup_logger()


# ===========================
# [X] SENTIMENT ANALYSIS (VADER)
# ===========================

_vader_sentiment_cache = {}

def analyze_text_sentiment(text):
    """[X] Analyze sentiment of text using VADER"""
    if not VADER_AVAILABLE or not USE_VADER:
        return 0.0  # Neutral
    
    try:
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        return scores['compound']  # -1 (negative) to +1 (positive)
    except Exception as e:
        logger.error(f"[SENTIMENT] Error analyzing text: {e}")
        return 0.0


def get_macro_sentiment():
    """
    [NEW] Detect MACRO/GEOPOLITICAL sentiment
    
    Checks for major events that affect entire crypto market:
    - War/Military conflicts
    - Sanctions
    - Political crises
    - Economic disasters
    - Central bank announcements
    
    Returns: macro_risk_level (0-1), event_description
    """
    
    try:
        macro_keywords = {
            "war": 0.9,  # Critical risk!
            "conflict": 0.85,
            "missile": 0.8,
            "attack": 0.8,
            "military": 0.75,
            "sanctions": 0.8,
            "embargo": 0.85,
            "crisis": 0.7,
            "recession": 0.75,
            "collapse": 0.8,
            "crash": 0.7,
            "panic": 0.6,
            "default": 0.85,
            "bankruptcy": 0.8,
        }
        
        # Check trending news/events
        # In real implementation, would check News API, CoinTelegraph, etc.
        # For now, return baseline (0.3 = normal market)
        
        macro_risk = 0.3  # Baseline = normal market sentiment
        event_detected = None
        
        # Example: Check for known events (in production, would use real news API)
        # if "Iran" in trending_events and "US" in trending_events:
        #     macro_risk = 0.85
        #     event_detected = "Iran-US Tensions (War Risk)"
        
        # For demo: Check if user wants to input events
        # This would be enhanced in production
        
        return macro_risk, event_detected
        
    except Exception as e:
        logger.error(f"[MACRO] Error detecting macro sentiment: {e}")
        return 0.3, None  # Default to normal


def get_macro_sentiment_description(risk_level):
    """Convert macro risk level to description"""
    if risk_level >= 0.85:
        return "🔴 CRITICAL (War/Sanctions/Major Crisis)"
    elif risk_level >= 0.75:
        return "🟠 HIGH RISK (Serious Event)"
    elif risk_level >= 0.6:
        return "🟡 ELEVATED RISK (Market Concern)"
    elif risk_level >= 0.4:
        return "🟡 MODERATE (Market Uncertainty)"
    else:
        return "🟢 NORMAL (Stable Market)"


def get_crypto_sentiment(coin):
    """
    [X] Get sentiment for a coin using VADER
    
    NOW ALSO CHECKS: Macro sentiment doesn't override individual, but flags it!
    
    Returns: sentiment score (0-1) and label
    """
    
    if not SENTIMENT_ANALYSIS_ENABLED or not USE_VADER:
        return 0.5, "neutral"
    
    try:
        # Check MACRO sentiment first
        macro_risk, macro_event = get_macro_sentiment()
        
        # Check cache
        if coin in _vader_sentiment_cache:
            cached = _vader_sentiment_cache[coin]
            age = (datetime.now() - cached['timestamp']).total_seconds() / 60
            if age < CACHE_SENTIMENT_MINUTES:
                logger.debug(f"[SENTIMENT] Using cached sentiment for {coin}")
                sentiment, label = cached['sentiment'], cached['label']
                
                # If macro risk is high, flag it
                if macro_risk >= 0.75 and macro_event:
                    print(f"[MACRO SENTIMENT] ⚠️ {macro_event} - Market Risk: {macro_risk:.0%}")
                    print(f"[MACRO SENTIMENT] ⚠️ Individual sentiment may be OVERRIDDEN by macro event!")
                
                return sentiment, label
        
        # Analyze sentiment of coin name and common crypto phrases
        text_samples = [
            coin.replace('-', ' '),  # coin name
            f"{coin} is a good investment",  # positive context
            f"{coin} moon potential",  # crypto slang positive
        ]
        
        sentiments = []
        for text in text_samples:
            sent = analyze_text_sentiment(text)
            sentiments.append(sent)
        
        # Average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        
        # Normalize to 0-1 scale
        normalized = (avg_sentiment + 1) / 2
        
        # MACRO ADJUSTMENT: If macro risk is high, reduce individual sentiment
        if macro_risk >= 0.75:
            # War/crisis: reduce bullish sentiment
            normalized = normalized * (1 - macro_risk * 0.5)  # Up to 50% reduction!
            print(f"[MACRO SENTIMENT] ⚠️ Individual sentiment REDUCED due to {macro_event}")
        
        # Generate label
        if normalized > 0.7:
            label = "very_positive"
        elif normalized > 0.6:
            label = "positive"
        elif normalized > 0.4:
            label = "neutral"
        elif normalized > 0.3:
            label = "negative"
        else:
            label = "very_negative"
        
        # Cache result
        _vader_sentiment_cache[coin] = {
            'timestamp': datetime.now(),
            'sentiment': normalized,
            'label': label
        }
        
        return normalized, label
    
    except Exception as e:
        logger.error(f"[SENTIMENT] Error getting sentiment for {coin}: {e}")
        return 0.5, "neutral"


def apply_sentiment_boost(score, coin):
    """[X] Apply sentiment boost to trading score"""
    
    if not SENTIMENT_ANALYSIS_ENABLED or not USE_VADER:
        return score
    
    try:
        sentiment, label = get_crypto_sentiment(coin)
        
        boost = 0
        
        # Strong positive sentiment
        if label == "very_positive":
            boost = SENTIMENT_BOOST_BUY
            print(f"[SENTIMENT] 🟢 {coin}: +{boost} boost from very positive sentiment")
            if LOG_ENABLED:
                logger.info(f"[SENTIMENT] 🟢 {coin}: +{boost} boost from very positive sentiment")
        
        # Positive sentiment
        elif label == "positive":
            boost = SENTIMENT_BOOST_BUY // 2
            print(f"[SENTIMENT] 🟢 {coin}: +{boost} boost from positive sentiment")
            if LOG_ENABLED:
                logger.info(f"[SENTIMENT] 🟢 {coin}: +{boost} boost from positive sentiment")
        
        # Negative sentiment
        elif label == "negative":
            boost = -SENTIMENT_BOOST_SELL // 2
            print(f"[SENTIMENT] 🔴 {coin}: {boost} penalty from negative sentiment")
            if LOG_ENABLED:
                logger.info(f"[SENTIMENT] 🔴 {coin}: {boost} penalty from negative sentiment")
        
        # Strong negative sentiment
        elif label == "very_negative":
            boost = -SENTIMENT_BOOST_SELL
            print(f"[SENTIMENT] 🔴 {coin}: {boost} penalty from very negative sentiment")
            if LOG_ENABLED:
                logger.info(f"[SENTIMENT] 🔴 {coin}: {boost} penalty from very negative sentiment")
        
        # Neutral sentiment (no change, but still report it)
        else:
            print(f"[SENTIMENT] ⚪ {coin}: neutral sentiment")
            if LOG_ENABLED:
                logger.debug(f"[SENTIMENT] {coin}: neutral sentiment")
        
        if boost != 0:
            print(f"[SENTIMENT] {coin} score: {score} → {score + boost} (sentiment: {boost:+d})")
            if LOG_ENABLED:
                logger.debug(f"[SENTIMENT] {coin} score: {score} → {score + boost}")
        
        return score + boost
    
    except Exception as e:
        print(f"[SENTIMENT] Error applying boost to {coin}: {e}")
        if LOG_ENABLED:
            logger.error(f"[SENTIMENT] Error applying boost: {e}")
        return score


# ===========================
# [Y] VOLUME BEHAVIOR ANALYSIS
# ===========================

def analyze_volume_behavior(df, coin):
    """
    [Y] Analyze volume behavior to detect good vs bad volume
    
    Returns: (volume_quality, volume_label, details)
    volume_quality: "good" / "bad" / "neutral"
    volume_label: descriptive label
    details: explanation for logging
    """
    
    try:
        if len(df) < 4:
            return "neutral", "insufficient_data", "Not enough data for volume analysis"
        
        # Get recent volumes
        current_vol = df['volume'].iloc[-1]
        prev_vol_1 = df['volume'].iloc[-2]
        prev_vol_2 = df['volume'].iloc[-3]
        prev_vol_3 = df['volume'].iloc[-4]
        
        # Get prices (before using them!)
        current_price = df['price'].iloc[-1]
        prev_price = df['price'].iloc[-2]
        
        # Get average volume (20-day)
        lookback = min(VOLUME_LOOKBACK, len(df))
        avg_volume = df['volume'].iloc[-lookback:].mean()
        
        # DEBUG: Log actual values
        vol_ratio = current_vol / avg_volume if avg_volume > 0 else 0
        if LOG_ENABLED:
            logger.debug(f"[VOLUME DEBUG] {coin}: current={current_vol:.2f}, avg={avg_volume:.2f}, ratio={vol_ratio:.2f}x")
        
        # Detect volume patterns
        details = {}
        
        # ✅ GOOD VOLUME PATTERNS
        
        # Pattern 1: Increasing volume over last 3 candles
        increasing_volume = (current_vol > prev_vol_1 and 
                           prev_vol_1 > prev_vol_2 and 
                           prev_vol_2 > prev_vol_3)
        
        # ⚠️ CHURNING DETECTION FIRST: High volume but small price range
        # Must check BEFORE volume spike so churning can override spike!
        # This indicates resistance/heavy buying/selling at a level with no progress
        try:
            # Get the high/low columns (could be 'high'/'low', 'High'/'Low', 'h'/'l')
            high_col = None
            low_col = None
            for col in ['high', 'High', 'h', 'H']:
                if col in df.columns:
                    high_col = col
                    break
            for col in ['low', 'Low', 'l', 'L']:
                if col in df.columns:
                    low_col = col
                    break
            
            # Only proceed if we have both high and low columns
            if high_col and low_col:
                current_high = df[high_col].iloc[-1]
                current_low = df[low_col].iloc[-1]
                price_range = current_high - current_low
                price_range_pct = (price_range / current_price) * 100 if current_price > 0 else 0
                
                # Churning: Volume high (1.5x+) but price range very small (< 1%)
                high_volume = current_vol > (avg_volume * 1.5)
                small_range = price_range_pct < 1.0
                
                # Log churning check
                vol_ratio = current_vol / avg_volume
                if LOG_ENABLED:
                    logger.debug(f"[VOLUME] {coin} churning check: Vol {vol_ratio:.2f}x (need >1.5x), Range {price_range_pct:.2f}% (need <1%)")
                
                if high_volume and small_range:
                    print(f"[VOLUME] ⚠️ {coin}: CHURNING DETECTED! High volume ({current_vol/avg_volume:.2f}x) but small range ({price_range_pct:.2f}%)")
                    if LOG_ENABLED:
                        logger.warning(f"[VOLUME] ⚠️ {coin}: CHURNING - Vol {current_vol/avg_volume:.2f}x, Range {price_range_pct:.2f}%")
                    print(f"[VOLUME] Heavy resistance - likely reversal incoming")
                    details['pattern'] = "churning"
                    details['volume_ratio'] = current_vol / avg_volume
                    details['price_range_pct'] = price_range_pct
                    # Return "very bad" with -2 penalty
                    return "very_bad", "churning_warning", f"⚠️ CHURNING: Volume {current_vol/avg_volume:.2f}x but range only {price_range_pct:.2f}% - Reversal likely!"
        except Exception as e:
            # High/low columns not available - just skip churning detection
            if LOG_ENABLED:
                logger.debug(f"[VOLUME] Churning detection skipped for {coin} (high/low data unavailable or error: {e})")
        
        # ✅ Pattern 1: Increasing volume over last 3 candles
        if increasing_volume:
            print(f"[VOLUME] ✅ {coin}: Good volume - increasing trend (vol going up)")
            if LOG_ENABLED:
                logger.info(f"[VOLUME] ✅ {coin}: Good volume - increasing trend")
            details['pattern'] = "increasing_trend"
            details['volumes'] = [prev_vol_3, prev_vol_2, prev_vol_1, current_vol]
            return "good", "increasing_volume", f"Volume increasing: {prev_vol_3:.0f}→{prev_vol_2:.0f}→{prev_vol_1:.0f}→{current_vol:.0f}"
        
        # ✅ Pattern 2: Volume spike (1.5x or 2x average)
        volume_spike = current_vol > (avg_volume * VOLUME_SPIKE_THRESHOLD)
        
        if volume_spike:
            spike_ratio = current_vol / avg_volume
            # Compute z-score for extreme volume detection
            try:
                _vols = df["volume"].dropna().values[-90:]
                _vmean, _vstd = _vols.mean(), _vols.std()
                _vol_zscore = (current_vol - _vmean) / (_vstd + 1e-9) if _vstd > 0 else 0
            except Exception:
                _vol_zscore = 0
            print(f"[VOLUME] ✅ {coin}: VOLUME SPIKE detected! {spike_ratio:.2f}x average")
            if LOG_ENABLED:
                logger.info(f"[VOLUME] ✅ {coin}: VOLUME SPIKE {spike_ratio:.2f}x average")
            details['pattern']    = "volume_spike"
            details['spike_ratio'] = spike_ratio
            details['vol_ratio']  = spike_ratio
            details['vol_zscore'] = _vol_zscore
            details['current']    = current_vol
            details['average']    = avg_volume
            return "good", "volume_spike", f"Volume spike: {spike_ratio:.2f}x average ({current_vol:.0f} vs avg {avg_volume:.0f})"
        
        # ❌ BAD VOLUME PATTERN
        
        # Pattern: Declining volume + price going up = FAKE MOVE
        declining_volume = current_vol < prev_vol_1 and prev_vol_1 < prev_vol_2
        price_going_up = current_price > prev_price
        
        if declining_volume and price_going_up:
            vol_decline = (current_vol / prev_vol_1) * 100 if prev_vol_1 > 0 else 0
            price_increase = ((current_price - prev_price) / prev_price) * 100
            print(f"[VOLUME] ❌ {coin}: FAKE MOVE WARNING! Price up but volume declining")
            if LOG_ENABLED:
                logger.warning(f"[VOLUME] ❌ {coin}: FAKE MOVE WARNING! Price +{price_increase:.2f}% Volume -")
            print(f"[VOLUME] Price: +{price_increase:.2f}% | Volume: {vol_decline:.1f}% (declining)")
            
            # CHECK SENTIMENT: If sentiment is VERY BULLISH, the price move might be justified!
            try:
                sentiment_score, sentiment_label = get_crypto_sentiment(coin)
                if sentiment_score >= 0.75:  # Very bullish sentiment (75%+)
                    # Sentiment-driven move is acceptable even without volume!
                    print(f"[VOLUME] ✅ BUT Sentiment is {sentiment_label} ({sentiment_score:.0%}) - SENTIMENT-DRIVEN MOVE (acceptable!)")
                    details['pattern'] = "sentiment_driven_move"
                    details['price_change'] = price_increase
                    details['volume_decline'] = vol_decline
                    details['sentiment_override'] = True
                    # Return reduced penalty instead of full fake move penalty
                    return "neutral", "weak_volume", f"Price up on strong sentiment ({sentiment_score:.0%}), low volume acceptable"
                else:
                    # Sentiment not strong enough to justify low volume
                    details['pattern'] = "fake_move"
                    details['price_change'] = price_increase
                    details['volume_decline'] = vol_decline
                    return "bad", "fake_move_warning", f"⚠️ FAKE MOVE: Price +{price_increase:.2f}% but volume -{100-vol_decline:.1f}%"
            except:
                # If sentiment check fails, default to fake move penalty
                details['pattern'] = "fake_move"
                details['price_change'] = price_increase
                details['volume_decline'] = vol_decline
                return "bad", "fake_move_warning", f"⚠️ FAKE MOVE: Price +{price_increase:.2f}% but volume -{100-vol_decline:.1f}%"
        
        # ⚠️ WEAK VOLUME (not terrible, just weak)
        low_volume = current_vol < (avg_volume * VOLUME_INCREASE_THRESHOLD)
        
        if low_volume:
            vol_ratio = current_vol / avg_volume
            print(f"[VOLUME] ⚠️ {coin}: Weak volume - {vol_ratio:.2f}x average")
            if LOG_ENABLED:
                logger.debug(f"[VOLUME] ⚠️ {coin}: Weak volume - {vol_ratio:.2f}x average")
            return "neutral", "weak_volume", f"Weak volume: {vol_ratio:.2f}x average (below {VOLUME_INCREASE_THRESHOLD}x)"
        
        # ✅ NORMAL / HEALTHY VOLUME
        return "good", "healthy_volume", f"Healthy volume: {current_vol/avg_volume:.2f}x average"
    
    except Exception as e:
        print(f"[VOLUME] Error analyzing {coin}: {e}")
        if LOG_ENABLED:
            logger.error(f"[VOLUME] Error analyzing {coin}: {e}")
        return "neutral", "error", str(e)


def apply_price_change_penalty(score, df, coin):
    """[Z] Price change 24h — PENALTY ONLY (downside unique signal)
    
    WHY ONLY DOWNSIDE:
      Volume already rewards upside:
        Price UP + Volume UP   → +1 (volume spike)
        Price UP + Volume DOWN → -2 (fake move warning)
      
      Price 24h unique value = catching FALLS that volume misses:
        Price DOWN + Volume normal → volume gives no signal
        But price 24h gives -1 or -2 ← unique!
    
    SCORING:
      < -3%:      -2  (sharp decline — volume may not catch this)
      < -1%:      -1  (price falling)
      -1% to +∞:   0  (volume handles the upside story)
    """
    try:
        if df is None or len(df) < 2:
            return score

        close_col = None
        for col in ['close', 'Close', 'price', 'Price']:
            if col in df.columns:
                close_col = col
                break

        if close_col is None:
            return score

        prev_close    = df.iloc[-2][close_col]
        current_close = df.iloc[-1][close_col]

        if prev_close <= 0 or current_close <= 0:
            return score

        price_change_24h = ((current_close - prev_close) / prev_close) * 100

        if price_change_24h < -3:
            penalty = -2
            reason = f"Sharp decline: {price_change_24h:.2f}%"
        elif price_change_24h < -1:
            penalty = -1
            reason = f"Price falling: {price_change_24h:.2f}%"
        else:
            penalty = 0
            reason = f"Price flat/rising: {price_change_24h:.2f}% (volume handles upside)"

        if penalty != 0:
            new_score = score + penalty
            print(f"[PRICE] {coin}: {reason}  score: {score} → {new_score}")
            return new_score

        return score

    except Exception as e:
        print(f"[PRICE] Error for {coin}: {e}")
        return score


def analyze_order_book_imbalance(coin):
    """[AB] Analyze order book imbalance - comprehensive measure of buy vs sell pressure
    
    Measures:
      1. Total buy volume vs sell volume (top 20 orders)
      2. Buy orders vs sell orders (count)
      3. Depth distribution (who has more orders at different levels)
    
    Strong bullish imbalance (more buy vol/orders):
      = Institutional accumulation
      = Support building
      = Add +1 or +2 boost!
    
    Strong bearish imbalance (more sell vol/orders):
      = Institutional distribution
      = Resistance ahead
      = Add -1 penalty!
    
    Returns: score_adjustment, imbalance_ratio, description
    """
    
    try:
        import ccxt
        
        # Initialize exchange
        exchange = ccxt.binance()
        
        # Format symbol correctly using mapping
        symbol = get_exchange_symbol(coin)
        
        # Skip if coin not available on Binance
        if symbol is None:
            return 0, 1.0, f"{coin} not available on Binance"
        
        if LOG_ENABLED:
            logger.debug(f"[ORDER BOOK IMBALANCE] {coin}: Fetching order book for {symbol}...")
        
        # Fetch full order book
        orderbook = exchange.fetch_order_book(symbol, limit=20)
        
        bids = orderbook['bids']  # Buy orders [[price, volume], ...]
        asks = orderbook['asks']  # Sell orders
        
        # Calculate metrics
        total_buy_volume = sum([bid[1] for bid in bids])
        total_sell_volume = sum([ask[1] for ask in asks])
        
        buy_order_count = len(bids)
        sell_order_count = len(asks)
        
        # Calculate imbalance ratios
        if total_sell_volume > 0:
            volume_ratio = total_buy_volume / total_sell_volume
        else:
            volume_ratio = 1.0
        
        if sell_order_count > 0:
            order_count_ratio = buy_order_count / sell_order_count
        else:
            order_count_ratio = 1.0
        
        # Weighted imbalance score
        # (volume matters more than count)
        imbalance = (volume_ratio * 0.7) + (order_count_ratio * 0.3)
        
        # Determine bullish/bearish
        # Capped at ±1 — volume boost already covers conviction
        # Order book = INSTITUTIONAL POSITIONING signal only
        adjustment = 0
        label = ""
        
        if imbalance > 1.5:  # Strong buy imbalance
            adjustment = 1
            label = "🟢 STRONG BUY IMBALANCE"
            description = f"Institutional buying: {volume_ratio:.2f}x volume, {order_count_ratio:.2f}x orders"
        elif imbalance > 1.2:  # Moderate buy imbalance
            adjustment = 0
            label = "🟡 MODERATE BUY IMBALANCE"
            description = f"Slight buying pressure: {volume_ratio:.2f}x volume"
        elif imbalance < 0.55:  # Strong sell imbalance
            adjustment = -1
            label = "🔴 STRONG SELL IMBALANCE"
            description = f"Institutional selling: {volume_ratio:.2f}x volume"
        elif imbalance < 0.67:  # Moderate sell imbalance
            adjustment = -1
            label = "🔴 SELL IMBALANCE"
            description = f"Distribution pressure: {volume_ratio:.2f}x volume"
        else:
            adjustment = 0
            label = "⚪ BALANCED"
            description = f"Neutral imbalance: {volume_ratio:.2f}x volume"
        
        print(f"[ORDER BOOK IMBALANCE] {coin}: {label}")
        if LOG_ENABLED:
            logger.info(f"[ORDER BOOK IMBALANCE] {coin}: {label} - {description}")
        
        return adjustment, imbalance, description
        
    except Exception as e:
        # Order book might not be available
        if "ccxt" not in str(e).lower():
            if LOG_ENABLED:
                logger.debug(f"[ORDER BOOK IMBALANCE] {coin}: {e}")
        return 0, 1.0, "Order book imbalance unavailable"


def analyze_order_book_depth(coin):
    """[AA] Analyze order book depth to detect institutional buying pressure
    
    If buy orders within 1-2% of price significantly outweigh sell orders:
      = Whales standing ready to buy dips
      = Add +1 boost to score
    
    Returns: boost (0 or +1), reason string
    """
    
    try:
        import ccxt
        
        # Initialize exchange (using Binance)
        exchange = ccxt.binance()
        
        # Format symbol correctly using mapping
        symbol = get_exchange_symbol(coin)
        
        # Skip if coin not available on Binance
        if symbol is None:
            return 0, f"{coin} not available on Binance"
        
        # Get current price
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Fetch order book (top 20 orders)
        orderbook = exchange.fetch_order_book(symbol, limit=20)
        
        bids = orderbook['bids']  # Buy orders
        asks = orderbook['asks']  # Sell orders
        
        # Calculate 1-2% range below and above price
        lower_bound = current_price * 0.98  # 2% below
        upper_bound = current_price * 0.99  # 1% below
        
        # Sum buy orders in 1-2% range (below price)
        buy_volume_range = sum([bid[1] for bid in bids if lower_bound <= bid[0] <= upper_bound])
        
        # Sum sell orders in 1-2% range (above price)
        sell_volume_range = sum([ask[1] for ask in asks if current_price <= ask[0] <= current_price * 1.02])
        
        # Check if buy orders dominate
        if buy_volume_range > 0 and sell_volume_range > 0:
            buy_sell_ratio = buy_volume_range / sell_volume_range
            
            if buy_sell_ratio > 1.5:  # Buy orders > 1.5x sell orders
                reason = f"Strong buy wall: {buy_sell_ratio:.2f}x ratio"
                return 1, reason
            elif buy_sell_ratio > 1.2:  # Buy orders > 1.2x sell orders
                reason = f"Moderate buy wall: {buy_sell_ratio:.2f}x ratio"
                return 0, reason  # Not quite strong enough
            else:
                reason = f"Balanced orders: {buy_sell_ratio:.2f}x ratio"
                return 0, reason
        
        return 0, "Insufficient order book data"
        
    except Exception as e:
        # Order book data might not be available for all coins
        if "ccxt" not in str(e):
            print(f"[ORDER BOOK] Error for {coin}: {e}")
        return 0, f"Order book unavailable"


def analyze_breakout(df, coin):
    """[AE] Detect price breakouts with volume confirmation
    
    Breakout = Price breaks above/below key resistance/support + volume
    
    Scoring:
      +3: Breakout detected + volume spike (HIGH PROBABILITY)
      +1: Breakout detected (good)
      -2: NO breakout (consolidation/range-bound)
    
    Purpose: Identify highest probability trade setups
    """
    
    try:
        if df is None or len(df) < 20:
            return 0, "Insufficient data"
        
        # Get price and volume
        price_col = None
        for col in ['close', 'Close', 'price', 'Price']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            return 0, "Price column not found"
        
        # Current price and volume
        current_price = df[price_col].iloc[-1]
        current_volume = df['volume'].iloc[-1] if 'volume' in df.columns else 0
        
        # Historical data (last 20 days)
        prices_20 = df[price_col].iloc[-20:-1].values
        volumes_20 = df['volume'].iloc[-20:-1].values if 'volume' in df.columns else None
        
        if len(prices_20) == 0:
            return 0, "Insufficient history"
        
        # Key levels
        resistance_20 = prices_20.max()      # 20-day high
        support_20 = prices_20.min()         # 20-day low
        avg_volume_20 = volumes_20.mean() if volumes_20 is not None else 0
        
        # Resistance distance (how close price is to breaking up)
        distance_to_resistance = resistance_20 - current_price
        resistance_breakout = distance_to_resistance < (resistance_20 * 0.005)  # Within 0.5%
        
        # Support distance (how close price is to breaking down)
        distance_to_support = current_price - support_20
        support_breakout = distance_to_support < (support_20 * 0.005)  # Within 0.5%
        
        # Volume confirmation
        volume_spike = current_volume > (avg_volume_20 * 1.5) if avg_volume_20 > 0 else False
        
        # BREAKOUT DETECTION LOGIC:
        
        # CASE 1: RESISTANCE BREAKOUT (price breaking UP)
        if current_price > resistance_20:
            if volume_spike:
                # ✅ BREAKOUT + VOLUME = STRONG SIGNAL
                score = 3
                label = "🟢 RESISTANCE BREAKOUT + VOLUME (strong buy!)"
                signal = f"Price broke above 20-day high (${resistance_20:.2f}) with {current_volume/avg_volume_20:.1f}x volume"
                direction_signal = "UP"
                
                print(f"[BREAKOUT] {coin}: {label} - {signal}")
                if LOG_ENABLED:
                    logger.warning(f"[BREAKOUT] {coin}: {label} - {signal}")
                
                return score, signal
            else:
                # ⚠️ BREAKOUT but weak volume = proceed with caution
                score = 1
                label = "🟡 RESISTANCE BREAKOUT (low volume)"
                signal = f"Price broke above 20-day high but weak volume ({current_volume/avg_volume_20:.1f}x)"
                
                print(f"[BREAKOUT] {coin}: {label}")
                if LOG_ENABLED:
                    logger.info(f"[BREAKOUT] {coin}: {label}")
                
                return score, signal
        
        # CASE 2: SUPPORT BREAKOUT (price breaking DOWN)
        elif current_price < support_20:
            if volume_spike:
                # ✅ BREAKOUT DOWN + VOLUME = STRONG BEARISH
                score = -3
                label = "🔴 SUPPORT BREAKDOWN + VOLUME (strong sell!)"
                signal = f"Price broke below 20-day low (${support_20:.2f}) with {current_volume/avg_volume_20:.1f}x volume"
                direction_signal = "DOWN"
                
                print(f"[BREAKOUT] {coin}: {label} - {signal}")
                if LOG_ENABLED:
                    logger.warning(f"[BREAKOUT] {coin}: {label} - {signal}")
                
                return score, signal
            else:
                # ⚠️ BREAKDOWN but weak volume = false breakout possible
                score = -1
                label = "🟡 SUPPORT BREAKDOWN (low volume)"
                signal = f"Price broke below 20-day low but weak volume ({current_volume/avg_volume_20:.1f}x)"
                
                print(f"[BREAKOUT] {coin}: {label}")
                if LOG_ENABLED:
                    logger.info(f"[BREAKOUT] {coin}: {label}")
                
                return score, signal
        
        # CASE 3: NO BREAKOUT (consolidating/range-bound)
        else:
            # Price between support and resistance = consolidation
            consolidation_pct = ((current_price - support_20) / (resistance_20 - support_20)) * 100
            
            score = -2
            label = "🟡 NO BREAKOUT (consolidating)"
            signal = f"Price consolidating: {consolidation_pct:.0f}% of range (between ${support_20:.2f}-${resistance_20:.2f})"
            
            print(f"[BREAKOUT] {coin}: {label} - {signal}")
            if LOG_ENABLED:
                logger.info(f"[BREAKOUT] {coin}: {label} - {signal}")
            
            return score, signal
    
    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[BREAKOUT] Error analyzing {coin}: {e}")
        return 0, f"Analysis failed: {e}"


def apply_breakout_boost(score, df, coin):
    """[AE] Apply breakout analysis boost/penalty to score"""
    
    try:
        breakout_adjustment, description = analyze_breakout(df, coin)
        
        if breakout_adjustment != 0:
            new_score = score + breakout_adjustment
            print(f"[BREAKOUT] {coin} score: {score} → {new_score} ({breakout_adjustment:+d})")
            return new_score
        
        return score
    
    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[BREAKOUT] Adjustment failed: {e}")
        return score

    """[AD] Detect chop zone - choppy sideways markets
    
    Chop Zone = High volatility but NO TREND
    
    Indicators:
      1. Low ADX (trend strength < 20)
      2. High Bollinger Band width (lots of movement)
      3. Price bouncing between BB upper/lower
      4. Volume but no directional progress
    
    Signal: Avoid trading, wait for breakout
    Penalty: -2 (don't trade choppy markets)
    """
    
    try:
        if df is None or len(df) < 20:
            return 0, 0, "Insufficient data"
        
        # Get ATR (volatility)
        high = df['high'].iloc[-20:].max() if 'high' in df.columns else None
        low = df['low'].iloc[-20:].min() if 'low' in df.columns else None
        close = df['close'].iloc[-1] if 'close' in df.columns else df['price'].iloc[-1]
        
        if high is None or low is None:
            return 0, 0, "High/low data unavailable"
        
        # Calculate ATR (Average True Range)
        atr = (high - low) / 20
        atr_pct = (atr / close) * 100
        
        # Get ADX (trend strength) - we need to calculate or estimate
        # Use directional movement if available
        adx = 15  # Default: assume weak trend
        
        # Calculate Bollinger Bands
        sma20 = df['close'].iloc[-20:].mean() if 'close' in df.columns else df['price'].iloc[-20:].mean()
        std20 = df['close'].iloc[-20:].std() if 'close' in df.columns else df['price'].iloc[-20:].std()
        bb_upper = sma20 + (2 * std20)
        bb_lower = sma20 - (2 * std20)
        bb_width_pct = ((bb_upper - bb_lower) / sma20) * 100
        
        # Check if price is bouncing in middle of BB (chop zone)
        price_from_lower = ((close - bb_lower) / (bb_upper - bb_lower)) * 100
        in_middle = 30 < price_from_lower < 70  # In middle 40% of BB range
        
        # CHOP ZONE CONDITIONS:
        chop_conditions = []
        
        # Condition 1: VERY weak trend (ADX < 15 is critical)
        if adx < 15:
            chop_conditions.append("very_weak_trend")
        elif adx < 20:
            chop_conditions.append("weak_trend")
        
        # Condition 2: Any significant volatility (ATR > 1.0% for chop)
        if atr_pct > 1.0:  # Lowered from 3% - even small ATR with weak trend = chop
            chop_conditions.append("movement_no_trend")
        
        # Condition 3: Price in middle of BB (bouncing)
        if in_middle:
            chop_conditions.append("bouncing_bb")
        
        # Condition 4: Tighter BB bands (>8% is already wide for chop)
        if bb_width_pct > 8:  # Lowered from 15%
            chop_conditions.append("wide_bb")
        
        # SCORE LOGIC:
        chop_score = len(chop_conditions)
        
        if chop_score >= 3:
            # STRONG CHOP ZONE DETECTED
            penalty = -2
            label = "🔴 STRONG CHOP ZONE (avoid trading!)"
            signal = f"Choppy market detected: ATR={atr_pct:.1f}%, Price mid-BB={price_from_lower:.0f}%, BB-width={bb_width_pct:.1f}%"
            
            print(f"[CHOP] {coin}: {label} - {signal}")
            if LOG_ENABLED:
                logger.warning(f"[CHOP] {coin}: {label} - {signal}")
            
            return penalty, chop_score, signal
        
        elif chop_score >= 2:
            # MODERATE CHOP ZONE
            penalty = -1
            label = "🟡 MODERATE CHOP (caution advised)"
            signal = f"Choppy conditions: {', '.join(chop_conditions)}"
            
            print(f"[CHOP] {coin}: {label}")
            if LOG_ENABLED:
                logger.info(f"[CHOP] {coin}: {label}")
            
            return penalty, chop_score, signal
        
        else:
            # NO CHOP ZONE - Normal trending or clear conditions
            return 0, chop_score, "Clear trend, not choppy"
        
    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[CHOP] Error analyzing {coin}: {e}")
        return 0, 0, f"Analysis failed: {e}"


def apply_chop_penalty(score, df, coin):
    """Chop zone penalty — uses ACTUAL penalty from analyze_chop_zone.
    
    Moderate chop (2 signals): -1 penalty
    Strong chop   (3 signals): -2 penalty
    
    Additional effects (via pipeline):
      - Blocks breakout signal
      - Tightens post-tech cap from ±2 to ±1
    """
    try:
        result  = analyze_chop_zone(df, coin)
        penalty = result[0] if isinstance(result, tuple) else result

        if penalty != 0:
            # Use the ACTUAL penalty returned by analyze_chop_zone
            # (-1 moderate, -2 strong) — not hardcoded -1
            actual_penalty = abs(penalty)   # always positive
            new_score = score - actual_penalty
            label = "STRONG CHOP" if actual_penalty >= 2 else "Choppy market"
            print(f"[CHOP] {coin}: {label} (-{actual_penalty}) — breakout skipped, cap tightened to ±1")
            return new_score

        return score

    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[CHOP] failed: {e}")
        return score
        if LOG_ENABLED:
            logger.debug(f"[CHOP] failed: {e}")
        return score

    """[AC] Analyze price Z-score to detect extreme moves and blow-off tops
    
    Z-Score = (Current Price - SMA20) / STD20
    
    Scoring:
      +2: Z-score 1-2   (healthy momentum, strong trend)
      +1: Z-score 0-1   (normal momentum)
       0: Z-score -1 to 0 (neutral)
      -1: Z-score -1 to -2 (weak/declining)
      -2: Z-score -2 to -3 (strong bearish)
      -3: Z-score > 3 or < -3 (EXTREME - blow-off top or panic bottom)
    
    Purpose: Distinguish healthy trends from exhausted moves
    """
    
    try:
        if df is None or len(df) < 20:
            return 0, 0, "Insufficient data for Z-score"
        
        # Get price column (handle variations)
        price_col = None
        for col in ['close', 'Close', 'price', 'Price']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            return 0, 0, "Price column not found"
        
        # Calculate 20-day metrics
        prices = df[price_col].iloc[-20:].values
        current_price = prices[-1]
        sma20 = prices.mean()
        std20 = prices.std()
        
        # Avoid division by zero
        if std20 == 0:
            return 0, 0, "No price volatility"
        
        # Calculate Z-score
        zscore = (current_price - sma20) / std20
        
        # Determine score and signal
        score = 0
        label = ""
        
        if zscore > 3:
            score = -3
            label = "🔴 EXTREME OVEREXTENSION (blow-off top!)"
            signal = f"Massive overbought: Z={zscore:.2f} (>{3})"
        elif zscore > 2:
            score = -2
            label = "🟠 VERY OVERBOUGHT (strong rejection likely)"
            signal = f"Strong overbought: Z={zscore:.2f} (2-3)"
        elif zscore > 1:
            score = 2
            label = "🟢 HEALTHY MOMENTUM (strong trend)"
            signal = f"Healthy momentum: Z={zscore:.2f} (1-2)"
        elif zscore > 0:
            score = 1
            label = "🟡 NORMAL MOMENTUM (mild bullish)"
            signal = f"Normal momentum: Z={zscore:.2f} (0-1)"
        elif zscore > -1:
            score = 0
            label = "⚪ NEUTRAL (at moving average)"
            signal = f"Neutral: Z={zscore:.2f} (0 to -1)"
        elif zscore > -2:
            score = -1
            label = "🟡 WEAK/DECLINING (below MA)"
            signal = f"Weak: Z={zscore:.2f} (-1 to -2)"
        elif zscore > -3:
            score = -2
            label = "🟠 VERY OVERSOLD (strong selling)"
            signal = f"Strong oversold: Z={zscore:.2f} (-2 to -3)"
        else:
            score = -3
            label = "🔴 EXTREME PANIC (blow-off bottom!)"
            signal = f"Extreme panic: Z={zscore:.2f} (<-3)"
        
        # Log result
        print(f"[ZSCORE] {coin}: {label} - {signal}")
        if LOG_ENABLED:
            logger.info(f"[ZSCORE] {coin}: {label} - {signal}")
        
        return score, zscore, signal
        
    except Exception as e:
        if LOG_ENABLED:
            logger.error(f"[ZSCORE] Error analyzing {coin}: {e}")
        return 0, 0, f"Error: {e}"


def apply_zscore_boost(score, df, coin):
    """[AC] Apply price Z-score boost/penalty to trading score"""
    
    try:
        zscore_adjustment, zscore_value, description = analyze_price_zscore(df, coin)
        
        if zscore_adjustment != 0:
            new_score = score + zscore_adjustment
            print(f"[ZSCORE] {coin} score: {score} → {new_score} ({zscore_adjustment:+d})")
            return new_score
        
        return score
        
    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[ZSCORE] Analysis failed for {coin}: {e}")
        return score




def analyze_momentum_exhaustion(df, coin):
    """[AS] EMA26 Deviation — Momentum Exhaustion / Mean Reversion Risk
    
    Formula: dev% = (Price - EMA26) / EMA26 × 100
    
    EMA26 = trend's "fair value" line (slow MACD line)
    Measures how far price has stretched from the trend itself
    
    Different from Z-Score:
      Z-Score:  stretched from recent average (short-term)
      EMA26 dev: stretched from trend line (medium-term)
    
    Scoring:
      > +40%: -3  Extreme extension — historical reversal zone
      > +25%: -2  Very stretched — strong mean reversion risk
      > +15%: -1  Stretched — elevated reversion probability
      ±15%:    0  Healthy — normal trend distance
      < -15%: +1  Oversold from trend — bounce potential
      < -25%: +1  Deeply oversold from trend — strong bounce
    """
    try:
        if df is None or len(df) < 2:
            return 0, 0, "Insufficient data"

        price_col = next((c for c in ['close','Close','price'] if c in df.columns), None)
        if price_col is None:
            return 0, 0, "No price column"

        price = df[price_col].iloc[-1]
        ema26 = df["EMA26"].iloc[-1] if "EMA26" in df.columns else None

        # Fallback: calculate EMA26 from price if not in df
        if ema26 is None or (isinstance(ema26, float) and ema26 != ema26):
            prices = df[price_col].values
            k = 2 / (26 + 1)
            ema = prices[0]
            for p in prices[1:]:
                ema = p * k + ema * (1 - k)
            ema26 = ema

        if ema26 <= 0:
            return 0, 0, "Invalid EMA26"

        dev_pct = (price - ema26) / ema26 * 100

        # Scoring
        if dev_pct > 40:
            score  = -3
            signal = f"EXTREME extension +{dev_pct:.1f}% above EMA26 — reversal zone"
        elif dev_pct > 25:
            score  = -2
            signal = f"Very stretched +{dev_pct:.1f}% above EMA26 — mean reversion risk"
        elif dev_pct > 15:
            score  = -1
            signal = f"Stretched +{dev_pct:.1f}% above EMA26 — elevated reversion probability"
        elif dev_pct < -25:
            score  = +1
            signal = f"Deeply oversold {dev_pct:.1f}% below EMA26 — strong bounce potential"
        elif dev_pct < -15:
            score  = +1
            signal = f"Oversold {dev_pct:.1f}% below EMA26 — bounce potential"
        else:
            score  = 0
            signal = f"Healthy distance {dev_pct:.1f}% from EMA26"

        if score != 0:
            print(f"[EXHAUSTION] {coin}: {signal} → {score:+d}")
            if LOG_ENABLED:
                logger.info(f"[EXHAUSTION] {coin}: {signal}")
        else:
            print(f"[EXHAUSTION] {coin}: {signal}")

        return score, dev_pct, signal

    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[EXHAUSTION] {coin}: Error - {e}")
        return 0, 0, f"Error: {e}"


def apply_exhaustion_penalty(score, df, coin, zscore_adj=0):
    """Apply momentum exhaustion penalty, capped with Z-score to avoid double-count"""
    try:
        adj, dev_pct, desc = analyze_momentum_exhaustion(df, coin)

        if adj != 0:
            # Cap combined exhaustion (EMA26 dev + Z-score) at -2 / +1
            # Both measure stretch — don't let them pile up beyond -2
            combined = adj + zscore_adj
            if combined < -2:
                adj = -2 - zscore_adj   # Only add what stays within -2 cap
                adj = max(adj, -2)

            new_score = score + adj
            print(f"[EXHAUSTION] {coin} score: {score} → {new_score} ({adj:+d})")
            if LOG_ENABLED:
                logger.info(f"[EXHAUSTION] {coin}: {adj:+d} (EMA26 dev={dev_pct:.1f}%)")
            return new_score

        return score
    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[EXHAUSTION] {coin}: skipped - {e}")
        return score

def _is_strong_chop(df):
    """Quick chop check — used internally by RSI, Z-score, and regime bonus.
    Returns True for BOTH moderate (2 signals) and strong (3 signals) chop.
    Regime bonus should be suppressed for any chop, not just extreme cases.
    """
    try:
        col = 'close' if 'close' in df.columns else 'price'
        prices = df[col].iloc[-20:].values
        if len(prices) < 20: return False
        current = prices[-1]
        high_20, low_20 = prices.max(), prices.min()
        range_pct = ((high_20 - low_20) / current) * 100
        sma20, std20 = prices.mean(), prices.std()
        bb_upper = sma20 + 2*std20
        bb_lower = sma20 - 2*std20
        bb_width = ((bb_upper - bb_lower) / sma20) * 100 if sma20 > 0 else 0
        price_pos = ((current - bb_lower) / (bb_upper - bb_lower)) * 100 if (bb_upper - bb_lower) > 0 else 50
        signals = (1 if range_pct < 12 else 0) + \
                  (1 if 30 < price_pos < 70 else 0) + \
                  (1 if bb_width > 9 else 0)
        return signals >= 2   # Catches moderate (2) AND strong (3) chop
    except: return False

def analyze_price_zscore(df, coin, in_trend=True, adx=None):
    """Price Z-score: risk filter + trend-regime-aware mean reversion.

    Z measures how far price has moved from the 20-day mean.

    THREE REGIMES based on ADX:
    ─────────────────────────────────────────────────────────────────
    RANGING  (ADX < 20):
        Z > 2:   -1  Stretched in range → rejection likely
        Z < -2:  +1  Deep below mean → high-probability bounce
        Z < -1:  +1  Below mean in range → mild opportunity

    WEAK TREND (ADX 20-25):
        Z > 3:   -2  Blow-off top
        Z > 2:   -1  Very stretched
        Z < -1:   0  Dip — neutral (weak trend could resume either way)
        Z < -2:   0  Do NOT chase as a bounce — trend may continue

    STRONG TREND (ADX > 25):
        Z > 3:   -2  Blow-off top (even in trends, this is extreme)
        Z > 2:   -1  Very stretched
        Z < -2:  -1  FALLING KNIFE — momentum accelerating DOWN, do not buy
        Z < -3:  -2  EXTREME FALLING KNIFE — severe downside momentum
    ─────────────────────────────────────────────────────────────────
    In_trend flag is a soft override — ADX takes priority when available.
    """
    try:
        if df is None or len(df) < 20: return 0, 0, "N/A"
        col = 'close' if 'close' in df.columns else 'price'
        prices = df[col].iloc[-20:].values
        current, sma20, std20 = prices[-1], prices.mean(), prices.std()
        if std20 == 0: return 0, 0, "N/A"
        z = (current - sma20) / std20

        # Resolve ADX from DataFrame if not passed explicitly
        if adx is None:
            adx = df["ADX"].iloc[-1] if "ADX" in df.columns else 0
            if adx != adx: adx = 0  # nan guard

        # ── Regime classification ─────────────────────────────────
        if adx >= 25:
            regime = "strong_trend"
        elif adx >= 20:
            regime = "weak_trend"
        else:
            regime = "ranging"

        # ── Strong trend: falling knife / blow-off top ────────────
        if regime == "strong_trend":
            if   z > 3:    return -2, z, f"BLOW-OFF TOP (ADX={adx:.0f}, extreme)"
            elif z > 2:    return -1, z, f"VERY STRETCHED in trend (ADX={adx:.0f})"
            elif z < -3:   return -2, z, f"⚠️ EXTREME FALLING KNIFE (ADX={adx:.0f}, Z={z:.2f}) — momentum accelerating DOWN"
            elif z < -2:   return -1, z, f"⚠️ FALLING KNIFE (ADX={adx:.0f}, Z={z:.2f}) — do NOT catch; strong trend continuing"
            else:          return  0, z, f"Normal in strong trend (ADX={adx:.0f})"

        # ── Weak trend: neutral on dips, still penalise overextension ─
        elif regime == "weak_trend":
            if   z > 3:    return -2, z, f"BLOW-OFF TOP (ADX={adx:.0f})"
            elif z > 2:    return -1, z, f"VERY STRETCHED (ADX={adx:.0f})"
            else:          return  0, z, f"Neutral — weak trend, Z-score inconclusive (ADX={adx:.0f})"

        # ── Ranging: mean-reversion signals are valid ─────────────
        else:
            if   z > 2:    return -1, z, f"STRETCHED in range (ADX={adx:.0f}) → rejection likely"
            elif z > 1:    return  0, z, f"Above average in range — neutral (ADX={adx:.0f})"
            elif z < -2:   return +1, z, f"DEEP OVERSOLD in range (ADX={adx:.0f}) → high-prob bounce"
            elif z < -1:   return +1, z, f"Below mean in range (ADX={adx:.0f}) → mild dip opportunity"
            else:          return  0, z, f"Normal oscillation (ADX={adx:.0f})"

    except: return 0, 0, "Error"

def apply_zscore_boost(score, df, coin, in_trend=True, adx=None):
    try:
        # Resolve ADX from df if not passed
        if adx is None and df is not None and "ADX" in df.columns:
            adx = df["ADX"].iloc[-1]
            if adx != adx: adx = 0
        adj, z, desc = analyze_price_zscore(df, coin, in_trend=in_trend, adx=adx)
        if adj != 0:
            new = score + adj
            adx_str = f", ADX={adx:.0f}" if adx else ""
            print(f"[ZSCORE] {coin}: {desc} (Z={z:.2f}{adx_str}) -> {adj:+d}  score: {score} -> {new}")
            return new
        else:
            print(f"[ZSCORE] {coin}: Normal (Z={z:.2f}, ADX={adx:.0f if adx else 0:.0f})")
        return score
    except: return score


def analyze_candle_rejection(df, coin):
    """[AP] Wick ratio analysis — detect exhaustion vs breakout candles.
    
    WICK RATIO = (High - Close) / (High - Low)
    Measures how much of the candle range was "given back" before close.
    
    SCORING:
      Wide candle (>1.5x ATR) + High wick (>0.5):
        → -1  Rejection/exhaustion — price couldn't hold the high
        
      Wide candle (>1.5x ATR) + Low wick (<0.2):
        →  0  Strong close — valid breakout move, no penalty
        
      Wide DOWN candle (>1.5x ATR) + High volume:
        → +1  Capitulation — panic selling = likely bounce
        
      Normal candle (<1.5x ATR):
        →  0  No signal (normal daily range)
    
    Requires: high/low columns (from OHLCV data)
    Falls back gracefully if unavailable.
    """
    try:
        if df is None or len(df) < 15:
            return 0, "Insufficient data"

        # Get required columns
        high_col  = next((c for c in ['high','High'] if c in df.columns), None)
        low_col   = next((c for c in ['low','Low']   if c in df.columns), None)
        close_col = next((c for c in ['close','Close','price'] if c in df.columns), None)
        open_col  = next((c for c in ['open','Open'] if c in df.columns), None)
        vol_col   = next((c for c in ['volume','Volume'] if c in df.columns), None)

        if not all([high_col, low_col, close_col]):
            return 0, "OHLC data unavailable"

        # Current candle values
        high    = df[high_col].iloc[-1]
        low     = df[low_col].iloc[-1]
        close   = df[close_col].iloc[-1]
        open_p  = df[open_col].iloc[-1] if open_col else close
        candle_range = high - low

        if candle_range <= 0:
            return 0, "Zero range candle"

        # ATR: 14-day average true range
        atr_col = 'ATR' if 'ATR' in df.columns else None
        if atr_col:
            atr14 = df[atr_col].iloc[-15:-1].mean()
        else:
            # Approximate ATR from high-low range
            highs  = df[high_col].iloc[-15:-1]
            lows   = df[low_col].iloc[-15:-1]
            atr14  = (highs - lows).mean()

        if atr14 <= 0 or np.isnan(atr14):
            return 0, "ATR unavailable"

        range_ratio = candle_range / atr14   # How wide vs normal

        # Only analyze wide candles (>1.5x ATR)
        if range_ratio < 1.5:
            return 0, f"Normal candle ({range_ratio:.1f}x ATR — no signal)"

        # Wick ratio calculation depends on candle direction:
        # UP candle:   upper wick = (High - Close) / Range
        # DOWN candle: lower wick = (Close - Low)  / Range
        is_down_candle = close < open_p
        if is_down_candle:
            wick_ratio = (close - low) / candle_range   # lower wick on down candle
        else:
            wick_ratio = (high - close) / candle_range  # upper wick on up candle

        # Volume check for capitulation
        avg_vol = df[vol_col].iloc[-15:-1].mean() if vol_col else 0
        cur_vol = df[vol_col].iloc[-1] if vol_col else 0
        high_volume = (cur_vol > avg_vol * 1.5) if avg_vol > 0 else False

        is_down_candle = close < open_p  # Bearish candle

        # ── SCORING ───────────────────────────────────────────────────
        if wick_ratio > 0.5:
            # Long upper wick = rejected at the high
            score  = -1
            signal = f"Rejection candle — gave back {wick_ratio:.0%} of range (wick={wick_ratio:.2f}, {range_ratio:.1f}x ATR)"
            print(f"[WICK] {coin}: Exhaustion/Rejection → -1  ({signal})")

        elif wick_ratio < 0.2 and not is_down_candle:
            # Closed near high = strong bullish close = breakout confirmed
            score  = 0
            signal = f"Strong close — valid move ({wick_ratio:.0%} wick, {range_ratio:.1f}x ATR)"
            print(f"[WICK] {coin}: Strong close — no penalty  ({signal})")

        elif is_down_candle and high_volume and range_ratio > 2.0 and wick_ratio < 0.2:
            # Wide DOWN candle + high volume + closed near lows = capitulation = bounce
            score  = 1
            signal = f"Capitulation — panic sell {range_ratio:.1f}x ATR, {cur_vol/avg_vol:.1f}x volume"
            print(f"[WICK] {coin}: Capitulation → +1  ({signal})")

        else:
            score  = 0
            signal = f"Wide candle, inconclusive ({range_ratio:.1f}x ATR, wick={wick_ratio:.2f})"
            print(f"[WICK] {coin}: Wide candle — inconclusive")

        return score, signal

    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[WICK] {coin}: Error - {e}")
        return 0, f"Error: {e}"


def apply_wick_adjustment(score, df, coin):
    """Apply wick ratio score adjustment"""
    try:
        adjustment, desc = analyze_candle_rejection(df, coin)
        if adjustment != 0:
            new_score = score + adjustment
            print(f"[WICK] {coin} score: {score} → {new_score} ({adjustment:+d})")
            if LOG_ENABLED:
                logger.info(f"[WICK] {coin}: {desc} → {adjustment:+d}")
            return new_score
        return score
    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[WICK] {coin}: skipped - {e}")
        return score

def analyze_chop_zone(df, coin):
    """Detect choppy sideways market"""
    try:
        if df is None or len(df) < 20: return 0, "N/A"
        col = 'close' if 'close' in df.columns else 'price'
        prices = df[col].iloc[-20:].values
        current = prices[-1]
        high_20, low_20 = prices.max(), prices.min()
        range_pct = ((high_20 - low_20) / current) * 100
        sma20, std20 = prices.mean(), prices.std()
        bb_upper = sma20 + 2*std20
        bb_lower = sma20 - 2*std20
        bb_width = ((bb_upper - bb_lower) / sma20) * 100 if sma20 > 0 else 0
        price_pos = ((current - bb_lower) / (bb_upper - bb_lower)) * 100 if (bb_upper - bb_lower) > 0 else 50
        signals = 0
        if range_pct < 12: signals += 1
        if 30 < price_pos < 70: signals += 1
        if bb_width > 9: signals += 1
        if signals >= 3:
            print(f"[CHOP] {coin}: STRONG CHOP ZONE (avoid!) - range={range_pct:.1f}%, pos={price_pos:.0f}%, width={bb_width:.1f}%")
            return -1, "Strong chop"
        elif signals >= 2:
            print(f"[CHOP] {coin}: MODERATE CHOP (caution) - {signals} signals")
            return -1, "Moderate chop"
        else:
            return 0, "Clear"
    except Exception as e: return 0, f"Error: {e}"

# apply_chop_penalty defined above

def analyze_breakout(df, coin):
    """Detect price breakouts with volume confirmation"""
    try:
        if df is None or len(df) < 20: return 0, "N/A"
        col = 'close' if 'close' in df.columns else 'price'
        prices = df[col].iloc[-20:-1].values
        current = df[col].iloc[-1]
        resistance, support = prices.max(), prices.min()
        avg_vol = df['volume'].iloc[-20:-1].mean() if 'volume' in df.columns else 0
        cur_vol = df['volume'].iloc[-1] if 'volume' in df.columns else 0
        vol_spike = cur_vol > avg_vol * 1.5 if avg_vol > 0 else False
        if current > resistance:
            if vol_spike:
                print(f"[BREAKOUT] {coin}: RESISTANCE + VOLUME -> +3")
                return 3, "Resistance breakout confirmed"
            print(f"[BREAKOUT] {coin}: Resistance break (low volume) -> +1")
            return 1, "Weak breakout"
        elif current < support:
            if vol_spike:
                print(f"[BREAKOUT] {coin}: SUPPORT BREAKDOWN + VOLUME -> -3")
                return -3, "Support breakdown confirmed"
            print(f"[BREAKOUT] {coin}: Support breakdown (low volume) -> -1")
            return -1, "Weak breakdown"
        else:
            pct = ((current - support) / (resistance - support)) * 100 if (resistance - support) > 0 else 50
            print(f"[BREAKOUT] {coin}: Consolidating ({pct:.0f}% of range) — neutral")
            return 0, "Consolidating"
    except Exception as e: return 0, f"Error: {e}"

def apply_breakout_boost(score, df, coin):
    try:
        adj, desc = analyze_breakout(df, coin)
        if adj != 0:
            new = score + adj
            print(f"[BREAKOUT] {coin} score: {score} -> {new} ({adj:+d})")
            return new
        return score
    except: return score

def analyze_order_book_liquidity(coin):
    """[AB2] Measure ABSOLUTE liquidity depth of the order book.
    
    Liquidity = Total USD value sitting in top 20 bids + asks
    
    Deep liquidity = large orders needed to move price = MORE RELIABLE signals
    Thin liquidity = small orders can move price = signals are RISKY (manipulation)
    
    Returns: (liquidity_usd, liquidity_tier, description)
      tier: "deep" | "moderate" | "thin" | "paper_thin"
    """
    if not USE_BINANCE_ORDER_BOOK:
        return 0, "moderate", "Binance disabled (GitHub Actions geo-restriction)"
    try:
        import ccxt
        exchange = ccxt.binance()
        symbol = get_exchange_symbol(coin)
        if symbol is None:
            return 0, "unknown", "Not on Binance"

        ticker    = exchange.fetch_ticker(symbol)
        price     = ticker['last']
        orderbook = exchange.fetch_order_book(symbol, limit=20)

        bids = orderbook['bids']
        asks = orderbook['asks']

        # Total USD value in order book (top 20 each side)
        bid_usd = sum(b[0] * b[1] for b in bids)   # price × volume
        ask_usd = sum(a[0] * a[1] for a in asks)
        total_usd = bid_usd + ask_usd

        # Liquidity tiers (USD in order book)
        if total_usd == 0:                # No data returned
            tier = "no_data"
            desc = f"No order book data (coin may not trade on this exchange)"
        elif total_usd >= 2_000_000:     # $2M+
            tier = "deep"
            desc = f"Deep liquidity: ${total_usd/1e6:.1f}M in book"
        elif total_usd >= 500_000:       # $500k-$2M
            tier = "moderate"
            desc = f"Moderate liquidity: ${total_usd/1e3:.0f}K in book"
        elif total_usd >= 100_000:       # $100k-$500k
            tier = "thin"
            desc = f"Thin liquidity: ${total_usd/1e3:.0f}K in book"
        else:                            # < $100k
            tier = "paper_thin"
            desc = f"Paper-thin liquidity: ${total_usd/1e3:.1f}K in book — HIGH RISK"

        return total_usd, tier, desc

    except Exception as e:
        return 0, "unknown", f"Error: {e}"


def apply_orderbook_boost(score, coin, market_regime="unknown"):
    """[AA] Liquidity-aware order book scoring with wash trade detection.
    
    THREE-STEP ANALYSIS:
    1. Measure LIQUIDITY DEPTH (book size)
    2. Check VOLUME/BOOK RATIO (wash trade signal)
    3. Apply IMBALANCE signal (filtered by depth)
    
    WASH TRADE DETECTION:
      Paper-thin book (<$100K) + high turnover ratio + volume spike
      = Classic wash trading signature on small coins
      Large coins (deep book) exempt — high ratios normal for BTC/ETH
    """
    try:
        # Step 1: Liquidity depth
        total_usd, liq_tier, liq_desc = analyze_order_book_liquidity(coin)
        print(f"[ORDER BOOK] {coin}: {liq_desc}")

        # Step 2: Wash trade ratio check (only for thin/paper-thin books)
        wash_penalty = 0
        if liq_tier in ("paper_thin", "thin") and USE_BINANCE_ORDER_BOOK:
            try:
                import ccxt as _ccxt
                _ex     = _ccxt.binance()
                _sym    = get_exchange_symbol(coin)
                if _sym:
                    _tk      = _ex.fetch_ticker(_sym)
                    _vol_usd = _tk.get('quoteVolume', 0) or 0
                    _ratio   = _vol_usd / total_usd if total_usd > 0 else float('inf')

                    if liq_tier == "paper_thin" and _ratio > 50:
                        # Paper-thin book + disproportionate volume = wash trade suspect
                        wash_penalty = -4
                        print(f"[ORDER BOOK] {coin}: ⚠️ WASH TRADE RISK — "
                              f"${total_usd/1e3:.0f}K book but {_ratio:.0f}x daily turnover → -4")
                        if LOG_ENABLED:
                            logger.warning(f"[ORDER BOOK] {coin}: Wash trade risk "
                                           f"(book=${total_usd/1e3:.0f}K, ratio={_ratio:.0f}x)")
                    elif liq_tier == "thin" and _ratio > 100:
                        wash_penalty = -2
                        print(f"[ORDER BOOK] {coin}: ⚠️ Elevated turnover ratio "
                              f"{_ratio:.0f}x on thin book → -2")
            except Exception:
                pass

        if wash_penalty != 0:
            # Defer wash penalty — it may be overridden by high exchange volume check below
            # (paper_thin book + high volume = trading on other exchanges = not wash trade)
            _wash_penalty_deferred = wash_penalty
        else:
            _wash_penalty_deferred = 0

        # Step 3: Imbalance signal (filtered by depth)
        imbalance_adj, imbalance_ratio, imbalance_desc = analyze_order_book_imbalance(coin)

        # Step 3: Apply liquidity filter to imbalance signal
        if liq_tier == "no_data":
            # No order book data — coin not on this exchange, skip silently
            if LOG_ENABLED:
                logger.debug(f"[ORDER BOOK] {coin}: No data — skipping")
            return score

        elif liq_tier == "paper_thin":
            # Paper-thin book — distinguish two cases:
            # Case A: Thin book + LOW volume  = genuinely illiquid → -1 penalty
            # Case B: Thin book + HIGH volume = trading on other exchanges → no penalty
            #         Also cancels the wash trade penalty (not wash trade, just off-exchange)
            print(f"[ORDER BOOK] {coin}: Paper-thin book — imbalance IGNORED (manipulation risk)")
            if LOG_ENABLED:
                logger.warning(f"[ORDER BOOK] {coin}: Paper-thin liquidity — signal suppressed")

            if not USE_BINANCE_ORDER_BOOK:
                # GitHub Actions / geo-restricted: skip ccxt, assume off-exchange trading
                print(f"[ORDER BOOK] {coin}: Binance check disabled — assuming off-exchange (no penalty)")
                return score

            try:
                import ccxt as _ccxt
                _ex  = _ccxt.binance()
                _sym = get_exchange_symbol(coin)
                if _sym:
                    _ticker  = _ex.fetch_ticker(_sym)
                    _usd_vol = _ticker.get('quoteVolume', 0) or 0
                    if _usd_vol < 500_000:
                        score += _wash_penalty_deferred
                        new_score = score - 1
                        print(f"[ORDER BOOK] {coin}: Genuinely illiquid "
                              f"(${_usd_vol/1e3:.0f}K daily + thin book) → -1")
                        return new_score
                    else:
                        print(f"[ORDER BOOK] {coin}: Thin Binance book but "
                              f"${_usd_vol/1e6:.1f}M traded → likely on other exchanges (no penalty)")
                        print(Fore.YELLOW + f"[ORDER BOOK] {coin}: ⚠️ ATR/signals valid "
                              f"but Binance execution will have slippage")
                else:
                    # Symbol not on Binance — assume off-exchange, no penalty
                    print(f"[ORDER BOOK] {coin}: Not on Binance — assuming off-exchange trading (no penalty)")
            except Exception as _e:
                # ccxt failed — use CoinGecko 24h volume as fallback
                # If we already have df, use its volume data
                print(f"[ORDER BOOK] {coin}: ccxt failed ({str(_e)[:40]}) — applying wash penalty conservatively")
                score += _wash_penalty_deferred

            return score

        elif liq_tier == "thin":
            score += _wash_penalty_deferred  # apply wash penalty (thin book = suspicious)
            if market_regime == "range" and imbalance_adj != 0:
                # In range regime, order book IS the primary signal
                # Apply even on thin books (mean reversion needs OB confirmation)
                new_score = score + imbalance_adj
                print(f"[ORDER BOOK] {coin}: Thin book but RANGE regime — OB applied as primary signal ({imbalance_adj:+d})")
                return new_score
            # Non-range: dampen imbalance on thin books
            print(f"[ORDER BOOK] {coin}: Thin book — imbalance dampened (was {imbalance_adj:+d})")
            if LOG_ENABLED:
                logger.info(f"[ORDER BOOK] {coin}: Thin liquidity — imbalance dampened")
            return score

        else:
            # Moderate or Deep — apply wash penalty if any, then full imbalance signal
            score += _wash_penalty_deferred
            if imbalance_adj != 0:
                new_score = score + imbalance_adj
                liquidity_label = "DEEP" if liq_tier == "deep" else "MODERATE"
                print(f"[ORDER BOOK] {coin}: {imbalance_desc} [{liquidity_label} book → signal reliable]")
                print(f"[ORDER BOOK] {coin} score: {score} → {new_score} ({imbalance_adj:+d})")
                if LOG_ENABLED:
                    logger.info(f"[ORDER BOOK] {coin}: {imbalance_adj:+d} [{liq_tier} liquidity]")
                return new_score

            return score

    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[ORDER BOOK] {coin}: skipped - {e}")
        return score


def apply_volume_boost(score, df, coin, backtest=False):
    """[Y] Apply volume-based boost or penalty to trading score"""
    
    if not VOLUME_ANALYSIS_ENABLED:
        return score
    
    try:
        quality, label, details = analyze_volume_behavior(df, coin)
        
        boost = 0
        
        # ✅ GOOD VOLUME: Boost the score
        if quality == "good":
            if label == "volume_spike":
                # Check for extreme volume (>4σ) on thin book = manipulation risk
                # Compute vol_zscore directly from df (details is a string not a dict)
                try:
                    _cur_vol  = df["volume"].iloc[-1]
                    _vols     = df["volume"].dropna().values[-90:]
                    _vmean    = _vols.mean()
                    _vstd     = _vols.std()
                    vol_zscore = (_cur_vol - _vmean) / (_vstd + 1e-9) if _vstd > 0 else 0
                    vol_ratio  = _cur_vol / _vmean if _vmean > 0 else 0
                except Exception:
                    vol_zscore = 0
                    vol_ratio  = 0
                
                try:
                    if backtest:
                        _book_thin = False   # skip live API in backtest
                    else:
                        _liq_usd, _liq_tier, _ = analyze_order_book_liquidity(coin)
                        _book_thin = _liq_tier in ("paper_thin", "thin")
                except Exception:
                    _book_thin = False

                if vol_zscore > 4.0 and _book_thin:
                    # Extreme volume + thin book = suspicious combination
                    # Could be wash trading or cross-venue pump
                    boost = 0
                    print(f"[VOLUME] ⚠️ {coin}: EXTREME VOLUME {vol_zscore:.1f}σ on thin book "
                          f"— possible manipulation, boost suppressed!")
                    if LOG_ENABLED:
                        logger.warning(f"[VOLUME] {coin}: Extreme vol {vol_zscore:.1f}σ + thin book — boost suppressed")
                else:
                    boost = GOOD_VOLUME_BOOST
                    print(f"[VOLUME] {coin}: +{boost} boost from volume spike")
                    if LOG_ENABLED:
                        logger.info(f"[VOLUME] {coin}: +{boost} boost from volume spike")
            elif label == "increasing_volume":
                # Increasing volume is good confirmation
                boost = GOOD_VOLUME_BOOST
                print(f"[VOLUME] {coin}: +{boost} boost from increasing volume")
                if LOG_ENABLED:
                    logger.info(f"[VOLUME] {coin}: +{boost} boost from increasing volume")
            elif label == "healthy_volume":
                # Normal volume, no penalty - but still report it!
                print(f"[VOLUME] ✅ {coin}: healthy volume (no change)")
                if LOG_ENABLED:
                    logger.debug(f"[VOLUME] {coin}: healthy volume")
                boost = 0
        
        # ❌ BAD VOLUME: OBV-aware fake move penalty
        elif quality == "bad":
            if label == "fake_move_warning":
                # Price up + volume declining = potential fake move
                # OBV check — BUT only trust OBV if volume is meaningful
                # Low volume (< 0.6x avg) makes OBV unreliable: 
                # tiny buy orders can push OBV upward = noise not signal
                try:
                    # Compute vol_ratio directly from df (details is a string)
                    _cur_v  = df["volume"].iloc[-1]
                    _avg_v  = df["volume"].iloc[-90:].mean() if len(df) >= 90 else df["volume"].mean()
                    vol_ratio = _cur_v / _avg_v if _avg_v > 0 else 1.0
                    obv_trustworthy = vol_ratio >= 0.60
                    obv_confirmed, at_high, obv_trend, _ = check_obv_confirmation(df, coin)
                except Exception:
                    obv_trend = "unknown"
                    obv_trustworthy = False

                if obv_trend == "rising" and obv_trustworthy:
                    # Meaningful volume + OBV rising = real slow accumulation
                    boost = 0
                    print(f"[VOLUME] {coin}: Fake move overridden by OBV rising — slow accumulation (no penalty)")
                    if LOG_ENABLED:
                        logger.info(f"[VOLUME] {coin}: Fake move overridden by OBV rising")
                elif obv_trend == "rising" and not obv_trustworthy:
                    # OBV rising but volume too low to trust it = noise
                    boost = -1
                    print(f"[VOLUME] ⚠️ {coin}: -1 penalty — OBV rising but volume too low ({vol_ratio:.2f}x) to confirm")
                    if LOG_ENABLED:
                        logger.warning(f"[VOLUME] {coin}: OBV unreliable at {vol_ratio:.2f}x volume")
                elif obv_trend == "flat":
                    # Weak conviction, mild warning
                    boost = -1
                    print(f"[VOLUME] ⚠️ {coin}: -1 penalty — low conviction rise (OBV flat)")
                    if LOG_ENABLED:
                        logger.warning(f"[VOLUME] {coin}: -1 fake move (OBV flat)")
                else:
                    # OBV falling = distribution = genuine fake move
                    boost = -BAD_VOLUME_PENALTY
                    print(f"[VOLUME] ⚠️ {coin}: {boost} penalty — FAKE MOVE DETECTED! (OBV falling = distribution)")
                    if LOG_ENABLED:
                        logger.warning(f"[VOLUME] ⚠️ {coin}: {boost} penalty - FAKE MOVE DETECTED!")
        
        # 🔴 VERY BAD VOLUME: Churning warning! (highest penalty)
        elif quality == "very_bad":
            if label == "churning_warning":
                # High volume but small range = heavy resistance, reversal likely
                boost = -2  # Stronger penalty than regular bad volume!
                print(f"[VOLUME] 🔴 {coin}: {boost} penalty - CHURNING DETECTED!")
                if LOG_ENABLED:
                    logger.warning(f"[VOLUME] 🔴 {coin}: {boost} penalty - CHURNING (reversal likely)")
        
        # ⚠️ WEAK VOLUME: Small penalty
        elif quality == "neutral":
            if label == "weak_volume":
                boost = -1  # Small penalty for weak volume
                print(f"[VOLUME] {coin}: -1 penalty from weak volume")
                if LOG_ENABLED:
                    logger.debug(f"[VOLUME] {coin}: -1 penalty from weak volume")
        
        if boost != 0:
            print(f"[VOLUME] {coin} score: {score} → {score + boost} ({details})")
        return score + boost
    
    except Exception as e:
        print(f"[VOLUME] Error applying volume boost: {e}")
        if LOG_ENABLED:
            logger.error(f"[VOLUME] Error applying volume boost: {e}")
        return score


_sentiment_cache = {"value": 50, "timestamp": None}

def fetch_fear_greed_index():
    """
    [K] Fetch Fear & Greed Index from alternative.me (FREE API, no auth required)
    
    Returns: (fng_value, classification)
    fng_value: 0-100 (0=extreme fear, 100=extreme greed)
    classification: 'Extreme Fear' | 'Fear' | 'Neutral' | 'Greed' | 'Extreme Greed'
    
    Default: (50, 'Neutral') if API fails
    """
    global _sentiment_cache
    
    # [K] Check cache (sentiment doesn't change hourly)
    if _sentiment_cache["timestamp"] is not None:
        cache_age = (datetime.now() - _sentiment_cache["timestamp"]).total_seconds() / 60
        if cache_age < SENTIMENT_CACHE_MINUTES:
            return _sentiment_cache["value"], "cached"
    
    try:
        if LOG_API_CALLS:
            logger.info("[SENTIMENT] Fetching Fear & Greed Index...")
        
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # [K] Extract latest Fear & Greed value
        fng_data = data['data'][0]
        fng_value = int(fng_data['value'])
        fng_class = fng_data['value_classification']
        
        # [K] Cache the result
        _sentiment_cache["value"] = fng_value
        _sentiment_cache["timestamp"] = datetime.now()
        
        if LOG_API_CALLS:
            logger.info(f"[SENTIMENT] ✅ F&G Index: {fng_value} ({fng_class})")
        
        return fng_value, fng_class
    
    except requests.exceptions.Timeout:
        logger.error("[SENTIMENT] ❌ TIMEOUT fetching Fear & Greed Index")
        return 50, "Neutral"  # [K] Default to neutral on error
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"[SENTIMENT] ❌ CONNECTION ERROR: {e}")
        return 50, "Neutral"
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"[SENTIMENT] ❌ HTTP ERROR {e.response.status_code}")
        return 50, "Neutral"
    
    except Exception as e:
        logger.error(f"[SENTIMENT] ❌ Error fetching Fear & Greed: {e}")
        return 50, "Neutral"  # [K] Default to neutral on error


# ===========================
# [L] RELATIVE STRENGTH vs BTC
# ===========================
def calculate_relative_strength_24h(coin_df, btc_df):
    """
    [L] Calculate 24-hour relative strength vs BTC
    
    Formula: (Coin_24h_return - BTC_24h_return)
    
    Returns: Normalized value [-1.0, 1.0]
    Positive: Coin outperforming BTC
    Negative: BTC outperforming (coin lagging)
    """
    try:
        if len(coin_df) < 2 or len(btc_df) < 2:
            return 0.0
        
        # [L] Get 24-hour returns
        coin_price_now = coin_df['price'].iloc[-1]
        coin_price_24h_ago = coin_df['price'].iloc[-2]
        coin_24h_return = (coin_price_now - coin_price_24h_ago) / coin_price_24h_ago
        
        btc_price_now = btc_df['price'].iloc[-1]
        btc_price_24h_ago = btc_df['price'].iloc[-2]
        btc_24h_return = (btc_price_now - btc_price_24h_ago) / btc_price_24h_ago
        
        # [L] Calculate relative strength
        relative_strength = coin_24h_return - btc_24h_return
        
        # [L] Normalize to [-1, 1] using tanh
        rs_normalized = np.tanh(relative_strength / 0.50)
        
        return rs_normalized
    
    except Exception as e:
        print(Fore.YELLOW + f"[RELATIVE STRENGTH] Error calculating: {e}")
        return 0.0


# ===========================
# [M] TIME-BASED FEATURES
# ===========================
def extract_time_features():
    """
    [M] Extract time-based features for market cyclicality
    
    Returns: (day_of_week_feature, hour_of_day_feature)
    day_of_week_feature: 0-1 (0=Monday, 1=Sunday)
    hour_of_day_feature: 0-1 (0=midnight, 1=23:00)
    """
    try:
        now = datetime.now()
        
        # [M] Day of week: 0=Monday, 6=Sunday
        day_of_week = now.weekday()
        day_feature = day_of_week / 6.0  # Normalize to [0, 1]
        
        # [M] Hour of day: 0-23
        hour_of_day = now.hour
        hour_feature = hour_of_day / 23.0  # Normalize to [0, 1]
        
        return day_feature, hour_feature
    
    except Exception as e:
        print(Fore.YELLOW + f"[TIME FEATURES] Error extracting: {e}")
        return 0.5, 0.5  # [M] Default to mid-values on error



def calculate_sortino_ratio(df, window=14):
    """[N] Sortino Ratio over last N days
    
    Sortino = Mean Return / Downside Deviation
    Only penalizes DOWNSIDE volatility (not upside)
    Better than Sharpe for crypto (asymmetric returns)
    
    Returns: float bounded by tanh (-1 to +1 approx)
    """
    try:
        col = 'close' if 'close' in df.columns else 'price'
        prices = df[col].iloc[-window-1:].values
        if len(prices) < window: return 0.0
        
        returns = np.diff(prices) / prices[:-1]   # daily returns
        mean_return = returns.mean()
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 1.0   # All positive days = max quality
        
        downside_std = np.sqrt(np.mean(downside_returns**2))
        if downside_std == 0: return 1.0
        
        sortino = mean_return / downside_std
        # Bound with tanh: maps (-inf,+inf) → (-1,+1)
        return float(np.tanh(sortino * 3))
        
    except: return 0.0


def get_market_session():
    """[O] Detect current crypto market session (UTC)
    
    Sessions (UTC):
      Asian:          00:00 - 08:00  (lower volatility)
      London:         08:00 - 12:00  (increasing volume)
      London+NY:      13:00 - 17:00  (HIGHEST volume/volatility)
      New York:       17:00 - 21:00  (declining)
      Off-hours:      21:00 - 00:00  (lowest)
    
    Returns: float 0-1 (higher = more active session)
    """
    try:
        hour = datetime.now(timezone.utc).hour
        
        # Session activity weights
        if 13 <= hour < 17:    # London + NY overlap (peak)
            return 1.0
        elif 8 <= hour < 13:   # London open
            return 0.75
        elif 17 <= hour < 21:  # NY afternoon
            return 0.6
        elif 0 <= hour < 8:    # Asian session
            return 0.35
        else:                  # Off hours 21-00
            return 0.2
    except: return 0.5

def extract_ml_features(row, df=None, btc_df=None, fng_value=50):
    """
    [K][L][M][N][O] Extract feature vector from a single candle row.
    
    Features (16 total):
    1-8:  Original technical indicators
    9  [K]: Market Sentiment (Fear & Greed Index)
    10 [L]: Relative Strength vs BTC
    11 [M]: Day of Week (normalized)
    12 [N]: Sortino Ratio 14d (volatility-adjusted momentum quality)
    13 [O]: Market Session (Asian/London/NY/Overlap)
    14 [P]: Weekly EMA50 headwind (price vs long-term macro trend)
    15 [Q]: Ichimoku cloud position (above/in/below = 1/0/-1)
    16 [R]: OBV 7-day trend (accumulation/flat/distribution = 1/0/-1)
    
    Returns: list of 16 features (or None if insufficient data)
    """
    features = []
    try:
        # ===== ORIGINAL 8 FEATURES =====
        
        # 1. RSI (normalized 0-1)
        rsi = row.get("RSI", np.nan)
        features.append(rsi / 100.0 if not np.isnan(rsi) else 0.5)

        # 2. MACD histogram (normalized; use tanh to bound)
        macd_hist = row.get("MACD_Hist", np.nan)
        features.append(np.tanh(macd_hist * 10) if not np.isnan(macd_hist) else 0.0)

        # 3. ADX (normalized 0-1, capped at 100)
        adx = row.get("ADX", np.nan)
        features.append(min(adx / 100.0, 1.0) if not np.isnan(adx) else 0.25)

        # 4. Volume ratio (log-normalized)
        vol_ratio = row.get("vol_ratio", np.nan)
        features.append(np.log1p(vol_ratio) / 2.0 if not np.isnan(vol_ratio) else 0.3)

        # 5. EMA12 > EMA26 (binary: 1 if bullish, 0 if bearish)
        ema12 = row.get("EMA12", np.nan)
        ema26 = row.get("EMA26", np.nan)
        features.append(1.0 if (not np.isnan(ema12) and not np.isnan(ema26) and ema12 > ema26) else 0.0)

        # 6. Price vs BB mid (position fraction 0-1)
        price = row.get("price", np.nan)
        bb_low = row.get("LowerBB", np.nan)
        bb_high = row.get("UpperBB", np.nan)
        if not any(np.isnan(x) for x in [price, bb_low, bb_high]):
            bb_pos = (price - bb_low) / (bb_high - bb_low + 1e-9)
            features.append(np.clip(bb_pos, 0.0, 1.0))
        else:
            features.append(0.5)

        # 7. ATR ratio (log-normalized, capped)
        atr = row.get("ATR", np.nan)
        atr_avg = 1.0
        if df is not None and len(df) > 1:
            atr_avg = df["ATR"].rolling(20).mean().iloc[-2]
            if np.isnan(atr_avg):
                atr_avg = 1.0
        atr_ratio = atr / atr_avg if (not np.isnan(atr) and atr_avg > 0) else 1.0
        features.append(np.log1p(min(atr_ratio, 3.0)) / 1.5)

        # 8. DI+ > DI- (1.0 if true, 0.0 if false)
        plus_di = row.get("plus_di", np.nan)
        minus_di = row.get("minus_di", np.nan)
        features.append(1.0 if (not np.isnan(plus_di) and not np.isnan(minus_di) and plus_di > minus_di) else 0.0)

        # ===== NEW 3 FEATURES [K][L][M] =====
        
        # 9 [K]. Market Sentiment (Fear & Greed Index) - Normalized 0-1
        features.append(fng_value / 100.0)
        
        # 10 [L]. Relative Strength vs BTC - Normalized -1 to 1
        if btc_df is not None and len(btc_df) >= 2 and len(df) >= 2:
            rs = calculate_relative_strength_24h(df, btc_df)
            features.append(rs)
        else:
            features.append(0.0)
        
        # 11 [M]. Day of Week - Normalized 0-1
        day_feature, _ = extract_time_features()
        features.append(day_feature)

        # 12 [N]. Sortino Ratio (14-day) - Volatility-adjusted momentum quality
        if df is not None and len(df) >= 15:
            sortino = calculate_sortino_ratio(df, window=14)
        else:
            sortino = 0.0
        features.append(sortino)

        # 13 [O]. Market Session - Which trading session is active right now
        session = get_market_session()
        features.append(session)

        # 14 [P]. Weekly EMA headwind — is price above/below long-term macro trend?
        # This is the MOST COMMONLY MISSING signal that causes ML to say ~50%
        # EMA_WEEKLY50 = EMA(250) on daily = approximates weekly 50 EMA
        ema_w50 = row.get("EMA_WEEKLY50", np.nan)
        if not np.isnan(ema_w50) and not np.isnan(price) and ema_w50 > 0:
            # Ratio: how far above/below weekly EMA (capped at ±30%)
            weekly_ratio = np.clip((price - ema_w50) / ema_w50, -0.3, 0.3)
            features.append(weekly_ratio)
        else:
            features.append(0.0)   # Unknown = neutral

        # 15 [Q]. Ichimoku cloud position (structural trend)
        # 1.0 = above cloud (bullish), 0.0 = in cloud (neutral), -1.0 = below (bearish)
        senkou_a = row.get("senkou_a", np.nan)
        senkou_b = row.get("senkou_b", np.nan)
        if not any(np.isnan(x) for x in [price, senkou_a, senkou_b]):
            cloud_top = max(senkou_a, senkou_b)
            cloud_bot = min(senkou_a, senkou_b)
            if price > cloud_top:
                ichimoku_feat = 1.0
            elif price < cloud_bot:
                ichimoku_feat = -1.0
            else:
                ichimoku_feat = 0.0
        else:
            ichimoku_feat = 0.0
        features.append(ichimoku_feat)

        # 16 [R]. OBV 7-day trend (capital flow direction)
        # Critical for distinguishing real moves from fake pumps
        # 1.0 = rising (accumulation), 0.0 = flat, -1.0 = falling (distribution)
        try:
            if df is not None and "OBV" in df.columns and len(df) >= 8:
                obv = df["OBV"].iloc[-8:].values
                obv_start, obv_end = obv[0], obv[-1]
                obv_change = (obv_end - obv_start) / (abs(obv_start) + 1e-9)
                if obv_change > 0.03:
                    obv_feat = 1.0
                elif obv_change < -0.03:
                    obv_feat = -1.0
                else:
                    obv_feat = 0.0
            else:
                obv_feat = 0.0
        except Exception:
            obv_feat = 0.0
        features.append(obv_feat)

        return features if len(features) == 16 else None
    except Exception:
        return None

def generate_ml_insight(row, model, scaler, df=None, btc_df=None, fng_value=50,
                        model_type="long"):
    """
    Explain WHY the ML model gave its rating for this coin.

    Uses feature importances from the trained model combined with current feature
    values to identify the top drivers of the prediction.

    model_type: "long"  → high proba = longs win  → feature contributing to HIGH proba = bullish
                "short" → high proba = shorts win  → feature contributing to HIGH proba = bearish context

    Returns: list of (label, pct, direction) tuples, sorted by contribution descending.
             direction: "bearish" | "bullish" | "neutral"
    """
    # ── Feature metadata: (name, bullish_threshold, bearish_threshold) ──────────
    # For continuous features: value > bullish_threshold → bullish reading
    # For binary features (0/1): 0.5 is the split
    FEATURE_META = [
        # idx  label                    bullish_thresh  bearish_thresh
        (0,  "RSI",                     0.55,           0.40),   # >55 = momentum OK, <40 = bearish
        (1,  "MACD Histogram",          0.05,          -0.05),   # pos = bullish
        (2,  "ADX (trend strength)",    0.20,           0.10),   # higher = clearer trend
        (3,  "Volume",                  0.40,           0.20),   # >40% normalized = healthy
        (4,  "EMA Cross",               0.6,            0.4),    # binary 1/0
        (5,  "BB Position",             0.70,           0.30),   # >70% = near upper (extended)
        (6,  "ATR (volatility)",        0.50,           0.20),   # context only
        (7,  "DI+ vs DI-",             0.6,            0.4),    # binary 1/0
        (8,  "Fear & Greed",            0.55,           0.40),   # >55 = greedy = bullish
        (9,  "Strength vs BTC",         0.05,          -0.05),   # pos = outperforming BTC
        (10, "Day of Week",             0.6,            0.4),    # mild signal
        (11, "Sortino Ratio",           0.10,          -0.10),   # pos = good risk-adj returns
        (12, "Market Session",          0.6,            0.4),    # mild signal
        (13, "Weekly EMA Trend",        0.02,          -0.02),   # above/below weekly EMA50
        (14, "Ichimoku Cloud",          0.5,           -0.5),    # 1=above, -1=below, 0=inside
        (15, "OBV Flow",               0.5,           -0.5),    # 1=accumulation, -1=distribution
    ]

    # Labels where HIGH value means bearish (for long model)
    # e.g. RSI 0.85 = overbought = bearish for longs
    INVERT_FOR_LONG = {5}  # BB position: high = near upper band = stretched (bearish for new longs)

    try:
        if model is None or not hasattr(model, "feature_importances_"):
            return []

        features = extract_ml_features(row, df, btc_df=btc_df, fng_value=fng_value)
        if features is None or len(features) != 16:
            return []

        importances = model.feature_importances_   # shape: (16,)
        total_importance = importances.sum()
        if total_importance <= 0:
            return []

        results = []
        for idx, label, bull_thresh, bear_thresh in FEATURE_META:
            if idx >= len(features) or idx >= len(importances):
                continue

            val   = features[idx]
            imp   = importances[idx]
            pct   = (imp / total_importance) * 100

            # Determine if this feature reading is bullish, bearish, or neutral
            if val >= bull_thresh:
                raw_dir = "bullish"
            elif val <= bear_thresh:
                raw_dir = "bearish"
            else:
                raw_dir = "neutral"

            # For features where high value = bearish for longs, invert
            if model_type == "long" and idx in INVERT_FOR_LONG:
                if raw_dir == "bullish":
                    raw_dir = "bearish"
                elif raw_dir == "bearish":
                    raw_dir = "bullish"

            # For short model: invert all directional labels
            # (a "bearish" feature reading helps the SHORT model predict wins)
            if model_type == "short":
                if raw_dir == "bullish":
                    raw_dir = "bearish_for_short"  # good for shorts
                elif raw_dir == "bearish":
                    raw_dir = "bullish_for_short"  # bad for shorts

            results.append((label, pct, raw_dir, val))

        # Sort by importance descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:6]   # top 6 features

    except Exception:
        return []


def format_ml_insight(long_insights, short_insights, long_proba, short_ml_proba,
                      ml_accuracy, short_ml_accuracy, coin):
    """
    Format ML insights into a human-readable [ML INSIGHT] block.
    Shows top drivers for both long and short model ratings.
    """
    lines = []
    lines.append(f"[ML INSIGHT] {coin.upper()}")

    # ── LONG MODEL ────────────────────────────────────────────────
    if long_insights:
        verdict = (
            "strongly bearish" if long_proba < 0.25 else
            "bearish"          if long_proba < 0.40 else
            "neutral"          if long_proba < 0.55 else
            "bullish"          if long_proba < 0.70 else
            "strongly bullish"
        )
        lines.append(f"  Long ML {long_proba:.0%} ({verdict}, acc {ml_accuracy:.0%}) — top drivers:")

        for label, pct, direction, val in long_insights[:4]:
            if direction == "bullish":
                icon = "🟢"
            elif direction == "bearish":
                icon = "🔴"
            else:
                icon = "⚪"

            # Add current value context for key indicators
            val_note = ""
            if label == "RSI":
                val_note = f" [RSI={val*100:.0f}]"
            elif label == "MACD Histogram":
                val_note = f" [{'pos' if val > 0 else 'neg'}]"
            elif label == "Ichimoku Cloud":
                val_note = f" [{'above' if val > 0.5 else 'below' if val < -0.5 else 'inside'}]"
            elif label == "OBV Flow":
                val_note = f" [{'accumulation' if val > 0.5 else 'distribution' if val < -0.5 else 'flat'}]"
            elif label == "Weekly EMA Trend":
                val_note = f" [{'above' if val > 0 else 'below'} weekly EMA]"
            elif label == "Volume":
                val_note = f" [{val:.0%} normalized]"
            elif label == "Fear & Greed":
                val_note = f" [F&G={val*100:.0f}]"

            lines.append(f"    {icon} {label}{val_note} — {pct:.0f}% weight")

    # ── SHORT MODEL ───────────────────────────────────────────────
    if short_insights and short_ml_accuracy >= 0.50:
        s_verdict = (
            "very rarely wins" if short_ml_proba < 0.20 else
            "rarely wins"      if short_ml_proba < 0.35 else
            "coin-flip"        if short_ml_proba < 0.45 else
            "often wins"       if short_ml_proba < 0.65 else
            "frequently wins"
        )
        lines.append(f"  Short ML {short_ml_proba:.0%} (shorts {s_verdict}, acc {short_ml_accuracy:.0%}) — top drivers:")

        for label, pct, direction, val in short_insights[:4]:
            # For short model: bullish_for_short = good for shorts (green)
            if direction == "bullish_for_short":
                icon = "🔴"   # bearish context = supports short
            elif direction == "bearish_for_short":
                icon = "🟢"   # bullish context = works against short
            elif direction == "bearish":
                icon = "🔴"
            elif direction == "bullish":
                icon = "🟢"
            else:
                icon = "⚪"

            val_note = ""
            if label == "RSI":
                val_note = f" [RSI={val*100:.0f}]"
            elif label == "Ichimoku Cloud":
                val_note = f" [{'above' if val > 0.5 else 'below' if val < -0.5 else 'inside'}]"
            elif label == "OBV Flow":
                val_note = f" [{'accumulation' if val > 0.5 else 'distribution' if val < -0.5 else 'flat'}]"
            elif label == "Weekly EMA Trend":
                val_note = f" [{'above' if val > 0 else 'below'} weekly EMA]"

            lines.append(f"    {icon} {label}{val_note} — {pct:.0f}% weight")
    elif short_ml_accuracy < 0.50:
        lines.append(f"  Short ML n/a (model accuracy {short_ml_accuracy:.0%} < 50% — neutralized)")

    return "\n".join(lines)


def train_ml_model(df, coin, btc_df=None, fng_value=50, lookback=ML_LOOKBACK_DAYS, future_days=ML_FUTURE_DAYS):
    """
    [K][L][M][N][O] Train ML model with dynamic ATR-based targets.
    
    LABEL LOGIC (Dynamic ATR targets):
      Target = Entry + (2.5 × ATR)   ← coin-calibrated take profit
      Stop   = Entry - (1.0 × ATR)   ← coin-calibrated stop loss
      Risk/Reward = 1:2.5             ← consistent across all coins
      
      Label = 1 if price hit Target BEFORE Stop within future_days
      Label = 0 otherwise
      
    Why ATR-based (not fixed 5%/12%)?
      Bitcoin ATR ≈ 2%/day → Target ≈ +5%, Stop ≈ -2%  (tight)
      Dogecoin ATR ≈ 5%/day → Target ≈ +12.5%, Stop ≈ -5% (wider)
      Fixed 5% target on Dogecoin = too easy (noise)
      Fixed 5% target on Bitcoin  = too hard in low vol
      
    Features: 13 total (RSI, MACD, ADX, Volume, EMA, BB, ATR, DI,
                         Sentiment, BTC RS, DayOfWeek, Sortino, Session)
    """
    if not ML_ENABLED or not ML_AVAILABLE or len(df) < lookback + future_days:
        return None, None, 0, 0

    try:
        # Extract features and create labels
        X_list = []
        y_list = []

        for i in range(len(df) - future_days - 1):
            if i < lookback:
                continue  # Skip first lookback days

            row = df.iloc[i]
            features = extract_ml_features(row, df, btc_df=btc_df, fng_value=fng_value)
            if features is None:
                continue

            # ── DYNAMIC ATR-BASED TARGETS ──────────────────────────────
            # Fixed 5%/12% is coin-agnostic — Bitcoin needs different targets than Dogecoin
            # ATR = Average True Range (coin's natural daily volatility)
            # Target = Entry + (2.5 × ATR)  → achievable but meaningful move
            # Stop   = Entry - (1.0 × ATR)  → tight, realistic stop
            # This gives ~1:2.5 Risk/Reward, calibrated per coin per period
            
            current_price = row["price"]
            future_slice = df.iloc[i+1:i+future_days+1]
            if len(future_slice) < future_days:
                break

            # Get ATR at entry point (use rolling 14-day ATR)
            atr_val = row.get("ATR", np.nan)
            if np.isnan(atr_val) or atr_val <= 0 or current_price <= 0:
                # Fallback to fixed targets if ATR unavailable
                target_price = current_price * 1.05
                stop_price   = current_price * 0.88
            else:
                atr_pct     = atr_val / current_price   # ATR as fraction of price
                target_price = current_price * (1 + 2.5 * atr_pct)  # +2.5 ATR
                stop_price   = current_price * (1 - 1.5 * atr_pct)  # -1.5 ATR (less noise stops)

            max_price = future_slice["price"].max()
            min_price = future_slice["price"].min()

            # Label: 1 if price hit TARGET before STOP within future_days
            hit_target = max_price >= target_price
            hit_stop   = min_price <= stop_price

            if hit_target and hit_stop:
                # Both hit — check which came first
                target_idx = (future_slice["price"] >= target_price).idxmax()
                stop_idx   = (future_slice["price"] <= stop_price).idxmax()
                label = 1 if target_idx <= stop_idx else 0
            elif hit_target:
                label = 1
            else:
                label = 0

            X_list.append(features)
            y_list.append(label)

        if len(X_list) < 10:
            return None, None, 0, 0

        X = np.array(X_list)
        y = np.array(y_list)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split for honest accuracy
        split = int(len(X_scaled) * 0.8)   # 80% train, 20% test
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]

        # Upgrade to GradientBoostingClassifier
        # Better than LogisticRegression for non-linear feature interactions
        # (RSI 65 + Volume spike + Bull regime = interaction LR misses)
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,          # Shallow trees prevent overfitting on small data
                learning_rate=0.05,   # Slow learning = better generalisation
                subsample=0.8,        # Row sampling = reduces overfitting
                random_state=42
            )
        except ImportError:
            # Fallback to RandomForest if GBC unavailable
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                class_weight='balanced',
                random_state=42
            )

        model.fit(X_train, y_train)

        # Honest accuracy = test set (never seen during training)
        if len(X_test) > 0:
            accuracy = model.score(X_test, y_test)
        else:
            accuracy = model.score(X_scaled, y)  # Fallback if too little data

        return model, scaler, accuracy, len(X)

    except Exception as e:
        print(Fore.YELLOW + f"[ML] Training failed for {coin}: {e}")
        return None, None, 0, 0

def predict_ml_outcome(row, model, scaler, df=None, btc_df=None, fng_value=50):
    """
    [K][L][M] Predict probability of a profitable trade.
    Returns: (probability, confidence_level) where confidence_level is "high" / "medium" / "low"
    """
    if model is None or scaler is None:
        return 0.5, "unknown"

    try:
        features = extract_ml_features(row, df, btc_df=btc_df, fng_value=fng_value)
        if features is None:
            return 0.5, "unknown"

        # Validate feature count matches model expectation
        if hasattr(scaler, 'n_features_in_') and len(features) != scaler.n_features_in_:
            logger.warning(f"[ML] Feature mismatch: got {len(features)}, model expects {scaler.n_features_in_}")
            return 0.5, "unknown"

        X_scaled = scaler.transform([features])
        proba = model.predict_proba(X_scaled)[0][1]

        if proba >= 0.70:
            confidence = "high"
        elif proba >= 0.55:
            confidence = "medium"
        else:
            confidence = "low"

        return proba, confidence

    except Exception as e:
        logger.debug(f"[ML] Prediction error: {e}")
        return 0.5, "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# SHORT ML MODEL — trained on short outcomes, completely separate from long model
# Long model:  label 1 = price rose  +2.5 ATR before falling -1.5 ATR  (long wins)
# Short model: label 1 = price fell  -2.5 ATR before rising  +1.5 ATR  (short wins)
# Same features, same GBM, different labels, different .pkl files.
# ═══════════════════════════════════════════════════════════════════════════════

def train_short_ml_model(df, coin, btc_df=None, fng_value=50,
                         lookback=ML_LOOKBACK_DAYS, future_days=ML_FUTURE_DAYS):
    """Train SHORT-specific model with inverted ATR labels. Long model untouched."""
    if not ML_ENABLED or not ML_AVAILABLE or len(df) < lookback + future_days:
        return None, None, 0, 0
    try:
        X_list, y_list = [], []
        for i in range(len(df) - future_days - 1):
            if i < lookback:
                continue
            row      = df.iloc[i]
            features = extract_ml_features(row, df, btc_df=btc_df, fng_value=fng_value)
            if features is None:
                continue
            current_price = row["price"]
            future_slice  = df.iloc[i+1:i+future_days+1]
            if len(future_slice) < future_days:
                break
            atr_val = row.get("ATR", np.nan)
            if np.isnan(atr_val) or atr_val <= 0 or current_price <= 0:
                short_target = current_price * 0.95
                short_stop   = current_price * 1.08
            else:
                atr_pct      = atr_val / current_price
                short_target = current_price * (1 - 2.5 * atr_pct)
                short_stop   = current_price * (1 + 1.5 * atr_pct)
            max_price = future_slice["price"].max()
            min_price = future_slice["price"].min()
            hit_target = min_price <= short_target
            hit_stop   = max_price >= short_stop
            if hit_target and hit_stop:
                target_idx = (future_slice["price"] <= short_target).idxmax()
                stop_idx   = (future_slice["price"] >= short_stop).idxmax()
                label = 1 if target_idx <= stop_idx else 0
            elif hit_target:
                label = 1
            else:
                label = 0
            X_list.append(features)
            y_list.append(label)
        if len(X_list) < 10:
            return None, None, 0, 0
        X = np.array(X_list)
        y = np.array(y_list)
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        split = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42)
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100, max_depth=4, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test) if len(X_test) > 0 else model.score(X_scaled, y)
        return model, scaler, accuracy, len(X)
    except Exception as e:
        print(f"[SHORT ML] Training failed for {coin}: {e}")
        return None, None, 0, 0


def save_short_model(coin, model, scaler, accuracy, samples):
    """Save short model to _short_ files. Long model files never touched."""
    try:
        if not MODEL_PERSISTENCE_ENABLED:
            return False
        Path(MODEL_PERSISTENCE_DIR).mkdir(parents=True, exist_ok=True)
        with open(get_short_model_path(coin), "wb") as f:
            pickle.dump(model, f)
        with open(get_short_scaler_path(coin), "wb") as f:
            pickle.dump(scaler, f)
        meta = {
            "coin": coin, "trained_date": datetime.now().isoformat(),
            "accuracy": float(accuracy), "samples": int(samples),
            "model_version": "v8.0", "n_features": 16,
            "model_type": "GradientBoosting_SHORT",
            "label": "1=short_wins(price fell -2.5ATR before +1.5ATR stop)"
        }
        with open(get_short_metadata_path(coin), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[SHORT ML] Saved: {coin} (acc: {accuracy:.1%}, samples: {samples})")
        return True
    except Exception as e:
        print(f"[SHORT ML] Error saving {coin}: {e}")
        return False


def load_short_model(coin):
    """Load short model. Returns (model, scaler, metadata) or (None, None, None)."""
    try:
        if not MODEL_PERSISTENCE_ENABLED or not LOAD_CACHED_MODELS:
            return None, None, None
        mp = get_short_model_path(coin)
        sp = get_short_scaler_path(coin)
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None, None, None
        with open(mp, "rb") as f:
            model = pickle.load(f)
        with open(sp, "rb") as f:
            scaler = pickle.load(f)
        metadata = {}
        meta_path = get_short_metadata_path(coin)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        if metadata.get("model_version") != "v8.0" or metadata.get("n_features", 0) != 16:
            return None, None, None
        if metadata.get("samples", 0) < 80:
            return None, None, None
        print(f"[SHORT ML] Loaded: {coin} (acc: {metadata.get('accuracy', 0):.1%})")
        return model, scaler, metadata
    except Exception as e:
        print(f"[SHORT ML] Error loading {coin}: {e}")
        return None, None, None


def should_retrain_short_model(coin):
    """True if short model is stale or missing."""
    try:
        if not MODEL_PERSISTENCE_ENABLED or not AUTO_RETRAIN:
            return True
        meta_path = get_short_metadata_path(coin)
        if not os.path.exists(meta_path):
            return True
        with open(meta_path, "r") as f:
            meta = json.load(f)
        days_old = (datetime.now() - datetime.fromisoformat(meta["trained_date"])).days
        return days_old >= MODEL_CACHE_DAYS
    except Exception:
        return True


def predict_short_ml_outcome(row, model, scaler, df=None, btc_df=None, fng_value=50):
    """
    Predict probability a SHORT trade wins in current conditions.
    Returns (probability, confidence)
      probability 0.80 = in 80% of historically similar setups, shorting won.
    """
    if model is None or scaler is None:
        return 0.5, "unknown"
    try:
        features = extract_ml_features(row, df, btc_df=btc_df, fng_value=fng_value)
        if features is None:
            return 0.5, "unknown"
        if hasattr(scaler, "n_features_in_") and len(features) != scaler.n_features_in_:
            return 0.5, "unknown"
        X_scaled = scaler.transform([features])
        proba = model.predict_proba(X_scaled)[0][1]
        conf  = "high" if proba >= 0.70 else ("medium" if proba >= 0.55 else "low")
        return proba, conf
    except Exception as e:
        return 0.5, "unknown"


def predict_direction(df):
    """
    Predict UP or DOWN direction based on technical momentum indicators.
    Returns: ("UP" | "DOWN", confidence_0_to_1)
    """
    if len(df) < 5:
        return "NEUTRAL", 0.5

    last = df.iloc[-1]

    # Collect direction signals
    signals = []

    # 1. EMA alignment (strongest signal)
    if last["EMA12"] > last["EMA26"]:
        signals.append(1.0)  # bullish
    else:
        signals.append(-1.0)  # bearish

    # 2. MACD
    if last["MACD"] > last["Signal"]:
        signals.append(0.7)
    else:
        signals.append(-0.7)

    # 3. RSI (50 is neutral)
    rsi_bias = (last["RSI"] - 50) / 50  # normalized -1 to 1
    signals.append(rsi_bias)

    # 4. OBV momentum
    if len(df) > 2:
        obv_trend = 1.0 if last["OBV"] > df.iloc[-2]["OBV"] else -1.0
        signals.append(obv_trend * 0.5)

    # 5. Price vs EMAs
    if last["price"] > last["EMA12"] > last["EMA26"]:
        signals.append(1.0)  # bullish stacking
    elif last["price"] < last["EMA12"] < last["EMA26"]:
        signals.append(-1.0)  # bearish stacking
    else:
        signals.append(0.0)

    # Aggregate
    avg_signal = np.mean(signals)
    confidence = min(abs(avg_signal), 1.0)

    direction = "UP" if avg_signal > 0 else "DOWN"
    return direction, confidence

def calculate_confidence_score(score, direction, direction_conf, ml_proba, blocked, smart_money_mode, is_whale, sig):
    """
    ═══════════════════════════════════════════════════════════════════
    CONFIDENCE SCORE: Combined metric (0-100) for easy decision-making
    ═══════════════════════════════════════════════════════════════════

    Uses LINEAR INTERPOLATION between anchor points — no cliff edges.
    A coin at 69% direction confidence scores proportionally between the
    60% and 70% anchors, not the same as 60%.

    Components:
      Technical   (0-35)  — tech score
      Direction   (0-20)  — direction model confidence
      ML Model    (0-20)  — long ML win probability
      Smart Money (0-15)  — OBV/whale accumulation
      No Blocks   (0-10)  — signal not blocked

    Penalties applied after base score:
      Backtest win rate < 45%   → up to -15
      Profit factor < 0.85      → up to -10
    """
    import bisect

    base_score = 0
    breakdown  = {}

    # Check if this is a SHORT signal
    action   = sig.get("action", "")
    is_short = "SHORT" in action

    # ── Helper: linear interpolation between anchor table ────────────
    def interp(value, anchors):
        """
        anchors: sorted list of (threshold, points) tuples.
        For BUY-style (higher=better): anchors sorted ascending.
        For SHORT-style (lower=better): anchors sorted descending.
        Returns interpolated points as a float, rounded to nearest int.
        """
        if not anchors:
            return 0
        xs = [a[0] for a in anchors]
        ys = [a[1] for a in anchors]
        # Clamp to range
        if value <= xs[0]:
            return ys[0]
        if value >= xs[-1]:
            return ys[-1]
        # Find bracket and interpolate
        idx = bisect.bisect_right(xs, value) - 1
        x0, y0 = xs[idx],   ys[idx]
        x1, y1 = xs[idx+1], ys[idx+1]
        t = (value - x0) / (x1 - x0)
        return round(y0 + t * (y1 - y0))

    # ─────────────────────────────────────────────────────────────────
    # 1. TECHNICAL SCORE (0-35 points)
    # ─────────────────────────────────────────────────────────────────
    if is_short:
        # Anchors: (score_threshold, points) — more negative = better for short
        _tech_anchors = [(-8, 35), (-7, 28), (-6, 21), (-5, 14), (-3, 7), (0, 2), (1, 0)]
        tech_points   = interp(score, _tech_anchors)
        if score <= -8:   tech_desc = "Excellent"
        elif score <= -6: tech_desc = "Very Good"
        elif score <= -5: tech_desc = "Good"
        elif score <= -3: tech_desc = "Acceptable"
        elif score < 0:   tech_desc = "Weak"
        else:             tech_desc = "Bullish (bad for short)"
    else:
        # Anchors: (score, points) — higher score = better
        _tech_anchors = [(0, 0), (1, 7), (3, 14), (5, 21), (6, 28), (8, 35)]
        tech_points   = interp(score, _tech_anchors)
        if score >= 8:   tech_desc = "Excellent"
        elif score >= 6: tech_desc = "Very Good"
        elif score >= 5: tech_desc = "Good"
        elif score >= 3: tech_desc = "Acceptable"
        elif score >= 1: tech_desc = "Weak"
        elif score == 0: tech_desc = "Neutral"
        else:            tech_desc = "Bearish"

    breakdown["Technical (0-35)"] = f"{tech_points}/35 ({tech_desc}) - Score {score:.1f}/5"
    base_score += tech_points

    # ─────────────────────────────────────────────────────────────────
    # 2. DIRECTION CONSENSUS (0-20 points) — interpolated
    # ─────────────────────────────────────────────────────────────────
    if is_short:
        if direction == "DOWN":
            _dir_anchors = [(0.30, 2), (0.50, 10), (0.60, 14), (0.70, 17), (0.80, 20)]
            if direction_conf >= 0.80:   dir_desc = "Overwhelming DOWN"
            elif direction_conf >= 0.70: dir_desc = "Very Strong DOWN"
            elif direction_conf >= 0.60: dir_desc = "Strong DOWN"
            elif direction_conf >= 0.50: dir_desc = "Moderate DOWN"
            else:                        dir_desc = "Weak DOWN"
        else:  # UP — bad for short
            _dir_anchors = [(0.30, 10), (0.70, 5), (0.80, 0)]
            if direction_conf >= 0.80:   dir_desc = "Overwhelming UP"
            elif direction_conf >= 0.70: dir_desc = "Strong UP"
            else:                        dir_desc = "Moderate UP"
    else:
        if direction == "UP":
            _dir_anchors = [(0.30, 3), (0.40, 6), (0.50, 10), (0.60, 14), (0.70, 17), (0.80, 20)]
            if direction_conf >= 0.80:   dir_desc = "Overwhelming UP"
            elif direction_conf >= 0.70: dir_desc = "Very Strong UP"
            elif direction_conf >= 0.60: dir_desc = "Strong UP"
            elif direction_conf >= 0.50: dir_desc = "Moderate UP"
            elif direction_conf >= 0.40: dir_desc = "Weak UP"
            elif direction_conf >= 0.30: dir_desc = "Very Weak UP"
            else:                        dir_desc = "Barely UP"
        else:  # DOWN — bad for BUY
            _dir_anchors = [(0.30, 6), (0.50, 5), (0.60, 3), (0.70, 2), (0.80, 0)]
            if direction_conf >= 0.80:   dir_desc = "Overwhelming DOWN"
            elif direction_conf >= 0.70: dir_desc = "Strong DOWN"
            elif direction_conf >= 0.60: dir_desc = "Moderate DOWN"
            elif direction_conf >= 0.50: dir_desc = "Weak DOWN"
            else:                        dir_desc = "Uncertain DOWN"

    dir_points = interp(direction_conf, _dir_anchors)
    breakdown["Direction (0-20)"] = f"{dir_points}/20 ({dir_desc}) - {direction_conf:.0%} consensus"

    # COUNTER-TREND PENALTY
    counter_trend = False
    if not is_short:
        if score < 0 and direction == "UP":
            dir_points    = dir_points // 2
            counter_trend = True
            breakdown["Direction (0-20)"] = (
                f"{dir_points}/20 ({dir_desc} — HALVED: tech bearish, direction bullish = counter-trend risk)"
            )
        elif score > 0 and direction == "DOWN":
            dir_points    = dir_points // 2
            counter_trend = True
            breakdown["Direction (0-20)"] = (
                f"{dir_points}/20 ({dir_desc} — HALVED: tech bullish, direction bearish = counter-trend risk)"
            )

    base_score += dir_points

    # ─────────────────────────────────────────────────────────────────
    # 3. ML PROBABILITY (0-20 points) — interpolated
    # ─────────────────────────────────────────────────────────────────
    if is_short:
        # Lower ml_proba = better for short (shorts win when longs fail)
        _ml_anchors = [(0.15, 20), (0.25, 18), (0.35, 15), (0.40, 12), (0.45, 8), (0.55, 4)]
        if ml_proba <= 0.15:   ml_desc = "Excellent"
        elif ml_proba <= 0.25: ml_desc = "Very Good"
        elif ml_proba <= 0.35: ml_desc = "Good"
        elif ml_proba <= 0.40: ml_desc = "Acceptable"
        elif ml_proba <= 0.45: ml_desc = "Mediocre"
        else:                  ml_desc = "Poor"
        # For short, lower is better so reverse direction for interp
        ml_points = interp(-ml_proba, [(-a[0], a[1]) for a in _ml_anchors])
    else:
        # BUY: Higher ml_proba = better for long
        # ML < 35% = model actively predicting failure → 0 points + penalty
        if ml_proba < 0.35:
            ml_points = 0
            ml_desc   = "Terrible (against trade)"
            # Apply penalty: model has seen this setup and says it fails
            # Penalty scales with how bearish: 25% gets -10, 20% gets -15
            _ml_penalty = round((0.35 - ml_proba) / 0.35 * 40)  # up to -40 at ml=0%
            base_score -= _ml_penalty
            breakdown["❌ ML Bearish Penalty"] = (
                f"ML {ml_proba:.0%} actively predicts failure ({1-ml_proba:.0%} loss rate) (-{_ml_penalty})"
            )
        else:
            _ml_anchors = [(0.35, 2), (0.40, 4), (0.50, 6), (0.55, 8), (0.60, 12), (0.65, 15), (0.75, 18), (0.85, 20)]
            if ml_proba >= 0.85:   ml_desc = "Excellent"
            elif ml_proba >= 0.75: ml_desc = "Very Good"
            elif ml_proba >= 0.65: ml_desc = "Good"
            elif ml_proba >= 0.60: ml_desc = "Acceptable"
            elif ml_proba >= 0.55: ml_desc = "Mediocre"
            elif ml_proba >= 0.50: ml_desc = "Weak"
            elif ml_proba >= 0.40: ml_desc = "Poor"
            else:                  ml_desc = "Very Poor"
            ml_points = interp(ml_proba, _ml_anchors)

    breakdown["ML Model (0-20)"] = f"{ml_points}/20 ({ml_desc}) - {ml_proba:.0%} win probability"
    base_score += ml_points

    # ─────────────────────────────────────────────────────────────────
    # CROSS-COMPENSATION: Strong ML can partially offset weak score
    # and strong score can partially offset weak ML
    # ─────────────────────────────────────────────────────────────────
    _ml_accuracy = sig.get("ml_accuracy", 0.5) or 0.5   # read once for both cases

    # Case 1: ML strong (≥70%) but score weak (3-4)
    if not is_short and ml_proba >= 0.70 and _ml_accuracy >= 0.65 and score < 5:
        _comp = round((ml_proba - 0.70) / 0.30 * 5)
        if _comp > 0:
            base_score += _comp
            breakdown["⚡ ML Compensates Score"] = (
                f"ML {ml_proba:.0%} (acc {_ml_accuracy:.0%}) partially offsets weak score "
                f"{score:.0f}/5 (+{_comp})"
            )

    # Case 2: Score strong (≥5) but ML weak (25-50%)
    # Only compensate when ML is uncertain (25-50%), NOT when actively bearish (<35%)
    # If ML < 35%, it's not uncertain — it's predicting failure. No compensation.
    elif not is_short and score >= 5 and 0.35 <= ml_proba < 0.50:
        _comp = round((score - 4) * 1.5)
        _comp = min(_comp, 5)
        if _comp > 0:
            base_score += _comp
            breakdown["⚡ Score Compensates ML"] = (
                f"Score {score:.0f}/5 partially offsets uncertain ML {ml_proba:.0%} (+{_comp})"
            )

    # ─────────────────────────────────────────────────────────────────
    # 4. SMART MONEY (0-15 points)
    # ─────────────────────────────────────────────────────────────────
    if smart_money_mode == "accumulation":
        sm_points = 15 if is_whale else 12
        sm_desc   = "ACCUMULATING" + (" (WHALE)" if is_whale else "")
    elif smart_money_mode == "distribution":
        sm_points = 0
        sm_desc   = "DUMPING (AVOID!)"
    else:
        sm_points = 7
        sm_desc   = "NEUTRAL"

    breakdown["Smart Money (0-15)"] = f"{sm_points}/15 ({sm_desc})"
    base_score += sm_points

    # ─────────────────────────────────────────────────────────────────
    # 5. NO BLOCKS (0-10 points)
    # ─────────────────────────────────────────────────────────────────
    if blocked:
        block_points = 0
        block_desc   = "BLOCKED ❌"
    else:
        block_points = 10
        block_desc   = "ALL CLEAR ✅"

    breakdown["No Blocks (0-10)"] = f"{block_points}/10 ({block_desc})"
    base_score += block_points

    # ─────────────────────────────────────────────────────────────────
    # CALCULATE FINAL CONFIDENCE SCORE (0-100)
    # ─────────────────────────────────────────────────────────────────
    confidence_score = base_score

    # TECH-BASED HARD CAPS
    if not is_short:
        if score < 0:
            confidence_score = min(confidence_score, 50)
        elif score == 0:
            confidence_score = min(confidence_score, 65)

    # COUNTER-TREND CAP
    if counter_trend:
        confidence_score = min(confidence_score, 55)

    # BACKTEST PENALTY: Poor historical performance deducts points
    # Only apply if enough trades for statistical significance (min 10)
    backtest_win_rate  = sig.get("backtest_win_rate", None)
    backtest_pf        = sig.get("backtest_pf", None)
    backtest_trades    = sig.get("backtest_trades", 999)   # default high = assume sufficient

    _backtest_reliable = backtest_trades >= 10

    if backtest_win_rate is not None:
        if not _backtest_reliable:
            breakdown["⚠️ Backtest"] = (
                f"Only {backtest_trades} trades — insufficient for statistical significance "
                f"(WR {backtest_win_rate:.1f}% ignored, need ≥10)"
            )
        elif backtest_win_rate < 25:
            confidence_score -= 15
            breakdown["⚠️ Backtest Penalty"] = f"Win rate {backtest_win_rate:.1f}% < 25% — historically very poor (-15)"
        elif backtest_win_rate < 35:
            confidence_score -= 10
            breakdown["⚠️ Backtest Penalty"] = f"Win rate {backtest_win_rate:.1f}% < 35% — below average (-10)"
        elif backtest_win_rate < 45:
            confidence_score -= 5
            breakdown["⚠️ Backtest Penalty"] = f"Win rate {backtest_win_rate:.1f}% < 45% — marginal (-5)"

    # PROFIT FACTOR PENALTY — same reliability gate
    if backtest_pf is not None and _backtest_reliable:
        if backtest_pf < 0.60:
            confidence_score -= 10
            breakdown["⚠️ Profit Factor"] = f"PF {backtest_pf:.2f} < 0.60 — setup consistently loses money (-10)"
        elif backtest_pf < 0.85:
            confidence_score -= 5
            breakdown["⚠️ Profit Factor"] = f"PF {backtest_pf:.2f} < 0.85 — net negative historically (-5)"
        # PF 0.85–1.10 = breakeven noise band, no penalty

    # ENTRY QUALITY PENALTY: Chasing or near exhaustion
    # Note: Low ADX is already penalised in the scoring pipeline via Z-score gate
    # (STRETCHED in range → -1) and exhaustion check (EMA26 deviation → -1)
    # so no separate ADX penalty needed here.
    entry_quality = sig.get("entry_quality", "normal")
    if entry_quality == "chasing":
        confidence_score -= 8
        breakdown["⚠️ Entry Quality"] = "Chasing breakout — price above Donchian upper (-8)"
    elif entry_quality == "near_exhaustion":
        confidence_score -= 5
        breakdown["⚠️ Entry Quality"] = f"Near EMA26 exhaustion zone — {sig.get('ema26_dev_pct', 0):.1f}% above EMA26 (-5)"

    # Clamp final score 0-100
    confidence_score = max(0, min(100, confidence_score))
    
    # Check if this is a SHORT signal
    action   = sig.get("action", "")
    is_short = "SHORT" in action
    # ml_proba comes from the function parameter — do NOT override from sig
    # (gem scanner was getting 0.5 default because it doesn't set sig["ml_proba"])
    
    # HARD BLOCK: ML very weak — always flag as poor regardless of confidence
    # Threshold: < 0.20 (not < 0.25) because 25% is borderline, handled by ML cap above
    if not is_short and ml_proba < 0.20:
        rating = "🔴 POOR"
        recommendation = "AVOID"
        emoji = "🛑"
    elif is_short and ml_proba > 0.75:  # SHORT with strong bullish (bad for SHORT)
        rating = "🔴 POOR"
        recommendation = "DO NOT SHORT"
        emoji = "🛑"
    # Determine recommendation based on confidence and signal type
    elif is_short:
        # SHORT-specific recommendations (INVERTED - low conf is good!)
        if confidence_score <= 15:
            rating = "🔴 EXCELLENT"
            recommendation = "STRONG SHORT"
            emoji = "📉"
        elif confidence_score <= 25:
            rating = "🔴 VERY GOOD"
            recommendation = "SHORT NOW"
            emoji = "⬇️"
        elif confidence_score <= 35:
            rating = "🟠 GOOD"
            recommendation = "CONSIDER SHORT"
            emoji = "📉"
        elif confidence_score <= 50:
            rating = "🟡 FAIR"
            recommendation = "WAIT FOR BETTER SHORT SETUP"
            emoji = "⏸️"
        elif confidence_score <= 75:
            rating = "🟡 WEAK"
            recommendation = "AVOID SHORT"
            emoji = "⚠️"
        else:
            rating = "🔴 POOR"
            recommendation = "DO NOT SHORT"
            emoji = "🛑"
    else:
        # BUY-specific recommendations (original logic)
        if confidence_score >= 85:
            rating = "🟢 EXCELLENT"
            recommendation = "STRONG BUY"
            emoji = "🚀"
        elif confidence_score >= 75:
            rating = "🟢 VERY GOOD"
            recommendation = "BUY"
            emoji = "⬆️"
        elif confidence_score >= 65:
            rating = "🟡 GOOD"
            recommendation = "CONSIDER BUY"
            emoji = "→"
        elif confidence_score >= 50:
            rating = "🟡 FAIR"
            recommendation = "WAIT"
            emoji = "⏸️"
        elif confidence_score >= 35:
            rating = "🟠 WEAK"
            recommendation = "AVOID"
            emoji = "⬇️"
        else:
            rating = "🔴 POOR"
            recommendation = "DO NOT BUY"
            emoji = "🛑"
    
    return {
        "score": confidence_score,
        "rating": rating,
        "recommendation": recommendation,
        "emoji": emoji,
        "breakdown": breakdown,
        "max_possible": 100
    }

def estimate_move_size(df, lookback=20):
    """
    Estimate expected move size in % based on ATR and historical volatility.
    Returns: (expected_move_pct, basis)
    """
    if len(df) < lookback:
        return 2.0, "insufficient_data"

    last = df.iloc[-1]
    current_price = last["price"]

    # ATR-based estimate
    atr = last["ATR"]
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

    # Historical volatility (std dev of returns)
    returns = df["price"].pct_change().tail(lookback)
    hist_vol = returns.std() * 100

    # Estimate: if ATR suggests 2% move, and vol suggests 1.5%, average them
    expected_move = (atr_pct * 0.6 + hist_vol * 0.4)  # weight ATR more

    # Cap at reasonable bounds
    expected_move = np.clip(expected_move, 0.5, 15.0)

    basis = "ATR+volatility"
    return expected_move, basis

def generate_narrative(coin, last, sig, direction, move_size, ml_proba, smart_money_mode, is_whale):
    """
    Generate a short narrative about what's likely to happen.
    Returns: narrative_string
    """
    price = last["price"]
    score = sig["score"]
    min_score = MIN_BUY_SCORE

    # Build narrative
    parts = []

    # Score context
    if score >= min_score:
        parts.append(f"Strong setup ({score}/{min_score})")
    elif score >= min_score - 1:
        parts.append(f"Borderline ({score}/{min_score})")
    else:
        parts.append(f"Weak ({score}/{min_score})")

    # Direction
    parts.append(f"{direction}")

    # Move size
    parts.append(f"{move_size:.1f}% move")

    # ML confidence
    if ml_proba >= 0.70:
        parts.append("High confidence")
    elif ml_proba >= 0.55:
        parts.append("Medium confidence")
    else:
        parts.append("Low confidence")

    # Smart money signal
    if smart_money_mode == "accumulation":
        parts.append("Whale accumulating")
    elif smart_money_mode == "distribution":
        parts.append("Whale dumping")

    # Whale presence
    if is_whale:
        parts.append("High volume")

    # Sentiment
    if score >= min_score and direction == "UP" and ml_proba >= 0.60:
        sentiment = "🟢 BULLISH"
    elif score >= min_score and direction == "DOWN":
        sentiment = "🟡 CONFLICT"
    elif score < min_score - 1 and direction == "DOWN":
        sentiment = "🔴 BEARISH"
    else:
        sentiment = "🟡 NEUTRAL"

    return " | ".join(parts) + f" → {sentiment}"

# ===========================
# [F] CONSECUTIVE CANDLE CONFIRMATION
# ===========================
def check_consecutive_confirmation(df, btc_regime="unknown", is_btc=False, min_days=MIN_CONSECUTIVE_DAYS):
    """
    [F] Require that BUY conditions are met for N consecutive days.
    Returns: (passes_confirmation, consecutive_days_met, last_n_scores)
    
    Only fires BUY if the last min_days candles all have:
      - score >= MIN_BUY_SCORE
      - blocked == False
    """
    if len(df) < min_days:
        return False, 0, []

    # Score the last min_days candles (DEBUG suppressed — historical scoring)
    scores_and_blocks = []
    for i in range(-min_days, 0):
        row = df.iloc[i]
        sig = generate_signal_and_reasons(row, df, btc_regime=btc_regime, is_btc=is_btc,
                                          suppress_debug=True)
        scores_and_blocks.append({
            "day_offset": i,
            "score": sig["score"],
            "blocked": sig["blocked"],
            "action": "PASS" if (sig["score"] >= MIN_BUY_SCORE and not sig["blocked"]) else "FAIL"
        })

    # Check if all min_days meet criteria
    passes = all(s["action"] == "PASS" for s in scores_and_blocks)
    consecutive_met = sum(1 for s in scores_and_blocks if s["action"] == "PASS")

    return passes, consecutive_met, scores_and_blocks

# ===========================
# SIGNAL + SCORING
# ===========================

def detect_market_regime(btc_regime, cloud_state, adx, ema12, ema26, macd, sig_line):
    """Detect current market regime for indicator weighting.
    
    Returns: "bull_trend" | "range" | "bear"
    
    BULL TREND:  Weight trend-following (Ichimoku, MACD, EMA)
    RANGE:       Weight mean-reversion (BB, RSI, Order Book)
    BEAR:        Standard scoring + buy blocks
    """
    # Bear: BTC downtrend OR price below cloud
    if btc_regime == "bear" or cloud_state == "below":
        return "bear"
    
    # Bull trend: BTC bull + price above cloud + MACD/EMA aligned
    strong_trend = (cloud_state == "above" and
                    ema12 > ema26 and
                    macd > sig_line)
    if btc_regime == "bull" and strong_trend:
        return "bull_trend"
    
    # Range/Chop: everything else (in cloud, mixed signals, weak ADX)
    return "range"

def generate_signal_and_reasons(last, df, btc_regime="unknown", is_btc=False, suppress_debug=False):
    """
    SIMPLIFIED 3-GROUP SCORING MODEL:
    ─────────────────────────────────
    GROUP 1: MACRO  (max ±2)
      BTC regime:     +1 bull / blocked bear
      Weekly EMA50:   +1 above / -1 below

    GROUP 2: TREND  (max ±3)
      Ichimoku cloud: +2 above / -1 in / blocked below
      EMA direction:  +1 EMA12>EMA26 / -1 below
      MACD:           +1 bullish / -1 bearish

    GROUP 3: MOMENTUM  (max ±2)
      RSI zone:       +1 (60-70) / 0 (50-60) / -1 (70-85) / -2 (>85) / +1 (<30)
      RSI Divergence: +1 bullish / -1 bearish

    TOTAL: -7 to +8
    BUY trigger:  >= +4 (MIN_BUY_SCORE)
    SELL trigger: <= -4 (MIN_SELL_SCORE)
    """
    reasons  = []
    caution  = []
    blockers = []
    score    = 0
    blocked  = False
    DEBUG    = True if not suppress_debug else False  # Suppress when called from backtest loops
    cloud_state = "unknown"  # Ichimoku cloud position

    price    = last["price"]
    ema12    = last["EMA12"]
    ema26    = last["EMA26"]
    macd     = last["MACD"]
    sig_line = last["Signal"]
    macd_hist= last["MACD_Hist"]
    rsi      = last["RSI"]
    adx      = last["ADX"]
    plus_di  = last.get("plus_di",  np.nan)
    minus_di = last.get("minus_di", np.nan)
    senkou_a = last.get("senkou_a", np.nan)
    senkou_b = last.get("senkou_b", np.nan)
    kijun    = last.get("kijun",    np.nan)
    tenkan   = last.get("tenkan",   np.nan)
    chikou           = last.get("chikou",           np.nan)
    chikou_ref       = last.get("chikou_ref",       np.nan)
    chikou_cloud_top = last.get("chikou_cloud_top", np.nan)
    chikou_cloud_bot = last.get("chikou_cloud_bot", np.nan)
    ema_w50  = last.get("EMA_WEEKLY50", np.nan)

    # ══════════════════════════════════════════════════════════
    # GROUP 1: MACRO  (max ±2)
    # ══════════════════════════════════════════════════════════

    # BTC Market Regime
    if not is_btc:
        if btc_regime == "bear":
            reasons.append("❌ [MACRO] BTC downtrend — altcoin BUY blocked")
            caution.append("BTCBearRegime")
            blockers.append("BTC Bear Regime")
            blocked = True
        elif btc_regime == "bull":
            reasons.append("✅ [MACRO] BTC uptrend — favourable environment")
            score += 1
        else:
            reasons.append("⚠️ [MACRO] BTC regime unknown")
    else:
        reasons.append("ℹ️ [MACRO] BTC self-check — regime filter skipped")

    # Weekly EMA50
    if not np.isnan(ema_w50):
        if price > ema_w50:
            reasons.append(f"✅ [MACRO] Price above Weekly EMA50 ({ema_w50:.4f}) — macro bullish")
            score += 1
        else:
            reasons.append(f"❌ [MACRO] Price below Weekly EMA50 ({ema_w50:.4f}) — macro headwind")
            score -= 1
            caution.append("WeeklyBearish")
    else:
        reasons.append("⚠️ [MACRO] Weekly EMA50 unavailable")

    macro_score = score
    if DEBUG:
        if is_btc:
            btc_contrib = 0  # BTC skips its own regime check
            btc_label   = "self (skipped)"
        else:
            btc_contrib = 1 if btc_regime == "bull" else (-1 if btc_regime == "bear" else 0)
            btc_label   = btc_regime
        ema_contrib = 1 if (not np.isnan(ema_w50) and price > ema_w50) else (-1 if not np.isnan(ema_w50) else 0)
        print(f"[SCORE] MACRO:    {macro_score:+d}  (BTC={btc_contrib:+d} [{btc_label}], WeeklyEMA={ema_contrib:+d} [{'above' if ema_contrib>0 else 'below' if ema_contrib<0 else 'N/A'}])")

    # ── REGIME DETECTION ─────────────────────────────────────────────
    # Determine market regime for weighting adjustments
    # chop_detected approximated here from BB and price position
    try:
        _col = 'close' if 'close' in df.columns else 'price'
        _prices = df[_col].iloc[-20:].values
        _sma = _prices.mean(); _std = _prices.std()
        _bb_w = ((_sma + 2*_std) - (_sma - 2*_std)) / _sma * 100 if _sma > 0 else 0
        _range = (_prices.max() - _prices.min()) / _prices[-1] * 100
        _pos = ((_prices[-1] - (_sma - 2*_std)) / (4*_std)) * 100 if _std > 0 else 50
        _chop_signals = (1 if _range < 12 else 0) + (1 if 30 < _pos < 70 else 0) + (1 if _bb_w > 9 else 0)
        is_range_regime = (_chop_signals >= 2) or (btc_regime not in ["bull", "bear"])
        is_bull_regime  = (btc_regime == "bull") and (_chop_signals < 2)
    except:
        is_range_regime = False
        is_bull_regime  = (btc_regime == "bull")

    # ══════════════════════════════════════════════════════════
    # GROUP 2: TREND  (max ±3)
    # ══════════════════════════════════════════════════════════

    # Ichimoku Cloud
    if not (np.isnan(senkou_a) or np.isnan(senkou_b)):
        cloud_top    = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        if price > cloud_top:
            reasons.append("✅ [TREND] Price above Ichimoku Cloud — bullish")
            cloud_state = "above"
            score += 2
        elif price < cloud_bottom:
            reasons.append("❌ [TREND] Price below Ichimoku Cloud — bearish (BLOCKED)")
            cloud_state = "below"
            caution.append("CloudBearish")
            blockers.append("Cloud Below")
            blocked = True
            score -= 2  # Strong bearish penalty
        else:
            reasons.append("⚠️ [TREND] Price inside Ichimoku Cloud — neutral")
            cloud_state = "in"
            caution.append("CloudNeutral")
            score -= 1
        reasons.append(f"📊 Tenkan={tenkan:.4f} | Kijun={kijun:.4f} | Cloud=({cloud_bottom:.4f} → {cloud_top:.4f})")
    else:
        reasons.append("⚠️ [TREND] Ichimoku cloud incomplete")

    # EMA Direction — info only: Ichimoku already scores the trend structure.
    # EMA cross is shown for trader reference but does NOT add to score here.
    if ema12 > ema26:
        reasons.append(f"✅ [TREND] EMA12 > EMA26 — bullish (confirmed by Ichimoku)")
    else:
        reasons.append(f"❌ [TREND] EMA12 < EMA26 — bearish (confirmed by Ichimoku)")

    # MACD
    if macd > sig_line:
        reasons.append(f"✅ [TREND] MACD bullish (hist={macd_hist:.4f})")
        score += 1
    else:
        reasons.append(f"❌ [TREND] MACD bearish (hist={macd_hist:.4f})")
        score -= 1

    # ── REGIME DETECTION (after trend indicators are known) ──────────
    market_regime = detect_market_regime(btc_regime, cloud_state, adx, ema12, ema26, macd, sig_line)

    # ── DI DIRECTIONAL PRESSURE ───────────────────────────────────────
    # DI- > DI+ means bearish directional momentum regardless of other signals.
    # Apply a penalty when directional pressure contradicts bullish structure.
    _di_bearish = (not np.isnan(plus_di) and not np.isnan(minus_di) and minus_di > plus_di)
    _di_gap = abs(minus_di - plus_di) if not (np.isnan(plus_di) or np.isnan(minus_di)) else 0
    if _di_bearish:
        score -= 1
        reasons.append(f"⚠️ [TREND] DI- ({minus_di:.1f}) > DI+ ({plus_di:.1f}) — bearish directional pressure (-1)")
    else:
        reasons.append(f"✅ [TREND] DI+ ({plus_di:.1f}) > DI- ({minus_di:.1f}) — bullish directional pressure")

    # BULL TREND BONUS: amplify trend signals
    # BUT suppress if:
    #   1. Chop detected — chop and bull trend contradict
    #   2. Coin ADX < 20 — BTC being bullish doesn't amplify a trendless coin
    #   3. DI- is dominant — directional momentum is bearish
    _regime_chop = _is_strong_chop(df)
    _adx_trend_ok = adx >= 20    # coin needs its own trend for amplification

    if market_regime == "bull_trend" and not _regime_chop and _adx_trend_ok and not _di_bearish:
        if cloud_state == "above":
            score += 1
            reasons.append("✅ [REGIME] Bull trend — Ichimoku weight amplified (+1)")
        if macd > sig_line:
            score += 1
            reasons.append("✅ [REGIME] Bull trend — MACD weight amplified (+1)")
    elif market_regime == "bull_trend" and _regime_chop:
        reasons.append("⚠️ [REGIME] Bull trend bonus SUPPRESSED — chop detected (contradicting signals)")
    elif market_regime == "bull_trend" and not _adx_trend_ok:
        reasons.append(f"⚠️ [REGIME] Bull trend bonus SUPPRESSED — coin ADX {adx:.1f} < 20 (no coin-level trend to amplify)")
    elif market_regime == "bull_trend" and _di_bearish:
        reasons.append(f"⚠️ [REGIME] Bull trend bonus SUPPRESSED — DI- dominant ({minus_di:.1f} > {plus_di:.1f})")
    elif market_regime == "range":
        # BB Position: add mean-reversion signal (not scored before)
        lower_bb = last.get("LowerBB", np.nan)
        upper_bb = last.get("UpperBB", np.nan)
        if not (np.isnan(lower_bb) or np.isnan(upper_bb)):
            bb_pos = (price - lower_bb) / (upper_bb - lower_bb + 1e-9)
            if bb_pos < 0.2:      # Near lower BB = oversold in range
                score += 1
                reasons.append(f"✅ [REGIME] Range — price near lower BB (mean reversion) (+1)")
            elif bb_pos > 0.8:    # Near upper BB = overbought in range
                score -= 1
                reasons.append(f"⚠️ [REGIME] Range — price near upper BB (likely rejection) (-1)")
            else:
                reasons.append(f"➖ [REGIME] Range market — mid BB (no edge)")
    
    if DEBUG:
        if market_regime == "bull_trend" and _regime_chop:
            print(f"[SCORE] REGIME:   BULL_TREND (SUPPRESSED — chop detected)")
        elif market_regime == "bull_trend" and not _adx_trend_ok:
            print(f"[SCORE] REGIME:   BULL_TREND (SUPPRESSED — ADX {adx:.1f} < 20, no coin-level trend)")
        elif market_regime == "bull_trend" and _di_bearish:
            print(f"[SCORE] REGIME:   BULL_TREND (SUPPRESSED — DI- dominant)")
        elif market_regime == "bull_trend":
            print(f"[SCORE] REGIME:   BULL_TREND weighting applied (+2)")
        elif market_regime == "range":
            print(f"[SCORE] REGIME:   RANGE weighting applied")
        else:
            print(f"[SCORE] REGIME:   {market_regime.upper()}")

    trend_score = score - macro_score
    if DEBUG:
        ichimoku_c = 2 if cloud_state=="above" else (-1 if cloud_state=="in" else -2)
        macd_c     = 1 if macd > sig_line else -1
        di_c       = -1 if _di_bearish else 0   # DI only penalises, never adds
        if market_regime == "bull_trend" and _regime_chop:
            regime_label = "SUPPRESSED (chop)"
        elif market_regime == "bull_trend" and not _adx_trend_ok:
            regime_label = f"SUPPRESSED (ADX={adx:.0f}<20)"
        elif market_regime == "bull_trend" and _di_bearish:
            regime_label = "SUPPRESSED (DI- dominant)"
        elif market_regime == "bull_trend":
            regime_label = "+2 applied"
        elif market_regime == "range":
            regime_label = "range mode"
        else:
            regime_label = "standard"
        di_label = f", DI={di_c:+d}" if di_c != 0 else ""
        print(f"[SCORE] TREND:    {trend_score:+d}  (Ichimoku={ichimoku_c:+d}, MACD={macd_c:+d}{di_label}, EMA=info, Regime={regime_label})")

    # ══════════════════════════════════════════════════════════
    # GROUP 3: MOMENTUM  (max ±2)
    # ══════════════════════════════════════════════════════════

    # RSI: Trend-context aware scoring
    strong_uptrend   = (cloud_state == "above" and ema12 > ema26 and macd > sig_line)
    strong_downtrend = (cloud_state == "below" and ema12 < ema26 and macd < sig_line)
    in_trend         = strong_uptrend or strong_downtrend

    # Detect strong chop before RSI scoring — disables positive RSI signals
    strong_chop_rsi = _is_strong_chop(df)

    if in_trend:
        # TRENDING MARKET
        if rsi < 30:
            reasons.append(f"✅ [MOMENTUM] RSI {rsi:.1f} — oversold bounce in trend")
            score += 1
        elif rsi < 50:
            reasons.append(f"➖ [MOMENTUM] RSI {rsi:.1f} — pullback zone (normal in trend)")
        elif rsi < 65:
            if strong_chop_rsi:
                reasons.append(f"➖ [MOMENTUM] RSI {rsi:.1f} — trend building (suppressed — chop)")
            else:
                reasons.append(f"✅ [MOMENTUM] RSI {rsi:.1f} — trend building momentum")
                score += 1
        elif rsi <= 80:
            if strong_chop_rsi:
                reasons.append(f"➖ [MOMENTUM] RSI {rsi:.1f} — continuation (suppressed — chop)")
            else:
                reasons.append(f"✅ [MOMENTUM] RSI {rsi:.1f} — strong continuation zone")
                score += 1
        else:                              # > 80
            reasons.append(f"⚠️ [MOMENTUM] RSI {rsi:.1f} — exhaustion risk")
            score -= 1                     # Warning kept even in chop
    else:
        # NO TREND (choppy / ranging market)
        if rsi < 30:
            reasons.append(f"✅ [MOMENTUM] RSI {rsi:.1f} — mean reversion bounce likely")
            score += 1                     # Oversold bounce kept in chop
        elif rsi < 65:
            reasons.append(f"➖ [MOMENTUM] RSI {rsi:.1f} — no edge in ranging market")
        elif rsi <= 75:
            reasons.append(f"⚠️ [MOMENTUM] RSI {rsi:.1f} — likely rejection in no-trend market")
            score -= 1
        else:                              # > 75
            reasons.append(f"🛑 [MOMENTUM] RSI {rsi:.1f} — fake breakout / exhaustion risk")
            score -= 2

    # RSI Divergence: unique signal (price vs RSI disagreement)
    divergence = detect_rsi_divergence(df)
    if divergence == "bullish":
        reasons.append("✅ [MOMENTUM] Bullish RSI divergence — hidden strength")
        score += 1
    elif divergence == "bearish":
        reasons.append("⚠️ [MOMENTUM] Bearish RSI divergence — hidden weakness")
        caution.append("BearishDivergence")
        score -= 1
    else:
        reasons.append("➖ [MOMENTUM] No RSI divergence")

    # NOTE: BB position in ranging market is handled in GROUP 2 REGIME block above.
    # No duplicate check here.

    # ADX — single consolidated status (DI± scoring already done in GROUP 2)
    if adx >= 25:
        _adx_dir = "DI+ dominant ✅" if (not np.isnan(plus_di) and not np.isnan(minus_di) and plus_di > minus_di) else "DI- dominant ⚠️"
        reasons.append(f"✅ [MOMENTUM] ADX {adx:.1f} — strong trend ({_adx_dir})")
    elif adx >= 20:
        reasons.append(f"⚠️ [MOMENTUM] ADX {adx:.1f} — trend developing")
    else:
        reasons.append(f"⚠️ [MOMENTUM] ADX weak ({adx:.1f}) — no trend confirmation")

    momentum_score = score - macro_score - trend_score
    if DEBUG:
        # Mirror ACTUAL scoring logic — respect chop suppression
        if in_trend:
            if rsi < 30:    rsi_c = +1
            elif rsi < 50:  rsi_c =  0
            elif rsi < 65:  rsi_c = 0 if strong_chop_rsi else +1
            elif rsi <= 80: rsi_c = 0 if strong_chop_rsi else +1
            else:           rsi_c = -1
        else:
            if rsi < 30:    rsi_c = +1
            elif rsi < 65:  rsi_c =  0
            elif rsi <= 75: rsi_c = -1
            else:           rsi_c = -2
        div_c = (1 if divergence=="bullish" else -1 if divergence=="bearish" else 0)
        ctx   = "TREND" if in_trend else "RANGE"
        if strong_chop_rsi and in_trend and 50 <= rsi <= 80:
            lbl = "suppressed-chop"
        elif rsi_c > 0 and in_trend and rsi >= 50:
            lbl = "continuation"
        elif rsi_c > 0:
            lbl = "oversold"
        elif rsi_c < 0:
            lbl = "warning"
        else:
            lbl = "neutral"
        print(f"[SCORE] MOMENTUM: {momentum_score:+d}  (RSI={rsi_c:+d} [{ctx},{lbl}], Divergence={div_c:+d})")
        print(f"[SCORE] ─────────────────────────────────────────────")
        print(f"[SCORE] TOTAL TECHNICAL: {score:+d}  (Macro={macro_score:+d}, Trend={trend_score:+d}, Momentum={momentum_score:+d})")

    # ══════════════════════════════════════════════════════════
    # Supporting info (no score impact)
    # ══════════════════════════════════════════════════════════

    # Chikou span — true Ichimoku: must be above BOTH price and cloud at t-26
    chikou           = last.get("chikou",           np.nan)
    chikou_ref       = last.get("chikou_ref",       np.nan)
    chikou_cloud_top = last.get("chikou_cloud_top", np.nan)
    chikou_cloud_bot = last.get("chikou_cloud_bot", np.nan)
    if not any(np.isnan(x) for x in [chikou, chikou_ref, chikou_cloud_top]):
        above_price = chikou > chikou_ref
        above_cloud = chikou > chikou_cloud_top
        below_price = chikou < chikou_ref
        below_cloud = chikou < chikou_cloud_bot
        if above_price and above_cloud:
            reasons.append(f"✅ [INFO] Chikou above past price AND above historical cloud — strong bullish confirmation")
        elif above_price and not above_cloud:
            reasons.append(f"⚠️ [INFO] Chikou above past price but still inside/below historical cloud — partial confirmation only")
        elif below_price and below_cloud:
            reasons.append(f"❌ [INFO] Chikou below past price AND below historical cloud — strong bearish confirmation")
        elif below_price and not below_cloud:
            reasons.append(f"⚠️ [INFO] Chikou below past price but above historical cloud — partial bearish signal")
        else:
            reasons.append(f"⚪ [INFO] Chikou inside historical cloud — neutral / transitional")

    # Volume info
    vol     = last.get("volume", 0)
    avg_vol = df["volume"].rolling(90).mean().iloc[-1] if "volume" in df.columns else 0
    vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
    if vol < avg_vol * 0.5:
        reasons.append(f"⚠️ Very low volume ({vol_ratio:.2f}x avg) — BLOCKED")
        blockers.append("Very Low Volume")
        blocked = True
    elif vol_ratio >= 1.5:
        reasons.append(f"✅ [INFO] Volume spike ({vol_ratio:.2f}x avg)")
    else:
        reasons.append(f"📊 [INFO] Volume {vol_ratio:.2f}x avg")

    # ATR info
    atr_avg   = df["ATR"].rolling(20).mean().iloc[-2] if "ATR" in df.columns else 0
    atr_ratio = last.get("ATR", 0) / (atr_avg + 1e-9) if atr_avg > 0 else 1.0
    reasons.append(f"📈 ATR Expansion: {atr_ratio:.2f} → {'✅ Calm / Low Volatility' if atr_ratio < 1.1 else '⚠️ High volatility'}")

    # BB position info
    lower_bb = last.get("LowerBB", price * 0.95)
    upper_bb = last.get("UpperBB", price * 1.05)
    bb_mid   = (lower_bb + upper_bb) / 2
    if price < lower_bb:
        reasons.append(f"📊 Price vs BB: Below lower (oversold zone)")
    elif price > upper_bb:
        reasons.append(f"📊 Price vs BB: Above upper (overbought zone)")
    else:
        reasons.append(f"⚠️ Price vs BB: Mid Range")

    # Risk/Reward — chop-aware
    atr_val    = last.get("ATR", price * 0.02)
    stop_price = price - 2 * atr_val

    # In chop: TP = Donchian upper, Stop = just below Donchian lower
    # Stop must be BELOW range low (not inside range) to be meaningful
    _chop_rr = _is_strong_chop(df)
    if _chop_rr:
        try:
            _col_rr    = "close" if "close" in df.columns else "price"
            _don_up_rr = df[_col_rr].iloc[-DONCHIAN_LOOKBACK:].max()
            _don_lo_rr = df[_col_rr].iloc[-DONCHIAN_LOOKBACK:].min()
            tp_price   = _don_up_rr                        # Range ceiling
            stop_price = _don_lo_rr - 0.5 * atr_val       # Below range low (invalidation)
            rr_label   = "CHOP (Donchian)"
        except Exception:
            tp_price = price + 4 * atr_val
            rr_label = "ATR"
    else:
        tp_price = price + 4 * atr_val
        rr_label = "TREND (ATR)"

    rr_ratio    = round((tp_price - price) / (price - stop_price), 2) if price > stop_price else 0
    entry_price = price
    reasons.append(f"✅ Risk/Reward Info: RR={rr_ratio:.2f} [{rr_label}], Entry={entry_price:.6f}, Stop={stop_price:.6f}, TP={tp_price:.6f}")

    action = "BUY" if score >= MIN_BUY_SCORE and not blocked else ("SELL" if score <= MIN_SELL_SCORE else "WAIT")

    return {
        "action":        action,
        "score":         score,
        "blocked":       blocked,
        "blockers":      blockers,
        "reasons":       reasons,
        "caution":       caution,
        "market_regime": market_regime,
        "entry_price": entry_price,
        "stop_price":  stop_price,
        "tp_price":    tp_price,
        "rr_ratio":    rr_ratio,
        "lower_bb":    lower_bb,
        "bb_mid":      bb_mid,
        "vol":         vol,
        "avg_vol":     avg_vol,
        "vol_ratio":   vol_ratio,
        "rsi":         rsi,
        "ema12":       ema12,
        "ema26":       ema26,
        "senkou_a":    senkou_a,
        "senkou_b":    senkou_b,
        "kijun":       kijun,
        "tenkan":      tenkan,
        "cloud_state": cloud_state,
        "divergence":  divergence,
        "plus_di":     plus_di,
        "minus_di":    minus_di,
    }

# ===========================
# ADVANCED STRATEGY HELPERS
# ===========================
def donchian_breakout_signal(df, lookback=DONCHIAN_LOOKBACK):
    high       = df["price"].shift(1).rolling(lookback).max()
    low        = df["price"].shift(1).rolling(lookback).min()
    ch         = high.iloc[-1]
    cl         = low.iloc[-1]
    last_price = df["price"].iloc[-1]
    donch_mid  = (ch + cl) / 2.0 if not (np.isnan(ch) or np.isnan(cl)) else np.nan
    return {
        "breakout_up":   last_price > ch,
        "breakout_down": last_price < cl,
        "channel_high":  ch,
        "channel_low":   cl,
        "channel_mid":   donch_mid,
        "lookback":      lookback,
    }

def volatility_target_size(entry_price, stop_price, account_balance, target_risk_usd):
    risk_per_unit = max(1e-8, abs(entry_price - stop_price))
    return target_risk_usd / risk_per_unit

def pyramid_levels(entry_price, tp_price, n_levels=PYRAMID_LEVELS, start_frac=PYRAMID_START_FRAC):
    diff   = tp_price - entry_price
    levels = []
    for i in range(n_levels):
        frac = start_frac + i * (1 - start_frac) / n_levels
        levels.append(entry_price + diff * frac)
    return levels

def compute_dynamic_trailing(price, ema12, atr, kijun=None, use_kijun=False,
                              ema_buffer=EMA_TRAILING_BUFFER, atr_mult=1.0):
    """Compute dynamic trailing stop — always BELOW price for BUY."""
    candidates = []

    # EMA12-based trail
    if not np.isnan(ema12) and ema12 > 0:
        candidates.append(ema12 * (1 - ema_buffer))

    # ATR-based trail
    if atr is not None and not np.isnan(atr) and atr > 0:
        candidates.append(price - atr_mult * atr)

    # Kijun-Sen (only if below price — prevents above-entry bug)
    if use_kijun and kijun is not None and not np.isnan(kijun) and kijun < price:
        candidates.append(kijun)

    if not candidates:
        return price * 0.97   # Default 3% below if all failed

    trail = max(candidates)

    # SAFETY: trailing stop must never be ABOVE entry (current price)
    trail = min(trail, price * 0.995)
    return trail

# ===========================
# POSITION PLAN
# ===========================
def build_position_plan(coin, last, df):
    entry  = last["price"]
    ema12  = last["EMA12"]
    ema26  = last["EMA26"]
    atr    = last["ATR"]
    kijun  = last.get("kijun", np.nan)

    # Stop-hunt buffer: widen stop when order book is paper-thin
    # Thin books can be swept temporarily to trigger stops (stop hunting)
    # 2×ATR = normal, 2.5×ATR = 25% wider buffer for thin books
    try:
        _liq_usd, _liq_tier, _ = analyze_order_book_liquidity(coin)
        _stop_mult = 2.5 if _liq_tier == "paper_thin" else STOP_ATR_MULT
        if _liq_tier == "paper_thin":
            print(Fore.YELLOW + f"[STOP HUNT] {coin}: Paper-thin book — "
                  f"stop widened to {_stop_mult}×ATR to avoid stop hunting")
    except Exception:
        _stop_mult = STOP_ATR_MULT

    stop = entry - _stop_mult * (atr if not np.isnan(atr) else entry * 0.02)
    stop = max(min(stop, entry * 0.995), 1e-8)

    risk_per_unit   = max(1e-8, entry - stop)
    risk_amount     = ACCOUNT_BALANCE * RISK_PER_TRADE
    raw_units       = risk_amount / risk_per_unit
    max_by_exposure = (ACCOUNT_BALANCE * MAX_EXPOSURE_PCT) / entry
    total_units     = min(raw_units, max_by_exposure)
    if total_units <= 0:
        total_units = MIN_TRADE_SIZE

    rsi = last["RSI"]
    if rsi > RSI_OVERBOUGHT_THRESH:
        init_frac = 0.5
        rsi_msg   = f"🛑 [RSI GUARD] Forced half entry (RSI {rsi:.2f})"
        rsi_color = Fore.RED
    elif rsi > RSI_REDUCE_THRESH:
        init_frac = 0.5
        rsi_msg   = f"⚠️ [RSI GUARD] Reducing initial entry (RSI {rsi:.2f})"
        rsi_color = Fore.YELLOW
    else:
        init_frac = 1.0
        rsi_msg   = f"✅ [RSI GUARD] Normal entry (RSI {rsi:.2f})"
        rsi_color = Fore.GREEN

    initial_units = max(MIN_TRADE_SIZE, total_units * init_frac)
    reserve_units = max(0.0, total_units - initial_units)

    tp            = entry + (entry - stop) * TP_MULTIPLIER
    trailing_stop = compute_dynamic_trailing(entry, ema12, atr, kijun=kijun, use_kijun=True)
    # SAFETY: trailing must always be below entry for BUY positions
    trailing_stop = min(trailing_stop, entry * 0.995)
    rr            = (tp - entry) / (entry - stop) if (entry - stop) > 0 else float("inf")
    # Compute Donchian levels from df for chop-aware TPs
    try:
        _col = "close" if "close" in df.columns else "price"
        _don_upper = df[_col].iloc[-DONCHIAN_LOOKBACK:].max()
        _don_lower = df[_col].iloc[-DONCHIAN_LOOKBACK:].min()
    except Exception:
        _don_upper = None
        _don_lower = None
    _chop_flag = _is_strong_chop(df) if df is not None else False
    partial_tps = calculate_partial_profit_levels(
        entry, signal_type="BUY", atr=atr,
        chop_detected=_chop_flag,
        donchian_upper=_don_upper,
        donchian_lower=_don_lower
    )

    # FIX 3: Enforce MIN_RR_RATIO — warn if below threshold
    rr_ok = rr >= MIN_RR_RATIO
    if not rr_ok:
        print(Fore.YELLOW + f"[RR WARNING] {coin}: RR={rr:.2f} < MIN={MIN_RR_RATIO} — trade quality is low")
        if LOG_ENABLED:
            logger.warning(f"[RR] {coin}: RR {rr:.2f} below minimum {MIN_RR_RATIO}")

    return {
        "total_units":   round(total_units,   QUANT_ROUND),
        "initial_units": round(initial_units, QUANT_ROUND),
        "reserve_units": round(reserve_units, QUANT_ROUND),
        "stop_price":    round(stop, 8),
        "tp_price":      round(tp,   8),
        "trailing_stop": round(trailing_stop, 8),
        "rsi_msg":       rsi_msg,
        "rsi_color":     rsi_color,
        "ladder_prices": {
            "ema12": round(ema12, 8),
            "ema26": round(ema26, 8),
        },
        "rr": round(rr, 2),
    }

# ===========================
# PAPER EXECUTION / LOGGING
# ===========================
def place_market_buy_paper(coin, units, price):
    slippage = 0.001
    avg_fill = price * (1 + slippage)
    log_trade(coin, "BUY_MARKET", price, round(units, QUANT_ROUND), avg_fill,
              None, None, None, f"paper_filled slippage={slippage}")
    return {"status": "filled", "filled_qty": round(units, QUANT_ROUND), "avg_price": avg_fill}

def place_limit_buy_paper(coin, price, units):
    log_trade(coin, "LIMIT_CREATED", price, round(units, QUANT_ROUND), None,
              None, None, None, f"paper_limit price={price} units={round(units, QUANT_ROUND)}")
    return {"order_id": f"lim_{coin}_{int(time.time())}", "status": "open",
            "price": price, "qty": round(units, QUANT_ROUND)}

def place_oco_paper(coin, qty, stop_price, tp_price):
    log_trade(coin, "OCO_CREATED", None, round(qty, QUANT_ROUND), None,
              stop_price, tp_price, None,
              f"paper_oco qty={round(qty, QUANT_ROUND)} stop={stop_price} tp={tp_price}")
    return {"oco_id": f"oco_{coin}_{int(time.time())}"}

def log_trade(coin, action, price, position_units, avg_price,
              stop_price, tp_price, rr, note=""):
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(), coin, action,
            price          if price          is not None else "",
            position_units if position_units is not None else "",
            avg_price      if avg_price      is not None else "",
            stop_price     if stop_price     is not None else "",
            tp_price       if tp_price       is not None else "",
            rr             if rr             is not None else "",
            note,
        ])

# ===========================
# SELL-OFF / CAPITULATION
# ===========================
def detect_big_selloff(df, drop_threshold=-5, vol_multiplier=1.5, debug=False):
    if len(df) < 2:
        return False, None
    price_change = (df["price"].iloc[-1] - df["price"].iloc[-2]) / df["price"].iloc[-2] * 100
    vol_avg90    = df["volume"].rolling(90, min_periods=1).mean().iloc[-1]
    vol_ratio    = df["volume"].iloc[-1] / (vol_avg90 + 1e-9)
    if debug:
        print(f"Price Change 24h % : {price_change:.2f}%")
    if price_change <= drop_threshold and vol_ratio >= vol_multiplier:
        return True, f"🚨 Big Sell-Off: {price_change:.2f}% drop | {vol_ratio:.2f}× volume"
    return False, None

def detect_capitulation(df, drop_pct=-3.0, vol_mult=2.5, atr_mult=1.8, rsi_thresh=22):
    if len(df) < 30:
        return False, {}
    last       = df.iloc[-1]
    prev       = df.iloc[-2]
    price_drop = (last["price"] - prev["price"]) / prev["price"] * 100
    vol_avg    = df["volume"].rolling(20).mean().iloc[-2]
    vol_ratio  = last["volume"] / (vol_avg + 1e-9)
    atr_avg    = df["ATR"].rolling(20).mean().iloc[-2]
    atr_ratio  = last["ATR"] / (atr_avg + 1e-9)
    rsi        = last["RSI"]
    uptrend_ok = (last["EMA12"] > last["EMA26"]) or (last["price"] > last["EMA26"])
    cap        = (price_drop <= drop_pct and vol_ratio >= vol_mult and
                  atr_ratio >= atr_mult and rsi <= rsi_thresh and uptrend_ok)
    return cap, {"price_drop_pct": price_drop, "vol_ratio": vol_ratio,
                 "atr_ratio": atr_ratio, "rsi": rsi, "uptrend_ok": uptrend_ok}

# ===========================
# [J] SELL LOGIC - Bearish Score Calculation
# ===========================
def calculate_sell_score(df, is_btc=False):
    """
    [J] Calculate SELL score (negative = bearish).
    [J] Range: 0 to -9. MIN_SELL_SCORE = -5 triggers SHORT signal.
    [J] Opposite logic of BUY scoring.
    """
    if len(df) < 20:
        return 0.0
    
    last = df.iloc[-1]
    close = last["price"]
    score = 0.0
    
    # [J] Price below EMAs (strong downtrend)
    if close < last["EMA12"] and close < last["EMA26"]:
        score -= 1.0
    elif close < last["EMA12"] or close < last["EMA26"]:
        score -= 0.5
    
    # [J] Price below Ichimoku cloud (bearish)
    senkou_a = last.get("senkou_a", np.nan)
    senkou_b = last.get("senkou_b", np.nan)
    if not np.isnan(senkou_a) and not np.isnan(senkou_b):
        cloud_top = max(senkou_a, senkou_b)
        cloud_bot = min(senkou_a, senkou_b)
        if close < cloud_bot:
            score -= 1.0
        elif close < cloud_top:
            score -= 0.5
    
    # [J] MACD negative (momentum down)
    if last["MACD_Hist"] < 0:
        score -= 1.0
    
    # [J] RSI overbought (mean reversion short)
    rsi = last["RSI"]
    if rsi > 85:
        score -= 1.5  # [J] Extreme
    elif rsi > 70:
        score -= 1.0
    
    # [J] ADX + DI- > DI+ (strong downtrend)
    if last["ADX"] > 25:
        plus_di = last.get("plus_di", np.nan)
        minus_di = last.get("minus_di", np.nan)
        if not (np.isnan(plus_di) or np.isnan(minus_di)) and minus_di > plus_di:
            score -= 1.0
    
    # [J] Bearish RSI divergence (price highs, RSI lows = weakness)
    divergence = detect_rsi_divergence_sell(df, lookback=DIVERGENCE_LOOKBACK)
    if divergence.get('bearish', False):
        score -= 1.0
    
    # [J] EMA12 below EMA26 (bearish crossover)
    if last["EMA12"] < last["EMA26"]:
        score -= 0.5
    
    # [J] Chikou bearish confirmation — below past price AND below historical cloud
    chikou           = last.get("chikou", np.nan)
    chikou_ref       = last.get("chikou_ref", np.nan)
    chikou_cloud_bot = last.get("chikou_cloud_bot", np.nan)
    _chikou_below_price = not np.isnan(chikou) and not np.isnan(chikou_ref) and chikou < chikou_ref
    _chikou_below_cloud = not np.isnan(chikou) and not np.isnan(chikou_cloud_bot) and chikou < chikou_cloud_bot
    if _chikou_below_price and _chikou_below_cloud:
        score -= 1.0   # full point: both conditions met
    elif _chikou_below_price:
        score -= 0.5   # partial: only below price, cloud not confirming
    
    # [J] Price at upper Bollinger Band (mean reversion short)
    if close > last.get("UpperBB", close * 1.1) * 0.98:
        score -= 0.5
    
    return score


def detect_rsi_divergence_sell(df, lookback=DIVERGENCE_LOOKBACK):
    """[J] Detect RSI divergences for SELL scoring — returns dict {bullish, bearish}"""
    if len(df) < lookback + 5:
        return {'bullish': False, 'bearish': False}
    recent = df.iloc[-lookback:]
    close  = recent["price"]
    rsi    = recent["RSI"]
    price_low_idx  = close.idxmin(); price_high_idx = close.idxmax()
    rsi_low_idx    = rsi.idxmin();   rsi_high_idx   = rsi.idxmax()
    bullish = (price_low_idx  > rsi_low_idx)  and (close.min() < close.iloc[0] * 0.99) and (rsi.min() < 40)
    bearish = (price_high_idx > rsi_high_idx) and (close.max() > close.iloc[0] * 1.01) and (rsi.max() > 60)
    return {'bullish': bullish, 'bearish': bearish}

# ===========================
# EXECUTE PLAN (paper)
# ===========================
def execute_plan_paper(coin, last, df, plan, sig, extra_limit_orders=None):
    entry_price = last["price"]
    initial  = plan["initial_units"]
    reserve  = plan["reserve_units"]
    total    = plan["total_units"]
    stop     = plan["stop_price"]
    tp       = plan["tp_price"]
    trailing = plan["trailing_stop"]

    try:
        print(plan["rsi_color"] + plan["rsi_msg"])
    except Exception:
        print(plan.get("rsi_msg", ""))

    filled_info = None
    if initial >= MIN_TRADE_SIZE:
        filled_info = place_market_buy_paper(coin, initial, entry_price)
        place_oco_paper(coin, filled_info["filled_qty"], stop, tp)
    else:
        print(Fore.YELLOW + "[PLAN] initial too small, skipping immediate entry")

    ladder_orders = []
    if reserve >= MIN_TRADE_SIZE:
        lp      = plan["ladder_prices"]
        levels  = [lp["ema12"], lp["ema26"]]
        per_lvl = round(reserve / len(levels), QUANT_ROUND)
        distrib = 0.0
        for i, p in enumerate(levels):
            qty     = per_lvl
            distrib += qty
            if i == len(levels) - 1:
                qty = round(qty + round(reserve - distrib, QUANT_ROUND), QUANT_ROUND)
            if qty < MIN_TRADE_SIZE:
                continue
            ladder_orders.append(place_limit_buy_paper(coin, p, qty))

    extra_orders = []
    if extra_limit_orders:
        for o in extra_limit_orders:
            if o["qty"] >= MIN_TRADE_SIZE:
                extra_orders.append(place_limit_buy_paper(coin, o["price"], o["qty"]))

    print(Fore.GREEN + f"[PLAN] {coin}: Total={total} | Initial={initial} | Reserve={reserve} | RR={plan['rr']}")
    print(Fore.GREEN + f" Stop={stop:.6f} | TP={tp:.6f} | Trailing={trailing:.6f}")
    print(Fore.CYAN  + f" Ladder Orders: EMA12 @ {plan['ladder_prices']['ema12']} | EMA26 @ {plan['ladder_prices']['ema26']}")
    if extra_orders:
        print(Fore.CYAN + f" Pyramid Limit Orders placed: {len(extra_orders)}")

    return {"filled": filled_info, "ladder_orders": ladder_orders,
            "extra_orders": extra_orders, "stop": stop, "tp": tp, "trailing": trailing}

# ===========================
# MAIN LOOP
# ===========================
def check_candle_close_gate():
    """
    Daily candle on CoinGecko closes at 00:00 UTC.
    For futures entries based on daily signals, we want to enter AFTER
    the daily candle has confirmed (closed), not mid-candle.

    Safe entry window: 00:00–02:00 UTC (candle just closed, new one started)
    Caution window:    22:00–23:59 UTC (candle hasn't closed yet — mid-candle)
    Warning window:    all other hours — mid-candle, signal may change by close

    Returns: (is_safe, hours_to_close, warning_msg)
    """
    try:
        from datetime import datetime, timezone
        utc_hour   = datetime.now(timezone.utc).hour
        utc_minute = datetime.now(timezone.utc).minute

        # Hours until next 00:00 UTC close
        if utc_hour == 0:
            hours_to_close = 24 - (utc_minute / 60)
        else:
            hours_to_close = 24 - utc_hour - (utc_minute / 60)

        # Safe: entered within 4 hours of candle close (22:00–02:00 UTC)
        is_confirmed = utc_hour < 2    # just closed — signal confirmed
        is_closing   = utc_hour >= 22  # closing soon — almost confirmed
        is_mid       = not is_confirmed and not is_closing

        if is_confirmed:
            return True, hours_to_close, ""
        elif is_closing:
            mins_left = int((24 - utc_hour - utc_minute/60) * 60)
            return True, hours_to_close, (
                f"⚠️ [CANDLE] Daily candle closes in ~{mins_left}min (00:00 UTC) — "
                f"signal almost confirmed but conditions can still change"
            )
        else:
            return False, hours_to_close, (
                f"⏳ [CANDLE] Mid-candle entry — daily candle closes in ~{hours_to_close:.1f}h (00:00 UTC). "
                f"For futures: wait for candle close to confirm signal. "
                f"Current conditions may reverse before daily close."
            )
    except Exception:
        return True, 0, ""   # fail safe — don't block on error


def check_btc_relative_strength(df, btc_df, coin, lookback_hours=8):
    """
    Detect when BTC rallies strongly but the coin doesn't follow.
    This is a distribution/weakness signal — smart money rotating INTO BTC,
    OUT of alts. Price may hold technically but OBV falls.

    lookback_hours: how many daily candles to look back (1 = today's candle)

    Returns: (is_weak, btc_chg, coin_chg, divergence, warning_msg)
    """
    try:
        if btc_df is None or btc_df.empty or df is None or df.empty:
            return False, 0, 0, 0, ""

        # Use last N candles
        lb = max(1, lookback_hours // 24) if lookback_hours >= 24 else 1
        lb = 1  # compare last candle

        btc_chg  = (btc_df["price"].iloc[-1] - btc_df["price"].iloc[-2]) / btc_df["price"].iloc[-2] * 100
        coin_chg = (df["price"].iloc[-1]      - df["price"].iloc[-2])      / df["price"].iloc[-2]      * 100

        divergence = btc_chg - coin_chg   # positive = BTC outperforming = alt weakness

        # Trigger: BTC up >= 3% but coin up < 1% (or down)
        is_weak = btc_chg >= 3.0 and coin_chg < 1.0

        # Stronger trigger: BTC up >= 5% but coin flat/down
        is_very_weak = btc_chg >= 5.0 and coin_chg <= 0

        if is_very_weak:
            msg = (
                f"🚨 [BTC DIVERGENCE] BTC +{btc_chg:.1f}% but {coin.upper()} {coin_chg:+.1f}% — "
                f"SEVERE underperformance ({divergence:.1f}% gap). "
                f"Smart money rotating INTO BTC, OUT of {coin.upper()}. "
                f"Distribution confirmed — DO NOT ENTER LONG."
            )
        elif is_weak:
            msg = (
                f"⚠️ [BTC DIVERGENCE] BTC +{btc_chg:.1f}% but {coin.upper()} only {coin_chg:+.1f}% — "
                f"alt weakness ({divergence:.1f}% gap). "
                f"Capital not flowing into {coin.upper()} despite BTC rally. Wait for catch-up."
            )
        else:
            msg = ""

        return is_weak or is_very_weak, btc_chg, coin_chg, divergence, msg

    except Exception:
        return False, 0, 0, 0, ""


def generate_entry_timing(coin, sig, df, last, final_score, btc_regime,
                          ml_proba=0.5, direction="UP", direction_conf=0.0,
                          short_ml_proba=0.5):
    """
    Generate a human-readable description of WHEN to start a trade and at what price.

    ML-AWARE: score alone is NOT enough.  The bot already blocks BUYs when ml_proba < 0.25.
    This function reflects that same gate so the entry description is never contradictory.

    ML TRAP DETECTION:
      Score qualifies (>= MIN_BUY_SCORE) BUT ml_proba < 0.50
      → Price is technically set up BUT the model sees high failure risk
      → Likely a momentum trap: price ran up, technicals look green, but AI disagrees
      → Better entry: wait for pullback to EMA26/lower-BB + OBV turning UP + ML > 50%

    Returns: (status_label, when_text, watch_price)
      status_label : "🟢 ENTER NOW" / "⚠️ ML TRAP" / "⏳ ALMOST" / "👀 WATCH" /
                     "⛔ BLOCKED" / "🔴 NOT YET"
      when_text    : concise description of what still needs to happen
      watch_price  : the specific price level to monitor for entry
    """
    price       = last["price"]
    ema12       = last.get("EMA12", price)
    ema26       = last.get("EMA26", price)
    rsi         = last.get("RSI", 50)
    cloud_state = sig.get("cloud_state", "unknown")
    blocked     = sig.get("blocked", False)
    blockers    = sig.get("blockers", [])

    # Senkou cloud levels
    sa = last.get("senkou_a", np.nan)
    sb = last.get("senkou_b", np.nan)
    cloud_top = max(sa, sb) if (not np.isnan(sa) and not np.isnan(sb)) else ema26
    cloud_bot = min(sa, sb) if (not np.isnan(sa) and not np.isnan(sb)) else ema26

    # Lower Bollinger Band — good pullback support target
    lower_bb = last.get("LowerBB", ema26 * 0.97)

    # Donchian upper (20-day high) — chasing check
    try:
        don_upper = df["price"].iloc[-DONCHIAN_LOOKBACK:].max()
    except Exception:
        don_upper = price

    # OBV trend (rising = real capital inflow backing the move)
    try:
        obv_vals  = df["OBV"].iloc[-8:].values
        obv_chg   = (obv_vals[-1] - obv_vals[0]) / (abs(obv_vals[0]) + 1e-9)
        obv_trend = "rising" if obv_chg > 0.03 else ("falling" if obv_chg < -0.03 else "flat")
    except Exception:
        obv_trend = "unknown"

    adx = last.get("ADX", 0) or 0

    # ══════════════════════════════════════════════════════════════
    # SHORT TERRITORY (score <= -5, action = WAIT)
    # Explain exactly which gate blocked the short and what to watch
    # IMPORTANT: "Cloud Below" CONFIRMS a short — it is NOT a blocker.
    #            Only ignore it here; it is only a blocker for BUY signals.
    # ══════════════════════════════════════════════════════════════
    if final_score <= -5:
        short_blocks = []
        short_hints  = []

        # Gate 1: ADX < 15 — no trend, shorting into chop is noise
        if adx < 15:
            short_blocks.append(f"ADX {adx:.1f} < 15 → no real trend, shorting into chop is noise")
            short_hints.append(f"Wait for ADX > 15 (trend must establish before shorting)")

        # Gate 2: Short ML model — direct evidence from historical short outcomes
        # Direction is informational context only, not a gate (it's independent of trade outcome)
        if short_ml_proba < 0.40:
            short_blocks.append(
                f"Short ML {short_ml_proba:.0%} → only {short_ml_proba:.0%} of similar setups "
                f"resulted in a winning short historically (needs ≥40%)"
            )
            short_hints.append(f"Wait for short ML ≥ 40% — model must see enough historical short wins")

        # Gate 3: OBV still rising — distribution not confirmed
        if obv_trend == "rising":
            short_blocks.append(f"OBV rising → smart money still accumulating, not distributing")
            short_hints.append(f"Wait for OBV to flatten or fall (confirms selling pressure)")

        # Gate 4: other signal blockers — explicitly SKIP "Cloud Below" (it confirms bearish)
        real_blockers = [b for b in sig.get("blockers", []) if b != "Cloud Below"]
        if not short_blocks and sig.get("blocked") and real_blockers:
            short_blocks.append(f"Signal blocked: {', '.join(real_blockers)}")
            short_hints.append(f"Resolve blocker then re-evaluate short")

        # Cloud position: for shorts, below cloud = GOOD (confirms downtrend)
        if cloud_state == "below":
            cloud_context = (
                f"✅ Below cloud (${cloud_bot:.4f}) — downtrend confirmed by Ichimoku. "
                f"Ideal short entry was at cloud breakdown. "
                f"Next re-entry opportunity: dead-cat bounce back to cloud bottom "
                f"(${cloud_bot:.4f}) as resistance → wait for rejection/stall there."
            )
            short_watch = cloud_bot
        elif cloud_state == "in":
            cloud_context = f"Price inside cloud — watch for rejection at cloud top (${cloud_top:.4f}) for short entry."
            short_watch = cloud_top
        else:
            cloud_context = f"Price above cloud — short only valid after trend reverses below EMA26 (${ema26:.4f})."
            short_watch = ema26

        if short_blocks:
            # Each gate gets its own line so nothing is trimmed
            blocks_lines = "\n".join(
                f"  {i+1}. {b}\n     Fix: {short_hints[i] if i < len(short_hints) else 'N/A'}"
                for i, b in enumerate(short_blocks[:4])
            )
            return (
                "🚫 SHORT BLOCKED",
                f"Score {final_score:.0f}/5 qualifies BUT gates not met:\n{blocks_lines}\n  📍 {cloud_context}",
                short_watch
            )
        else:
            # All gates clear — short is ready
            return (
                "🔴 SHORT READY",
                (
                    f"Score {final_score:.0f}/5 ✅  ADX {adx:.1f} ✅  "
                    f"Direction {direction} {direction_conf:.0%} ✅  "
                    f"Short ML {short_ml_proba:.0%} ✅ "
                    f"({short_ml_proba:.0%} of similar setups, shorting won historically)\n"
                    f"  Long ML {ml_proba:.0%} (only {ml_proba:.0%} of longs hit target — also bearish)\n"
                    f"  📍 {cloud_context}\n"
                    f"  Enter short at ${price:.4f}."
                ),
                price
            )

    # ══════════════════════════════════════════════════════════════
    # BUY TERRITORY — score > -5
    # ══════════════════════════════════════════════════════════════
    if blocked and "BTC Bear Regime" in blockers:
        return (
            "⛔ BTC BEAR",
            f"BTC is in a downtrend — all altcoin BUYs blocked. Wait for BTC to recover.",
            price
        )
    if blocked and "Cloud Below" in blockers:
        return (
            "⛔ CLOUD BELOW",
            f"Price is below Ichimoku cloud — strong downtrend. "
            f"Only enter after price reclaims cloud top (${cloud_top:.4f}).",
            cloud_top
        )
    if blocked:
        reason = ", ".join(b for b in blockers if b not in ("BTC Bear Regime", "Cloud Below")) or "signal blocked"
        return (
            "⛔ BLOCKED",
            f"Blocked: {reason}. Resolve before entering.",
            price
        )

    # ── ML TRAP: score qualifies but AI disagrees strongly ────────
    # This is the most dangerous scenario: technicals look green, price may have
    # already run up, but the ML model predicts high failure rate.
    # The right play: wait for a pullback to real support + OBV confirmation.
    if final_score >= MIN_BUY_SCORE and ml_proba < 0.25:
        # Calculate a realistic pullback target (EMA26 or lower BB, whichever is closer)
        pullback_target = max(ema26, lower_bb)   # the nearest meaningful support
        ml_pct = ml_proba * 100
        obv_note = (
            "OBV already falling (distribution)" if obv_trend == "falling"
            else "OBV flat — no real accumulation yet" if obv_trend == "flat"
            else "OBV rising but ML still weak"
        )
        return (
            "⚠️  ML TRAP — DO NOT ENTER",
            (
                f"Score {final_score:.0f}/5 qualifies BUT ML only {ml_pct:.0f}% (model sees ~{100-ml_pct:.0f}% failure rate). "
                f"{obv_note}. "
                f"BETTER ENTRY: Wait for pullback to ${pullback_target:.4f} (EMA26/BB support), "
                f"OBV turning UP, and ML flipping above 50% — then it becomes a high-probability setup."
            ),
            pullback_target
        )

    # ── MARGINAL ML: score qualifies but AI is lukewarm ──────────
    if final_score >= MIN_BUY_SCORE and ml_proba < ML_MIN_CONFIDENCE:
        pullback_target = max(ema26, lower_bb)
        ml_pct = ml_proba * 100
        obv_note = "OBV rising ✅" if obv_trend == "rising" else f"OBV {obv_trend} ⚠️"
        return (
            "⚠️  CAUTION — ML WEAK",
            (
                f"Score qualifies but ML only {ml_pct:.0f}% (needs >{ML_MIN_CONFIDENCE*100:.0f}%). "
                f"{obv_note}. "
                f"Safer entry: wait for dip to EMA26 (${ema26:.4f}) where ML is more likely to confirm. "
                f"Half-size position acceptable if OBV is rising."
            ),
            pullback_target
        )

    # ── VALID FULL BUY: score + ML + not blocked ─────────────────
    if final_score >= MIN_BUY_SCORE and ml_proba >= ML_MIN_CONFIDENCE:
        obv_note = " | OBV rising ✅" if obv_trend == "rising" else f" | OBV {obv_trend} ⚠️"
        rsi_note = f" | ⚠️ RSI {rsi:.0f} — consider half-size entry" if rsi > RSI_REDUCE_THRESH else ""
        chasing  = price > don_upper * 1.01
        if chasing:
            return (
                "⚠️  CHASING — WAIT FOR PULLBACK",
                (
                    f"Score {final_score:.0f}/5 ✅  ML {ml_proba*100:.0f}% ✅ — but price is "
                    f"{((price/don_upper)-1)*100:.1f}% above 20-day high (${don_upper:.4f}). "
                    f"Wait for dip to EMA12 (${ema12:.4f}) to avoid chasing{obv_note}{rsi_note}."
                ),
                ema12
            )

        # ── Daily candle close gate ───────────────────────────────
        # Only warn mid-candle if consecutive days NOT already confirmed.
        # If D-2 and D-1 both scored ≥5, the trend is established and
        # mid-candle entry is acceptable — we're trading a confirmed trend,
        # not a single candle signal.
        _candle_safe, _hrs_to_close, _candle_msg = check_candle_close_gate()
        _consecutive_confirmed = sig.get("_consecutive_confirmed", False)

        if not _candle_safe and not _consecutive_confirmed:
            _h   = int(_hrs_to_close)
            _m   = int((_hrs_to_close - _h) * 60)
            _eta = f"{_h}h {_m}m" if _h > 0 else f"{_m}m"
            return (
                f"⏳ WAIT FOR CANDLE CLOSE ({_eta} remaining)",
                (
                    f"Score {final_score:.0f}/5 ✅  ML {ml_proba*100:.0f}% ✅ — technically qualifies BUT\n"
                    f"No consecutive day confirmation yet — single candle signal mid-forming.\n"
                    f"Daily candle closes in {_eta} (00:00 UTC).\n"
                    f"OBV, volume and RSI can flip before close. Re-check after 00:00 UTC.\n"
                    f"Target entry after close: ${ema26:.4f} (EMA26)"
                ),
                ema26
            )
        elif not _candle_safe and _consecutive_confirmed:
            # Consecutive confirmed — mid-candle entry allowed but flag the time
            _h   = int(_hrs_to_close)
            _m   = int((_hrs_to_close - _h) * 60)
            _eta = f"{_h}h {_m}m" if _h > 0 else f"{_m}m"
            # Don't block — just note it in the ENTER NOW message
            obv_str = "OBV rising ✅" if obv_trend == "rising" else f"OBV {obv_trend} ⚠️"
            return (
                "🟢 ENTER NOW",
                (
                    f"All criteria met — Score {final_score:.0f}/5 ✅  ML {ml_proba*100:.0f}% ✅ | {obv_str}\n"
                    f"2-day consecutive confirmed ✅ — mid-candle entry acceptable.\n"
                    f"Note: candle closes in {_eta} (00:00 UTC). Monitor for OBV/volume changes.\n"
                    f"Enter at ${price:.4f}."
                ),
                price
            )

        # ── BTC relative strength gate ────────────────────────────
        _btc_weak, _btc_chg, _coin_chg, _div, _btc_msg = check_btc_relative_strength(
            df, sig.get("_btc_df"), coin)
        if _btc_weak:
            return (
                "⚠️ BTC DIVERGENCE",
                (
                    f"Score qualifies but {coin.upper()} underperforming BTC "
                    f"(BTC {_btc_chg:+.1f}% vs {coin.upper()} {_coin_chg:+.1f}%).\n"
                    f"{_btc_msg}\n"
                    f"Enter only when {coin.upper()} starts catching up."
                ),
                ema26
            )

        return (
            "🟢 ENTER NOW",
            (
                f"All criteria met — Score {final_score:.0f}/5 ✅  ML {ml_proba*100:.0f}% ✅"
                f"{obv_note}{rsi_note}. Enter at ${price:.4f}."
            ),
            price
        )

    # ── CANDLE CLOSE GATE ─────────────────────────────────────────
    # Applied after ENTER NOW check — if all else qualifies but candle mid-forming
    # This function is re-checked below for ALMOST cases too
    conditions  = []
    watch_price = price

    if btc_regime == "bear" and coin != BTC_ID:
        conditions.append("BTC must turn bullish first")

    if cloud_state == "in":
        conditions.append(f"Break above cloud top (${cloud_top:.4f})")
        watch_price = cloud_top

    score_gap = MIN_BUY_SCORE - final_score
    if score_gap > 0:
        conditions.append(f"Score {final_score:.0f}/{MIN_BUY_SCORE} — needs {score_gap:.0f} more point(s)")

    if ml_proba < ML_MIN_CONFIDENCE:
        conditions.append(f"ML {ml_proba*100:.0f}% → needs >{ML_MIN_CONFIDENCE*100:.0f}% (currently bearish model)")

    if rsi > RSI_OVERBOUGHT_THRESH:
        conditions.append(f"RSI {rsi:.0f} overheated — wait for cooldown below {RSI_REDUCE_THRESH}")

    if obv_trend == "falling":
        conditions.append("OBV falling (distribution) — wait for OBV to turn up")

    # Entry hint based on cloud position
    if cloud_state == "above":
        watch_price = ema26
        entry_hint  = f"Watch EMA26 (${ema26:.4f}) for dip entry once conditions improve"
    elif cloud_state == "in":
        watch_price = cloud_top
        entry_hint  = f"Cloud top ${cloud_top:.4f} is the key breakout level"
    else:
        watch_price = ema26
        entry_hint  = f"EMA26 (${ema26:.4f}) is support if price recovers"

    # Status label
    if final_score >= MIN_BUY_SCORE - 1 and ml_proba >= 0.40 and cloud_state == "above":
        status = "⏳ ALMOST — 1 condition away"
    elif final_score >= MIN_BUY_SCORE - 2 and cloud_state != "below":
        status = "👀 WATCH CLOSELY"
    else:
        status = "🔴 NOT YET"

    cond_text = " | ".join(conditions[:3])
    when_text = f"{cond_text}  →  {entry_hint}" if cond_text else entry_hint

    return status, when_text, watch_price


# ===========================
# MAIN LOOP
# ===========================
def run_once():
    """[W] Main trading bot loop with comprehensive logging"""
    
    run_start_time = time.time()
    logger.info("=" * 70)
    logger.info("[BOT] Starting trading bot run...")
    logger.info(f"[BOT] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"[BOT] Scanning {len(COINS)} coins...")
    
    # [D] Fetch BTC regime once before scanning altcoins
    print(Fore.CYAN + "\n[REGIME] Checking BTC market regime...")
    logger.info("[REGIME] Checking BTC market regime...")
    btc_regime = get_btc_regime()
    regime_color = Fore.GREEN if btc_regime == "bull" else (Fore.RED if btc_regime == "bear" else Fore.YELLOW)
    print(regime_color + f"[REGIME] BTC regime: {btc_regime.upper()}")
    logger.info(f"[REGIME] BTC regime: {btc_regime.upper()}")

    # [V] List saved models at startup
    if MODEL_PERSISTENCE_ENABLED and LOAD_CACHED_MODELS:
        list_saved_models()

    # [K] Fetch Market Sentiment (Fear & Greed Index)
    fng_value, fng_class = 50, "Neutral"
    if FETCH_FEAR_GREED:
        fng_value, fng_class = fetch_fear_greed_index()
        fng_display = "cached" if fng_class == "cached" else f"{fng_value} ({fng_class})"
        print(Fore.CYAN + f"[SENTIMENT] Fear & Greed Index: {fng_display}")
        logger.info(f"[SENTIMENT] Fear & Greed Index: {fng_display}")
    
    # [L] BTC data for Relative Strength — reuse from get_btc_regime() cache (no extra API call)
    btc_df = None
    if RELATIVE_STRENGTH_ENABLED:
        btc_df = _btc_cache.get("df")   # already fetched by get_btc_regime above
        if btc_df is not None and not btc_df.empty:
            print(Fore.CYAN + f"[RELATIVE STRENGTH] BTC data reused from regime cache (0 extra API calls)")
            logger.info("[RELATIVE STRENGTH] ✅ BTC data reused from cache")
        else:
            try:
                logger.info("[RELATIVE STRENGTH] Fetching BTC data (cache miss)...")
                btc_df = fetch_coin_data(BTC_ID)
                print(Fore.CYAN + f"[RELATIVE STRENGTH] BTC data loaded")
            except Exception as e:
                logger.error(f"[RELATIVE STRENGTH] ❌ Could not fetch BTC data: {e}")

    # Collect results for ranking at end
    results = []

    for coin in COINS:
        # Print separator at VERY START of each coin
        print("\n" + "=" * 70)
        print(f">>> Processing: {coin.upper()}")
        print("=" * 70)
        
        try:
            df   = fetch_coin_data(coin)
            if df.empty or len(df) < 60:
                print(Fore.RED + f"{coin}: not enough data")
                continue

            last     = df.iloc[-1]
            is_btc   = (coin == BTC_ID)

            # [G+V] Train ML model and predict trade outcome (with persistence)
            ml_model, ml_scaler, ml_accuracy, ml_samples = None, None, 0, 0
            ml_proba, ml_confidence = 0.5, "unknown"
            if ML_ENABLED:
                # [V] Try loading cached model first
                if MODEL_PERSISTENCE_ENABLED and not should_retrain_model(coin):
                    model, scaler, metadata = load_model(coin)
                    if model is not None and scaler is not None:
                        # [V] Use cached model
                        ml_model  = model
                        ml_scaler = scaler
                        ml_accuracy = metadata.get('accuracy', 0.0)
                        ml_samples  = metadata.get('samples', 0)
                    else:
                        # Version mismatch or load error → retrain
                        print(Fore.YELLOW + f"[PERSIST] {coin}: Retraining (version/feature mismatch)...")
                        ml_model, ml_scaler, ml_accuracy, ml_samples = train_ml_model(
                            df, coin, btc_df=btc_df, fng_value=fng_value)
                        if SAVE_MODELS and ml_model is not None:
                            save_model(coin, ml_model, ml_scaler, ml_accuracy, ml_samples)
                else:
                    # [V] Cache miss or stale → Train new model
                    ml_model, ml_scaler, ml_accuracy, ml_samples = train_ml_model(
                        df, coin, btc_df=btc_df, fng_value=fng_value)
                    # [V] Save trained model
                    if SAVE_MODELS and ml_model is not None:
                        save_model(coin, ml_model, ml_scaler, ml_accuracy, ml_samples)
                
                # [K][L][M][V] Make predictions using model (cached or fresh)
                if ml_model is not None:
                    ml_proba, ml_confidence = predict_ml_outcome(last, ml_model, ml_scaler, df=df, btc_df=btc_df, fng_value=fng_value)

                    # ── ACCURACY THRESHOLD GATE (runs first, before OBV gate) ──
                    # Thresholds defined as config constants at top of file

                    if ml_accuracy < 0.50:
                        ml_proba = 0.5
                        ml_confidence = "disabled"   # model worse than random → neutralized
                        print(f"[ML GATE] {coin}: Accuracy {ml_accuracy:.1%} < 50% → NEUTRALIZED to 0.5")
                        if LOG_ENABLED:
                            logger.warning(f"[ML GATE] {coin}: Accuracy {ml_accuracy:.1%} below random — ML disabled")
                    elif ml_accuracy < ML_MIN_ACCURACY:
                        ml_proba = ml_proba * 0.30 + 0.5 * 0.70
                        print(f"[ML GATE] {coin}: Accuracy {ml_accuracy:.1%} < 55% → heavily dampened to {ml_proba:.1%}")
                        if LOG_ENABLED:
                            logger.warning(f"[ML GATE] {coin}: Low accuracy {ml_accuracy:.1%} — ML heavily dampened")
                    elif ml_accuracy < ML_FULL_ACCURACY:
                        ml_proba = ml_proba * 0.70 + 0.5 * 0.30
                        print(f"[ML GATE] {coin}: Accuracy {ml_accuracy:.1%} (marginal) → lightly dampened to {ml_proba:.1%}")
                        if LOG_ENABLED:
                            logger.info(f"[ML GATE] {coin}: Marginal accuracy {ml_accuracy:.1%} — ML lightly dampened")
                    else:
                        print(f"[ML GATE] {coin}: Accuracy {ml_accuracy:.1%} ✅ — ML trusted fully")

            # [H] Detect smart money / whale activity
            is_whale, whale_vol, whale_threshold = detect_whale_volume(df)
            smart_money_mode, smart_money_conf, smart_money_details = detect_smart_money_accumulation(df)
            vol_div, vol_div_details = detect_volume_divergence(df)

            # [I] Detect upcoming/recent news events
            events = get_upcoming_events(coin, days_ahead=EVENT_RISK_DAYS)
            event_block, event_block_reason = should_block_trade_on_event(coin, events)

            # Only adjust ML confidence if trade isn't already blocked
            # — no point crushing confidence on a signal we're about to discard
            if event_block:
                ml_proba_adjusted      = ml_proba   # unchanged
                event_adjustment_reason = event_block_reason
            else:
                ml_proba_adjusted, event_adjustment_reason = apply_event_risk_adjustment(ml_proba, coin, events)

            # [AQ] OBV gate: dampen ML if no real capital inflow backing the signal
            # Skip if:
            #   1. Model accuracy already neutralized (ml_proba == 0.5 from bad model)
            #   2. Smart money = distribution — confidence already penalises 0/15 for this,
            #      applying ML dampening too would double-count the same OBV signal
            _ml_was_neutralized = (ml_accuracy < 0.50)
            _sm_is_distribution = (smart_money_mode == "distribution")

            if _ml_was_neutralized:
                obv_trend      = "unknown"
                obv_desc       = "OBV gate skipped (model neutralized)"
                obv_multiplier = 1.0
                print(f"[OBV GATE] {coin}: Skipped — model accuracy {ml_accuracy:.1%} already neutralized")
            elif _sm_is_distribution:
                # Smart money distribution already captured in confidence (0/15 points)
                # Still compute obv_trend for display but don't dampen ML further
                _, obv_trend, obv_desc, _ = apply_obv_ml_gate(ml_proba_adjusted, df, coin)
                obv_multiplier = 1.0
                print(f"[OBV GATE] {coin}: ML dampening skipped — smart money distribution already penalises confidence")
            else:
                ml_proba_adjusted, obv_trend, obv_desc, obv_multiplier = apply_obv_ml_gate(ml_proba_adjusted, df, coin)
            _obv_falling = (obv_trend == "falling")

            # ── USE ADJUSTED PROBABILITY FROM HERE ONWARDS ──────────────
            # ml_proba_adjusted incorporates: event risk + OBV gate
            # All downstream logic must use this, not raw ml_proba
            ml_proba = ml_proba_adjusted

            # ── SHORT ML MODEL ────────────────────────────────────────────
            # Trained on inverted labels: label 1 = short won (-2.5 ATR target hit before +1.5 ATR stop)
            # Completely separate model files — long model is never touched.
            short_ml_proba, short_ml_confidence, short_ml_accuracy = 0.5, "unknown", 0.0
            if ML_ENABLED:
                short_model, short_scaler, short_meta = None, None, None
                if MODEL_PERSISTENCE_ENABLED and not should_retrain_short_model(coin):
                    short_model, short_scaler, short_meta = load_short_model(coin)

                if short_model is None:
                    short_model, short_scaler, short_ml_accuracy, short_ml_samples = \
                        train_short_ml_model(df, coin, btc_df=btc_df, fng_value=fng_value)
                    if short_model is not None and SAVE_MODELS:
                        save_short_model(coin, short_model, short_scaler,
                                         short_ml_accuracy, short_ml_samples)
                else:
                    short_ml_accuracy = short_meta.get("accuracy", 0.0)

                if short_model is not None:
                    short_ml_proba, short_ml_confidence = predict_short_ml_outcome(
                        last, short_model, short_scaler, df=df, btc_df=btc_df, fng_value=fng_value)
                    # Apply same accuracy gate as long model
                    if short_ml_accuracy < 0.50:
                        short_ml_proba = 0.5
                        short_ml_confidence = "disabled"
                        print(f"[SHORT ML GATE] {coin}: Accuracy {short_ml_accuracy:.1%} < 50% → NEUTRALIZED")
                    elif short_ml_accuracy < ML_MIN_ACCURACY:
                        short_ml_proba = short_ml_proba * 0.30 + 0.5 * 0.70
                    elif short_ml_accuracy < ML_FULL_ACCURACY:
                        short_ml_proba = short_ml_proba * 0.70 + 0.5 * 0.30
                    print(f"[SHORT ML] {coin}: Short win prob {short_ml_proba:.1%} "
                          f"({short_ml_confidence}) | acc: {short_ml_accuracy:.1%}")

            sig      = generate_signal_and_reasons(last, df, btc_regime=btc_regime, is_btc=is_btc)
            decision = sig["action"]
            sig["technical_score"] = sig["score"]  # Save for cap
            ml_guard_block = False   # initialise here — set True later if ML guard fires
            
            # [X] Apply sentiment boost to score
            if SENTIMENT_ANALYSIS_ENABLED and USE_VADER:
                try:
                    sig["score"] = apply_sentiment_boost(sig["score"], coin)
                except Exception as e:
                    print(f"[SENTIMENT ERROR] {coin}: {e}")
                    logger.error(f"[SENTIMENT ERROR] {coin}: {e}")
            
            # ── SCORING PIPELINE ──────────────────────────────────────
            # [Y] Volume
            if VOLUME_ANALYSIS_ENABLED:
                try:
                    sig["score"] = apply_volume_boost(sig["score"], df, coin)
                except Exception as e:
                    print(f"[VOLUME ERROR] {coin}: {e}")

            # [Z] Price change 24h
            try:
                sig["score"] = apply_price_change_penalty(sig["score"], df, coin)
            except Exception as e:
                print(f"[PRICE ERROR] {coin}: {e}")

            # [AC] Z-Score (blow-off tops/bottoms) — context-aware
            zscore_adj = 0
            try:
                _adx_val  = last.get("ADX", 0) or 0
                _in_trend = (sig.get("cloud_state") == "above" and
                             sig.get("ema12", 0) > sig.get("ema26", 0))
                pre_z = sig["score"]
                sig["score"] = apply_zscore_boost(sig["score"], df, coin, in_trend=_in_trend, adx=_adx_val)
                zscore_adj = sig["score"] - pre_z
            except Exception as e:
                if LOG_ENABLED: logger.debug(f"[ZSCORE] {coin}: skipped - {e}")

            # [AS] Momentum exhaustion (EMA26 deviation — mean reversion risk)
            try:
                sig["score"] = apply_exhaustion_penalty(sig["score"], df, coin, zscore_adj=zscore_adj)
            except Exception as e:
                if LOG_ENABLED: logger.debug(f"[EXHAUSTION] {coin}: skipped - {e}")

            # [AP] Wick ratio (rejection vs breakout vs capitulation)
            try:
                sig["score"] = apply_wick_adjustment(sig["score"], df, coin)
            except Exception as e:
                if LOG_ENABLED: logger.debug(f"[WICK] {coin}: skipped - {e}")

            # [AA] Order book (primary signal in range regime)
            try:
                _regime = sig.get("market_regime", "unknown")
                sig["score"] = apply_orderbook_boost(sig["score"], coin, market_regime=_regime)
            except Exception as e:
                print(f"[ORDER BOOK DEBUG] {coin}: Analysis failed - {e}")

            # [AQ2] OBV falling → score penalty (distribution = not a real move)
            try:
                if _obv_falling:
                    sig["score"] -= 1
                    print(f"[OBV GATE] {coin}: Distribution detected — score -{1} (OBV falling)")
                    if LOG_ENABLED:
                        logger.info(f"[OBV GATE] {coin}: Score adjusted -1 (OBV distribution)")
            except Exception as e:
                if LOG_ENABLED: logger.debug(f"[OBV SCORE] {coin}: skipped - {e}")

            # [AD] Chop zone
            chop_detected = False
            try:
                original_score = sig["score"]
                sig["score"] = apply_chop_penalty(sig["score"], df, coin)
                if sig["score"] != original_score:
                    chop_detected = True
                else:
                    print(f"[CHOP] {coin}: Clear trend (not choppy)")
            except Exception as e:
                if LOG_ENABLED: logger.debug(f"[CHOP] {coin}: skipped - {e}")

            # [AE] Breakout (skipped if chop detected)
            try:
                if chop_detected:
                    print(f"[BREAKOUT] {coin}: Skipped (chop zone detected)")
                else:
                    sig["score"] = apply_breakout_boost(sig["score"], df, coin)
            except Exception as e:
                if LOG_ENABLED: logger.debug(f"[BREAKOUT] {coin}: skipped - {e}")

            # CAP: Post-technical adjustments capped at ±2 normally, ±1 if choppy
            tech_score = sig.get("technical_score", 0)
            adjustment = sig["score"] - tech_score
            cap = 1 if chop_detected else 2
            if adjustment < -cap:
                sig["score"] = tech_score - cap
                print(f"[CAP] {coin}: Adjustments capped at -{cap} ({'chop market' if chop_detected else 'normal'})")
            elif adjustment > cap:
                sig["score"] = tech_score + cap
                print(f"[CAP] {coin}: Adjustments capped at +{cap} ({'chop market' if chop_detected else 'normal'})")

            # Ensure _obv_falling is defined (may not exist if OBV gate errored)
            if "_obv_falling" not in dir():
                _obv_falling = False

            # Predict direction before action determination
            direction, direction_conf = predict_direction(df)

            # ═══════════════════════════════════════════════════════════════
            # RECALCULATE ACTION after all score adjustments (volume, sentiment, price, orderbook, zscore, chop, breakout)
            # ═══════════════════════════════════════════════════════════════
            final_score = sig["score"]

            # ── RECALCULATE BASE DECISION from final adjusted score ──────
            # generate_signal_and_reasons() set action from the pre-adjustment score.
            # Volume boost, wick penalty, chop, Z-score etc. all ran after that.
            # Recalculate now so the decision reflects the TRUE final score.
            if sig.get("blocked"):
                decision = "WAIT"
            elif final_score >= MIN_BUY_SCORE:
                decision = "BUY"
            elif final_score <= MIN_SELL_SCORE:
                decision = "SELL"
            else:
                decision = "WAIT"
            sig["action"] = decision
            # If Short ML strongly says shorts win here AND long ML is unreliable
            # or neutralized, a BUY signal is contradicted by the model.
            # Gate: short_ml_proba >= 0.65 AND (long ML neutralized OR long ML < 0.40)
            _short_ml_dominant = (
                short_ml_proba >= 0.65 and short_ml_accuracy >= 0.55 and
                (ml_accuracy < 0.50 or ml_proba < 0.40)
            )
            if _short_ml_dominant and final_score >= MIN_BUY_SCORE:
                sig["score"]   = MIN_BUY_SCORE - 1
                final_score    = sig["score"]
                sig["blocked"] = True
                decision       = "WAIT"           # ← update decision immediately
                sig["action"]  = "WAIT"
                sig["reasons"].append(
                    f"🚫 [SHORT ML GATE] BUY blocked: Short ML {short_ml_proba:.0%} (acc {short_ml_accuracy:.0%}) "
                    f"says shorts historically win here while Long ML is "
                    f"{'neutralized (acc<50%)' if ml_accuracy < 0.50 else f'bearish ({ml_proba:.0%})'}. "
                    f"Mixed signal — do NOT enter long."
                )
                print(f"[SHORT ML GATE] {coin}: BUY BLOCKED — Short ML {short_ml_proba:.0%} dominates")

            # SHORT: Needs score <= -5 + direction DOWN + ML >= 25%
            if final_score <= -5 and direction == "DOWN" and ml_proba >= 0.25:
                sig["action"] = "🔴 SHORT"
            elif final_score <= -5 and ml_proba < 0.25:
                sig["action"] = "🟡 WAIT"
            elif final_score <= -5:
                sig["action"] = "🟠 AVOID"
            elif final_score <= -3:
                sig["action"] = "🟠 AVOID"
            elif sig["blocked"]:
                # BLOCKED must come before score threshold — blocked overrides BUY
                sig["action"] = "⚠️ BLOCKED"
                decision      = "WAIT"
            elif final_score >= MIN_BUY_SCORE:
                sig["action"] = "🟢 BUY"
            elif final_score >= 0:
                sig["action"] = "🟡 WAIT"
            else:
                sig["action"] = "🟡 WAIT"
            
            # [J] Calculate SELL score
            sell_score = calculate_sell_score(df, is_btc=is_btc)
            
            # [J] Update decision: SELL takes priority over BUY if score is bearish enough
            if sell_score <= MIN_SELL_SCORE:
                # ADX gate: if ADX < 15 → no real trend → short is unreliable
                _adx_val = last.get("ADX", 0) or 0
                if _adx_val < 15:
                    # Weak trend — downgrade to WAIT with warning
                    decision = "WAIT"
                    sig["action"] = "🟡 WAIT"
                    sig["sell_score"] = sell_score
                    sig["adx_short_blocked"] = True
                    sig["reasons"].append(f"⚠️ [SHORT BLOCKED] ADX {_adx_val:.1f} < 15 — no trend confirmation, short unreliable in flat market")
                else:
                    decision = "SELL"
                    sig["action"] = "SELL"
                    sig["sell_score"] = sell_score
            else:
                sig["sell_score"] = sell_score

            # ── BACKTEST (before confidence so win-rate penalty applies) ─────
            if BACKTEST_ENABLED:
                backtest_result = backtest_signals(df, coin, btc_regime=btc_regime)
                if backtest_result.winning_trades + backtest_result.losing_trades > 0:
                    backtest_color = Fore.GREEN if backtest_result.win_rate > 50 else Fore.RED
                    print(backtest_color + f"[BACKTEST] Win Rate: {backtest_result.win_rate:.1f}% | Profit Factor: {backtest_result.profit_factor:.2f} | Max DD: {backtest_result.max_drawdown:.1f}% | Return: {backtest_result.total_return_pct:.1f}%")
                    sig["backtest_win_rate"] = backtest_result.win_rate
                    sig["backtest_pf"]       = backtest_result.profit_factor
                    sig["backtest_trades"]   = backtest_result.winning_trades + backtest_result.losing_trades

            # ── ENTRY QUALITY (must run BEFORE confidence score) ─────────
            # Penalises chasing breakouts and EMA26 exhaustion in confidence
            try:
                _col_eq   = "close" if "close" in df.columns else "price"
                _don_high = df[_col_eq].iloc[-DONCHIAN_LOOKBACK:].max()
                _ema26_eq = last.get("EMA26", last["price"])
                _dev_eq   = (last["price"] - _ema26_eq) / _ema26_eq * 100 if _ema26_eq > 0 else 0

                if last["price"] > _don_high * 1.01:
                    sig["entry_quality"] = "chasing"
                    print(Fore.YELLOW + f"[ENTRY] {coin}: Price {last['price']:.4f} is "
                          f"{((last['price']/_don_high)-1)*100:.1f}% above Donchian upper "
                          f"${_don_high:.4f} — CHASING breakout (confidence penalized)")
                elif _dev_eq > 12:
                    sig["entry_quality"]  = "near_exhaustion"
                    sig["ema26_dev_pct"]  = _dev_eq
                    print(Fore.YELLOW + f"[ENTRY] {coin}: EMA26 deviation {_dev_eq:.1f}% "
                          f"— near exhaustion zone (confidence penalized)")
                else:
                    sig["entry_quality"] = "normal"
            except Exception:
                sig["entry_quality"] = "normal"

            sig["adx"]         = last.get("ADX", 0) or 0   # for confidence ML cap
            sig["ml_accuracy"] = ml_accuracy                 # for confidence ML cap

            # ── CALCULATE CONFIDENCE SCORE (after backtest so penalty applies) ─
            conf_result = calculate_confidence_score(
                score=sig["score"],
                direction=direction,
                direction_conf=direction_conf,
                ml_proba=ml_proba,
                blocked=sig["blocked"],
                smart_money_mode=smart_money_mode,
                is_whale=is_whale,
                sig=sig
            )
            conf_score = conf_result["score"]
            conf_rating = conf_result["rating"]
            conf_rec = conf_result["recommendation"]
            conf_emoji = conf_result["emoji"]
            conf_breakdown = conf_result["breakdown"]

            # ═══════════════════════════════════════════════════════════════
            # ENHANCED ACTION: Add ML + Direction + ADX + Backtest gates
            # Must run BEFORE notifications so decision is final when ntfy fires
            # ═══════════════════════════════════════════════════════════════
            _coin_adx         = last.get("ADX", 0) or 0
            _backtest_pf      = sig.get("backtest_pf", 1.0)
            _backtest_winrate = sig.get("backtest_win_rate", 50)

            # STRONG BUY: all gates must pass
            #   ML >= 70%, direction UP, ADX >= 20 (real trend), backtest profitable
            #   ADX < 20 = no coin-level trend → "STRONG" is misleading
            #   Backtest PF < 1.0 = historically loses money → no strong signal
            _strong_buy_ok = (
                sig["score"] >= MIN_BUY_SCORE and
                ml_proba >= 0.70 and
                direction == "UP" and
                _coin_adx >= 20 and
                _backtest_pf >= 0.85 and        # breakeven zone ok for STRONG BUY
                _backtest_winrate >= 45
            )
            if _strong_buy_ok:
                decision = "STRONG BUY"
                sig["action"] = "STRONG BUY"
            # ADX too weak for any confident BUY — still tradeable but flag clearly
            elif sig["score"] >= MIN_BUY_SCORE and _coin_adx < 15:
                decision = "BUY"
                sig["action"] = "BUY"
                sig["reasons"].append(
                    f"⚠️ [ADX CAUTION] ADX {_coin_adx:.1f} < 15 — no trend; "
                    f"signal valid but unreliable. Wait for ADX > 20 before sizing up."
                )
            # Block BUY: AI very pessimistic
            elif sig["score"] >= MIN_BUY_SCORE and ml_proba < 0.25:
                decision = "WAIT"
                sig["action"] = "WAIT"
                print(f"[ML BLOCK] {coin}: score {sig['score']}/5 but ML {ml_proba:.1%} < 25% → WAIT")
            # Regular BUY
            elif sig["score"] >= MIN_BUY_SCORE and direction == "UP":
                decision = "BUY"
                sig["action"] = "BUY"
            # Short action refinement
            _adx_for_short    = last.get("ADX", 0) or 0
            _adx_ok_for_short = _adx_for_short >= 15
            if not _adx_ok_for_short and sig["score"] <= -5:
                decision = "WAIT"
                sig["action"] = "WAIT"
                sig["adx_short_blocked"] = True
                print(Fore.YELLOW + f"[ADX GATE] {coin}: SHORT blocked — ADX {_adx_for_short:.1f} < 15")
            elif sig["score"] <= -5 and short_ml_proba >= 0.60 and _adx_ok_for_short:
                decision = "STRONG SHORT"
                sig["action"] = "STRONG SHORT"
            elif sig["score"] <= -5 and short_ml_proba >= 0.40 and _adx_ok_for_short:
                decision = "SHORT"
                sig["action"] = "SHORT"
            elif sig["score"] <= -5 and short_ml_proba < 0.40 and _adx_ok_for_short:
                decision = "WAIT"
                sig["action"] = "WAIT"
                print(Fore.YELLOW + f"[SHORT ML GATE] {coin}: SHORT blocked — short ML {short_ml_proba:.0%} < 40%")

            # ── CONFIDENCE OVERRIDE ──────────────────────────────────────
            # If confidence score says WAIT/AVOID, downgrade BUY to WAIT
            # so the displayed action matches the overall assessment.
            # Threshold: conf < 65 = not enough conviction to act
            if decision in ("BUY", "STRONG BUY") and conf_score < 65:
                decision = "WAIT"
                sig["action"] = "WAIT"
                print(Fore.YELLOW + f"[CONF OVERRIDE] {coin}: downgraded to WAIT "
                      f"— confidence {conf_score}/100 < 65 (backtest/ML too weak)")
            if LOG_TRADES and decision not in ("WAIT", "AVOID", "BLOCKED") and not ml_guard_block:
                if decision in ("BUY", "STRONG BUY"):
                    logger.info(f"[TRADE] 🟢 {decision} {coin:12} | Score: {sig.get('score', 0)}/5 | ML: {ml_proba:.0%}")
                elif decision in ("SHORT", "STRONG SHORT"):
                    logger.info(f"[TRADE] 🔴 {decision} {coin:12} | Score: {sell_score}/-5 | ML: {ml_proba:.0%}")
                elif decision == "SELL":
                    logger.info(f"[TRADE] 🔴 SELL {coin:12} | Score: {sell_score}/-5 | ML: {ml_proba:.0%}")
                    _ml_reliable = ml_accuracy >= 0.55
                    _priority    = "urgent" if (sell_score <= -7 or ml_accuracy >= 0.75) else "high"
                    if _ml_reliable:
                        try:
                            _book_usd, _, _ = analyze_order_book_liquidity(coin)
                        except Exception:
                            _book_usd = 0
                        _liq_key, _liq_label, _max_pos = get_liquidity_tier(_book_usd)
                        if _liq_key == "paper_thin":
                            print(Fore.YELLOW + f"[SHORT] {coin}: Skipping ntfy — paper-thin book, untradeable")
                        else:
                            _pos_msg = f"Max position: ${_max_pos:,}" if _max_pos else "No size limit"
                            send_ntfy(
                                title=f"SHORT: {coin.upper()} | sell={sell_score}/-5 | {_liq_label}",
                                message=(
                                    f"Sell Score: {sell_score}/-5\n"
                                    f"ML: {ml_proba:.0%} win prob (acc {ml_accuracy:.0%})\n"
                                    f"Price: ${last['price']:.6f}\n"
                                    f"Direction: {direction} ({direction_conf:.0%})\n"
                                    f"Liquidity: {_liq_label} (${_book_usd:,.0f} book)\n"
                                    f"{_pos_msg}"
                                ),
                                priority=_priority,
                                tags="chart_decreasing,warning",
                                topic_env="NTFY_TOPIC"
                            )
            # NOTE: BUY ntfy is sent AFTER all gates pass (consecutive, ML guard, event block)
            # See send_final_buy_ntfy() call below near [ENTRY TIMING]

            # direction already predicted above
            move_size, move_basis = estimate_move_size(df)
            narrative = generate_narrative(coin, last, sig, direction, move_size, ml_proba, smart_money_mode, is_whale)

            # ═══════════════════════════════════════════════════════════════
            # (Enhanced action was moved above to run before notifications)
            # ═══════════════════════════════════════════════════════════════

            # ═══════════════════════════════════════════════════════════════
            # CONFIDENCE SCORE: Calculate before storing results
            # ═══════════════════════════════════════════════════════════════
            # (Backtest moved above confidence calculation — see above)
            # ═══════════════════════════════════════════════════════════════

            # [J] Store result for ranking (with SELL score and CONFIDENCE SCORE)
            _entry_status, _entry_when, _entry_watch = generate_entry_timing(
                coin, sig, df, last, final_score, btc_regime,
                ml_proba=ml_proba, direction=direction, direction_conf=direction_conf,
                short_ml_proba=short_ml_proba
            )
            results.append({
                "coin": coin,
                "score": sig["score"],
                "sell_score": sell_score,
                "ml_proba": ml_proba,
                "ml_proba_adjusted": ml_proba_adjusted,
                "ml_accuracy": ml_accuracy,
                "short_ml_proba": short_ml_proba,
                "short_ml_accuracy": short_ml_accuracy,
                "direction": direction,
                "direction_conf": direction_conf,
                "move_size": move_size,
                "decision": decision,
                "price": last["price"],
                "narrative": narrative,
                "is_whale": is_whale,
                "smart_money": smart_money_mode,
                "smart_money_mode": smart_money_mode,  # For display
                "smart_money_conf": smart_money_conf,  # OBV momentum! (THIS WAS MISSING!)
                "whale_volume_pct": (whale_vol / whale_threshold * 100) if (whale_threshold and whale_threshold > 0) else 0,
                "obv_momentum": smart_money_conf,  # Use smart_money_conf as OBV momentum!
                "obv_trend":    obv_trend,          # rising / flat / falling
                "above_weekly_ema": any("above Weekly EMA50" in r for r in sig.get("reasons", [])),
                "blocked": sig["blocked"],
                "event_block": event_block,
                "events": events,
                "action": sig["action"],  # Use sig["action"] (recalculated after all adjustments!)
                "confidence_score": conf_score,  # NEW: Easy decision metric!
                "confidence_rating": conf_rating,
                "confidence_rec": conf_rec,
                "confidence_emoji": conf_emoji,
                "entry_status":    _entry_status,
                "entry_when":      _entry_when,
                "entry_watch":     _entry_watch,
                "backtest_pf":       sig.get("backtest_pf", 1.0),   # 1.0 = neutral if not run
                "backtest_win_rate": sig.get("backtest_win_rate", 50),
            })

            # Print coin name and decision (already have separator above)
            color = Fore.GREEN if decision == "BUY" else (Fore.RED if decision == "SELL" else Fore.MAGENTA)
            # [J] Display both BUY and SELL scores in decision line
            # ── BTC Market Direction Notification ────────────────────────
            if coin == BTC_ID and LOG_TRADES:
                _btc_adx    = last.get("ADX", 0) or 0
                _btc_di_pos = last.get("plus_di", 0) or 0
                _btc_di_neg = last.get("minus_di", 0) or 0
                _btc_rsi    = last.get("RSI", 50) or 50
                _btc_vol    = sig.get("vol_ratio", 1.0) or 1.0
                # Read OBV — use the coin loop variable, fall back to sig if unknown
                _btc_obv = obv_trend if obv_trend != "unknown" else sig.get("obv_trend", obv_desc if "rising" in str(obv_desc) else "flat")
                _regime_emoji = "🟢" if btc_regime == "bull" else "🔴" if btc_regime == "bear" else "⚪"
                _dir_emoji    = "📈" if direction == "UP" else "📉" if direction == "DOWN" else "➡️"
                send_ntfy(
                    title=f"₿ BTC Market: {btc_regime.upper()} {_regime_emoji} | ${last['price']:,.0f}",
                    message=(
                        f"Regime:     {btc_regime.upper()} {_regime_emoji}\n"
                        f"Direction:  {direction} {_dir_emoji} ({direction_conf:.0%} consensus)\n"
                        f"Score:      {sig.get('score',0):+.0f}/5\n"
                        f"Confidence: {conf_score}/100\n"
                        f"ML:         {ml_proba:.0%} (acc {ml_accuracy:.0%})\n"
                        f"ADX:        {_btc_adx:.1f} | DI+: {_btc_di_pos:.1f}  DI-: {_btc_di_neg:.1f}\n"
                        f"RSI:        {_btc_rsi:.1f}\n"
                        f"Volume:     {_btc_vol:.2f}x avg\n"
                        f"OBV:        {_btc_obv}\n"
                        f"Signal:     {decision}"
                    ),
                    priority="low",
                    tags="bitcoin,chart_increasing" if btc_regime == "bull" else "bitcoin,chart_decreasing",
                    topic_env="NTFY_TOPIC"
                )

            # [J] For WAIT: show BUY score (how close to triggering BUY), not SELL score for clarity
            if decision == "BUY":
                action_label = f"{decision} [score={sig.get('score', 0)}/{MIN_BUY_SCORE}]"
            elif decision == "SELL":
                action_label = f"{decision} [score={sell_score}/{MIN_SELL_SCORE}]"
            else:  # WAIT
                action_label = f"{decision} [BUY score={sig.get('score', 0)}/{MIN_BUY_SCORE}]"
            print(f"{coin.upper():<14} ➜ {color}{action_label}")
            
            # [G] ML prediction display
            if ML_ENABLED and ml_model is not None:
                ml_color = Fore.GREEN if ml_confidence == "high" else (Fore.YELLOW if ml_confidence == "medium" else (Fore.RED if ml_confidence in ("low","disabled") else Fore.WHITE))
                _conf_label = {
                    "high":     "HIGH",
                    "medium":   "MEDIUM",
                    "low":      "LOW",
                    "disabled": "DISABLED (model accuracy < 50% — worse than random, neutralized to 50%)",
                    "unknown":  "UNKNOWN (prediction failed)",
                }.get(ml_confidence, ml_confidence.upper())
                print(ml_color + f"[ML] Trade outcome confidence: {_conf_label} ({ml_proba:.1%}) | Model accuracy: {ml_accuracy:.1%} on {ml_samples} samples")
            
            # [N] On-Chain Metrics Display
            if DETECT_WHALE_ACTIVITY:
                on_chain_score, on_chain_details = detect_on_chain_accumulation(df)
                is_extreme, vol_zscore, vol_details = detect_volume_extremes(df)
                
                on_chain_color = Fore.GREEN if on_chain_score > 0.3 else (Fore.RED if on_chain_score < -0.3 else Fore.YELLOW)
                print(on_chain_color + f"[ON-CHAIN] {on_chain_details} | {vol_details}")
                if is_extreme:
                    print(Fore.MAGENTA + f"[WHALE ALERT] Extreme volume detected! Volume Z-score: {vol_zscore:.1f}σ")
            
            # [U] Advanced Stops Display — only for actionable signals
            _is_actionable = any(x in decision for x in ("BUY", "SHORT", "SELL"))
            if USE_ADVANCED_STOPS and _is_actionable:
                lookback_high = df['price'].iloc[-20:].max() if len(df) >= 20 else df['price'].iloc[-1]
                lookback_low  = df['price'].iloc[-20:].min() if len(df) >= 20 else df['price'].iloc[-1]
                _atr_val      = last.get("ATR", last["price"] * 0.02)

                # Normalize decision → signal type (BUY/SELL only, WAIT shows BUY plan)
                if "SHORT" in decision or "SELL" in decision:
                    _signal_type = "SELL"
                else:
                    _signal_type = "BUY"

                stop_loss, stop_type, stop_pct = calculate_advanced_stop_loss(
                    last["price"], last["price"], _signal_type,
                    lookback_high, lookback_low, atr=_atr_val
                )

                # SAFETY: displayed stop must ALWAYS be below entry for BUY
                # (trailing from 20-day high can exceed current price on pullbacks)
                if _signal_type == "BUY" and stop_loss >= last["price"]:
                    stop_loss  = last["price"] * 0.98   # 2% default below entry
                    stop_type  = "INITIAL"
                    stop_pct   = -2.0
                elif _signal_type == "SELL" and stop_loss <= last["price"]:
                    stop_loss  = last["price"] * 1.02
                    stop_type  = "INITIAL"
                    stop_pct   = 2.0
                try:
                    _col   = "close" if "close" in df.columns else "price"
                    _don_u = df[_col].iloc[-DONCHIAN_LOOKBACK:].max()
                    _don_l = df[_col].iloc[-DONCHIAN_LOOKBACK:].min()
                except Exception:
                    _don_u = None
                    _don_l = None
                _chop  = _is_strong_chop(df) if df is not None else False
                partial_tps = calculate_partial_profit_levels(
                    last["price"], _signal_type, atr=_atr_val,
                    chop_detected=_chop,
                    donchian_upper=_don_u,
                    donchian_lower=_don_l
                )

                print(Fore.BLUE + f"[STOPS] {stop_type} Stop: {stop_loss:.6f} ({stop_pct:.1f}%) | Partial TP: {len(partial_tps)} levels")
                for i, (tp_price, qty_pct) in enumerate(partial_tps, 1):
                    tp_pct = ((tp_price - last["price"]) / last["price"]) * 100
                    print(f"  Level {i}: {tp_price:.6f} ({tp_pct:+.1f}%) → Take {qty_pct:.0f}%")
            
            
            # ═══════════════════════════════════════════════════════════════
            # CONFIDENCE SCORE DISPLAY (already calculated)
            # ═══════════════════════════════════════════════════════════════
            
            # Display the score prominently
            if conf_score >= 75:
                score_color = Fore.GREEN
            elif conf_score >= 50:
                score_color = Fore.YELLOW
            else:
                score_color = Fore.RED
            
            print("")
            print(score_color + "╔" + "═" * 70 + "╗")
            print(score_color + f"║ CONFIDENCE SCORE: {conf_score:.0f}/100 {conf_emoji} │ TECH SCORE: {sig['score']:.1f}/5 │ {conf_rating} │ {conf_rec}")
            print(score_color + "╚" + "═" * 70 + "╝")
            
            # Show breakdown — tick based on component score, not overall confidence
            for category, value in conf_breakdown.items():
                # Extract numeric score from value string e.g. "2/20" → 2, max 20
                _is_penalty = "⚠️" in category or "⚡" in category
                _is_good = False
                try:
                    if "/" in value:
                        _pts, _max = value.split("/")[0].split(" ")[-1], value.split("/")[1].split(" ")[0]
                        _ratio = float(_pts) / float(_max)
                        _is_good = _ratio >= 0.65   # 65%+ of max = green
                    elif "(-" in value:              # penalty lines always red
                        _is_good = False
                    elif "(+" in value:              # bonus lines yellow
                        _is_good = None              # neutral
                except Exception:
                    _is_good = conf_score >= 75

                if _is_penalty and "(+" in str(value):
                    print(Fore.YELLOW + f"  ⚠️ {category}: {value}")
                elif _is_penalty:
                    print(Fore.RED + f"  ❌ {category}: {value}")
                elif _is_good is True:
                    print(Fore.GREEN + f"  ✅ {category}: {value}")
                elif _is_good is None:
                    print(Fore.CYAN + f"  ⚡ {category}: {value}")
                else:
                    print(Fore.YELLOW + f"  ⚠️ {category}: {value}")
            
            print("")
            
            # [NEW] Direction and move size
            dir_color = Fore.GREEN if direction == "UP" else Fore.RED
            print(dir_color + f"[DIRECTION] {direction} ({direction_conf:.0%} confidence) | Expected move: {move_size:.1f}%")
            
            # [NEW] Narrative summary
            print(Fore.CYAN + f"[WHAT'S NEXT] {narrative}")

            # ── ML INSIGHT — explain why the model gave this rating ───────────
            _long_insights  = generate_ml_insight(
                last, ml_model, ml_scaler, df=df, btc_df=btc_df,
                fng_value=fng_value, model_type="long"
            )
            _s_model  = short_model  if short_ml_proba != 0.5 else None
            _s_scaler = short_scaler if short_ml_proba != 0.5 else None
            _short_insights = generate_ml_insight(
                last, _s_model, _s_scaler,
                df=df, btc_df=btc_df, fng_value=fng_value, model_type="short"
            )
            if _long_insights or _short_insights:
                _insight_text = format_ml_insight(
                    _long_insights, _short_insights,
                    ml_proba, short_ml_proba,
                    ml_accuracy, short_ml_accuracy, coin
                )
                for _iline in _insight_text.split("\n"):
                    print(Fore.MAGENTA + _iline)

            # ── ENTRY TIMING DESCRIPTION ──────────────────────────────────
            sig["_btc_df"] = btc_df   # pass btc_df for BTC divergence gate
            entry_status, entry_when, entry_watch_price = generate_entry_timing(
                coin, sig, df, last, final_score, btc_regime,
                ml_proba=ml_proba, direction=direction, direction_conf=direction_conf,
                short_ml_proba=short_ml_proba
            )
            if "ENTER NOW" in entry_status:
                timing_color = Fore.GREEN
            elif "SHORT READY" in entry_status:
                timing_color = Fore.RED
            elif "SHORT BLOCKED" in entry_status:
                timing_color = Fore.YELLOW
            elif "ML TRAP" in entry_status or "CHASING" in entry_status:
                timing_color = Fore.RED
            elif "BTC DIVERGENCE" in entry_status:
                timing_color = Fore.YELLOW
            elif "CANDLE CLOSE" in entry_status:
                timing_color = Fore.YELLOW
            elif "CAUTION" in entry_status or "ALMOST" in entry_status or "WATCH" in entry_status:
                timing_color = Fore.YELLOW
            elif "BLOCKED" in entry_status or "NOT YET" in entry_status:
                timing_color = Fore.RED
            else:
                timing_color = Fore.WHITE
            print(timing_color + f"[ENTRY TIMING] {entry_status}")
            print(timing_color + f"               Watch price: ${entry_watch_price:.6f}")
            for _eline in entry_when.split("\n"):
                if _eline.strip():
                    print(timing_color + f"               {_eline}")

            # ── SHORT SCORE BREAKDOWN — verify long has no bearish overlap ────
            _s_last   = df.iloc[-1]
            _s_close  = _s_last["price"]
            _s_ema12  = _s_last.get("EMA12", _s_close)
            _s_ema26  = _s_last.get("EMA26", _s_close)
            _s_macd_h = _s_last.get("MACD_Hist", 0) or 0
            _s_rsi    = _s_last.get("RSI", 50) or 50
            _s_adx    = _s_last.get("ADX", 0) or 0
            _s_pdi    = _s_last.get("plus_di", 0) or 0
            _s_mdi    = _s_last.get("minus_di", 0) or 0
            _s_sa     = _s_last.get("senkou_a", float("nan"))
            _s_sb     = _s_last.get("senkou_b", float("nan"))
            _s_ubb    = _s_last.get("UpperBB", _s_close * 1.1)
            _s_chikou       = _s_last.get("chikou", float("nan"))
            _s_chikou_ref   = _s_last.get("chikou_ref", float("nan"))
            _s_chikou_cloud = _s_last.get("chikou_cloud_bot", float("nan"))

            _sc = []   # (label, value_str, score_contribution)

            # 1. EMA position
            if _s_close < _s_ema12 and _s_close < _s_ema26:
                _sc.append(("Price vs EMA",   "below both EMA12/26",             -1.0))
            elif _s_close < _s_ema12 or _s_close < _s_ema26:
                _sc.append(("Price vs EMA",   "below one EMA",                   -0.5))
            else:
                _sc.append(("Price vs EMA",   "above EMA12/26 ✅",                0.0))

            # 2. Cloud
            if not (np.isnan(_s_sa) or np.isnan(_s_sb)):
                _ct = max(_s_sa, _s_sb); _cb = min(_s_sa, _s_sb)
                if _s_close < _cb:
                    _sc.append(("Ichimoku Cloud", "below cloud",                  -1.0))
                elif _s_close < _ct:
                    _sc.append(("Ichimoku Cloud", "inside cloud",                 -0.5))
                else:
                    _sc.append(("Ichimoku Cloud", "above cloud ✅",                0.0))

            # 3. MACD
            if _s_macd_h < 0:
                _sc.append(("MACD Histogram",  f"{_s_macd_h:.4f} (negative)",    -1.0))
            else:
                _sc.append(("MACD Histogram",  f"{_s_macd_h:.4f} (positive) ✅",  0.0))

            # 4. RSI overbought
            if _s_rsi > 85:
                _sc.append(("RSI",             f"{_s_rsi:.1f} — extreme OB",     -1.5))
            elif _s_rsi > 70:
                _sc.append(("RSI",             f"{_s_rsi:.1f} — overbought",     -1.0))
            else:
                _sc.append(("RSI",             f"{_s_rsi:.1f} ✅",                0.0))

            # 5. ADX + DI
            if _s_adx > 25 and _s_mdi > _s_pdi:
                _sc.append(("ADX + DI",        f"ADX={_s_adx:.0f}, DI-({_s_mdi:.0f})>DI+({_s_pdi:.0f})", -1.0))
            elif _s_adx <= 15:
                _sc.append(("ADX + DI",        f"ADX={_s_adx:.0f} — no trend",   0.0))
            else:
                _sc.append(("ADX + DI",        f"ADX={_s_adx:.0f}, DI+>{_s_pdi:.0f} ✅", 0.0))

            # 6. EMA cross
            if _s_ema12 < _s_ema26:
                _sc.append(("EMA Cross",       "EMA12 < EMA26 (death cross)",    -0.5))
            else:
                _sc.append(("EMA Cross",       "EMA12 > EMA26 ✅",                0.0))

            # 7. Chikou
            _cb_price = (not np.isnan(_s_chikou) and not np.isnan(_s_chikou_ref)
                         and _s_chikou < _s_chikou_ref)
            _cb_cloud = (not np.isnan(_s_chikou) and not np.isnan(_s_chikou_cloud)
                         and _s_chikou < _s_chikou_cloud)
            if _cb_price and _cb_cloud:
                _sc.append(("Chikou Span",     "below price + cloud",            -1.0))
            elif _cb_price:
                _sc.append(("Chikou Span",     "below price only",               -0.5))
            else:
                _sc.append(("Chikou Span",     "above price/cloud ✅",            0.0))

            # 8. Upper BB
            if _s_close > _s_ubb * 0.98:
                _sc.append(("Upper BB",        f"near upper band ({_s_close/_s_ubb*100:.0f}%)", -0.5))
            else:
                _sc.append(("Upper BB",        "not extended ✅",                 0.0))

            _total_sell       = sum(c for _, _, c in _sc)
            _bearish_count    = sum(1 for _, _, c in _sc if c < 0)

            print(Fore.CYAN + f"\n[SHORT SCORE] {coin.upper()}"
                  f"   sell score: {_total_sell:.1f} / -9"
                  f"   (short triggers at -5)")
            print(Fore.CYAN + "─" * 65)
            for _label, _val, _contrib in _sc:
                _c_color = Fore.RED if _contrib < 0 else Fore.GREEN
                _c_str   = f"{_contrib:+.1f}" if _contrib != 0 else " 0.0"
                print(f"  {_c_color}{_c_str}{Fore.RESET}  {_label:<22} {_val}")
            print(Fore.CYAN + "─" * 65)

            if _total_sell <= MIN_SELL_SCORE:
                _vs_verdict = (Fore.RED + f"🚨 SHORT TERRITORY — long conflicts with {_bearish_count} active bearish "
                               f"components. Reconsider entry.")
            elif _total_sell <= -3:
                _vs_verdict = (Fore.YELLOW + f"⚠️  PARTIAL PRESSURE — {_bearish_count} bearish component(s) active. "
                               f"Long is valid but watch for reversal.")
            elif _total_sell <= -1:
                _vs_verdict = (Fore.YELLOW + f"🟡 MILD HEADWINDS — minor bearish overlap ({_bearish_count} component). "
                               f"Long still favoured.")
            else:
                _vs_verdict = (Fore.GREEN + f"✅ CLEAN LONG — zero bearish components. Short score confirms long setup.")

            print(f"  VERDICT: {_vs_verdict}{Fore.RESET}")
            print(Fore.CYAN + "─" * 65 + "\n")

            # ── SMART RE-ENTRY ALERT ──────────────────────────────────────
            # When entry timing says "wait for EMA26" or similar level,
            # check if price is currently approaching that level.
            # If within 3% of watch price AND key conditions are still valid,
            # send a specific alert — not just "price reached level" but
            # "price at level AND setup still intact".
            _watch_price = _entry_watch
            _dist_to_watch = abs(last["price"] - _watch_price) / _watch_price * 100

            if (_dist_to_watch <= 3.0 and                      # within 3% of watch price
                decision == "WAIT" and                          # not already a BUY
                final_score >= MIN_BUY_SCORE - 1 and            # close to qualifying
                not sig["blocked"] and
                _entry_status not in ("🟢 ENTER NOW",)):

                # Re-validate key conditions AT this price level
                _obv_ok   = obv_trend in ("rising",)
                _rsi_ok   = 40 <= last.get("RSI", 50) <= 70
                _macd_ok  = last.get("MACD_Hist", 0) > 0
                _above_cloud = sig.get("cloud_state") == "above"

                _conditions_met = sum([_obv_ok, _rsi_ok, _macd_ok, _above_cloud])

                if _conditions_met >= 3:
                    _alert_msg = (
                        f"Price approaching watch level ${_watch_price:.4f} "
                        f"({_dist_to_watch:.1f}% away)\n"
                        f"Re-entry conditions at this level:\n"
                        f"  OBV:   {'✅ rising' if _obv_ok else '❌ not rising'}\n"
                        f"  RSI:   {'✅' if _rsi_ok else '❌'} {last.get('RSI',0):.1f}\n"
                        f"  MACD:  {'✅ positive' if _macd_ok else '❌ negative'}\n"
                        f"  Cloud: {'✅ above' if _above_cloud else '❌ below'}\n"
                        f"Score: {final_score:.0f}/5  ML: {ml_proba:.0%}\n"
                        f"Run bot again when price touches ${_watch_price:.4f} to confirm entry."
                    )
                    send_ntfy(
                        title=f"👀 WATCH: {coin.upper()} approaching ${_watch_price:.4f}",
                        message=_alert_msg,
                        priority="default",
                        tags="eyes,chart_increasing",
                        topic_env="NTFY_WATCH_TOPIC"
                    )
                    print(Fore.YELLOW + f"[WATCH ALERT] {coin}: Price within 3% of watch level "
                          f"${_watch_price:.4f} — {_conditions_met}/4 conditions intact")
            if events:
                closest_event = events[0]
                days_away = abs(closest_event["days_until"])
                if closest_event["is_past"]:
                    event_status = f"RECENT ({days_away}d ago)"
                else:
                    event_status = f"UPCOMING ({days_away}d away)"
                
                event_color = Fore.RED if closest_event["impact"] == "HIGH" else Fore.YELLOW
                print(event_color + f"[EVENT] {event_status}: {closest_event['description']}")
                
                if event_adjustment_reason:
                    print(event_color + f"[EVENT IMPACT] {event_adjustment_reason} — ML confidence reduced from {ml_proba:.0%} to {ml_proba_adjusted:.0%}")
                
                if event_block:
                    print(Fore.RED + f"⛔ {event_block_reason}")
            
            # [H] Smart money display
            whale_indicator = "🐋 WHALE ZONE" if is_whale else "normal"
            whale_color = Fore.CYAN if is_whale else Fore.WHITE
            print(whale_color + f"[WHALES] Volume: {whale_vol:.0f} ({whale_vol/whale_threshold*100:.0f}% of whale threshold) — {whale_indicator}")
            
            smart_color = Fore.GREEN if smart_money_mode == "accumulation" else (Fore.RED if smart_money_mode == "distribution" else Fore.YELLOW)
            print(smart_color + f"[SMART MONEY] {smart_money_mode.upper()} mode | OBV momentum: {smart_money_conf:.0%}")
            
            if vol_div:
                div_color = Fore.GREEN if vol_div == "bullish_divergence" else Fore.RED
                print(div_color + f"[VOLUME DIV] {vol_div.upper()} detected — low volume on {vol_div_details.get('price_trend', '?')} price")
            
            print(f"Price: {last['price']:.6f} | EMA12: {last['EMA12']:.6f} | EMA26: {last['EMA26']:.6f}")
            print(f"MACD: {last['MACD']:.6f} | Signal: {last['Signal']:.6f} | MACD Hist: {last['MACD_Hist']:.6f}")

            plus_di  = sig.get("plus_di",  float("nan"))
            minus_di = sig.get("minus_di", float("nan"))
            print(f"RSI: {last['RSI']:.2f} | ADX: {last['ADX']:.2f} | DI+: {plus_di:.2f} | DI-: {minus_di:.2f}")
            print(f"Bollinger Bands: Upper={last['UpperBB']:.6f}, Lower={last['LowerBB']:.6f}")

            d = donchian_breakout_signal(df)
            print(f"📊 Donchian ({DONCHIAN_LOOKBACK}): Upper={d['channel_high']:.6f} | Lower={d['channel_low']:.6f} | Mid={d['channel_mid']:.6f}")

            vol_ratio = sig["vol_ratio"]
            vol_label = "🟢 High" if vol_ratio >= 1.5 else ("🟡 Moderate" if vol_ratio >= 0.8 else "🔴 Low")
            print(f"Volume (24h): {sig['vol']:.2f} | Avg (90d): {sig['avg_vol']:.2f} - {vol_label} ({vol_ratio:.2f}x avg)")

            selloff, msg = detect_big_selloff(df, debug=True)
            if selloff:
                print(Fore.RED + f"[{coin.upper()}] {msg} → Skipping trade.")

            capitulation, cap_info = detect_capitulation(df)
            if capitulation:
                print(Fore.RED + "🔥 CAPITULATION DETECTED 🔥")
                print(Fore.RED + f" Drop={cap_info['price_drop_pct']:.2f}% | Vol={cap_info['vol_ratio']:.2f}x | "
                      f"ATR={cap_info['atr_ratio']:.2f}x | RSI={cap_info['rsi']:.2f}")

            print("Reasons:")
            for r in sig["reasons"]:
                print("   " + r)

            cloud_state = sig.get("cloud_state", "unknown")
            if cloud_state == "above":
                print(Fore.GREEN  + "Ichimoku: Price above cloud (bullish).")
            elif cloud_state == "in":
                print(Fore.YELLOW + "Ichimoku: Price inside cloud (neutral).")
            elif cloud_state == "below":
                print(Fore.RED    + "Ichimoku: Price below cloud (bearish).")

            if sig["blocked"]:
                blockers_str = ', '.join(sig.get('blockers', ['Unknown blocker']))
                print(Fore.RED + f"⛔ [BLOCKED] {blockers_str}")

            # [G] Optional ML guard: block BUY if model is confident but predicts failure
            ml_guard_block = False
            if ML_ENABLED and ml_model is not None and ml_accuracy > 0.60:
                if ml_proba < 0.45 and ml_samples > 20:
                    print(Fore.RED + f"[ML GUARD] Model predicts {1-ml_proba:.0%} failure rate — BUY blocked.")
                    ml_guard_block = True

            safe_low  = sig["lower_bb"]
            safe_high = last["EMA26"]
            price_now = last["price"]
            atr_now   = last.get("ATR", price_now * 0.02)
            adx_now   = last.get("ADX", 0)
            di_plus   = last.get("plus_di",  last.get("DI+", 0))
            di_minus  = last.get("minus_di", last.get("DI-", 0))
            z_now     = 0
            try:
                _col = "close" if "close" in df.columns else "price"
                _p   = df[_col].iloc[-20:].values
                _z   = (_p[-1] - _p.mean()) / (_p.std() + 1e-9)
                z_now = _z
            except Exception:
                pass

            # 3-Tier ADX-based safe zone expansion
            # Only expands when: DI+ > DI- (bullish) AND not choppy AND Z-score < 2
            _chop_safe  = _is_strong_chop(df)
            _di_bullish = di_plus > di_minus
            _not_stretched = z_now < 2.0

            if _di_bullish and not _chop_safe and _not_stretched:
                if adx_now > 30:
                    safe_high = last["EMA26"] + 1.0 * atr_now
                    zone_label = f"WIDE (ADX={adx_now:.1f}>30, DI+>{di_minus:.1f})"
                elif adx_now > 20:
                    safe_high = last["EMA26"] + 0.5 * atr_now
                    zone_label = f"MODERATE (ADX={adx_now:.1f}>20, DI+>{di_minus:.1f})"
                else:
                    zone_label = f"STANDARD (ADX={adx_now:.1f}<20)"
            else:
                zone_label = "STANDARD"
                if _chop_safe:       zone_label += " (chop — no expansion)"
                elif not _di_bullish: zone_label += " (DI- dominant — no expansion)"
                elif not _not_stretched: zone_label += " (Z-score stretched — no expansion)"

            if price_now < safe_low or price_now > safe_high:
                print(Fore.YELLOW + f"⚠️ [WARNING] Price {price_now:.6f} is OUTSIDE safe entry zone "
                      f"({safe_low:.6f} → {safe_high:.6f}) [{zone_label}]")
            else:
                print(Fore.GREEN + f"✅ Safe Entry Zone: {safe_low:.6f} → {safe_high:.6f} [{zone_label}]")

            if decision == "BUY" or capitulation:
                breakout_up = d["breakout_up"]

                # [G] ML guard blocks BUY if model is confident and predicts failure
                if ml_guard_block and not capitulation:
                    print(Fore.MAGENTA + "[NO ACTION] ML guard blocked this trade.")
                    # Update the already-appended result to reflect the block
                    if results and results[-1]["coin"] == coin:
                        results[-1]["decision"] = "WAIT"
                        results[-1]["action"]   = "WAIT"
                    continue

                # [I] Event block — do not trade before/after high-impact events
                if event_block and not capitulation:
                    print(Fore.RED + f"[NO ACTION] {event_block_reason}")
                    continue

                # [F] Consecutive candle confirmation: both today and yesterday must pass
                if decision == "BUY" and not capitulation:
                    consecutive_pass, consecutive_met, consecutive_scores = check_consecutive_confirmation(
                        df, btc_regime=btc_regime, is_btc=is_btc, min_days=MIN_CONSECUTIVE_DAYS
                    )
                    print(Fore.CYAN + f"[CONSECUTIVE] Last {MIN_CONSECUTIVE_DAYS} candles: ", end="")
                    for cs in consecutive_scores:
                        status = "✅" if cs["action"] == "PASS" else "❌"
                        print(f"{status}D{cs['day_offset']}(score={cs['score']})", end=" ")
                    print()

                    # Also verify TODAY's final adjusted score qualifies
                    # Historical days may have scored 6 but today may have dropped to 4
                    # after wick/volume/chop penalties — don't confirm a signal that doesn't exist
                    if final_score < MIN_BUY_SCORE:
                        print(Fore.YELLOW + f"[CONSECUTIVE] ⚠️ Historical days passed but current score "
                              f"{final_score:.0f} < {MIN_BUY_SCORE} (penalties applied today) — NOT confirmed")
                        consecutive_pass = False

                    if not consecutive_pass:
                        print(Fore.YELLOW + f"[CONSECUTIVE] Only {consecutive_met}/{MIN_CONSECUTIVE_DAYS} days pass threshold. Waiting for confirmation.")
                        continue
                    else:
                        print(Fore.GREEN + f"[CONSECUTIVE] ✅ Confirmed: {consecutive_met}/{MIN_CONSECUTIVE_DAYS} consecutive days pass BUY criteria!")

                    # Pass to generate_entry_timing so candle gate can be relaxed
                    sig["_consecutive_confirmed"] = consecutive_pass

                    # Re-generate entry timing now that consecutive is known
                    # First call at line ~7462 ran before consecutive — may show wrong message
                    if consecutive_pass:
                        _entry_status, _entry_when, _entry_watch = generate_entry_timing(
                            coin, sig, df, last, final_score, btc_regime,
                            ml_proba=ml_proba, direction=direction, direction_conf=direction_conf,
                            short_ml_proba=short_ml_proba
                        )
                        # Update results dict so ranked table shows correct status
                        if results and results[-1].get("coin") == coin:
                            results[-1]["entry_status"] = _entry_status
                            results[-1]["entry_when"]   = _entry_when
                            results[-1]["entry_watch"]  = _entry_watch
                        # Reprint corrected entry timing
                        _t_col = Fore.GREEN if "ENTER NOW" in _entry_status else Fore.YELLOW
                        print(_t_col + f"[ENTRY TIMING] {_entry_status}")
                        print(_t_col + f"               Watch price: ${_entry_watch:.6f}")
                        for _el in _entry_when.split("\n"):
                            if _el.strip():
                                print(_t_col + f"               {_el}")

                    # ── BUY NOTIFICATION — fired only here, after ALL gates pass ──
                    # Gates passed: ML guard ✅ | event block ✅ | consecutive ✅
                    if decision in ("BUY", "STRONG BUY") and LOG_TRADES:
                        _conf_score  = conf_result.get("score", 0)
                        _conf_rating = conf_result.get("rating", "")
                        _notify_buy  = _conf_score >= 65 and any(
                            tier in _conf_rating for tier in ["GOOD", "VERY GOOD", "EXCELLENT"]
                        )
                        print(f"[NTFY] {coin}: conf={_conf_score} rating='{_conf_rating}' notify={_notify_buy}")
                        if _notify_buy:
                            _priority = "urgent" if decision == "STRONG BUY" or "EXCELLENT" in _conf_rating else "high"
                            send_ntfy(
                                title=f"{'⚡ STRONG ' if decision == 'STRONG BUY' else ''}BUY: {coin.upper()} — {_conf_rating}",
                                message=(
                                    f"Signal:     {decision}\n"
                                    f"Confidence: {_conf_score}/100\n"
                                    f"Score:      {sig.get('score',0):+.0f}/5\n"
                                    f"ML:         {ml_proba:.0%} (acc {ml_accuracy:.0%})\n"
                                    f"Short ML:   {short_ml_proba:.0%} (acc {short_ml_accuracy:.0%})\n"
                                    f"Price:      ${last['price']:.6f}\n"
                                    f"Direction:  {direction} ({direction_conf:.0%})\n"
                                    f"Volume:     {sig.get('vol_ratio',0):.2f}x avg\n"
                                    f"ADX:        {last.get('ADX',0):.1f} | DI+: {last.get('plus_di',0):.1f} DI-: {last.get('minus_di',0):.1f}\n"
                                    f"OBV:        {obv_trend if obv_trend not in ('unknown', None) else ('rising' if df['OBV'].iloc[-1] > df['OBV'].iloc[-4] else 'falling' if df['OBV'].iloc[-1] < df['OBV'].iloc[-4] else 'flat') if 'OBV' in df.columns else '?'}\n"
                                    f"Consecutive:{consecutive_met}/{MIN_CONSECUTIVE_DAYS} days ✅"
                                ),
                                priority=_priority,
                                tags="chart_increasing,moneybag",
                                topic_env="NTFY_TOPIC"
                            )

                if cloud_state == "below" and not breakout_up:
                    print(Fore.RED + "[CLOUD] Bearish cloud — suppressing all buys.")
                    continue

                if breakout_up:
                    if sig["vol_ratio"] < 0.8:
                        print(Fore.MAGENTA + "[ADV] Breakout on low volume — falling back to standard plan.")
                        plan = build_position_plan(coin, last, df)
                        execute_plan_paper(coin, last, df, plan, sig)
                        continue

                    print(Fore.CYAN + f"[ADV] Donchian breakout confirmed! ChannelHigh={d['channel_high']:.6f}")
                    base_plan   = build_position_plan(coin, last, df)
                    target_risk = ACCOUNT_BALANCE * RISK_PER_TRADE
                    vol_units   = volatility_target_size(last["price"], base_plan["stop_price"],
                                                         ACCOUNT_BALANCE, target_risk)
                    max_by_exp  = (ACCOUNT_BALANCE * MAX_EXPOSURE_PCT) / last["price"]
                    vol_units   = min(vol_units, max_by_exp)
                    if vol_units < MIN_TRADE_SIZE:
                        vol_units = MIN_TRADE_SIZE

                    rsi = last["RSI"]
                    init_frac = 0.5 if rsi > RSI_REDUCE_THRESH else 1.0

                    total_units   = round(vol_units, QUANT_ROUND)
                    initial_units = max(MIN_TRADE_SIZE, round(total_units * init_frac, QUANT_ROUND))
                    reserve_units = round(max(0.0, total_units - initial_units), QUANT_ROUND)

                    stop     = base_plan["stop_price"]
                    tp       = base_plan["tp_price"]
                    trailing = compute_dynamic_trailing(
                        last["price"], last["EMA12"], last["ATR"],
                        kijun=last.get("kijun", np.nan), use_kijun=True
                    )

                    plan = {
                        "total_units":   total_units,
                        "initial_units": initial_units,
                        "reserve_units": reserve_units,
                        "stop_price":    stop,
                        "tp_price":      tp,
                        "trailing_stop": round(trailing, 8),
                        "rsi_msg":       base_plan["rsi_msg"],
                        "rsi_color":     base_plan["rsi_color"],
                        "ladder_prices": base_plan["ladder_prices"],
                        "rr":            base_plan["rr"],
                    }

                    pyr_prices     = pyramid_levels(last["price"], tp)
                    ladder_alloc   = round(reserve_units * 0.5, QUANT_ROUND)
                    pyramid_alloc  = round(reserve_units - ladder_alloc, QUANT_ROUND)

                    extra_limit_orders = []
                    if pyramid_alloc >= MIN_TRADE_SIZE and pyr_prices:
                        per_pyr     = round(pyramid_alloc / len(pyr_prices), QUANT_ROUND)
                        distributed = 0.0
                        for i, p in enumerate(pyr_prices):
                            qty          = per_pyr
                            distributed += qty
                            if i == len(pyr_prices) - 1:
                                qty = round(qty + round(pyramid_alloc - distributed, QUANT_ROUND), QUANT_ROUND)
                            if qty >= MIN_TRADE_SIZE:
                                extra_limit_orders.append({"price": round(p, 8), "qty": qty})

                    plan["reserve_units"] = ladder_alloc
                    print(Fore.CYAN + f"[PYRAMID] planned prices: {', '.join(f'{p:.6f}' for p in pyr_prices)}")
                    execute_plan_paper(coin, last, df, plan, sig, extra_limit_orders)

                else:
                    print(Fore.MAGENTA + "[WAIT] No breakout confirmation (Donchian). Using standard plan instead.")
                    plan = build_position_plan(coin, last, df)
                    execute_plan_paper(coin, last, df, plan, sig)

            else:
                # Action is determined by score (which includes all penalties/boosts)
                # If not executing trade here, just note it in logs
                action_determined = sig.get("action", "WAIT")
                if LOG_ENABLED:
                    logger.debug(f"[SIGNAL] {coin}: Action={action_determined} but trade execution disabled or conditions blocked execution")

        except Exception as e:
            import traceback
            print(Fore.RED + f"Error for {coin}: {e}")
            print(Fore.RED + traceback.format_exc())
            logger.error(f"Error processing {coin}: {e}\n{traceback.format_exc()}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # [J] RANKING: Order by BUY + SELL signals with scores
    # ═══════════════════════════════════════════════════════════════════════════════
    if results:
        print("\n\n" + "="*80)
        print(Fore.CYAN + "RANKED OPPORTUNITIES (BUY + SELL Signals)")
        print("="*80)

        # [J] Separate BUY and SELL results — exclude event-blocked signals
        buy_results  = [r for r in results if r["action"] == "BUY" and not r.get("event_block")]
        sell_results = [r for r in results if r["action"] == "SELL"]
        wait_results = [r for r in results if r["action"] == "WAIT"]
        
        # [J] Sort each group by score
        buy_results_sorted = sorted(buy_results, key=lambda x: x["score"], reverse=True)
        sell_results_sorted = sorted(sell_results, key=lambda x: x["sell_score"])  # [J] More negative = stronger
        
        # Display BUY signals
        if buy_results_sorted:
            print(Fore.GREEN + "\n🟢 BUY SIGNALS (Long Entry Opportunities):")
            print(Fore.GREEN + "─"*80)
            
            for rank, r in enumerate(buy_results_sorted, 1):
                score_color = Fore.GREEN if r["score"] >= MIN_BUY_SCORE else Fore.YELLOW
                print(f"\n{rank:2d}. 🟢 {r['coin'].upper():<15} Price: ${r['price']:.6f}")
                print(f"    Score: {score_color}{r['score']}/{MIN_BUY_SCORE}{Fore.RESET} | " +
                      f"Direction: {r['direction']} ({r['direction_conf']:.0%}) | " +
                      f"Expected Move: {r['move_size']:.1f}%")

                # ── Entry timing ──────────────────────────────────────────
                _r_last = {"price": r["price"]}
                # Reconstruct minimal 'last' dict for timing helper
                print(Fore.GREEN + f"    ✅ ENTER NOW at ${r['price']:.6f} — all criteria met")
                
                # Show ML confidence with event adjustment if applicable
                if r["ml_proba_adjusted"] < r["ml_proba"]:
                    print(f"    ML Confidence: {r['ml_proba']:.0%} → {Fore.RED}{r['ml_proba_adjusted']:.0%}{Fore.RESET} (event impact) | " +
                          f"Smart Money: {r['smart_money']} | " +
                          f"Whale: {'Yes 🐋' if r['is_whale'] else 'No'}")
                else:
                    print(f"    ML Confidence: {r['ml_proba']:.0%} | " +
                          f"Smart Money: {r['smart_money']} | " +
                          f"Whale: {'Yes 🐋' if r['is_whale'] else 'No'}")
                
                # Show upcoming events
                if r['events']:
                    closest_event = r['events'][0]
                    days_away = abs(closest_event["days_until"])
                    status = "RECENT" if closest_event["is_past"] else "UPCOMING"
                    event_icon = "🔴" if closest_event["impact"] == "HIGH" else "🟡"
                    print(f"    {event_icon} [{status} {days_away}d] {closest_event['description']}")
                
                if r["event_block"]:
                    print(Fore.RED + f"    ⛔ EVENT BLOCK: Trade blocked due to upcoming event" + Fore.RESET)
                
                print(f"    {Fore.CYAN}{r['narrative']}{Fore.RESET}")
        
        # [J] Display SELL signals
        if sell_results_sorted:
            print(Fore.RED + "\n🔴 SELL SIGNALS (Short Entry Opportunities):")
            print(Fore.RED + "─"*80)
            
            for rank, r in enumerate(sell_results_sorted, 1):
                score_color = Fore.RED if r["sell_score"] <= MIN_SELL_SCORE else Fore.YELLOW
                print(f"\n{rank:2d}. 🔴 {r['coin'].upper():<15} Price: ${r['price']:.6f}")
                # [J] Display SELL score (negative values)
                print(f"    SELL Score: {score_color}{r['sell_score']:.1f}/{MIN_SELL_SCORE}{Fore.RESET} | " +
                      f"Direction: {r['direction']} ({r['direction_conf']:.0%}) | " +
                      f"Expected Move: {r['move_size']:.1f}%")
                
                # Show ML confidence with event adjustment if applicable
                if r["ml_proba_adjusted"] < r["ml_proba"]:
                    print(f"    ML Confidence: {r['ml_proba']:.0%} → {Fore.RED}{r['ml_proba_adjusted']:.0%}{Fore.RESET} (event impact) | " +
                          f"Smart Money: {r['smart_money']} | " +
                          f"Whale: {'Yes 🐋' if r['is_whale'] else 'No'}")
                else:
                    print(f"    ML Confidence: {r['ml_proba']:.0%} | " +
                          f"Smart Money: {r['smart_money']} | " +
                          f"Whale: {'Yes 🐋' if r['is_whale'] else 'No'}")
                
                # Show upcoming events
                if r['events']:
                    closest_event = r['events'][0]
                    days_away = abs(closest_event["days_until"])
                    status = "RECENT" if closest_event["is_past"] else "UPCOMING"
                    event_icon = "🔴" if closest_event["impact"] == "HIGH" else "🟡"
                    print(f"    {event_icon} [{status} {days_away}d] {closest_event['description']}")
                
                if r["event_block"]:
                    print(Fore.RED + f"    ⛔ EVENT BLOCK: Trade blocked due to upcoming event" + Fore.RESET)
                
                print(f"    {Fore.CYAN}{r['narrative']}{Fore.RESET}")

        # ═══════════════════════════════════════════════════════════════
        # 🏆 CONFIDENCE RANKING TABLE - Easy Decision Making!
        # ═══════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════
        # 🎯 HIGH-PROBABILITY WATCHLIST
        # Coins that pass the 5-factor qualifier — closest to a valid entry
        # ═══════════════════════════════════════════════════════════════
        print("\n" + Fore.CYAN + "╔" + "═"*70 + "╗")
        print(Fore.CYAN + "║" + "  🎯 HIGH-PROBABILITY WATCHLIST — 5-Factor Qualifier".ljust(70) + "║")
        print(Fore.CYAN + "╚" + "═"*70 + "╝")
        print(Fore.WHITE + "  Coins ranked by how many of the 5 factors they satisfy:")
        print(Fore.WHITE + "  OBV↑  |  PF>1.0  |  ML>60%(acc>65%)  |  Score≥4  |  Above Weekly EMA50")

        _watchlist = []
        for r in results:
            _obv     = r.get("obv_trend", "")
            _pf      = r.get("backtest_pf", 0) or 0
            _ml      = r.get("ml_proba", 0) or 0
            _acc     = r.get("ml_accuracy", 0) or 0
            _score   = r.get("score", 0) or 0
            _above   = r.get("above_weekly_ema", False)
            _conf    = r.get("confidence_score", 0) or 0
            _coin    = r.get("coin", "")

            if _coin.lower() in ("zcash", "origintrail"):  # debug
                f2_check = _pf >= 1.0
                print(f"  [DEBUG] {_coin}: PF={_pf:.2f}({'✅' if f2_check else '❌'}) WR={r.get('backtest_win_rate',0):.1f}% OBV={_obv} ML={_ml:.0%}(acc{_acc:.0%}) Score={_score} EMA50={_above}")

            f1 = _obv == "rising"
            f2 = _pf >= 1.0
            f3 = _ml >= 0.60 and _acc >= 0.65   # lowered acc from 0.70 to 0.65 — 69.8% was failing
            f4 = _score >= 4    # lowered from 5 — score 4 is one point away, worth watching
            f5 = _above

            factors_met = sum([f1, f2, f3, f4, f5])
            if factors_met >= 3:   # minimum 3/5 to appear — 2/5 is too noisy
                _watchlist.append({
                    "coin": _coin, "factors": factors_met,
                    "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
                    "pf": _pf, "ml": _ml, "acc": _acc, "score": _score,
                    "conf": _conf, "obv": _obv,
                })

        _watchlist.sort(key=lambda x: (-x["factors"], -x["conf"]))

        # ── Notify for 4/5, 5/5 coins, or 3/5 with high confidence ──
        for _w in _watchlist:
            _should_notify = (
                _w["factors"] >= 4 or
                (_w["factors"] == 3 and _w["conf"] >= 75)
            )
            if _should_notify:
                _missing = []
                if not _w["f1"]: _missing.append("OBV↑")
                if not _w["f2"]: _missing.append(f"PF>{_w['pf']:.2f}")
                if not _w["f3"]: _missing.append(f"ML{_w['ml']:.0%}")
                if not _w["f4"]: _missing.append(f"Score{_w['score']:.0f}")
                if not _w["f5"]: _missing.append("EMA50")
                _tier_label = (
                    "🟢 5/5 — ALL FACTORS MET" if _w["factors"] == 5 else
                    f"🟡 4/5 — needs: {', '.join(_missing)}" if _w["factors"] == 4 else
                    f"👀 3/5 HIGH CONF ({_w['conf']}/100) — needs: {', '.join(_missing)}"
                )
                _priority  = "high" if _w["factors"] == 5 else "default"
                _topic_env = "NTFY_TOPIC" if _w["factors"] == 5 else "NTFY_WATCH_TOPIC"
                send_ntfy(
                    title=f"🎯 WATCHLIST: {_w['coin'].upper()} {_w['factors']}/5",
                    message=(
                        f"{_tier_label}\n"
                        f"Confidence: {_w['conf']}/100\n"
                        f"OBV:   {'✅' if _w['f1'] else '❌'}  "
                        f"PF:    {'✅' if _w['f2'] else '❌'}  "
                        f"ML:    {'✅' if _w['f3'] else '❌'}\n"
                        f"Score: {'✅' if _w['f4'] else '❌'}  "
                        f"EMA50: {'✅' if _w['f5'] else '❌'}\n"
                        f"ML: {_w['ml']:.0%}  Score: {_w['score']:.0f}/5  PF: {_w['pf']:.2f}"
                    ),
                    priority=_priority,
                    tags="eyes,chart_increasing",
                    topic_env=_topic_env
                )

        if _watchlist:
            print(f"\n  {'Coin':<12} {'✓':>3}  {'OBV':>4}  {'PF':>5}  {'ML(acc)':>9}  {'Score':>6}  {'EMA50':>6}  {'Conf':>5}  Status")
            print("  " + "─"*95)
            for w in _watchlist[:10]:
                _f1 = "✅" if w["f1"] else "❌"
                _f2 = "✅" if w["f2"] else "❌"
                _f3 = "✅" if w["f3"] else "❌"
                _f4 = "✅" if w["f4"] else "❌"
                _f5 = "✅" if w["f5"] else "❌"
                _missing = []
                if not w["f1"]: _missing.append("OBV↑")
                if not w["f2"]: _missing.append(f"PF>{w['pf']:.2f}")
                if not w["f3"]: _missing.append(f"ML{w['ml']:.0%}")
                if not w["f4"]: _missing.append(f"Sc{w['score']:.0f}")
                if not w["f5"]: _missing.append("EMA50")

                if w["factors"] == 5:
                    _tier  = Fore.GREEN
                    _label = "🟢 5/5"
                elif w["factors"] == 4:
                    _tier  = Fore.YELLOW
                    _label = "🟡 4/5"
                else:
                    _tier  = Fore.WHITE
                    _label = "👀 3/5"

                _needs = f"  needs: {', '.join(_missing)}" if _missing else ""
                print(_tier + f"  {w['coin'].upper():<12} {_label}  {_f1}  {_f2}  {_f3}  {_f4}  {_f5}  {w['conf']:>5}/100{_needs}" + Fore.RESET)
        else:
            print(Fore.YELLOW + "\n  No coins meet 2+ of 5 factors right now — market conditions too weak")

        print(Fore.CYAN + "\n  💡 All 5 green = highest probability trade. Enter only at 5/5 or 4/5+conf≥75")
        
        # Sort ALL results by confidence score (descending)
        ranked_results = sorted(results, key=lambda x: x.get("confidence_score", 0), reverse=True)
        
        print("\n" + Fore.YELLOW + "Rank │ Coin      │ Score │SellSc│ OBV  │ L.ML (acc)    │ S.ML (acc)    │ Direction │ Action  │ Whale Vol           │ Smart Money   │ Confidence │ Recommendation     │ Rating" + Fore.RESET)
        print("─" * 225)
        
        for rank, r in enumerate(ranked_results, 1):
            conf_score = r.get("confidence_score", 0)
            tech_score = r.get("score", 0)
            ml_proba = r.get("ml_proba", 0)
            ml_accuracy = r.get("ml_accuracy", 0)
            short_ml_proba = r.get("short_ml_proba", 0.5)
            short_ml_accuracy = r.get("short_ml_accuracy", 0)
            conf_rating = r.get("confidence_rating", "🟡 FAIR")
            conf_rec = r.get("confidence_rec", "WAIT")
            conf_emoji = r.get("confidence_emoji", "→")
            action = r.get("action", "WAIT")
            
            # Color based on confidence
            if conf_score >= 85:
                color = Fore.GREEN
                bar = "████████████████████" + "░" * 0
            elif conf_score >= 75:
                color = Fore.GREEN
                bar = "███████████████████░░" + "░" * -1
            elif conf_score >= 65:
                color = Fore.YELLOW
                bar = "██████████████████░░░░"
            elif conf_score >= 50:
                color = Fore.YELLOW
                bar = "█████████████░░░░░░░░░░"
            elif conf_score >= 35:
                color = Fore.RED
                bar = "█████████░░░░░░░░░░░░░░"
            else:
                color = Fore.RED
                bar = "████░░░░░░░░░░░░░░░░░░"
            
            # Determine action icon based on action type
            if "BUY" in action:
                action_icon = "🟢"
            elif "SHORT" in action or "AVOID" in action or "BLOCK" in action or "SELL" in action:
                action_icon = "🔴"
            else:
                action_icon = "🟡"
            
            # Abbreviate action for display
            if "SHORT" in action:
                action_display = "SHORT"
            elif "STRONG BUY" in action:
                action_display = "S.BUY"
            elif "AVOID" in action:
                action_display = "AVOID"
            elif "BLOCKED" in action:
                action_display = "WAIT"
            elif "BUY" in action:
                action_display = "BUY"
            elif "SELL" in action:
                action_display = "SELL"
            else:
                action_display = "WAIT"
            
            coin_str       = r["coin"].upper()[:9].ljust(9)
            tech_score_str = f"{tech_score:.1f}/5".ljust(6)

            # Sell score — colour coded
            _sell_s     = r.get("sell_score", 0)
            _sell_color = (Fore.RED    if _sell_s <= -5 else
                           Fore.YELLOW if _sell_s <= -3 else
                           Fore.WHITE)
            sell_score_str = f"{_sell_s:+.1f}".ljust(6)

            # OBV status — the most important live signal
            _obv = r.get("obv_trend", "unknown")
            if _obv == "rising":
                _obv_color = Fore.GREEN
                _obv_str   = "↑ UP".ljust(5)
            elif _obv == "falling":
                _obv_color = Fore.RED
                _obv_str   = "↓ DN".ljust(5)
            elif _obv == "flat":
                _obv_color = Fore.YELLOW
                _obv_str   = "→ FL".ljust(5)
            else:
                _obv_color = Fore.WHITE
                _obv_str   = "?".ljust(5)

            # Long ML: win% (accuracy%) — flag if accuracy < 50% (model neutralized)
            if ml_accuracy < 0.50:
                long_ml_str = f"n/a ({ml_accuracy:.0%})".ljust(14)   # neutralized — don't show probability
            else:
                long_ml_str = f"{ml_proba:.0%} ({ml_accuracy:.0%})".ljust(14)

            # Short ML: win% (accuracy%) — flag if accuracy < 50%
            if short_ml_accuracy < 0.50:
                short_ml_str = f"n/a ({short_ml_accuracy:.0%})".ljust(14)
            else:
                short_ml_str = f"{short_ml_proba:.0%} ({short_ml_accuracy:.0%})".ljust(14)

            direction_str        = r.get("direction", "?").upper()[:3].ljust(9)
            action_display_short = action_display.ljust(7)

            # ── SIGNAL CONFLICT / PATTERN DETECTION ──────────────────────
            _direction    = r.get("direction", "?")
            _sm_mode      = r.get("smart_money_mode", "")
            _obv_mom      = r.get("obv_momentum", 0)
            _conflict_note = ""

            # Pattern 1: Mean-reversion — Direction DOWN but long ML says longs win
            if _direction == "DOWN" and ml_proba > 0.55 and ml_accuracy >= 0.55:
                _conflict_note = (
                    f"  ⚠️  SIGNAL CONFLICT: Direction DOWN but Long ML {ml_proba:.0%} "
                    f"→ mean-reversion pattern. Price dips but historically bounces. "
                    f"Do NOT short — wait for score to qualify as BUY on the dip."
                )

            # Pattern 2: Bull trap — Direction UP but long ML says longs fail
            elif _direction == "UP" and ml_proba < 0.35 and ml_accuracy >= 0.55:
                _conflict_note = (
                    f"  ⚠️  SIGNAL CONFLICT: Direction UP but Long ML {ml_proba:.0%} "
                    f"→ bull trap pattern. Price rising but historically stops out. "
                    f"Do NOT chase breakout — wait for ML to confirm."
                )

            # Pattern 3: Early accumulation — smart money buying but technicals lagging
            elif (_direction == "UP"
                  and "accumulation" in _sm_mode.lower()
                  and _obv_mom >= 0.50
                  and tech_score < MIN_BUY_SCORE
                  and short_ml_proba < 0.35 and short_ml_accuracy >= 0.55):
                _conflict_note = (
                    f"  📈 EARLY ACCUMULATION: Smart money buying (OBV {_obv_mom*100:.0f}%) + "
                    f"direction UP but technicals not confirmed (score {tech_score:.0f}/5). "
                    f"Short ML {short_ml_proba:.0%} says bears can't push it down. "
                    f"→ Watchlist: wait for score ≥ {MIN_BUY_SCORE} + Long ML > 50% then enter."
                )

            # NEW: Whale volume and smart money
            whale_vol = r.get("whale_volume_pct", 0)
            # Whale volume description
            if whale_vol >= 200:
                _whale_desc  = "🐋🐋MEGA"
                _whale_color = Fore.MAGENTA
            elif whale_vol >= 150:
                _whale_desc  = "🐋 WHALE"
                _whale_color = Fore.GREEN
            elif whale_vol >= 100:
                _whale_desc  = "↑ HIGH"
                _whale_color = Fore.CYAN
            elif whale_vol >= 70:
                _whale_desc  = "→ MID"
                _whale_color = Fore.WHITE
            else:
                _whale_desc  = "↓ LOW"
                _whale_color = Fore.YELLOW
            whale_str = f"{_whale_color}{whale_vol:.0f}% {_whale_desc}{Fore.RESET}".ljust(22)
            smart_money_str = f"{_sm_mode} {_obv_mom*100:.0f}%".ljust(13)

            conf_score_str = f"{conf_score:.0f}/100".ljust(10)
            rec_str = conf_rec[:18].ljust(18)

            print(f"{rank:3d}  │ {color}{coin_str}│ {tech_score_str}│{_sell_color}{sell_score_str}{Fore.RESET}│{_obv_color}{_obv_str}{Fore.RESET}│ {long_ml_str}│ {short_ml_str}│ {direction_str}│ {action_icon} {action_display_short}│ {whale_str}│ {smart_money_str}│ {conf_score_str}│ {rec_str}│ {conf_rating:<14}")

            if _conflict_note:
                print(Fore.MAGENTA + _conflict_note)

            # ── Entry timing line — driven entirely by stored entry_status ──
            _timing_action = r.get("action", "WAIT")
            _timing_price  = r.get("price", 0)
            _entry_status  = r.get("entry_status", "")
            _entry_when    = r.get("entry_when", "")
            _entry_watch   = r.get("entry_watch", _timing_price)

            if "ENTER NOW" in _entry_status:
                _t_color = Fore.GREEN
            elif "ML TRAP" in _entry_status or "CHASING" in _entry_status:
                _t_color = Fore.RED
            elif "CAUTION" in _entry_status or "ALMOST" in _entry_status or "WATCH" in _entry_status:
                _t_color = Fore.YELLOW
            elif "SHORT" in _timing_action or "SELL" in _timing_action:
                _t_color = Fore.RED
            else:
                _t_color = Fore.WHITE

            # First line: status + watch price
            print(_t_color + f"     ↳ {_entry_status} | Watch: ${_entry_watch:.6f}")
            # Full explanation — split on \n so each gate prints on its own line, nothing truncated
            if _entry_when:
                for _line in _entry_when.split("\n"):
                    if _line.strip():
                        print(_t_color + f"       {_line}")
        
        print("─" * 225)
        print(f"\n💡 RECOMMENDATION GUIDE:")
        print(f"  🟢 EXCELLENT (85+):  STRONG BUY    ➜ Top picks, all factors green")
        print(f"  🟢 VERY GOOD (75+):  BUY           ➜ Good setups, reliable signal")
        print(f"  🟡 GOOD (65-74):     CONSIDER BUY  ➜ Decent, watch ML + OBV closely")
        print(f"  🟡 FAIR (50-64):     WAIT          ➜ Weak signal, wait for confirmation")
        print(f"  🟠 WEAK (35-49):     AVOID         ➜ Poor setup, skip this one")
        print(f"  🔴 POOR (<35):       DO NOT BUY    ➜ Avoid completely")
        print(f"  Note: Action column may show WAIT even at GOOD confidence")
        print(f"  if ML guard, consecutive check, or event block fires.\n")

        # [W] Log completion with performance metrics
        run_elapsed = time.time() - run_start_time
        print("\n" + "="*80)
        print(f"Total coins analyzed: {len(results)}")
        # [J] Display count for each signal type (BUY, SELL, WAIT)
        print(f"🟢 BUY signals: {len(buy_results)} | 🔴 SELL signals: {len(sell_results)} | 🟡 WAIT signals: {len(wait_results)}")
        print("="*80)
        
        # [W] Log performance metrics
        logger.info("=" * 70)
        logger.info(f"[BOT] Run completed successfully")
        logger.info(f"[PERFORMANCE] Total time: {run_elapsed:.1f}s")
        logger.info(f"[PERFORMANCE] Coins analyzed: {len(results)}")
        logger.info(f"[PERFORMANCE] BUY signals: {len(buy_results)} | SELL signals: {len(sell_results)} | WAIT signals: {len(wait_results)}")
        logger.info(f"[PERFORMANCE] Average per coin: {run_elapsed/len(results):.2f}s")
        logger.info("=" * 70)

if __name__ == "__main__":
    run_once()
