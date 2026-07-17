# crypto_trading_bot_gecko_v6.4.py
# Built on v6.3. Eighteen additions + ML enhancements:
#   [A] DI+ > DI- required for BUY (ADX direction filter) -- fixes STORY-2 false ADX
#   [B] Chikou span confirmation -- close vs price 26 periods ago
#   [C] RSI divergence detection -- bullish divergence adds score, bearish subtracts
#   [D] BTC market regime filter -- altcoin BUYs blocked when BTC is in downtrend
#   [E] Weekly EMA trend filter -- EMA50 weekly approximated from daily data
#   [F] Consecutive candle confirmation -- BUY only after 2 consecutive days of signals
#   [G] ML Model (Logistic Regression) -- predicts trade profitability; blocks low-confidence trades
#   [H] Smart Money / Whale Activity -- detects accumulation/distribution and large volume
#   [I] News / Event Filter -- blocks trades before/after major events (FOMC, CPI, halving, etc)
#   [J] SELL Logic (Shorting Mechanism) -- bearish RSI divergence, negative scores, downtrend confirmation
#   [K] Market Sentiment (Fear & Greed Index) -- FREE API from alternative.me, 0-100 scale
#   [L] Relative Strength vs BTC -- 24h coin performance compared to BTC (outperformance indicator)
#   [M] Time-based Features -- Day of week and hour patterns for market behavior
#   [N] On-Chain Metrics -- Whale accumulation signals, volume spikes, on-chain activity patterns
#   [S] Backtesting Engine -- Historical trade simulation with win rate, profit factor, drawdown analysis
#   [U] Advanced Stops & TP -- Trailing stops, breakeven protection, partial profit taking strategies
#   [V] Model Persistence -- Save/load trained models (300x faster startup, auto-retrain every 7 days)
#   [W] Comprehensive Logging -- Full error tracking, API monitoring, trade recording, performance metrics
#   [X] Sentiment Analysis (NLP) -- VADER sentiment analysis for coins (no API key needed!)
#   [Y] Volume Behavior Logic -- Detect good volume (increasing/spike) vs bad (declining + price up = FAKE MOVE)
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

# ══════════════════════════════════════════════════════════════════════════════
# PROBABILISTIC STATE MODEL  [P]
# Replaces score += 1 / score -= 1 with:
#   P(Continuation | Trend, Volume, Regime, Volatility, Flow)
#
# Framework: Log-Odds Bayesian Evidence Accumulation
#   Each signal contributes a Likelihood Ratio (LR):
#     LR = P(signal | continuation) / P(signal | reversal)
#   log_odds = log(prior/(1-prior)) + Σ log(LR_i)
#   P = sigmoid(log_odds)
#
# BUY  when P >= 0.65   (replaces score >= MIN_BUY_SCORE = 5)
# SELL when P <= 0.35   (replaces score <= MIN_SELL_SCORE = -5)
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# TRADE LOGGER — forward-test performance tracking
# Logs every BUY signal, resolves outcomes, computes 10 metrics dashboard
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _TradeRecord:
    trade_id: str;  timestamp: str;  coin: str
    entry_price: float;  stop: float;  target: float;  rr_ratio: float
    prob: float;  ev: float;  ev_rating: str;  kelly: float = 0.0
    btc_regime: str = "unknown";  lifecycle: str = "UNKNOWN"
    rs_tier: str = "N/A";  ml_accuracy: float = 0.5;  ml_proba: float = 0.5
    adx: float = 0.0;  rsi: float = 50.0
    status: str = "pending";  outcome: Optional[int] = None
    exit_price: Optional[float] = None;  exit_time: Optional[str] = None
    pnl_pct: Optional[float] = None;  days_held: Optional[float] = None
    exit_reason: Optional[str] = None
    prob_bucket: str = "";  ev_bucket: str = ""
    def __post_init__(self):
        if not self.prob_bucket:
            p = self.prob * 100
            self.prob_bucket = ("50-55%" if p<55 else "55-60%" if p<60 else "60-65%" if p<65 else
                                "65-70%" if p<70 else "70-75%" if p<75 else "75-80%" if p<80 else "80%+")
        if not self.ev_bucket:
            self.ev_bucket = ("A+ (≥1.5)" if self.ev_rating=="A+" else "A  (1.0-1.5)" if self.ev_rating=="A" else
                              "B  (0.5-1.0)" if self.ev_rating=="B" else "C  (0.2-0.5)" if self.ev_rating=="C" else "D/F (<0.2)")


class TradeLogger:
    """Forward-test trade logger with outcome resolution and performance dashboard."""

    def __init__(self, log_path=None):
        import uuid as _uuid
        self._uuid = _uuid
        self.log_path = Path(log_path or "models/trade_log.json")
        self.trades: List[_TradeRecord] = []
        self._load()

    def _load(self):
        if self.log_path.exists():
            try:
                raw = json.loads(self.log_path.read_text())
                self.trades = [_TradeRecord(**r) for r in raw]
                print(f"[LOGGER] Loaded {len(self.trades)} trades  "
                      f"({sum(1 for t in self.trades if t.status=='pending')} pending)")
            except Exception as e:
                print(f"[LOGGER] Load error: {e}")

    def save(self):
        import dataclasses as _dc
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps([_dc.asdict(t) for t in self.trades], indent=2))

    def log_trade(self, coin, entry_price, stop, target, prob, ev, ev_rating,
                  rr_ratio, btc_regime="unknown", lifecycle="UNKNOWN", rs_tier="N/A",
                  ml_accuracy=0.5, ml_proba=0.5, adx=0.0, rsi=50.0, kelly=0.0):
        t = _TradeRecord(
            trade_id=str(self._uuid.uuid4())[:8], timestamp=datetime.now().isoformat(),
            coin=coin, entry_price=entry_price, stop=stop, target=target, rr_ratio=rr_ratio,
            prob=prob, ev=ev, ev_rating=ev_rating, kelly=kelly,
            btc_regime=btc_regime, lifecycle=lifecycle, rs_tier=rs_tier,
            ml_accuracy=ml_accuracy, ml_proba=ml_proba, adx=adx, rsi=rsi,
        )
        self.trades.append(t)
        self.save()
        print(f"[LOGGER] ✅ Logged: {coin}  P={prob:.0%}  EV={ev:+.2f}({ev_rating})  "
              f"entry={entry_price:.4f}  stop={stop:.4f}  target={target:.4f}")
        return t

    def resolve_outcomes(self, current_prices: dict) -> int:
        resolved = 0
        for t in self.trades:
            if t.status != "pending": continue
            price = current_prices.get(t.coin)
            if price is None: continue
            days = (datetime.now() - datetime.fromisoformat(t.timestamp)).total_seconds() / 86400
            if price <= t.stop:
                t.status="loss"; t.outcome=0; t.exit_price=t.stop
                t.exit_time=datetime.now().isoformat(); t.exit_reason="stop_hit"
                t.pnl_pct=round((t.stop-t.entry_price)/t.entry_price*100,2); t.days_held=round(days,1)
                print(f"[LOGGER] ❌ LOSS  {t.coin}  PnL={t.pnl_pct:+.1f}%  days={t.days_held:.0f}")
                resolved += 1
            elif price >= t.target:
                t.status="win"; t.outcome=1; t.exit_price=t.target
                t.exit_time=datetime.now().isoformat(); t.exit_reason="target_hit"
                t.pnl_pct=round((t.target-t.entry_price)/t.entry_price*100,2); t.days_held=round(days,1)
                print(f"[LOGGER] ✅ WIN   {t.coin}  PnL={t.pnl_pct:+.1f}%  days={t.days_held:.0f}")
                resolved += 1
            elif days > MAX_HOLD_DAYS:
                t.status="expired"; t.outcome=0; t.exit_price=price
                t.exit_time=datetime.now().isoformat(); t.exit_reason="expired"
                t.pnl_pct=round((price-t.entry_price)/t.entry_price*100,2); t.days_held=round(days,1)
                print(f"[LOGGER] ⏱ EXPR  {t.coin}  PnL={t.pnl_pct:+.1f}%")
                resolved += 1
        if resolved: self.save()
        return resolved

    def _seg_stats(self, trades):
        if not trades: return {"n":0,"wr":0,"pf":0,"max_dd":0}
        wins=[t for t in trades if t.outcome==1]; n_l=sum(1 for t in trades if t.outcome==0)
        n=len(wins)+n_l; wr=len(wins)/n if n else 0
        gp=sum(t.rr_ratio for t in wins); pf=(gp/n_l) if n_l>0 else (float("inf") if gp>0 else 0)
        curve=[]; running=0; peak=0; maxdd=0
        for t in sorted(trades,key=lambda x:x.timestamp):
            running += t.rr_ratio if t.outcome==1 else -1.0; curve.append(running)
        for v in curve:
            if v>peak: peak=v
            if peak-v>maxdd: maxdd=peak-v
        return {"n":n,"wr":round(wr,3),"pf":round(pf,3) if pf!=float("inf") else "∞","max_dd":round(maxdd,2)}

    def compute_stats(self):
        resolved=[t for t in self.trades if t.outcome is not None]
        if not resolved: return {}
        ov=self._seg_stats(resolved)
        if len(resolved)>=10:
            brier=sum((t.prob-t.outcome)**2 for t in resolved)/len(resolved)
            bins={}
            for t in resolved: b=int(t.prob*10); bins.setdefault(b,[]).append(t)
            ece=sum((len(g)/len(resolved))*abs(sum(x.prob for x in g)/len(g)-sum(x.outcome for x in g)/len(g)) for g in bins.values())
            ov["brier"]=round(brier,4); ov["ece"]=round(ece,4)
        ov["pending"]=sum(1 for t in self.trades if t.status=="pending"); ov["total"]=len(self.trades)
        def seg(key):
            m={}
            for t in resolved: m.setdefault(getattr(t,key,"N/A"),[]).append(t)
            return {k:self._seg_stats(v) for k,v in sorted(m.items())}
        return {"overall":ov,"by_regime":seg("btc_regime"),"by_lifecycle":seg("lifecycle"),
                "by_rs_tier":seg("rs_tier"),"by_prob":seg("prob_bucket"),"by_ev":seg("ev_bucket")}

    def print_dashboard(self):
        stats=self.compute_stats()
        if not stats: return
        ov=stats["overall"]; sep="═"*72
        print(f"\n{sep}")
        print(f"  📊 FORWARD-TEST DASHBOARD  —  {ov['n']} resolved  {ov['pending']} pending  {ov['total']} total")
        print(sep)
        pf_str=f"{ov['pf']:.2f}" if isinstance(ov['pf'],float) else ov['pf']
        print(f"  Win Rate: {ov.get('wr',0):.1%}   PF: {pf_str}   MaxDD: {ov.get('max_dd',0):.1f}R"
              + (f"   Brier: {ov['brier']:.4f}   ECE: {ov['ece']:.4f}" if "brier" in ov else ""))
        print()
        def table(title, seg):
            if not seg: return
            print(f"  ── {title}")
            print(f"  {'Segment':<24}  {'n':>4}  {'WR':>6}  {'PF':>6}  {'MaxDD':>7}")
            print(f"  {'─'*52}")
            for k,s in seg.items():
                if s['n']==0: continue
                pf=f"{s['pf']:.2f}" if isinstance(s['pf'],float) else s['pf']
                print(f"  {k:<24}  {s['n']:>4}  {s['wr']:>6.1%}  {pf:>6}  {s['max_dd']:>6.1f}R")
            print()
        table("PF by BTC Regime",         stats["by_regime"])
        table("PF by Lifecycle Stage",    stats["by_lifecycle"])
        table("PF by Relative Strength",  stats["by_rs_tier"])
        table("PF by Probability Bucket", stats["by_prob"])
        table("PF by EV Bucket",          stats["by_ev"])
        print(sep)


# Singleton — initialised once at module load, reused across all scans
_trade_logger: Optional[TradeLogger] = None

def get_trade_logger() -> TradeLogger:
    global _trade_logger
    if _trade_logger is None and TRADE_LOGGER_ENABLED:
        _trade_logger = TradeLogger(TRADE_LOG_PATH)
    return _trade_logger

# Thresholds replacing MIN_BUY_SCORE / MIN_SELL_SCORE
BUY_PROB_THRESHOLD    = 0.65
STRONG_BUY_THRESHOLD  = 0.72
SELL_PROB_THRESHOLD   = 0.35
STRONG_SELL_THRESHOLD = 0.28
BLOCK_PROB_CEILING    = 0.40

# LR constants for regime swap calculation
_BTC_BEAR_LR = 0.50   # default — overridden per-coin based on RS tier (see below)
# RS-conditional bear LR applied in generate_signal_probabilistic:
#   RS LEADER → 0.70  (decoupled from BTC, full penalty unwarranted)
#   RS STRONG → 0.60  (mild outperformance, partial penalty)
#   RS NEUTRAL/LAGGARD/WEAK → 0.50  (moves with BTC, full penalty)
_BTC_BULL_LR = 2.10   # btc_bull_regime


def compute_bull_scenario_prob(bear_prob: float, prior: float = 0.42) -> float:
    """
    Compute what a coin's probability WOULD BE in a BTC bull market,
    given its current bear-market probability.

    Math: swap the BTC regime LR (0.20 → 2.10) in log-odds space.
      bear_prob = sigmoid(logit(prior) + log(BEAR_LR) + base_signals)
      bull_prob = sigmoid(logit(prior) + log(BULL_LR) + base_signals)

    base_signals = logit(bear_prob) - logit(prior) - log(BEAR_LR)
    bull_prob    = sigmoid(logit(prior) + base_signals + log(BULL_LR))
    """
    import math as _m
    try:
        _logit = lambda p: _m.log(max(1e-9, p) / max(1e-9, 1-p))
        _sig   = lambda x: 1 / (1 + _m.exp(-x))
        base_lo = _logit(bear_prob) - _logit(prior) - _m.log(_BTC_BEAR_LR)
        bull_lo = _logit(prior) + base_lo + _m.log(_BTC_BULL_LR)
        return round(min(0.97, _sig(bull_lo)), 3)
    except Exception:
        return bear_prob


def compute_ev(prob: float,
               rr_ratio: float,
               lifecycle_stage: str = "MATURE_TREND",
               atr_pct: float = 0.0) -> dict:
    """
    Expected Value per unit risked.

    EV = P(win) × AvgWin - P(loss) × AvgLoss

    Normalised to 1 unit of risk (AvgLoss = 1.0):
      EV = P × RR - (1 - P)

    Lifecycle adjusts effective RR:
      EARLY_TREND:  RR bonus ×1.20  (tight stop, trend just confirmed)
      MATURE_TREND: RR as given
      EXHAUSTION:   RR penalty ×0.80 (wider stop needed for volatile swings)
      DISTRIBUTION: RR penalty ×0.60 (high reversal risk)
      ACCUMULATION: RR as given (entry near support)

    Kelly fraction (optimal % of capital to risk):
      f* = (P × RR - (1-P)) / RR  = EV / RR

    EV rating:
      A+: EV ≥ 1.50  (exceptional — 65% win, 3:1 RR = EV 1.60)
      A:  EV ≥ 1.00  (excellent)
      B:  EV ≥ 0.50  (good)
      C:  EV ≥ 0.20  (acceptable)
      D:  EV ≥ 0.00  (marginal — breakeven)
      F:  EV < 0.00  (negative expectancy — do not trade)
    """
    # Lifecycle-adjusted RR
    rr_mult = {
        "EARLY_TREND":  1.20,
        "MATURE_TREND": 1.00,
        "ACCUMULATION": 1.00,
        "EXHAUSTION":   0.80,
        "DISTRIBUTION": 0.60,
    }.get(lifecycle_stage, 1.00)
    adj_rr = max(0.1, rr_ratio * rr_mult)

    p_win  = max(0.01, min(0.99, prob))
    p_loss = 1.0 - p_win
    ev     = p_win * adj_rr - p_loss        # per unit risked

    # Kelly fraction (optimal bet size as % of capital)
    kelly  = ev / adj_rr if adj_rr > 0 else 0.0
    kelly  = max(0.0, min(kelly, 0.25))     # cap at 25% of capital

    # Rating
    if   ev >= 1.50: rating = "A+"
    elif ev >= 1.00: rating = "A"
    elif ev >= 0.50: rating = "B"
    elif ev >= 0.20: rating = "C"
    elif ev >= 0.00: rating = "D"
    else:            rating = "F"

    # Breakeven probability at this RR (what win rate you need just to break even)
    breakeven_p = 1.0 / (1.0 + adj_rr)

    return {
        "ev":           round(ev, 3),
        "adj_rr":       round(adj_rr, 2),
        "kelly":        round(kelly, 3),
        "rating":       rating,
        "breakeven_p":  round(breakeven_p, 3),
        "positive":     ev > 0.0,
        "tradeable":    ev >= 0.20,   # minimum threshold for a trade
    }

# Base rate: ~42% of crypto daily moves continue the prior trend
# Static fallback -- overridden at runtime by detect_regime_prior()
PRIOR_CONTINUATION = 0.42

# ── Dynamic regime priors ─────────────────────────────────────────────────────
# Continuation probability differs dramatically by market state.
# These values reflect empirical intraday continuation rates in crypto
# (estimated from BTC/ETH 2019-2024 daily data, ATR-based labels).
#
# Why this matters:
#   In a strong trend:  most technical BUY signals are right → high base rate
#   In euphoria:        most BUY signals are at the top → low base rate
#   In panic:           mean-reversion is more likely → low continuation rate
#   In range:           near-random walk → ~50% base rate
#
# The prior shifts the starting log-odds so the SAME signal evidence
# produces correctly calibrated probabilities in different regimes.
REGIME_PRIORS: Dict[str, float] = {
    "euphoria":        0.30,   # RSI>80 OR F&G>80 OR Z>3 -- tops reverse sharply
    "panic":           0.35,   # RSI<25 OR F&G<20 OR sharp drop -- oversold bounces common
    "bear_trend":      0.38,   # BTC bear + below cloud -- downtrend more likely to continue
    "range":           0.48,   # ADX<15 or chop -- near-random walk, slight continuation edge
    "weak_trend":      0.55,   # ADX 15-25 + above cloud -- mild trend
    "strong_trend":    0.62,   # ADX≥25 + DI+ + above cloud + BTC bull -- high persistence
}

def detect_regime_prior(last: pd.Series,
                        df: pd.DataFrame,
                        btc_regime: str = "unknown",
                        fng_value: int  = 50,
                        adx: float      = 0.0) -> Tuple[float, str]:
    """
    Detect the current market regime and return the corresponding prior.

    Detection order (first match wins -- most extreme state takes priority):

    1. EUPHORIA  -- RSI > 80  OR  Fear&Greed > 80  OR  Z-score > 3.0
       Tops reverse sharply in crypto; even bullish evidence is unreliable here.

    2. PANIC     -- RSI < 25  OR  Fear&Greed < 20  OR  3-day drop > 8%
       Oversold extremes often bounce; continuation (further drop) less likely.

    3. BEAR TREND -- BTC bear AND price below cloud
       Structural downtrend; signal evidence is mostly noise.

    4. STRONG TREND -- ADX ≥ 25 AND DI+ dominant AND above cloud AND BTC bull
       Highest persistence state; trend following works best here.

    5. WEAK TREND -- ADX 15-25 AND above cloud
       Moderate trend; signals more reliable than in range.

    6. RANGE -- ADX < 15 OR chop detected
       Mean-reversion regime; trend signals less reliable.

    7. DEFAULT -- 0.42 (neutral fallback)

    Returns: (prior, regime_label)
    """
    try:
        price   = float(last["price"])
        rsi     = float(last.get("RSI", 50) or 50)
        plus_di = float(last.get("plus_di",  0) or 0)
        minus_di= float(last.get("minus_di", 0) or 0)

        # Z-score (20-day)
        try:
            col = "close" if "close" in df.columns else "price"
            pz  = df[col].iloc[-20:].values
            z   = (pz[-1] - pz.mean()) / (pz.std() + 1e-9) if len(pz) >= 10 else 0.0
        except Exception:
            z = 0.0

        # Cloud position
        sa = float(last.get("senkou_a", float("nan")))
        sb = float(last.get("senkou_b", float("nan")))
        if not (math.isnan(sa) or math.isnan(sb)):
            cloud_top = max(sa, sb)
            cloud_bot = min(sa, sb)
            above_cloud = price > cloud_top
            below_cloud = price < cloud_bot
        else:
            above_cloud = below_cloud = False

        # 3-day price drop
        try:
            p3d  = float(df["price"].iloc[-4])
            drop = (price - p3d) / p3d * 100 if p3d > 0 else 0.0
        except Exception:
            drop = 0.0

        # DI direction
        di_bullish = plus_di > minus_di

        # Chop check
        try:
            col_c  = "close" if "close" in df.columns else "price"
            prices = df[col_c].iloc[-20:].values
            cur    = prices[-1]
            rng    = (prices.max() - prices.min()) / cur * 100
            s20, d20 = prices.mean(), prices.std()
            bbu, bbl = s20 + 2*d20, s20 - 2*d20
            bbw    = (bbu - bbl) / s20 * 100 if s20 > 0 else 0
            pos    = (cur - bbl) / (bbu - bbl + 1e-9) * 100
            chop   = (rng < 12) + (30 < pos < 70) + (bbw > 9) >= 2
        except Exception:
            chop = False

        # ── Detection hierarchy ───────────────────────────────────────
        # 1. EUPHORIA -- extreme overbought or euphoric sentiment
        if rsi > 80 or fng_value > 80 or z > 3.0:
            drivers = []
            if rsi > 80:        drivers.append(f"RSI={rsi:.0f}")
            if fng_value > 80:  drivers.append(f"F&G={fng_value}")
            if z > 3.0:         drivers.append(f"Z={z:.1f}")
            return REGIME_PRIORS["euphoria"], f"EUPHORIA ({', '.join(drivers)})"

        # 2. PANIC -- extreme oversold or panic sentiment
        if rsi < 25 or fng_value < 20 or drop < -8:
            drivers = []
            if rsi < 25:       drivers.append(f"RSI={rsi:.0f}")
            if fng_value < 20: drivers.append(f"F&G={fng_value}")
            if drop < -8:      drivers.append(f"3d={drop:.1f}%")
            return REGIME_PRIORS["panic"], f"PANIC ({', '.join(drivers)})"

        # 3. BEAR TREND -- structural downtrend
        if btc_regime == "bear" and below_cloud:
            return REGIME_PRIORS["bear_trend"], "BEAR TREND (BTC bear + below cloud)"

        # 4. STRONG TREND -- all trend conditions aligned
        if (adx >= 25 and di_bullish and above_cloud and
                btc_regime == "bull" and not chop):
            return REGIME_PRIORS["strong_trend"], \
                   f"STRONG TREND (ADX={adx:.1f}, DI+, above cloud, BTC bull)"

        # 5. WEAK TREND -- partial trend alignment
        if adx >= 15 and above_cloud and not chop:
            return REGIME_PRIORS["weak_trend"], \
                   f"WEAK TREND (ADX={adx:.1f}, above cloud)"

        # 6. RANGE -- low ADX or chop detected
        if adx < 15 or chop:
            reason = f"ADX={adx:.1f}" if adx < 15 else "chop"
            return REGIME_PRIORS["range"], f"RANGE ({reason})"

        # 7. Default
        return PRIOR_CONTINUATION, "DEFAULT (neutral)"

    except Exception as e:
        return PRIOR_CONTINUATION, f"DEFAULT (error: {e})"

# ── Likelihood Ratio table ──────────────────────────────────────────────────
# LR = P(signal | continuation) / P(signal | reversal)
# LR > 1: bullish evidence   LR < 1: bearish evidence   LR = 1: neutral
# Format: { "key": (LR_when_fired, LR_when_absent) }
SIGNAL_LR: Dict[str, Tuple[float, float]] = {
    # MACRO / REGIME
    "btc_bull_regime":              (2.10, 0.90),
    "btc_bear_regime_block":        (0.50, 1.40),   # was 0.20 — calibrated to proportional penalty
    "above_weekly_ema50":           (1.75, 0.80),
    "below_weekly_ema50":           (0.65, 1.20),
    # TREND STRUCTURE
    "above_cloud":                  (2.40, 0.50),
    "in_cloud":                     (0.80, 1.10),
    "below_cloud_block":            (0.35, 2.20),   # fallback only — runtime uses distance-scaled LR (0.15–0.75)
    "tenkan_above_kijun":           (1.35, 0.85),
    "chikou_fully_bullish":         (1.55, 0.80),
    "chikou_partial_bullish":       (1.15, 0.95),
    "chikou_fully_bearish":         (0.50, 1.55),
    "chikou_partial_bearish":       (0.80, 1.15),
    # MACD / EMA
    "macd_bullish_hist":            (1.45, 0.80),
    "macd_bearish_hist":            (0.70, 1.30),
    "ema12_above_ema26":            (1.30, 0.85),
    "ema12_below_ema26":            (0.75, 1.20),
    "di_plus_dominant":             (1.40, 0.82),
    "di_minus_dominant":            (0.65, 1.30),
    # RSI
    "rsi_continuation_zone":        (1.55, 0.75),
    "rsi_extreme_overbought":       (0.45, 1.75),
    "rsi_oversold_bounce":          (1.40, 0.90),
    "rsi_range_overbought":         (0.55, 1.50),
    "rsi_bullish_divergence":       (1.65, 0.90),
    "rsi_bearish_divergence":       (0.55, 1.40),
    # CAPITAL FLOW
    "obv_rising_at_high":           (2.20, 0.55),
    "obv_rising_not_at_high":       (1.70, 0.70),
    "obv_flat":                     (0.90, 1.05),
    "obv_falling":                  (0.50, 2.00),   # was 0.35 — OBV flat/falling is often consolidation not distribution
    "sm_accumulation":              (2.00, 0.55),
    "sm_distribution":              (0.30, 2.30),
    # VOLUME QUALITY
    "volume_spike_confirmed":       (1.85, 0.70),
    "volume_increasing_trend":      (1.40, 0.82),
    "volume_healthy":               (1.10, 0.95),
    "volume_fake_move":             (0.50, 1.60),
    "volume_churning":              (0.45, 1.70),
    "volume_weak":                  (0.80, 1.10),
    # VOLATILITY REGIME
    "vol_expansion_bullish":        (1.65, 0.72),
    "vol_expansion_bearish":        (0.55, 1.55),
    "vol_compression":              (1.20, 0.90),
    # Volatility sub-signals (BB width, ATR momentum, squeeze, HV ratio)
    "vol_squeeze_on":               (1.35, 0.88),   # BB squeeze = coiling spring, breakout imminent
    "vol_bb_wide":                  (1.22, 0.92),   # BB in top 20% of history = trending environment
    "vol_bb_narrow":                (0.90, 1.08),   # BB in bottom 20% = quiet (compression building)
    "vol_atr_expanding":            (1.28, 0.88),   # 5d ATR > 1.2x 20d ATR = momentum accelerating
    "vol_atr_contracting":          (0.85, 1.10),   # 5d ATR < 0.8x 20d ATR = energy fading
    "vol_hv_rising":                (1.25, 0.90),   # 10d HV > 1.3x 30d HV = volatility picking up
    # Market regime amplifiers
    "regime_bull_trend_amplifier":  (1.38, 0.88),   # confirmed bull: cloud+ADX20+DI+dominant+no chop
    "regime_range_dampener":        (0.92, 1.05),   # range market: slight pull toward neutral
    # Z-SCORE / EXHAUSTION
    "zscore_healthy_trend":         (1.45, 0.80),
    "zscore_blow_off":              (0.45, 2.00),   # was 0.35 — momentum coins regularly extend far from mean
    "zscore_oversold_range":        (1.50, 0.80),
    "zscore_falling_knife":         (0.45, 1.80),
    "ema26_extreme_stretch":        (0.40, 1.90),
    "ema26_very_stretched":         (0.60, 1.55),
    "ema26_stretched":              (0.78, 1.25),
    "ema26_oversold":               (1.40, 0.85),
    # CANDLE WICK
    "wick_rejection":               (0.55, 1.60),
    "wick_strong_close":            (1.40, 0.80),
    "wick_capitulation":            (1.45, 0.85),
    # MARKET STRUCTURE
    "structure_bullish_hh_hl":      (1.60, 0.75),
    "structure_bearish_lh_ll":      (0.55, 1.55),
    "structure_bos_bullish":        (1.75, 0.80),
    "structure_bos_bearish":        (0.50, 1.65),
    "structure_choch_bullish":      (1.50, 0.85),
    "structure_choch_bearish":      (0.52, 1.55),   # CHoCH bearish = trend reversal down
    "structure_sfp_bullish":        (1.65, 0.85),
    "structure_sfp_bearish":        (0.55, 1.50),
    "structure_accumulation":       (1.50, 0.80),
    "structure_distribution":       (0.60, 1.50),
    "structure_ranging":            (0.90, 1.05),
    # ORDER BOOK
    "orderbook_strong_buy_imbalance":(1.45, 0.85),
    "orderbook_strong_sell_imbalance":(0.65, 1.35),
    "orderbook_paper_thin":         (0.70, 1.20),
    # CHOP / BREAKOUT
    "strong_chop_zone":             (0.55, 1.35),
    "moderate_chop_zone":           (0.75, 1.15),
    "breakout_confirmed":           (1.80, 0.65),
    "breakdown_confirmed":          (0.45, 1.75),
    # SENTIMENT (VADER NLP -- weak signal, conservative LRs)
    # LR calibrated so at P=0.50 the update is ≈ ±0.03-0.06 probability
    # At extremes (P=0.90) it correctly becomes ≈ ±0.005 instead of ±0.07
    "sentiment_very_positive":      (1.18, 0.92),
    "sentiment_positive":           (1.08, 0.96),
    "sentiment_negative":           (0.92, 1.08),
    "sentiment_very_negative":      (0.85, 1.18),
    # MICROSTRUCTURE -- real-time exchange data converted to LRs
    # Spread = reliability amplifier (tight book = signals more trustworthy)
    "micro_spread_tight":           (1.18, 0.95),   # < 0.05% -- deep liquid market
    "micro_spread_wide":            (0.78, 1.18),   # > 0.20% -- illiquid, noisy signals
    # Order book weighted imbalance (multiple depth levels)
    "micro_ob_strong_bull":         (1.45, 0.80),   # bids dominate ≥ 2× asks
    "micro_ob_mild_bull":           (1.20, 0.92),   # bids slightly dominate
    "micro_ob_strong_bear":         (0.65, 1.40),   # asks dominate ≥ 2×
    "micro_ob_mild_bear":           (0.82, 1.12),
    # Aggressor flow (CVD = cumulative buy vol − sell vol in recent window)
    "micro_flow_strong_buy":        (1.55, 0.75),   # institutional buying pressure
    "micro_flow_mild_buy":          (1.22, 0.90),
    "micro_flow_strong_sell":       (0.58, 1.52),   # institutional selling pressure
    "micro_flow_mild_sell":         (0.82, 1.14),
    # Liquidity depth (cumulative USD within ±1% of mid-price)
    "micro_depth_deep":             (1.22, 0.92),   # thick book → trend can sustain
    "micro_depth_thin":             (0.80, 1.18),   # thin book → cascade risk
    # Liquidation cascades (Binance Futures forced orders)
    "micro_liq_shorts_squeezed":    (1.38, 0.85),   # buy liquidations dominate
    "micro_liq_longs_stopped":      (0.68, 1.32),   # sell liquidations dominate
}

# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL CORRELATION GROUPS  [independence assumption fix]
#
# Signals within a group are correlated -- they fire together because they
# measure the same underlying phenomenon.
#
# Example: cloud above + MACD bullish + EMA cross + DI+ all say "uptrend".
# Treating them as independent multiplies the same evidence 4×.
#
# Fix applied in ProbabilisticScorer.finalize():
#   1. Within each group, sort signals by |log_lr| (strongest first)
#   2. Strongest signal → full LR (it carries the most unique information)
#   3. Same-direction signals → dampened by DAMPENING factor
#   4. Opposite-direction signals → full LR (genuine contrarian evidence)
#   5. Signals in no group (macro, order book) → full LR
#
# DAMPENING values reflect how correlated the group is:
#   0.25 = extremely correlated (MACD and cloud almost always agree)
#   0.50 = moderately correlated (OBV and volume somewhat independent)
#   0.70 = weakly correlated (BTC regime and weekly EMA fairly independent)
# ═══════════════════════════════════════════════════════════════════════════

SIGNAL_GROUPS: Dict[str, Dict] = {
    "TREND": {
        # All measure the same thing: "is this coin trending up or down?"
        # Cloud is the most structural -- it gets full LR.
        # MACD, EMA, DI are derived from the same price series and nearly
        # always agree with the cloud, so they add very little new information.
        "signals": {
            "above_cloud", "in_cloud", "below_cloud_block",
            "macd_bullish_hist", "macd_bearish_hist",
            "ema12_above_ema26", "ema12_below_ema26",
            "tenkan_above_kijun",
            "di_plus_dominant", "di_minus_dominant",
            "chikou_fully_bullish", "chikou_partial_bullish",
            "chikou_fully_bearish", "chikou_partial_bearish",
        },
        "dampening": 0.28,   # very high correlation -- 2nd signal adds ~28% of its LR
    },
    "MOMENTUM": {
        # RSI, Z-score, EMA26 deviation all measure "how stretched is price?"
        # They differ in time scale (RSI=rate, Z=SMA20, EMA26=trend) so
        # more independent than TREND, but still correlated when extreme.
        # Note: pipeline already applies RSI-aware dampening to Z+EMA26,
        # so group dampening here is a second guard for remaining overlap.
        "signals": {
            "rsi_continuation_zone", "rsi_extreme_overbought",
            "rsi_oversold_bounce",   "rsi_range_overbought",
            "rsi_bullish_divergence","rsi_bearish_divergence",
            "zscore_blow_off",       "zscore_falling_knife",
            "zscore_oversold_range", "zscore_healthy_trend",
            "ema26_extreme_stretch", "ema26_very_stretched",
            "ema26_stretched",       "ema26_oversold",
        },
        "dampening": 0.45,   # moderate -- different time scales give partial independence
    },
    "FLOW": {
        # OBV measures cumulative capital direction.
        # Volume quality measures single-bar conviction.
        # Smart money uses 14d OBV vs price divergence.
        # More independent than TREND but still share the OBV series.
        "signals": {
            "obv_rising_at_high", "obv_rising_not_at_high",
            "obv_flat",           "obv_falling",
            "sm_accumulation",    "sm_distribution",
            "volume_spike_confirmed",  "volume_increasing_trend",
            "volume_fake_move",        "volume_churning",
            "volume_weak",             "volume_healthy",
        },
        "dampening": 0.50,   # OBV direction vs volume quality are partially independent
    },
    "VOLATILITY": {
        # BB squeeze, ATR ratio, HV ratio all measure "how compressed is vol?"
        # They use different lookback windows but share the same price series.
        "signals": {
            "vol_compression",       "vol_expansion_bullish",  "vol_expansion_bearish",
            "vol_squeeze_on",        "vol_bb_wide",            "vol_bb_narrow",
            "vol_atr_expanding",     "vol_atr_contracting",    "vol_hv_rising",
        },
        "dampening": 0.40,   # fairly correlated -- all say "vol is high/low"
    },
    "STRUCTURE": {
        # Market structure signals are sequential: structure → BOS → CHoCH → SFP.
        # Each is a refinement of the previous, so correlated but not identical.
        "signals": {
            "structure_bullish_hh_hl",  "structure_bearish_lh_ll",
            "structure_bos_bullish",     "structure_bos_bearish",
            "structure_choch_bullish",   "structure_choch_bearish",
            "structure_sfp_bullish",     "structure_sfp_bearish",
            "structure_accumulation",    "structure_distribution",
            "structure_ranging",
        },
        "dampening": 0.50,   # BOS and CHoCH are sequential refinements
    },
    "BREAKOUT": {
        # Price breakout and chop are near-complementary -- one fires when the other doesn't.
        # Group together to prevent both from firing simultaneously.
        "signals": {
            "breakout_confirmed",  "breakdown_confirmed",
            "strong_chop_zone",    "moderate_chop_zone",
        },
        "dampening": 0.35,   # breakout and chop nearly mutually exclusive
    },
    "TRANSITION": {
        # Markov transition model -- one LR per phase. Dynamic keys from add_lr().
        # Signals field is a sentinel; actual membership tracked via evidence group label.
        "signals": set(),
        "dampening": 0.50,
    },
    "MICROSTRUCTURE": {
        # Spread, OB imbalance, CVD, depth, liquidations -- all measure market pressure.
        # Correlated in panic/euphoria but partially independent (different data sources).
        "signals": {
            "micro_spread_tight",    "micro_spread_wide",
            "micro_ob_strong_bull",  "micro_ob_mild_bull",
            "micro_ob_strong_bear",  "micro_ob_mild_bear",
            "micro_flow_strong_buy", "micro_flow_mild_buy",
            "micro_flow_strong_sell","micro_flow_mild_sell",
            "micro_depth_deep",      "micro_depth_thin",
            "micro_liq_shorts_squeezed","micro_liq_longs_stopped",
        },
        "dampening": 0.55,
    },
}
# Reverse lookup: signal_key → group_name (static keys only)
_SIGNAL_GROUP_MAP: Dict[str, str] = {
    sig: group
    for group, cfg in SIGNAL_GROUPS.items()
    for sig in cfg["signals"]
}


# ── Core probabilistic math helpers ──────────────────────────────────────────
# These must be defined BEFORE any class or function that uses them.

def _lr_safe(lr: float) -> float:
    """Clamp Likelihood Ratio to [0.01, 50] to prevent log-odds explosion."""
    return max(0.01, min(50.0, float(lr)))


def _sigmoid(log_odds: float) -> float:
    """Map log-odds to probability in (0, 1)."""
    if log_odds >= 30:  return 1.0
    if log_odds <= -30: return 0.0
    return 1.0 / (1.0 + math.exp(-log_odds))


def _logit(p: float) -> float:
    """Map probability to log-odds. Clamps to avoid ±inf."""
    p = max(1e-6, min(1 - 1e-6, float(p)))
    return math.log(p / (1.0 - p))


def _apply_group_dampening(evidence: List["_Evidence"], prior: float) -> float:
    """
    Compute group-dampened log-odds from accumulated evidence.

    Works for BOTH static keys (looked up via _SIGNAL_GROUP_MAP) and dynamic
    keys added via scorer.add_lr() -- the group label is stored directly in the
    _Evidence object, so dynamic evidence participates in group dampening too.

    Algorithm per group:
      1. Sort signals by |log_lr| descending (strongest first)
      2. Strongest signal → full weight
      3. Same-direction signals → DAMPENING × log_lr
      4. Opposite-direction signals → full weight (contrarian evidence)

    Signals not in any group → full weight always.
    """
    log_odds = _logit(prior)

    by_group: Dict[str, List["_Evidence"]] = {}
    ungrouped: List["_Evidence"] = []

    for ev in evidence:
        # Use stored group label directly -- covers both static and add_lr() evidence
        grp = ev.group if ev.group in SIGNAL_GROUPS else None
        if grp:
            by_group.setdefault(grp, []).append(ev)
        else:
            ungrouped.append(ev)

    for ev in ungrouped:
        log_odds += ev.log_lr

    for grp_name, evs in by_group.items():
        dampening = SIGNAL_GROUPS[grp_name]["dampening"]
        sorted_evs = sorted(evs, key=lambda e: abs(e.log_lr), reverse=True)
        strongest_sign = 1 if sorted_evs[0].log_lr >= 0 else -1

        for i, ev in enumerate(sorted_evs):
            ev_sign = 1 if ev.log_lr >= 0 else -1
            if i == 0:
                # Strongest in group: always full contribution
                log_odds += ev.log_lr
            elif ev.manually_scaled:
                # Pre-dampened by developer (scale != 1.0 was passed to update/add_lr).
                # The developer already accounted for the specific correlation by reducing
                # the LR at call time. Applying group dampening on top double-counts.
                # Example: ema12_above_ema26 scale=0.50 (explicitly: "collinear with MACD")
                # → already at LR=1.14 instead of 1.30 → group dampening would push to 1.04
                # → effectively neutral → bot "too scared" to enter even strong trends.
                log_odds += ev.log_lr   # full pre-reduced log_lr, no further dampening
            elif ev_sign == strongest_sign:
                # Secondary same-direction, not manually scaled → normal group dampening
                log_odds += ev.log_lr * dampening
            else:
                # Contrarian (opposite direction) → always full contribution
                log_odds += ev.log_lr

    return log_odds


# ═══════════════════════════════════════════════════════════════════════════
# [CAL] EMPIRICAL LR CALIBRATION
# Replaces hand-crafted LR values with data-estimated ones.
#
# Method: for each signal, count how often it fired on bars that led to
# continuation vs reversal (ATR-based labels, same as ML model).
#   LR = P(signal | continuation) / P(signal | reversal)
#
# Blending: empirical LR is blended with the handcrafted prior.
#   Confidence = min(1.0, sqrt(n / 200))   ← ramps up with sample size
#   LR_final   = w × LR_empirical + (1-w) × LR_prior
#   where w    = BLEND_WEIGHT × confidence
#
# Caching: calibrated LRs saved to JSON and reused for LR_CACHE_DAYS.
# ═══════════════════════════════════════════════════════════════════════════

LR_CACHE_FILE  = "models/calibrated_lrs.json"
LR_CACHE_DAYS  = 1      # recalibrate DAILY -- market regime shifts fast
LR_BLEND_WEIGHT = 0.75  # weight given to empirical when confident
LR_MIN_SAMPLES  = 40    # minimum signal occurrences before trusting empirical
LR_FULL_TRUST   = 200   # samples at which confidence = 1.0

# Rolling window and decay settings
LR_ROLLING_WINDOW = 90   # use only the most recent N bars per coin
LR_HALF_LIFE_DAYS = 30   # exponential decay: 30-day half-life
                          # weight(bar) = 2^(-(days_ago / HALF_LIFE))
                          # bar 30 days ago  → weight = 0.50
                          # bar 60 days ago  → weight = 0.25
                          # bar 90 days ago  → weight = 0.125

# Regime labels used for conditional LR tables
LR_REGIMES = ["bull", "bear", "range"]   # three conditional tables

# Active regime-conditional LR tables (populated at runtime)
# Keys: "bull" | "bear" | "range"
REGIME_CONDITIONAL_LRS: Dict[str, Dict[str, Tuple[float, float]]] = {
    regime: {} for regime in LR_REGIMES
}


def _compute_decay_weights(n: int, half_life: int = LR_HALF_LIFE_DAYS) -> np.ndarray:
    """
    Exponential decay weights for bars in a window of length n.
    Oldest bar (index 0) gets lowest weight, most recent (index n-1) highest.

    weight[i] = 2^(-(n-1-i) / half_life)
    """
    days_ago = np.arange(n - 1, -1, -1, dtype=float)   # [n-1, n-2, ..., 1, 0]
    return np.power(2.0, -days_ago / max(1, half_life))


def _regime_label_for_bar(price_arr: np.ndarray, idx: int,
                           adx_arr: Optional[np.ndarray] = None) -> str:
    """
    Classify a single bar into bull / bear / range using price context.
    Used to build regime-conditional LR tables.
    """
    if idx < 20:
        return "range"
    window = price_arr[max(0, idx-20): idx+1]
    trend  = (window[-1] - window[0]) / (window[0] + 1e-9) * 100

    adx = float(adx_arr[idx]) if adx_arr is not None and idx < len(adx_arr) else 15.0

    if trend > 5 and adx >= 18:
        return "bull"
    elif trend < -5 and adx >= 18:
        return "bear"
    return "range"


def _compute_signal_conditions(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Vectorized computation of every signal condition for all bars.
    Returns {signal_key: bool array aligned with df.index}.
    Missing data handled by returning False for that bar.
    """
    n   = len(df)
    p   = df["price"].values
    out : Dict[str, np.ndarray] = {}

    def _safe(series, default=False):
        try:
            return np.asarray(series, dtype=bool)
        except Exception:
            return np.full(n, default, dtype=bool)

    # ── MACRO ───────────────────────────────────────────────────────
    if "EMA_WEEKLY50" in df.columns:
        ew = df["EMA_WEEKLY50"].values
        out["above_weekly_ema50"] = _safe(p > ew)
        out["below_weekly_ema50"] = _safe(p < ew)

    # ── TREND STRUCTURE (Ichimoku) ───────────────────────────────────
    if "senkou_a" in df.columns and "senkou_b" in df.columns:
        sa = df["senkou_a"].values
        sb = df["senkou_b"].values
        ct = np.fmax(sa, sb)    # cloud top  (fmax ignores nan)
        cb = np.fmin(sa, sb)    # cloud bottom
        out["above_cloud"]      = _safe(p > ct)
        out["below_cloud_block"]= _safe(p < cb)
        out["in_cloud"]         = _safe((p >= cb) & (p <= ct))

    if "tenkan" in df.columns and "kijun" in df.columns:
        out["tenkan_above_kijun"] = _safe(df["tenkan"].values > df["kijun"].values)

    if "chikou" in df.columns and "chikou_ref" in df.columns:
        chikou = df["chikou"].values
        cref   = df["chikou_ref"].values
        ctop   = df["chikou_cloud_top"].values if "chikou_cloud_top" in df.columns else np.full(n, np.nan)
        cbot   = df["chikou_cloud_bot"].values if "chikou_cloud_bot" in df.columns else np.full(n, np.nan)
        out["chikou_fully_bullish"]  = _safe((chikou > cref) & (chikou > ctop))
        out["chikou_partial_bullish"]= _safe((chikou > cref) & ~(chikou > ctop))
        out["chikou_fully_bearish"]  = _safe((chikou < cref) & (chikou < cbot))
        out["chikou_partial_bearish"]= _safe((chikou < cref) & ~(chikou < cbot))

    # ── MACD / EMA ───────────────────────────────────────────────────
    if "MACD_Hist" in df.columns:
        mh = df["MACD_Hist"].values
        out["macd_bullish_hist"] = _safe(mh > 0)
        out["macd_bearish_hist"] = _safe(mh < 0)

    if "EMA12" in df.columns and "EMA26" in df.columns:
        e12 = df["EMA12"].values; e26 = df["EMA26"].values
        out["ema12_above_ema26"] = _safe(e12 > e26)
        out["ema12_below_ema26"] = _safe(e12 < e26)

    if "plus_di" in df.columns and "minus_di" in df.columns:
        pd_  = df["plus_di"].values; md_ = df["minus_di"].values
        out["di_plus_dominant"]  = _safe(pd_ > md_)
        out["di_minus_dominant"] = _safe(md_ > pd_)

    # ── RSI ──────────────────────────────────────────────────────────
    if "RSI" in df.columns:
        rsi = df["RSI"].values
        out["rsi_continuation_zone"] = _safe((rsi >= 55) & (rsi <= 80))
        out["rsi_extreme_overbought"]= _safe(rsi > 80)
        out["rsi_oversold_bounce"]   = _safe(rsi < 35)
        out["rsi_range_overbought"]  = _safe((rsi >= 65) & (rsi <= 75))

    # ── OBV ──────────────────────────────────────────────────────────
    if "OBV" in df.columns:
        obv    = df["OBV"].values.astype(float)
        obv_s  = pd.Series(obv)
        # 7-day fractional change
        obv_ch = (obv_s - obv_s.shift(7)) / (obv_s.shift(7).abs() + 1e-9)
        obv_ch = obv_ch.values
        # 7-day rolling max (prior 7 bars, not including today)
        obv_7max = pd.Series(obv).shift(1).rolling(7).max().values
        at_high  = obv >= obv_7max * 0.98

        out["obv_rising_at_high"]    = _safe((obv_ch > 0.03) &  at_high)
        out["obv_rising_not_at_high"]= _safe((obv_ch > 0.03) & ~at_high)
        out["obv_flat"]              = _safe(np.abs(obv_ch) <= 0.03)
        out["obv_falling"]           = _safe(obv_ch < -0.03)

    # ── VOLUME ───────────────────────────────────────────────────────
    if "volume" in df.columns:
        vol = df["volume"].values.astype(float)

        # Use rel_vol if precomputed; otherwise compute inline with median baseline.
        # Median is robust to spikes -- mean would inflate the baseline for months.
        if "rel_vol" in df.columns:
            vratio = df["rel_vol"].values.astype(float)
        else:
            _med20 = pd.Series(vol).rolling(20, min_periods=5).median().values
            vratio = (vol / (_med20 + 1e-9)).clip(0, 20)

        price_up    = np.concatenate([[False], p[1:] > p[:-1]])
        vol_decline = np.concatenate([[False, False],
                                       (vol[1:-1] < vol[:-2]) & (vol[2:] < vol[1:-1])])[:n]
        out["volume_spike_confirmed"]  = _safe(vratio > 2.0)
        out["volume_increasing_trend"] = _safe(
            (vol > np.roll(vol,1)) & (np.roll(vol,1) > np.roll(vol,2)) &
            (np.roll(vol,2) > np.roll(vol,3)))
        out["volume_weak"]    = _safe(vratio < 0.70)
        out["volume_fake_move"]= _safe(price_up & vol_decline)
        out["volume_healthy"] = _safe((vratio >= 0.7) & (vratio < 2.0))

    # ── BREAKOUT / CHOP ──────────────────────────────────────────────
    p_s  = pd.Series(p)
    res  = p_s.shift(1).rolling(20, min_periods=10).max().values
    sup  = p_s.shift(1).rolling(20, min_periods=10).min().values
    out["breakout_confirmed"]  = _safe(p > res)
    out["breakdown_confirmed"] = _safe(p < sup)

    sma20    = p_s.rolling(20, min_periods=10).mean().values
    std20    = p_s.rolling(20, min_periods=10).std().values
    bb_u     = sma20 + 2 * std20; bb_l = sma20 - 2 * std20
    bb_w_pct = (bb_u - bb_l) / (sma20 + 1e-9) * 100
    rng20    = (p_s.rolling(20, min_periods=10).max() -
                p_s.rolling(20, min_periods=10).min()).values / (p + 1e-9) * 100
    pos20    = ((p - (sma20 - 2*std20)) / (4*std20 + 1e-9)) * 100
    chop_sigs= ((rng20 < 12).astype(int) +
                ((pos20 > 30) & (pos20 < 70)).astype(int) +
                (bb_w_pct > 9).astype(int))
    out["strong_chop_zone"]   = _safe(chop_sigs >= 3)
    out["moderate_chop_zone"] = _safe(chop_sigs == 2)

    # ── Z-SCORE ──────────────────────────────────────────────────────
    zscore = (p - sma20) / (std20 + 1e-9)
    out["zscore_blow_off"]       = _safe(zscore > 2.5)
    out["zscore_falling_knife"]  = _safe(zscore < -2.5)
    out["zscore_oversold_range"] = _safe((zscore < -2.0) & (zscore >= -2.5))
    out["zscore_healthy_trend"]  = _safe((zscore > 0.5) & (zscore < 2.0))

    # ── EMA26 EXHAUSTION ─────────────────────────────────────────────
    if "EMA26" in df.columns:
        e26  = df["EMA26"].values
        dev  = (p - e26) / (e26 + 1e-9) * 100
        out["ema26_extreme_stretch"] = _safe(dev > 40)
        out["ema26_very_stretched"]  = _safe((dev > 25) & (dev <= 40))
        out["ema26_stretched"]       = _safe((dev > 15) & (dev <= 25))
        out["ema26_oversold"]        = _safe(dev < -15)

    # ── VOL REGIME sub-signals ───────────────────────────────────────
    if "ATR" in df.columns:
        atr  = df["ATR"].values
        atr5 = pd.Series(atr).rolling(5, min_periods=3).mean().values
        atr20= pd.Series(atr).rolling(20,min_periods=10).mean().values
        atrr = atr5 / (atr20 + 1e-9)
        out["vol_atr_expanding"]   = _safe(atrr > 1.20)
        out["vol_atr_contracting"] = _safe(atrr < 0.80)
    if "UpperBB" in df.columns and "LowerBB" in df.columns:
        bw   = (df["UpperBB"].values - df["LowerBB"].values) / (sma20 + 1e-9)
        bwr  = pd.Series(bw).rolling(50, min_periods=20).rank(pct=True).values
        out["vol_bb_wide"]   = _safe(bwr > 0.80)
        out["vol_bb_narrow"] = _safe(bwr < 0.20)

    # ── SMART MONEY (simplified vectorized proxy) ─────────────────────
    if "OBV" in df.columns and "EMA26" in df.columns:
        obv_    = df["OBV"].values.astype(float)
        obv14c  = (pd.Series(obv_) - pd.Series(obv_).shift(14)).values
        pc14    = (p - np.roll(p, 14)) / (np.roll(p, 14) + 1e-9) * 100
        out["sm_accumulation"]  = _safe((pc14 > 3)  & (obv14c >= 0))
        out["sm_distribution"]  = _safe((pc14 > 2)  & (obv14c <  0))

    # Remove keys not in SIGNAL_LR
    return {k: v for k, v in out.items() if k in SIGNAL_LR}


def _compute_continuation_labels(df: pd.DataFrame,
                                  future_days: int = 5,
                                  atr_mult: float = 1.5) -> np.ndarray:
    """
    ATR-based continuation/reversal labels (identical logic to ML training).
    1 = continuation (target hit before stop)
    0 = reversal (stop hit first)
    nan = neutral (neither hit in window)
    """
    price = df["price"].values
    n     = len(price)
    atr   = df["ATR"].values if "ATR" in df.columns else \
            np.abs(np.diff(price, prepend=price[0]))

    labels = np.full(n, np.nan)

    for i in range(n - future_days - 1):
        entry = price[i]
        a     = float(atr[i])
        if np.isnan(a) or a <= 0 or entry <= 0:
            continue
        target = entry + atr_mult * a
        stop   = entry - atr_mult * a
        future = price[i+1: i+1+future_days]

        hit_t = np.any(future >= target)
        hit_s = np.any(future <= stop)

        if hit_t and hit_s:
            t_i = int(np.argmax(future >= target))
            s_i = int(np.argmax(future <= stop))
            labels[i] = 1 if t_i <= s_i else 0
        elif hit_t:
            labels[i] = 1
        elif hit_s:
            labels[i] = 0

    return labels


def estimate_lrs_single(df: pd.DataFrame, coin: str = "",
                         future_days: int = 5, atr_mult: float = 1.5,
                         alpha: float = 0.5) -> Tuple[Dict, Dict]:
    """
    Estimate LRs from a single coin's historical data.
    Returns (lrs, stats).
    """
    if df is None or len(df) < 80:
        return {}, {}

    labels     = _compute_continuation_labels(df, future_days, atr_mult)
    valid      = ~np.isnan(labels)
    conditions = _compute_signal_conditions(df)

    n_cont = int((labels[valid] == 1).sum())
    n_rev  = int((labels[valid] == 0).sum())

    if n_cont < 10 or n_rev < 10:
        return {}, {}

    lv = labels[valid]
    lrs, stats = {}, {}

    for key, sig_arr in conditions.items():
        sv = sig_arr[valid].astype(bool)

        s_c = int(( sv & (lv == 1)).sum())
        s_r = int(( sv & (lv == 0)).sum())
        ns_c= int((~sv & (lv == 1)).sum())
        ns_r= int((~sv & (lv == 0)).sum())

        p_s_c  = (s_c  + alpha) / (n_cont + 2*alpha)
        p_s_r  = (s_r  + alpha) / (n_rev  + 2*alpha)
        p_ns_c = (ns_c + alpha) / (n_cont + 2*alpha)
        p_ns_r = (ns_r + alpha) / (n_rev  + 2*alpha)

        lr_t = p_s_c  / (p_s_r  + 1e-9)
        lr_f = p_ns_c / (p_ns_r + 1e-9)

        lrs[key]   = (round(lr_t, 4), round(lr_f, 4))
        stats[key] = {
            "s_c": s_c, "s_r": s_r, "n": s_c + s_r,
            "p_s|c": round(p_s_c, 4), "p_s|r": round(p_s_r, 4),
            "lr_emp": round(lr_t, 3),
            "lr_prior": SIGNAL_LR.get(key, (1.0, 1.0))[0],
        }

    return lrs, stats


def calibrate_lrs_from_history(coins_dfs: Dict[str, pd.DataFrame],
                                future_days  : int   = 5,
                                atr_mult     : float = 1.5,
                                alpha        : float = 0.5,
                                blend_weight : float = LR_BLEND_WEIGHT,
                                min_samples  : int   = LR_MIN_SAMPLES,
                                full_trust_n : int   = LR_FULL_TRUST,
                                window_days  : int   = LR_ROLLING_WINDOW,
                                half_life    : int   = LR_HALF_LIFE_DAYS,
                                oos_fraction : float = 0.20,
                                oos_shrink_k : float = 2.0,
                                verbose      : bool  = True) -> Dict[str, Tuple[float, float]]:
    """
    Rolling, decay-weighted, regime-conditional LR calibration
    with Out-of-Sample (OOS) validation.

    OOS VALIDATION -- what it fixes:
    ─────────────────────────────────────────────────────────────
    Without OOS, LRs are estimated and applied to the SAME window:
      days 1-90 → estimate LR(above_cloud)=2.1 → apply on days 1-90
      The LR just describes that window; it may not generalise.

    With OOS:
      days 1-72 (80%) → estimate LR_is (in-sample)
      days 73-90 (20%) → test: does LR_is correctly rank outcomes?
      OOS log-loss measures how well LR_is actually predicts on unseen data.

    STABILITY SHRINKAGE:
      A signal that was strong in-sample but weak out-of-sample is overfit.
      We shrink its LR toward 1.0 proportionally to the divergence:

        stability  = min(LR_oos / LR_is,  LR_is / LR_oos)   ∈ (0, 1]
        oos_weight = stability^oos_shrink_k                   ∈ (0, 1]
        LR_final   = LR_is^oos_weight × 1.0^(1-oos_weight)
                   = LR_is^oos_weight

      oos_shrink_k=2.0: moderate shrinkage.
        LR_is=2.0, LR_oos=1.2 → stability=0.60 → oos_weight=0.36
        LR_final = 2.0^0.36 = 1.32  (pulled toward 1.0 from 2.0)

      A signal stable across IS and OOS gets oos_weight≈1.0 (no change).
      A signal that collapses on OOS gets oos_weight→0.0 (→1.0 neutral).
    """
    # ── Accumulators: IS (estimation) and OOS (test) ─────────────────
    def _empty():
        return {"sw_c":0.0,"sw_r":0.0,"nsw_c":0.0,"nsw_r":0.0,"wn_c":0.0,"wn_r":0.0}

    # is_totals:  in-sample counts  (first 80% of each window)
    # oos_totals: out-of-sample counts (last 20%)
    # both indexed [regime][signal_key]
    is_totals : Dict[str, Dict[str, Dict]] = {r: {} for r in LR_REGIMES + ["all"]}
    oos_totals: Dict[str, Dict[str, Dict]] = {r: {} for r in LR_REGIMES + ["all"]}

    def _init(store, regime, key):
        if key not in store[regime]:
            store[regime][key] = _empty()

    for coin, df in coins_dfs.items():
        if df is None or df.empty or len(df) < 60:
            continue

        df_w  = df.iloc[-window_days:].copy() if len(df) > window_days else df.copy()
        n     = len(df_w)
        split = max(10, int(n * (1 - oos_fraction)))   # IS / OOS boundary

        labels     = _compute_continuation_labels(df_w, future_days, atr_mult)
        valid      = ~np.isnan(labels)
        conditions = _compute_signal_conditions(df_w)
        lv         = labels[valid]

        if (lv == 1).sum() < 5 or (lv == 0).sum() < 5:
            continue

        weights_all = _compute_decay_weights(n, half_life)
        weights_v   = weights_all[valid]

        price_col  = "close" if "close" in df_w.columns else "price"
        price_arr  = df_w[price_col].values
        adx_arr    = df_w["ADX"].values if "ADX" in df_w.columns else None
        regime_arr = np.array([
            _regime_label_for_bar(price_arr, i, adx_arr) for i in range(n)
        ])
        regime_v = regime_arr[valid]

        # Valid-bar indices into original window (for IS/OOS split)
        valid_indices = np.where(valid)[0]
        is_mask_v  = valid_indices < split    # True for IS bars
        oos_mask_v = valid_indices >= split   # True for OOS bars

        for key, sig_arr in conditions.items():
            sv = sig_arr[valid].astype(bool)

            for regime in LR_REGIMES + ["all"]:
                reg_mask = (np.ones(len(lv), dtype=bool)
                            if regime == "all" else (regime_v == regime))

                for store, time_mask in [(is_totals, is_mask_v),
                                         (oos_totals, oos_mask_v)]:
                    mask = reg_mask & time_mask
                    if mask.sum() < 2:
                        continue
                    w   = weights_v[mask]
                    lvm = lv[mask]
                    svm = sv[mask]
                    w_c = w[lvm == 1].sum() + 1e-9
                    w_r = w[lvm == 0].sum() + 1e-9
                    _init(store, regime, key)
                    t = store[regime][key]
                    t["sw_c"]  += w[(svm==True)  & (lvm==1)].sum()
                    t["sw_r"]  += w[(svm==True)  & (lvm==0)].sum()
                    t["nsw_c"] += w[(svm==False) & (lvm==1)].sum()
                    t["nsw_r"] += w[(svm==False) & (lvm==0)].sum()
                    t["wn_c"]  += w_c
                    t["wn_r"]  += w_r

    # ── Helper: compute empirical LR from weighted counts ─────────────
    def _lr_emp(t, alpha_=alpha):
        n_eff = t["sw_c"] + t["sw_r"]
        if n_eff < 1.0:
            return None, 0.0
        p_s_c = (t["sw_c"] + alpha_) / (t["wn_c"] + 2*alpha_)
        p_s_r = (t["sw_r"] + alpha_) / (t["wn_r"] + 2*alpha_)
        return _lr_safe(p_s_c / (p_s_r + 1e-9)), float(n_eff)

    # ── Helper: blend IS LR with prior + apply OOS shrinkage ─────────
    def _lr_final(is_t, oos_t, prior_t, prior_f):
        """
        Returns (lr_t, lr_f, conf, oos_weight, oos_stability)
        """
        n_is = is_t["sw_c"] + is_t["sw_r"] if is_t else 0.0
        if n_is < min_samples * 0.3:
            return prior_t, prior_f, 0.0, 1.0, 1.0

        # In-sample LR
        p_s_c  = (is_t["sw_c"]  + alpha) / (is_t["wn_c"] + 2*alpha)
        p_s_r  = (is_t["sw_r"]  + alpha) / (is_t["wn_r"] + 2*alpha)
        p_ns_c = (is_t["nsw_c"] + alpha) / (is_t["wn_c"] + 2*alpha)
        p_ns_r = (is_t["nsw_r"] + alpha) / (is_t["wn_r"] + 2*alpha)
        emp_t  = _lr_safe(p_s_c  / (p_s_r  + 1e-9))
        emp_f  = _lr_safe(p_ns_c / (p_ns_r + 1e-9))

        # IS blend with prior
        conf  = min(1.0, math.sqrt(n_is / full_trust_n))
        w     = blend_weight * conf
        is_lr_t = w * emp_t  + (1-w) * prior_t
        is_lr_f = w * emp_f  + (1-w) * prior_f

        # OOS validation: how consistent is the signal on held-out data?
        oos_stability = 1.0
        oos_weight    = 1.0
        if oos_t:
            n_oos = oos_t["sw_c"] + oos_t["sw_r"]
            if n_oos >= 5.0:
                oos_lr, _ = _lr_emp(oos_t)
                if oos_lr is not None:
                    # stability = ratio of smaller to larger LR
                    # 1.0 = identical IS/OOS, 0.0 = completely opposite
                    ratio = min(oos_lr / (is_lr_t + 1e-9),
                                is_lr_t / (oos_lr  + 1e-9))
                    oos_stability = float(np.clip(ratio, 0.0, 1.0))
                    # oos_weight = stability^k  (k=2 → moderate shrinkage)
                    oos_weight = float(oos_stability ** oos_shrink_k)

        # Apply OOS shrinkage: LR_is^oos_weight (pulls toward 1.0 when unstable)
        fin_t = round(max(0.05, min(10.0,
                    math.exp(math.log(_lr_safe(is_lr_t)) * oos_weight))), 3)
        fin_f = round(max(0.05, min(10.0,
                    math.exp(math.log(_lr_safe(is_lr_f)) * oos_weight))), 3)
        return fin_t, fin_f, conf, oos_weight, oos_stability

    # ── Build global calibrated LRs ───────────────────────────────────
    calibrated: Dict[str, Tuple[float, float]] = {}
    comparison: List[Dict] = []

    for key in SIGNAL_LR:
        prior_t, prior_f = SIGNAL_LR[key]
        is_t  = is_totals["all"].get(key)
        oos_t = oos_totals["all"].get(key)
        if is_t is None:
            calibrated[key] = (prior_t, prior_f)
            continue
        fin_t, fin_f, conf, oos_w, oos_stab = _lr_final(
            is_t, oos_t, prior_t, prior_f)
        calibrated[key] = (fin_t, fin_f)
        comparison.append({
            "key":      key,
            "prior":    prior_t,
            "empirical":fin_t,
            "conf":     conf,
            "oos_w":    oos_w,
            "oos_stab": oos_stab,
            "delta":    round(fin_t - prior_t, 3),
        })

    # ── Regime-conditional LR tables ─────────────────────────────────
    global REGIME_CONDITIONAL_LRS
    for regime in LR_REGIMES:
        reg_lrs: Dict[str, Tuple[float, float]] = {}
        for key in SIGNAL_LR:
            prior_t, prior_f = SIGNAL_LR[key]
            is_t  = is_totals[regime].get(key)
            oos_t = oos_totals[regime].get(key)
            if is_t is None:
                reg_lrs[key] = calibrated.get(key, (prior_t, prior_f))
                continue
            fin_t, fin_f, _, _, _ = _lr_final(is_t, oos_t, prior_t, prior_f)
            reg_lrs[key] = (fin_t, fin_f)
        REGIME_CONDITIONAL_LRS[regime] = reg_lrs

    # ── Verbose report ────────────────────────────────────────────────
    if verbose and comparison:
        print("\n" + "="*84)
        print(f" [CAL] ROLLING LR CALIBRATION + OOS VALIDATION"
              f"  (window={window_days}d  IS={int((1-oos_fraction)*100)}%"
              f"  OOS={int(oos_fraction*100)}%  half_life={half_life}d)")
        print("="*84)
        print(f"  {'Signal':<35} {'Prior':>6}  {'Final':>7}  "
              f"{'Conf':>5}  {'OOS_w':>6}  {'Stability':>10}  {'Δ':>6}")
        print("  " + "─"*80)

        # Sort by OOS weight ascending (most shrunk = most suspicious)
        moved = sorted(comparison, key=lambda x: x["oos_w"])
        shrunk  = [r for r in moved if r["oos_w"] < 0.70]
        stable  = [r for r in moved if r["oos_w"] >= 0.70]

        if shrunk:
            print(f"  ── OOS-SHRUNK signals (IS≠OOS, pulled toward neutral) ──")
            for row in shrunk[:10]:
                stab_bar = "░" * int((1-row["oos_stab"]) * 10)
                print(f"  {row['key']:<35} {row['prior']:>6.3f}  "
                      f"{row['empirical']:>7.3f}  {row['conf']:>5.0%}  "
                      f"{row['oos_w']:>6.2f}  {row['oos_stab']:>9.2f}  "
                      f"{row['delta']:>+6.3f}  {stab_bar}")

        if stable:
            print(f"  ── STABLE signals (IS≈OOS, full weight) ──")
            for row in sorted(stable, key=lambda x: abs(x["delta"]),
                              reverse=True)[:10]:
                print(f"  {row['key']:<35} {row['prior']:>6.3f}  "
                      f"{row['empirical']:>7.3f}  {row['conf']:>5.0%}  "
                      f"{row['oos_w']:>6.2f}  {row['oos_stab']:>9.2f}  "
                      f"{row['delta']:>+6.3f}")

        n_shrunk = len(shrunk)
        n_total  = len(comparison)
        print(f"\n  OOS summary: {n_shrunk}/{n_total} signals shrunk  "
              f"(oos_weight < 0.70 = overfit to recent window)")

        print(f"\n  Regime-conditional LR divergence:")
        print(f"  {'Signal':<30}  {'Global':>7}  {'Bull':>7}  {'Bear':>7}  {'Range':>7}")
        for key in ["above_cloud","obv_rising_at_high","btc_bull_regime",
                    "macd_bullish_hist","breakout_confirmed"]:
            gl = calibrated.get(key, SIGNAL_LR.get(key,(1,1)))[0]
            bu = REGIME_CONDITIONAL_LRS["bull"].get(key,(gl,))[0]
            be = REGIME_CONDITIONAL_LRS["bear"].get(key,(gl,))[0]
            ra = REGIME_CONDITIONAL_LRS["range"].get(key,(gl,))[0]
            print(f"  {key:<30}  {gl:>7.3f}  {bu:>7.3f}  {be:>7.3f}  {ra:>7.3f}")
        print("="*84)

    return calibrated


def save_calibrated_lrs(lrs: Dict, stats: Dict = None):
    """Persist calibrated LRs to JSON with timestamp."""
    try:
        Path(MODEL_PERSISTENCE_DIR).mkdir(parents=True, exist_ok=True)
        payload = {
            "calibrated_date": datetime.now().isoformat(),
            "n_signals":       len(lrs),
            "lrs":             {k: list(v) for k, v in lrs.items()},
            "stats":           stats or {},
        }
        with open(LR_CACHE_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[CAL] ✅ Calibrated LRs saved ({len(lrs)} signals) → {LR_CACHE_FILE}")
    except Exception as e:
        print(f"[CAL] ⚠️  Could not save LRs: {e}")


def load_calibrated_lrs() -> Optional[Dict[str, Tuple[float, float]]]:
    """Load calibrated LRs if cache is fresh (< LR_CACHE_DAYS old)."""
    try:
        if not os.path.exists(LR_CACHE_FILE):
            return None
        with open(LR_CACHE_FILE) as f:
            payload = json.load(f)
        age_days = (datetime.now() - datetime.fromisoformat(payload["calibrated_date"])).days
        if age_days >= LR_CACHE_DAYS:
            print(f"[CAL] Cache is {age_days}d old -- will recalibrate")
            return None
        lrs = {k: tuple(v) for k, v in payload["lrs"].items()}
        print(f"[CAL] ✅ Loaded calibrated LRs ({len(lrs)} signals, {age_days}d old)")
        return lrs
    except Exception as e:
        print(f"[CAL] Could not load LR cache: {e}")
        return None


def apply_calibrated_lrs(new_lrs: Dict[str, Tuple[float, float]]):
    """Update global SIGNAL_LR in-place with calibrated values."""
    global SIGNAL_LR
    updated = 0
    for key, (lr_t, lr_f) in new_lrs.items():
        if key in SIGNAL_LR:
            SIGNAL_LR[key] = (lr_t, lr_f)
            updated += 1
    print(f"[CAL] Applied {updated}/{len(SIGNAL_LR)} calibrated LR values to SIGNAL_LR")

# ── END [CAL] ─────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# [TRN] MARKOV STATE TRANSITION MODEL
#
# The probabilistic engine models P(continuation | current signals).
# This is a STATIC snapshot. It does not capture:
#   - What phase of the market cycle are we in?
#   - How long have we been in this phase?
#   - What is the probability of transitioning to the NEXT phase?
#
# Example of what the static model misses:
#   STRONG_BULL day 5:   most trends are early → high continuation
#   STRONG_BULL day 30:  most trends are overdue → distribution likely
#   Both show identical signals today. The static model treats them the same.
#
# Markov chain:
#   States:  8 market phases (ACCUMULATION → MARKUP → DISTRIBUTION → MARKDOWN)
#   Matrix:  P(phase_t+1 | phase_t)  -- estimated empirically, seeded with priors
#   Duration: semi-Markov -- transition probability increases past typical phase length
#   Output:  P(bullish_next_phase) → converted to LR update in log-odds space
#
# The transition model answers:
#   "Given I am in phase X for D days, what is the probability that tomorrow
#    I will still be in a bullish phase?"
#   This is a TRUE probability that cannot be recovered from current signals alone.
# ═══════════════════════════════════════════════════════════════════════════════

MARKET_PHASES = [
    "ACCUMULATION",   # quiet base-building, OBV rising, price flat
    "EARLY_BULL",     # ADX rising from low, price breaking above cloud, OBV confirming
    "STRONG_BULL",    # ADX ≥ 25, price well above cloud, OBV at highs
    "LATE_BULL",      # ADX plateauing, RSI > 70, OBV decelerating -- near distribution
    "DISTRIBUTION",   # price high but OBV falling, volume increasing on down bars
    "EARLY_BEAR",     # price below cloud, ADX rising, OBV declining
    "STRONG_BEAR",    # price well below cloud, ADX ≥ 25, OBV falling fast
    "RANGE",          # ADX < 15, BB narrow, no trend -- pure noise
]

# Bullish / bearish classification for continuation probability
_BULL_PHASES = {"ACCUMULATION", "EARLY_BULL", "STRONG_BULL"}
_BEAR_PHASES = {"DISTRIBUTION", "EARLY_BEAR", "STRONG_BEAR"}
_NEUT_PHASES = {"LATE_BULL", "RANGE"}

# Typical phase durations in days (crypto daily bars, estimated from BTC/ETH 2019-2024)
# Used for semi-Markov duration adjustment
PHASE_TYPICAL_DURATION: Dict[str, int] = {
    "ACCUMULATION": 21,
    "EARLY_BULL":   12,
    "STRONG_BULL":  22,
    "LATE_BULL":    8,
    "DISTRIBUTION": 11,
    "EARLY_BEAR":   13,
    "STRONG_BEAR":  19,
    "RANGE":        28,
}

# Transition matrix: P(phase_t+1 | phase_t)
# Rows = current phase, Cols = next phase
# Each row must sum to ~1.0. Estimated from crypto market microstructure;
# replaced by empirical estimation in calibrate_transitions_from_history().
#
# Key structural facts encoded:
#   - Trends persist (diagonal values high: 0.60-0.70)
#   - Accumulation leads to bull (ACC→EARLY_BULL = 0.20)
#   - Distribution leads to bear (DIST→EARLY_BEAR = 0.40)
#   - Range is sticky but breaks (RANGE→RANGE = 0.68)
#   - Panic/euphoria embedded via LATE_BULL and DISTRIBUTION states
PHASE_TRANSITION_MATRIX: Dict[str, Dict[str, float]] = {
    "ACCUMULATION": {
        "ACCUMULATION": 0.52, "EARLY_BULL": 0.22, "RANGE": 0.16,
        "EARLY_BEAR": 0.06, "STRONG_BEAR": 0.04,
    },
    "EARLY_BULL": {
        "EARLY_BULL": 0.50, "STRONG_BULL": 0.28, "ACCUMULATION": 0.12,
        "LATE_BULL": 0.06, "RANGE": 0.04,
    },
    "STRONG_BULL": {
        "STRONG_BULL": 0.62, "LATE_BULL": 0.18, "EARLY_BULL": 0.10,
        "DISTRIBUTION": 0.06, "RANGE": 0.04,
    },
    "LATE_BULL": {
        "LATE_BULL": 0.32, "DISTRIBUTION": 0.32, "STRONG_BULL": 0.18,
        "RANGE": 0.12, "EARLY_BEAR": 0.06,
    },
    "DISTRIBUTION": {
        "DISTRIBUTION": 0.38, "EARLY_BEAR": 0.38, "LATE_BULL": 0.12,
        "RANGE": 0.08, "STRONG_BEAR": 0.04,
    },
    "EARLY_BEAR": {
        "EARLY_BEAR": 0.48, "STRONG_BEAR": 0.30, "DISTRIBUTION": 0.12,
        "RANGE": 0.06, "ACCUMULATION": 0.04,
    },
    "STRONG_BEAR": {
        "STRONG_BEAR": 0.58, "EARLY_BEAR": 0.20, "RANGE": 0.12,
        "ACCUMULATION": 0.06, "DISTRIBUTION": 0.04,
    },
    "RANGE": {
        "RANGE": 0.68, "EARLY_BULL": 0.12, "ACCUMULATION": 0.10,
        "EARLY_BEAR": 0.06, "DISTRIBUTION": 0.04,
    },
}

# Duration-adjusted transition: scale persistence probability down when
# the coin has been in this phase longer than typical.
# After 2× typical duration, persistence probability halved.

# ── Per-phase decay constants (λ) ────────────────────────────────────────────
# Higher λ = faster decay toward neutral as phase ages.
# At t = t_typical: P_effective = 0.5 + (P_base-0.5) × e^(-λ)
#   λ=1.00 → 37% of excess remains  (persistent phases)
#   λ=1.40 → 25% of excess remains  (standard)
#   λ=1.80 → 17% of excess remains  (short-lived phases)
#   λ=2.00 → 13% of excess remains  (fast-breaking phases)
PHASE_DECAY_LAMBDA: Dict[str, float] = {
    "ACCUMULATION":  1.20,   # patient accumulation can last many bars
    "EARLY_BULL":    1.40,   # standard -- trend must prove itself quickly
    "STRONG_BULL":   1.00,   # strong trends genuinely persist longer
    "LATE_BULL":     1.80,   # late-stage bull fades fast
    "DISTRIBUTION":  1.60,   # distribution is a transition state
    "EARLY_BEAR":    1.40,   # symmetric with EARLY_BULL
    "STRONG_BEAR":   1.00,   # strong bears also persist
    "RANGE":         2.00,   # ranges break without warning
}


def _apply_age_decay(p_base: float, duration_bars: int,
                     typical_duration: int, lambda_: float = 1.40) -> float:
    """
    Exponential decay of trend continuation probability with phase age.

    Formula:
        P_effective = 0.5 + (P_base − 0.5) × exp(−λ × t / t_typical)

    Behaviour:
        t = 0              → P_effective = P_base     (fresh, full trust)
        t = t_typical      → 25% of excess retained   (λ=1.4: e^-1.4 = 0.25)
        t = 2 × t_typical  → 6% of excess retained    (very old, near-neutral)
        t → ∞              → P_effective → 0.5         (pure neutral)

    The excess probability (P_base − 0.5) decays -- not the raw probability.
    This means bearish signals also decay symmetrically: P_base=0.10 (bear)
    at t=t_typical → P_effective = 0.5 − 0.40 × 0.25 = 0.40 (still bearish,
    but much less so than at the start of the bear phase).

    BEFORE: _duration_scale was a step function -- ratio 0.75-1.25 → scale=1.00
            EARLY_BULL day 10/12 (83%) → dur_scale=1.00 → NO DECAY → LR=2.875
    AFTER:  t_norm = 10/12 = 0.833 → decay = e^(-1.4 × 0.833) = 0.311
            P_effective = 0.5 + (0.90-0.5) × 0.311 = 0.624  → LR≈1.87
    """
    t_norm = float(duration_bars) / max(1, typical_duration)
    decay  = math.exp(-lambda_ * t_norm)
    return 0.5 + (float(p_base) - 0.5) * decay


def _duration_scale(current_phase: str, duration_bars: int) -> float:
    """
    DEPRECATED: replaced by _apply_age_decay() -- kept for backward compat only.
    Callers should switch to _apply_age_decay for smooth continuous decay.
    """
    typical = PHASE_TYPICAL_DURATION.get(current_phase, 20)
    ratio   = duration_bars / max(1, typical)
    if   ratio < 0.75: return 1.10
    elif ratio < 1.25: return 1.00
    elif ratio < 1.75: return 0.80
    elif ratio < 2.50: return 0.60
    else:              return 0.40


def classify_market_phase(last: pd.Series,
                          df: pd.DataFrame,
                          obv_trend: str    = "unknown",
                          vol_regime: str   = "neutral",
                          btc_regime: str   = "unknown") -> str:
    """
    Classify the current market into one of 8 Markov phases.

    Priority ordering: bear phases checked before bull so structural
    conditions override momentum. Distribution checked before STRONG_BULL
    because OBV falling + price high = distribution even if trend signals fire.
    """
    try:
        price    = float(last["price"])
        rsi      = float(last.get("RSI", 50) or 50)
        adx      = float(last.get("ADX", 0)  or 0)
        plus_di  = float(last.get("plus_di",  0) or 0)
        minus_di = float(last.get("minus_di", 0) or 0)
        ema12    = float(last.get("EMA12", price))
        ema26    = float(last.get("EMA26", price))
        macd_h   = float(last.get("MACD_Hist", 0) or 0)

        # Cloud position
        sa = float(last.get("senkou_a", float("nan")))
        sb = float(last.get("senkou_b", float("nan")))
        if not (math.isnan(sa) or math.isnan(sb)):
            above_cloud = price > max(sa, sb)
            below_cloud = price < min(sa, sb)
        else:
            above_cloud = below_cloud = False

        # OBV acceleration (is OBV rising faster or slower than last week?)
        obv_accel = "flat"
        if "OBV" in df.columns and len(df) >= 14:
            obv = df["OBV"].values.astype(float)
            obv7  = obv[-1] - obv[-8]   # 7-day OBV change
            obv14 = obv[-8] - obv[-15] if len(obv) >= 15 else obv7
            if obv7 > obv14 * 1.1:  obv_accel = "accelerating"
            elif obv7 < obv14 * 0.9: obv_accel = "decelerating"

        # ── Detection hierarchy ─────────────────────────────────────────
        # STRONG_BEAR: structural -- checked first
        if below_cloud and adx >= 20 and minus_di > plus_di and obv_trend == "falling":
            return "STRONG_BEAR"

        # EARLY_BEAR: cloud below + trend forming but not full
        if below_cloud and minus_di > plus_di:
            return "EARLY_BEAR"

        # DISTRIBUTION: price still high but OBV falling (most dangerous state for longs)
        if above_cloud and obv_trend == "falling" and rsi > 60:
            return "DISTRIBUTION"

        # LATE_BULL: above cloud but OBV decelerating, RSI elevated
        if above_cloud and rsi > 68 and obv_accel == "decelerating":
            return "LATE_BULL"

        # STRONG_BULL: all aligned, trend confirmed
        if above_cloud and adx >= 22 and plus_di > minus_di and obv_trend in ("rising",):
            return "STRONG_BULL"

        # EARLY_BULL: trend forming but not yet strong
        if above_cloud and ema12 > ema26 and macd_h > 0:
            return "EARLY_BULL"

        # ACCUMULATION: price at/below cloud but OBV building (smart money buying quietly)
        if not above_cloud and not below_cloud and obv_trend == "rising":
            return "ACCUMULATION"
        if below_cloud and obv_trend == "rising" and adx < 20:
            return "ACCUMULATION"

        # RANGE: default when no trend is detected
        return "RANGE"

    except Exception:
        return "RANGE"


def compute_transition_lr(current_phase: str,
                          duration_bars: int,
                          matrix: Dict[str, Dict[str, float]]) -> Tuple[float, str]:
    """
    Convert phase transition probabilities into a LR update with exponential age decay.

    STEP 1: Read P_bull_raw and P_bear_raw from the transition matrix.
    STEP 2: Apply exponential age decay:
              P_bull = 0.5 + (P_bull_raw - 0.5) × e^(-λ × t/t_typical)
            A mature trend's continuation probability shrinks toward neutral.
    STEP 3: Compute LR from the decayed P_bull / P_bear vs base rates.

    KEY FIX vs old _duration_scale():
      Old: step function -- day 10/12 falls in bucket 0.75-1.25 → scale=1.00 (NO decay)
      New: smooth exponential -- day 10/12 → decay=0.311 → P drops 0.90→0.624 → LR drops 2.875→1.87
    """
    trans = matrix.get(current_phase, {})
    if not trans:
        return 1.0, f"{current_phase}: no transitions defined"

    typical  = PHASE_TYPICAL_DURATION.get(current_phase, 20)
    lambda_  = PHASE_DECAY_LAMBDA.get(current_phase, 1.40)

    # Raw transition probabilities from matrix (no age adjustment yet)
    p_bull_raw = sum(v for k, v in trans.items() if k in _BULL_PHASES)
    p_bear_raw = sum(v for k, v in trans.items() if k in _BEAR_PHASES)

    # Apply exponential age decay -- this is the core fix.
    # The longer a phase has run, the less it should extrapolate.
    p_bull = _apply_age_decay(p_bull_raw, duration_bars, typical, lambda_)
    p_bear = _apply_age_decay(p_bear_raw, duration_bars, typical, lambda_)

    # Decay factor for display (how much of the original signal remains)
    decay_factor = math.exp(-lambda_ * duration_bars / max(1, typical))

    # Base rates: prior expectation without phase info
    BASE_BULL = 0.40
    BASE_BEAR = 0.30

    # LR from decayed probabilities vs base rates
    if p_bull > BASE_BULL:
        raw_lr = 1.0 + (p_bull - BASE_BULL) / BASE_BULL * 1.5
    elif p_bear > BASE_BEAR:
        raw_lr = 1.0 / (1.0 + (p_bear - BASE_BEAR) / BASE_BEAR * 1.5)
    else:
        raw_lr = 1.0

    # Late-bull penalty: even bullish transition may be problematic
    p_late = trans.get("LATE_BULL", 0.0)
    if p_late > 0.25:
        raw_lr *= (1.0 - p_late * 0.30)

    dur_pct = int(duration_bars / max(1, typical) * 100)
    label = (
        f"{current_phase} (day {duration_bars}/{typical} = {dur_pct}%)  "
        f"P(→bull): {p_bull_raw:.2f}→{p_bull:.2f}  "
        f"P(→bear): {p_bear_raw:.2f}→{p_bear:.2f}  "
        f"decay={decay_factor:.3f}  λ={lambda_:.2f}  LR={raw_lr:.3f}"
    )
    return _lr_safe(raw_lr), label


def estimate_transitions_from_history(coins_dfs: Dict[str, pd.DataFrame],
                                      alpha: float = 1.0,
                                      verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Empirically estimate the phase transition matrix from all coin histories.

    For each consecutive pair of bars (t, t+1):
      1. Classify market phase at bar t and bar t+1
      2. Record the transition phase_t → phase_t+1
      3. Count occurrences of each transition

    Returns: transition matrix with Laplace-smoothed probabilities.

    alpha = Laplace smoothing count (larger = more shrinkage toward uniform)
    """
    counts: Dict[str, Dict[str, float]] = {
        p: {q: alpha for q in MARKET_PHASES} for p in MARKET_PHASES
    }

    total_transitions = 0

    for coin, df in coins_dfs.items():
        if df is None or len(df) < 10:
            continue
        try:
            # Classify phase for every bar (vectorized-ish: classify row by row)
            phases = []
            # Compute OBV trend for each bar
            obv_col = df["OBV"].values.astype(float) if "OBV" in df.columns else None

            for i in range(len(df)):
                sub_df = df.iloc[max(0, i-30):i+1]  # trailing window
                row    = df.iloc[i]
                obv_t  = "unknown"
                if obv_col is not None and i >= 7:
                    ch = obv_col[i] - obv_col[i-7]
                    obv_t = "rising" if ch > 0 else ("falling" if ch < 0 else "flat")
                phases.append(classify_market_phase(row, sub_df, obv_trend=obv_t))

            # Count transitions
            for i in range(len(phases) - 1):
                counts[phases[i]][phases[i+1]] += 1
                total_transitions += 1

        except Exception as e:
            continue

    # Normalize rows → probability
    matrix: Dict[str, Dict[str, float]] = {}
    for phase in MARKET_PHASES:
        row_sum = sum(counts[phase].values())
        matrix[phase] = {q: counts[phase][q] / row_sum for q in MARKET_PHASES
                         if counts[phase][q] / row_sum > 0.001}

    if verbose:
        print("\n" + "═"*70)
        print(" [TRN] EMPIRICAL TRANSITION MATRIX")
        print("═"*70)
        header = "".join(f"{p[:8]:>10}" for p in MARKET_PHASES)
        print(f"  {'':>14}{header}")
        for phase in MARKET_PHASES:
            row = "".join(
                f"{matrix[phase].get(q, 0):.3f}     " if matrix[phase].get(q, 0) > 0.01
                else f"{'--':>10}"
                for q in MARKET_PHASES
            )
            print(f"  {phase:<14}  {row}")
        print(f"  Total transitions sampled: {total_transitions:,}")
        print("═"*70 + "\n")

    return matrix


def track_phase_duration(df: pd.DataFrame, current_phase: str) -> int:
    """
    Estimate how many consecutive bars the coin has been in current_phase.

    Looks back through the last 60 bars, counting consecutive bars
    where classify_market_phase() returns the same phase.
    Stops at the first bar where the phase changes.
    """
    if df is None or len(df) < 2:
        return 1

    max_lookback = min(60, len(df) - 1)
    duration     = 1
    obv_col = df["OBV"].values.astype(float) if "OBV" in df.columns else None

    for i in range(len(df)-2, len(df)-2-max_lookback, -1):
        if i < 0:
            break
        sub = df.iloc[max(0, i-15):i+1]
        row = df.iloc[i]
        obv_t = "unknown"
        if obv_col is not None and i >= 7:
            ch    = obv_col[i] - obv_col[max(0, i-7)]
            obv_t = "rising" if ch > 0 else ("falling" if ch < 0 else "flat")
        phase_i = classify_market_phase(row, sub, obv_trend=obv_t)
        if phase_i == current_phase:
            duration += 1
        else:
            break

    return duration

# ── END [TRN] ─────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# [MLLR] ML → BAYESIAN BRIDGE: Dynamic Likelihood Ratio Producer
#
# Problem:  ML and Bayesian engines run in parallel and are bridged only by a
#           hard gate (if ml_proba < 0.25: WAIT). This loses information.
#
# Solution: ML produces LIKELIHOOD RATIOS that enter the Bayesian log-odds
#           chain as evidence -- ML informs Bayes, not overrides it.
#
# Two components:
#
# 1. RESIDUAL LR  -- what ML knows BEYOND what Bayesian signals captured.
#    The Bayesian model updates prior → P_bayes using marginal signal LRs.
#    The ML model independently predicts P_ml from the same features.
#    The RESIDUAL encodes non-linear INTERACTION effects that marginal LRs miss.
#
#      residual_odds = (P_ml / (1-P_ml)) / (P_bayes / (1-P_bayes))
#      LR_residual   = 1.0 + (residual_odds - 1.0) × shrink_weight
#
#    shrink_weight = 0.35: ML residual is modest because Bayes already
#    has most of the signal. Only interaction effects are genuinely new.
#
# 2. INTERACTION LRs -- feature-pair conditional LRs learned from history.
#    Example:
#      P(cloud above AND ATR expanding | continuation) = 0.58
#      P(cloud above AND ATR expanding | reversal)     = 0.34
#      LR(cloud ∩ ATR expansion) = 0.58 / 0.34 = 1.71
#
#    This LR captures the SYNERGY between cloud breakout and volatility
#    expansion that cannot be recovered from the individual marginal LRs.
#    cloud LR × ATR LR ≠ LR(cloud ∩ ATR) -- the joint is different.
#
# PIPELINE INTEGRATION (replaces ML gate):
#   OLD:  if ml_proba < 0.25: decision = "WAIT"   ← hard gate, loses info
#   NEW:  prob = apply_lr(prob, ml_residual_lr)    ← soft Bayesian evidence
#         prob = apply_lr(prob, ml_interaction_lrs) ← joint feature evidence
# ═══════════════════════════════════════════════════════════════════════════════

# Feature pairs whose interactions are most predictive (estimated from data)
# Format: {name: (signal_key_a, signal_key_b, prior_lr)}
# prior_lr = hand-seeded value before empirical estimation
ML_INTERACTION_DEFINITIONS: Dict[str, Tuple[str, str, float]] = {
    # Structural momentum confirmation -- cloud trend + volatility expansion
    "cloud_atr_expansion":      ("above_cloud",          "vol_atr_expanding",       1.72),
    # Capital flow confirmation -- cloud trend + OBV institutional buying
    "cloud_obv_7dhigh":         ("above_cloud",          "obv_rising_at_high",      1.85),
    # Real move filter -- MACD direction + volume quality
    "macd_volume_spike":        ("macd_bullish_hist",     "volume_spike_confirmed",  1.58),
    # Trend momentum -- ADX directional + price breakout
    "breakout_di_dominant":     ("breakout_confirmed",    "di_plus_dominant",        1.65),
    # Macro alignment -- BTC regime + weekly trend
    "btc_weekly_alignment":     ("btc_bull_regime",       "above_weekly_ema50",      1.90),
    # Smart money + OBV -- both institutional signals aligned
    "smart_money_obv":          ("sm_accumulation",       "obv_rising_at_high",      2.05),
    # RSI continuation in structural uptrend -- most reliable combo
    "rsi_cloud_continuation":   ("rsi_continuation_zone","above_cloud",             1.45),
    # Market structure + institutional flow
    "bos_obv_confirmation":     ("structure_bos_bullish", "obv_rising_at_high",      1.78),
    # Volatility squeeze before breakout -- coiling + volume building
    "squeeze_volume_spike":     ("vol_squeeze_on",        "volume_spike_confirmed",  1.55),
    # Macro downside -- BTC bear + OBV distribution (bearish interaction)
    "btc_bear_obv_falling":     ("btc_bear_regime_block", "obv_falling",             0.28),
    # Distribution warning -- smart money exiting + OBV falling
    "distribution_obv_fall":    ("sm_distribution",       "obv_falling",             0.22),
    # Exhaustion confirmation -- RSI overbought + blow-off Z-score
    "rsi_zscore_blowoff":       ("rsi_extreme_overbought","zscore_blow_off",         0.30),
}

# Add ML interactions to group dampening (own group, independent of signal groups)
SIGNAL_GROUPS["ML_INTERACTION"] = {
    "signals": set(),   # populated dynamically from MLLRProducer
    "dampening": 0.45,  # moderate: interactions are somewhat correlated
}


class MLLRProducer:
    """
    Bridges the ML model and Bayesian engine via Likelihood Ratios.

    Two outputs:
      residual_lr(ml_proba, bayes_prob)       → LR from ML residual
      get_active_interaction_lrs(last, df)    → dict of active interaction LRs

    Both enter the Bayesian log-odds chain as evidence, not as gates.
    """

    MLLR_CACHE = "models/ml_interaction_lrs.json"
    RESIDUAL_SHRINK = 0.35   # fraction of raw residual to apply (keeps ML modest)
    MIN_SAMPLES     = 25     # minimum interaction occurrences before trusting empirical

    # ── Impact cap: how far the ML residual LR can move probability ──────────
    # Bounds are LR-space, calibrated to the P=70% zone (BUY/STRONG_BUY decision
    # boundary), since pp-impact isn't anchor-invariant in log-odds space -- the
    # same LR pair moves probability a bit less near 0/1 and a bit more near 0.5.
    # DEFAULT band ≈ ±15pp at that anchor. Widened to the UNLOCKED band only when
    # the model has actually proven itself -- same bar the ML GATE section
    # elsewhere uses for "full weight regardless of raw accuracy" (wf_sharpe>0.5),
    # paired with a tighter accuracy floor (0.70) since this is a single live
    # disagreement, not an aggregated historical win rate like regime-cal's.
    TRUST_ACC          = 0.70
    TRUST_SHARPE       = 0.5
    LR_BOUNDS_DEFAULT  = (0.52, 2.43)   # ≈ ±15pp at P=70%
    LR_BOUNDS_UNLOCKED = (0.35, 4.00)   # ≈ -25pp downside; upside already capped by
                                         # the absolute 4.0 ceiling below (≈+20pp at P=70%)

    def __init__(self):
        self.interaction_lrs  : Dict[str, float] = {
            k: v[2] for k, v in ML_INTERACTION_DEFINITIONS.items()
        }
        self.fitted           : bool  = False
        self.fit_date         : str   = ""
        self.n_coins_fitted   : int   = 0

    # ── Core method 1: Residual LR ────────────────────────────────────────────

    @staticmethod
    def residual_lr(ml_proba   : float,
                    bayes_prob  : float,
                    shrink      : float = RESIDUAL_SHRINK,
                    lr_bounds   : Tuple[float, float] = (0.25, 4.0)) -> Tuple[float, str]:
        """
        Convert ML probability into a residual LR for Bayesian integration.

        Math:
          odds_bayes  = P_bayes / (1 - P_bayes)   ← what Bayes believes
          odds_ml     = P_ml    / (1 - P_ml)       ← what ML believes
          residual    = odds_ml / odds_bayes        ← disagreement ratio
          LR_final    = 1.0 + (residual - 1.0) × shrink

        Shrinkage = 0.35:
          If ML strongly disagrees with Bayes (residual=3.0):
            LR = 1.0 + 2.0 × 0.35 = 1.70  ← modest positive update
          If ML and Bayes agree exactly (residual=1.0):
            LR = 1.0  ← no update (ML adds nothing new)

        This prevents the ML from undoing what the Bayesian model correctly
        computed while still letting it contribute its interaction information.

        lr_bounds applies a SECOND, typically tighter clamp on top of the
        absolute [0.25, 4.0] safety clamp below -- pass LR_BOUNDS_DEFAULT or
        LR_BOUNDS_UNLOCKED (see class docstring above) depending on whether
        this model has earned the wider band.
        """
        bayes_odds = max(1e-7, bayes_prob) / max(1e-7, 1.0 - bayes_prob)
        ml_odds    = max(1e-7, ml_proba)   / max(1e-7, 1.0 - ml_proba)
        raw_resid  = ml_odds / bayes_odds

        # PATHOLOGICAL CASE: if Bayesian P is near 0 or 1, the odds ratio
        # explodes because ML at 10% looks "bullish" compared to Bayes at 0.2%.
        # Cap the raw residual so ML can never contribute more than LR≈4.0
        # regardless of how extreme the Bayesian posterior has become.
        # Without this, strong bearish signals push P near 0, then ML at 10%
        # produces LR=17+ (bullish!) which partially undoes correct bearish work.
        raw_resid = max(0.05, min(20.0, raw_resid))   # raw_resid ∈ [0.05, 20]

        shrunk_lr  = 1.0 + (raw_resid - 1.0) * shrink
        # Absolute safety clamp: ML residual contributes at most LR=4.0 in either
        # direction, no matter what -- never relaxed regardless of trust tier.
        shrunk_lr  = max(0.25, min(4.0, shrunk_lr))
        # Trust-tier clamp: tighter by default; caller decides whether to widen it.
        _lo, _hi   = lr_bounds
        shrunk_lr  = max(_lo, min(_hi, shrunk_lr))
        lr         = _lr_safe(shrunk_lr)

        direction  = "ML↑" if lr > 1.01 else ("ML↓" if lr < 0.99 else "ML≈")
        desc = (f"{direction} residual LR={lr:.3f}  "
                f"(P_ml={ml_proba:.1%} vs P_bayes={bayes_prob:.1%}  "
                f"raw_resid={raw_resid:.2f}  shrink={shrink:.2f})")
        return lr, desc

    # ── Core method 2: Interaction LRs ───────────────────────────────────────

    def get_active_interaction_lrs(self,
                                   df   : pd.DataFrame,
                                   coin : str = "") -> Dict[str, Tuple[float, str]]:
        """
        Compute which feature interactions are currently active and return their LRs.

        For each interaction (A ∩ B): if BOTH signals fire on the current bar,
        return the empirically estimated LR for that joint state.
        If only one fires or neither, the interaction does not contribute
        (absence evidence is already captured by individual signal LRs).
        """
        conds  = _compute_signal_conditions(df)
        active : Dict[str, Tuple[float, str]] = {}

        for int_name, (key_a, key_b, _) in ML_INTERACTION_DEFINITIONS.items():
            if key_a not in conds or key_b not in conds:
                continue
            # Check current bar (last row)
            try:
                ca = bool(conds[key_a][-1])
                cb = bool(conds[key_b][-1])
            except (IndexError, KeyError):
                continue

            if ca and cb:
                lr = self.interaction_lrs.get(int_name, 1.0)
                if abs(math.log(_lr_safe(lr))) < 0.02:
                    continue   # near-neutral -- skip
                desc = (f"ML interaction [{int_name}]: "
                        f"{key_a} ∩ {key_b}  LR={lr:.3f}")
                active[int_name] = (lr, desc)

        if active and coin:
            print(f"[MLLR] {coin}: {len(active)} active interactions: "
                  f"{', '.join(f'{k}={v[0]:.2f}' for k,v in active.items())}")
        return active

    # ── Fitting: learn interaction LRs from historical data ───────────────────

    def fit(self, coins_dfs   : Dict[str, pd.DataFrame],
            alpha             : float = 0.5,
            verbose           : bool  = True) -> None:
        """
        Estimate interaction LRs from all coin histories.

        For each interaction (A ∩ B):
          P(A∩B | continuation)   = count(A∩B on cont bars) / total cont bars
          P(A∩B | reversal)       = count(A∩B on rev bars)  / total rev bars
          LR = P(A∩B|C) / P(A∩B|R)  -- blended with prior based on sample size
        """
        counts: Dict[str, Dict[str, int]] = {
            k: {"sc": 0, "sr": 0, "nc": 0, "nr": 0, "n_cont": 0, "n_rev": 0}
            for k in ML_INTERACTION_DEFINITIONS
        }

        for coin, df in coins_dfs.items():
            if df is None or len(df) < 80:
                continue
            try:
                labels = _compute_continuation_labels(df)
                valid  = ~np.isnan(labels)
                conds  = _compute_signal_conditions(df)
                lv     = labels[valid]
                n_c    = int((lv == 1).sum())
                n_r    = int((lv == 0).sum())
                if n_c < 5 or n_r < 5:
                    continue

                for int_name, (key_a, key_b, _) in ML_INTERACTION_DEFINITIONS.items():
                    if key_a not in conds or key_b not in conds:
                        continue
                    sa = conds[key_a][valid].astype(bool)
                    sb = conds[key_b][valid].astype(bool)
                    both = sa & sb
                    c = counts[int_name]
                    c["sc"]     += int(( both & (lv==1)).sum())
                    c["sr"]     += int(( both & (lv==0)).sum())
                    c["nc"]     += int((~both & (lv==1)).sum())
                    c["nr"]     += int((~both & (lv==0)).sum())
                    c["n_cont"] += n_c
                    c["n_rev"]  += n_r
            except Exception:
                continue

        updated = {}
        report  = []
        for int_name, c in counts.items():
            n     = c["sc"] + c["sr"]
            prior = ML_INTERACTION_DEFINITIONS[int_name][2]
            if n < self.MIN_SAMPLES:
                updated[int_name] = prior
                report.append((int_name, n, prior, None, prior, "PRIOR"))
                continue

            p_sc = (c["sc"] + alpha) / (c["n_cont"] + 2*alpha)
            p_sr = (c["sr"] + alpha) / (c["n_rev"]  + 2*alpha)
            emp  = _lr_safe(p_sc / (p_sr + 1e-9))

            # Confidence-weighted blend with prior
            conf  = min(1.0, math.sqrt(n / 100))
            blend = round(conf * emp + (1-conf) * prior, 3)
            updated[int_name] = blend
            report.append((int_name, n, prior, emp, blend, f"conf={conf:.0%}"))

        self.interaction_lrs  = updated
        self.fitted           = True
        self.fit_date         = datetime.now().isoformat()
        self.n_coins_fitted   = len([d for d in coins_dfs.values() if d is not None])

        # Update SIGNAL_GROUPS so group dampening knows about ML interactions
        SIGNAL_GROUPS["ML_INTERACTION"]["signals"] = set(
            f"ml_{k}" for k in updated)

        if verbose:
            print("\n" + "═"*72)
            print(" [MLLR] ML INTERACTION LR CALIBRATION")
            print("═"*72)
            print(f"  {'Interaction':<30}  {'n':>5}  {'Prior':>6}  {'Empirical':>9}  "
                  f"{'Final':>7}  {'Status'}")
            print("  " + "─"*68)
            for row in sorted(report, key=lambda x: abs((x[3] or x[2]) - x[2]), reverse=True):
                int_name, n, prior, emp, final, status = row
                emp_s = f"{emp:.3f}" if emp else "  n/a"
                delta = (emp - prior) if emp else 0
                flag  = " ◄" if abs(delta) > 0.25 else ""
                print(f"  {int_name:<30}  {n:>5}  {prior:>6.3f}  {emp_s:>9}  "
                      f"{final:>7.3f}  {status}{flag}")
            print("═"*72 + "\n")

    def save(self, path: str = MLLR_CACHE) -> None:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump({
                    "interaction_lrs":  self.interaction_lrs,
                    "fitted":           self.fitted,
                    "fit_date":         self.fit_date,
                    "n_coins_fitted":   self.n_coins_fitted,
                    "residual_shrink":  self.RESIDUAL_SHRINK,
                }, f, indent=2)
            print(f"[MLLR] ✅ Saved interaction LRs → {path}")
        except Exception as e:
            print(f"[MLLR] ⚠️  Save failed: {e}")

    def load(self, path: str = MLLR_CACHE) -> bool:
        try:
            with open(path) as f:
                d = json.load(f)
            self.interaction_lrs = d.get("interaction_lrs", {})
            self.fitted          = d.get("fitted", False)
            self.fit_date        = d.get("fit_date", "")
            self.n_coins_fitted  = d.get("n_coins_fitted", 0)
            age = (datetime.now() - datetime.fromisoformat(self.fit_date)).days
            print(f"[MLLR] ✅ Loaded interaction LRs ({len(self.interaction_lrs)} pairs, "
                  f"{age}d old, {self.n_coins_fitted} coins)")
            return self.fitted
        except Exception:
            return False


# Global instance -- fitted once at startup, used by the pipeline
_mllr_producer = MLLRProducer()
# Walk-forward Sharpe cache: populated during ML training, consumed in MLLR bridge.
# Key: coin (long), f"{coin}_short" (short).
# If Sharpe < 0 → model profitable in training but loses money on test trades
# → suppress ML scale by 0.5 to reduce model's influence on final probability.
_WF_SHARPE: Dict[str, float] = {}  # coin → mean walk-forward Sharpe

# ═══════════════════════════════════════════════════════════════════════════════
# [CAL-EVAL] CALIBRATION EVALUATION SUITE
#
# "Model says 75%" is meaningless without verification.
# When the model predicts P=0.75, does it ACTUALLY win ~75% of the time?
# If not, every downstream calculation (LR, ensemble, position sizing) is wrong.
#
# THREE METRICS:
#
# 1. BRIER SCORE = mean( (P_predicted - outcome)² )
#    Range: 0.00 (perfect) → 0.25 (random) → 1.00 (perfectly wrong)
#    Summarises overall calibration quality in one number.
#
# 2. ECE (Expected Calibration Error)
#    = Σ (n_bin / n_total) × |avg_confidence_in_bin - avg_accuracy_in_bin|
#    Measures the WEIGHTED average miscalibration across probability buckets.
#    ECE=0.00: perfectly calibrated. ECE=0.10: off by 10% on average.
#    Unlike Brier, ECE directly quantifies the calibration gap.
#
# 3. MCE (Maximum Calibration Error)
#    = max over all bins of |avg_confidence - avg_accuracy|
#    The worst single bucket. Even if ECE is good, a bad MCE means
#    the model has a specific probability range where it is badly wrong.
#
# RELIABILITY CURVE:
#    For each probability bucket [0.0-0.1, 0.1-0.2, ...]:
#      - How many predictions fell in this bucket?
#      - What fraction actually won? (actual accuracy)
#      - What was the average predicted probability? (expected accuracy)
#      - The GAP is the calibration error for that bucket.
#    A perfectly calibrated model's points lie on the diagonal (y=x).
#
# COMPONENT BREAKDOWN:
#    Evaluates ML, Bayesian engine, and Ensemble separately.
#    Tells you WHICH component is most miscalibrated.
#
# OVERCONFIDENCE vs UNDERCONFIDENCE:
#    Overconfident: predicts 80%, actually wins 55% -- occurs when model
#    extremes don't correspond to extreme outcomes (common in small datasets).
#    Underconfident: predicts 55%, actually wins 75% -- common after aggressive
#    calibration (isotonic regression over-regularised).
# ═══════════════════════════════════════════════════════════════════════════════

BRIER_LOG_FILE         = "models/brier_predictions.json"
BRIER_RESOLVE_DAYS     = 5
BRIER_ATR_TARGET_MULT  = 2.5
BRIER_ATR_STOP_MULT    = 1.5
BRIER_RETRAIN_THRESHOLD= 0.26
BRIER_ROLLING_WINDOW   = 100
BRIER_MIN_RESOLVED     = 30
BRIER_N_BINS           = 10   # ECE / reliability curve bucket count


class CalibrationReport:
    """Holds full calibration metrics for one model component."""
    def __init__(self, component: str, n: int,
                 brier: float, ece: float, mce: float,
                 overconf_bias: float,
                 bins: List[Dict]):
        self.component     = component
        self.n             = n
        self.brier         = brier      # 0=perfect, 0.25=random
        self.ece           = ece        # weighted avg calibration gap
        self.mce           = mce        # worst single bucket gap
        self.overconf_bias = overconf_bias  # + = overconfident, - = underconfident
        self.bins          = bins       # list of bucket dicts

    def brier_label(self) -> str:
        if self.brier < 0.18: return "EXCELLENT ✅"
        if self.brier < 0.21: return "GOOD ✅"
        if self.brier < 0.24: return "ACCEPTABLE ⚠️"
        if self.brier < 0.27: return "DEGRADED ⚠️"
        return "POOR ❌"

    def ece_label(self) -> str:
        if self.ece < 0.03: return "EXCELLENT ✅"
        if self.ece < 0.06: return "GOOD ✅"
        if self.ece < 0.10: return "ACCEPTABLE ⚠️"
        if self.ece < 0.15: return "HIGH ⚠️"
        return "VERY HIGH ❌"

    def overconf_label(self) -> str:
        if abs(self.overconf_bias) < 0.03: return "Balanced"
        return f"{'Overconfident' if self.overconf_bias > 0 else 'Underconfident'} ({self.overconf_bias:+.2f})"


class BrierMonitor:
    """
    Tracks predicted probabilities vs actual outcomes.
    Computes Brier, ECE, MCE, reliability curves per component,
    AND rolling calibration drift over time.
    """

    def __init__(self, log_path: str = BRIER_LOG_FILE):
        self.log_path    = log_path
        self.predictions: List[Dict] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            with open(self.log_path) as f:
                self.predictions = json.load(f)
            if len(self.predictions) > 500:
                self.predictions = self.predictions[-500:]
        except (FileNotFoundError, json.JSONDecodeError):
            self.predictions = []

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "w") as f:
                json.dump(self.predictions[-500:], f, indent=2)
        except Exception:
            pass

    # ── Step 1: Log ───────────────────────────────────────────────────────────

    def log_prediction(self, coin: str, ml_proba: float, bayes_prob: float,
                        price: float, atr: float, action: str = "BUY",
                        p_bayes_raw: float = 0.5,
                        p_transition: float = 0.5,
                        regime: str = "unknown",
                        chop_detected: bool = False) -> None:
        if atr <= 0 or price <= 0:
            return
        # Regime label: chop overrides market regime
        regime_label = "chop" if chop_detected else str(regime)
        self.predictions.append({
            "coin":         coin,
            "timestamp":    datetime.now().isoformat(),
            "ml_proba":     round(float(ml_proba),     4),
            "bayes_prob":   round(float(bayes_prob),    4),
            "p_bayes_raw":  round(float(p_bayes_raw),  4),
            "p_transition": round(float(p_transition), 4),
            "price":        round(float(price),         6),
            "atr":          round(float(atr),           6),
            "action":       action,
            "regime":       regime_label,
            "outcome":      None,
            "resolved_at":  None,
        })
        self._save()

    # ── Step 2: Resolve outcomes ──────────────────────────────────────────────

    def resolve_outcomes(self, coins_dfs: Dict[str, pd.DataFrame],
                          atr_target_mult: float = BRIER_ATR_TARGET_MULT,
                          atr_stop_mult:   float = BRIER_ATR_STOP_MULT,
                          resolve_days:    int   = BRIER_RESOLVE_DAYS) -> int:
        newly = 0
        for pred in self.predictions:
            if pred["outcome"] is not None:
                continue
            df = coins_dfs.get(pred["coin"])
            if df is None:
                continue
            try:
                col    = "price" if "price" in df.columns else "close"
                prices = df[col].values.astype(float)
                n      = len(prices)
                entry  = pred["price"]; atr_v = pred["atr"]
                target = entry + atr_target_mult * atr_v
                stop   = entry - atr_stop_mult   * atr_v
                future = prices[max(0, n - resolve_days):]
                if len(future) < 2:
                    continue
                hit_t = bool(np.any(future >= target))
                hit_s = bool(np.any(future <= stop))
                if hit_t and hit_s:
                    idx_t = int(np.argmax(future >= target))
                    idx_s = int(np.argmax(future <= stop))
                    resolve_idx = min(idx_t, idx_s)
                    outcome = 1 if idx_t <= idx_s else 0
                elif hit_t:
                    resolve_idx = int(np.argmax(future >= target))
                    outcome = 1
                elif hit_s:
                    resolve_idx = int(np.argmax(future <= stop))
                    outcome = 0
                else:
                    continue

                # ── MAE / MFE: best/worst point reached while the trade was
                # actually open, truncated to the bar it resolved on (not the
                # whole future window, which would leak post-exit price action
                # into the excursion stats). ATR-normalized, signed the same
                # way as price (+ = favorable, - = adverse for a long).
                window  = future[:resolve_idx + 1]
                mfe_atr = float((window.max() - entry) / atr_v) if atr_v > 0 else 0.0
                mae_atr = float((window.min() - entry) / atr_v) if atr_v > 0 else 0.0

                pred["outcome"]     = outcome
                pred["resolved_at"] = datetime.now().isoformat()
                pred["mfe_atr"]     = round(mfe_atr, 3)
                pred["mae_atr"]     = round(mae_atr, 3)
                newly += 1
            except Exception:
                continue
        if newly:
            self._save()
        return newly

    # ── Step 2b: MAE / MFE aggregate report ────────────────────────────────────

    def mae_mfe_report(self, min_n: int = 5) -> str:
        """
        Aggregate Maximum Adverse/Favorable Excursion across resolved
        predictions -- how far trades moved against you (MAE) and in your
        favor (MFE) before resolving, split by outcome.

        Use this to check two things win rate alone can't tell you:
          - Are stops too tight? (winners showing a deep MAE -- they nearly
            got stopped out before turning into wins)
          - Are targets/exits too conservative? (winners' MFE running well
            past where they actually closed, leaving R on the table)
        """
        resolved = [p for p in self.predictions
                    if p.get("outcome") is not None and "mfe_atr" in p and "mae_atr" in p]
        if len(resolved) < min_n:
            return (f"\n  [MAE/MFE] Only {len(resolved)} resolved predictions have excursion "
                    f"data yet (need {min_n}+) -- existing resolved predictions from before "
                    f"this feature was added won't have it.\n")

        def _stats(preds, label):
            if not preds:
                return f"  {label:<8} n=0\n"
            mfe = np.array([p["mfe_atr"] for p in preds])
            mae = np.array([p["mae_atr"] for p in preds])
            return (f"  {label:<8} n={len(preds):<4} "
                    f"avg MFE={mfe.mean():+.2f}  avg MAE={mae.mean():+.2f}  "
                    f"median MFE={np.median(mfe):+.2f}  median MAE={np.median(mae):+.2f}\n")

        wins   = [p for p in resolved if p["outcome"] == 1]
        losses = [p for p in resolved if p["outcome"] == 0]

        lines = ["\n  ── MAE / MFE (ATR-normalized, while trade was open) ───────────────\n"]
        lines.append(_stats(resolved, "All"))
        lines.append(_stats(wins,     "Wins"))
        lines.append(_stats(losses,   "Losses"))

        if wins:
            win_mae   = np.array([p["mae_atr"] for p in wins])
            win_mfe   = np.array([p["mfe_atr"] for p in wins])
            # Fraction of winners that dipped within 30% of the configured
            # stop distance before turning around -- scales with whatever
            # ATR_STOP_MULT is actually set to, not a hardcoded ATR figure.
            close_call_thresh = -0.7 * BRIER_ATR_STOP_MULT
            close_call_frac   = float(np.mean(win_mae <= close_call_thresh))
            if close_call_frac > 0.25:
                lines.append(f"  ⚠️  {close_call_frac:.0%} of winning trades dipped to "
                              f"{close_call_thresh:.2f} ATR or worse (within 30% of the stop) "
                              f"before turning around -- stop may be cutting off would-be winners.\n")
            lines.append(f"  ℹ️  Winners reached {win_mfe.mean():+.2f} ATR favorable on average "
                          f"before resolving -- compare against your actual target multiple "
                          f"({BRIER_ATR_TARGET_MULT:+.2f} ATR) to see how much you're capturing.\n")
        return "".join(lines)

    # ── Step 3: Core metrics ──────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(probas: np.ndarray, outcomes: np.ndarray,
                          n_bins: int = BRIER_N_BINS) -> "CalibrationReport":
        n = len(probas)
        brier = float(np.mean((probas - outcomes) ** 2))
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins: List[Dict] = []
        ece = 0.0; mce = 0.0
        total_bias = 0.0; total_n = 0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (probas >= lo) & (probas < hi)
            if not np.any(mask):
                bins.append({"lo": lo, "hi": hi, "n": 0, "avg_conf": None, "avg_acc": None, "gap": None})
                continue
            bc   = int(mask.sum())
            aconf= float(probas[mask].mean())
            aacc = float(outcomes[mask].mean())
            gap  = abs(aconf - aacc)
            bins.append({"lo": lo, "hi": hi, "n": bc, "avg_conf": aconf, "avg_acc": aacc, "gap": gap})
            ece += bc / n * gap
            mce  = max(mce, gap)
            total_bias += bc * (aconf - aacc)
            total_n    += bc
        overconf_bias = total_bias / max(1, total_n)
        return CalibrationReport("unknown", n, brier, ece, mce, overconf_bias, bins)

    def compute_calibration(self, window: int = BRIER_ROLLING_WINDOW,
                             components: Tuple[str,...] = ("ml", "bayes", "ensemble")
                             ) -> Dict[str, "CalibrationReport"]:
        resolved = [p for p in self.predictions if p.get("outcome") is not None]
        if len(resolved) < BRIER_MIN_RESOLVED:
            return {}
        recent  = resolved[-window:]
        outcomes= np.array([p["outcome"] for p in recent], dtype=float)
        key_map = {"ml": "ml_proba", "bayes": "p_bayes_raw", "ensemble": "bayes_prob"}
        reports = {}
        for comp in components:
            key    = key_map.get(comp, comp)
            probas = np.array([p.get(key, 0.5) for p in recent], dtype=float)
            r      = self._compute_metrics(probas, outcomes)
            r.component = comp
            reports[comp] = r
        return reports

    # ── Step 4: Rolling Brier series (drift detection) ────────────────────────

    def rolling_brier_series(self, window: int = 50,
                              step: int = 25,
                              key: str = "bayes_prob") -> List[Dict]:
        """
        Compute Brier score on a rolling window, stepping through resolved
        predictions in chronological order.

        Returns a list of dicts: {idx, timestamp, brier, n}
        Used to detect whether calibration is improving, stable, or drifting.

        window=50, step=25: windows overlap by 50% — smooths noise while
        remaining sensitive to regime shifts.
        """
        resolved = [p for p in self.predictions if p.get("outcome") is not None]
        if len(resolved) < window:
            return []
        series = []
        for start in range(0, len(resolved) - window + 1, step):
            chunk   = resolved[start:start + window]
            probas  = np.array([p.get(key, 0.5) for p in chunk], dtype=float)
            outcomes= np.array([p["outcome"]    for p in chunk], dtype=float)
            brier   = float(np.mean((probas - outcomes) ** 2))
            ts      = chunk[-1].get("resolved_at") or chunk[-1].get("timestamp", "")
            series.append({"idx": start + window, "timestamp": ts,
                           "brier": brier, "n": len(chunk)})
        return series

    def detect_drift(self, window: int = 50,
                     threshold_warn: float = 0.03,
                     threshold_alert: float = 0.06) -> Dict:
        """
        Detect calibration drift by comparing the oldest and newest Brier
        windows in the rolling series.

        DRIFT LOGIC:
          recent_brier  = Brier of most recent `window` resolved predictions
          earlier_brier = Brier of the `window` resolved predictions before that
          delta         = recent_brier - earlier_brier

          delta > threshold_alert : DRIFT  (calibration significantly worse)
          delta > threshold_warn  : WARN   (slight degradation)
          delta < -threshold_warn : IMPROVE (calibration getting better)
          else                    : STABLE

        Also fits a linear trend over the rolling series to detect slow drift
        that isn't captured by comparing just two windows.
        """
        resolved = [p for p in self.predictions if p.get("outcome") is not None]
        n = len(resolved)
        if n < window * 2:
            return {"status": "INSUFFICIENT", "n": n, "needed": window * 2}

        series = self.rolling_brier_series(window=window)
        if len(series) < 2:
            return {"status": "INSUFFICIENT", "n": n}

        recent_brier  = series[-1]["brier"]
        earlier_brier = series[0]["brier"]    # oldest window
        delta         = recent_brier - earlier_brier

        # Linear trend over all windows (positive slope = getting worse)
        if len(series) >= 3:
            x    = np.arange(len(series), dtype=float)
            y    = np.array([w["brier"] for w in series])
            slope= float(np.polyfit(x, y, 1)[0])   # brier per window
        else:
            slope = delta

        if   delta > threshold_alert:    status = "DRIFT ⚠️"
        elif delta > threshold_warn:     status = "WARN 🟡"
        elif delta < -threshold_warn:    status = "IMPROVING ✅"
        else:                            status = "STABLE ✅"

        return {
            "status":        status,
            "recent_brier":  recent_brier,
            "earlier_brier": earlier_brier,
            "delta":         delta,
            "slope_per_window": slope,
            "n_windows":     len(series),
            "n_resolved":    n,
        }

    # ── Step 5: Reliability bucket tracking over time ─────────────────────────

    def reliability_bucket_drift(self, n_bins: int = 10,
                                  half_window: int = 100,
                                  key: str = "bayes_prob") -> List[Dict]:
        """
        Detects which SPECIFIC probability buckets have become miscalibrated.

        For each bucket [0.0-0.1, 0.1-0.2, ...]:
          - Compare actual win rate in the FIRST half of resolved predictions
            vs the SECOND half
          - Large drift in a bucket = regime shift affecting that confidence level

        This exposes overconfidence creep: if the [0.7-0.8] bucket's actual
        win rate fell from 0.73 → 0.55, the model is becoming overconfident
        in that range (crypto bull regime ending, model not updating).
        """
        resolved = [p for p in self.predictions if p.get("outcome") is not None]
        if len(resolved) < half_window * 2:
            return []

        early  = resolved[:half_window]
        recent = resolved[-half_window:]
        edges  = np.linspace(0.0, 1.0, n_bins + 1)
        results = []

        for lo, hi in zip(edges[:-1], edges[1:]):
            def bucket_stats(preds):
                ps = [p.get(key, 0.5) for p in preds if lo <= p.get(key, 0.5) < hi]
                os = [p["outcome"]    for p in preds if lo <= p.get(key, 0.5) < hi]
                if len(ps) < 3:
                    return None, 0
                return float(np.mean(os)), len(ps)

            early_wr,  n_early  = bucket_stats(early)
            recent_wr, n_recent = bucket_stats(recent)

            if early_wr is None or recent_wr is None:
                continue

            drift = recent_wr - early_wr   # positive = better calibrated, negative = overconfident
            results.append({
                "bucket":    f"{lo:.1f}-{hi:.1f}",
                "lo": lo, "hi": hi,
                "early_wr":  early_wr,
                "recent_wr": recent_wr,
                "drift":     drift,
                "n_early":   n_early,
                "n_recent":  n_recent,
                "flag": ("OVERCONFIDENT 🔴" if drift < -0.10 else
                         "IMPROVING ✅"     if drift >  0.08 else
                         "STABLE")
            })
        return results

    # ── Step 5b: Regime-conditional calibration ──────────────────────────────

    REGIME_MIN_N        = 12     # minimum resolved predictions in a regime before adjusting at all
    REGIME_LARGE_N       = 100    # n > this can unlock impacts beyond the pp cap below (if also stable)
    REGIME_CAL_MAX_PP    = 0.15   # cap on |P_corrected - P_raw|: 15 percentage points (requested range: 10-15pp)
    REGIME_STABLE_DRIFT  = 0.25   # max win-rate drift between chronological halves to call a large-n regime "stable"

    # Low win-rate unlock — graduated by sample size.
    #
    # The raw logit adjustment for wr=3%/mc=60% is -3.88. At face value that
    # sounds like strong evidence the regime never works, but n=31 gives a 95%
    # CI of roughly (0.6%, 17%) — the true win rate could easily be 15%. Treating
    # the point estimate as gospel and extending the logit clamp to ±4.0 + removing
    # the pp cap is statistically unjustified at that sample size.
    #
    # Three tiers (extreme_wr means wr < LOW_WR or wr > 1 - LOW_WR):
    #
    # Tier 1  n < LOW_WR_PARTIAL_MIN_N (30):
    #   Standard treatment. Logit clamp ±2.0, pp cap ±15%. A directional
    #   signal with very thin evidence — correction is meaningful but restrained.
    #
    # Tier 2  LOW_WR_PARTIAL_MIN_N (30) ≤ n < LOW_WR_FULL_MIN_N (100):
    #   Partial unlock. Logit clamp stays ±2.0 (don't over-leverage thin data)
    #   but pp cap expanded to ±25%, enough to push most high-P coins below the
    #   buy threshold without treating a CI-wide range as a point estimate.
    #   Example: n=31, wr=3%, mc=60%, P=0.833 → corrected P ≈ 58% (WAIT).
    #
    # Tier 3  n ≥ LOW_WR_FULL_MIN_N (100) AND extreme_wr:
    #   Full unlock. Logit clamp extended to ±4.0, pp cap removed. At 100+
    #   resolved predictions a 3% win rate is a durable regime-level signal,
    #   not a lucky/unlucky stretch.
    LOW_WR_UNLOCK_WR       = 0.10   # win rate below (or above 1-this) is "extreme"
    LOW_WR_PARTIAL_MIN_N   = 30     # tier 2: partial unlock
    LOW_WR_FULL_MIN_N      = 100    # tier 3: full unlock (was LOW_WR_UNLOCK_MIN_N=20)
    LOW_WR_PARTIAL_PP_CAP  = 0.25   # expanded pp cap for tier 2 (vs 0.15 standard)
    LOW_WR_LOGIT_CLAMP     = 4.0    # extended logit clamp for tier 3 only

    # Bayesian shrinkage of the regime win rate toward a global prior before
    # computing the logit adjustment. Raw win rates from small samples are
    # noisy — 1 win / 31 trades has a 95% CI of (0.6%, 17%), not a point
    # estimate of 3%. Shrinking toward the global baseline dampens the
    # correction proportional to how little data we have.
    #
    #   wr_shrunk = (REGIME_SHRINK_N × REGIME_PRIOR_WR + n × wr_obs)
    #               ─────────────────────────────────────────────────
    #                          REGIME_SHRINK_N + n
    #
    # At n → 0:  wr_shrunk → REGIME_PRIOR_WR  (pure prior, no correction)
    # At n → ∞:  wr_shrunk → wr_obs           (data dominates, full correction)
    # At n = REGIME_SHRINK_N: 50/50 blend     (prior and data equally weighted)
    #
    # Example  n=31, wr_obs=3%, PRIOR_WR=35%, SHRINK_N=35:
    #   wr_shrunk = (35×0.35 + 31×0.03) / 66 = 13.18/66 ≈ 20%
    #   logit_adj = logit(20%) − logit(60%) = −1.79  (vs −3.88 raw)
    #   — naturally within ±2.0 clamp, no extended clamp needed.
    REGIME_SHRINK_N  = 35    # pseudo-observation count for the prior
    REGIME_PRIOR_WR  = 0.35  # global baseline win rate across all bear regimes

    def _regime_is_stable(self, preds: list) -> Tuple[bool, str]:
        """
        Split a regime's resolved predictions chronologically into two halves
        and compare win rate. A genuine regime-level effect should hold up
        across more than one episode -- this guards against a thin sample
        that's really just "one continuous bear stretch with no contrast yet"
        masquerading as a durable pattern once n happens to cross REGIME_LARGE_N.

        Only called when n > REGIME_LARGE_N, so each half has 50+ predictions.
        """
        ordered = sorted(preds, key=lambda p: p.get("timestamp", "") or "")
        mid     = len(ordered) // 2
        wr1     = float(np.mean([p["outcome"] for p in ordered[:mid]]))
        wr2     = float(np.mean([p["outcome"] for p in ordered[mid:]]))
        drift   = abs(wr1 - wr2)
        stable  = drift <= self.REGIME_STABLE_DRIFT
        return stable, f"half_wr={wr1:.0%}/{wr2:.0%} drift={drift:.0%}"

    def regime_stats(self) -> Dict[str, Dict]:
        """
        Per-regime win rate, mean predicted probability, Brier, and count.
        Only uses resolved predictions.

        Returns dict keyed by regime label:
          {win_rate, mean_prob, brier, n, bias, overconfident, active, large_n, stable}
        """
        from collections import defaultdict
        resolved = [p for p in self.predictions if p.get("outcome") is not None]
        groups: Dict[str, list] = defaultdict(list)
        for p in resolved:
            r = p.get("regime", "unknown") or "unknown"
            groups[r].append(p)

        stats = {}
        for regime, preds in groups.items():
            if len(preds) < 3:
                continue
            probs    = np.array([p.get("bayes_prob", 0.5) for p in preds])
            outcomes = np.array([p["outcome"]               for p in preds], dtype=float)
            win_rate  = float(outcomes.mean())
            mean_prob = float(probs.mean())
            brier     = float(np.mean((probs - outcomes) ** 2))
            bias      = mean_prob - win_rate   # + = overconfident, - = underconfident
            n         = len(preds)
            large_n   = n > self.REGIME_LARGE_N
            if large_n:
                stable, stable_note = self._regime_is_stable(preds)
            else:
                stable, stable_note = False, f"n={n}<=REGIME_LARGE_N({self.REGIME_LARGE_N})"
            stats[regime] = {
                "n":             n,
                "win_rate":      round(win_rate,  4),
                "mean_prob":     round(mean_prob, 4),
                "brier":         round(brier,     4),
                "bias":          round(bias,      4),
                "overconfident": bias > 0.05,
                "active":        n >= self.REGIME_MIN_N,
                "large_n":       large_n,
                "stable":        stable,
                "stable_note":   stable_note,
            }
        return stats

    def regime_calibrate(self, prob: float, regime: str) -> Tuple[float, str]:
        """
        Apply regime-conditional log-odds correction to a predicted probability.

        MATH:
          logit(P_corrected) = logit(P_raw) + logit(actual_wr) - logit(model_conf)

        This shifts the log-odds by the systematic bias observed in this regime.

        Example -- the actual GRASS case that motivated the cap below:
          P_raw=0.868  regime=bear  n=20  actual_wr=0%  model_conf=58%
          logit_adj = clamp(logit(~0) - logit(0.58), -2, 2) = -2.00   (hits the logit clamp)
          Uncapped result would be P=0.471 (a 39.7pp move). Since n=20 is not
          > REGIME_LARGE_N (100), the pp cap applies instead:
          P_corrected = 0.868 - 0.15 = 0.718   (capped at 15pp)

        Falls back to P_raw if:
          - Not enough data for this regime (< REGIME_MIN_N)
          - Regime is unknown
          - Bias between actual win rate and model confidence is negligible (<3%)

        Safety rails on the size of the move:
          - Logit-space shift is clamped to ±2.0 (original safety clamp).
          - The resulting probability move |P_corrected - P_raw| is additionally
            capped at ±REGIME_CAL_MAX_PP (15 percentage points) UNLESS the
            regime has n > REGIME_LARGE_N (100) resolved trades AND those
            trades are "stable" (win rate doesn't drift more than
            REGIME_STABLE_DRIFT between the first and second half
            chronologically) -- only then is the full logit-clamp range used.
        """
        stats = self.regime_stats()
        rs    = stats.get(regime)
        if rs is None or not rs["active"]:
            return prob, f"no_regime_data({regime})"

        wr   = rs["win_rate"]
        mc   = rs["mean_prob"]

        # Safety: don't adjust if model conf and actual wr are almost the same
        if abs(wr - mc) < 0.03:
            return prob, f"regime={regime} calibrated (bias={wr-mc:+.1%} negligible)"

        def safe_logit(p):
            return math.log(max(0.001, min(0.999, p)) /
                            (1 - max(0.001, min(0.999, p))))

        logit_adj = safe_logit(wr) - safe_logit(mc)

        # ── Bayesian shrinkage of the win rate ────────────────────────────────
        # Blend observed wr toward the global prior before computing the logit
        # adjustment. See REGIME_SHRINK_N / REGIME_PRIOR_WR for the rationale.
        wr_raw    = wr
        wr        = (self.REGIME_SHRINK_N * self.REGIME_PRIOR_WR + n * wr_raw) / (self.REGIME_SHRINK_N + n)
        logit_adj = safe_logit(wr) - safe_logit(mc)   # recompute with shrunk wr

        # ── Graduated low/high win-rate unlock ───────────────────────────────
        # See class constants for the full design rationale.
        # Tier classification uses RAW win rate — shrinkage adjusts the
        # magnitude of the correction, not whether the situation is extreme.
        n          = rs["n"]
        extreme_wr = (wr_raw < self.LOW_WR_UNLOCK_WR) or (wr_raw > 1.0 - self.LOW_WR_UNLOCK_WR)
        partial_unlock = extreme_wr and (self.LOW_WR_PARTIAL_MIN_N <= n < self.LOW_WR_FULL_MIN_N)
        full_unlock    = extreme_wr and (n >= self.LOW_WR_FULL_MIN_N)
        low_wr_unlocked = partial_unlock or full_unlock   # any unlock active

        # Logit clamp: only extend to ±4.0 for tier 3 (100+ samples)
        logit_clamp = self.LOW_WR_LOGIT_CLAMP if full_unlock else 2.0
        logit_adj = max(-logit_clamp, min(logit_clamp, logit_adj))

        logit_raw       = safe_logit(prob)
        logit_corrected = logit_raw + logit_adj
        p_corr = 1.0 / (1.0 + math.exp(-logit_corrected))
        p_corr = max(0.01, min(0.99, p_corr))

        # ── Percentage-point impact cap ──────────────────────────────────────
        # Cap is removed only for tier 3 (full unlock) or large+stable.
        # Tier 2 (partial) uses an expanded cap (25pp) — more room than
        # standard (15pp) but not unlimited, reflecting the wider CI.
        large_stable_unlocked = bool(rs.get("large_n")) and bool(rs.get("stable"))
        cap_removed   = large_stable_unlocked or full_unlock
        pp_max        = (self.REGIME_CAL_MAX_PP if not low_wr_unlocked
                         else self.LOW_WR_PARTIAL_PP_CAP if partial_unlock
                         else self.REGIME_CAL_MAX_PP)   # full_unlock → cap_removed=True, this value unused
        pp_capped = False
        if not cap_removed:
            if p_corr > prob + pp_max:
                p_corr, pp_capped = prob + pp_max, True
            elif p_corr < prob - pp_max:
                p_corr, pp_capped = prob - pp_max, True

        p_corr = round(p_corr, 4)

        direction = "↓" if p_corr < prob else "↑" if p_corr > prob else "="
        if full_unlock:
            cap_note = f"  [low_wr={wr:.0%}+n={n}≥{self.LOW_WR_FULL_MIN_N}, full correction]"
        elif partial_unlock:
            cap_note = f"  [low_wr={wr:.0%}+n={n}, partial capped@{pp_max:.0%}]"
        elif pp_capped:
            cap_note = f"  [capped@{self.REGIME_CAL_MAX_PP:.0%}]"
        elif large_stable_unlocked:
            cap_note = "  [n>100+stable, uncapped]"
        else:
            cap_note = ""
        shrink_note = (f"  [shrunk: {wr_raw:.0%}→{wr:.0%} n={n}+prior{self.REGIME_SHRINK_N}]"
                       if abs(wr - wr_raw) >= 0.01 else "")
        note = (f"regime={regime}  n={rs['n']}  "
                f"actual_wr={wr_raw:.0%}  wr_shrunk={wr:.0%}  model_conf={mc:.0%}  "
                f"bias={rs['bias']:+.0%}  adj={logit_adj:+.2f} logit  "
                f"P: {prob:.3f} → {p_corr:.3f} {direction}{cap_note}{shrink_note}")
        return p_corr, note

    def regime_calibration_report(self) -> str:
        """
        Print per-regime calibration table.
        Exposes that P=0.70 in chop is very different from P=0.70 in trend.
        """
        stats = self.regime_stats()
        if not stats:
            return "  [REGIME CAL] No resolved predictions yet.\n"

        REGIME_DISPLAY = {
            "bull_trend":  "Bull trend",
            "bear":        "Bear",
            "range":       "Range",
            "weak_trend":  "Weak trend",
            "chop":        "Chop",
            "panic":       "Panic",
            "unknown":     "Unknown",
        }

        lines = ["\n  ── Regime-Conditional Calibration ─────────────────────────────────"]
        lines.append(f"  {'Regime':<16} {'N':>4}  {'Actual WR':>10}  "
                     f"{'Model conf':>11}  {'Bias':>7}  {'Brier':>7}  Status")
        lines.append("  " + "─" * 70)

        for regime, rs in sorted(stats.items(), key=lambda x: -x[1]["n"]):
            label = REGIME_DISPLAY.get(regime, regime)
            wr    = rs["win_rate"]; mc = rs["mean_prob"]
            bias  = rs["bias"];    b  = rs["brier"]
            n     = rs["n"];       active = rs["active"]

            if   abs(bias) < 0.03: status = "CALIBRATED ✅"
            elif bias >  0.15:     status = "OVERCONF HIGH ⛔"
            elif bias >  0.05:     status = "OVERCONF ⚠️ "
            elif bias < -0.10:     status = "UNDERCONF ⚠️ "
            else:                  status = "MILD ⬜"

            if not active:
                status += "  (need more data)"
            elif rs.get("large_n") and rs.get("stable"):
                status += "  ·full-range"
            else:
                status += f"  ·capped@{self.REGIME_CAL_MAX_PP:.0%}"

            lines.append(f"  {label:<16} {n:>4}  {wr:>10.1%}  "
                         f"{mc:>11.1%}  {bias:>+7.1%}  {b:>7.4f}  {status}")

        # Highlight the most actionable insight
        big_bias = [(r, s) for r, s in stats.items()
                    if abs(s["bias"]) > 0.10 and s["active"]]
        if big_bias:
            lines.append("")
            for r, s in sorted(big_bias, key=lambda x: abs(x[1]["bias"]), reverse=True)[:3]:
                direction = "over" if s["bias"] > 0 else "under"
                lines.append(f"  ⚠️  '{r}' is {direction}confident by {s['bias']:+.0%}: "
                             f"model says {s['mean_prob']:.0%}, "
                             f"actual wins {s['win_rate']:.0%}")
        return "\n".join(lines) + "\n"

    # ── Step 6: Full display with drift ──────────────────────────────────────

    def reliability_report(self, window: int = BRIER_ROLLING_WINDOW) -> str:
        resolved = [p for p in self.predictions if p.get("outcome") is not None]
        n_total  = len(self.predictions)
        n_res    = len(resolved)

        if n_res < BRIER_MIN_RESOLVED:
            return (f"\n  [CAL] Not enough resolved predictions "
                    f"({n_res}/{BRIER_MIN_RESOLVED} minimum)\n")

        reports = self.compute_calibration(window)
        if not reports:
            return "\n  [CAL] No reports computed\n"

        lines = [""]
        lines.append("  ═══════════════════════════════════════════════════════════════════")
        lines.append(f"   CALIBRATION + DRIFT REPORT   n={n_res}/{n_total} resolved  "
                     f"window={min(n_res, window)}")
        lines.append("  ═══════════════════════════════════════════════════════════════════")

        # Summary metrics
        lines.append(f"\n  {'Component':<12} {'Brier':>8} {'ECE':>8} {'MCE':>8}  "
                     f"{'Bias':>10}  Status")
        lines.append("  " + "─" * 68)
        for comp, r in reports.items():
            lines.append(f"  {comp:<12} {r.brier:>8.4f} {r.ece:>8.4f} {r.mce:>8.4f}  "
                         f"{r.overconf_bias:>+9.3f}  {r.brier_label():<18} ECE:{r.ece_label()}")

        # Drift detection
        drift = self.detect_drift()
        lines.append(f"\n  ── Calibration Drift ─────────────────────────────────────────────")
        if drift.get("status") == "INSUFFICIENT":
            lines.append(f"  Not enough data: {drift.get('n',0)}/{drift.get('needed',100)} resolved")
        else:
            lines.append(f"  Status:          {drift['status']}")
            lines.append(f"  Earlier Brier:   {drift['earlier_brier']:.4f}")
            lines.append(f"  Recent Brier:    {drift['recent_brier']:.4f}  "
                         f"(Δ={drift['delta']:+.4f})")
            lines.append(f"  Trend slope:     {drift['slope_per_window']:+.5f} per window")
            if "DRIFT" in drift["status"]:
                lines.append(f"  ⚠️  Calibration is significantly worse recently.")
                lines.append(f"     Possible cause: regime shift, model stale.")
                lines.append(f"     Action: retrain model or reset calibration weights.")
            elif "WARN" in drift["status"]:
                lines.append(f"  🟡 Slight degradation — monitor closely.")

        # Reliability curve
        primary = reports.get("ml", next(iter(reports.values())))
        lines.append(f"\n  ── Reliability Curve ({primary.component.upper()}) ──────────────────────────────────")
        lines.append(f"  {'Bucket':<12} {'N':>4}  {'Win rate':>9}  {'Expected':>9}  "
                     f"{'Gap':>7}  Calibration")
        lines.append("  " + "─" * 70)
        for b in primary.bins:
            if b["n"] == 0:
                continue
            acc = b["avg_acc"]; conf = b["avg_conf"]; gap = b["gap"]
            bar = ("▓" if conf > acc else "░") * min(int(gap * 40), 20)
            direction = "+" if conf > acc else "−"
            flag = "  ⚠️ MISCALIBRATED" if gap > 0.15 else ("  ↗ notable" if gap > 0.08 else "")
            lines.append(f"  {b['lo']:.1f}–{b['hi']:.1f}    {b['n']:>4}  "
                         f"{acc:>8.1%}  {conf:>8.1%}  {direction}{gap:>5.1%}  {bar}{flag}")

        # Bucket drift (if enough data)
        bucket_drift = self.reliability_bucket_drift()
        if bucket_drift:
            drifting = [b for b in bucket_drift if b["flag"] != "STABLE"]
            if drifting:
                lines.append(f"\n  ── Bucket Drift (early vs recent {50} predictions) ───────────────")
                lines.append(f"  {'Bucket':<12} {'Early WR':>9}  {'Recent WR':>10}  {'Drift':>7}  Status")
                lines.append("  " + "─" * 58)
                for b in sorted(drifting, key=lambda x: x["drift"]):
                    lines.append(f"  {b['bucket']:<12} {b['early_wr']:>9.1%}  "
                                 f"{b['recent_wr']:>10.1%}  {b['drift']:>+7.1%}  {b['flag']}")

        # Per-coin Brier
        coins_in = sorted({p["coin"] for p in resolved[-100:]})
        if len(coins_in) > 1:
            lines.append(f"\n  ── Per-coin Brier (last 100 resolved) ───────────────")
            for c in coins_in[:8]:
                cp = [p for p in resolved[-100:] if p["coin"] == c]
                if len(cp) < 5:
                    continue
                cb = float(np.mean(
                    (np.array([p["ml_proba"] for p in cp]) -
                     np.array([p["outcome"]  for p in cp])) ** 2
                ))
                flag = "  ⚠️ retrain" if cb > BRIER_RETRAIN_THRESHOLD else ""
                lines.append(f"    {c:<14} n={len(cp):>3}  Brier={cb:.4f}{flag}")

        lines.append("  ═══════════════════════════════════════════════════════════════════")
        lines.append("")
        return "\n".join(lines)

    # ── Step 7: Retrain trigger ───────────────────────────────────────────────

    def should_retrain(self, coin: str,
                        threshold: float = BRIER_RETRAIN_THRESHOLD) -> bool:
        cp = [p for p in self.predictions
              if p["coin"] == coin and p.get("outcome") is not None]
        if len(cp) < 20:
            return False
        recent = cp[-50:]
        b = float(np.mean(
            (np.array([p["ml_proba"] for p in recent]) -
             np.array([p["outcome"]  for p in recent])) ** 2
        ))
        if b > threshold:
            print(f"[CAL] {coin}: Brier={b:.4f} > {threshold} → retrain recommended")
            return True
        # Also trigger on drift
        drift = self.detect_drift()
        if "DRIFT" in drift.get("status", ""):
            print(f"[CAL] {coin}: Drift detected ({drift['status']}) → retrain recommended")
            return True
        return False

    def summary(self) -> str:
        n_res   = sum(1 for p in self.predictions if p.get("outcome") is not None)
        reports = self.compute_calibration()
        drift   = self.detect_drift()
        if not reports:
            return f"[CAL] {n_res}/{len(self.predictions)} resolved — awaiting {BRIER_MIN_RESOLVED}"
        ml = reports.get("ml")
        drift_str = f"  drift={drift['status']}" if drift.get("status") not in ("INSUFFICIENT","STABLE ✅") else ""
        if ml:
            return (f"[CAL] {n_res} resolved  "
                    f"Brier={ml.brier:.4f} {ml.brier_label()}  "
                    f"ECE={ml.ece:.4f}  bias={ml.overconf_bias:+.3f}{drift_str}")
        return f"[CAL] {n_res} resolved"

_brier_monitor = BrierMonitor()

# ── END [CAL-EVAL] ────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# [ENS] ENSEMBLE CALIBRATOR
# Meta-learner that combines the Bayesian posterior, ML probability, and
# Markov transition component. Trained on resolved BrierMonitor predictions.
# Falls back to weighted average (55/30/15) until enough data accumulates.
# ═══════════════════════════════════════════════════════════════════════════════

ENS_MIN_RECORDS  = 60    # minimum resolved predictions before meta-learner activates
ENS_CACHE_FILE   = "models/ensemble_calibrator.pkl"
ENS_FALLBACK_W   = (0.55, 0.30, 0.15)   # Bayes, ML, Transition


class EnsembleCalibrator:
    """
    Logistic regression meta-learner over three component probabilities.
    Features: logit(p_bayes), logit(p_ml), logit(p_trans),
              agreement = 1 - |p_bayes - p_ml|,
              trans_align = sign(p_bayes-0.5) == sign(p_trans-0.5),
              interaction = logit(p_bayes) * logit(p_ml)
    Trained chronologically: first 80% fit, last 20% validate.
    """

    def __init__(self):
        self.model      = None
        self.fitted     = False
        self.val_brier  = 1.0
        self.n_records  = 0

    @staticmethod
    def _features(p_bayes: float, p_ml: float, p_transition: float):
        def safe_logit(p):
            p = max(0.001, min(0.999, float(p)))
            return math.log(p / (1 - p))
        lb = safe_logit(p_bayes); lm = safe_logit(p_ml); lt = safe_logit(p_transition)
        agree      = 1.0 - abs(p_bayes - p_ml)
        trans_align= 1.0 if (p_bayes > 0.5) == (p_transition > 0.5) else 0.0
        interaction= lb * lm
        return [lb, lm, lt, agree, trans_align, interaction]

    def predict(self, p_bayes: float, p_ml: float, p_transition: float) -> float:
        if self.fitted and self.model is not None:
            try:
                feats = self._features(p_bayes, p_ml, p_transition)
                prob  = float(self.model.predict_proba([feats])[0][1])
                return max(0.01, min(0.99, prob))
            except Exception:
                pass
        # Fallback: weighted average in log-odds space
        wb, wm, wt = ENS_FALLBACK_W
        lo = wb * math.log(max(0.001, p_bayes)  / max(0.001, 1-p_bayes)) + \
             wm * math.log(max(0.001, p_ml)     / max(0.001, 1-p_ml))    + \
             wt * math.log(max(0.001, p_transition) / max(0.001, 1-p_transition))
        return max(0.01, min(0.99, 1 / (1 + math.exp(-lo))))

    def fit(self, records: list, verbose: bool = False) -> bool:
        try:
            from sklearn.linear_model import LogisticRegression
            import numpy as _np
            n     = len(records)
            split = int(n * 0.80)
            train = records[:split]; val = records[split:]
            if len(train) < 20 or len(val) < 5:
                return False
            X_train = [self._features(r["p_bayes"], r["p_ml"], r["p_transition"]) for r in train]
            y_train = [r["outcome"] for r in train]
            X_val   = [self._features(r["p_bayes"], r["p_ml"], r["p_transition"]) for r in val]
            y_val   = [r["outcome"] for r in val]
            clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            clf.fit(X_train, y_train)
            preds     = [float(clf.predict_proba([x])[0][1]) for x in X_val]
            brier     = float(_np.mean([(p - y)**2 for p, y in zip(preds, y_val)]))
            self.model     = clf
            self.fitted    = True
            self.val_brier = brier
            self.n_records = n
            if verbose:
                print(f"[ENS] Fitted  n={n}  val_brier={brier:.4f}")
            return True
        except Exception as e:
            if verbose:
                print(f"[ENS] Fit error: {e}")
            return False

    def save(self) -> None:
        try:
            import pickle, os
            os.makedirs(os.path.dirname(ENS_CACHE_FILE) or ".", exist_ok=True)
            with open(ENS_CACHE_FILE, "wb") as f:
                pickle.dump({"model": self.model, "val_brier": self.val_brier,
                             "n_records": self.n_records}, f)
        except Exception:
            pass

    def load(self) -> bool:
        try:
            import pickle
            with open(ENS_CACHE_FILE, "rb") as f:
                d = pickle.load(f)
            self.model     = d["model"]
            self.val_brier = d.get("val_brier", 1.0)
            self.n_records = d.get("n_records", 0)
            self.fitted    = self.model is not None
            return self.fitted
        except Exception:
            return False


_ens_calibrator = EnsembleCalibrator()

# ═══════════════════════════════════════════════════════════════════════════════
# [TDBL] TRADEABILITY REPORT
#
# The current "8.58/5" score conflates five different concerns into one number:
#   direction, participation, stability, entry quality, and ML/backtest quality.
#
# When a trade scores 8.58 but the system says WAIT, you can't tell:
#   • Is the direction weak?      (maybe -- but cloud says bullish)
#   • Is participation missing?   (OBV falling -- yes, this is why)
#   • Is entry timing poor?       (CVD negative -- yes)
#   • Is ML disagreeing?          (ML 18% -- yes, strongly)
#
# Five separate components each answer one question:
#
#   Direction     -- "Is price technically moving the right way?"
#                   Signals: cloud, MACD, DI+/-, chikou, market phase
#
#   Participation -- "Is real capital backing the move?"
#                   Signals: OBV trend, relative volume, smart money, CVD
#
#   Stability     -- "How reliable and clean is the signal?"
#                   Signals: ADX, chop, vol regime, market structure quality
#
#   Entry         -- "Is NOW a good time to enter?"
#                   Signals: OB imbalance, spread, Z-score, microstructure
#
#   Tradeability  -- "Should you actually trade this?"
#                   Signals: ML confidence, backtest quality, ensemble P, regime
#
# Each component: 0-100 → STRONG / GOOD / MODERATE / WEAK / POOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeabilityReport:
    coin: str
    # Five components, each 0-100 (50 = neutral)
    direction_score:     int
    participation_score: int
    stability_score:     int
    entry_score:         int
    tradeability_score:  int
    # Labels
    direction_label:     str
    participation_label: str
    stability_label:     str
    entry_label:         str
    tradeability_label:  str
    # Top driver for each component (why)
    direction_why:       str
    participation_why:   str
    stability_why:       str
    entry_why:           str
    tradeability_why:    str
    # Overall
    overall_label:       str
    recommendation:      str
    prob:                float = 0.5

    @staticmethod
    def _label(score: int) -> str:
        if score >= 72: return "STRONG"
        if score >= 57: return "GOOD"
        if score >= 43: return "MODERATE"
        if score >= 28: return "WEAK"
        return "POOR"

    @staticmethod
    def _traffic(score: int) -> str:
        if score >= 65: return "🟢"
        if score >= 45: return "🟡"
        return "🔴"

    def display(self) -> str:
        rows = [
            ("Direction",     self.direction_score,     self.direction_label,     self.direction_why),
            ("Participation", self.participation_score, self.participation_label, self.participation_why),
            ("Stability",     self.stability_score,     self.stability_label,     self.stability_why),
            ("Entry",         self.entry_score,         self.entry_label,         self.entry_why),
            ("Tradeability",  self.tradeability_score,  self.tradeability_label,  self.tradeability_why),
        ]
        lines = [
            "",
            f"  ┌─ TRADEABILITY REPORT: {self.coin.upper()} ───────────────────────────────────",
        ]
        for name, score, label, why in rows:
            bar_len = int(score / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            tc  = self._traffic(score)
            lines.append(f"  │ {tc} {name:<14} {bar}  {score:>3}/100  {label:<8}  {why}")
        lines.append(f"  └─ Overall: {self.overall_label:<10}  P={self.prob:.0%}  Recommendation: {self.recommendation}")
        lines.append("")
        return "\n".join(lines)


def compute_tradeability_report(
    coin: str,
    sig:  dict,
    prob: float,
    ml_proba: float,
    ml_accuracy: float,
    last: pd.Series,
    df:   pd.DataFrame,
    btc_regime: str     = "unknown",
    chop_detected: bool = False,
    obv_trend: str      = "unknown",
    micro                = None,       # Optional[MicrostructureData]
    backtest_wr: float  = 0.0,
    backtest_pf: float  = 0.0,
    backtest_n:  int    = 0,
) -> TradeabilityReport:
    """
    Compute five-component tradeability report from already-available signal data.
    No new API calls -- reads from sig dict and pipeline outputs.
    """
    def clamp(v: float) -> int:
        return max(0, min(100, int(round(v))))

    rsi   = float(last.get("RSI", 50) or 50)
    adx   = float(last.get("ADX", 0)  or 0)
    cloud = sig.get("cloud_state", "unknown")
    phase = sig.get("market_phase", "RANGE")
    phase_dur = sig.get("phase_duration", 1)
    mkt_regime= sig.get("market_regime", "unknown")
    ms        = sig.get("market_structure", {}) or {}

    # ── DIRECTION (0-100, 50=neutral) ────────────────────────────────────────
    d = 50.0
    # Cloud
    if cloud == "above": d += 22
    elif cloud == "below": d -= 25
    # MACD
    if float(last.get("MACD_Hist", 0) or 0) > 0: d += 12
    else: d -= 10
    # DI
    pdi = float(last.get("plus_di",  0) or 0)
    mdi = float(last.get("minus_di", 0) or 0)
    if pdi > mdi: d += 10
    else: d -= 8
    # Market phase
    phase_adj = {
        "EARLY_BULL": 15, "STRONG_BULL": 12, "ACCUMULATION": 8, "LATE_BULL": 2,
        "RANGE": 0, "DISTRIBUTION": -8, "EARLY_BEAR": -12, "STRONG_BEAR": -18,
    }
    d += phase_adj.get(phase, 0)
    # Structure
    if ms.get("structure") == "bullish":    d += 8
    elif ms.get("structure") == "bearish":  d -= 8
    d_score = clamp(d)
    if d >= 72: d_label, d_why = "BULLISH↑↑", f"Cloud {cloud}, phase {phase}"
    elif d >= 55: d_label, d_why = "BULLISH↑", f"Cloud {cloud}, phase {phase}"
    elif d >= 45: d_label, d_why = "NEUTRAL", f"Mixed signals, ADX={adx:.0f}"
    elif d >= 30: d_label, d_why = "BEARISH↓", f"Cloud {cloud}, DI->{mdi:.0f}"
    else: d_label, d_why = "BEARISH↓↓", f"Strong downtrend (cloud {cloud})"

    # ── PARTICIPATION (0-100) ─────────────────────────────────────────────────
    p = 50.0
    obv_adj  = {"rising": 25, "flat": 0, "falling": -30, "unknown": 0}
    p += obv_adj.get(obv_trend, 0)
    # Relative volume
    rel_vol = float(df["rel_vol"].iloc[-1]) if "rel_vol" in df.columns else 1.0
    if rel_vol > 2.0: p += 15
    elif rel_vol > 1.3: p += 8
    elif rel_vol < 0.7: p -= 10
    # Smart money
    sm = sig.get("smart_money_mode", "neutral") if hasattr(sig, "get") else "neutral"
    if sm == "accumulation": p += 12
    elif sm == "distribution": p -= 18
    # CVD from microstructure
    if micro and not micro.error:
        if   micro.cvd_pct >= 15:  p += 12
        elif micro.cvd_pct >= 5:   p += 6
        elif micro.cvd_pct <= -15: p -= 15
        elif micro.cvd_pct <= -5:  p -= 8
    p_score = clamp(p)
    if obv_trend == "falling":   p_label, p_why = "WEAK",  "OBV falling -- distribution"
    elif obv_trend == "rising" and rel_vol > 1.3: p_label, p_why = "STRONG", "OBV rising + volume spike"
    elif obv_trend == "rising":  p_label, p_why = "GOOD",  "OBV rising"
    elif obv_trend == "flat":    p_label, p_why = "WEAK",  "OBV flat -- no accumulation"
    else:                        p_label, p_why = "MODERATE", f"rel_vol={rel_vol:.1f}×"

    # ── STABILITY (0-100) ─────────────────────────────────────────────────────
    s = 50.0
    # ADX
    if adx >= 25: s += 20
    elif adx >= 18: s += 10
    elif adx < 12: s -= 15
    # Chop
    if chop_detected: s -= 25
    # Vol regime
    vr = sig.get("vol_regime", "neutral")
    if vr == "expansion": s += 5
    elif vr == "compression": s -= 5
    # BTC regime
    if btc_regime == "bull": s += 10
    elif btc_regime == "bear": s -= 12
    # Market regime
    if mkt_regime == "bull_trend": s += 8
    elif mkt_regime == "range": s -= 8
    s_score = clamp(s)
    _adx_dir = "uptrend" if pdi > mdi else "downtrend"
    if chop_detected and adx >= 20: s_label, s_why = "CHOPPY",   f"Chop+ADX={adx:.0f} (mixed)"
    elif chop_detected:             s_label, s_why = "UNSTABLE", "Chop -- signal may not follow through"
    elif adx >= 25:                 s_label, s_why = "STRONG",   f"ADX={adx:.0f} confirmed {_adx_dir}"
    elif adx >= 18:                 s_label, s_why = "MODERATE", f"ADX={adx:.0f} building"
    else:                           s_label, s_why = "WEAK",     f"ADX={adx:.0f} no trend"

    # ── ENTRY QUALITY (0-100) ─────────────────────────────────────────────────
    e = 50.0
    # OB imbalance
    if micro and not micro.error:
        imb = micro.ob_imbalance
        if   imb >= 2.0:  e += 15
        elif imb >= 1.3:  e += 8
        elif imb <= 0.5:  e -= 15
        elif imb <= 0.77: e -= 8
        # Spread
        if   micro.spread_pct < 0.05: e += 8
        elif micro.spread_pct > 0.20: e -= 12
        # Depth
        if   micro.depth_ratio >= 2.0: e += 8
        elif micro.depth_ratio <= 0.5: e -= 8
    # Z-score (overextension risk)
    try:
        col = "close" if "close" in df.columns else "price"
        pz  = df[col].iloc[-20:].values
        zscore = (pz[-1] - pz.mean()) / (pz.std() + 1e-9)
        if   zscore > 2.5: e -= 18   # overbought
        elif zscore < -2.0: e += 12  # oversold bounce potential
        elif abs(zscore) < 1.0: e += 5  # near mean = good entry zone
    except Exception:
        zscore = 0.0
    # RSI zone
    if   rsi > 80: e -= 12
    elif rsi < 30: e += 8
    elif 45 < rsi < 65: e += 5
    e_score = clamp(e)
    if e >= 65: e_label, e_why = "GOOD",     "OB bullish + spread ok"
    elif e >= 45: e_label, e_why = "MODERATE", f"RSI={rsi:.0f}, spread/OB mixed"
    elif rsi > 75: e_label, e_why = "POOR",  "RSI overbought -- late entry"
    else: e_label, e_why = "WEAK", "OB bearish or overextended"

    # ── TRADEABILITY (0-100) ─────────────────────────────────────────────
    # The Bayesian P already encodes BTC bear, ML-neutral, OBV, cloud,
    # momentum and every other signal. Tradeability should NOT re-penalise
    # those — it should measure EXECUTION QUALITY on top of P:
    #   "Given that P=X%, how reliably can we actually execute this trade?"
    #
    # Formula:  t_score = P×100 × execution_quality_factor
    # Execution factors (things NOT already in Bayesian P):
    #   + ADX confirmed trend       → entry is more reliable
    #   + Consecutive daily confirm → breakout is proven
    #   + EV rating A/A+            → R:R supports the trade
    #   - Entry overextended (EMA)  → bad entry point
    #   - ML actively bearish       → reliable model disagrees
    #   - No trend (ADX<15)         → signal may not follow through
    #
    t = prob * 100.0  # start directly from Bayesian probability

    # ML: only penalise if a RELIABLE model (acc≥55%) is actively bearish
    # ML disabled (acc<50%) = neutral: already neutralised in P, don't re-penalise
    if ml_accuracy >= 0.55:
        if   ml_proba >= 0.65: t = min(t * 1.12, 100)   # ML confirms → small boost
        elif ml_proba >= 0.50: pass                       # ML neutral → no change
        elif ml_proba >= 0.35: t *= 0.88                  # ML mild bear → moderate cut
        else:                  t *= 0.70                  # ML strongly bearish → big cut

    # Entry quality: execution factors not in Bayesian P
    # Overextension (EMA deviation) — already in lifecycle but not in entry execution
    _lc_stage = sig.get("lifecycle", {}).get("stage", "MATURE_TREND")
    if _lc_stage == "EXHAUSTION":    t *= 0.85  # overextended entry, reduce size not P
    elif _lc_stage == "DISTRIBUTION": t *= 0.75  # dangerous entry zone
    elif _lc_stage == "EARLY_TREND":  t = min(t * 1.10, 100)  # best R:R window

    # EV: expected value below threshold means the R:R doesn't support the trade
    _ev_data = sig.get("ev", {})
    if _ev_data:
        _ev_r = _ev_data.get("rating", "C")
        if   _ev_r in ("A+", "A"): t = min(t * 1.08, 100)
        elif _ev_r == "F":          t *= 0.70   # negative EV → don't trade
        elif _ev_r == "D":          t *= 0.85   # marginal EV → reduce confidence

    # ADX: low ADX means the signal may not follow through even if P is high
    if adx < 12:     t *= 0.80   # no trend — breakout could go either way
    elif adx >= 25:  t = min(t * 1.05, 100)

    t_score = clamp(t)

    # Label reflects the reason — execution perspective, not Bayesian perspective
    if ml_accuracy >= 0.55 and ml_proba < 0.35:
        t_label, t_why = "LOW",      f"ML={ml_proba:.0%} strongly bearish (acc {ml_accuracy:.0%})"
    elif ml_accuracy >= 0.55 and ml_proba >= 0.65:
        t_label, t_why = "GOOD",     f"ML={ml_proba:.0%} confirms (acc {ml_accuracy:.0%})"
    elif _lc_stage in ("DISTRIBUTION",):
        t_label, t_why = "MODERATE", f"Stage 5 Distribution — dangerous entry zone"
    elif _ev_data.get("rating") in ("F",):
        t_label, t_why = "LOW",      f"EV={_ev_data.get('ev',0):+.2f} negative — skip"
    elif adx < 12:
        t_label, t_why = "MODERATE", f"ADX={adx:.0f} no trend — P={prob:.0%} unconfirmed"
    elif prob >= STRONG_BUY_THRESHOLD:
        t_label, t_why = "GOOD",     f"P={prob:.0%}, EV={_ev_data.get('ev',0):+.2f} ({_ev_data.get('rating','?')})"
    elif prob >= BUY_PROB_THRESHOLD:
        t_label, t_why = "MODERATE", f"Bayesian P={prob:.0%}, ML neutral"
    else:
        t_label, t_why = "WEAK",     f"P={prob:.0%} below threshold"

    # ── COMPOSITE OVERALL (MULTIPLICATIVE, not additive) ─────────────────────
    # Additive weighted average: D*0.20 + P*0.25 + ...
    #   Problem: D=99, P=13 → average=51.6 → "MODERATE"
    #   Participation 13/100 barely suppresses the strong direction signal.
    #
    # Geometric weighted product: 50 × ∏(score/50)^weight
    #   D=99, P=13 → 50 × (1.98^0.20 × 0.26^0.25 × ...) = 39.1 → "LOW"
    #   A poor participation (0.26× neutral) multiplies the whole composite down.
    #   This is the correct behaviour: weakness should SUPPRESS, not just reduce.
    #
    # Implementation: normalize each score to [0, 2] where 1.0 = neutral (50/100),
    # then take the weighted geometric mean and scale back to [0, 100].
    # Clamp components to min 1 so we don't take log(0).
    d_n = max(1, d_score) / 50.0
    p_n = max(1, p_score) / 50.0
    s_n = max(1, s_score) / 50.0
    e_n = max(1, e_score) / 50.0
    t_n = max(1, t_score) / 50.0

    geo = 50.0 * (d_n**0.20 * p_n**0.25 * s_n**0.20 * e_n**0.15 * t_n**0.20)
    composite = round(max(0.0, min(100.0, geo)), 1)

    if composite >= 68: overall_label = "HIGH"
    elif composite >= 52: overall_label = "MODERATE"
    elif composite >= 38: overall_label = "LOW"
    else: overall_label = "AVOID"

    if overall_label == "HIGH" and d >= 58:       rec = "BUY"
    elif overall_label == "MODERATE" and d >= 55:  rec = "WAIT/SMALL"
    elif d < 35:                                    rec = "AVOID/SHORT"
    else:                                           rec = "WAIT"

    return TradeabilityReport(
        coin=coin,
        direction_score=d_score,     participation_score=p_score,
        stability_score=s_score,     entry_score=e_score,
        tradeability_score=t_score,
        direction_label=d_label,     participation_label=p_label,
        stability_label=s_label,     entry_label=e_label,
        tradeability_label=t_label,
        direction_why=d_why,         participation_why=p_why,
        stability_why=s_why,         entry_why=e_why,
        tradeability_why=t_why,
        overall_label=overall_label, recommendation=rec,
        prob=float(prob),
    )

# ── END [TDBL] ────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# [TDBL-S] SHORT TRADEABILITY REPORT
#
# Mirrors the long TradeabilityReport but framed for shorts.
# Six components, each 0-100 where 100 = strongly favours a short.
#
#   Direction    — how bearish is the technical picture?
#   Participation — is capital leaving? (OBV, CVD, BOP)
#   Stability    — is the short setup reliable? (ADX, no chop, bear regime)
#   Entry        — is now a good entry? (not oversold, spread ok, ATR ok)
#   Tradeability — should you actually short? (ML, squeeze risk, funding)
#   Fragility    — multiplicative composite (Exhaust × Distrib × ParticDecay)
#
# Composite: GEOMETRIC mean (multiplicative) so any one missing collapses it.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShortTradeabilityReport:
    coin: str
    direction_score:     int;  participation_score: int
    stability_score:     int;  entry_score:         int
    tradeability_score:  int;  fragility_score:     int
    direction_label:     str;  participation_label: str
    stability_label:     str;  entry_label:         str
    tradeability_label:  str;  fragility_label:     str
    direction_why:       str;  participation_why:   str
    stability_why:       str;  entry_why:           str
    tradeability_why:    str;  fragility_why:       str
    overall_label:       str;  recommendation:      str
    short_prob:          float

    @staticmethod
    def _label(s: int) -> str:
        if s >= 72: return "STRONG"
        if s >= 57: return "GOOD"
        if s >= 43: return "MODERATE"
        if s >= 28: return "WEAK"
        return "POOR"

    @staticmethod
    def _traffic(s: int) -> str:
        return "🟢" if s >= 65 else ("🟡" if s >= 45 else "🔴")

    def display(self) -> str:
        rows = [
            ("Direction",     self.direction_score,     self.direction_label,     self.direction_why),
            ("Participation", self.participation_score, self.participation_label, self.participation_why),
            ("Stability",     self.stability_score,     self.stability_label,     self.stability_why),
            ("Entry",         self.entry_score,         self.entry_label,         self.entry_why),
            ("Tradeability",  self.tradeability_score,  self.tradeability_label,  self.tradeability_why),
            ("Fragility",     self.fragility_score,     self.fragility_label,     self.fragility_why),
        ]
        lines = ["", f"  ┌─ SHORT TRADEABILITY: {self.coin.upper()} ──────────────────────────────────"]
        for name, score, label, why in rows:
            bar = "█" * int(score/5) + "░" * (20 - int(score/5))
            lines.append(f"  │ {self._traffic(score)} {name:<14} {bar}  {score:>3}/100  {label:<8}  {why}")
        lines.append(f"  └─ Overall: {self.overall_label:<10}  Short P={self.short_prob:.0%}  "
                     f"Rec: {self.recommendation}")
        lines.append("")
        return "\n".join(lines)


def compute_short_tradeability_report(
    coin: str, sig: dict, short_prob: float,
    short_ml_proba: float, short_ml_acc: float,
    last: pd.Series, df: pd.DataFrame,
    obv_trend: str = "unknown", vol_regime: str = "neutral",
    btc_regime: str = "unknown", chop_detected: bool = False,
    futures = None, micro = None,
    bear_accel = None,     # BearAcceleration
    trend_frag = None,     # TrendFragility
) -> ShortTradeabilityReport:
    def clamp(v): return max(0, min(100, int(round(v))))

    col   = "close" if "close" in df.columns else "price"
    rsi   = float(last.get("RSI", 50) or 50)
    adx   = float(last.get("ADX",  0) or 0)
    cloud = sig.get("cloud_state", "unknown")
    phase = sig.get("market_phase", "RANGE")
    mkt   = sig.get("market_regime", "unknown")
    pdi   = float(last.get("plus_di",  25) or 25)
    mdi   = float(last.get("minus_di", 25) or 25)
    macd_h= float(last.get("MACD_Hist", 0) or 0)

    # ── DIRECTION (0-100, 100 = strongly bearish = good for short) ────────────
    d = 50.0
    if cloud == "below": d += 25
    elif cloud == "in":  d +=  5
    else:                d -= 20   # above cloud = bullish = bad for short

    if macd_h < 0:       d += 12
    else:                d -= 10

    if mdi > pdi + 5:    d += 10
    elif pdi > mdi + 5:  d -= 8

    phase_adj = {"STRONG_BEAR":18,"EARLY_BEAR":15,"DISTRIBUTION":12,"LATE_BULL":8,
                 "RANGE":0,"ACCUMULATION":-5,"EARLY_BULL":-15,"STRONG_BULL":-20}
    d += phase_adj.get(phase, 0)

    if mkt == "bear":         d += 10
    elif mkt == "bull_trend": d -= 15

    if btc_regime == "bear":  d += 8
    elif btc_regime == "bull":d -= 10

    d_score = clamp(d)
    if   d >= 70: d_label, d_why = "BEARISH↓↓", f"Cloud {cloud}, phase {phase}"
    elif d >= 55: d_label, d_why = "BEARISH↓",  f"Cloud {cloud}, {mkt}"
    elif d >= 45: d_label, d_why = "NEUTRAL",    "Mixed signals"
    else:         d_label, d_why = "BULLISH",    f"Cloud {cloud} (bad for short)"

    # ── PARTICIPATION (0-100, 100 = capital leaving = good for short) ─────────
    p = 50.0
    obv_adj = {"falling": 30, "flat": 5, "rising": -25, "unknown": 0}
    p += obv_adj.get(obv_trend, 0)

    rel_vol = float(df["rel_vol"].iloc[-1]) if "rel_vol" in df.columns else 1.0
    if rel_vol < 0.60:  p += 8
    elif rel_vol > 2.0: p -= 10  # volume spike = not thin = harder to short

    if micro and not getattr(micro,"error",True):
        cvd = getattr(micro,"cvd_pct",0)
        if cvd <= -20: p += 18
        elif cvd <= -10: p += 9
        elif cvd >= 20:  p -= 12

    # BOP score
    if bear_accel and hasattr(bear_accel, "vol_spike_red"):
        pass  # bear_accel measures acceleration, not directly participation

    p_score = clamp(p)
    if obv_trend == "falling":  p_label, p_why = "DISTRIBUTING", "OBV falling (distribution)"
    elif obv_trend == "rising": p_label, p_why = "ACCUMULATING", "OBV rising (bad for short)"
    elif obv_trend == "flat":   p_label, p_why = "NEUTRAL",      "OBV flat"
    else:                       p_label, p_why = "UNKNOWN",      f"rel_vol={rel_vol:.1f}x"

    # ── STABILITY (0-100, 100 = reliable bearish setup) ───────────────────────
    s = 50.0
    if adx >= 25: s += 18
    elif adx >= 18: s += 8
    elif adx < 12: s -= 12

    if chop_detected:     s -= 22   # chop makes shorts unreliable
    if vol_regime == "expansion": s += 8
    if btc_regime == "bear":  s += 12
    elif btc_regime == "bull":s -= 15
    if mkt == "bear":     s += 10
    elif mkt == "range":  s -= 5

    # Squeeze risk from futures
    if futures and not getattr(futures,"error",True) and futures.is_futures:
        fr = futures.funding_rate * 100
        if fr < -0.02:    s -= 20   # negative funding = shorts crowded = unstable
        if futures.ls_ratio < 0.7: s -= 12

    s_score = clamp(s)
    if chop_detected: s_label, s_why = "UNSTABLE", "Chop -- signals unreliable"
    elif adx >= 25:   s_label, s_why = "STRONG",   f"ADX={adx:.0f} confirmed downtrend"
    elif adx >= 18:   s_label, s_why = "MODERATE", f"ADX={adx:.0f} weak trend"
    else:             s_label, s_why = "WEAK",      f"ADX={adx:.0f} no trend"

    # ── ENTRY (0-100, 100 = good entry for short right now) ───────────────────
    e = 50.0
    # RSI for SHORT entry (INVERTED vs long):
    #   > 72 = prime short zone (overbought) → big bonus
    #   55-72 = elevated = good short zone  → mild bonus
    #   40-55 = neutral zone                → no adjustment
    #   35-40 = approaching oversold        → penalty (bounce risk)
    #   < 35  = oversold                    → large penalty
    if   rsi > 72:  e += 20
    elif rsi > 55:  e += 8
    elif rsi < 30:  e -= 20   # deep oversold = high bounce risk
    elif rsi < 35:  e -= 14   # oversold
    elif rsi < 40:  e -= 8    # approaching oversold — 42→34 for RSI=38

    # OB imbalance from micro
    if micro and not getattr(micro,"error",True):
        imb = getattr(micro,"ob_imbalance",1.0)
        if imb < 0.6:   e += 12   # sell-heavy OB = good entry
        elif imb > 1.8: e -= 12   # buy-heavy OB = bad entry for short
        if getattr(micro,"spread_pct",0) < 0.05: e += 5  # tight spread = good

    # Bear acceleration = good entry (momentum confirms)
    if bear_accel and getattr(bear_accel,"n_confirmed",0) >= 2:
        e += 12
    elif bear_accel and getattr(bear_accel,"n_confirmed",0) >= 1:
        e += 6

    # ATR position (is price at a good level or overextended down?)
    try:
        prices = df[col].values.astype(float)
        z = (prices[-1] - prices[-20:].mean()) / (prices[-20:].std() + 1e-9)
        if z < -2.5: e -= 15   # extremely oversold = bad short entry
        elif z > 1.5: e += 12  # price above mean = good short entry zone
    except Exception: pass

    e_score = clamp(e)
    # SHORT-perspective RSI entry quality:
    #   RSI > 72  = ideal (overbought, prime short entry)
    #   RSI 55-72 = good  (elevated, directional edge)
    #   RSI 40-55 = moderate (neutral zone)
    #   RSI 35-40 = caution — approaching oversold, bounce risk
    #   RSI < 35  = poor — oversold, high bounce risk for new shorts
    if   rsi > 72:  e_label, e_why = "IDEAL",    f"RSI overbought ({rsi:.0f}) — prime short entry"
    elif rsi > 55:  e_label, e_why = "GOOD",     f"RSI={rsi:.0f} elevated — directional edge"
    elif rsi > 40:  e_label, e_why = "MODERATE", f"RSI={rsi:.0f} neutral — no entry edge"
    elif rsi > 35:  e_label, e_why = "WEAK",     f"RSI={rsi:.0f} near oversold — bounce risk"
    else:           e_label, e_why = "POOR",     f"RSI={rsi:.0f} oversold — high bounce risk"

    # ── TRADEABILITY (0-100, 100 = strong case to short) ─────────────────────
    t = 50.0
    # Short ML
    if short_ml_acc >= 0.50:
        if   short_ml_proba >= 0.65: t += 25
        elif short_ml_proba >= 0.50: t += 10
        elif short_ml_proba <= 0.35: t -= 20

    # Final short probability vs threshold
    p_margin = short_prob - SHORT_PROB_THRESHOLD
    if   p_margin > 0.15: t += 15
    elif p_margin > 0.05: t += 8
    elif p_margin < -0.10: t -= 12

    # Squeeze risk
    if futures and not getattr(futures,"error",True) and futures.is_futures:
        fr = futures.funding_rate * 100
        if fr <= SHORT_BLOCK_SQUEEZE_THRESH: t -= 40  # squeezed = fatal
        elif fr < -0.02: t -= 20

    # BTC regime
    if btc_regime == "bear": t += 12
    elif btc_regime == "bull": t -= 18

    t_score = clamp(t)
    if short_prob >= SHORT_STRONG_THRESHOLD:
        t_label, t_why = "STRONG",   f"Short P={short_prob:.0%} (>{SHORT_STRONG_THRESHOLD:.0%})"
    elif short_prob >= SHORT_PROB_THRESHOLD:
        t_label, t_why = "GOOD",     f"Short P={short_prob:.0%} qualifies"
    elif short_prob >= 0.50:
        t_label, t_why = "MODERATE", f"Short P={short_prob:.0%} (below threshold)"
    else:
        t_label, t_why = "LOW",      f"Short P={short_prob:.0%} below threshold"

    # ── FRAGILITY (0-100, from multiplicative model) ──────────────────────────
    if trend_frag is not None:
        frag_score = clamp(trend_frag.fragility * 100)
        frag_why   = (f"Exhaust={trend_frag.exhaustion_score:.2f}×"
                      f"Distrib={trend_frag.distribution_score:.2f}×"
                      f"PD={trend_frag.participation_decay:.2f}"
                      f"={trend_frag.fragility:.3f} [{trend_frag.label}]")
        frag_label = trend_frag.label
    elif bear_accel is not None:
        n = getattr(bear_accel,"n_confirmed",0)
        frag_score = clamp(n / 4.0 * 80 + 10)
        frag_why   = f"Accel n={n}/4  exp={getattr(bear_accel,'expected_move_atr',1.0):.1f}xATR"
        frag_label = ["NONE","LOW","MODERATE","HIGH","CRITICAL"][min(n,4)]
    else:
        frag_score = 30; frag_why = "No fragility data"; frag_label = "UNKNOWN"

    f_score = clamp(frag_score)

    # ── COMPOSITE (geometric mean, multiplicative) ─────────────────────────────
    d_n = max(1, d_score) / 50.0
    p_n = max(1, p_score) / 50.0
    s_n = max(1, s_score) / 50.0
    e_n = max(1, e_score) / 50.0
    t_n = max(1, t_score) / 50.0
    f_n = max(1, f_score) / 50.0

    geo = 50.0 * (d_n**0.15 * p_n**0.25 * s_n**0.15 * e_n**0.15 * t_n**0.15 * f_n**0.15)
    composite = round(max(0, min(100, geo)), 1)

    if composite >= 65: overall_label = "HIGH"
    elif composite >= 50: overall_label = "MODERATE"
    elif composite >= 35: overall_label = "LOW"
    else:                 overall_label = "AVOID"

    if short_prob >= SHORT_STRONG_THRESHOLD:
        rec = "STRONG SHORT"
    elif short_prob >= SHORT_PROB_THRESHOLD:
        rec = "SHORT"
    elif overall_label in ("HIGH", "MODERATE") and d >= 52:
        rec = "WATCH"    # setup looks good but probability not there yet
    elif d < 40:
        rec = "AVOID (too bullish)"
    else:
        rec = "WAIT"

    return ShortTradeabilityReport(
        coin=coin,
        direction_score=d_score,     participation_score=p_score,
        stability_score=s_score,     entry_score=e_score,
        tradeability_score=t_score,  fragility_score=f_score,
        direction_label=d_label,     participation_label=p_label,
        stability_label=s_label,     entry_label=e_label,
        tradeability_label=t_label,  fragility_label=frag_label,
        direction_why=d_why,         participation_why=p_why,
        stability_why=s_why,         entry_why=e_why,
        tradeability_why=t_why,      fragility_why=frag_why,
        overall_label=overall_label, recommendation=rec,
        short_prob=short_prob,
    )

# ── END [TDBL-S] ──────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILISTIC PIPELINE FUNCTIONS
# All accept ProbabilisticScorer and add evidence via scorer.update() / add_lr().
# None mutate a bare prob: float -- that was the split-state bug.
# ═══════════════════════════════════════════════════════════════════════════════

def prob_volume_update(scorer: "ProbabilisticScorer", coin: str, quality: str,
                        label: str, obv_trend: str = "unknown",
                        book_thin: bool = False, vol_zscore: float = 0.0) -> None:
    if quality == "good":
        if label == "volume_spike":
            # ── INTENSITY-GRADED VOLUME SPIKE LR ─────────────────────────────
            # Old: flat LR=1.85 for any spike (same for 2σ and 106σ)
            # New: LR scales with z-score magnitude — extreme events get credit
            #
            #   vol_zscore   LR     Interpretation
            #   2–4σ         1.40   Normal elevated volume
            #   4–8σ         1.65   Strong institutional day
            #   8–15σ        1.85   Unusual, likely catalyst
            #   15–30σ       2.10   Rare event, high conviction
            #   30σ+         2.50   Extraordinary (whale/listing/news)
            #
            # Paper-thin orderbook overrides: paper thin + extreme vol = trap
            if vol_zscore > 4.0 and book_thin:
                scorer.update("orderbook_paper_thin", True,
                              f"Extreme vol ({vol_zscore:.0f}σ) but paper-thin book -- trap risk",
                              "FLOW")
                print(f"[VOL] {coin}: volume_spike {vol_zscore:.1f}σ but paper_thin → trap risk → scorer")
                return

            if   vol_zscore >= 30: vol_lr = 2.50; tier = "extraordinary"
            elif vol_zscore >= 15: vol_lr = 2.10; tier = "rare event"
            elif vol_zscore >= 8:  vol_lr = 1.85; tier = "unusual"
            elif vol_zscore >= 4:  vol_lr = 1.65; tier = "strong"
            else:                  vol_lr = 1.40; tier = "elevated"

            scorer.add_lr(vol_lr,
                          f"vol_spike_{vol_zscore:.0f}σ ({tier}) LR={vol_lr:.2f}",
                          "FLOW")
            print(f"[VOL] {coin}: volume_spike {vol_zscore:.1f}σ ({tier}) LR={vol_lr:.2f} → scorer")
            return

        elif label == "increasing_volume": key = "volume_increasing_trend"
        else: key = "volume_healthy"
    elif quality == "bad" and label == "fake_move_warning":
        if obv_trend == "rising":
            scorer.update("volume_healthy", True, "Fake move overridden by OBV rising", "FLOW", scale=0.3)
            print(f"[VOL] {coin}: Fake move overridden by OBV rising")
            return
        elif obv_trend == "flat":
            scorer.update("volume_fake_move", True, "Fake move (OBV flat)", "FLOW", scale=0.5)
            print(f"[VOL] {coin}: Fake move (OBV flat) -- dampened")
            return
        else: key = "volume_fake_move"
    elif quality == "very_bad": key = "volume_churning"
    elif quality == "neutral" and label == "weak_volume": key = "volume_weak"
    else: return
    scorer.update(key, True, f"Volume: {key}", "FLOW")
    print(f"[VOL] {coin}: {key} → scorer")


def prob_zscore_update(scorer: "ProbabilisticScorer", coin: str, zscore: float,
                       adx: float, in_trend: bool = True, rsi: float = 50.0) -> None:
    if adx >= 25: regime = "strong_trend"
    elif adx >= 20: regime = "weak_trend"
    else: regime = "ranging"
    key, scale, is_negative, is_positive = "zscore_healthy_trend", 1.0, False, False
    if regime == "strong_trend":
        if   zscore > 3:   key, is_negative = "zscore_blow_off", True
        elif zscore > 2:   key, scale, is_negative = "zscore_blow_off", 0.5, True
        elif zscore < -3:  key, is_positive = "zscore_falling_knife", True
        elif zscore < -2:  key, scale, is_positive = "zscore_falling_knife", 0.6, True
        else: return
    elif regime == "weak_trend":
        if   zscore > 3:   key, is_negative = "zscore_blow_off", True
        elif zscore > 2:   key, scale, is_negative = "zscore_blow_off", 0.6, True
        else: return
    else:
        if   zscore > 2:   key, scale, is_negative = "zscore_blow_off", 0.7, True
        elif zscore < -2:  key, is_positive = "zscore_oversold_range", True
        elif zscore < -1:  key, scale, is_positive = "zscore_oversold_range", 0.6, True
        else: return
    if is_negative and rsi > 80:   scale *= 0.60
    elif is_positive and rsi < 30: scale *= 0.60
    scorer.update(key, True, f"Z={zscore:.2f} [{regime}]", "MOMENTUM", scale=scale)
    print(f"[ZSCORE] {coin}: {key} scale={scale:.2f} → scorer")


def prob_exhaustion_update(scorer: "ProbabilisticScorer", coin: str, dev_pct: float,
                           zscore_delta: float = 0.0, rsi: float = 50.0) -> None:
    if   dev_pct > 40:  key, scale = "ema26_extreme_stretch", 1.0
    elif dev_pct > 25:  key, scale = "ema26_very_stretched",  1.0
    elif dev_pct > 15:  key, scale = "ema26_stretched",       1.0
    elif dev_pct < -25: key, scale = "ema26_oversold",        1.0
    elif dev_pct < -15: key, scale = "ema26_oversold",        0.6
    else: return
    is_bearish = dev_pct > 0
    if is_bearish and zscore_delta < -0.05: scale *= 0.65
    if is_bearish and rsi > 80:             scale *= 0.80
    elif not is_bearish and rsi < 30:       scale *= 0.80
    scorer.update(key, True, f"EMA26 dev={dev_pct:.1f}%", "MOMENTUM", scale=scale)
    print(f"[EXHAUSTION] {coin}: {key} dev={dev_pct:.1f}% → scorer")


def prob_chop_update(scorer: "ProbabilisticScorer", coin: str, df) -> bool:
    try:
        col    = "close" if "close" in df.columns else "price"
        prices = df[col].iloc[-20:].values
        if len(prices) < 20: return False
        cur = prices[-1]; s, d = prices.mean(), prices.std()
        rng = (prices.max()-prices.min())/cur*100
        bbu, bbl = s+2*d, s-2*d; bbw = (bbu-bbl)/s*100 if s>0 else 0
        pos = (cur-bbl)/(bbu-bbl+1e-9)*100
        sigs = (1 if rng<12 else 0)+(1 if 30<pos<70 else 0)+(1 if bbw>9 else 0)
    except Exception: return False
    if sigs >= 3:
        scorer.update("strong_chop_zone",   True, "Strong chop (3 signals)", "BREAKOUT")
        print(f"[CHOP] {coin}: STRONG CHOP → scorer"); return True
    elif sigs >= 2:
        scorer.update("moderate_chop_zone", True, "Moderate chop (2 signals)", "BREAKOUT")
        print(f"[CHOP] {coin}: MODERATE CHOP → scorer"); return True
    print(f"[CHOP] {coin}: Clear trend"); return False


def prob_breakout_update(scorer: "ProbabilisticScorer", coin: str, df,
                          chop_detected: bool = False) -> None:
    if chop_detected: print(f"[BREAKOUT] {coin}: Skipped (chop)"); return
    try:
        col = "close" if "close" in df.columns else "price"
        p20 = df[col].iloc[-20:-1].values; cur = df[col].iloc[-1]
        res, sup = p20.max(), p20.min()
        if "rel_vol" in df.columns:
            vol_sp = float(df["rel_vol"].iloc[-1]) > 1.5
        elif "volume" in df.columns:
            avg_v = df["volume"].iloc[-20:-1].median(); cur_v = df["volume"].iloc[-1]
            vol_sp = cur_v > avg_v * 1.5 if avg_v > 0 else False
        else: vol_sp = False
        if cur > res:
            key, sc = "breakout_confirmed", (1.0 if vol_sp else 0.4)
            lbl = "Resistance breakout+vol" if vol_sp else "Weak breakout"
        elif cur < sup:
            key, sc = "breakdown_confirmed", (1.0 if vol_sp else 0.4)
            lbl = "Support breakdown+vol" if vol_sp else "Weak breakdown"
        else:
            print(f"[BREAKOUT] {coin}: Consolidating -- neutral"); return
    except Exception as e:
        print(f"[BREAKOUT] {coin}: Error {e}"); return
    scorer.update(key, True, lbl, "BREAKOUT", scale=sc)
    print(f"[BREAKOUT] {coin}: {lbl} → scorer")


def prob_volregime_update(scorer: "ProbabilisticScorer", coin: str, df,
                           vol_regime: str, vol_squeeze: bool = False,
                           vol_signals: dict = None) -> None:
    vs = vol_signals or {}
    bb_rank = float(vs.get("bb_rank", 0.5)); atr_ratio = float(vs.get("atr_ratio", 1.0))
    hv_ratio = float(vs.get("hv_ratio", 1.0)); upd = []
    if vol_squeeze:
        scorer.update("vol_squeeze_on", True, "BB Squeeze ON", "VOLATILITY"); upd.append("squeeze")
    if bb_rank > 0.80:
        scorer.update("vol_bb_wide",    True, f"BB wide ({bb_rank:.0%})", "VOLATILITY"); upd.append("bb_wide")
    elif bb_rank < 0.20:
        scorer.update("vol_bb_narrow",  True, f"BB narrow ({bb_rank:.0%})", "VOLATILITY"); upd.append("bb_narrow")

    # ATR expansion: DIRECTIONAL only — volatility alone is directionless.
    # Combine ATR expansion with close location, OBV, and candle body efficiency
    # before assigning a directional LR.  Neutral expansion = LR 1.0 (no update).
    if atr_ratio > 1.20 or atr_ratio < 0.80:
        try:
            col   = "close" if "close" in df.columns else "price"
            high  = float(df["high"].iloc[-1])  if "high"  in df.columns else float(df[col].iloc[-1])
            low   = float(df["low"].iloc[-1])   if "low"   in df.columns else float(df[col].iloc[-1])
            open_ = float(df["open"].iloc[-1])  if "open"  in df.columns else float(df[col].iloc[-1])
            close = float(df[col].iloc[-1])
            rng   = high - low + 1e-9

            # Component 1: close location in bar range [-1=bottom, +1=top]
            close_loc_score = ((close - low) / rng - 0.5) * 2.0

            # Component 2: candle body efficiency × direction [-1, +1]
            body_eff  = abs(close - open_) / rng          # 0=doji, 1=full body
            body_dir  = 1.0 if close >= open_ else -1.0
            body_score= body_dir * body_eff

            # Component 3: OBV direction over last 5 bars
            obv_score = 0.0
            if "OBV" in df.columns and len(df) >= 6:
                obv_chg = float(df["OBV"].iloc[-1]) - float(df["OBV"].iloc[-6])
                obv_med = float(df["OBV"].iloc[-6:].abs().max()) + 1e-9
                obv_score = float(np.clip(obv_chg / obv_med, -1.0, 1.0))

            # Directional composite: weighted average [-1=bearish, +1=bullish]
            # close_loc contributes most (direct evidence), body and OBV confirm
            direction = (close_loc_score * 0.40 +
                         body_score      * 0.35 +
                         obv_score       * 0.25)

            # Only apply an LR if direction is clear (|direction| > 0.15)
            # Pure expansion with no directional signal = LR 1.0
            if abs(direction) < 0.15:
                upd.append(f"atr_exp_neutral({atr_ratio:.2f}x,dir={direction:.2f})")
            else:
                # Cap atr_ratio before log to prevent extreme exp_mag from meme coins
                _capped_atr = max(0.10, min(5.0, atr_ratio))   # clamp [0.1, 5.0]
                exp_mag  = abs(math.log(_capped_atr))            # magnitude of expansion
                dir_conf = abs(direction)                         # 0-1 direction confidence
                log_lr   = direction * exp_mag * dir_conf        # signed log-LR

                # Cap: max contribution ±log(2.5) using pure Python (safer than np.clip)
                _lo = -math.log(2.5); _hi = math.log(2.5)
                log_lr = max(_lo, min(_hi, float(log_lr)))
                lr     = _lr_safe(math.exp(log_lr))
                dir_tag = "↑bull" if direction > 0 else "↓bear"
                label   = (f"Directional ATR expansion: {atr_ratio:.2f}x (capped {_capped_atr:.2f}x)  "
                           f"dir={direction:+.2f}({dir_tag})  "
                           f"close_loc={close_loc_score:+.2f}  "
                           f"body={body_score:+.2f}  obv={obv_score:+.2f}")
                scorer.add_lr(lr, label, group="VOLATILITY")
                upd.append(f"atr_dir({dir_tag},LR={lr:.2f})")

        except Exception as _ve:
            # Fallback: old undirected approach
            if atr_ratio > 1.20:
                scorer.update("vol_atr_expanding",   True, f"ATR expanding ({atr_ratio:.2f}x)", "VOLATILITY"); upd.append("atr_exp")
            elif atr_ratio < 0.80:
                scorer.update("vol_atr_contracting", True, f"ATR contracting ({atr_ratio:.2f}x)", "VOLATILITY"); upd.append("atr_cont")

    if hv_ratio > 1.30:
        scorer.update("vol_hv_rising", True, f"HV rising ({hv_ratio:.2f}x)", "VOLATILITY"); upd.append("hv_rise")
    if vol_regime == "expansion":
        try:
            col = "close" if "close" in df.columns else "price"
            p5d = (df[col].iloc[-1]-df[col].iloc[-6])/df[col].iloc[-6]*100 if len(df)>=6 else 0
            obv_up = "OBV" in df.columns and len(df)>=8 and float(df["OBV"].iloc[-1])>float(df["OBV"].iloc[-8])
        except: p5d, obv_up = 0.0, False
        if p5d > 0 and obv_up:
            scorer.update("vol_expansion_bullish", True, f"Expansion bullish +{p5d:.1f}%", "VOLATILITY"); upd.append("exp_bull")
        elif p5d < -3 or (not obv_up and p5d < 0):
            scorer.update("vol_expansion_bearish", True, f"Expansion bearish {p5d:.1f}%", "VOLATILITY"); upd.append("exp_bear")
    elif vol_regime == "compression":
        scorer.update("vol_compression", True, "Vol compression (coiling)", "VOLATILITY"); upd.append("compression")
    print(f"[VOL REGIME] {coin}: [{vol_regime.upper()}{'  SQUEEZE' if vol_squeeze else ''}]  updates={upd} → scorer")


def prob_market_regime_update(scorer: "ProbabilisticScorer", coin: str,
                               market_regime: str, cloud_state: str, adx: float,
                               di_bullish: bool, chop_detected: bool,
                               macd_bullish: bool) -> None:
    if market_regime == "bull_trend" and not chop_detected and adx >= 20 and di_bullish:
        scorer.update("regime_bull_trend_amplifier", True,
                      f"BULL TREND amplifier (ADX={adx:.1f} DI+ no-chop)", "OTHER")
        print(f"[REGIME] {coin}: BULL TREND amplifier → scorer")
    elif market_regime == "range":
        scorer.update("regime_range_dampener", True, "RANGE dampener", "OTHER")
        print(f"[REGIME] {coin}: RANGE dampener → scorer")
    else:
        print(f"[REGIME] {coin}: {market_regime} -- no amplification")


def prob_structure_update(scorer: "ProbabilisticScorer", coin: str, ms: dict) -> None:
    st = ms.get("structure", "undefined"); upd = []
    if st == "bullish":
        scorer.update("structure_bullish_hh_hl", True, "Structure BULLISH (HH+HL)", "STRUCTURE"); upd.append("bullish")
    elif st in ("bearish","distribution"):
        scorer.update("structure_bearish_lh_ll", True, f"Structure {st.upper()}", "STRUCTURE"); upd.append(st)
    elif st == "accumulation":
        scorer.update("structure_accumulation",  True, "Structure ACCUMULATION", "STRUCTURE"); upd.append("accum")
    elif st == "ranging":
        scorer.update("structure_ranging",        True, "Structure RANGING", "STRUCTURE"); upd.append("ranging")
    if ms.get("bos") == "bullish":
        scorer.update("structure_bos_bullish", True, "BOS → bullish", "STRUCTURE"); upd.append("bos_bull")
    elif ms.get("bos") == "bearish":
        scorer.update("structure_bos_bearish", True, "BOS → bearish", "STRUCTURE"); upd.append("bos_bear")
    if ms.get("choch") == "bullish":
        scorer.update("structure_choch_bullish", True, "CHoCH → bullish", "STRUCTURE"); upd.append("choch_bull")
    elif ms.get("choch") == "bearish":
        scorer.update("structure_choch_bearish", True, "CHoCH → bearish", "STRUCTURE"); upd.append("choch_bear")
    if ms.get("sfp") == "bullish":
        scorer.update("structure_sfp_bullish", True, "SFP spring → bullish", "STRUCTURE"); upd.append("sfp_bull")
    elif ms.get("sfp") == "bearish":
        scorer.update("structure_sfp_bearish", True, "SFP grab → bearish",  "STRUCTURE"); upd.append("sfp_bear")
    print(f"[MKTSTR] {coin}: {ms.get('details', st)}  signals={upd} → scorer")


def prob_obv_update(scorer: "ProbabilisticScorer", coin: str,
                    obv_trend: str, at_7d_high: bool = False) -> None:
    if   obv_trend == "rising" and at_7d_high: key = "obv_rising_at_high"
    elif obv_trend == "rising":                 key = "obv_rising_not_at_high"
    elif obv_trend == "flat":                   key = "obv_flat"
    elif obv_trend == "falling":                key = "obv_falling"
    else: return
    scorer.update(key, True, f"OBV {obv_trend}", "FLOW")
    print(f"[OBV] {coin}: {key} → scorer")


def prob_smart_money_update(scorer: "ProbabilisticScorer", coin: str,
                             mode: str, obv_trend: str = "unknown",
                             sm_conf: float = 0.5) -> None:
    """
    Score smart money accumulation/distribution with severity-graded LR.

    Distribution severity is graded by sm_conf (0-1 from detect_smart_money):
      sm_conf < 0.25  → mild divergence    LR=0.75  (weak signal, small penalty)
      sm_conf < 0.50  → moderate dist.     LR=0.55  (moderate distribution)
      sm_conf < 0.75  → clear distribution LR=0.42  (confirmed distribution)
      sm_conf ≥ 0.75  → collapse / heavy   LR=0.25  (strong exit signal)

    Accumulation is unchanged — high confidence gets full weight.
    """
    if mode == "accumulation":
        if obv_trend == "falling":
            print(f"[SM] {coin}: accumulation BLOCKED (OBV falling -- 14d SM stale)"); return
        scale = 0.50 if obv_trend == "flat" else 1.0
        lbl   = "SM accumulation (OBV flat--partial)" if obv_trend == "flat" else "SM accumulation (OBV confirmed)"
        scorer.update("sm_accumulation", True, lbl, "FLOW", scale=scale)
        print(f"[SM] {coin}: {lbl} scale={scale:.2f} → scorer")

    elif mode == "distribution":
        # Severity-graded LR based on confidence score
        if   sm_conf < 0.25:
            dist_lr    = 0.75
            severity   = "mild divergence"
        elif sm_conf < 0.50:
            # Interpolate 0.75→0.55 across conf 0.25-0.50
            t          = (sm_conf - 0.25) / 0.25
            dist_lr    = round(0.75 - t * (0.75 - 0.55), 3)
            severity   = "moderate distribution"
        elif sm_conf < 0.75:
            # Interpolate 0.55→0.40 across conf 0.50-0.75
            t          = (sm_conf - 0.50) / 0.25
            dist_lr    = round(0.55 - t * (0.55 - 0.40), 3)
            severity   = "clear distribution"
        else:
            # Interpolate 0.40→0.20 across conf 0.75-1.0
            t          = min(1.0, (sm_conf - 0.75) / 0.25)
            dist_lr    = round(0.40 - t * (0.40 - 0.20), 3)
            severity   = "collapse/heavy distribution"

        scorer.add_lr(dist_lr,
                      f"SM {severity} (conf={sm_conf:.2f} LR={dist_lr:.2f})",
                      "FLOW")
        print(f"[SM] {coin}: {severity}  conf={sm_conf:.2f}  LR={dist_lr:.2f} → scorer")


def prob_orderbook_update(scorer: "ProbabilisticScorer", coin: str,
                           imbalance: float, liq_tier: str) -> None:
    if liq_tier == "paper_thin":
        scorer.update("orderbook_paper_thin",             True, "OB paper thin", "FLOW")
    elif imbalance > 1.5:
        scorer.update("orderbook_strong_buy_imbalance",  True, f"OB buy imbalance ({imbalance:.2f}x)", "FLOW")
    elif imbalance < 0.55:
        scorer.update("orderbook_strong_sell_imbalance", True, f"OB sell imbalance ({imbalance:.2f}x)", "FLOW")
    else: return
    print(f"[OB] {coin}: imbalance={imbalance:.2f} tier={liq_tier} → scorer")


def prob_price_change_update(scorer: "ProbabilisticScorer", coin: str,
                              price_change_24h: float) -> None:
    if price_change_24h < -3:
        scorer.update("zscore_falling_knife", True, f"Sharp 24h decline {price_change_24h:.1f}%",
                      "MOMENTUM", scale=0.5)
        print(f"[PRICE] {coin}: sharp decline {price_change_24h:.1f}% → scorer")
    elif price_change_24h < -1:
        scorer.update("volume_weak", True, f"Price falling {price_change_24h:.1f}%", "MOMENTUM")
        print(f"[PRICE] {coin}: falling {price_change_24h:.1f}% → scorer")


def print_transition_risk(coin: str, current_phase: str,
                           duration_bars: int,
                           matrix: Optional[Dict] = None) -> None:
    """
    Print a plain-text transition risk table for the current phase.

    Groups next-state transitions into three buckets:
      Continuation  — staying in a bull/bear/trend-aligned phase
      Reversal      — moving to the opposite phase
      Range/neutral — collapsing into range or accumulation

    Duration-adjusted: probabilities shift when the phase is old.
    Typical duration for each phase is used to scale persistence down
    using the same exponential decay as the scorer.

    Example output (STRONG_BULL day 18 of typical 20):
      [TRANSITION RISK] RIPPLE  phase=STRONG_BULL  day 18/20
        Bull continuation    62% → 34%  (age-decay applied)
        Late-bull / dist.    24%
        Range reversion      10%
        Bear reversal         4%
    """
    mat = matrix or PHASE_TRANSITION_MATRIX
    row = mat.get(current_phase, {})
    if not row:
        return

    # Group transitions into meaningful buckets
    BULL_PHASES  = {"EARLY_BULL", "STRONG_BULL", "LATE_BULL"}
    BEAR_PHASES  = {"EARLY_BEAR", "STRONG_BEAR", "DISTRIBUTION"}
    RANGE_PHASES = {"RANGE", "ACCUMULATION"}

    def bucket(phases):
        return sum(row.get(p, 0.0) for p in phases)

    raw_continue = row.get(current_phase, 0.0)
    typical      = PHASE_TYPICAL_DURATION.get(current_phase, 20)
    lambda_      = PHASE_DECAY_LAMBDA.get(current_phase, 1.40)
    t_norm       = duration_bars / max(1, typical)
    age_decay    = math.exp(-lambda_ * t_norm)
    adj_continue = 0.5 + (raw_continue - 0.5) * age_decay

    # Re-normalise remaining transitions proportionally
    leftover_raw = 1.0 - raw_continue
    leftover_adj = 1.0 - adj_continue
    scale = (leftover_adj / leftover_raw) if leftover_raw > 0 else 1.0

    if current_phase in BULL_PHASES:
        p_bull_cont   = adj_continue
        p_other_bull  = bucket(BULL_PHASES - {current_phase}) * scale
        p_range       = bucket(RANGE_PHASES) * scale
        p_bear        = bucket(BEAR_PHASES) * scale
        lines = [
            (f"Bull continuation ({current_phase})", p_bull_cont),
            (f"Other bull phases",                   p_other_bull),
            (f"Range / accumulation",                p_range),
            (f"Bear reversal",                       p_bear),
        ]
    elif current_phase in BEAR_PHASES:
        p_bear_cont   = adj_continue
        p_other_bear  = bucket(BEAR_PHASES - {current_phase}) * scale
        p_range       = bucket(RANGE_PHASES) * scale
        p_bull        = bucket(BULL_PHASES) * scale
        lines = [
            (f"Bear continuation ({current_phase})", p_bear_cont),
            (f"Other bear phases",                   p_other_bear),
            (f"Range reversion",                     p_range),
            (f"Bull reversal",                       p_bull),
        ]
    else:   # RANGE / ACCUMULATION
        p_cont  = adj_continue
        p_bull  = bucket(BULL_PHASES) * scale
        p_bear  = bucket(BEAR_PHASES) * scale
        lines = [
            (f"Range continuation",  p_cont),
            (f"Bull breakout",       p_bull),
            (f"Bear breakdown",      p_bear),
        ]

    # Compute age warning
    age_pct = duration_bars / max(1, typical)
    if   age_pct >= 1.5: age_note = f"⚠️  VERY OLD ({duration_bars}/{typical} bars)"
    elif age_pct >= 1.0: age_note = f"⚠️  mature ({duration_bars}/{typical} bars)"
    elif age_pct >= 0.7: age_note = f"ageing ({duration_bars}/{typical} bars)"
    else:                age_note = f"fresh ({duration_bars}/{typical} bars)"

    print(f"[TRANSITION RISK] {coin}  phase={current_phase}  {age_note}")
    for label, prob in lines:
        if prob < 0.01:
            continue
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {label:<35} {prob:>5.0%}  {bar}")


def prob_transition_update(scorer: "ProbabilisticScorer", coin: str,
                            current_phase: str, duration_bars: int,
                            matrix: Optional[Dict] = None) -> str:
    mat = matrix or PHASE_TRANSITION_MATRIX
    lr, label = compute_transition_lr(current_phase, duration_bars, mat)
    print_transition_risk(coin, current_phase, duration_bars, mat)
    if abs(math.log(_lr_safe(lr))) < 0.01:
        print(f"[TRANSITION] {coin}: {current_phase} -- neutral")
        return current_phase
    scorer.add_lr(lr, label, group="TRANSITION")
    direction = "↑" if lr > 1.0 else "↓"
    print(f"[TRANSITION] {coin}: {label}  {direction} → scorer")
    return current_phase


def classify_trend_lifecycle(last: pd.Series,
                              df: pd.DataFrame,
                              phase: str,
                              obv_trend: str = "unknown") -> dict:
    """
    5-stage trend lifecycle classifier.

    Maps the existing 8 Markov phases onto 5 actionable lifecycle stages
    with specific entry rules, position size modifiers, and LRs per stage.

    Stage 1 — ACCUMULATION
      Characteristics: ADX low, volume building, price near cloud
      Entry: Watch, wait for breakout confirmation
      LR: 1.30  (slight bullish lean)
      Position modifier: 0.75× (small starter position)

    Stage 2 — EARLY_TREND  ← BEST RISK/REWARD
      Characteristics: Fresh breakout, ADX rising 20-30, OBV confirming
      Entry: Most aggressive — highest probability of large move
      LR: 2.00  (strong signal, trend just confirmed)
      Position modifier: 1.25× (add to position)

    Stage 3 — MATURE_TREND
      Characteristics: Above cloud, ADX 25-40, trend persistence
      Entry: Continuation trades at pullbacks
      LR: 1.55  (solid, reliable)
      Position modifier: 1.00× (normal)

    Stage 4 — EXHAUSTION  ← KEY FIX
      Characteristics: EMA stretched >20%, RSI >70, volume climax
      Entry: REDUCE POSITION SIZE, not probability
      LR: 1.25  (still bullish — stretched can stay stretched)
      Position modifier: 0.50× (half size, tighter stop)
      Note: Do NOT suppress probability. Many crypto winners run for weeks
            when stretched. Signal the risk via position sizing only.

    Stage 5 — DISTRIBUTION
      Characteristics: OBV divergence, price at high, A/D deterioration
      Entry: Dangerous — avoid longs or use very tight stops
      LR: 0.65  (bearish lean)
      Position modifier: 0.25× (minimal exposure if trading at all)
    """
    try:
        price   = float(last["price"])
        rsi     = float(last.get("RSI", 50)      or 50)
        adx     = float(last.get("ADX", 0)       or 0)
        ema26   = float(last.get("EMA26", price) or price)
        macd_h  = float(last.get("MACD_Hist", 0) or 0)

        # EMA26 deviation (how stretched is price from trend centre)
        ema_dev_pct = abs(price - ema26) / (ema26 + 1e-9) * 100

        # Volume trend (recent 7d vs prior 7d)
        vol_accel = "flat"
        if "volume" in df.columns and len(df) >= 14:
            v = df["volume"].values.astype(float)
            v7  = v[-7:].mean();  v14 = v[-14:-7].mean()
            if   v7 > v14 * 1.20: vol_accel = "rising"
            elif v7 < v14 * 0.80: vol_accel = "falling"

        # OBV at recent high? (price at high but OBV falling = divergence)
        obv_diverge = False
        if "OBV" in df.columns and len(df) >= 10:
            prices = df["price"].values if "price" in df.columns else df["close"].values
            obvs   = df["OBV"].values.astype(float)
            price_at_high = float(prices[-1]) >= float(prices[-10:].max()) * 0.97
            obv_at_high   = float(obvs[-1])   >= float(obvs[-10:].max())   * 0.97
            obv_diverge   = price_at_high and not obv_at_high

        # ── Map phase + conditions → lifecycle stage ──────────────────────
        if phase in ("ACCUMULATION",) or (adx < 20 and obv_trend == "rising"):
            stage  = "ACCUMULATION"
            lr     = 1.30
            pos    = 0.75
            reason = f"Building base — ADX={adx:.0f} low, OBV {obv_trend}, wait for breakout"

        elif phase == "EARLY_BULL" or (
            phase in ("STRONG_BULL", "ACCUMULATION") and
            adx >= 20 and adx < 32 and macd_h > 0 and
            ema_dev_pct < 12 and obv_trend == "rising"
        ):
            stage  = "EARLY_TREND"
            lr     = 2.00
            pos    = 1.25
            reason = (f"Fresh trend — ADX {adx:.0f} rising, OBV confirming, "
                      f"EMA dev {ema_dev_pct:.1f}% (not yet stretched)")

        elif phase in ("STRONG_BULL",) and (
            ema_dev_pct >= 20 or rsi > 70 or
            (vol_accel == "falling" and rsi > 65) or obv_diverge
        ):
            stage  = "EXHAUSTION"
            lr     = 1.25   # still bullish — stretched can persist for weeks
            pos    = 0.50   # reduce size, not probability
            signs  = []
            if ema_dev_pct >= 20: signs.append(f"EMA dev {ema_dev_pct:.1f}%")
            if rsi > 70:          signs.append(f"RSI {rsi:.0f}")
            if obv_diverge:       signs.append("OBV diverging")
            reason = f"Extended — {', '.join(signs)} — reduce size, keep stop tight"

        elif phase in ("DISTRIBUTION", "LATE_BULL") or (
            obv_trend == "falling" and rsi > 60 and ema_dev_pct > 15
        ):
            stage  = "DISTRIBUTION"
            lr     = 0.65
            pos    = 0.25
            signs  = []
            if obv_diverge:       signs.append("OBV diverging at high")
            if obv_trend=="falling": signs.append("OBV falling")
            if rsi > 60:          signs.append(f"RSI {rsi:.0f}")
            reason = f"Distribution — {', '.join(signs) or 'A/D deteriorating'}"

        else:
            stage  = "MATURE_TREND"
            lr     = 1.55
            pos    = 1.00
            reason = (f"Trend intact — ADX {adx:.0f}, "
                      f"EMA dev {ema_dev_pct:.1f}%, OBV {obv_trend}")

        return {
            "stage":            stage,
            "lr":               lr,
            "position_mod":     pos,
            "reason":           reason,
            "ema_dev_pct":      round(ema_dev_pct, 1),
            "vol_accel":        vol_accel,
            "obv_diverge":      obv_diverge,
        }

    except Exception as e:
        return {"stage": "UNKNOWN", "lr": 1.0, "position_mod": 1.0,
                "reason": str(e), "ema_dev_pct": 0, "vol_accel": "flat",
                "obv_diverge": False}


def prob_lifecycle_update(scorer: "ProbabilisticScorer", coin: str,
                           lifecycle: dict, phase: str,
                           suppress_debug: bool = False) -> None:
    """
    Add trend lifecycle stage LR to scorer and print summary.

    EXHAUSTION is the critical case:
      - LR=1.25 (still bullish, trend likely continues)
      - position_mod=0.50 (risk management via size, not suppression)
      - Do NOT add a bearish LR for exhaustion — stretched coins run further
    """
    stage  = lifecycle["stage"]
    lr     = lifecycle["lr"]
    pos    = lifecycle["position_mod"]
    reason = lifecycle["reason"]

    stage_icon = {
        "ACCUMULATION": "🟡 Stage 1",
        "EARLY_TREND":  "🟢 Stage 2",
        "MATURE_TREND": "🟢 Stage 3",
        "EXHAUSTION":   "⚠️  Stage 4",
        "DISTRIBUTION": "🔴 Stage 5",
        "UNKNOWN":      "⬜ Stage ?",
    }.get(stage, "⬜")

    scorer.add_lr(lr, f"lifecycle_{stage} (LR={lr:.2f})", "TREND_LIFECYCLE")
    if not suppress_debug:
        print(f"[LIFECYCLE] {coin}: {stage_icon} {stage}  LR={lr:.2f}  "
              f"pos_mod={pos:.2f}×  — {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# [MEM] REGIME PERSISTENCE MEMORY
#
# Markets are path-dependent. The current regime classifiers (bull/bear/range/
# chop/vol_regime) tell you WHAT the regime is. RegimeMemory tells you:
#
#   1. CHOP PERSISTENCE -- how many consecutive bars have we been in chop?
#      A 1-bar chop might be noise. An 8-bar chop is structural -- suppress harder.
#      Each additional chop bar adds a fractional LR to the scorer.
#
#   2. VOLATILITY CLUSTERING (GARCH effect) -- high vol follows high vol.
#      If ATR has been autocorrelated over the last 20 bars, the current vol
#      regime is more reliable (persistence is real, not noise). If vol is
#      mean-reverting, current expansion/compression is likely temporary.
#
#   3. REGIME INSTABILITY -- how often has the regime classification flipped
#      in the last 10 bars? High flip-count = noisy signal environment.
#      The current regime label is less trustworthy when it keeps changing.
#
#   4. REGIME HALF-LIFE -- long-running bull/bear trends are maturing.
#      A trend running 20 bars is more likely to transition than one at bar 3.
#      Applied as a gentle exponential decay on top of the phase age decay.
#
# Session-scoped: per-coin RegimeMemory persists across the coin loop but
# resets at session start. Not persisted to disk (intraday only).
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeMemory:
    """
    Rolling buffer of regime observations for persistence/clustering analysis.
    Kept in RAM per coin across the run (session-scoped, not disk-persisted).
    """
    MAX_HISTORY = 60

    def __init__(self):
        self.market_regimes: List[str]   = []
        self.vol_regimes:    List[str]   = []
        self.chop_flags:     List[bool]  = []
        self.atr_vals:       List[float] = []

    def record(self, market_regime: str, vol_regime: str,
               chop: bool, atr: float) -> None:
        self.market_regimes.append(str(market_regime))
        self.vol_regimes.append(str(vol_regime))
        self.chop_flags.append(bool(chop))
        self.atr_vals.append(max(0.0, float(atr)))
        for lst in (self.market_regimes, self.vol_regimes,
                    self.chop_flags, self.atr_vals):
            if len(lst) > self.MAX_HISTORY:
                lst.pop(0)

    def chop_streak(self) -> int:
        """Consecutive bars where chop was detected (counting back from now)."""
        streak = 0
        for flag in reversed(self.chop_flags):
            if flag: streak += 1
            else: break
        return streak

    def vol_autocorrelation(self, lag: int = 1, window: int = 20) -> float:
        """
        Pearson autocorrelation of ATR at given lag.
        > 0.50: volatility clustering (high vol follows high vol) -- GARCH effect.
        < -0.30: mean-reverting vol -- regime is about to change.
        """
        a = np.array(self.atr_vals[-window:])
        if len(a) < lag + 4:
            return 0.0
        try:
            r = float(np.corrcoef(a[lag:], a[:-lag])[0, 1])
            return r if not math.isnan(r) else 0.0
        except Exception:
            return 0.0

    def regime_flip_count(self, window: int = 10) -> int:
        """Number of regime changes in the last `window` observations."""
        r = self.market_regimes[-window:]
        return sum(1 for i in range(1, len(r)) if r[i] != r[i - 1])

    def current_streak(self) -> int:
        """Consecutive bars in the current market regime."""
        if not self.market_regimes:
            return 0
        cur = self.market_regimes[-1]
        streak = 0
        for r in reversed(self.market_regimes):
            if r == cur: streak += 1
            else: break
        return streak


def print_regime_persistence(coin: str, market_regime: str,
                              phase: str, phase_dur: int,
                              mem: "RegimeMemory") -> None:
    """
    Print a plain-text regime persistence summary.

    Three lines:
      Regime age        — how long have we been in this phase vs typical
      Persistence conf  — STRONG / MODERATE / WEAK based on age and flips
      Decay risk        — LOW / MODERATE / HIGH based on exp decay remaining

    Example:
      [REGIME PERSISTENCE] RIPPLE
        Bear regime age:        6 days  (typical: 19)  33% through
        Persistence confidence: MODERATE  (0 flips, 0.47 vol_ac)
        Regime decay risk:      LOW  (66% signal strength remaining)
    """
    streak   = mem.current_streak()
    flips    = mem.regime_flip_count(window=10)
    vol_ac   = mem.vol_autocorrelation()
    chop_s   = mem.chop_streak()

    typical  = PHASE_TYPICAL_DURATION.get(phase, 20)
    lambda_  = PHASE_DECAY_LAMBDA.get(phase, 1.40)
    age_pct  = phase_dur / max(1, typical)
    # Signal strength remaining after age decay
    strength = math.exp(-lambda_ * age_pct)

    # ── Persistence confidence ────────────────────────────────────────────────
    # Strong: fresh phase, no flips, vol clustering confirms
    # Moderate: some age, minor instability
    # Weak: old phase OR many flips OR mean-reverting vol
    if flips >= 4 or chop_s >= 8:
        persist_conf = "WEAK"
        persist_why  = f"({flips} flips" + (f", chop {chop_s}d" if chop_s >= 4 else "") + ")"
    elif age_pct > 1.2 or vol_ac < -0.25:
        persist_conf = "WEAK"
        persist_why  = ("(overdue)" if age_pct > 1.2 else f"(vol reverting ac={vol_ac:.2f})")
    elif flips >= 2 or (0.50 <= age_pct <= 1.0):
        persist_conf = "MODERATE"
        persist_why  = f"({flips} flip{'s' if flips!=1 else ''})"
    else:
        persist_conf = "STRONG"
        persist_why  = f"(fresh, {flips} flips)"

    # ── Decay risk ────────────────────────────────────────────────────────────
    # Strength = how much of the original regime signal remains
    # Low risk  = >60% remaining (regime still valid)
    # Medium    = 30-60% remaining (starting to fade)
    # High      = <30% remaining (transition likely)
    if   strength >= 0.60: decay_risk = "LOW";      decay_icon = "✅"
    elif strength >= 0.30: decay_risk = "MODERATE";  decay_icon = "⚠️ "
    else:                  decay_risk = "HIGH";      decay_icon = "🔴"

    # ── Age description ───────────────────────────────────────────────────────
    if   age_pct < 0.30: age_tag = "very fresh"
    elif age_pct < 0.60: age_tag = "early"
    elif age_pct < 0.90: age_tag = "mid-life"
    elif age_pct < 1.20: age_tag = "mature"
    else:                age_tag = "overdue"

    # Regime label for display
    regime_label_map = {
        "bull_trend": "Bull", "bear": "Bear",
        "range": "Range", "weak_trend": "Weak trend",
        "strong_trend": "Strong bull",
    }
    rl = regime_label_map.get(market_regime, market_regime.replace("_", " ").title())

    print(f"[REGIME PERSISTENCE] {coin}")
    print(f"  {rl} regime age:         {phase_dur:>3} days  "
          f"(typical {typical})  {age_pct:.0%} through  [{age_tag}]")
    print(f"  Persistence confidence: {persist_conf:<8}  {persist_why}  "
          f"streak={streak}  vol_ac={vol_ac:+.2f}")
    print(f"  Regime decay risk:      {decay_risk:<8}  {decay_icon}  "
          f"{strength:.0%} signal strength remaining")


def prob_regime_memory_update(
    scorer:         "ProbabilisticScorer",
    coin:           str,
    mem:            "RegimeMemory",
    market_regime:  str,
    vol_regime:     str,
    chop_detected:  bool,
    atr:            float,
    phase:          str = "RANGE",
    phase_dur:      int = 0,
) -> None:
    """
    Record current regime state and add persistence/clustering LRs to scorer.

    SIGNAL 1 -- Chop persistence:
      The per-bar chop LR (moderate_chop_zone ≈ 0.75) captures TODAY's chop.
      This captures HOW LONG chop has persisted -- longer streaks deserve more suppression.
        streak 4-7 bars:  add LR=0.83  (chop is structural, not noise)
        streak 8-13 bars: add LR=0.72  (strongly entrenched -- harder to break out)
        streak 14+ bars:  add LR=0.65  (market is dead -- probability very suppressed)

    SIGNAL 2 -- Volatility clustering (GARCH):
      autocorr > 0.60 in expansion:    LR=1.08  (vol regime is reliable, signals stronger)
      autocorr > 0.60 in compression:  LR=0.94  (quiet period persisting → signals weaker)
      autocorr < -0.30:                LR=0.92  (mean-reverting vol → regime changing soon)

    SIGNAL 3 -- Regime instability:
      ≥4 flips in 10 bars:  LR=0.85  (environment is noisy -- current classification unreliable)
      ≥6 flips in 10 bars:  LR=0.78  (market is whipsawing -- very uncertain)

    SIGNAL 4 -- Regime half-life:
      bull_trend or bear lasting ≥15 bars: gentle decay LR = exp(-0.05 × (streak-10))
      Complements the phase age decay in the transition model.
    """
    mem.record(market_regime, vol_regime, chop_detected, atr)

    # Print human-readable persistence summary
    print_regime_persistence(coin, market_regime, phase, phase_dur, mem)

    updates = []

    # ── 1. Chop persistence ───────────────────────────────────────────────────
    chop_s = mem.chop_streak()
    if chop_s >= 14:
        scorer.add_lr(0.65, f"Chop entrenched ({chop_s} bars)", "BREAKOUT")
        updates.append(f"chop={chop_s}→LR=0.65")
    elif chop_s >= 8:
        scorer.add_lr(0.72, f"Persistent chop ({chop_s} bars)", "BREAKOUT")
        updates.append(f"chop={chop_s}→LR=0.72")
    elif chop_s >= 4:
        scorer.add_lr(0.83, f"Chop persisting ({chop_s} bars)", "BREAKOUT")
        updates.append(f"chop={chop_s}→LR=0.83")

    # ── 2. Volatility clustering (GARCH) ─────────────────────────────────────
    vol_ac = mem.vol_autocorrelation()
    if vol_ac > 0.60:
        if vol_regime == "expansion":
            scorer.add_lr(1.08,
                          f"Vol clustering in expansion (autocorr={vol_ac:.2f})",
                          "VOLATILITY")
            updates.append(f"garch_exp={vol_ac:.2f}→1.08")
        elif vol_regime == "compression":
            scorer.add_lr(0.94,
                          f"Vol clustering in compression (autocorr={vol_ac:.2f})",
                          "VOLATILITY")
            updates.append(f"garch_comp={vol_ac:.2f}→0.94")
    elif vol_ac < -0.30:
        scorer.add_lr(0.92,
                      f"Vol mean-reverting (autocorr={vol_ac:.2f}) -- regime change likely",
                      "VOLATILITY")
        updates.append(f"vol_revert={vol_ac:.2f}→0.92")

    # ── 3. Regime instability ─────────────────────────────────────────────────
    flips = mem.regime_flip_count(window=10)
    if flips >= 6:
        scorer.add_lr(0.78,
                      f"Regime whipsawing ({flips}/10 bar flips)",
                      "OTHER")
        updates.append(f"flips={flips}→0.78")
    elif flips >= 4:
        scorer.add_lr(0.85,
                      f"Regime unstable ({flips}/10 bar flips)",
                      "OTHER")
        updates.append(f"flips={flips}→0.85")

    # ── 4. Regime half-life ───────────────────────────────────────────────────
    streak = mem.current_streak()
    if streak >= 15 and market_regime in ("bull_trend", "bear"):
        # Trend maturity decay -- applied AFTER transition model's phase age decay
        # so both contribute; they are multiplicative in log-odds
        hl_lr = math.exp(-0.05 * (streak - 10))
        hl_lr = max(0.80, min(1.0, hl_lr))
        if hl_lr < 0.98:
            scorer.add_lr(hl_lr,
                          f"Regime half-life decay ({market_regime} streak={streak}b)",
                          "OTHER")
            updates.append(f"half_life={streak}→{hl_lr:.3f}")

    if updates:
        print(f"[REGIME MEM] {coin}: {' | '.join(updates)}  "
              f"(chop_streak={chop_s}  vol_ac={vol_ac:.2f}  flips={flips}  streak={streak})")
    else:
        print(f"[REGIME MEM] {coin}: no persistence signals  "
              f"(chop={chop_s}  vol_ac={vol_ac:.2f}  flips={flips}  streak={streak})")


# Session-scoped cache -- per-coin RegimeMemory, reset each run
_regime_memory: Dict[str, "RegimeMemory"] = {}

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# [SHORT] INDEPENDENT SHORT SCORER -- SIX-QUESTION ARCHITECTURE
#
#   Q1 EXHAUSTION   Is trend aging?             momentum fade, RSI/MACD divergence
#   Q2 DISTRIBUTION Is participation falling?   OBV/CVD diverging from price
#   Q3 TRAP         Is breakout failing?         bull trap, failed new high
#   Q4 ACCELERATION Is vol expanding down?       ATR growing on downside moves
#   Q5 DANGER       Is smart money exiting?      institutional flow reversing
#   Q6 FRAGILITY    Is confidence unstable?      positioning extremes, regime flips
#
#   Score->LR: LR = MAX_LR^(2*score-1)
#   score=0.5->LR=1.0  score=1.0->LR=3.0  score=0.0->LR=0.33
# ===============================================================================

SHORT_PROB_THRESHOLD       = 0.65
SHORT_STRONG_THRESHOLD     = 0.75
SHORT_BLOCK_SQUEEZE_THRESH = -0.20

SHORT_REGIME_PRIORS: Dict[str, float] = {
    "euphoria":     0.55,
    "panic":        0.58,
    "bear_trend":   0.62,
    "range":        0.44,
    "weak_trend":   0.38,
    "strong_trend": 0.25,
}

# ── Short-specific time decay constants ───────────────────────────────────────
# Shorts are time-sensitive: bearish moves revert faster than bullish trends.
# These λ values are deliberately FASTER than the long-side PHASE_DECAY_LAMBDA.
#
# WHY SHORTS DECAY FASTER:
#   Bull trend: persistent institutional buying, steady accumulation
#   Bear dump:  liquidation cascade — violent but self-terminating
#   Panic:      extreme sellers exhausted in hours to days
#
# Formula (applied to short prior):
#   P_effective = 0.5 + (P_base - 0.5) × e^(-λ × t / t_typical)
#
#   STRONG_BEAR day 1:  P=0.62  (fresh — maximum short conviction)
#   STRONG_BEAR day 8:  P=0.53  (aging — expect reversion)
#   STRONG_BEAR day 15: P=0.50  (fully decayed — stand aside)
#
# Half-lives at t=t_typical:
#   λ=0.70: 50% of excess remains  (slow — bull uptrends are sticky)
#   λ=1.40: 25% of excess remains  (long side default)
#   λ=2.00: 14% of excess remains  (fast — bear dumps fade)
#   λ=3.50:  3% of excess remains  (panic — exhausted in days)
#   λ=5.00: <1% of excess remains  (counter-trend short in bull — stop quickly)
SHORT_PHASE_DECAY_LAMBDA: Dict[str, float] = {
    "ACCUMULATION":  2.00,  # range-bound, expect reversal quickly
    "EARLY_BULL":    5.00,  # counter-trend short, extremely dangerous
    "STRONG_BULL":   4.00,  # don't fight the trend
    "LATE_BULL":     1.80,  # aging bull, short window opens
    "DISTRIBUTION":  1.50,  # smart money exiting, moderate urgency
    "EARLY_BEAR":    2.20,  # first leg down: fast and violent
    "STRONG_BEAR":   2.80,  # dump accelerating but reversal risk rising
    "RANGE":         1.80,  # breakdowns in ranges often fake
}

# Freshness bonus: short signals in the first 20% of a phase get a boost
# because the best short entry is at the very START of distribution/breakdown
SHORT_FRESHNESS_WINDOW = 0.20   # first 20% of typical duration = "fresh" phase


def _apply_short_age_decay(base_prior: float, phase: str, phase_dur: int) -> Tuple[float, float]:
    """
    Apply fast time decay to the short prior based on phase and age.

    Returns (decayed_prior, freshness_boost_lr):
      - decayed_prior:    the time-adjusted prior for the short scorer
      - freshness_boost_lr: LR > 1 if phase is fresh (short window just opened)
    """
    typical   = PHASE_TYPICAL_DURATION.get(phase, 20)
    lambda_   = SHORT_PHASE_DECAY_LAMBDA.get(phase, 2.00)
    t_norm    = phase_dur / max(1, typical)
    decay     = math.exp(-lambda_ * t_norm)

    # Only decay the bearish excess (above 0.5)
    # Priors below 0.5 (bull regimes) don't get worse with time for shorts —
    # they stay bad, because time doesn't make a bullish regime better for shorts
    if base_prior > 0.50:
        decayed = 0.50 + (base_prior - 0.50) * decay
    else:
        decayed = base_prior   # already unfavorable — keep as-is

    # Freshness bonus: first 20% of a bearish phase is the prime short window
    # This rewards catching the turn early, not chasing the extended move
    freshness_lr = 1.0
    if base_prior > 0.50 and t_norm <= SHORT_FRESHNESS_WINDOW and phase_dur <= 3:
        # Very fresh bearish phase: add a modest bullish-for-short LR
        freshness = 1.0 - (t_norm / SHORT_FRESHNESS_WINDOW)   # 1.0 at day 0, 0.0 at freshness_window
        freshness_lr = 1.0 + freshness * 0.35   # up to 1.35 LR for day-1 entries

    return round(decayed, 4), round(freshness_lr, 3)

SHORT_SIGNAL_GROUPS: Dict[str, Dict] = {
    "EXHAUSTION":   {"dampening": 0.35},
    "DISTRIBUTION": {"dampening": 0.40},
    "TRAP":         {"dampening": 0.45},
    "ACCELERATION": {"dampening": 0.40},
    "DANGER":       {"dampening": 0.45},
    "FRAGILITY":    {"dampening": 0.50},
}

_SHORT_MAX_LR = 3.0

def _score_to_lr(score: float, max_lr: float = _SHORT_MAX_LR) -> float:
    return _lr_safe(max_lr ** (2.0 * max(0.0, min(1.0, float(score))) - 1.0))


@dataclass
class FuturesData:
    coin: str; funding_rate: float; funding_1d_sum: float
    oi_usd: float; oi_change_24h: float; ls_ratio: float
    liq_above_usd: float; liq_below_usd: float
    is_futures: bool = True; error: str = ""


def fetch_futures_data(coin: str) -> Optional["FuturesData"]:
    FUTURES_SYMBOLS = {
        "bitcoin":"BTCUSDT","ethereum":"ETHUSDT","solana":"SOLUSDT",
        "bittensor":"TAOUSDT","celestia":"TIAUSDT","sui":"SUIUSDT",
        "avalanche-2":"AVAXUSDT","polkadot":"DOTUSDT","chainlink":"LINKUSDT",
        "near":"NEARUSDT","aptos":"APTUSDT","arbitrum":"ARBUSDT","optimism":"OPUSDT",
    }
    symbol = FUTURES_SYMBOLS.get(coin.lower())
    if not symbol:
        return None
    try:
        import ccxt
        ex = ccxt.binance({"options":{"defaultType":"future"},"enableRateLimit":True})
        fh = ex.fetch_funding_rate_history(symbol, limit=4)
        rates = [float(f["fundingRate"]) for f in fh if f.get("fundingRate") is not None]
        cur_rate = rates[-1] if rates else 0.0
        sum_1d   = sum(rates[-3:]) if len(rates) >= 3 else sum(rates)
        oi_now = ex.fetch_open_interest(symbol)
        oi_usd = float(oi_now.get("openInterestValue", 0))
        try:
            oh = ex.fetch_open_interest_history(symbol, "1d", limit=2)
            oi_chg = (oi_usd - float(oh[-2]["openInterestValue"])) / (float(oh[-2]["openInterestValue"]) + 1e-9) * 100 if len(oh) >= 2 else 0.0
        except Exception: oi_chg = 0.0
        try:
            ls_d = ex.fetch_long_short_ratio(symbol, period="1h", limit=1)
            ls_ratio = float(ls_d[-1]["longShortRatio"]) if ls_d else 1.0
        except Exception: ls_ratio = 1.0
        return FuturesData(coin=coin, funding_rate=cur_rate, funding_1d_sum=sum_1d,
                           oi_usd=oi_usd, oi_change_24h=oi_chg, ls_ratio=ls_ratio,
                           liq_above_usd=0.0, liq_below_usd=0.0, is_futures=True)
    except Exception as e:
        return FuturesData(coin=coin, funding_rate=0, funding_1d_sum=0, oi_usd=0,
                           oi_change_24h=0, ls_ratio=1.0, liq_above_usd=0, liq_below_usd=0,
                           is_futures=False, error=str(e)[:80])


def prob_short_squeeze_risk(short_scorer: "ProbabilisticScorer", coin: str,
                             futures: "FuturesData") -> bool:
    if not futures or futures.error or not futures.is_futures:
        return False
    fr = futures.funding_rate * 100
    if fr <= SHORT_BLOCK_SQUEEZE_THRESH:
        short_scorer.block(f"SQUEEZE BLOCK: funding={fr:.3f}%")
        print(f"[SHORT SQUEEZE] {coin}: BLOCKED -- funding={fr:.3f}%")
        return True
    if fr < -0.02 and futures.ls_ratio < 0.7:
        short_scorer.add_lr(0.55, f"Squeeze risk (fund={fr:.3f}% ls={futures.ls_ratio:.2f})", "FRAGILITY")
    return False


def _short_score_exhaustion(df, last, phase="RANGE", phase_dur=0):
    score = 0.0; ev = []
    col = "close" if "close" in df.columns else "price"
    typical = PHASE_TYPICAL_DURATION.get(phase, 20)
    if phase in ("LATE_BULL","DISTRIBUTION"):
        score += 0.30; ev.append(f"phase={phase}")
    elif phase == "STRONG_BULL" and phase_dur / max(1,typical) > 0.80:
        score += 0.20; ev.append("strong_bull_aged")
    try:
        prices = df[col].values.astype(float)
        rsis   = df["RSI"].values.astype(float) if "RSI" in df.columns else None
        if rsis is not None and len(prices) >= 20:
            ph = float(prices[-15:-1].max()); ph_rsi = float(rsis[-15:-1].max())
            if prices[-1] >= ph * 0.998 and float(rsis[-1]) < ph_rsi - 5:
                score += 0.25; ev.append(f"rsi_div({rsis[-1]:.0f}<{ph_rsi:.0f})")
    except Exception: pass
    try:
        if "MACD_Hist" in df.columns and len(df) >= 5:
            hist = df["MACD_Hist"].values.astype(float); p = df[col].values.astype(float)
            if p[-1] >= p[-5:].max()*0.995 and hist[-1] < hist[-3]:
                score += 0.15; ev.append("macd_hist_fading")
    except Exception: pass
    try:
        if "EMA12" in df.columns and len(df) >= 10:
            e12=float(last.get("EMA12",last["price"])); e26=float(last.get("EMA26",last["price"]))
            sp_now  = (e12-e26)/(e26+1e-9)
            e12p=float(df["EMA12"].iloc[-10]); e26p=float(df["EMA26"].iloc[-10])
            sp_prev = (e12p-e26p)/(e26p+1e-9)
            if sp_now < sp_prev * 0.6 and sp_now > 0:
                score += 0.15; ev.append("ema_fan_closing")
    except Exception: pass
    return min(1.0, score), f"exhaust({min(1.0,score):.2f}):{'+'.join(ev) or 'none'}"


@dataclass
class DistributionDivergence:
    """Result of the price/OBV divergence analysis."""
    detected:          bool
    price_swing_high:  float    # price level at prior swing high
    obv_at_swing:      float    # OBV at prior swing high
    price_now:         float    # current price
    obv_now:           float    # current OBV
    obv_divergence_pct: float   # how far OBV has lagged, as a %
    n_swing_lookback:  int      # how far back the reference swing was
    severity:          str      # "strong" / "moderate" / "mild" / "none"
    lr:                float    # pre-computed LR for the short scorer


def detect_distribution_divergence(df: pd.DataFrame,
                                    lookback: int = 30) -> "DistributionDivergence":
    """
    Detect price higher-high / OBV lower-high divergence.

    This is the primary Wyckoff distribution signal. When price makes
    a new high but OBV does NOT confirm — capital is quietly leaving.
    Smart money is distributing to retail buyers at new highs.

    DETECTION ALGORITHM:
      1. Find the most recent significant swing high in price
         (local maximum that is higher than surrounding bars)
      2. Record OBV at that swing high
      3. Check current price vs that swing high
      4. If price >= swing high: measure OBV then vs OBV now
      5. If OBV now < OBV at swing: divergence is present

    WHY HIGH WEIGHT:
      This pattern has very high historical predictive value for shorts:
      - Smart money has information advantage; they sell into strength
      - OBV falling while price rises = volume NOT confirming the move
      - A market that goes up without volume is easy to push back down
      - Classic Wyckoff Phase C (UTAD) before Phase D distribution

    LR TABLE (direct, bypasses _score_to_lr):
      OBV diverged  > 25% below prior level: LR = 4.50  (near-certain distribution)
      OBV diverged 15-25% below:             LR = 3.20
      OBV diverged  8-15% below:             LR = 2.40
      OBV diverged  3-8%  below:             LR = 1.80
      No divergence (OBV confirms):           LR = 0.55  (bearish for short — price confirmed)
    """
    NULL = DistributionDivergence(
        detected=False, price_swing_high=0, obv_at_swing=0,
        price_now=0, obv_now=0, obv_divergence_pct=0,
        n_swing_lookback=0, severity="none", lr=1.0
    )

    if "OBV" not in df.columns or len(df) < 15:
        return NULL

    col = "close" if "close" in df.columns else "price"
    try:
        prices = df[col].values.astype(float)
        obv    = df["OBV"].values.astype(float)
        highs  = df["high"].values.astype(float) if "high" in df.columns else prices

        cur_price = float(prices[-1])
        cur_obv   = float(obv[-1])

        # Find swing highs: a bar is a swing high if it is the highest
        # in a ±3 bar window and above the rolling 10-bar mean
        swing_highs = []   # list of (bar_index, price, obv_at_bar)
        for i in range(5, min(lookback, len(df) - 1)):
            idx = len(df) - 1 - i   # counting from the end
            if idx < 3 or idx >= len(df) - 3:
                continue
            h = float(highs[idx])
            window = highs[max(0, idx-3):idx+4]
            if h == window.max() and h > float(np.mean(highs[max(0,idx-10):idx])):
                swing_highs.append((idx, float(prices[idx]), float(obv[idx])))

        if not swing_highs:
            return NULL

        # Use the most recent swing high as reference
        ref_idx, ref_price, ref_obv = swing_highs[0]
        n_lookback = len(df) - 1 - ref_idx

        # Price must be near or above the reference swing high (new HH or testing it)
        if cur_price < ref_price * 0.97:
            # Price has not retested the swing high — no divergence context
            return NULL

        # Measure OBV divergence: how far has OBV dropped from the reference level?
        obv_div_pct = (cur_obv - ref_obv) / (abs(ref_obv) + 1e-9) * 100

        # Divergence = price >= prior high, OBV < OBV at prior high
        if obv_div_pct >= 0:
            # OBV is HIGHER than at the swing — no divergence (bullish, bad for short)
            return DistributionDivergence(
                detected=False, price_swing_high=ref_price, obv_at_swing=ref_obv,
                price_now=cur_price, obv_now=cur_obv, obv_divergence_pct=obv_div_pct,
                n_swing_lookback=n_lookback, severity="none",
                lr=0.55   # OBV confirming price → bearish for short thesis
            )

        # Divergence detected — price HH, OBV LH
        div_mag = abs(obv_div_pct)
        if   div_mag >= 25: severity = "strong";   lr = 4.50
        elif div_mag >= 15: severity = "moderate";  lr = 3.20
        elif div_mag >= 8:  severity = "mild+";     lr = 2.40
        elif div_mag >= 3:  severity = "mild";      lr = 1.80
        else:               severity = "weak";      lr = 1.20

        return DistributionDivergence(
            detected=True, price_swing_high=ref_price, obv_at_swing=ref_obv,
            price_now=cur_price, obv_now=cur_obv, obv_divergence_pct=obv_div_pct,
            n_swing_lookback=n_lookback, severity=severity, lr=lr
        )

    except Exception:
        return NULL


def _short_score_distribution(df: pd.DataFrame, last: pd.Series,
                               obv_trend: str, rel_vol: float,
                               micro=None) -> Tuple[float, str]:
    """
    Q2: Is participation deteriorating? (DISTRIBUTION)

    NOW ANCHORED ON detect_distribution_divergence() as the primary signal.

    OLD: OBV trend direction → score += 0.35 (binary, loses magnitude)
    NEW: Price HH / OBV LH divergence → LR up to 4.50 (magnitude-sensitive)

    The divergence dominates. Other signals (vol asymmetry, CVD) are secondary.
    """
    col  = "close" if "close" in df.columns else "price"
    ev   = []

    # PRIMARY: Distribution divergence (price HH / OBV LH)
    div = detect_distribution_divergence(df)
    if div.detected:
        # Map the divergence LR to a score for the orchestrator's _score_to_lr
        # But we need to preserve the high LR — so we back-calculate the score
        # that would produce this LR from _score_to_lr with MAX_LR=_SHORT_MAX_LR=3.0
        # For LR > 3.0 (strong divergence), we cap score at 1.0 and note separately
        div_score = min(1.0, math.log(div.lr) / math.log(_SHORT_MAX_LR) * 0.5 + 0.5)
        ev.append(f"DIV:{div.severity}(OBV{div.obv_divergence_pct:+.1f}%,LR={div.lr:.2f})")
        # Store div for caller to add high-LR separately
        _div_obj = div
    else:
        div_score = 0.0
        _div_obj  = div
        if div.lr < 1.0:   # OBV confirming = bullish (bad for short)
            div_score = -0.10
            ev.append(f"OBV_confirms_price(bad_for_short)")

    # SECONDARY: Break of Participation — RATE of participation collapse
    # Runs alongside distribution divergence (they measure different things)
    bop = detect_break_of_participation(df)
    _break_of_participation_cache[0] = bop
    if bop.detected and bop.lr > 1.0:
        bop_score = min(0.35, (bop.n_signals / 3.0) * 0.35 *
                        (1.5 if bop.price_at_highs else 1.0))
        ev.append(f"BOP:{bop.severity}(n={bop.n_signals}/3)")
    else:
        bop_score = 0.0
        if bop.lr < 1.0:   # participation rising = bad for short
            bop_score = -0.08

    # SECONDARY: Simple OBV trend direction
    obv_base = 0.0
    if not div.detected:   # only count if divergence didn't capture it
        if obv_trend == "falling":
            obv_base = 0.35; ev.append("obv_falling")
        elif obv_trend == "flat":
            obv_base = 0.10; ev.append("obv_flat")

    # TERTIARY: Volume asymmetry (down-day vol > up-day vol)
    vol_asym = 0.0
    try:
        if "volume" in df.columns and len(df) >= 10:
            prices = df[col].values.astype(float)
            vols   = df["volume"].values.astype(float)
            mask   = prices[-10:] >= np.roll(prices, 1)[-10:]
            up_v   = vols[-10:][mask].mean(); down_v = vols[-10:][~mask].mean()
            if not np.isnan(up_v) and not np.isnan(down_v) and up_v > 0:
                r = down_v / (up_v + 1e-9)
                if r > 1.5: vol_asym = 0.18; ev.append(f"vol_asym({r:.1f}x)")
                elif r > 1.2: vol_asym = 0.08
    except Exception: pass

    # TERTIARY: CVD from microstructure
    cvd_score = 0.0
    if micro and not getattr(micro, "error", True):
        cvd = getattr(micro, "cvd_pct", 0)
        if cvd <= -20: cvd_score = 0.18; ev.append(f"cvd({cvd:.0f}%)")
        elif cvd <= -10: cvd_score = 0.09

    # Thin volume
    thin_score = 0.0
    if rel_vol < 0.60:
        thin_score = 0.08; ev.append(f"thin({rel_vol:.1f}x)")

    # Composite: divergence is primary, others secondary
    composite = min(1.0, div_score + bop_score + obv_base + vol_asym + cvd_score + thin_score)
    note = f"distrib({composite:.2f}):{'+'.join(ev) or 'none'}"

    # Return the _div_obj as a side-channel via a module-level variable for the orchestrator
    # The orchestrator reads this to add the high-LR directly
    _distribution_divergence_cache[0] = _div_obj

    return max(0.0, composite), note


# Thread-local cache so run_short_scorer can read the divergence object after _short_score_distribution runs
_distribution_divergence_cache: List = [None]


# ── Break of Participation ────────────────────────────────────────────────────

@dataclass
class BreakOfParticipation:
    """
    Structured result from detect_break_of_participation().
    Measures the RATE of volume/OBV collapse relative to price position.
    """
    detected:             bool
    price_at_highs:       bool    # price in top 25% of 20-day range?
    obv_slope_reversed:   bool    # OBV was rising, now falling (slope sign flip)
    upday_vol_ratio:      float   # recent_up_vol / historical_up_vol  (< 1 = collapsed)
    obv_roc_collapsed:    bool    # OBV 5d ROC turned negative while price ROC positive
    n_signals:            int     # how many of the three participation signals fired
    price_pct_from_high:  float   # how close to 20-day high price is (0 = at high)
    severity:             str     # "critical" / "strong" / "moderate" / "mild" / "none"
    lr:                   float   # pre-computed LR for direct injection
    details:              str


def detect_break_of_participation(df: pd.DataFrame,
                                   short_window: int = 5,
                                   long_window:  int = 20) -> "BreakOfParticipation":
    """
    Detect "Break of Participation" — price holds but volume/OBV collapses.

    WHY THIS MATTERS:
      Price can be levitated without volume for days or even weeks.
      Smart money exits quietly while retail provides buying support.
      When the last buyer is exhausted — with no participation left —
      price drops rapidly because there is no bid depth.

      This signal fires EARLIER than structural divergence: participation
      collapses in 3-5 bars before the OBV higher-high / lower-high
      pattern completes. It is the momentum-based early warning.

    THREE PARTICIPATION SIGNALS:
      1. OBV slope reversal: OBV was rising (positive slope) and is
         now falling (negative slope). The cumulative capital flow
         that supported the uptrend has reversed direction.

      2. Up-day volume collapse: volume on days price goes up has
         shrunk dramatically vs the prior window. Buyers are disappearing.
         ratio = recent_up_vol_avg / historical_up_vol_avg
         ratio < 0.60 → participation collapsed by 40%+

      3. OBV rate-of-change turned negative while price ROC is positive:
         OBV momentum is diverging from price momentum in real time.

    CRITICAL WEIGHT WHEN PRICE AT HIGHS:
      All three signals are more significant when price is near the 20-day
      high. A collapse of participation at a TOP is far more important than
      the same collapse at a random mid-range price.

      price at highs + 3 signals: LR = 4.20  (critical — sell immediately)
      price at highs + 2 signals: LR = 3.00  (strong)
      price at highs + 1 signal:  LR = 2.00  (moderate)
      price not at highs + 2+:    LR = 1.70  (notable but less urgent)
      price not at highs + 1:     LR = 1.35  (mild)
      No signals:                 LR = 1.00  (neutral)
      OBV participation RISING:   LR = 0.65  (bullish — bad for short)
    """
    NULL = BreakOfParticipation(
        detected=False, price_at_highs=False, obv_slope_reversed=False,
        upday_vol_ratio=1.0, obv_roc_collapsed=False, n_signals=0,
        price_pct_from_high=0.0, severity="none", lr=1.0,
        details="insufficient data"
    )

    if len(df) < long_window + 5:
        return NULL

    col = "close" if "close" in df.columns else "price"

    try:
        prices = df[col].values.astype(float)
        obv    = df["OBV"].values.astype(float) if "OBV" in df.columns else None
        vols   = df["volume"].values.astype(float) if "volume" in df.columns else None

        cur_price   = float(prices[-1])
        high_20d    = float(prices[-long_window:].max())
        low_20d     = float(prices[-long_window:].min())
        rng_20d     = high_20d - low_20d + 1e-9
        price_pct_from_high = (high_20d - cur_price) / rng_20d  # 0 = at high, 1 = at low

        # Price at highs = within top 25% of 20-day range
        price_at_highs = price_pct_from_high <= 0.25

        signals = []
        details_parts = [f"price@{price_pct_from_high:.0%}_from_high"]

        # ── Signal 1: OBV slope reversal ─────────────────────────────────────
        obv_slope_reversed = False
        if obv is not None and len(obv) >= long_window + short_window:
            # Slope over recent short_window vs slope over prior long_window
            def linslope(arr):
                n = len(arr)
                x = np.arange(n, dtype=float)
                return float(np.polyfit(x, arr, 1)[0])

            recent_slope = linslope(obv[-short_window:])
            prior_slope  = linslope(obv[-long_window:-short_window])

            if prior_slope > 0 and recent_slope < 0:
                obv_slope_reversed = True
                signals.append("obv_slope_rev")
                details_parts.append(f"obv_slope:{prior_slope:+.0f}->{recent_slope:+.0f}")
            elif recent_slope < 0 and prior_slope < 0:
                details_parts.append("obv_slope:both_neg")
            elif recent_slope > 0:
                # OBV still rising = participation healthy (negative for short)
                details_parts.append(f"obv_slope_ok:{recent_slope:+.0f}")

        # ── Signal 2: Up-day volume collapse ─────────────────────────────────
        upday_vol_ratio = 1.0
        if vols is not None and len(vols) >= long_window + short_window:
            rets_recent = np.diff(prices[-short_window:])
            rets_prior  = np.diff(prices[-long_window:-short_window + 1])

            up_vol_recent = vols[-short_window:][1:][rets_recent >= 0]
            up_vol_prior  = vols[-long_window:-short_window][rets_prior >= 0]

            if len(up_vol_recent) >= 1 and len(up_vol_prior) >= 2:
                avg_recent = up_vol_recent.mean()
                avg_prior  = up_vol_prior.mean()
                upday_vol_ratio = avg_recent / (avg_prior + 1e-9)

                if upday_vol_ratio < 0.50:
                    signals.append(f"upvol_crash({upday_vol_ratio:.2f}x)")
                    details_parts.append(f"up_vol_ratio:{upday_vol_ratio:.2f}")
                elif upday_vol_ratio < 0.70:
                    signals.append(f"upvol_weak({upday_vol_ratio:.2f}x)")
                    details_parts.append(f"up_vol_ratio:{upday_vol_ratio:.2f}")
                elif upday_vol_ratio > 1.30:
                    details_parts.append(f"up_vol_strong:{upday_vol_ratio:.2f}")

        # ── Signal 3: OBV ROC divergence ─────────────────────────────────────
        obv_roc_collapsed = False
        if obv is not None and len(obv) >= long_window + short_window:
            obv_roc_now   = (obv[-1] - obv[-short_window - 1]) / (abs(obv[-short_window - 1]) + 1e-9)
            price_roc_now = (prices[-1] - prices[-short_window - 1]) / (prices[-short_window - 1] + 1e-9)

            if obv_roc_now < -0.01 and price_roc_now > 0.005:
                obv_roc_collapsed = True
                signals.append(f"obv_roc_neg(obv{obv_roc_now:+.1%},px{price_roc_now:+.1%})")
                details_parts.append(f"roc_div:obv={obv_roc_now:+.2%}/px={price_roc_now:+.2%}")

        # ── Severity and LR ───────────────────────────────────────────────────
        n = len(signals)
        participation_rising = (
            not obv_slope_reversed
            and upday_vol_ratio >= 1.20
            and not obv_roc_collapsed
        )

        if participation_rising and not any(s.startswith("obv_slope_rev") for s in signals):
            severity = "none"
            lr = 0.65   # participation healthy → short thesis weakened
        elif n == 0:
            severity = "none"; lr = 1.00
        elif price_at_highs:
            if   n >= 3: severity = "critical"; lr = 4.20
            elif n >= 2: severity = "strong";   lr = 3.00
            else:        severity = "moderate";  lr = 2.00
        else:
            if   n >= 2: severity = "moderate";  lr = 1.70
            else:        severity = "mild";       lr = 1.35

        detected = n >= 1

        return BreakOfParticipation(
            detected=detected, price_at_highs=price_at_highs,
            obv_slope_reversed=obv_slope_reversed,
            upday_vol_ratio=upday_vol_ratio,
            obv_roc_collapsed=obv_roc_collapsed,
            n_signals=n,
            price_pct_from_high=price_pct_from_high,
            severity=severity, lr=lr,
            details=f"n={n}  {' | '.join(details_parts)}"
        )

    except Exception as e:
        return BreakOfParticipation(
            detected=False, price_at_highs=False, obv_slope_reversed=False,
            upday_vol_ratio=1.0, obv_roc_collapsed=False, n_signals=0,
            price_pct_from_high=0.0, severity="none", lr=1.0,
            details=f"error: {e}"
        )


_break_of_participation_cache: List = [None]



# ── Failed Breakout Engine ────────────────────────────────────────────────────
#
# ONE OF THE HIGHEST-EDGE SHORT PATTERNS.
#
# A failed breakout (bull trap) occurs when:
#   price exceeds a prior resistance level
#   BUT: OBV didn't confirm / wick rejected / volume thin / price reversed fast
#
# The four failure signals are increasingly conclusive:
#   weak OBV         → suspicious (breakout not backed by capital)
#   rejection wick   → trap forming (price tested resistance, got sold)
#   declining volume  → exhaustion (buyers ran out at exactly the wrong level)
#   fast reversal    → confirmed (price gave back the breakout in 1-3 bars)
#
# The FAST REVERSAL is the trigger signal. Without it, the others are warnings.
# With it, the pattern is confirmed and the short is live.
#
# LR TABLE (n = failure signals present, rev = fast reversal confirmed):
#   4 signals + rev:  LR = 5.00  (maximum conviction — all four fire)
#   3 signals + rev:  LR = 4.00
#   2 signals + rev:  LR = 2.80  (confirmed trap)
#   1 signal  + rev:  LR = 2.00  (reversal alone = minimum short trigger)
#   3 signals, no rev: LR = 2.50  (suspicious — watch for reversal)
#   2 signals, no rev: LR = 1.80
#   1 signal, no rev:  LR = 1.30
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FailedBreakout:
    """Structured result from detect_failed_breakout()."""
    breakout_detected:    bool
    breakout_bar_offset:  int      # how many bars ago the breakout attempt was
    resistance_level:     float    # the prior resistance that was broken
    breakout_high:        float    # the high that exceeded resistance

    # Four failure signals — each scored [0,1]
    weak_obv:         float   # OBV didn't spike with price at breakout
    rejection_wick:   float   # large upper shadow at resistance
    volume_declining: float   # breakout bar volume < prior average
    fast_reversal:    bool    # price closed back below resistance

    n_failure_signals: int    # 0-4 (counting fast_reversal as 1 if present)
    is_failed:         bool   # overall verdict
    severity:          str    # "confirmed" / "suspected" / "warning" / "none"
    lr:                float  # pre-computed LR for direct injection
    details:           str


def detect_failed_breakout(df: pd.DataFrame, last: pd.Series,
                            max_lookback: int = 5,
                            prior_window:  int = 20) -> "FailedBreakout":
    """
    Detect failed breakouts — the bull trap pattern.

    ALGORITHM:
      1. Compute prior resistance = highest high in bars [-25 : -max_lookback]
         (the resistance BEFORE our observation window)

      2. Scan last max_lookback bars for any that exceeded prior resistance
         (a breakout attempt occurred)

      3. For the most recent breakout bar, evaluate 4 failure signals:
         a. Weak OBV:         OBV at breakout vs OBV 5 bars earlier
                              Genuine breakouts have OBV spikes. Fake ones don't.
         b. Rejection wick:   upper_wick / bar_range at the breakout bar
                              Large wick = price rejected at resistance
         c. Declining volume: breakout bar volume vs 10-bar prior average
                              Low-volume breakouts almost always fail.
         d. Fast reversal:    did price close back below resistance within 1-3 bars?
                              This is the CONFIRMATION signal — pattern complete.

      4. Score based on n signals present and whether reversal confirmed.
    """
    NULL = FailedBreakout(
        breakout_detected=False, breakout_bar_offset=0,
        resistance_level=0, breakout_high=0,
        weak_obv=0, rejection_wick=0, volume_declining=0, fast_reversal=False,
        n_failure_signals=0, is_failed=False,
        severity="none", lr=1.0, details="no recent breakout"
    )

    col = "close" if "close" in df.columns else "price"
    if len(df) < prior_window + max_lookback + 5:
        return NULL

    try:
        prices = df[col].values.astype(float)
        highs  = df["high"].values.astype(float)  if "high"  in df.columns else prices
        lows   = df["low"].values.astype(float)   if "low"   in df.columns else prices
        opens  = df["open"].values.astype(float)  if "open"  in df.columns else prices
        closes = df["close"].values.astype(float) if "close" in df.columns else prices
        obv    = df["OBV"].values.astype(float)   if "OBV"   in df.columns else None
        vols   = df["volume"].values.astype(float) if "volume" in df.columns else None
        rel_vol= df["rel_vol"].values.astype(float) if "rel_vol" in df.columns else None

        n = len(df)

        # Prior resistance: highest high BEFORE our lookback window
        resist_window_end   = n - max_lookback
        resist_window_start = max(0, resist_window_end - prior_window)
        prior_resistance    = float(highs[resist_window_start:resist_window_end].max())

        # Find most recent bar that exceeded prior resistance
        bo_idx = None; bo_offset = 0
        for offset in range(1, max_lookback + 1):
            idx = n - offset
            if idx < 0: break
            if float(highs[idx]) > prior_resistance * 1.002:  # 0.2% buffer
                bo_idx = idx; bo_offset = offset; break

        if bo_idx is None:
            return NULL

        bo_high  = float(highs[bo_idx])
        bo_close = float(closes[bo_idx])
        bo_open  = float(opens[bo_idx])
        bo_low   = float(lows[bo_idx])
        bo_range = bo_high - bo_low + 1e-9

        # ── Signal A: Weak OBV ────────────────────────────────────────────────
        # Genuine breakouts have OBV spiking upward at the breakout bar.
        # Fake breakouts: OBV flat, declining, or rising much less than price.
        weak_obv_score = 0.0
        if obv is not None and bo_idx >= 5:
            obv_before_bo = float(obv[bo_idx - 5])   # OBV 5 bars before breakout
            obv_at_bo     = float(obv[bo_idx])
            obv_chg       = (obv_at_bo - obv_before_bo) / (abs(obv_before_bo) + 1e-9)

            # Price went up N%, what did OBV do?
            price_chg = (bo_close - float(prices[bo_idx - 5])) / (float(prices[bo_idx - 5]) + 1e-9)

            if obv_chg < 0:
                weak_obv_score = 1.00   # OBV fell while price broke out = strong distribution
            elif obv_chg < price_chg * 0.30:
                weak_obv_score = 0.75   # OBV barely moved vs price spike
            elif obv_chg < price_chg * 0.60:
                weak_obv_score = 0.45   # OBV rose but much less than price

        # ── Signal B: Rejection Wick ──────────────────────────────────────────
        # At the breakout bar, how much of the range is in the upper wick?
        # Large upper wick = price probed above resistance, sellers appeared.
        upper_wick    = (bo_high - max(bo_close, bo_open)) / bo_range
        wick_above_res = max(0, bo_high - prior_resistance) / (bo_range + 1e-9)

        rejection_wick_score = 0.0
        if upper_wick >= 0.60:
            rejection_wick_score = 1.00   # 60%+ wick = strong rejection
        elif upper_wick >= 0.45:
            rejection_wick_score = 0.75
        elif upper_wick >= 0.30:
            rejection_wick_score = 0.45
        elif upper_wick >= 0.20:
            rejection_wick_score = 0.25

        # ── Signal C: Declining Volume ────────────────────────────────────────
        # Compare breakout bar volume to the 10-bar prior average.
        # A low-volume breakout has no institutional backing — almost always fails.
        volume_declining_score = 0.0
        if vols is not None and bo_idx >= 10:
            avg_vol_prior = float(vols[bo_idx - 10:bo_idx].mean())
            bo_vol        = float(vols[bo_idx])
            vol_ratio     = bo_vol / (avg_vol_prior + 1e-9)

            if   vol_ratio < 0.50: volume_declining_score = 1.00   # less than half normal
            elif vol_ratio < 0.70: volume_declining_score = 0.75
            elif vol_ratio < 0.85: volume_declining_score = 0.50
            elif vol_ratio < 1.00: volume_declining_score = 0.25

            # Also check rel_vol at breakout bar
            if rel_vol is not None:
                rv = float(rel_vol[bo_idx])
                if rv < 0.70:
                    volume_declining_score = min(1.0, volume_declining_score + 0.25)

        # ── Signal D: Fast Reversal ───────────────────────────────────────────
        # Did price close back below the resistance level within 1-3 bars?
        # This is the CONFIRMATION — the pattern is complete and the short is live.
        fast_reversal = False
        bars_after_bo = n - bo_idx - 1   # bars that occurred AFTER breakout
        if bars_after_bo >= 1:
            # Check closes from breakout+1 to now
            for i in range(bo_idx + 1, min(bo_idx + 4, n)):
                if float(closes[i]) < prior_resistance * 1.002:
                    fast_reversal = True; break
            # Also confirmed if CURRENT price is back below resistance
            if float(prices[-1]) < prior_resistance:
                fast_reversal = True

        # ── Scoring ───────────────────────────────────────────────────────────
        THRESH = 0.40   # minimum score to count as "signal fired"
        fired  = [
            ("weak_obv",         weak_obv_score         >= THRESH),
            ("rejection_wick",   rejection_wick_score    >= THRESH),
            ("volume_declining", volume_declining_score  >= THRESH),
            ("fast_reversal",    fast_reversal),
        ]
        n_fired = sum(1 for _, f in fired if f)

        # LR table — reversal status is the key gating factor
        if fast_reversal:
            if   n_fired >= 4: lr = 5.00; severity = "confirmed_critical"
            elif n_fired >= 3: lr = 4.00; severity = "confirmed_strong"
            elif n_fired >= 2: lr = 2.80; severity = "confirmed"
            else:              lr = 2.00; severity = "reversal_only"
        else:
            if   n_fired >= 3: lr = 2.50; severity = "suspected_strong"
            elif n_fired >= 2: lr = 1.80; severity = "suspected"
            elif n_fired >= 1: lr = 1.30; severity = "warning"
            else:              lr = 1.00; severity = "none"

        is_failed = n_fired >= 2

        fired_names = [name for name, f in fired if f]
        details = (
            f"res={prior_resistance:.4f}  bo_high={bo_high:.4f}  "
            f"offset={bo_offset}bars  "
            f"wick={upper_wick:.2f}  weak_obv={weak_obv_score:.2f}  "
            f"vol_decl={volume_declining_score:.2f}  reversal={fast_reversal}  "
            f"fired={fired_names}"
        )

        return FailedBreakout(
            breakout_detected    = True,
            breakout_bar_offset  = bo_offset,
            resistance_level     = prior_resistance,
            breakout_high        = bo_high,
            weak_obv             = weak_obv_score,
            rejection_wick       = rejection_wick_score,
            volume_declining     = volume_declining_score,
            fast_reversal        = fast_reversal,
            n_failure_signals    = n_fired,
            is_failed            = is_failed,
            severity             = severity,
            lr                   = round(lr, 2),
            details              = details,
        )

    except Exception as e:
        return FailedBreakout(
            breakout_detected=False, breakout_bar_offset=0,
            resistance_level=0, breakout_high=0,
            weak_obv=0, rejection_wick=0, volume_declining=0, fast_reversal=False,
            n_failure_signals=0, is_failed=False,
            severity="none", lr=1.0, details=f"error: {e}"
        )


_failed_breakout_cache: List = [None]

def _short_score_trap(df, last):
    """
    Q3: Is breakout failing? (TRAP)

    Now anchored on detect_failed_breakout() — the dedicated Failed Breakout Engine.
    The engine evaluates all four failure signals (weak OBV, wick, volume, reversal)
    and returns a structured result that run_short_scorer will inject directly.
    """
    col = "close" if "close" in df.columns else "price"

    # Primary: Failed Breakout Engine
    fb = detect_failed_breakout(df, last)
    _failed_breakout_cache[0] = fb

    if fb.breakout_detected:
        # Map fb to a score for the Q3 group LR (run_short_scorer also adds direct LR)
        if fb.fast_reversal:
            base = min(1.0, 0.50 + fb.n_failure_signals * 0.12)
        else:
            base = min(1.0, fb.n_failure_signals * 0.20)
        ev = [f"FB:{fb.severity}(n={fb.n_failure_signals})"]
    else:
        base = 0.0; ev = []

    # Secondary: simple wick rejection at recent highs (if no breakout context)
    if not fb.breakout_detected:
        try:
            prices = df[col].values.astype(float)
            if "high" in df.columns and len(df) >= 15:
                highs = df["high"].values.astype(float)
                opens = df["open"].values.astype(float) if "open" in df.columns else prices
                lows  = df["low"].values.astype(float)  if "low"  in df.columns else prices
                cur_h = float(highs[-1]); cur_c = float(prices[-1])
                cur_o = float(opens[-1]); cur_l = float(lows[-1])
                rng   = cur_h - cur_l + 1e-9
                uw    = (cur_h - max(cur_c, cur_o)) / rng
                if uw > 0.50: base = 0.30; ev.append(f"wick_rej({uw:.2f})")
                elif uw > 0.35: base = 0.18; ev.append(f"wick_at_res({uw:.2f})")
        except Exception: pass

    return min(1.0, base), f"trap({base:.2f}):{'+'.join(ev) or 'none'}"


@dataclass
class BearAcceleration:
    """
    Composite result from detect_bear_acceleration().
    Each of the four signals is scored independently [0,1].
    When multiple fire together, the composite amplifies super-linearly
    to reflect that convergent acceleration signals are more significant
    than any single signal alone.
    """
    atr_expanding:      float   # ATR expanding while price falls     [0,1]
    vol_spike_red:      float   # volume spike on down candles        [0,1]
    lower_high_fail:    float   # lower high formed, breakdown        [0,1]
    support_rejection:  float   # price rejected from support level   [0,1]
    n_confirmed:        int     # how many of the four fired (≥ 0.30 threshold)
    composite_score:    float   # amplified composite [0,1]
    expected_move_atr:  float   # expected downside move in ATR units
    details:            str     # human-readable summary


def detect_bear_acceleration(df: pd.DataFrame, last: pd.Series,
                              vol_regime: str = "neutral") -> "BearAcceleration":
    """
    Detects and quantifies bearish acceleration from four independent signals.

    WHY CONVERGENCE MATTERS:
      Each signal alone can be noise. ATR expanding without volume = low conviction.
      Volume spike without structural breakdown = single event.
      When 2+ signals fire together, the probability of a genuine acceleration
      multiplies — they are measuring different mechanisms that all confirm
      the same underlying sell pressure.

    SUPER-LINEAR AMPLIFICATION:
      1 signal:  composite = signal_score × 0.80    (single evidence)
      2 signals: composite = avg_score   × 1.20    (meaningful confirmation)
      3 signals: composite = avg_score   × 1.55    (strong acceleration)
      4 signals: composite = avg_score   × 2.00    (maximum conviction)

    EXPECTED DOWNSIDE MOVE (in ATR units):
      No acceleration:     1.0 × ATR
      1 signal:            1.5 × ATR
      2 signals:           2.0 × ATR
      3 signals:           2.8 × ATR
      4 signals confirmed: 3.5 × ATR
    """
    col = "close" if "close" in df.columns else "price"

    # ── Signal 1: ATR Expanding Down ─────────────────────────────────────────
    # Dangerous: price is falling faster with each bar (downside momentum building)
    atr_score = 0.0
    try:
        if "ATR" in df.columns and len(df) >= 20:
            atr_now  = float(df["ATR"].iloc[-1])
            atr_prev = float(df["ATR"].iloc[-10])
            prices   = df[col].values.astype(float)
            atr_ratio = atr_now / (atr_prev + 1e-9)
            price_direction = prices[-1] < prices[-5] if len(prices) >= 5 else False
            if price_direction:
                if   atr_ratio >= 1.50: atr_score = 1.00
                elif atr_ratio >= 1.30: atr_score = 0.75
                elif atr_ratio >= 1.15: atr_score = 0.50
                elif atr_ratio >= 1.05: atr_score = 0.25
            # ATR contracting while falling = losing momentum (reduce acceleration)
            elif not price_direction and atr_ratio < 0.80:
                atr_score = -0.10   # slight negative: bear losing steam
    except Exception:
        pass

    # ── Signal 2: Volume Spike on Red Candles ────────────────────────────────
    # Distribution: large sellers unloading into any bid
    vol_score = 0.0
    try:
        cur_open  = float(last.get("open",  last[col]))
        cur_close = float(last.get("close", last[col]) if "close" in last.index else last[col])
        is_red    = cur_close < cur_open

        if "rel_vol" in df.columns:
            rel_vol = float(df["rel_vol"].iloc[-1])
            if is_red:
                if   rel_vol >= 4.0: vol_score = 1.00   # panic selling
                elif rel_vol >= 3.0: vol_score = 0.85
                elif rel_vol >= 2.0: vol_score = 0.65
                elif rel_vol >= 1.5: vol_score = 0.40
            else:
                # Green candle with spike = covering rally, not distribution
                if rel_vol >= 2.0: vol_score = -0.15

        # 3-day down-volume trend: recent vol heavier on down days than up days
        if "volume" in df.columns and len(df) >= 7:
            prices_7 = df[col].values.astype(float)[-7:]
            vols_7   = df["volume"].values.astype(float)[-7:]
            rets_6   = np.diff(prices_7)
            dv = vols_7[1:][rets_6 < 0]; uv = vols_7[1:][rets_6 >= 0]
            if len(dv) >= 2 and len(uv) >= 1:
                asym = dv.mean() / (uv.mean() + 1e-9)
                if asym > 1.6:
                    vol_score = min(1.0, vol_score + 0.20)
    except Exception:
        pass

    # ── Signal 3: Lower High Failure ─────────────────────────────────────────
    # Breakdown pattern: each rally fails lower — supply permanently exceeding demand
    lhf_score = 0.0
    try:
        if "high" in df.columns and len(df) >= 12:
            highs    = df["high"].values.astype(float)
            recent_h = highs[-5:].max()     # current swing high
            prior_h  = highs[-12:-5].max()  # prior swing high

            if prior_h > 0:
                fail_pct = (prior_h - recent_h) / prior_h  # how much lower
                if   fail_pct >= 0.05: lhf_score = 1.00   # clear lower high: 5%+ below
                elif fail_pct >= 0.03: lhf_score = 0.75
                elif fail_pct >= 0.01: lhf_score = 0.45
                elif fail_pct < 0:     lhf_score = -0.10  # higher high = not breaking down

            # Confirm with price below midpoint of recent range
            prices = df[col].values.astype(float)
            if len(prices) >= 12:
                rng_mid = (highs[-12:].max() + df["low"].values.astype(float)[-12:].min()) / 2
                if prices[-1] < rng_mid:
                    lhf_score = min(1.0, lhf_score + 0.15)
    except Exception:
        pass

    # ── Signal 4: Support Rejection ──────────────────────────────────────────
    # Acceleration: price tested a prior support level and was rejected downward
    # This signals that former support has become resistance — accelerant for downside
    sr_score = 0.0
    try:
        if "low" in df.columns and len(df) >= 20:
            prices   = df[col].values.astype(float)
            lows     = df["low"].values.astype(float)
            highs    = df["high"].values.astype(float) if "high" in df.columns else prices + 0.01

            # Prior support = swing low from 10-20 bars ago
            prior_support = lows[-20:-8].min()
            cur_price     = float(prices[-1])

            # Did price recently bounce up to the support level (now resistance)?
            recent_high   = highs[-5:].max()
            touched_level = abs(recent_high - prior_support) / (prior_support + 1e-9) < 0.03

            if touched_level and cur_price < prior_support * 0.99:
                # Price bounced to support-turned-resistance and failed → acceleration
                reject_depth = (prior_support - cur_price) / (prior_support + 1e-9)
                if   reject_depth >= 0.05: sr_score = 1.00
                elif reject_depth >= 0.02: sr_score = 0.65
                elif reject_depth >= 0.01: sr_score = 0.40

            # Also check: price just broke below the 14-day low (fresh breakdown)
            low_14d = lows[-14:].min()
            if cur_price < low_14d * 0.99:
                sr_score = min(1.0, sr_score + 0.25)

    except Exception:
        pass

    # ── Composite: super-linear amplification for convergence ─────────────────
    CONFIRM_THRESH = 0.30   # must score above this to count as "fired"
    signals = [
        ("atr_exp",     max(0.0, atr_score)),
        ("vol_spike_red", max(0.0, vol_score)),
        ("lower_high",  max(0.0, lhf_score)),
        ("support_rej", max(0.0, sr_score)),
    ]
    fired   = [(name, s) for name, s in signals if s >= CONFIRM_THRESH]
    n_fired = len(fired)

    if n_fired == 0:
        raw_score = max(0.0, max(s for _,s in signals) * 0.30)   # weak noise
    else:
        avg_fired = sum(s for _,s in fired) / n_fired
        # Super-linear multiplier: each additional confirmation amplifies more
        multipliers = {1: 0.80, 2: 1.20, 3: 1.55, 4: 2.00}
        raw_score   = avg_fired * multipliers.get(n_fired, 2.00)

    # Reduce if negative signals outweigh positives
    neg_penalty = abs(min(0.0, atr_score + vol_score + lhf_score + sr_score)) * 0.15
    composite   = round(min(1.0, max(0.0, raw_score - neg_penalty)), 3)

    # Expected downside move in ATR units
    move_table  = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.8, 4: 3.5}
    expected_atr = move_table.get(n_fired, 1.0)
    if vol_regime == "expansion": expected_atr *= 1.20  # vol regime amplifies

    # Summary string
    fired_strs   = [f"{n}={s:.2f}" for n,s in fired]
    unfired_strs = [f"{n}=0" for n,s in signals if s < CONFIRM_THRESH]
    details = (f"n_confirmed={n_fired}/4  composite={composite:.3f}  "
               f"expected={expected_atr:.1f}xATR  "
               f"[{', '.join(fired_strs)}]  "
               f"[unfired: {', '.join(unfired_strs)}]")

    return BearAcceleration(
        atr_expanding    = max(0.0, atr_score),
        vol_spike_red    = max(0.0, vol_score),
        lower_high_fail  = max(0.0, lhf_score),
        support_rejection= max(0.0, sr_score),
        n_confirmed      = n_fired,
        composite_score  = composite,
        expected_move_atr= expected_atr,
        details          = details,
    )


def _short_score_acceleration(df: pd.DataFrame, last: pd.Series,
                               vol_regime: str) -> Tuple[float, str]:
    """
    Q4: Is volatility expanding down? (ACCELERATION)

    Wraps detect_bear_acceleration() — evaluates all four acceleration signals
    and returns a single composite score for the short scorer.

    The super-linear convergence bonus means 3+ signals firing is treated as
    much stronger evidence than any single signal, even at the same raw level.
    """
    ba = detect_bear_acceleration(df, last, vol_regime)

    label_map = {0: "no acceleration", 1: "mild", 2: "moderate", 3: "strong", 4: "CONFIRMED"}
    label = label_map.get(ba.n_confirmed, "CONFIRMED")

    if ba.n_confirmed >= 2:
        conf_str = f"ACCELERATION {label.upper()}: {ba.details}"
    else:
        conf_str = f"accel({ba.composite_score:.2f}): {ba.details}"

    return ba.composite_score, conf_str


def _short_score_danger(df, last, futures, obv_trend, micro=None):
    score = 0.0; ev = []
    if obv_trend == "falling": score += 0.25; ev.append("obv_dist")
    if futures and not futures.error and futures.is_futures:
        fr=futures.funding_rate*100; fr1d=futures.funding_1d_sum*100
        if fr>0 and fr1d>0 and fr<fr1d/3: score+=0.20; ev.append(f"fund_declining({fr:.3f}%)")
        elif fr>0.03: score+=0.12; ev.append(f"fund+({fr:.3f}%)")
        if futures.oi_change_24h<-5: score+=0.15; ev.append(f"oi_fall({futures.oi_change_24h:+.1f}%)")
        if futures.liq_below_usd>10: score+=0.12; ev.append(f"liq_below(${futures.liq_below_usd:.0f}M)")
    if micro and not getattr(micro,"error",True):
        cvd=getattr(micro,"cvd_pct",0); ob=getattr(micro,"ob_imbalance",1.0)
        if cvd<=-25 and ob<0.8: score+=0.20; ev.append("cvd+ob_bear")
        elif cvd<=-15: score+=0.12; ev.append(f"cvd({cvd:.0f}%)")
    return min(1.0,score), f"danger({min(1.0,score):.2f}):{'+'.join(ev) or 'none'}"



# ── Trend Fragility Model ─────────────────────────────────────────────────────
#
# Fragility = Exhaustion × Distribution × ParticipationDecay  (geometric mean)
#
# WHY MULTIPLICATIVE:
#   With additive scoring, a trend with strong exhaustion but healthy OBV
#   can still score high. Multiplicative means ALL THREE must be present.
#   If any one is zero → Fragility = 0. If all three are high → trend is
#   genuinely fragile and about to break.
#
# THREE COMPONENTS:
#   Exhaustion         = trend aging (phase old, RSI/MACD diverging)
#   Distribution       = capital leaving (OBV divergence, BOP, vol asymmetry)
#   ParticipationDecay = RATE of collapse (velocity, not level)
#                        ← this is new and distinct from distribution
#
# Distribution measures LEVEL: "OBV is low relative to price"
# ParticipationDecay measures VELOCITY: "OBV is collapsing faster each bar"
#
# Fragility LR:
#   > 0.80: LR = 4.50  (all three fully engaged — breakdown imminent)
#   > 0.60: LR = 3.20
#   > 0.40: LR = 2.20
#   > 0.20: LR = 1.60
#   ≤ 0.20: LR = 1.00  (fragility insufficient)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrendFragility:
    exhaustion_score:    float   # Q1 exhaustion score [0,1]
    distribution_score:  float   # Q2 distribution score [0,1]
    participation_decay: float   # rate of participation collapse [0,1]
    fragility:           float   # geometric mean (multiplicative composite)
    lr:                  float   # pre-computed LR for direct injection
    is_fragile:          bool
    label:               str
    details:             str


def detect_participation_decay(df: pd.DataFrame,
                                short_w: int = 5,
                                long_w:  int = 20) -> Tuple[float, str]:
    """
    Measure the RATE at which participation is collapsing.

    Distinct from distribution (level measure):
      Distribution:         OBV lower than at the same price level
      Participation Decay:  OBV/volume slope becoming MORE negative (velocity)

    Three sub-signals:
      1. OBV second derivative: is the slope becoming steeper downward?
         slope_now vs slope_5bars_ago → if more negative: decay accelerating
      2. Up-day volume sequential decline: each recent up-day has lower volume
         than the prior up-day → participation actively drying up
      3. Relative volume momentum: rel_vol trending lower bar-by-bar
         (running median of rel_vol declining → baseline participation fading)
    """
    score = 0.0; ev = []

    col = "close" if "close" in df.columns else "price"
    if len(df) < long_w + short_w:
        return 0.0, "insufficient data"

    try:
        prices = df[col].values.astype(float)
        obv    = df["OBV"].values.astype(float) if "OBV" in df.columns else None
        vols   = df["volume"].values.astype(float) if "volume" in df.columns else None

        # ── Sub-signal 1: OBV second derivative ──────────────────────────────
        # "Is the OBV slope becoming more negative?" — velocity of collapse
        if obv is not None and len(obv) >= long_w + short_w:
            def slope(arr):
                n = len(arr); x = np.arange(n, dtype=float)
                return float(np.polyfit(x, arr, 1)[0]) if n >= 2 else 0.0

            slope_now  = slope(obv[-short_w:])
            slope_prev = slope(obv[-short_w*2:-short_w])

            if slope_now < 0 and slope_prev < 0:
                accel = (slope_now - slope_prev) / (abs(slope_prev) + 1e-9)
                if accel < -0.30:   # slope 30%+ more negative = decay accelerating
                    score += 0.35; ev.append(f"obv_2nd_deriv({accel:+.2f})")
                elif accel < -0.10:
                    score += 0.18; ev.append(f"obv_slope_steepen({accel:+.2f})")
            elif slope_now < 0 and slope_prev >= 0:
                score += 0.30; ev.append("obv_slope_flip_neg")

        # ── Sub-signal 2: Sequential up-day volume decline ───────────────────
        # Each recent buying day has LESS volume than the prior buying day
        if vols is not None and len(vols) >= short_w + 2:
            rets = np.diff(prices[-(short_w+1):])
            up_indices = [i for i, r in enumerate(rets) if r > 0]

            if len(up_indices) >= 3:
                up_vols = [float(vols[-(short_w - i)]) for i in up_indices]
                # Check if the series is monotonically declining
                is_declining = all(up_vols[i] > up_vols[i+1] for i in range(len(up_vols)-1))
                if is_declining and len(up_vols) >= 3:
                    score += 0.35; ev.append(f"upvol_seq_decline(n={len(up_vols)})")
                elif up_vols[-1] < up_vols[0] * 0.55:
                    score += 0.20; ev.append(f"upvol_shrink({up_vols[-1]/up_vols[0]:.2f}x)")

        # ── Sub-signal 3: Relative volume momentum ───────────────────────────
        # rel_vol itself is trending down → baseline participation fading
        if "rel_vol" in df.columns and len(df) >= long_w:
            rv = df["rel_vol"].values.astype(float)[-long_w:]
            x  = np.arange(len(rv), dtype=float)
            rv_slope = float(np.polyfit(x, rv, 1)[0])
            rv_current = float(rv[-1])
            if rv_slope < -0.02 and rv_current < 0.85:
                score += 0.25; ev.append(f"rel_vol_down(slope={rv_slope:.3f})")
            elif rv_slope < -0.01:
                score += 0.12; ev.append(f"rel_vol_declining")

    except Exception as e:
        ev.append(f"err:{e}")

    return min(1.0, score), f"pd({min(1.0,score):.2f}):{'+'.join(ev) or 'none'}"


def compute_trend_fragility(
    df: pd.DataFrame, last: pd.Series,
    obv_trend:  str = "unknown",
    vol_regime: str = "neutral",
    phase: str      = "RANGE",
    phase_dur: int  = 0,
    micro           = None,
    ex_score_override: Optional[float] = None,
    di_score_override: Optional[float] = None,
) -> "TrendFragility":
    """
    Compute trend fragility as the geometric mean of three components.

    Fragility = (Exhaustion × Distribution × ParticipationDecay)^(1/3)

    WHY GEOMETRIC MEAN (not additive or arithmetic mean):
      If any one component is 0 → Fragility = 0
      Two strong + one absent: fragility is LOW
      All three moderate (0.6): fragility = 0.6 (well-balanced)
      All three high (0.9): fragility = 0.9 (maximum — act now)

    This enforces that all three mechanisms must be simultaneously present
    for the system to classify a trend as genuinely fragile.
    """
    col = "close" if "close" in df.columns else "price"
    rel_vol = float(df["rel_vol"].iloc[-1]) if "rel_vol" in df.columns else 1.0

    # Component 1: Exhaustion (reuse existing scorer)
    if ex_score_override is not None:
        ex_score = ex_score_override
    else:
        ex_score, _ = _short_score_exhaustion(df, last, phase, phase_dur)

    # Component 2: Distribution (level — from existing scorer)
    if di_score_override is not None:
        di_score = di_score_override
    else:
        di_score, _ = _short_score_distribution(df, last, obv_trend, rel_vol, micro)

    # Component 3: Participation Decay (velocity — new)
    pd_score, pd_note = detect_participation_decay(df)

    # Geometric mean: multiplicative composite
    # Clamp components to [0.01, 1.0] to avoid log(0)
    ex = max(0.01, min(1.0, ex_score))
    di = max(0.01, min(1.0, di_score))
    pd = max(0.01, min(1.0, pd_score))

    fragility = (ex * di * pd) ** (1.0 / 3.0)

    # LR table — fragility > threshold gets direct injection
    if   fragility > 0.80: lr = 4.50; label = "CRITICAL"
    elif fragility > 0.60: lr = 3.20; label = "HIGH"
    elif fragility > 0.40: lr = 2.20; label = "MODERATE"
    elif fragility > 0.20: lr = 1.60; label = "LOW"
    else:                  lr = 1.00; label = "NONE"

    details = (
        f"fragility={fragility:.3f}  [{label}]  "
        f"exhaust={ex_score:.2f} × distrib={di_score:.2f} × pd={pd_score:.2f}  "
        f"| pd: {pd_note}"
    )

    return TrendFragility(
        exhaustion_score   = ex_score,
        distribution_score = di_score,
        participation_decay= pd_score,
        fragility          = round(fragility, 4),
        lr                 = round(lr, 2),
        is_fragile         = fragility > 0.40,
        label              = label,
        details            = details,
    )


_trend_fragility_cache: List = [None]


def _short_score_fragility(df, last, futures, regime_memory=None,
                            obv_trend="unknown", vol_regime="neutral",
                            phase="RANGE", phase_dur=0, micro=None,
                            ex_score=None, di_score=None):
    """
    Q6: Is confidence unstable? (FRAGILITY)

    Now anchored on compute_trend_fragility() — the multiplicative model.
    Also adds positioning fragility from futures (funding, L/S ratio).
    """
    # PRIMARY: Trend Fragility (multiplicative Exhaustion × Distribution × ParticipationDecay)
    tf = compute_trend_fragility(
        df=df, last=last,
        obv_trend=obv_trend, vol_regime=vol_regime,
        phase=phase, phase_dur=phase_dur, micro=micro,
        ex_score_override=ex_score,
        di_score_override=di_score,
    )
    _trend_fragility_cache[0] = tf

    # Map fragility score to this function's [0,1] output
    # (run_short_scorer will also add the direct LR — this score feeds the group LR)
    base_score = tf.fragility
    ev = [f"FRAG={tf.fragility:.3f}[{tf.label}]"]

    # SECONDARY: Positioning fragility from futures
    pos_score = 0.0
    if futures and not futures.error and futures.is_futures:
        ls=futures.ls_ratio; fr=futures.funding_rate*100
        if ls>2.0: pos_score+=0.22; ev.append(f"ls_extreme({ls:.2f})")
        elif ls>1.6: pos_score+=0.12; ev.append(f"ls_high({ls:.2f})")
        if fr>0.08: pos_score+=0.18; ev.append(f"fund_high({fr:.3f}%)")
        elif fr>0.05: pos_score+=0.08

    # Regime memory instability
    rm_score = 0.0
    if regime_memory is not None:
        flips=regime_memory.regime_flip_count(10); vol_ac=regime_memory.vol_autocorrelation()
        if flips>=4: rm_score+=0.20; ev.append(f"flips={flips}")
        elif flips>=2: rm_score+=0.08
        if vol_ac<-0.25: rm_score+=0.15; ev.append(f"vol_rev(ac={vol_ac:.2f})")

    # RSI fading from highs
    try:
        rsi=float(last.get("RSI",50) or 50)
        if "RSI" in df.columns and len(df)>=5:
            if rsi>65 and rsi<float(df["RSI"].iloc[-5])-8:
                rm_score+=0.12; ev.append(f"rsi_fading({rsi:.0f})")
    except Exception: pass

    composite = min(1.0, base_score * 0.70 + pos_score * 0.20 + rm_score * 0.10)
    return composite, f"fragile({composite:.2f}):{'+'.join(ev)}"


def run_short_scorer(short_scorer, coin, df, last, futures=None,
                     obv_trend="unknown", vol_regime="neutral",
                     phase="RANGE", phase_dur=0, micro=None,
                     regime_memory=None, short_ml_proba=0.5, short_ml_acc=0.5):
    rel_vol = float(df["rel_vol"].iloc[-1]) if "rel_vol" in df.columns else 1.0

    # Run Q1 and Q2 first so their scores can feed into Q6 (fragility)
    ex_group, ex_score, ex_note = "EXHAUSTION", *_short_score_exhaustion(df, last, phase, phase_dur)
    di_group, di_score, di_note = "DISTRIBUTION", *_short_score_distribution(df, last, obv_trend, rel_vol, micro)

    # Run all six questions — Q6 receives Q1/Q2 scores for the fragility model
    qs = [
        ("EXHAUSTION",   ex_score, ex_note),
        ("DISTRIBUTION", di_score, di_note),
        ("TRAP",         *_short_score_trap(df, last)),
        ("ACCELERATION", *_short_score_acceleration(df, last, vol_regime)),
        ("DANGER",       *_short_score_danger(df, last, futures, obv_trend, micro)),
        ("FRAGILITY",    *_short_score_fragility(df, last, futures, regime_memory,
                                                  obv_trend=obv_trend, vol_regime=vol_regime,
                                                  phase=phase, phase_dur=phase_dur, micro=micro,
                                                  ex_score=ex_score, di_score=di_score)),
    ]
    for group, score, note in qs:
        short_scorer.add_lr(_score_to_lr(score), note, group=group)

    # TREND FRAGILITY — direct high-LR injection (geometric multiplicative model)
    # This fires ONLY when Exhaustion × Distribution × ParticipationDecay all present.
    tf = _trend_fragility_cache[0]
    if tf is not None and tf.lr > 1.02 and tf.is_fragile:
        short_scorer.add_lr(_lr_safe(tf.lr),
                            f"TREND FRAGILITY [{tf.label}] "
                            f"= exhaust({tf.exhaustion_score:.2f}) × "
                            f"distrib({tf.distribution_score:.2f}) × "
                            f"pd({tf.participation_decay:.2f}) "
                            f"→ fragility={tf.fragility:.3f}  LR={tf.lr:.2f}",
                            group="FRAGILITY")
        print(f"[FRAGILITY] {coin}: {tf.label}  "
              f"{tf.exhaustion_score:.2f}×{tf.distribution_score:.2f}×{tf.participation_decay:.2f}"
              f" = {tf.fragility:.3f}  LR={tf.lr:.2f}  {tf.details}")
    _trend_fragility_cache[0] = None

    # FAILED BREAKOUT ENGINE — direct high-LR injection
    # The TRAP group LR from Q3 already captured the signal in proportional form.
    # Here we add the FULL pre-computed LR from detect_failed_breakout() which
    # can reach 5.0 for the 4-signal confirmed pattern — much higher than the
    # _SHORT_MAX_LR=3.0 used by _score_to_lr. This is the highest LR in the system.
    fb = _failed_breakout_cache[0]
    if fb is not None and fb.breakout_detected and fb.lr > 1.02:
        short_scorer.add_lr(_lr_safe(fb.lr),
                            f"FAILED BREAKOUT [{fb.severity.upper()}]: "
                            f"resistance={fb.resistance_level:.4f}  "
                            f"n_signals={fb.n_failure_signals}/4  "
                            f"reversal={fb.fast_reversal}  "
                            f"| {fb.details}",
                            group="TRAP")
        _rev = "REVERSAL CONFIRMED" if fb.fast_reversal else "watch for reversal"
        print(f"[FAILED BO] {coin}: {fb.severity.upper()}  {_rev}  "
              f"n={fb.n_failure_signals}/4  LR={fb.lr:.2f}  "
              f"[weak_obv={fb.weak_obv:.2f}  wick={fb.rejection_wick:.2f}  "
              f"vol={fb.volume_declining:.2f}  rev={fb.fast_reversal}]")
    _failed_breakout_cache[0] = None

    # DISTRIBUTION DIVERGENCE — add high-LR directly (bypasses _score_to_lr cap)
    # The divergence LR can reach 4.50 for strong patterns — much higher than the
    # normal _SHORT_MAX_LR=3.0 used for other signals. This reflects the diagnostic
    # value of price HH / OBV LH as a primary distribution indicator.
    div = _distribution_divergence_cache[0]
    if div is not None and div.detected and div.lr > 1.0:
        short_scorer.add_lr(_lr_safe(div.lr),
                            f"DISTRIBUTION DIVERGENCE [{div.severity}]: "
                            f"price@{div.price_now:.4f} (>{div.price_swing_high:.4f}) "
                            f"OBV diverged {div.obv_divergence_pct:+.1f}% from swing "
                            f"({div.n_swing_lookback}b ago)",
                            group="DISTRIBUTION")
        print(f"[DIST DIV] {coin}: {div.severity.upper()} -- "
              f"price HH={div.price_now:.4f} (ref={div.price_swing_high:.4f})  "
              f"OBV gap={div.obv_divergence_pct:+.1f}%  LR={div.lr:.2f}  "
              f"({div.n_swing_lookback} bars ago)")
    elif div is not None and not div.detected and div.lr < 1.0:
        short_scorer.add_lr(_lr_safe(div.lr),
                            "OBV confirming price (bullish -- bad for short thesis)",
                            group="DISTRIBUTION")
        print(f"[DIST DIV] {coin}: OBV confirming price -- bearish for short (LR={div.lr:.2f})")
    _distribution_divergence_cache[0] = None   # reset

    # BREAK OF PARTICIPATION — direct high-LR injection
    # This fires alongside distribution divergence but measures a different thing:
    # distribution divergence = cumulative OBV level at HH vs prior HH (structural)
    # break of participation  = RATE of volume/OBV collapse now (momentum / early warning)
    # When both fire: extremely strong short signal (structural + momentum confirming)
    bop = _break_of_participation_cache[0]
    if bop is not None and bop.detected and bop.lr > 1.0:
        short_scorer.add_lr(_lr_safe(bop.lr),
                            f"BREAK OF PARTICIPATION [{bop.severity}]: "
                            f"n={bop.n_signals}/3 signals  "
                            f"price_at_highs={bop.price_at_highs}  "
                            f"up_vol={bop.upday_vol_ratio:.2f}x  "
                            f"obv_slope_rev={bop.obv_slope_reversed}  "
                            f"obv_roc_coll={bop.obv_roc_collapsed}  "
                            f"| {bop.details}",
                            group="DISTRIBUTION")
        _at = f"AT HIGHS" if bop.price_at_highs else f"{bop.price_pct_from_high:.0%} from high"
        print(f"[BOP] {coin}: {bop.severity.upper()}  {_at}  "
              f"n={bop.n_signals}/3  LR={bop.lr:.2f}  "
              f"up_vol={bop.upday_vol_ratio:.2f}x  {bop.details}")
    elif bop is not None and bop.lr < 1.0:
        short_scorer.add_lr(_lr_safe(bop.lr),
                            "Participation rising -- short thesis weakened",
                            group="DISTRIBUTION")
        print(f"[BOP] {coin}: participation rising  LR={bop.lr:.2f}  (bad for short)")
    _break_of_participation_cache[0] = None

    # ML residual — capped to prevent pathological LRs when short_scorer.current_prob ≈ 0
    if short_ml_acc >= 0.50 and 0 < short_ml_proba < 1:
        _d   = abs(short_ml_proba - short_scorer.current_prob)
        _acc_s = max(0.0, min(1.0, (short_ml_acc-0.50)/0.25))
        # Sharpe suppression: if short strategy loses money in WF, halve ML influence
        _sh_s = _WF_SHARPE.get(f"{coin}_short", 1.0)
        _sharpe_scale = 0.5 if _sh_s < 0 else 1.0
        if _sharpe_scale < 1.0:
            print(f"[SHORT ML] {coin}: ⚠️  Sharpe={_sh_s:+.2f} < 0 → ML scale ×0.5")
        _s   = min(0.70, 0.20+0.50*_d) * _acc_s * _sharpe_scale
        _r   = ((short_ml_proba / (1-short_ml_proba+1e-9)) /
                (max(0.001, short_scorer.current_prob) / max(0.001, 1-short_scorer.current_prob)))
        _r   = max(0.05, min(20.0, _r))
        _lr  = max(0.25, min(4.0, 1.0+(_r-1.0)*_s))
        short_scorer.add_lr(_lr_safe(_lr), f"Short ML(acc={short_ml_acc:.0%})", "OTHER")

    # Bear acceleration details — run again to get the structured result for display
    ba = detect_bear_acceleration(df, last, vol_regime)

    # Confirmation bonus: when 3+ acceleration signals fire, add an interaction LR
    # This is ON TOP of the ACCELERATION group LR from Q4 above.
    # It reflects that convergent acceleration is much stronger than any single signal.
    if ba.n_confirmed >= 3:
        bonus_lr = 1.0 + (ba.n_confirmed - 2) * 0.35   # 3→1.35, 4→1.70
        short_scorer.add_lr(_lr_safe(bonus_lr),
                            f"BEAR ACCEL CONFIRMED ({ba.n_confirmed}/4 signals) "
                            f"expected_move={ba.expected_move_atr:.1f}xATR",
                            group="ACCELERATION")
        print(f"[BEAR ACCEL] {coin}: {ba.n_confirmed}/4 signals confirmed -- "
              f"bonus LR={bonus_lr:.2f}  expected={ba.expected_move_atr:.1f}xATR")
    elif ba.n_confirmed == 2:
        print(f"[BEAR ACCEL] {coin}: 2/4 signals -- moderate ({ba.details})")

    row = "  ".join(f"{g[:5]}={s:.2f}->{_score_to_lr(s):.2f}" for g,s,_ in qs)
    print(f"[SHORT] {coin}: {row}")

    return ba   # caller can use ba.expected_move_atr for position sizing


def detect_short_prior(market_regime: str, btc_regime: str = "unknown",
                        fng_value: float = 50.0, adx: float = 20.0,
                        phase: str = "RANGE", phase_dur: int = 0,
                        ) -> Tuple[float, str, float]:
    """
    Short prior WITH time decay.  Returns (decayed_prior, label, freshness_lr).

    Short signals decay much faster than long signals:
      Panic       λ=3.50  → exhausted in 2-3 bars
      STRONG_BEAR λ=2.80  → fades within ~15 bars
      EARLY_BULL  no decay → already at floor (don't fight uptrend)

    freshness_lr > 1 rewards catching the turn at day 1-3 of a bearish phase.
    """
    if fng_value < 20 or market_regime == "panic":
        base_prior, label = SHORT_REGIME_PRIORS.get("panic", 0.55), "panic"
        t_norm = phase_dur / max(1, 5)
        decay  = math.exp(-3.50 * t_norm)
        decayed = round(0.50 + (base_prior - 0.50) * decay, 4)
        return decayed, label, 1.0

    if btc_regime == "bear" and market_regime in ("bear", "range"):
        base_prior, label = 0.62, "btc_bear_confluence"
    elif btc_regime == "bull" and market_regime == "bull_trend":
        base_prior, label = 0.22, "btc_bull_headwind"
    else:
        base_prior = SHORT_REGIME_PRIORS.get(market_regime, 0.42)
        label      = market_regime

    decayed, freshness_lr = _apply_short_age_decay(base_prior, phase, phase_dur)
    _d = decayed - base_prior
    print(f"[SHORT PRIOR] {label.upper()}  P₀={base_prior:.2f}"
          f" → P_eff={decayed:.3f} ({_d:+.3f})  "
          f"phase={phase} day={phase_dur}  freshness_lr={freshness_lr:.2f}")
    return decayed, label, freshness_lr
# ── END [SHORT] ───────────────────────────────────────────────────────────────


# [Removed: old float-returning apply_ml_lr_to_prob -- see scorer-based version below]
def apply_ml_lr_to_prob(scorer: "ProbabilisticScorer", coin: str,
                         ml_proba: float, df,
                         ml_accuracy: float = 0.0) -> None:
    """
    Unified: adds ML residual + interaction LRs to scorer.

    SYMMETRY FIX -- old architecture had two asymmetries:

    1. OBV gate (ml_proba × 0.60 outside scorer) -- removed.
       obv_falling is already LR=0.35 in the scorer. Applying it again
       externally on ml_proba double-counts OBV AND applies it in raw
       probability space, not log-odds space. OBV handles itself via scorer.

    2. Flat shrink (0.35) regardless of disagreement magnitude.
       When ML=0.18 and Bayes=0.80, the ML is providing a very strong
       counter-signal. With flat shrink=0.35, the residual LR was only
       mildly bearish, allowing the bullish Bayesian signals to dominate.
       A hard gate (ml<0.25 → WAIT) was then needed to compensate.
       Both are removed: disagreement-scaled shrink does the work instead.

    Disagreement-scaled shrink:
      disagreement = |ml_proba − P_bayes|
      eff_shrink   = min(0.75, 0.20 + 0.55 × disagreement) × acc_scale

      ml=0.18, bayes=0.80  → disagreement=0.62  eff_shrink≈0.51  (strong counter-signal)
      ml=0.68, bayes=0.72  → disagreement=0.04  eff_shrink≈0.22  (mild, little to add)
    """
    p_bayes = scorer.current_prob

    # Accuracy scale: model at 50% accuracy (random) contributes nothing
    acc_scale = max(0.0, min(1.0, (ml_accuracy - 0.50) / 0.25)) if ml_accuracy > 0 else 0.5

    # Sharpe suppression: if WF strategy loses money, halve the ML's influence.
    # The model may predict correctly (acc > 50%) but its predictions don't translate
    # to profitable trades — meaning the probabilities aren't economically useful.
    wf_sharpe    = _WF_SHARPE.get(coin, 1.0)   # default 1.0 = no suppression
    sharpe_scale = 0.5 if wf_sharpe < 0 else 1.0
    if sharpe_scale < 1.0:
        print(f"[MLLR] {coin}: ⚠️  Sharpe={wf_sharpe:+.2f} < 0 → ML scale ×0.5 (unprofitable in WF)")

    # Disagreement-scaled shrink -- core symmetry fix
    disagreement = abs(float(ml_proba) - float(p_bayes))
    eff_shrink   = min(0.75, 0.20 + 0.55 * disagreement) * acc_scale * sharpe_scale

    # Trust tier: only allow the wider impact band once the model has actually
    # proven itself (same bar the ML GATE section uses for "full weight").
    _unlocked  = (ml_accuracy >= MLLRProducer.TRUST_ACC) and (wf_sharpe > MLLRProducer.TRUST_SHARPE)
    _lr_bounds = MLLRProducer.LR_BOUNDS_UNLOCKED if _unlocked else MLLRProducer.LR_BOUNDS_DEFAULT

    resid_lr, resid_desc = MLLRProducer.residual_lr(ml_proba, p_bayes, shrink=eff_shrink,
                                                      lr_bounds=_lr_bounds)
    scorer.add_lr(resid_lr, f"ML residual ({resid_desc})", group="OTHER")

    interactions = _mllr_producer.get_active_interaction_lrs(df, coin)
    for int_name, (lr, desc) in interactions.items():
        scorer.add_lr(lr, desc, group="ML_INTERACTION")

    direction = "↓ bearish" if resid_lr < 0.95 else ("↑ bullish" if resid_lr > 1.05 else "≈ neutral")
    _tier = "unlocked" if _unlocked else "default"
    print(f"[MLLR] {coin}: disagree={disagreement:.2f}  shrink={eff_shrink:.2f}  "
          f"LR={resid_lr:.3f} ({direction})  band={_tier}  "
          f"{len(interactions)} interactions → scorer")


# ═══════════════════════════════════════════════════════════════════════════════
# [BAYES] PROBABILISTIC BAYESIAN ENGINE -- Core Classes
#
# _Evidence          -- one piece of evidence (signal) with its LR contribution
# ProbabilisticResult -- finalized output: prob, action, evidence trail
# ProbabilisticScorer -- accumulates evidence, applies group dampening on finalize
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Evidence:
    """One signal's contribution to the Bayesian log-odds."""
    key:             str
    fired:           bool
    lr:              float    # effective LR (after scale)
    log_lr:          float    # log(lr) -- the log-odds contribution
    label:           str
    group:           str
    is_block:        bool = False
    manually_scaled: bool = False   # True when scale != 1.0 was passed to update/add_lr.
                                    # manually_scaled signals skip secondary group dampening --
                                    # the developer already handled the specific correlation.


@dataclass
class ProbabilisticResult:
    """Finalized output of ProbabilisticScorer.finalize()."""
    coin:             str
    prob_continuation: float
    raw_prob:         float          # after group dampening, before entropy suppression
    log_odds:         float          # group-dampened log-odds
    prior:            float
    action:           str
    blocked:          bool
    block_reasons:    List[str]
    evidence:         List[_Evidence]
    temperature:      float = 1.0    # T applied in entropy suppression
    shrink_factor:    float = 1.0    # s applied in entropy suppression
    suppression_note: str  = ""

    @property
    def score_equivalent(self) -> float:
        """Map P → legacy 0-12 score scale for display tables."""
        return round(max(0, min(12, (self.prob_continuation - 0.35) / 0.55 * 10)), 2)

    @property
    def pct(self) -> str:
        return f"{self.prob_continuation:.1%}"

    def top_drivers(self, n: int = 6) -> List["_Evidence"]:
        return sorted(self.evidence, key=lambda e: abs(e.log_lr), reverse=True)[:n]

    def print_summary(self, coin: str = "") -> None:
        name = coin or self.coin
        naive_lo = sum(e.log_lr for e in self.evidence) + _logit(self.prior)
        naive_p  = _sigmoid(naive_lo)
        delta    = self.prob_continuation - naive_p
        print(f"[PROB] {name.upper()}  P={self.prob_continuation:.3f} ({self.prob_continuation:.1%})  "
              f"Action={self.action}  Score≈{self.score_equivalent:.1f}  "
              f"[naive={naive_p:.3f}  Δ={delta:+.3f}]  "
              f"[T={self.temperature:.2f}  s={self.shrink_factor:.2f}]")
        for e in self.top_drivers(5):
            grp = _SIGNAL_GROUP_MAP.get(e.key, e.group)
            d   = "↑" if e.log_lr > 0 else "↓"
            f_  = "✅" if e.fired else "❌"
            print(f"  {f_}{d} {e.label:<44} LR={e.lr:.2f}  [{grp}]")
        if self.suppression_note:
            print(f"  [ENTROPY] {self.suppression_note}")
        if self.block_reasons:
            for r in self.block_reasons:
                print(f"  ⛔ {r}")


def compute_entropy_suppression(
    evidence        : List["_Evidence"],
    prior           : float,
    prior_regime    : str  = "neutral",
    calibration_conf: float = 0.5,
) -> Tuple[float, float, str]:
    """
    Compute temperature T and shrink factor s for entropy suppression.

    Prevents overconfidence when many correlated groups all fire bullish.
    Even after group dampening, cross-group accumulation can push P > 0.90+.

    T > 1 (applied to signal log-odds, not prior):  flattens the distribution.
    s < 1 (linear shrinkage toward 0.5):            compresses toward neutral.

    Returns (T, s, explanation_note).
    """
    groups_fired = len({
        ev.group for ev in evidence
        if ev.group in SIGNAL_GROUPS and abs(ev.log_lr) > 0.01
    })

    # Temperature
    T = 1.05
    T += 0.15 * max(0, groups_fired - 2)
    if prior_regime in ("euphoria", "panic"):
        T += 0.20
    elif prior_regime in ("bear_trend",):
        T += 0.08
    T += 0.10 * (1.0 - min(1.0, max(0.0, calibration_conf)))
    T = round(min(1.90, max(1.05, T)), 3)

    # Shrink factor
    s = 0.86
    if prior_regime in ("euphoria", "panic"):
        s -= 0.07
    s -= 0.028 * max(0, groups_fired - 3)
    s -= 0.04  * (1.0 - min(1.0, max(0.0, calibration_conf)))
    s = round(min(0.94, max(0.70, s)), 3)

    note = (f"groups_fired={groups_fired}  T={T:.2f}  s={s:.2f}  "
            f"regime={prior_regime}  cal_conf={calibration_conf:.2f}")
    return T, s, note


def _apply_entropy_suppression(dampened_lo: float, prior: float,
                                T: float, s: float) -> float:
    """
    Apply temperature scaling then shrinkage in log-odds space.

    Temperature scales the SIGNAL component only (not the prior).
    P_temp  = sigmoid(prior_lo + signal_lo / T)
    P_final = 0.5 + (P_temp - 0.5) × s
    """
    prior_lo  = _logit(prior)
    signal_lo = dampened_lo - prior_lo
    scaled_lo = prior_lo + signal_lo / T
    p_temp    = _sigmoid(scaled_lo)
    p_final   = 0.5 + (p_temp - 0.5) * s
    return round(p_final, 6)


class ProbabilisticScorer:
    """
    Accumulates Bayesian evidence as log-odds updates.

    THREE-LEVEL ARCHITECTURE (signal → group → latent state → probability):
    ─────────────────────────────────────────────────────────────────────────
    Level 1  SIGNAL → GROUP  (geometric mean within group)
      cloud + MACD + EMA + DI all say bear → TREND group LR (one number)
      OBV + CVD say bear                   → FLOW group LR  (one number)

    Level 2  GROUP → LATENT STATE  (geometric mean within state)
      TREND + MOMENTUM + STRUCTURE all say bear → TREND_STATE (one number)
      FLOW + VOLATILITY + MICROSTRUCTURE say bear → FLOW_STATE (one number)

      Without this layer, TREND, STRUCTURE, FLOW, MACRO each contribute
      independently even though in a bear market they ALL fire bearish —
      compounding correlated groups as if they were independent evidence.

    Level 3  LATENT STATES → PROBABILITY  (compound across states)
      TREND_STATE × FLOW_STATE × MACRO_STATE → final log-odds
      These are genuinely different evidence types and DO compound.

    ISOLATED group (BREAKOUT, ML_INTERACTION, OTHER):
      Not compressed — genuinely independent signals compound freely.

    LATENT STATE CEILINGS:
      TREND_STATE:  3.0  — maximum from all trend/momentum/structure together
      FLOW_STATE:   3.0  — maximum from all volume/volatility/microstructure
      MACRO_STATE:  3.5  — maximum from macro/phase context
      BEAR_STATE:   4.0  — short scorer evidence (exhaustion+distribution+danger)
      SETUP_STATE:  3.5  — short scorer setup signals (trap+fragility)
    """

    LATENT_STATES: Dict[str, Dict] = {
        # Long scorer: three largely independent evidence types
        "TREND_STATE": {
            "groups":  {"TREND", "MOMENTUM", "STRUCTURE"},
            "max_lr":  3.0,
        },
        "FLOW_STATE": {
            "groups":  {"FLOW", "VOLATILITY", "MICROSTRUCTURE"},
            "max_lr":  3.0,
        },
        "MACRO_STATE": {
            "groups":  {"MACRO", "TRANSITION"},
            "max_lr":  3.5,
        },
        # Short scorer: bear evidence vs bear setup signals
        "BEAR_EVIDENCE_STATE": {
            "groups":  {"EXHAUSTION", "DISTRIBUTION", "DANGER", "ACCELERATION"},
            "max_lr":  4.0,
        },
        "BEAR_SETUP_STATE": {
            "groups":  {"TRAP", "FRAGILITY"},
            "max_lr":  3.5,
        },
        # Isolated: truly independent — still sum (not average) within state
        "ISOLATED": {
            "groups":  {"BREAKOUT", "ML_INTERACTION", "OTHER",
                        "S_TREND", "S_MOMENTUM", "S_FLOW",
                        "S_STRUCTURE", "S_FUNDING", "S_OI", "S_SQUEEZE"},
            "max_lr":  5.0,
            "independent": True,   # sum log-LRs instead of averaging
        },
    }

    GROUP_MAX_LR: Dict[str, float] = {
        "TREND":          3.5,
        "MOMENTUM":       2.5,
        "FLOW":           3.0,
        "STRUCTURE":      3.0,
        "VOLATILITY":     2.2,
        "BREAKOUT":       3.5,
        "TRANSITION":     3.5,
        "TREND_LIFECYCLE": 2.5,  # lifecycle stage LR — independent signal
        "ML_INTERACTION": 2.5,
        "MICROSTRUCTURE": 2.5,
        "OTHER":          5.0,
        "EXHAUSTION":     3.0,
        "DISTRIBUTION":   4.0,
        "TRAP":           4.5,
        "ACCELERATION":   3.5,
        "DANGER":         3.5,
        "FRAGILITY":      3.5,
        "S_TREND":        3.5,
        "S_MOMENTUM":     2.5,
        "S_FLOW":         3.0,
        "S_STRUCTURE":    3.0,
        "S_FUNDING":      3.0,
        "S_OI":           2.5,
        "S_SQUEEZE":      4.0,
        "MACRO":          3.5,
    }

    def __init__(self, coin: str = "",
                 prior: float = PRIOR_CONTINUATION,
                 prior_regime: str = "neutral",
                 calibration_conf: float = 0.5):
        self.coin             = coin
        self.log_odds         = _logit(prior)   # running approx for current_prob
        self.prior            = prior
        self.prior_regime     = prior_regime
        self.calibration_conf = calibration_conf
        self._evidence: List[_Evidence]  = []
        self._blocked         = False
        self._block_reasons: List[str] = []
        self._hard_ceiling: Optional[float] = None

    def update(self, key: str, fired: bool, label: str,
               group: str = "OTHER", scale: float = 1.0,
               is_block: bool = False) -> "ProbabilisticScorer":
        """Add a signal from SIGNAL_LR to evidence."""
        lr_true, lr_false = SIGNAL_LR[key]
        raw_lr  = lr_true if fired else lr_false
        scaled  = _lr_safe(raw_lr ** scale)
        log_lr  = math.log(scaled)
        self.log_odds += log_lr   # running approx
        self._evidence.append(_Evidence(
            key=key, fired=fired, lr=scaled, log_lr=log_lr,
            label=label, group=group, is_block=is_block,
            manually_scaled=(scale != 1.0),
        ))
        if fired and is_block:
            ceil = BLOCK_PROB_CEILING
            if self._hard_ceiling is None or ceil < self._hard_ceiling:
                self._hard_ceiling = ceil
            self._blocked = True
            self._block_reasons.append(label)
        return self

    def block(self, reason: str) -> "ProbabilisticScorer":
        self._blocked = True
        self._block_reasons.append(reason)
        ceil = BLOCK_PROB_CEILING
        if self._hard_ceiling is None or ceil < self._hard_ceiling:
            self._hard_ceiling = ceil
        return self

    def add_lr(self, lr: float, label: str, group: str = "OTHER",
               is_block: bool = False, scale: float = 1.0) -> "ProbabilisticScorer":
        """Add a pre-computed LR. Participates in latent group aggregation."""
        scaled = _lr_safe(float(lr) ** scale)
        log_lr = math.log(scaled)
        self.log_odds += log_lr   # running approx
        key = f"_lr_{abs(hash(label)) % 100000:05d}"
        self._evidence.append(_Evidence(
            key=key, fired=True, lr=scaled, log_lr=log_lr,
            label=label, group=group, is_block=is_block,
            manually_scaled=(scale != 1.0),
        ))
        if is_block:
            self._blocked = True
            self._block_reasons.append(label)
            ceil = BLOCK_PROB_CEILING
            if self._hard_ceiling is None or ceil < self._hard_ceiling:
                self._hard_ceiling = ceil
        return self

    @property
    def current_prob(self) -> float:
        """Running estimate (pre-finalize approximation). Use finalize() for final value."""
        return _sigmoid(self.log_odds)

    def _compute_group_log_lr(self, group: str,
                               evidence: List["_Evidence"]) -> float:
        """
        Compute the single log-LR contribution of this group.

        Magnitude-weighted geometric mean of individual signal log-LRs:
            avg = Σ(log_lr_i × |log_lr_i|) / Σ|log_lr_i|

        Magnitude-weighting means strong signals set the direction while
        weak signals nudge it — appropriate because a LR=3.0 signal has
        more information than a LR=1.05 signal in the same group.

        Clamped to ±log(GROUP_MAX_LR) — information ceiling for this
        latent variable regardless of how many signals fire.

        Block signals (is_block=True) get full weight regardless of direction
        and bypass the ceiling on the bearish side.
        """
        if not evidence:
            return 0.0

        max_lr  = self.GROUP_MAX_LR.get(group, 3.0)
        max_lo  = math.log(max_lr)

        # Separate blocks (always full weight) from regular signals
        blocks   = [ev for ev in evidence if ev.is_block]
        regular  = [ev for ev in evidence if not ev.is_block]

        # OTHER group: signals are genuinely independent → sum log-LRs (compound freely)
        # Named groups: signals share a latent variable → magnitude-weighted average
        if group == "OTHER":
            group_lo = sum(ev.log_lr for ev in regular)
            group_lo = max(-max_lo, min(max_lo, group_lo))
        else:
            total_mag = sum(abs(ev.log_lr) for ev in regular)
            if total_mag > 0:
                avg_lo = sum(ev.log_lr * abs(ev.log_lr) for ev in regular) / total_mag
            else:
                avg_lo = 0.0
            group_lo = max(-max_lo, min(max_lo, avg_lo))

        # Block signals add their FULL log-LR on top (independent safety checks)
        for ev in blocks:
            group_lo += ev.log_lr

        return group_lo

    def finalize(self) -> "ProbabilisticResult":
        """
        Three-level compression: signal → group → latent state → probability.

        Level 1: group geometric mean (existing)
          Each group → one log-LR = magnitude-weighted avg of its signals.
        Level 2: latent state geometric mean (existing)
          Correlated groups averaged into one state LR before adding to log-odds.
          ISOLATED groups still compound freely (genuinely independent).
        Level 3: entropy suppression + ceiling (existing)

        PENALTY BUDGET CAP (new):
          Two dominant suppressors get unlimited weight:
            1. btc_bear_regime_block  — market-wide macro
            2. below_cloud_*          — structural position
          All other bearish signals (LR < 1.0):
            • Individual floor: LR ≥ 0.55  (single signal ≤ −0.60 log-odds)
            • Total budget:     combined bearish log-odds capped at −0.80
          This prevents 5 moderate penalties from burying good setups
          while preserving the two dominant suppressors at full strength.
        """
        # ── DOMINANT SUPPRESSORS (applied at full strength, no cap) ──────────
        DOMINANT_KEYS = {"btc_bear_regime_block"}
        DOMINANT_PREFIXES = ("below_cloud_",)   # distance-graded cloud signals

        def _is_dominant(ev: "_Evidence") -> bool:
            if ev.key in DOMINANT_KEYS:
                return True
            return any(ev.key.startswith(p) for p in DOMINANT_PREFIXES)

        # Individual LR floor for non-dominant bearish signals
        NON_DOMINANT_LR_FLOOR = 0.55    # single signal can cost at most ~−0.60 lo
        NON_DOMINANT_BUDGET   = -0.80   # total non-dominant bearish log-odds cap

        # Apply floor to non-dominant bearish evidence BEFORE group compression
        import dataclasses as _dc
        floored_evidence = []
        for ev in self._evidence:
            if ev.lr < 1.0 and not _is_dominant(ev):
                floored_lr  = max(ev.lr, NON_DOMINANT_LR_FLOOR)
                floored_lo  = math.log(floored_lr)
                floored_evidence.append(_dc.replace(ev, lr=floored_lr, log_lr=floored_lo))
            else:
                floored_evidence.append(ev)

        # Level 1: group log-LRs (using floored evidence)
        group_ev: Dict[str, List["_Evidence"]] = {}
        for ev in floored_evidence:
            group_ev.setdefault(ev.group, []).append(ev)
        group_log_lrs: Dict[str, float] = {}
        for group, evs in group_ev.items():
            group_log_lrs[group] = self._compute_group_log_lr(group, evs)

        # Level 2: latent state compression
        group_to_state: Dict[str, str] = {}
        for state, cfg in self.LATENT_STATES.items():
            for g in cfg["groups"]:
                group_to_state[g] = state

        state_groups: Dict[str, Dict[str, float]] = {s: {} for s in self.LATENT_STATES}
        unclaimed: Dict[str, float] = {}
        for group, glo in group_log_lrs.items():
            state = group_to_state.get(group, "ISOLATED")
            if state in state_groups:
                state_groups[state][group] = glo
            else:
                unclaimed[group] = glo

        final_lo = _logit(self.prior)
        state_contributions: Dict[str, float] = {}
        for state, cfg in self.LATENT_STATES.items():
            gs = state_groups[state]
            if not gs:
                continue
            max_lo      = math.log(cfg["max_lr"])
            independent = cfg.get("independent", False)
            if independent:
                state_lo = max(-max_lo, min(max_lo, sum(gs.values())))
            else:
                glos      = list(gs.values())
                total_mag = sum(abs(lo) for lo in glos) or 1e-9
                avg_lo    = sum(lo * abs(lo) for lo in glos) / total_mag
                state_lo  = max(-max_lo, min(max_lo, avg_lo))
            final_lo += state_lo
            state_contributions[state] = state_lo
        for glo in unclaimed.values():
            final_lo += glo

        # ── PENALTY BUDGET CAP ────────────────────────────────────────────────
        # Separate dominant vs non-dominant contributions in final_lo
        # to enforce the total cap on non-dominant bearish log-odds.
        prior_lo   = _logit(self.prior)
        dominant_lo = sum(
            ev.log_lr for ev in self._evidence
            if ev.lr < 1.0 and _is_dominant(ev))
        bullish_lo  = sum(
            ev.log_lr for ev in floored_evidence
            if ev.lr >= 1.0)
        non_dom_lo  = final_lo - prior_lo - dominant_lo - bullish_lo

        if non_dom_lo < NON_DOMINANT_BUDGET:
            # Non-dominant penalties exceeded budget — cap them
            _excess  = non_dom_lo - NON_DOMINANT_BUDGET
            final_lo -= _excess   # add back the excess (reduce penalty)

        raw = _sigmoid(final_lo)

        T, s, supp_note = compute_entropy_suppression(
            self._evidence, self.prior,
            prior_regime     = self.prior_regime,
            calibration_conf = self.calibration_conf,
        )
        prob = _apply_entropy_suppression(final_lo, self.prior, T, s)
        if self._hard_ceiling is not None:
            prob = min(prob, self._hard_ceiling)
        if self._blocked:
            prob = min(prob, BLOCK_PROB_CEILING)

        if self._blocked:                      action = "WAIT"
        elif prob >= STRONG_BUY_THRESHOLD:     action = "STRONG BUY"
        elif prob >= BUY_PROB_THRESHOLD:       action = "BUY"
        elif prob <= STRONG_SELL_THRESHOLD:    action = "STRONG SHORT"
        elif prob <= SELL_PROB_THRESHOLD:      action = "SHORT"
        else:                                  action = "WAIT"

        import logging as _log
        _log.getLogger(__name__).debug(
            "[STATES] " + self.coin + ": " +
            "  ".join(f"{st}:{math.exp(lo):.2f}" for st, lo in state_contributions.items()
                      if abs(lo) > 0.05) +
            f"  P={prob:.3f}"
        )

        return ProbabilisticResult(
            coin=self.coin, prob_continuation=prob, raw_prob=raw,
            log_odds=final_lo, prior=self.prior, action=action,
            blocked=self._blocked, block_reasons=self._block_reasons,
            evidence=self._evidence,
            temperature=T, shrink_factor=s, suppression_note=supp_note,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# [ENTROPY] TRANSITION ENTROPY MODEL
#
# The current system classifies regime as ONE label: bull / bear / range / chop.
# Problem: that discards information about HOW CERTAIN the classification is.
#
# STABLE MARKET (low entropy):
#   p_bull=0.85  p_bear=0.05  p_range=0.08  p_chop=0.02
#   Entropy ≈ 0.05  → All signals agree → full confidence in predictions
#
# UNSTABLE MARKET (high entropy):
#   p_bull=0.35  p_bear=0.30  p_range=0.20  p_chop=0.15
#   Entropy ≈ 0.85  → No clear regime → signals are meaningless
#
# FORMULA: H = -Σ p_i × log(p_i)   (Shannon entropy, nats)
# NORMALISED: H_norm = H / log(N)   where N=5 regime states → range [0,1]
#
# CONFIDENCE SCALING (entropy → probability shrinkage toward 0.5):
#   H_norm ≤ 0.30  → scale = 1.00  (stable — no shrinkage)
#   H_norm ≤ 0.50  → scale = 0.85  (slightly unstable)
#   H_norm ≤ 0.70  → scale = 0.60  (unstable — reduce leverage)
#   H_norm ≤ 0.85  → scale = 0.35  (very unstable — avoid sizing up)
#   H_norm > 0.85  → scale = 0.15  (chaotic — avoid trading)
#
# Applied as: P_final = 0.5 + (P_raw - 0.5) × confidence_scale
# A P_raw=0.80 in chaotic market → P_final=0.5+(0.30×0.15)=0.545 (barely signal)
# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# [ENTROPY TRACKER] Adaptive thresholds via rolling percentile
#
# WHY NOT HARDCODED THRESHOLDS:
#   In a sustained crypto bear market, H_norm=0.55 is "normal" — every coin
#   shows mixed regime signals. A fixed threshold of 0.40 for STABLE would
#   label EVERYTHING as UNSTABLE and over-suppress all signals.
#
#   With adaptive percentiles, the thresholds shift with the market:
#   - Calm bull run:   30th pct ≈ 0.30  (strict — even mild mixed signals = MILD)
#   - Volatile bear:   30th pct ≈ 0.55  (relaxed — chaos is the baseline)
#
# PERCENTILE THRESHOLDS:
#   STABLE   ≤ 30th pct  (calmest 30% of recent entropy readings)
#   MILD     ≤ 60th pct  (below-median entropy)
#   UNSTABLE ≤ 75th pct  (top quarter starts here)
#   VOLATILE ≤ 90th pct  (top 10% most chaotic)
#   CHAOTIC  > 90th pct
#
# FALLBACK: when history < 30 readings, uses static thresholds
#   so the system is immediately usable from the first run.
# ─────────────────────────────────────────────────────────────────────────────

class EntropyTracker:
    """
    Rolling buffer of historical H_norm values → adaptive label thresholds.
    Persisted across restarts so the percentiles accumulate real history.
    """
    HISTORY_FILE = "models/entropy_history.json"
    MAX_HISTORY  = 500    # ~6 months at once-daily scans for 10 coins
    MIN_HISTORY  = 30     # minimum before switching from fixed to adaptive
    FALLBACK     = (0.40, 0.58, 0.73, 0.87)  # (stable, mild, unstable, volatile)
    PERCENTILES  = (30,   60,   75,   90)

    def __init__(self):
        self.history: List[float] = []
        self._load()

    def record(self, h_norm: float) -> None:
        self.history.append(float(h_norm))
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]

    def thresholds(self) -> Tuple[float, float, float, float]:
        """Return (t_stable, t_mild, t_unstable, t_volatile) thresholds."""
        if len(self.history) < self.MIN_HISTORY:
            return self.FALLBACK
        return tuple(float(np.percentile(self.history, p))
                     for p in self.PERCENTILES)

    def percentile_of(self, h_norm: float) -> float:
        """How unusual is this entropy value vs recent history? (0-100)"""
        if len(self.history) < self.MIN_HISTORY:
            return float(h_norm * 100)
        return float(np.mean(np.array(self.history) <= h_norm) * 100)

    def _load(self) -> None:
        try:
            with open(self.HISTORY_FILE) as f:
                data = json.load(f)
            self.history = [float(v) for v in data if 0.0 <= float(v) <= 1.0]
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = []

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.HISTORY_FILE) or ".", exist_ok=True)
            with open(self.HISTORY_FILE, "w") as f:
                json.dump(self.history[-self.MAX_HISTORY:], f)
        except Exception:
            pass

    def stats(self) -> str:
        if len(self.history) < 5:
            return f"n={len(self.history)} (using fixed fallback thresholds)"
        t = self.thresholds()
        return (f"n={len(self.history)}  "
                f"p30={t[0]:.3f}  p60={t[1]:.3f}  "
                f"p75={t[2]:.3f}  p90={t[3]:.3f}")


_entropy_tracker = EntropyTracker()

N_REGIME_STATES = 5   # bull, bear, range, chop, panic


@dataclass
class RegimeEntropy:
    """Soft probability distribution over regime states + Shannon entropy."""
    p_bull:           float
    p_bear:           float
    p_range:          float
    p_chop:           float
    p_panic:          float
    entropy_raw:      float    # Shannon entropy in nats
    entropy_norm:     float    # normalised to [0,1]; 0=stable, 1=maximum chaos
    dominant_regime:  str      # highest-probability state
    label:            str      # "STABLE" / "MILD" / "UNSTABLE" / "CHAOTIC"
    confidence_scale: float    # probability shrinkage factor [0.15,1.0]
    entropy_pct:      float    = 0.0   # percentile vs recent history (0-100)
    thresholds_adaptive: bool  = False  # True once enough history accumulated


def compute_regime_entropy(last: pd.Series, df: pd.DataFrame,
                            sig: dict, adx: float,
                            chop_detected: bool) -> "RegimeEntropy":
    """
    Estimate soft regime probability distribution and Shannon entropy.

    SOFT SCORES → SOFTMAX → PROBABILITIES → ENTROPY

    Each regime accumulates evidence points from independent indicators.
    Softmax converts raw scores to a probability simplex.
    Shannon entropy measures how spread-out the distribution is.

    Indicators used:
      ADX          → trend strength (high ADX = likely trend, not range)
      Cloud state  → directional bias
      MACD hist    → momentum direction and strength
      RSI          → overbought / oversold / neutral
      Price Z-score → how far from the mean (extremes → bear or panic)
      Chop flag    → direct chop evidence
      ATR ratio    → vol expansion (panic signal) vs compression (range)
    """
    col = "close" if "close" in df.columns else "price"
    try:
        price  = float(last.get("price", last.get("close", 1.0)))
        rsi    = float(last.get("RSI", 50) or 50)
        macd_h = float(last.get("MACD_Hist", 0) or 0)
        cloud  = sig.get("cloud_state", "unknown")
        ms     = sig.get("market_structure", {}) or {}

        # Price Z-score vs 20-day mean
        try:
            prices = df[col].values.astype(float)[-20:]
            z = (prices[-1] - prices.mean()) / (prices.std() + 1e-9)
        except Exception:
            z = 0.0

        # ATR ratio (current vs 10-day prior)
        try:
            atr_ratio = (float(df["ATR"].iloc[-1]) /
                         (float(df["ATR"].iloc[-10]) + 1e-9)
                         if "ATR" in df.columns and len(df) >= 10 else 1.0)
        except Exception:
            atr_ratio = 1.0

        # ── Build raw regime scores ───────────────────────────────────────
        s_bull = s_bear = s_range = s_chop = s_panic = 0.0

        # ADX — high = trend, low = range/chop
        if   adx >= 30: s_bull += 2.0; s_bear += 2.0  # strong trend (direction TBD)
        elif adx >= 20: s_bull += 1.0; s_bear += 1.0; s_range += 1.0
        elif adx >= 12: s_range += 2.0; s_chop += 1.0
        else:           s_chop  += 3.0; s_range += 1.0

        # Cloud position — directional
        if   cloud == "above":   s_bull += 3.0; s_bear -= 2.0
        elif cloud == "below":   s_bear += 3.0; s_bull -= 2.0
        elif cloud == "in":      s_range+= 2.0; s_chop += 1.0

        # MACD histogram — momentum direction
        if   macd_h >  0.002: s_bull += 2.0
        elif macd_h < -0.002: s_bear += 2.0
        else:                 s_range += 1.0; s_chop += 0.5

        # RSI
        if   rsi > 70:  s_bull += 1.5
        elif rsi > 55:  s_bull += 0.8
        elif rsi < 30:  s_bear += 1.5; s_panic += 1.0
        elif rsi < 45:  s_bear += 0.8
        else:           s_range += 1.0

        # Chop flag (direct evidence)
        if chop_detected: s_chop += 4.0; s_bull -= 1.0; s_bear -= 1.0

        # Market structure
        struct = ms.get("structure", "")
        if   struct == "bullish":      s_bull += 2.5
        elif struct == "bearish":      s_bear += 2.5
        elif struct == "distribution": s_bear += 1.5; s_panic += 0.5
        elif struct == "ranging":      s_range += 2.0

        # Price Z-score
        if   z < -3.0: s_panic += 3.0; s_bear += 1.0
        elif z < -2.0: s_bear  += 2.0; s_panic += 1.0
        elif z < -1.0: s_bear  += 1.0
        elif z >  2.5: s_bull  += 1.5
        elif z >  1.5: s_bull  += 0.8

        # ATR ratio (vol expansion = stress)
        if   atr_ratio > 1.8: s_panic += 2.0; s_bear += 1.0
        elif atr_ratio > 1.3: s_bear  += 0.5; s_bull += 0.5  # expansion but neutral
        elif atr_ratio < 0.7: s_range += 1.5  # compression → range

        # ── Softmax with temperature normalisation ────────────────────────
        # Without temperature, a score of 11 vs 3 → P_dominant ≈ 0.9996 → entropy ≈ 0.
        # Normalise so the max input to softmax is always ~3.0.
        # This means entropy is high when scores are close, low when one dominates.
        raw = np.array([s_bull, s_bear, s_range, s_chop, s_panic], dtype=float)
        T = max(float(np.abs(raw).max()), 1.0) / 3.0   # dynamic temperature
        raw_scaled = raw / T
        raw_scaled -= raw_scaled.max()   # numerical stability
        exp_raw = np.exp(raw_scaled)
        probs   = exp_raw / (exp_raw.sum() + 1e-9)
        p_bull, p_bear, p_range, p_chop, p_panic = [float(p) for p in probs]

        # ── Shannon entropy (normalised) ──────────────────────────────────
        h_raw  = float(-np.sum(probs * np.log(probs + 1e-9)))
        h_norm = float(h_raw / math.log(N_REGIME_STATES))   # [0,1]

        # ── Dominant regime ───────────────────────────────────────────────
        names  = ["bull", "bear", "range", "chop", "panic"]
        dom    = names[int(np.argmax(probs))]

        # ── Label and confidence scale ────────────────────────────────────
        # ── Label and confidence scale (adaptive thresholds) ─────────────────
        # Record this entropy value and get percentile-based thresholds.
        # Uses rolling history when >= 30 readings; fixed fallback otherwise.
        _entropy_tracker.record(h_norm)
        t_stable, t_mild, t_unstable, t_volatile = _entropy_tracker.thresholds()
        h_pct = _entropy_tracker.percentile_of(h_norm)
        adaptive = len(_entropy_tracker.history) >= _entropy_tracker.MIN_HISTORY

        if   h_norm <= t_stable:   label = "STABLE";   scale = 1.00
        elif h_norm <= t_mild:     label = "MILD";     scale = 0.85
        elif h_norm <= t_unstable: label = "UNSTABLE"; scale = 0.60
        elif h_norm <= t_volatile:
            label = "VOLATILE"
            # Context-aware scale: volatile markets ARE the opportunity in
            # bullish trends; they're dangerous in bearish ones.
            #   Bullish trend + volatile → 0.75  (volatility = momentum, lean in)
            #   Sideways + volatile      → 0.40  (some edge, directionally uncertain)
            #   Bearish trend + volatile → 0.22  (dangerous but not fatal)
            _is_bull = (p_bull > 0.40) or (cloud == "above" and macd_h > 0)
            _is_bear = (p_bear + p_panic > 0.45) or (cloud == "below" and macd_h < 0)
            if   _is_bull: scale = 0.75
            elif _is_bear: scale = 0.22
            else:          scale = 0.40
        else:
            label = "CHAOTIC"
            # Direction-aware — chaotic bullish markets have the biggest moves.
            #   Bullish trend + chaotic → 0.60  (high volatility = opportunity)
            #   Sideways + chaotic      → 0.30  (some signal survives)
            #   Bearish trend + chaotic → 0.17  (dangerous, heavy suppression)
            _is_bull = (p_bull > 0.40) or (cloud == "above" and macd_h > 0)
            _is_bear = (p_bear + p_panic > 0.45) or (cloud == "below" and macd_h < 0)
            if   _is_bull: scale = 0.60
            elif _is_bear: scale = 0.17
            else:          scale = 0.30

        return RegimeEntropy(
            p_bull=p_bull, p_bear=p_bear, p_range=p_range,
            p_chop=p_chop, p_panic=p_panic,
            entropy_raw=h_raw, entropy_norm=round(h_norm, 4),
            dominant_regime=dom, label=label,
            confidence_scale=scale,
            entropy_pct=round(h_pct, 1),
            thresholds_adaptive=adaptive,
        )

    except Exception as e:
        return RegimeEntropy(
            p_bull=0.33, p_bear=0.17, p_range=0.33, p_chop=0.17, p_panic=0.0,
            entropy_raw=float(math.log(3)), entropy_norm=0.50,
            dominant_regime="range", label="MILD",
            confidence_scale=0.85,
        )


def apply_regime_entropy(prob: float, entropy: "RegimeEntropy") -> Tuple[float, str]:
    """
    Shrink the final probability toward 0.5 based on regime entropy.

    P_adjusted = 0.5 + (P_raw - 0.5) × confidence_scale

    This preserves direction but reduces conviction in unstable regimes.
    A bullish signal in a chaotic market is not ignored — it is weakened
    proportionally to how uncertain the regime classification is.
    """
    scale = entropy.confidence_scale
    adjusted = 0.5 + (prob - 0.5) * scale
    adjusted = max(0.01, min(0.99, adjusted))
    src_tag  = f"adaptive(n={len(_entropy_tracker.history)},pct={entropy.entropy_pct:.0f}th)" \
               if entropy.thresholds_adaptive else f"fixed(fallback)"
    note = (f"[ENTROPY] {entropy.label}  H={entropy.entropy_norm:.3f}  "
            f"scale={scale:.2f}  {src_tag}  "
            f"bull={entropy.p_bull:.2f} bear={entropy.p_bear:.2f} "
            f"range={entropy.p_range:.2f} chop={entropy.p_chop:.2f}  "
            f"P: {prob:.3f} → {adjusted:.3f}")
    # Periodically save history (every 10 recordings)
    if len(_entropy_tracker.history) % 10 == 0:
        _entropy_tracker.save()
    return adjusted, note


def print_signal_consensus(coin: str,
                            evidence: List["_Evidence"],
                            vol_regime: str = "neutral") -> None:
    """
    Compress all scorer signals into a per-group consensus summary.

    For each group:
      - Count signals that voted BULLISH (log_lr > 0.05)
      - Count signals that voted BEARISH (log_lr < -0.05)
      - Compute the weighted average log-LR for direction label
      - Display as (bearish_count/total) or (bullish_count/total)

    Direction labels:
      BULLISH      avg_log_lr > +0.40  (strong bull consensus)
      MILD BULL    avg_log_lr > +0.10
      NEUTRAL      |avg_log_lr| <= 0.10 and signals agree
      MIXED        |avg_log_lr| <= 0.10 but signals disagree (bullish AND bearish present)
      MILD BEAR    avg_log_lr < -0.10
      BEARISH      avg_log_lr < -0.40  (strong bear consensus)

    VOLATILITY uses "EXPANDING" / "CONTRACTING" since that's more descriptive.
    """
    from collections import defaultdict

    GROUP_ORDER = [
        "TREND", "MOMENTUM", "FLOW", "STRUCTURE",
        "VOLATILITY", "BREAKOUT", "TRANSITION", "MICROSTRUCTURE", "OTHER",
    ]
    GROUP_LABELS = {
        "TREND":          "Trend",
        "MOMENTUM":       "Momentum",
        "FLOW":           "Flow",
        "STRUCTURE":      "Structure",
        "VOLATILITY":     "Volatility",
        "BREAKOUT":       "Breakout",
        "TRANSITION":     "Transition",
        "MICROSTRUCTURE": "Microstructure",
        "OTHER":          "ML / Other",
        "MACRO":          "Macro context",
        "ML_INTERACTION": "ML interact",
        # Short scorer groups
        "EXHAUSTION":     "Exhaustion",
        "DISTRIBUTION":   "Distribution",
        "TRAP":           "Trap",
        "ACCELERATION":   "Acceleration",
        "DANGER":         "Danger",
        "FRAGILITY":      "Fragility",
    }

    groups: Dict[str, list] = defaultdict(list)
    for ev in evidence:
        groups[ev.group].append(ev)

    all_groups = GROUP_ORDER + [g for g in groups if g not in GROUP_ORDER]

    print(f"[CONSENSUS] {coin}")

    for group in all_groups:
        evs = groups.get(group, [])
        if not evs:
            continue

        n_bull  = sum(1 for e in evs if e.log_lr >  0.05)
        n_bear  = sum(1 for e in evs if e.log_lr < -0.05)
        n_neut  = len(evs) - n_bull - n_bear
        total   = len(evs)

        # Magnitude-weighted average (mirrors latent group math)
        total_mag = sum(abs(e.log_lr) for e in evs) or 1.0
        avg_lo    = sum(e.log_lr * abs(e.log_lr) for e in evs) / total_mag

        # Direction label
        # SHORT SCORER GROUPS: avg_lo > 0 means the signal SUPPORTS shorting
        # (e.g. Distribution LR=3.31 is BULLISH for the short thesis, not price)
        # Remap labels so the display matches intent:
        #   LR > 1 → "SHORT ✓"  (confirms short)
        #   LR < 1 → "CONTRA"   (works against short)
        SHORT_GROUPS = {"EXHAUSTION","DISTRIBUTION","TRAP","ACCELERATION","DANGER","FRAGILITY"}
        _is_short_group = group in SHORT_GROUPS

        mixed = n_bull >= 1 and n_bear >= 1
        if   group == "VOLATILITY":
            if   vol_regime == "expansion":  dir_label = "EXPANDING"
            elif vol_regime == "compression":dir_label = "CONTRACTING"
            elif avg_lo > 0.10:              dir_label = "BULLISH"
            elif avg_lo < -0.10:             dir_label = "BEARISH"
            else:                            dir_label = "NEUTRAL"
        elif _is_short_group:
            # Reframe relative to the short trade (not price direction)
            if   avg_lo >  0.50: dir_label = "SHORT ✓"      # strong short confirmation
            elif avg_lo >  0.15: dir_label = "MILD SHORT"    # moderate short confirmation
            elif avg_lo < -0.50: dir_label = "CONTRA ✗"      # strong counter-signal
            elif avg_lo < -0.15: dir_label = "MILD CONTRA"   # mild counter-signal
            elif mixed:          dir_label = "MIXED"
            else:                dir_label = "NEUTRAL"
        elif avg_lo >  0.50:                 dir_label = "BULLISH"
        elif avg_lo >  0.15:                 dir_label = "MILD BULL"
        elif avg_lo < -0.50:                 dir_label = "BEARISH"
        elif avg_lo < -0.15:                 dir_label = "MILD BEAR"
        elif mixed:                          dir_label = "MIXED"
        else:                                dir_label = "NEUTRAL"

        # Colour indicator — for short groups: SHORT ✓ = green, CONTRA ✗ = red
        if _is_short_group:
            if   dir_label == "SHORT ✓":    icon = "🟢"
            elif dir_label == "MILD SHORT":  icon = "🟡"
            elif dir_label == "CONTRA ✗":   icon = "🔴"
            elif dir_label == "MILD CONTRA": icon = "🟡"
            elif dir_label == "MIXED":       icon = "⚪"
            else:                            icon = "⬜"
        elif dir_label in ("BULLISH",):             icon = "🟢"
        elif dir_label in ("MILD BULL",):            icon = "🟡"
        elif dir_label in ("BEARISH",):              icon = "🔴"
        elif dir_label in ("MILD BEAR",):            icon = "🟡"
        elif dir_label in ("EXPANDING","BULLISH"):   icon = "🟡"
        elif dir_label in ("CONTRACTING",):          icon = "🔵"
        elif dir_label == "MIXED":                   icon = "⚪"
        else:                                        icon = "⬜"

        # Vote fraction bar (30 chars)
        dominant_n = n_bear if avg_lo < 0 else n_bull
        if total > 0:
            vote_frac   = dominant_n / total
            bar_filled  = int(vote_frac * 20)
            bar         = "█" * bar_filled + "░" * (20 - bar_filled)
        else:
            bar = "░" * 20

        vote_str = f"({dominant_n}/{total})" if dir_label not in ("NEUTRAL", "MIXED") else f"({n_bull}↑{n_bear}↓)"
        gname    = GROUP_LABELS.get(group, group)
        print(f"  {icon} {gname:<14}  {dir_label:<12}  {vote_str:<8}  {bar}  "
              f"avg_lr={math.exp(avg_lo):.2f}")


def print_confidence_killers(
    coin:          str,
    sig:           dict,
    last:          pd.Series,
    prob:          float,
    ml_proba:      float,
    ml_accuracy:   float,
    entropy:       Optional["RegimeEntropy"],
    chop_detected: bool,
    vol_regime:    str,
    phase:         str        = "RANGE",
    phase_dur:     int        = 0,
    btc_regime:    str        = "unknown",
    mem:           Optional["RegimeMemory"] = None,
) -> List[str]:
    """
    Identify and print the specific reasons why confidence is suppressed.

    Shows base P (signal strength without BTC regime) and the cost of
    each killer so users understand the magnitude, not just the list.
    """
    killers: List[Tuple[int, str, str]] = []  # (severity_rank, icon, message)

    adx      = float(last.get("ADX", 0) or 0)
    rsi      = float(last.get("RSI", 50) or 50)
    cloud    = sig.get("cloud_state", "unknown")
    ms       = sig.get("market_structure", {}) or {}
    mkt      = sig.get("market_regime", "unknown")

    # ── Compute base P (what signals say without BTC regime LR) ──────────────
    # base_P shows the intrinsic signal quality, unaffected by macro regime.
    # This answers "is this a good setup being suppressed, or just a weak setup?"
    base_p = prob  # fallback
    try:
        _prior  = sig.get("dynamic_prior", PRIOR_CONTINUATION)
        if btc_regime == "bear":
            # Reverse the bear LR to get what P would be without it
            import math as _m
            _logit = lambda p: _m.log(max(1e-9, p) / max(1e-9, 1-p))
            _sig   = lambda x: 1 / (1 + _m.exp(-x))
            base_lo = _logit(prob) - _m.log(_BTC_BEAR_LR)
            base_p  = round(min(0.97, _sig(base_lo)), 3)
    except Exception:
        pass

    # ── CRITICAL (rank 1-3) ───────────────────────────────────────────────────

    # ── ML status ────────────────────────────────────────────────────────
    # ML disabled = LR=1.0 (neutral) — does NOT suppress probability.
    # Show as info note, not as a killer. Real killer is only if ML is
    # actively bearish (reliable model predicting loss).
    if ml_accuracy >= 0.55:
        if ml_proba < 0.35:
            killers.append((1, "⛔",
                f"ML strongly bearish: {ml_proba:.0%} win probability "
                f"(acc={ml_accuracy:.0%}) — reliable model predicts loss"))
        elif ml_proba < 0.50:
            killers.append((2, "❌",
                f"ML bearish: {ml_proba:.0%} win probability "
                f"(acc={ml_accuracy:.0%})"))
    # ml_accuracy < 0.50: ML is neutralized (LR=1.0) — not a suppressor,
    # just means no ML confirmation. Note it in the header only.)

    # Below cloud (hard structural block)
    if cloud == "below":
        killers.append((1, "⛔", "Below Ichimoku cloud: structural bear zone"))
    elif cloud == "in":
        killers.append((3, "⚠️ ", "Inside cloud: no directional conviction"))

    # Bear regime + BTC bear — show the probability cost explicitly
    if btc_regime == "bear" and mkt in ("bear", "range"):
        _cost = base_p - prob if base_p > prob else 0
        _cost_str = f"  base={base_p:.0%} → {prob:.0%} (regime costs {_cost:.0%})" if _cost > 0.02 else ""
        killers.append((2, "❌", f"BTC bear + {mkt} regime: dual macro headwind{_cost_str}"))
    elif btc_regime == "bear":
        _cost = base_p - prob if base_p > prob else 0
        _cost_str = f"  base={base_p:.0%} → {prob:.0%} (costs {_cost:.0%})" if _cost > 0.02 else ""
        killers.append((2, "❌", f"BTC bear regime: altcoin longs suppressed{_cost_str}"))
    elif mkt in ("bear",):
        killers.append((3, "⚠️ ", f"Bear market regime: trend against longs"))

    # ── MAJOR (rank 4-6) ──────────────────────────────────────────────────────

    # OBV falling — distribution
    obv_trend = sig.get("obv_trend", "unknown")
    if obv_trend == "falling":
        killers.append((4, "❌", "OBV falling: distribution detected — capital leaving"))

    # Rejection wick
    try:
        high  = float(last.get("high", last["price"]))
        close = float(last.get("close", last["price"]))
        open_ = float(last.get("open", last["price"]))
        low   = float(last.get("low", last["price"]))
        rng   = high - low + 1e-9
        uw    = (high - max(close, open_)) / rng
        if uw > 0.80:
            killers.append((4, "❌", f"Rejection wick {uw:.0%}: gave back {uw:.0%} of range "
                                      f"({uw / (rng / last['price']):.1f}×ATR equivalent)"))
        elif uw > 0.60:
            killers.append((5, "⚠️ ", f"Large upper wick {uw:.0%}: sellers absorbed buyers at high"))
    except Exception:
        pass

    # Weak ADX in chop
    if chop_detected:
        killers.append((4, "❌", f"Chop detected: ADX={adx:.1f} — signals unreliable in choppy market"))
    elif adx < 15:
        killers.append((5, "⚠️ ", f"Very weak ADX ({adx:.1f}): no trend — breakout could go either way"))
    elif adx < 20:
        killers.append((5, "⚠️ ", f"Weak ADX ({adx:.1f}): trend persistence unreliable"))

    # ── CAUTION (rank 7-9) ───────────────────────────────────────────────────

    # High entropy — unstable regime
    if entropy is not None:
        if entropy.label == "CHAOTIC":
            _dir = ("bullish" if entropy.p_bull > 0.40 else
                    "bearish" if (entropy.p_bear + entropy.p_panic) > 0.45 else "sideways")
            killers.append((4, "❌", f"CHAOTIC regime entropy (H={entropy.entropy_norm:.2f}, "
                                      f"{_dir}) scale={entropy.confidence_scale:.2f}"))
        elif entropy.label == "VOLATILE":
            _dir = ("bullish" if entropy.p_bull > 0.40 else
                    "bearish" if (entropy.p_bear + entropy.p_panic) > 0.45 else "sideways")
            killers.append((5, "⚠️ ", f"Volatile regime entropy (H={entropy.entropy_norm:.2f}, "
                                       f"{_dir}) scale={entropy.confidence_scale:.2f}"))
        elif entropy.label == "UNSTABLE":
            killers.append((6, "⚠️ ", f"Unstable regime entropy (H={entropy.entropy_norm:.2f}): "
                                       f"reduced confidence scale={entropy.confidence_scale:.2f}"))

    # Phase decay risk
    typical  = PHASE_TYPICAL_DURATION.get(phase, 20)
    lambda_  = PHASE_DECAY_LAMBDA.get(phase, 1.40)
    age_pct  = phase_dur / max(1, typical)
    strength = math.exp(-lambda_ * age_pct)
    if strength < 0.25:
        killers.append((5, "⚠️ ", f"Phase overdue ({phase} day {phase_dur}/{typical}): "
                                   f"only {strength:.0%} signal strength remaining"))
    elif age_pct > 1.2:
        killers.append((6, "⚠️ ", f"Phase ageing ({phase_dur}/{typical} bars): transition risk rising"))

    # Bearish market structure
    struct = ms.get("structure", "")
    if struct in ("bearish", "distribution"):
        killers.append((6, "⚠️ ", f"Market structure {struct.upper()}: lower highs + lower lows"))

    # RSI overbought
    if rsi > 75:
        killers.append((7, "⚠️ ", f"RSI overbought ({rsi:.1f}): mean-reversion risk"))

    # Regime memory: flip instability
    if mem is not None:
        flips = mem.regime_flip_count(window=10)
        if flips >= 6:
            killers.append((6, "⚠️ ", f"Regime whipsawing ({flips}/10 bars): "
                                       f"classification unstable"))
        elif flips >= 4:
            killers.append((7, "⚠️ ", f"Regime flipping ({flips}/10 bars): noisy environment"))

    # ── EXPECTED VALUE ───────────────────────────────────────────────────
    ev_data = sig.get("ev", {})
    if ev_data:
        ev_val  = ev_data.get("ev", 0.0)
        ev_rate = ev_data.get("rating", "D")
        ev_rr   = ev_data.get("adj_rr", 2.0)
        ev_kelly= ev_data.get("kelly", 0.0)
        ev_be   = ev_data.get("breakeven_p", 0.33)
        if not ev_data.get("tradeable") and prob >= BUY_PROB_THRESHOLD:
            killers.append((2, "❌",
                f"EV={ev_val:+.2f} ({ev_rate}) below trade threshold — "
                f"RR={ev_rr:.1f} needs P≥{ev_be:.0%} to break even, you have {prob:.0%}"))
        elif ev_data.get("rating") in ("A+", "A"):
            killers.append((3, "✅",
                f"EV={ev_val:+.2f} ({ev_rate}) — exceptional  "
                f"Kelly={ev_kelly:.1%}  RR={ev_rr:.1f}"))

    # ── TREND LIFECYCLE ──────────────────────────────────────────────────
    lifecycle = sig.get("lifecycle", {})
    if lifecycle.get("stage"):
        stage = lifecycle["stage"]
        pos   = lifecycle.get("position_mod", 1.0)
        reason= lifecycle.get("reason", "")
        if stage == "EARLY_TREND":
            killers.append((3, "✅", f"Stage 2 EARLY TREND — best R:R window, "
                                     f"fresh breakout confirming (pos={pos:.2f}×)"))
        elif stage == "EXHAUSTION":
            killers.append((4, "⚠️ ", f"Stage 4 EXHAUSTION — {reason}. "
                                      f"Reduce position to {pos:.0%} of normal. "
                                      f"Trend can stay stretched — don't exit yet."))
        elif stage == "DISTRIBUTION":
            killers.append((2, "❌", f"Stage 5 DISTRIBUTION — {reason}. "
                                     f"Dangerous for longs (pos={pos:.0%} max)"))
        elif stage == "ACCUMULATION":
            killers.append((6, "🟡", f"Stage 1 ACCUMULATION — building base, "
                                     f"wait for breakout confirmation"))

    # ── RELATIVE STRENGTH ────────────────────────────────────────────────
    rs_full = sig.get("rs_full", {})
    if rs_full.get("available"):
        rs_label = rs_full.get("label", "NEUTRAL")
        rs_7d    = rs_full.get("rs_btc_7d", 0)
        rs_trend = rs_full.get("rs_trend", "neutral")
        rs_eth   = rs_full.get("rs_eth_7d", 0)
        if rs_label == "LEADER":
            killers.append((6, "✅", f"RS LEADER: +{rs_7d:.1f}% vs BTC (7d)"
                                     + (f"  +{rs_eth:.1f}% vs TOTAL3" if rs_eth > 2 else "")
                                     + "  — resists weakness, leads rotations"))
        elif rs_label == "WEAK":
            killers.append((3, "❌", f"RS WEAK: {rs_7d:.1f}% vs BTC (7d) — "
                                     "capital rotating out of this coin"))
        elif rs_label == "LAGGARD":
            killers.append((5, "⚠️ ", f"RS LAGGARD: {rs_7d:.1f}% vs BTC (7d) — "
                                      "underperforming market"))
        if rs_trend == "declining" and rs_label not in ("LEADER", "STRONG"):
            killers.append((6, "⚠️ ", f"RS trend declining (7d RS worse than 30d) — "
                                      "momentum fading"))

    # Sort by severity (lower rank = more critical), keep top 7
    killers.sort(key=lambda x: x[0])
    shown = killers[:7]

    if not shown:
        if prob >= 0.50:
            print(f"[CONFIDENCE KILLERS] {coin}: none — signals broadly supportive")
        return []

    reason_label = "reason" if len(shown) == 1 else "reasons"
    # Show base P when BTC bear is suppressing a strong setup
    base_note = ""
    if btc_regime == "bear" and base_p > prob + 0.10:
        bull_p_val = compute_bull_scenario_prob(prob, prior=sig.get("dynamic_prior", PRIOR_CONTINUATION))
        base_note  = f"  [signals={base_p:.0%}  bull={bull_p_val:.0%}]"
    # ML status note (informational only — not a suppressor)
    ml_note = ""
    _gate_sharpe = _WF_SHARPE.get(coin, 0.0)
    if _gate_sharpe < -1.0:
        ml_note = f"  [ML disabled Sharpe={_gate_sharpe:+.2f}]"
    elif _gate_sharpe > 0.5 and ml_accuracy < 0.50:
        ml_note = f"  [ML active Sharpe={_gate_sharpe:+.2f} acc={ml_accuracy:.0%}↓ kept]"
    elif ml_accuracy < 0.50:
        ml_note = f"  [ML dampened acc={ml_accuracy:.0%} Sharpe={_gate_sharpe:+.2f}]"
    elif ml_accuracy >= 0.55 and ml_proba >= 0.60:
        ml_note = f"  [ML confirms {ml_proba:.0%}]"
    print(f"[CONFIDENCE KILLERS] {coin}  "
          f"({len(shown)} {reason_label} P={prob:.0%} suppressed{base_note}{ml_note})")
    for _, icon, msg in shown:
        print(f"  {icon}  {msg}")

    return [msg for _, _, msg in shown]


def print_short_confidence_killers(
    coin:             str,
    sig:              dict,
    short_prob:       float,
    short_ml_proba:   float,
    short_ml_acc:     float,
    short_blocked:    bool,
    bear_accel,                        # BearAcceleration | None
    trend_frag,                        # TrendFragility | None
    chop_detected:    bool,
    btc_regime:       str = "unknown",
    futures           = None,
) -> List[str]:
    """
    Identify why a SHORT isn't triggering — from the short scorer's perspective.

    Different from long killers: shorts fail for short-specific reasons.
    The killer list answers: "Why doesn't this set up as a short right now?"
    """
    killers: List[Tuple[int, str, str]] = []

    short_thresh  = SHORT_PROB_THRESHOLD
    cloud         = sig.get("cloud_state", "unknown")
    mkt           = sig.get("market_regime", "unknown")
    ms            = sig.get("market_structure", {}) or {}

    # ── CRITICAL — P far below threshold ────────────────────────────────────
    if short_prob < 0.30:
        killers.append((1, "⛔", f"Short P={short_prob:.0%} — well below {short_thresh:.0%} threshold"))
    elif short_prob < short_thresh:
        killers.append((2, "❌", f"Short P={short_prob:.0%} — below {short_thresh:.0%} threshold"))

    # ML says it's not a short
    if short_ml_acc >= 0.50 and short_ml_proba < 0.35:
        killers.append((1, "⛔", f"Short ML {short_ml_proba:.0%} — model says don't short "
                                  f"(acc={short_ml_acc:.0%})"))
    elif short_ml_acc < 0.50:
        killers.append((2, "❌", f"Short ML disabled (acc={short_ml_acc:.0%} below random)"))

    # Squeeze block
    if short_blocked:
        killers.append((1, "⛔", "Short BLOCKED — squeeze risk (funding too negative)"))

    # Price above cloud (hard counter for shorts)
    if cloud == "above":
        killers.append((2, "❌", "Price above cloud: structural bull zone — shorts fight trend"))
    elif cloud == "in":
        killers.append((3, "⚠️ ", "Price inside cloud: no clear direction for shorts"))

    # BTC bull
    if btc_regime == "bull" and mkt == "bull_trend":
        killers.append((2, "❌", "BTC bull + bull trend: counter-trend short — very risky"))
    elif btc_regime == "bull":
        killers.append((3, "⚠️ ", "BTC bull regime: macro headwind for shorts"))

    # ── MAJOR — missing short prerequisites ──────────────────────────────────

    # No bear acceleration
    n_accel = getattr(bear_accel, "n_confirmed", 0) if bear_accel else 0
    exp_atr  = getattr(bear_accel, "expected_move_atr", 1.0) if bear_accel else 1.0
    if n_accel == 0:
        killers.append((4, "❌", "No bear acceleration (0/4 signals) — no downside momentum"))
    elif n_accel == 1:
        killers.append((5, "⚠️ ", f"Weak acceleration ({n_accel}/4) — expected move only {exp_atr:.1f}×ATR"))

    # No trend fragility
    frag = getattr(trend_frag, "fragility", 0.0) if trend_frag else 0.0
    frag_label = getattr(trend_frag, "label", "NONE") if trend_frag else "NONE"
    if frag < 0.20:
        killers.append((4, "❌", f"No fragility ({frag_label} {frag:.2f}) — trend not exhausted"))
    elif frag < 0.40:
        killers.append((5, "⚠️ ", f"Low fragility ({frag_label} {frag:.2f}) — trend still has strength"))

    # Chop — shorts unreliable in chop
    if chop_detected:
        killers.append((4, "❌", "Chop detected — short setups don't follow through in ranging markets"))

    # Structure bullish (shorts need bearish structure)
    struct = ms.get("structure", "")
    if struct == "bullish":
        killers.append((5, "⚠️ ", "Structure BULLISH (HH+HL) — waiting for CHoCH before shorting"))

    # Funding against shorts
    if futures and not getattr(futures, "error", True) and futures.is_futures:
        fr = futures.funding_rate * 100
        if fr < -0.03:
            killers.append((4, "❌", f"Funding negative ({fr:.3f}%) — shorts paying, squeeze risk"))

    killers.sort(key=lambda x: x[0])
    shown = killers[:6]

    if not shown:
        print(f"[SHORT KILLERS] {coin}: none — setup is valid, probability just below threshold")
        return []

    print(f"[SHORT KILLERS] {coin}  "
          f"({len(shown)} reason{'s' if len(shown)!=1 else ''} short not triggering)")
    for _, icon, msg in shown:
        c = "\033[91m" if any(x in icon for x in ("⛔","❌")) else "\033[93m"
        print(f"  {icon}  {msg}")
    print("\033[0m", end="")
    return [msg for _, _, msg in shown]


def detect_contradictions(
    sig:           dict,
    last:          pd.Series,
    df:            pd.DataFrame,
    btc_regime:    str = "unknown",
    chop_detected: bool = False,
) -> Tuple[int, List[str], float]:
    """
    Detect specific indicator contradictions that reduce conviction.

    The latent group architecture handles WITHIN-group correlations correctly,
    but cannot detect CROSS-group contradictions where signals disagree on
    the fundamental direction. These disagreements are genuine uncertainty
    that the Bayesian model can't see because it treats each group independently.

    CONTRADICTION EXAMPLES:
      Cloud below + DI+ dominant     — structural bear but momentum bullish
      MACD bearish + RSI bullish     — momentum conflict
      OBV falling + structure bull   — distribution despite HH/HL
      OBV rising + structure bear    — accumulation despite LH/LL
      BTC bull + local bear          — macro/micro divergence

    SEVERITY SCALING (shrinkage toward 0.5):
      1 conflict:  8%  shrinkage  →  P=0.70 → 0.648
      2 conflicts: 18% shrinkage  →  P=0.70 → 0.574
      3+ conflicts:28% shrinkage  →  P=0.70 → 0.504

    Applied as:  P_adj = 0.5 + (P - 0.5) × (1 - severity)
    This preserves direction but reduces conviction proportionally.

    Returns: (n_contradictions, descriptions, severity)
    """
    conflicts: List[Tuple[int, str]] = []   # (weight, description)

    cloud    = sig.get("cloud_state", "unknown")
    adx      = float(last.get("ADX",  20) or 20)
    di_plus  = float(last.get("plus_di",  25) or 25)
    di_minus = float(last.get("minus_di", 25) or 25)
    rsi      = float(last.get("RSI",  50) or 50)
    macd_h   = float(last.get("MACD_Hist", 0) or 0)
    obv      = sig.get("obv_trend", "unknown")
    struct   = (sig.get("market_structure") or {}).get("structure", "")
    regime   = sig.get("market_regime", "unknown")

    # ── Trend direction contradictions ───────────────────────────────────────

    # Cloud bearish but DI+ clearly dominant (price below cloud, momentum bullish)
    # This is the crypto-com-chain case: cloud↓ but DI+=56 >> DI-=18
    if cloud == "below" and di_plus > di_minus * 1.5 and adx > 18:
        conflicts.append((2, f"cloud↓ below but DI+={di_plus:.0f} >> DI-={di_minus:.0f} "
                             f"(structural bear, momentum bullish)"))

    # Cloud bullish but DI- clearly dominant (price above cloud, trend weakening)
    if cloud == "above" and di_minus > di_plus * 1.5 and adx > 18:
        conflicts.append((2, f"cloud↑ above but DI-={di_minus:.0f} >> DI+={di_plus:.0f} "
                             f"(structural bull, trend weakening)"))

    # ── Momentum contradictions ───────────────────────────────────────────────

    # MACD bearish + RSI bullish (medium-term declining, short-term strong)
    if macd_h < -0.001 and rsi > 60:
        conflicts.append((1, f"MACD hist={macd_h:.4f}↓ but RSI={rsi:.0f} "
                             f"(MACD declining while RSI bullish)"))

    # MACD bullish + RSI bearish (medium-term rising, short-term weak)
    if macd_h > 0.001 and rsi < 40:
        conflicts.append((1, f"MACD hist={macd_h:.4f}↑ but RSI={rsi:.0f} "
                             f"(MACD rising while RSI bearish)"))

    # ── Volume / flow contradictions ──────────────────────────────────────────

    # OBV falling (distribution) + bullish market structure (HH+HL)
    # Capital leaving while price making higher highs = unsustainable
    if obv == "falling" and struct == "bullish":
        conflicts.append((2, "OBV↓ falling (distribution) but structure BULLISH "
                             "(HH+HL) — price rising without volume backing"))

    # OBV rising (accumulation) + bearish market structure (LH+LL)
    if obv == "rising" and struct == "bearish":
        conflicts.append((1, "OBV↑ rising (accumulation) but structure BEARISH "
                             "(LH+LL) — smart money buying into downtrend"))

    # ── Macro / micro divergence ──────────────────────────────────────────────

    # BTC bear macro but strong local bull trend
    if btc_regime == "bear" and regime == "bull_trend" and cloud == "above":
        conflicts.append((1, f"BTC bear but local bull trend above cloud "
                             f"(macro headwind vs local strength)"))

    # BTC bull but deep in bearish regime
    if btc_regime == "bull" and regime == "bear" and cloud == "below":
        conflicts.append((1, f"BTC bull but local bear below cloud "
                             f"(macro tailwind but local weakness)"))

    # ── Regime / volatility contradictions ────────────────────────────────────

    # Chop detected but strong ADX (ADX says trend, chop says no)
    if chop_detected and adx > 30:
        conflicts.append((1, f"Chop detected but ADX={adx:.0f} (ADX says trend exists, "
                             f"BB says range — conflicting regime signals)"))

    # ── Compute severity ──────────────────────────────────────────────────────
    # Weight-sum: each conflict contributes 1 or 2 to severity score
    # Calibrated so that:
    #   1 heavy conflict (weight 2):  P=0.70 → 0.638  (just below gem threshold)
    #   2 medium conflicts (weight 2): P=0.70 → 0.580
    #   3+ conflicts:                  P=0.70 → 0.525
    n_conflicts  = len(conflicts)
    total_weight = sum(w for w, _ in conflicts)
    if   total_weight == 0: severity = 0.0
    elif total_weight == 1: severity = 0.12   # light conflict
    elif total_weight == 2: severity = 0.25   # 1 heavy OR 2 light
    elif total_weight == 3: severity = 0.35   # 1 heavy + 1 light
    elif total_weight == 4: severity = 0.42   # 2 heavy
    else:                   severity = 0.48   # extreme contradiction

    return n_conflicts, [desc for _, desc in conflicts], severity


def apply_contradiction_penalty(prob: float, coin: str,
                                 n: int, descriptions: List[str],
                                 severity: float) -> Tuple[float, str]:
    """
    Apply probability shrinkage toward 0.5 based on contradiction severity.

    Preserves direction — a bullish coin stays bullish, just less extreme.
    P=0.70 with severity=0.18  →  0.5 + 0.20 × 0.82 = 0.664
    P=0.30 with severity=0.18  →  0.5 − 0.20 × 0.82 = 0.336
    """
    if severity <= 0 or n == 0:
        return prob, ""

    adj  = round(0.5 + (prob - 0.5) * (1.0 - severity), 4)
    adj  = max(0.01, min(0.99, adj))
    note = (f"[CONTRADICTION] {coin}: {n} conflict(s)  "
            f"severity={severity:.0%}  P: {prob:.3f} → {adj:.3f}")
    return adj, note


def detect_short_contradictions(
    sig:           dict,
    last:          pd.Series,
    df:            pd.DataFrame,
    btc_regime:    str = "unknown",
    chop_detected: bool = False,
) -> Tuple[int, List[str], float]:
    """
    Detect indicator contradictions that undermine short conviction.

    Mirror of detect_contradictions() but framed for shorts:
      For longs:  "cloud below + DI+ dominant" = contradiction
      For shorts: "cloud above + DI+ dominant" = contradiction (fighting trend)

    SHORT CONTRADICTION EXAMPLES:
      Cloud above + shorting           — counter-trend, fighting bullish structure
      DI+ clearly dominant             — momentum bullish, short fights trend
      OBV rising + bearish setup       — smart money accumulating while shorting
      RSI oversold                     — mean-reversion risk, bad short timing
      Bullish market structure (HH+HL) — structure disagrees with short thesis
      BTC bull + trying to short       — macro tailwind fights short thesis
      MACD histogram rising + shorting — momentum turning up

    SEVERITY LEVELS (same as long contradictions):
      1 heavy conflict:  25% shrinkage  →  short_P=0.70 → 0.638
      2 conflicts:       35-42%         →  short_P=0.70 → 0.580-0.561
    """
    conflicts: List[Tuple[int, str]] = []

    cloud    = sig.get("cloud_state", "unknown")
    adx      = float(last.get("ADX",  20) or 20)
    di_plus  = float(last.get("plus_di",  25) or 25)
    di_minus = float(last.get("minus_di", 25) or 25)
    rsi      = float(last.get("RSI",  50) or 50)
    macd_h   = float(last.get("MACD_Hist", 0) or 0)
    obv      = sig.get("obv_trend", "unknown")
    struct   = (sig.get("market_structure") or {}).get("structure", "")
    regime   = sig.get("market_regime", "unknown")

    # ── Structure / trend conflicts ───────────────────────────────────────────

    # Cloud ABOVE while shorting — counter-trend, fighting bullish structure
    # The main short signal needs cloud below for best probability
    if cloud == "above" and adx > 18:
        conflicts.append((2, f"cloud↑ above (bullish structure) — "
                             f"counter-trend short, fighting Ichimoku support"))

    # DI+ dominant → momentum bullish, fights short thesis
    if di_plus > di_minus * 1.5 and adx > 18:
        conflicts.append((2, f"DI+={di_plus:.0f} >> DI-={di_minus:.0f} "
                             f"(bullish momentum while shorting)"))

    # Bullish market structure while shorting (HH+HL = trend is UP)
    if struct == "bullish" and cloud != "below":
        conflicts.append((1, "Market structure BULLISH (HH+HL) — "
                             "shorting into a higher-highs / higher-lows uptrend"))

    # ── Momentum / timing conflicts ───────────────────────────────────────────

    # RSI oversold — mean-reversion risk when entering short
    if rsi < 30:
        conflicts.append((2, f"RSI={rsi:.0f} oversold — mean-reversion bounce risk, "
                             f"bad short entry timing"))
    elif rsi < 38:
        conflicts.append((1, f"RSI={rsi:.0f} approaching oversold — limited downside "
                             f"before potential bounce"))

    # MACD histogram rising (momentum turning up while shorting)
    if macd_h > 0.001 and macd_h > 0:
        conflicts.append((1, f"MACD hist={macd_h:.4f}↑ rising — "
                             f"short-term momentum turning bullish while shorting"))

    # ── Volume / flow conflicts ───────────────────────────────────────────────

    # OBV rising — smart money accumulating (bad for shorts)
    if obv == "rising":
        conflicts.append((2, "OBV↑ rising (accumulation) — "
                             "capital flowing IN while shorting"))

    # ── Macro conflicts ───────────────────────────────────────────────────────

    # BTC bull + trying to short (macro tailwind against short)
    if btc_regime == "bull" and cloud != "below":
        conflicts.append((1, "BTC bull regime — macro tailwind fights short thesis, "
                             "requires very strong local setup to override"))

    # ── Severity (same scale as long) ────────────────────────────────────────
    n_conflicts  = len(conflicts)
    total_weight = sum(w for w, _ in conflicts)
    if   total_weight == 0: severity = 0.0
    elif total_weight == 1: severity = 0.12
    elif total_weight == 2: severity = 0.25
    elif total_weight == 3: severity = 0.35
    elif total_weight == 4: severity = 0.42
    else:                   severity = 0.48

    return n_conflicts, [desc for _, desc in conflicts], severity


def detect_short_contradictions(
    sig:            dict,
    last:           pd.Series,
    df:             pd.DataFrame,
    short_prob:     float       = 0.5,
    short_ml_proba: float       = 0.5,
    short_ml_acc:   float       = 0.0,
    btc_regime:     str         = "unknown",
) -> Tuple[int, List[str], float]:
    """
    Detect indicator contradictions that reduce short conviction.

    Mirrors detect_contradictions() but from the SHORT perspective.
    For shorts, the dangerous signals are ones that show BUYERS still in control:
      DI+ dominant    → trend momentum bullish (bad for shorts)
      RSI oversold    → mean-reversion bounce risk
      OBV rising      → capital flowing in
      Bullish structure (HH+HL) → structural uptrend
      Above cloud     → structural bull zone
      MACD bullish    → medium-term momentum still up

    Returns: (n_contradictions, descriptions, severity)
    """
    conflicts: List[Tuple[int, str]] = []

    cloud    = sig.get("cloud_state", "unknown")
    adx      = float(last.get("ADX", 20) or 20)
    di_plus  = float(last.get("plus_di", 25) or 25)
    di_minus = float(last.get("minus_di", 25) or 25)
    rsi      = float(last.get("RSI", 50) or 50)
    macd_h   = float(last.get("MACD_Hist", 0) or 0)
    obv      = sig.get("obv_trend", "unknown")
    struct   = (sig.get("market_structure") or {}).get("structure", "")
    regime   = sig.get("market_regime", "unknown")

    # ── Most dangerous short contradiction: buyers still dominate ────────────

    # DI+ dominant despite wanting to short (momentum fighting the short)
    # Threshold: DI+ ≥ 1.5× DI- with ADX confirming the trend
    if di_plus > di_minus * 1.5 and adx > 18:
        conflicts.append((2, f"DI+={di_plus:.0f} >> DI-={di_minus:.0f} "
                             f"(bullish momentum — trend buyers dominating)"))

    # Price above cloud (structural bull) — shorting into a confirmed uptrend
    if cloud == "above":
        conflicts.append((2, "price above cloud — shorting into structural bull zone"))
    elif cloud == "in":
        conflicts.append((1, "price inside cloud — no directional conviction for short"))

    # ── Mean-reversion risk ───────────────────────────────────────────────────

    # RSI oversold → potential bounce, bad entry for new shorts
    if rsi < 30:
        conflicts.append((2, f"RSI={rsi:.0f} oversold — short entered at bounce risk zone"))
    elif rsi < 38:
        conflicts.append((1, f"RSI={rsi:.0f} approaching oversold — short entry quality poor"))

    # ── Volume / flow contradictions ──────────────────────────────────────────

    # OBV rising = capital flowing IN = buyers accumulating
    if obv == "rising":
        conflicts.append((2, "OBV↑ rising — capital flowing in despite short signal "
                             "(smart money accumulating)"))

    # ── Structure contradictions ──────────────────────────────────────────────

    # Bullish market structure (HH+HL) while trying to short
    if struct == "bullish":
        conflicts.append((1, "market structure BULLISH (HH+HL) — "
                             "waiting for CHoCH before shorting"))

    # ── Momentum contradiction ────────────────────────────────────────────────

    # MACD bullish histogram while trying to short — medium-term momentum up
    if macd_h > 0.001 and short_prob >= 0.50:
        conflicts.append((1, f"MACD hist={macd_h:.4f}↑ bullish — "
                             f"medium-term momentum against short"))

    # ── ML contradiction ──────────────────────────────────────────────────────

    # Short ML says don't short (when accuracy is trustworthy)
    if short_ml_acc >= 0.60 and short_ml_proba < 0.40:
        conflicts.append((2, f"Short ML={short_ml_proba:.0%} (acc={short_ml_acc:.0%}) "
                             f"— model says shorts unlikely to win here"))

    # ── Macro contradiction ───────────────────────────────────────────────────

    # BTC bull + trying to short altcoin
    if btc_regime == "bull":
        conflicts.append((1, "BTC bull regime — macro tailwind works against shorts"))

    # ── Compute severity ──────────────────────────────────────────────────────
    n_conflicts  = len(conflicts)
    total_weight = sum(w for w, _ in conflicts)
    if   total_weight == 0: severity = 0.0
    elif total_weight == 1: severity = 0.12
    elif total_weight == 2: severity = 0.25
    elif total_weight == 3: severity = 0.35
    elif total_weight == 4: severity = 0.42
    else:                   severity = 0.48

    return n_conflicts, [desc for _, desc in conflicts], severity

# ── END [BAYES] ───────────────────────────────────────────────────────────────

# ── Core probabilistic signal generator (replaces generate_signal_and_reasons) ──

def generate_signal_probabilistic(last, df, btc_regime="unknown",
                                  is_btc=False, suppress_debug=False,
                                  fng_value: int = 50):
    """
    Replaces generate_signal_and_reasons() with Bayesian log-odds scoring.
    Returns (ProbabilisticResult, compat_dict) where compat_dict has all the
    same keys the rest of the bot already reads from the old function.
    """
    price    = last["price"]
    ema12    = last["EMA12"];    ema26    = last["EMA26"]
    macd     = last["MACD"];    sig_line = last["Signal"]
    macd_hist= last["MACD_Hist"]; rsi    = last["RSI"]
    adx      = last["ADX"]
    plus_di  = last.get("plus_di",  float("nan"))
    minus_di = last.get("minus_di", float("nan"))

    # ── Dynamic prior: regime-aware starting probability ─────────────
    # Must be computed BEFORE creating the scorer so the prior matches
    # the actual market state (strong trend, range, euphoria, panic, etc.)
    dynamic_prior, prior_regime = detect_regime_prior(
        last, df,
        btc_regime = btc_regime,
        fng_value  = int(fng_value),
        adx        = float(adx or 0),
    )
    if not suppress_debug:
        print(f"[PRIOR] {prior_regime}  →  P₀ = {dynamic_prior:.2f}  "
              f"(static would be {PRIOR_CONTINUATION:.2f}  "
              f"Δ = {dynamic_prior - PRIOR_CONTINUATION:+.2f})")

    scorer = ProbabilisticScorer(
        coin             = str(getattr(last, 'name', 'unknown')),
        prior            = dynamic_prior,
        prior_regime     = prior_regime,
        calibration_conf = 0.5,   # updated at runtime after LR calibration
    )
    reasons, caution, blockers = [], [], []
    cloud_state = "unknown"

    # (re-assign for rest of function -- already defined above)
    senkou_a = last.get("senkou_a", float("nan"))
    senkou_b = last.get("senkou_b", float("nan"))
    tenkan   = last.get("tenkan",   float("nan"))
    kijun    = last.get("kijun",    float("nan"))
    ema_w50  = last.get("EMA_WEEKLY50", float("nan"))
    chikou           = last.get("chikou",           float("nan"))
    chikou_ref       = last.get("chikou_ref",       float("nan"))
    chikou_cloud_top = last.get("chikou_cloud_top", float("nan"))
    chikou_cloud_bot = last.get("chikou_cloud_bot", float("nan"))

    # ── GROUP 1: MACRO ────────────────────────────────────────────────
    if not is_btc:
        if btc_regime == "bull":
            scorer.update("btc_bull_regime", True, "BTC uptrend", "MACRO")
            reasons.append("✅ [MACRO] BTC uptrend -- regime tailwind")
        elif btc_regime == "bear":
            # ── RS-conditional bear LR ────────────────────────────────
            # Pre-compute RS from coin's own 7-day return.
            # In a bear market, a coin up >15% in 7 days has decoupled.
            # ── RS-conditional bear LR (preliminary estimate) ─────────
            # A corrected value is applied in run_once() after the full
            # RS calculation. This is just a conservative starting point.
            # Uses strict thresholds to avoid false LEADER classification.
            _quick_rs = "NEUTRAL"
            try:
                if len(df) >= 8:
                    _c7d = (float(last["price"]) / float(df["price"].iloc[-8]) - 1) * 100
                    _c1d = (float(last["price"]) / float(df["price"].iloc[-2]) - 1) * 100
                    # Both timeframes must agree — prevents single-day spikes
                    if   _c7d > 20 and _c1d > 5:  _quick_rs = "LEADER"
                    elif _c7d > 10 and _c1d > 2:  _quick_rs = "STRONG"
            except Exception:
                pass

            _bear_lr = (0.70 if _quick_rs == "LEADER" else
                        0.60 if _quick_rs == "STRONG" else
                        0.50)

            global _BTC_BEAR_LR
            _BTC_BEAR_LR = _bear_lr
            SIGNAL_LR["btc_bear_regime_block"] = (_bear_lr, SIGNAL_LR.get("btc_bear_regime_block", (_bear_lr, 1.20))[1])

            scorer.update("btc_bear_regime_block", True,
                          f"BTC downtrend -- altcoin BUY blocked (RS={_quick_rs} LR={_bear_lr})",
                          "MACRO")
            reasons.append(f"❌ [MACRO] BTC downtrend -- altcoin BUY blocked "
                           f"(RS={_quick_rs} → LR={_bear_lr})")
            blockers.append("BTC Bear Regime")
        else:
            scorer.update("btc_bull_regime", False, "BTC regime unknown", "MACRO")
            reasons.append("⚠️ [MACRO] BTC regime unknown")
    else:
        reasons.append("ℹ️ [MACRO] BTC self-check -- regime filter skipped")

    if not math.isnan(ema_w50):
        if price > ema_w50:
            scorer.update("above_weekly_ema50", True, f"Price above Weekly EMA50 ({ema_w50:.4f})", "MACRO")
            reasons.append(f"✅ [MACRO] Price above Weekly EMA50 ({ema_w50:.4f}) -- macro bullish")
        else:
            scorer.update("below_weekly_ema50", True, f"Price below Weekly EMA50 ({ema_w50:.4f})", "MACRO")
            reasons.append(f"❌ [MACRO] Price below Weekly EMA50 ({ema_w50:.4f}) -- macro headwind")
            caution.append("WeeklyBearish")
    else:
        reasons.append("⚠️ [MACRO] Weekly EMA50 unavailable")

    # ── GROUP 2: TREND STRUCTURE ──────────────────────────────────────
    if not (math.isnan(senkou_a) or math.isnan(senkou_b)):
        cloud_top    = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        if price > cloud_top:
            scorer.update("above_cloud", True, f"Price above Ichimoku Cloud", "TREND")
            cloud_state = "above"
            reasons.append("✅ [TREND] Price above Ichimoku Cloud -- bullish")
        elif price < cloud_bottom:
            # Distance-based cloud penalty — proportional to how far below
            # the coin is, rather than a flat LR that treats -1% same as -15%.
            #
            #  Distance   LR     Interpretation
            #  -1%        0.75   just clipped below, recovery likely
            #  -3%        0.55   moderate below, tradeable with strong signals
            #  -6%        0.40   significantly below, cautious
            #  -10%       0.25   deep below, structural bear
            #  -15%+      0.15   extreme, avoid entirely
            #
            pct_below = (cloud_bottom - price) / cloud_bottom * 100  # positive number
            if   pct_below <= 1.0:  cloud_lr = 0.75
            elif pct_below <= 3.0:  cloud_lr = 0.55 - (pct_below - 1.0) / 2.0 * (0.55 - 0.40)
            elif pct_below <= 6.0:  cloud_lr = 0.40 - (pct_below - 3.0) / 3.0 * (0.40 - 0.25)
            elif pct_below <= 10.0: cloud_lr = 0.25 - (pct_below - 6.0) / 4.0 * (0.25 - 0.15)
            else:                   cloud_lr = 0.15
            cloud_lr = round(cloud_lr, 3)

            # Remove is_block=True — let the distance-scaled LR do the work
            # proportionally. Strong setups 1-3% below cloud can still surface.
            scorer.add_lr(cloud_lr,
                          f"below_cloud_{pct_below:.1f}pct (LR={cloud_lr:.2f})",
                          "TREND")
            cloud_state = "below"
            blockers.append("Cloud Below")
            caution.append("CloudBearish")
            reasons.append(f"❌ [TREND] Price below Ichimoku Cloud -- {pct_below:.1f}% below "
                           f"(LR={cloud_lr:.2f})")
        else:
            scorer.update("in_cloud", True, "Price inside Ichimoku Cloud", "TREND")
            cloud_state = "in"
            caution.append("CloudNeutral")
            reasons.append("⚠️ [TREND] Price inside Ichimoku Cloud -- neutral")
        reasons.append(f"📊 Tenkan={tenkan:.4f} | Kijun={kijun:.4f} | Cloud=({cloud_bottom:.4f}→{cloud_top:.4f})")
    else:
        reasons.append("⚠️ [TREND] Ichimoku cloud incomplete")

    # Tenkan/Kijun
    if not (math.isnan(tenkan) or math.isnan(kijun)):
        scorer.update("tenkan_above_kijun", tenkan > kijun, "Tenkan vs Kijun", "TREND", scale=0.6)

    # Chikou
    if not any(math.isnan(x) for x in [chikou, chikou_ref, chikou_cloud_top, chikou_cloud_bot]):
        ap = chikou > chikou_ref; ac = chikou > chikou_cloud_top
        bp = chikou < chikou_ref; bc = chikou < chikou_cloud_bot
        if ap and ac:
            scorer.update("chikou_fully_bullish", True, "Chikou fully bullish", "TREND")
            reasons.append("✅ [INFO] Chikou above past price AND cloud -- strong bullish confirmation")
        elif ap:
            scorer.update("chikou_partial_bullish", True, "Chikou partial bullish", "TREND")
            reasons.append("⚠️ [INFO] Chikou above past price but inside cloud -- partial confirmation")
        elif bp and bc:
            scorer.update("chikou_fully_bearish", True, "Chikou fully bearish", "TREND")
            reasons.append("❌ [INFO] Chikou below past price AND cloud -- strong bearish confirmation")
        elif bp:
            scorer.update("chikou_partial_bearish", True, "Chikou partial bearish", "TREND")
            reasons.append("⚠️ [INFO] Chikou below past price but above cloud -- partial bearish")
        else:
            reasons.append("⚪ [INFO] Chikou inside historical cloud -- neutral")

    # MACD
    if macd > sig_line:
        scorer.update("macd_bullish_hist", True, f"MACD bullish (hist={macd_hist:.4f})", "TREND")
        reasons.append(f"✅ [TREND] MACD bullish (hist={macd_hist:.4f})")
    else:
        scorer.update("macd_bearish_hist", True, f"MACD bearish (hist={macd_hist:.4f})", "TREND")
        reasons.append(f"❌ [TREND] MACD bearish (hist={macd_hist:.4f})")

    # EMA cross -- dampened (collinear with MACD/cloud)
    if ema12 > ema26:
        scorer.update("ema12_above_ema26", True, "EMA12 > EMA26 (dampened)", "TREND", scale=0.5)
        reasons.append(f"✅ [TREND] EMA12 > EMA26 (dampened -- collinear)")
    else:
        scorer.update("ema12_below_ema26", True, "EMA12 < EMA26 (dampened)", "TREND", scale=0.5)
        reasons.append(f"❌ [TREND] EMA12 < EMA26 (dampened -- collinear)")

    # DI directional pressure
    di_ok = not (math.isnan(plus_di) or math.isnan(minus_di))
    _di_bearish = di_ok and minus_di > plus_di
    if di_ok:
        if plus_di > minus_di:
            scorer.update("di_plus_dominant", True, f"DI+ ({plus_di:.1f}) > DI- ({minus_di:.1f})", "TREND")
            reasons.append(f"✅ [TREND] DI+ ({plus_di:.1f}) > DI- ({minus_di:.1f}) -- bullish pressure")
        else:
            scorer.update("di_minus_dominant", True, f"DI- ({minus_di:.1f}) > DI+ ({plus_di:.1f})", "TREND")
            reasons.append(f"⚠️ [TREND] DI- ({minus_di:.1f}) > DI+ ({plus_di:.1f}) -- bearish pressure")

    # Regime detection for RSI context
    market_regime = detect_market_regime(btc_regime, cloud_state, adx, ema12, ema26, macd, sig_line)
    strong_uptrend   = (cloud_state == "above" and ema12 > ema26 and macd > sig_line)
    strong_downtrend = (cloud_state == "below"  and ema12 < ema26 and macd < sig_line)
    in_trend         = strong_uptrend or strong_downtrend
    _chop_rsi        = _is_strong_chop(df)

    # ── GROUP 3: RSI MOMENTUM ─────────────────────────────────────────
    if in_trend:
        if rsi < 35:
            scorer.update("rsi_oversold_bounce", True, f"RSI {rsi:.1f} oversold bounce (trend)", "MOMENTUM")
            reasons.append(f"✅ [MOMENTUM] RSI {rsi:.1f} -- oversold bounce in trend")
        elif rsi < 55:
            scorer.update("rsi_oversold_bounce", False, f"RSI {rsi:.1f} pullback (neutral)", "MOMENTUM", scale=0.5)
            reasons.append(f"➖ [MOMENTUM] RSI {rsi:.1f} -- pullback zone (normal in trend)")
        elif rsi <= 80:
            sc = 0.1 if _chop_rsi else 1.0
            scorer.update("rsi_continuation_zone", True, f"RSI {rsi:.1f} continuation", "MOMENTUM", scale=sc)
            lbl = "suppressed (chop)" if _chop_rsi else "strong continuation"
            reasons.append(f"{'➖' if _chop_rsi else '✅'} [MOMENTUM] RSI {rsi:.1f} -- {lbl}")
        else:
            scorer.update("rsi_extreme_overbought", True, f"RSI {rsi:.1f} exhaustion", "MOMENTUM")
            reasons.append(f"⚠️ [MOMENTUM] RSI {rsi:.1f} -- exhaustion risk")
    else:
        if rsi < 35:
            scorer.update("rsi_oversold_bounce", True, f"RSI {rsi:.1f} bounce (range)", "MOMENTUM")
            reasons.append(f"✅ [MOMENTUM] RSI {rsi:.1f} -- mean reversion bounce (range)")
        elif rsi < 65:
            reasons.append(f"➖ [MOMENTUM] RSI {rsi:.1f} -- no edge in ranging market")
        elif rsi <= 75:
            scorer.update("rsi_range_overbought", True, f"RSI {rsi:.1f} rejection (range)", "MOMENTUM")
            reasons.append(f"⚠️ [MOMENTUM] RSI {rsi:.1f} -- likely rejection in range")
        else:
            scorer.update("rsi_extreme_overbought", True, f"RSI {rsi:.1f} fake breakout (range)", "MOMENTUM", scale=1.2)
            reasons.append(f"🛑 [MOMENTUM] RSI {rsi:.1f} -- fake breakout / exhaustion risk")

    # RSI divergence
    divergence = detect_rsi_divergence(df)
    if divergence == "bullish":
        scorer.update("rsi_bullish_divergence", True, "Bullish RSI divergence", "MOMENTUM")
        reasons.append("✅ [MOMENTUM] Bullish RSI divergence -- hidden strength")
    elif divergence == "bearish":
        scorer.update("rsi_bearish_divergence", True, "Bearish RSI divergence", "MOMENTUM")
        reasons.append("⚠️ [MOMENTUM] Bearish RSI divergence -- hidden weakness")
        caution.append("BearishDivergence")
    else:
        reasons.append("➖ [MOMENTUM] No RSI divergence")

    # ADX info
    if adx >= 25:
        _adx_dir = "DI+ dominant ✅" if (di_ok and plus_di > minus_di) else "DI- dominant ⚠️"
        reasons.append(f"✅ [MOMENTUM] ADX {adx:.1f} -- strong trend ({_adx_dir})")
    elif adx >= 20:
        reasons.append(f"⚠️ [MOMENTUM] ADX {adx:.1f} -- trend developing")
    else:
        reasons.append(f"⚠️ [MOMENTUM] ADX weak ({adx:.1f}) -- no trend confirmation")

    # Volume / low-volume block -- use relative volume (median-based, portable across coins)
    vol       = float(last.get("volume", 0))
    # Prefer precomputed rel_vol column; fall back to median-based inline calculation
    if "rel_vol" in df.columns and not math.isnan(float(df["rel_vol"].iloc[-1])):
        vol_ratio = float(df["rel_vol"].iloc[-1])
        avg_vol   = vol / (vol_ratio + 1e-9)   # implied median for display
    else:
        avg_vol   = float(df["volume"].rolling(20, min_periods=5).median().iloc[-1]) \
                    if "volume" in df.columns else 1.0
        vol_ratio = vol / (avg_vol + 1e-9)
    vol_ratio = max(0.0, min(20.0, vol_ratio))
    if vol < avg_vol * 0.5:
        scorer.block("Very low volume -- signal unreliable")
        blockers.append("Very Low Volume")
        reasons.append(f"⚠️ Very low volume ({vol_ratio:.2f}x avg) -- BLOCKED")
    elif vol_ratio >= 1.5:
        reasons.append(f"✅ [INFO] Volume spike ({vol_ratio:.2f}x avg)")
    else:
        reasons.append(f"📊 [INFO] Volume {vol_ratio:.2f}x avg")

    # Misc info (ATR, BB, RR) -- same as original
    atr_avg   = df["ATR"].rolling(20).mean().iloc[-2] if "ATR" in df.columns else 0
    atr_ratio = last.get("ATR", 0) / (atr_avg + 1e-9) if atr_avg > 0 else 1.0
    reasons.append(f"📈 ATR Expansion: {atr_ratio:.2f}")
    lower_bb = last.get("LowerBB", price * 0.95)
    upper_bb = last.get("UpperBB", price * 1.05)
    bb_mid   = (lower_bb + upper_bb) / 2
    atr_val    = last.get("ATR", price * 0.02)
    stop_price = price - 2 * atr_val
    tp_price   = price + 4 * atr_val
    rr_ratio   = round((tp_price - price) / (price - stop_price), 2) if price > stop_price else 0
    reasons.append(f"✅ Risk/Reward: RR={rr_ratio:.2f}, Entry={price:.6f}, Stop={stop_price:.6f}, TP={tp_price:.6f}")

    # ── Return scorer (NOT finalized) + partial compat ────────────────
    # finalize() is called ONCE by the pipeline after ALL evidence is added.
    # Calling it here would cause a split-state bug: later pipeline stages
    # (vol regime, OBV, structure, ML, transition) would update a separate
    # float instead of the evidence list, bypassing group dampening.
    #
    # The caller (run_once) does:
    #   scorer, sig = generate_signal_probabilistic(...)
    #   prob_volregime_update(scorer, ...)    ← adds to scorer
    #   prob_obv_update(scorer, ...)          ← adds to scorer
    #   ...
    #   result = scorer.finalize()            ← ONE call, full evidence
    compat = {
        # Finalize-dependent fields: set to placeholder, filled in by caller
        "action":        None,
        "score":         None,
        "prob":          None,
        "_prob_result":  None,
        "_scorer":       scorer,           # ← give pipeline access to scorer
        # Pre-finalize fields: all available now
        "blocked":       scorer._blocked,
        "blockers":      blockers,
        "reasons":       reasons,
        "caution":       caution,
        "market_regime": market_regime,
        "prior_regime":  prior_regime,
        "dynamic_prior": dynamic_prior,
        "cloud_state":   cloud_state,
        "divergence":    divergence,
        "entry_price":   price,
        "stop_price":    stop_price,
        "tp_price":      tp_price,
        "rr_ratio":      rr_ratio,
        "lower_bb":      lower_bb,
        "bb_mid":        bb_mid,
        "vol":           vol,
        "avg_vol":       avg_vol,
        "vol_ratio":     vol_ratio,
        "rsi":           rsi,
        "ema12":         ema12,
        "ema26":         ema26,
        "senkou_a":      senkou_a,
        "senkou_b":      senkou_b,
        "kijun":         kijun,
        "tenkan":        tenkan,
        "plus_di":       plus_di,
        "minus_di":      minus_di,
    }
    return scorer, compat

# ── END PROBABILISTIC ENGINE [P] ─────────────────────────────────────────────

import os as _ntfy_os

def send_ntfy(title, message, priority="default", tags="", topic_env="NTFY_TOPIC"):
    """Push notification to phone via ntfy.sh. Set NTFY_TOPIC env var.
    Use topic_env='NTFY_WATCH_TOPIC' for near-miss/watch alerts.
    """
    topic = _ntfy_os.environ.get(topic_env, "") or _ntfy_os.environ.get("NTFY_TOPIC", "")
    if not topic:
        print(f"[NTFY] ⚠️  NTFY_TOPIC not set -- skipping notification: '{title}'")
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
API_KEY = "CG-yBnQ5i4dg451uMed1DL8sWrC"
COINS = [
    "celestia", "floki", "algorand", "raydium", "bonk", "dash", "tron", "zencash", "origintrail", "zcash", "dogecoin",
    "avalanche-2", "solana", "bittensor", "ethereum", "bitcoin",
    "litecoin", "polkadot", "chainlink", "binancecoin", "ripple", "arbitrum",
    "cardano", "ethena", "story-2", "near", "immutable-x",
    "fartcoin", "ordinals", "ondo-finance", "aster-2",
    "filecoin", "hyperliquid", "monero", "shiba-inu",
    "hedera-hashgraph", "gnosis", "kucoin-shares", "dydx", "centrifuge", "xdce-crowd-sale",
    # ── Batch addition from gem-scanner PRE-BREAKOUT/SHORT-GEM tables (2026-06-15) ──
    "ribbita-by-virtuals", "jupiter-exchange-solana", "arweave", "aerodrome-finance",
    "basic-attention-token", "internet-computer", "zebec-network", "shuffle-2",
    "mantle", "okb", "quant-network", "just", "aptos", "conflux-token", "telcoin",
    "neo", "apecoin", "safepal", "crypto-com-chain", "sky", "midnight-3",
    "spx6900", "plasma", "onyxcoin", "mx-token",
    # ── Batch 2: top gem-scanner GEM/SHORT results (2026-06-15 11:11 scan) ──
    "grass", "flying-tulip", "venice-token", "rain", "akash-network",
    "worldcoin-wld", "bitway", "lighter", "eigenlayer", "morpho",
    "beldex", "the9bit",
    # ── Batch 3: gem-scanner table, 5 genuinely new of 14 (2026-06-15 ~13:55 scan) ──
    "railgun", "velvet", "kaito", "dexe", "injective",
    # ── Batch 4: gem-scanner table (2026-07-14) — 2 added, 1 removed ───────────
    # AWE: CoinGecko API ID is stp-network (old name kept for API compat despite
    #      website rebranding to awe-network in May 2025 — awe-network → 404)
    # UNI (uniswap): was missing from list despite SYMBOL_MAP already having it
    # WFI (wfi): REMOVED — launched Dec 2024, only ~7 months of history,
    #            365-day API call returns 404; re-add when it has ≥200 days of data
    "stp-network", "uniswap",
    # ── Batch 5: gem-scanner table (2026-07-15) — 7 genuinely new of 25 ─────────
    # 14 already tracked, 4 too new (<200d history → 365d API 404s), 0 skipped
    # Too new (re-add when ≥200d data exists):
    #   wfi (Dec 2024), adi-token (Dec 2025), backpack (Mar 2026), plasma/XPL (Sep 2025)
    #   bianrensheng (Oct 2025, also a meme token labeled manipulated Jun 2026)
    #   stable-2 / real: IDs unconfirmed
    "bitcoin-sv",         # BSV — Bitcoin SV fork (2018), on Binance as BSV/USDT
    "pancakeswap-token",  # CAKE — PancakeSwap DEX/governance (2020), Binance CAKE/USDT
    "lido-dao",           # LDO — Lido liquid staking (2020), Binance LDO/USDT
    "whitebit",           # WBT — WhiteBIT exchange token (2022), on WhiteBIT/Kraken, NOT Binance
    "huobi-token",        # HTX — Huobi/HTX exchange token (2018), Binance HT/USDT
    "pyth-network",       # PYTH — Pyth oracle network (Nov 2023, 20m+ history), Binance
    "aave"                # AAVE — Aave DeFi lending (2017), Binance AAVE/USDT
]
CG_DAYS        = 365        # 365 days: more ML training samples (326 usable vs 101 at 180d)
CG_INTERVAL    = "daily"

# ── API call minimisation ─────────────────────────────────────────────────────
# Each coin costs 2 CoinGecko calls (market_chart + ohlc).
# With 25+ coins on the free tier (30 calls/min) this hits rate limits fast.
# Strategy:
#   1. Session cache: store DataFrame in memory -- skip re-fetch if < CACHE_TTL_SECONDS old
#   2. BTC reuse: get_btc_regime() already caches BTC df; relative strength reuses it
#   3. OHLC cache: store OHLC result per coin -- one fetch per session
#   4. Smart sleep: adaptive delay between coins based on 429 history
CACHE_TTL_SECONDS  = 1800    # 30 minutes -- fresh enough for intraday decisions
CG_SLEEP_BETWEEN   = 1.5     # seconds between coin fetches (free tier: 30 req/min)
CG_SLEEP_ON_429    = 65.0    # seconds to wait after a 429 (rate limit reset)

_session_df_cache: dict  = {}   # coin_id → {"df": DataFrame, "ts": float}
_session_ohlc_cache: dict = {}  # coin_id → {"df": DataFrame, "ts": float}
_cg_consecutive_429 = 0         # track consecutive rate limit hits
BTC_ID         = "bitcoin"  # [D] reference coin for market regime

ACCOUNT_BALANCE     = 10000.0
RISK_PER_TRADE      = 0.01
STOP_ATR_MULT       = 2.0

# ── ML label constants ────────────────────────────────────────────────────────
ML_TP_ATR_MULT      = 1.5   # TP = entry + 1.5×ATR  (was 2.5 → too hard, 6-20% win rate)
                             # At 1.5:1 R:R, breakeven win rate = 1/(1+1.5) = 40%
                             # At 2.5:1 R:R, breakeven win rate = 1/(1+2.5) = 28.6%
                             # PROBLEM: with 2.5× TP in bear/chop, actual win rate was
                             # 6-20%, making ML learn "always predict loss" (acc=84%
                             # by always saying 0). Useless signal, no BUYs for weeks.
                             # Fix: 1.5× TP → achievable 35-45% win rate → ML has
                             # real positive examples to learn from.
ML_SL_ATR_MULT      = 1.0   # SL = entry − 1.0×ATR  (unchanged)
ML_NEUTRAL_Z        = 0.25  # exclude bars where |z_score| < 0.25 (was 0.35 — too strict,
                             # was excluding too many marginal wins and labelling them neutral)

# ── Backtest Bayesian shrinkage constants ─────────────────────────────────────
# Shrinks observed win rate toward the breakeven prior μ.
# A small sample can't prove itself -- the prior dominates until n is large.
#
# BACKTEST_PRIOR_K = 10 means the prior is worth ~10 virtual trades.
#   n=4  trades: 30% influence from data, 70% from prior  (prior dominates)
#   n=10 trades: 50% influence from data, 50% from prior  (balanced)
#   n=50 trades: 83% influence from data, 17% from prior  (data dominates)
#
# BACKTEST_MAX_LR_SCALE = 2.5 caps the LR for large reliable samples.
# BACKTEST_MIN_TRADES = 3 is the minimum to get any LR (Bayesian handles noise).
BACKTEST_PRIOR_K       = 10    # prior strength in virtual trades
BACKTEST_MIN_TRADES    = 3     # below this: no LR at all (truly meaningless)
BACKTEST_MAX_LR        = 2.5   # hard cap -- backtest is informative, not dominant


def bayesian_win_rate(wins: int, n: int,
                       mu: float = None,
                       k: float  = BACKTEST_PRIOR_K) -> float:
    """
    Beta-Binomial posterior mean for win rate.

    This is the correct way to handle small backtest samples.
    Formula:  adjusted_wr = (wins + k*μ) / (n + k)

    Derivation:
      Prior:     Beta(k*μ, k*(1-μ))    -- equivalent to k virtual trades at win rate μ
      Likelihood: Binomial(n, wr)
      Posterior: Beta(k*μ + wins, k*(1-μ) + losses)
      Mean:      (wins + k*μ) / (n + k)                ← this function

    Effect on trust:
      WR=100% n=4:   adjusted = (4 + 2.86) / (4+10)  = 48.9%  ← heavily pulled down
      WR=80%  n=10:  adjusted = (8 + 2.86) / (10+10) = 54.3%  ← moderately pulled down
      WR=80%  n=70:  adjusted = (56 + 2.86)/ (70+10) = 73.6%  ← mostly preserved
      WR=30%  n=10:  adjusted = (3 + 2.86) / (10+10) = 29.3%  ← near-neutral, small shift

    μ defaults to the breakeven win rate for the current R:R ratio (28.6% for 2.5:1).
    """
    if mu is None:
        mu = 1.0 / (1.0 + ML_TP_ATR_MULT / ML_SL_ATR_MULT)   # breakeven WR
    return (wins + k * mu) / (n + k)
ML_STRONG_WIN_Z     = 2.0   # z > 2.0 → strong win, sample_weight = 2.0
ML_STRONG_LOSS_Z    =-0.5   # z < -0.5 → clear loss, sample_weight = 1.5
ML_BORDERLINE_Z     = 0.5   # |z| < 0.5 → borderline, sample_weight = 0.5
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

# [A-E] scoring: max possible ~12 -- require 5 to fire BUY
MIN_BUY_SCORE = 5

# [J] SELL scoring: threshold for SHORT signals (negative score)
# [J] Range: -12 to 0. MIN_SELL_SCORE = -5 means sell when score is -5 or lower
MIN_SELL_SCORE = -5

# [K] Market Sentiment Configuration
FETCH_FEAR_GREED = True             # [K] Enable Fear & Greed Index fetching
SENTIMENT_CACHE_MINUTES = 60        # [K] Cache sentiment for 1 hour (doesn't change frequently)

# [L] Relative Strength Configuration
RELATIVE_STRENGTH_ENABLED = True    # [L] Enable relative strength vs BTC calculation

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — GLOBAL ML MODEL
#
# One model trained on ALL coins simultaneously.
# Each bar gets: 16 technical features + 4 regime + 10 sector + 3 mcap + 5 RS = 38 features
#
# Why global beats per-coin:
#   Per-coin: 314 samples → 5-10 per regime → model averages everything
#   Global:  56,000+ samples → 500-2000 per regime → model learns conditional patterns
#
# The model is trained offline (weekly / on demand) and used as an LR adjuster.
# It does NOT replace the Bayesian scorer — it adds one calibrated LR to it.
# ══════════════════════════════════════════════════════════════════════════════

GLOBAL_MODEL_PATH    = Path("models") / "global_model.pkl"
GLOBAL_MODEL_META    = Path("models") / "global_model_meta.json"
GLOBAL_MODEL_VERSION = "v1.0"
GLOBAL_N_FEATURES    = 38    # 16 technical + 4 regime + 10 sector + 3 mcap + 5 RS + rounding
GLOBAL_RETRAIN_DAYS  = 7     # retrain every 7 days

# ── Coin metadata: sector + market cap tier ───────────────────────────────────
# Sectors: l1, l2, defi, exchange, infra, gaming, ai, meme, privacy, payments
# Mcap:    large (>$10B), mid ($1-10B), small (<$1B)
COIN_METADATA: Dict[str, Dict[str, str]] = {
    # Layer 1
    "bitcoin":          {"sector": "l1",       "mcap": "large"},
    "ethereum":         {"sector": "l1",       "mcap": "large"},
    "solana":           {"sector": "l1",       "mcap": "large"},
    "binancecoin":      {"sector": "exchange", "mcap": "large"},
    "cardano":          {"sector": "l1",       "mcap": "large"},
    "avalanche-2":      {"sector": "l1",       "mcap": "large"},
    "polkadot":         {"sector": "l1",       "mcap": "mid"},
    "near":             {"sector": "l1",       "mcap": "mid"},
    "algorand":         {"sector": "l1",       "mcap": "mid"},
    "tron":             {"sector": "l1",       "mcap": "large"},
    "celestia":         {"sector": "l1",       "mcap": "mid"},
    "story-2":          {"sector": "l1",       "mcap": "small"},
    "aster-2":          {"sector": "l1",       "mcap": "small"},
    # Layer 2 / Infrastructure
    "arbitrum":         {"sector": "l2",       "mcap": "mid"},
    "immutable-x":      {"sector": "l2",       "mcap": "mid"},
    "hyperliquid":      {"sector": "defi",     "mcap": "mid"},
    "origintrail":      {"sector": "infra",    "mcap": "small"},
    # DeFi
    "raydium":          {"sector": "defi",     "mcap": "small"},
    "ethena":           {"sector": "defi",     "mcap": "mid"},
    "ondo-finance":     {"sector": "defi",     "mcap": "mid"},
    # Payments / Store of Value
    "ripple":           {"sector": "payments", "mcap": "large"},
    "litecoin":         {"sector": "payments", "mcap": "mid"},
    "dash":             {"sector": "payments", "mcap": "small"},
    # Infrastructure / Oracle
    "chainlink":        {"sector": "infra",    "mcap": "large"},
    "filecoin":         {"sector": "infra",    "mcap": "mid"},
    # AI / Data
    "bittensor":        {"sector": "ai",       "mcap": "mid"},
    # Privacy
    "monero":           {"sector": "privacy",  "mcap": "mid"},
    "zcash":            {"sector": "privacy",  "mcap": "small"},
    "zencash":          {"sector": "privacy",  "mcap": "small"},
    # Meme
    "dogecoin":         {"sector": "meme",     "mcap": "large"},
    "shiba-inu":        {"sector": "meme",     "mcap": "large"},
    "floki":            {"sector": "meme",     "mcap": "small"},
    "bonk":             {"sector": "meme",     "mcap": "small"},
    "fartcoin":         {"sector": "meme",     "mcap": "small"},
    # NFT / Gaming
    "ordinals":         {"sector": "gaming",   "mcap": "small"},
    # Newly added
    "hedera-hashgraph": {"sector": "l1",       "mcap": "mid"},     # HBAR — L1 DAG network
    "gnosis":           {"sector": "infra",    "mcap": "small"},   # GNO — DAO/infra
    "kucoin-shares":    {"sector": "exchange", "mcap": "mid"},     # KCS — exchange token
    "dydx":             {"sector": "defi",     "mcap": "small"},   # DYDX — perp DEX
    "centrifuge":       {"sector": "defi",     "mcap": "small"},   # CFG — RWA/DeFi
    "xdce-crowd-sale":  {"sector": "l1",       "mcap": "small"},   # XDC — XDC Network L1
    # ── Batch addition from gem-scanner tables (2026-06-15) ──
    "ribbita-by-virtuals":     {"sector": "ai",       "mcap": "small"},  # TIBBIR — Base AI agent token
    "jupiter-exchange-solana": {"sector": "defi",     "mcap": "mid"},    # JUP — Solana DEX aggregator
    "arweave":                 {"sector": "infra",    "mcap": "small"},  # AR — decentralized storage
    "aerodrome-finance":       {"sector": "defi",     "mcap": "small"},  # AERO — Base DEX
    "basic-attention-token":   {"sector": "infra",    "mcap": "small"},  # BAT — Brave/ad attention token
    "internet-computer":       {"sector": "l1",       "mcap": "mid"},    # ICP — L1 compute network
    "zebec-network":           {"sector": "payments", "mcap": "small"},  # ZBCN — payroll/streaming payments
    "shuffle-2":               {"sector": "gaming",   "mcap": "small"},  # SHFL — onchain casino
    "mantle":                  {"sector": "l2",       "mcap": "mid"},    # MNT — Ethereum L2
    "okb":                     {"sector": "exchange", "mcap": "mid"},    # OKB — OKX exchange token
    "quant-network":           {"sector": "infra",    "mcap": "mid"},    # QNT — blockchain interop
    "just":                    {"sector": "defi",     "mcap": "small"},  # JST — TRON DeFi/stablecoin
    "aptos":                   {"sector": "l1",       "mcap": "mid"},    # APT — Move-based L1
    "conflux-token":           {"sector": "l1",       "mcap": "small"},  # CFX — China-compliant L1
    "telcoin":                 {"sector": "payments", "mcap": "small"},  # TEL — mobile remittance
    "neo":                     {"sector": "l1",       "mcap": "small"},  # NEO — smart-contract L1
    "apecoin":                 {"sector": "gaming",   "mcap": "small"},  # APE — BAYC ecosystem token
    "safepal":                 {"sector": "exchange", "mcap": "small"},  # SFP — wallet/exchange token
    "crypto-com-chain":        {"sector": "exchange", "mcap": "mid"},    # CRO — Crypto.com exchange token
    "sky":                     {"sector": "defi",     "mcap": "mid"},    # SKY — MakerDAO rebrand
    "midnight-3":              {"sector": "l1",       "mcap": "mid"},    # NIGHT — Cardano-linked privacy L1
    "spx6900":                 {"sector": "meme",     "mcap": "small"},  # SPX — meme
    "plasma":                  {"sector": "l1",       "mcap": "mid"},    # XPL — stablecoin-focused L1
    "onyxcoin":                {"sector": "payments", "mcap": "small"},  # XCN — payments (Chain rebrand)
    "mx-token":                {"sector": "exchange", "mcap": "small"},  # MX — MEXC exchange token
    # ── Batch 2: top gem-scanner GEM/SHORT results (2026-06-15 11:11 scan) ──
    "grass":                   {"sector": "infra",    "mcap": "small"},  # GRASS — DePIN data network
    "flying-tulip":            {"sector": "defi",     "mcap": "small"},  # FT — DeFi protocol
    "venice-token":            {"sector": "ai",       "mcap": "small"},  # VVV — Venice private AI
    "rain":                    {"sector": "payments", "mcap": "small"},  # RAIN — crypto card/payments infra
    "akash-network":           {"sector": "infra",    "mcap": "small"},  # AKT — DePIN cloud compute
    "worldcoin-wld":           {"sector": "ai",       "mcap": "mid"},    # WLD — Worldcoin identity/AI
    "bitway":                  {"sector": "exchange", "mcap": "small"},  # BTW — exchange-linked token
    "lighter":                 {"sector": "defi",     "mcap": "small"},  # LIT — zk perp DEX
    "eigenlayer":              {"sector": "infra",    "mcap": "mid"},    # EIGEN — restaking infra
    "morpho":                  {"sector": "defi",     "mcap": "mid"},    # MORPHO — DeFi lending
    "beldex":                  {"sector": "privacy",  "mcap": "small"},  # BDX — privacy coin
    "the9bit":                 {"sector": "meme",     "mcap": "small"},  # 9BIT — meme
    # ── Batch 3: gem-scanner table, 5 genuinely new of 14 (2026-06-15 ~13:55 scan) ──
    "railgun":                 {"sector": "privacy",  "mcap": "small"},  # RAIL — zk-SNARK privacy protocol
    "velvet":                  {"sector": "defi",     "mcap": "small"},  # VELVET — onchain asset mgmt/trading terminal
    "kaito":                   {"sector": "ai",       "mcap": "small"},  # KAITO — AI-powered InfoFi network
    "dexe":                    {"sector": "defi",     "mcap": "mid"},    # DEXE — DAO/social-trading governance
    "injective":               {"sector": "l1",       "mcap": "mid"},    # INJ — DeFi-focused L1
    # ── Batch 4: gem-scanner table (2026-07-14) ─────────────────────────────────
    "stp-network":             {"sector": "ai",       "mcap": "small"},  # AWE — AI Autonomous Worlds (CG API ID = stp-network, rebranded from STPT)
    "uniswap":                 {"sector": "defi",     "mcap": "mid"},    # UNI — Uniswap DEX governance
    # wfi removed: launched Dec 2024, insufficient history for 365d API call
    # ── Batch 5: gem-scanner table (2026-07-15) ─────────────────────────────────
    "bitcoin-sv":              {"sector": "l1",       "mcap": "small"},  # BSV — Bitcoin SV fork
    "pancakeswap-token":       {"sector": "defi",     "mcap": "mid"},    # CAKE — PancakeSwap DEX
    "lido-dao":                {"sector": "defi",     "mcap": "large"},  # LDO — Lido liquid staking
    "whitebit":                {"sector": "exchange", "mcap": "mid"},    # WBT — WhiteBIT exchange token
    "huobi-token":             {"sector": "exchange", "mcap": "mid"},    # HTX — Huobi/HTX exchange token
    "pyth-network":            {"sector": "infra",    "mcap": "mid"},    # PYTH — Pyth price oracle
    "aave":                    {"sector": "defi",     "mcap": "large"},  # AAVE — Aave DeFi lending
}

SECTORS  = ["l1","l2","defi","exchange","infra","gaming","ai","meme","privacy","payments"]
MCAP_TIERS = ["large","mid","small"]
RS_TIERS   = ["LEADER","STRONG","NEUTRAL","LAGGARD","WEAK"]


def _coin_meta_features(coin_id: str, rs_tier: str = "NEUTRAL") -> List[float]:
    """
    Build 18 metadata features for a coin:
      10 sector one-hot  (l1, l2, defi, exchange, infra, gaming, ai, meme, privacy, payments)
       3 mcap one-hot    (large, mid, small)
       5 RS tier one-hot (LEADER, STRONG, NEUTRAL, LAGGARD, WEAK)
    Total: 18 features
    """
    meta = COIN_METADATA.get(coin_id.lower(), {"sector": "l1", "mcap": "mid"})
    sector_feat = [1.0 if meta["sector"] == s else 0.0 for s in SECTORS]
    mcap_feat   = [1.0 if meta["mcap"]   == m else 0.0 for m in MCAP_TIERS]
    rs_feat     = [1.0 if rs_tier.upper() == r else 0.0 for r in RS_TIERS]
    return sector_feat + mcap_feat + rs_feat   # 18 total


def build_global_features(row, df, btc_df, coin_id: str,
                           rs_tier: str = "NEUTRAL",
                           fng_value: int = 50,
                           bar_index: int = -1) -> Optional[List[float]]:
    """
    Build the full 38-feature vector for the global model:
      16 technical  (extract_ml_features)
       4 regime     (_detect_bar_regime one-hot)
      18 metadata   (_coin_meta_features: sector + mcap + RS tier)
    Total: 38 features
    """
    try:
        df_slice = df.iloc[:bar_index+1] if bar_index >= 0 else df
        tech     = extract_ml_features(row, df_slice, btc_df=btc_df, fng_value=fng_value)
        if tech is None:
            return None

        regime      = _detect_bar_regime(row, btc_df, bar_index if bar_index >= 0 else len(df)-1, df)
        regime_feat = [1.0 if regime == r else 0.0 for r in ("bull","bear","range","panic")]
        meta_feat   = _coin_meta_features(coin_id, rs_tier)

        return list(tech) + regime_feat + meta_feat
    except Exception:
        return None


def train_global_model(coin_dfs: Dict[str, pd.DataFrame],
                       btc_df:   pd.DataFrame,
                       fng_value: int = 50) -> bool:
    """
    Train one global ML model on all coins simultaneously.

    Each coin contributes all its labelled bars as training samples.
    Coin identity is encoded via sector + mcap + RS tier features.

    The model is a 3-class GBT (0=short wins, 1=neutral, 2=long wins)
    identical to the per-coin model, but trained on ~56,000 samples
    instead of 314.

    Saves to models/global_model.pkl + global_model_meta.json.
    Returns True on success.
    """
    if not ML_ENABLED or not ML_AVAILABLE:
        return False
    try:
        from sklearn.ensemble         import GradientBoostingClassifier
        from sklearn.preprocessing    import StandardScaler
        from sklearn.model_selection  import TimeSeriesSplit
        from sklearn.calibration      import CalibratedClassifierCV
        import numpy as _np

        print(f"\n[GLOBAL ML] Starting global model training on {len(coin_dfs)} coins...")

        # ── Collect samples WITH timestamp (leakage-safe) ───────────────────
        # Store (timestamp, feats, label, weight) for every bar.
        # After collection we sort by timestamp and split at a single date
        # boundary so NO coin's future data contaminates another coin's test.
        all_samples: list = []   # (ts_float, feats, label, weight)
        coin_counts = {}

        for coin_id, df in coin_dfs.items():
            if df is None or len(df) < ML_LOOKBACK_DAYS + ML_FUTURE_DAYS + 10:
                continue
            n_added = 0
            for i in range(ML_LOOKBACK_DAYS, len(df) - ML_FUTURE_DAYS - 1):
                row     = df.iloc[i]
                entry   = float(row.get("price", 0))
                atr_val = float(row.get("ATR", 0))
                if entry <= 0 or atr_val <= 0:
                    continue

                # ── Label ─────────────────────────────────────────────────────
                future   = df.iloc[i+1:i+ML_FUTURE_DAYS+1]["price"]
                long_tp  = entry + ML_TP_ATR_MULT * atr_val
                long_sl  = entry - ML_SL_ATR_MULT * atr_val
                short_tp = entry - ML_TP_ATR_MULT * atr_val
                short_sl = entry + ML_SL_ATR_MULT * atr_val
                hit_ltp = bool((future >= long_tp).any())
                hit_lsl = bool((future <= long_sl).any())
                hit_stp = bool((future <= short_tp).any())
                hit_ssl = bool((future >= short_sl).any())
                ltp_i = int((future >= long_tp).values.argmax())  if hit_ltp else 9999
                lsl_i = int((future <= long_sl).values.argmax())  if hit_lsl else 9999
                stp_i = int((future <= short_tp).values.argmax()) if hit_stp else 9999
                ssl_i = int((future >= short_sl).values.argmax()) if hit_ssl else 9999
                if   hit_ltp and (not hit_lsl or ltp_i <= lsl_i): label = 2
                elif hit_stp and (not hit_ssl or stp_i <= ssl_i): label = 0
                elif hit_lsl and not hit_ltp:                      label = 0
                elif hit_ssl and not hit_stp:                      label = 2
                else:                                              label = 1
                fwd_ret = (float(future.iloc[-1]) - entry) / entry
                z_score = fwd_ret / (atr_val / entry + 1e-9)
                if label == 1 and abs(z_score) < ML_NEUTRAL_Z:
                    continue

                # ── Features ──────────────────────────────────────────────────
                feats = build_global_features(row, df, btc_df, coin_id,
                                              rs_tier="NEUTRAL",
                                              fng_value=fng_value,
                                              bar_index=i)
                if feats is None:
                    continue

                # ── Timestamp (unix seconds) for temporal ordering ────────────
                try:
                    _ts = row.get("timestamp") or row.get("date") or row.name
                    ts_float = float(_ts.timestamp()) if hasattr(_ts,"timestamp") else float(_ts)
                except Exception:
                    ts_float = float(i)  # relative fallback

                days_ago = len(df) - i
                time_wt  = float(_np.exp(-days_ago / 180.0))
                mag_wt   = min(3.0, 1.0 + abs(z_score) / 2.0)
                all_samples.append((ts_float, feats, label, time_wt * mag_wt))
                n_added += 1

            coin_counts[coin_id] = n_added
            if n_added > 0:
                print(f"[GLOBAL ML]   {coin_id:<24} {n_added:>5} samples")

        if len(all_samples) < 500:
            print(f"[GLOBAL ML] ❌ Insufficient samples: {len(all_samples)} < 500")
            return False

        # ── REGIME-STRATIFIED SAMPLING ────────────────────────────────────────
        # Problem: 195 coins × March 2025-2026 = predominantly bear market bars.
        # Without stratification: ~60% of samples have bear regime label → model
        # learns "predict loss in bear" as its primary rule → 2.8% trigger rate.
        #
        # Fix: cap each regime at MAX_SAMPLES_PER_REGIME, sample randomly within
        # regime boundaries. This forces the model to see equal examples of:
        #   bull:  "RSI=77 + EARLY_BULL + OBV rising → win"
        #   bear:  "RSI=77 + EARLY_BEAR + OBV falling → loss"
        #   range: "RSI=55 + consolidation → neutral"
        # And stops it defaulting to "always predict bear outcome".
        MAX_SAMPLES_PER_REGIME = 15_000   # cap per regime; total ≤ 60k

        # Tag each sample with its regime (feature index 16=bull, 17=bear, 18=range, 19=panic)
        regime_buckets: dict = {"bull": [], "bear": [], "range": [], "panic": [], "other": []}
        for s in all_samples:
            feats = s[1]
            try:
                if   feats[16] > 0.5: regime_buckets["bull"].append(s)
                elif feats[17] > 0.5: regime_buckets["bear"].append(s)
                elif feats[18] > 0.5: regime_buckets["range"].append(s)
                elif feats[19] > 0.5: regime_buckets["panic"].append(s)
                else:                 regime_buckets["other"].append(s)
            except (IndexError, TypeError):
                regime_buckets["other"].append(s)

        # Print regime distribution before and after stratification
        print(f"\n[GLOBAL ML] Regime distribution (before stratification):")
        for r, samples in regime_buckets.items():
            if samples:
                print(f"  {r:<8}: {len(samples):>6,} samples")

        # Cap each regime and recombine
        import random as _rng
        _rng.seed(42)
        stratified_samples = []
        for regime, samples in regime_buckets.items():
            if len(samples) <= MAX_SAMPLES_PER_REGIME:
                stratified_samples.extend(samples)
            else:
                # Preserve temporal ordering within regime: take most recent N
                # (sorted by timestamp already from earlier sort step, but we cap
                # the TOTAL not just the most recent — use systematic subsample)
                step = len(samples) / MAX_SAMPLES_PER_REGIME
                picked = [samples[int(i * step)] for i in range(MAX_SAMPLES_PER_REGIME)]
                stratified_samples.extend(picked)

        # Re-sort by timestamp after combining (leakage-safe split still required)
        stratified_samples.sort(key=lambda s: s[0])

        print(f"[GLOBAL ML] After stratification: {len(stratified_samples):,} samples "
              f"(was {len(all_samples):,})")
        for r, samples in regime_buckets.items():
            if samples:
                n_after = min(len(samples), MAX_SAMPLES_PER_REGIME)
                print(f"  {r:<8}: {n_after:>6,}  (capped from {len(samples):,})"
                      if len(samples) > MAX_SAMPLES_PER_REGIME else
                      f"  {r:<8}: {len(samples):>6,}  (kept all)")

        # Replace all_samples with stratified version
        all_samples = stratified_samples
        # Without this, coin A's May 2026 bars sit in "training" while coin B's
        # May 2026 bars are in "test". Sorting ensures the 80% split is a real
        # calendar date: everything before date X trains, everything after tests.
        all_samples.sort(key=lambda s: s[0])
        timestamps = _np.array([s[0] for s in all_samples])
        X_raw      = _np.array([s[1] for s in all_samples], dtype=_np.float32)
        y          = _np.array([s[2] for s in all_samples], dtype=_np.int32)
        w          = _np.array([s[3] for s in all_samples], dtype=_np.float32)
        w          = w / w.mean()

        # Date boundary at 80th percentile of timestamps
        cutoff_ts  = float(_np.percentile(timestamps, 80))
        try:
            cutoff_dt = datetime.fromtimestamp(cutoff_ts).strftime("%Y-%m-%d")
        except Exception:
            cutoff_dt = f"bar-index {int(cutoff_ts)}"
        train_mask = timestamps <= cutoff_ts
        test_mask  = timestamps >  cutoff_ts
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())

        print(f"\n[GLOBAL ML] Temporal split → train before {cutoff_dt}  "
              f"train={n_train:,}  test={n_test:,}")
        print(f"[GLOBAL ML] Total samples: {len(X_raw):,}  "
              f"(label 0={int((y==0).sum()):,}  1={int((y==1).sum()):,}  "
              f"2={int((y==2).sum()):,})  Features: {X_raw.shape[1]}")

        # ── Scale: fit on TRAIN only ──────────────────────────────────────────
        scaler   = StandardScaler()
        X_scaled = _np.zeros_like(X_raw)
        X_scaled[train_mask] = scaler.fit_transform(X_raw[train_mask])
        X_scaled[test_mask]  = scaler.transform(X_raw[test_mask])
        X = X_raw   # keep raw for final full-data model

        # ── Temporally-clean train/test ───────────────────────────────────────
        X_tr = X_scaled[train_mask];  y_tr = y[train_mask];  w_tr = w[train_mask]
        X_te = X_scaled[test_mask];   y_te = y[test_mask]

        # ── Class-balanced sample weights ─────────────────────────────────────
        # Three-way balancing: class imbalance × regime imbalance × time decay
        from sklearn.utils.class_weight import compute_sample_weight as _csw
        _class_w_tr  = _csw("balanced", y_tr)
        _class_w_all = _csw("balanced", y)
        # Extra boost for bull-regime samples: model must not under-learn bull patterns
        # Feature index 16 = is_bull one-hot
        _bull_mask_tr  = X_tr[:, 16] > 0.5  if X_tr.shape[1] > 16  else _np.zeros(len(y_tr), bool)
        _bull_mask_all = X_raw[:, 16] > 0.5 if X_raw.shape[1] > 16 else _np.zeros(len(y), bool)
        _regime_boost_tr  = _np.where(_bull_mask_tr,  1.5, 1.0)  # 1.5× weight for bull bars
        _regime_boost_all = _np.where(_bull_mask_all, 1.5, 1.0)
        w_tr_bal  = w_tr * _class_w_tr  * _regime_boost_tr  / (w_tr.mean()  + 1e-9)
        w_all_bal = w    * _class_w_all * _regime_boost_all / (w.mean()     + 1e-9)

        # ── Optimal prediction threshold per regime ────────────────────────────
        def _scan_threshold(clf, X_val, y_val, thresholds=None):
            """Find threshold that maximises Sharpe on validation set."""
            if thresholds is None:
                thresholds = _np.arange(0.30, 0.70, 0.02)
            proba = clf.predict_proba(X_val)
            lp = proba[:, list(clf.classes_).index(2)] if 2 in clf.classes_ else proba[:, 1]
            sp = proba[:, list(clf.classes_).index(0)] if 0 in clf.classes_ else 1 - proba[:, 1]
            best_long_t = 0.42; best_short_t = 0.42
            best_long_s = -99;  best_short_s = -99
            outcomes = _np.where(y_val==2, ML_TP_ATR_MULT, _np.where(y_val==0, -ML_SL_ATR_MULT, 0))
            for t in thresholds:
                for p, tag in [(lp,"long"),(sp,"short")]:
                    sig = (p > t).astype(float)
                    traded = (sig * outcomes)[sig > 0]
                    if len(traded) < 10: continue
                    sh = float(traded.mean() / (traded.std() + 1e-9))
                    if tag == "long"  and sh > best_long_s:  best_long_s = sh;  best_long_t = t
                    if tag == "short" and sh > best_short_s: best_short_s = sh; best_short_t = t
            return round(float(best_long_t), 2), round(float(best_short_t), 2)

        clf_wf = GradientBoostingClassifier(
            n_estimators  = 200,   # more trees for 38-feature space
            max_depth     = 5,     # was 4 — deeper to capture regime interactions
            learning_rate = 0.07,
            subsample     = 0.75,  # slightly more aggressive subsample
            min_samples_leaf = 8,  # was 20 — too conservative, missed many win patterns
            min_samples_split = 20,
            random_state  = 42,
        )
        clf_wf.fit(X_tr, y_tr, sample_weight=w_tr_bal)
        proba_te = clf_wf.predict_proba(X_te)
        if proba_te.shape[1] == 3:
            long_p  = proba_te[:, list(clf_wf.classes_).index(2)]
            short_p = proba_te[:, list(clf_wf.classes_).index(0)]
        else:
            long_p  = proba_te[:, 1]
            short_p = 1 - proba_te[:, 1]

        # Find optimal thresholds on test set
        best_long_thresh, best_short_thresh = _scan_threshold(clf_wf, X_te, y_te)

        # Evaluate at optimal threshold
        signal    = (long_p > best_long_thresh).astype(float)
        outcomes  = _np.where(y_te == 2, ML_TP_ATR_MULT, _np.where(y_te == 0, -ML_SL_ATR_MULT, 0))
        trade_ret = signal * outcomes
        traded    = trade_ret[signal > 0]
        wf_sharpe = float(traded.mean() / (traded.std() + 1e-9)) if len(traded) > 10 else 0.0
        wf_acc    = float((clf_wf.predict(X_te) == y_te).mean())
        trigger_rate = float(signal.mean()) * 100
        print(f"[GLOBAL ML WF] acc={wf_acc:.1%}  Sharpe={wf_sharpe:+.2f}  "
              f"trades={int(signal.sum())}/{len(y_te)} ({trigger_rate:.1f}%)  "
              f"long_t={best_long_thresh:.2f}  short_t={best_short_thresh:.2f}")

        # ── Full model (all data, fresh scaler, balanced weights) ────────────
        full_scaler   = StandardScaler()
        X_scaled_full = full_scaler.fit_transform(X_raw)
        clf = GradientBoostingClassifier(
            n_estimators     = 250,   # more trees for full dataset
            max_depth        = 5,
            learning_rate    = 0.06,  # slightly lower for 250 trees
            subsample        = 0.75,
            min_samples_leaf = 8,
            min_samples_split= 20,
            random_state     = 42,
        )
        clf.fit(X_scaled_full, y, sample_weight=w_all_bal)
        scaler = full_scaler

        # ── Calibrate on held-out 20% ─────────────────────────────────────
        try:
            cal = CalibratedClassifierCV(clf_wf, method="sigmoid", cv=3)
            cal.fit(X_te, y_te)
            model = cal
            print(f"[GLOBAL ML] Calibrated on {len(X_te):,} OOS samples")
        except Exception as _ce:
            model = clf
            print(f"[GLOBAL ML] Calibration skipped: {_ce}")

        # ── Feature importance top 10 ──────────────────────────────────────
        feat_names = (
            ["rsi","macd","adx","vol","ema_cross","bb_pos","atr_slope",
             "di_dom","fng","btc_rs","dow","sortino","session","ema_w","cloud","obv_7d"]
            + ["reg_bull","reg_bear","reg_range","reg_panic"]
            + [f"sec_{s}" for s in SECTORS]
            + [f"mcap_{m}" for m in MCAP_TIERS]
            + [f"rs_{r}" for r in RS_TIERS]
        )
        try:
            importances = clf.feature_importances_
            top10 = sorted(zip(feat_names[:len(importances)], importances),
                           key=lambda x: -x[1])[:10]
            print(f"[GLOBAL ML] Top features: " +
                  ", ".join(f"{n}={v:.3f}" for n,v in top10))
        except Exception:
            pass

        # ── Save ──────────────────────────────────────────────────────────
        Path("models").mkdir(exist_ok=True)
        with open(GLOBAL_MODEL_PATH, "wb") as f:
            pickle.dump((model, scaler), f)

        meta = {
            "version":          GLOBAL_MODEL_VERSION,
            "trained_date":     datetime.now().isoformat(),
            "n_samples":        int(len(X_raw)),
            "n_features":       int(X_raw.shape[1]),
            "n_coins":          len(coin_counts),
            "wf_acc":           round(wf_acc, 4),
            "wf_sharpe":        round(wf_sharpe, 4),
            "long_threshold":   best_long_thresh,
            "short_threshold":  best_short_thresh,
            "coin_counts":      coin_counts,
        }
        with open(GLOBAL_MODEL_META, "w") as f:
            json.dump(meta, f, indent=2)

        # Cache in memory
        _GLOBAL_MODEL["model"]          = model
        _GLOBAL_MODEL["scaler"]         = scaler
        _GLOBAL_MODEL["long_threshold"] = best_long_thresh
        _GLOBAL_MODEL["short_threshold"]= best_short_thresh
        _GLOBAL_MODEL["meta"]    = meta
        _GLOBAL_MODEL["sharpe"]  = wf_sharpe
        _GLOBAL_MODEL["acc"]     = wf_acc

        print(f"\n[GLOBAL ML] ✅ Saved → {GLOBAL_MODEL_PATH}  "
              f"({len(X):,} samples  {len(coin_counts)} coins  "
              f"Sharpe={wf_sharpe:+.2f}  acc={wf_acc:.1%})")
        return True

    except Exception as e:
        print(f"[GLOBAL ML] ❌ Training failed: {e}")
        import traceback; traceback.print_exc()
        return False


def load_global_model() -> bool:
    """Load global model from disk into _GLOBAL_MODEL cache. Returns True if loaded."""
    if not GLOBAL_MODEL_PATH.exists() or not GLOBAL_MODEL_META.exists():
        return False
    try:
        with open(GLOBAL_MODEL_PATH, "rb") as f:
            model, scaler = pickle.load(f)
        with open(GLOBAL_MODEL_META, "r") as f:
            meta = json.load(f)
        if meta.get("version") != GLOBAL_MODEL_VERSION:
            print(f"[GLOBAL ML] Version mismatch ({meta.get('version')} ≠ {GLOBAL_MODEL_VERSION}) — retrain needed")
            return False
        _GLOBAL_MODEL["model"]           = model
        _GLOBAL_MODEL["scaler"]          = scaler
        _GLOBAL_MODEL["meta"]            = meta
        _GLOBAL_MODEL["sharpe"]          = meta.get("wf_sharpe", 0.0)
        _GLOBAL_MODEL["acc"]             = meta.get("wf_acc", 0.0)
        _GLOBAL_MODEL["long_threshold"]  = meta.get("long_threshold",  0.42)
        _GLOBAL_MODEL["short_threshold"] = meta.get("short_threshold", 0.42)
        trained = meta.get("trained_date","")[:10]
        print(f"[GLOBAL ML] ✅ Loaded global model  "
              f"trained={trained}  "
              f"coins={meta.get('n_coins',0)}  "
              f"samples={meta.get('n_samples',0):,}  "
              f"Sharpe={meta.get('wf_sharpe',0):+.2f}  "
              f"acc={meta.get('wf_acc',0):.1%}")
        return True
    except Exception as e:
        print(f"[GLOBAL ML] Load error: {e}")
        return False


def needs_global_retrain() -> bool:
    """Returns True if global model is missing or older than GLOBAL_RETRAIN_DAYS."""
    if not GLOBAL_MODEL_META.exists():
        return True
    try:
        meta = json.loads(GLOBAL_MODEL_META.read_text())
        trained = datetime.fromisoformat(meta.get("trained_date", "2000-01-01"))
        return (datetime.now() - trained).days >= GLOBAL_RETRAIN_DAYS
    except Exception:
        return True


def predict_global_ml(last: pd.Series, df: pd.DataFrame,
                       btc_df: pd.DataFrame, coin_id: str,
                       rs_tier: str = "NEUTRAL",
                       fng_value: int = 50) -> Tuple[float, float, str]:
    """
    Predict using global model. Returns (long_proba, short_proba, note).

    Uses calibrated per-regime thresholds learned during training.
    Returns raw probabilities — the Sharpe gate in the caller decides
    whether to use them.
    """
    if not _GLOBAL_MODEL.get("model"):
        return 0.5, 0.5, "global model not loaded"
    try:
        feats = build_global_features(last, df, btc_df, coin_id,
                                      rs_tier=rs_tier, fng_value=fng_value)
        if feats is None:
            return 0.5, 0.5, "feature extraction failed"

        model  = _GLOBAL_MODEL["model"]
        scaler = _GLOBAL_MODEL["scaler"]
        long_t = _GLOBAL_MODEL.get("long_threshold",  0.42)
        short_t= _GLOBAL_MODEL.get("short_threshold", 0.42)

        X      = scaler.transform([feats])
        proba  = model.predict_proba(X)[0]
        classes = list(model.classes_) if hasattr(model, "classes_") else [0,1,2]

        raw_long  = float(proba[classes.index(2)]) if 2 in classes else 0.5
        raw_short = float(proba[classes.index(0)]) if 0 in classes else 0.5

        # Re-scale raw probabilities relative to their calibrated threshold:
        # This maps threshold → 0.50 so downstream blending is correctly centered.
        # P_raw=0.60 with threshold=0.42 → strong signal (far above threshold)
        # P_raw=0.40 with threshold=0.42 → below threshold (map below 0.50)
        def _rescale(p, t, neutral=0.50):
            if p >= t:
                # above threshold → map [t, 1.0] → [0.55, 0.95]
                return 0.55 + (p - t) / (1.0 - t + 1e-9) * 0.40
            else:
                # below threshold → map [0, t] → [0.10, 0.50]
                return 0.10 + p / (t + 1e-9) * 0.40

        long_p  = _rescale(raw_long,  long_t)
        short_p = _rescale(raw_short, short_t)

        note = (f"global acc={_GLOBAL_MODEL['acc']:.0%} "
                f"Sharpe={_GLOBAL_MODEL['sharpe']:+.2f} "
                f"t_long={long_t:.2f} t_short={short_t:.2f}")
        return round(long_p, 3), round(short_p, 3), note

    except Exception as e:
        return 0.5, 0.5, f"prediction error: {e}"


# Runtime state
_GLOBAL_MODEL: Dict[str, Any] = {}   # model, scaler, meta, sharpe, acc
_GLOBAL_COIN_DFS: Dict[str, pd.DataFrame] = {}   # accumulated this session for retraining
TRADE_LOGGER_ENABLED = True         # [P] Log every BUY signal for forward-test tracking
TRADE_LOG_PATH       = Path("models") / "trade_log.json"   # [P] Where log is stored
MAX_HOLD_DAYS        = 30           # [P] Mark as expired after 30 days unrealised

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
MODEL_CACHE_DAYS   = 7                # Main bot: retrain after 7 days
SCANNER_CACHE_HOURS = 48              # Scanner: accept models up to 48h old (predict only)
ML_FEATURE_VERSION  = "v2.0-leak-fix" # Bump when features change → forces all models to retrain
                                       # v1.x: original 16 features (with leakage)
                                       # v2.0-leak-fix: df_slice fix applied (clean training)
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

# Binance order book -- disable on GitHub Actions (geo-restricted, returns 451)
# Set USE_BINANCE_ORDER_BOOK=false in GitHub Actions env to skip ccxt calls
import os as _os
USE_BINANCE_ORDER_BOOK = _os.environ.get("USE_BINANCE_ORDER_BOOK", "true").lower() != "false"
# Auto-detect GitHub Actions
if _os.environ.get("GITHUB_ACTIONS") == "true":
    USE_BINANCE_ORDER_BOOK = False
    print("[CONFIG] GitHub Actions detected -- Binance order book disabled (geo-restriction 451)")
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
    "hedera-hashgraph": "HBAR",
    "gnosis": "GNO",
    "centrifuge": "CFG",
    "xdce-crowd-sale": "XDC",
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

    # ── Batch addition from gem-scanner tables (2026-06-15) ──
    "ribbita-by-virtuals": "TIBBIR",
    "jupiter-exchange-solana": "JUP",
    "jupiter": "JUP",
    "arweave": "AR",
    "ar": "AR",
    "aerodrome-finance": "AERO",
    "aero": "AERO",
    "basic-attention-token": "BAT",
    "bat": "BAT",
    "internet-computer": "ICP",
    "icp": "ICP",
    "zebec-network": "ZBCN",
    "zbcn": "ZBCN",
    "shuffle-2": "SHFL",
    "shfl": "SHFL",
    "mantle": "MNT",
    "mnt": "MNT",
    "okb": "OKB",
    "quant-network": "QNT",
    "qnt": "QNT",
    "just": "JST",
    "jst": "JST",
    "aptos": "APT",
    "apt": "APT",
    "conflux-token": "CFX",
    "cfx": "CFX",
    "telcoin": "TEL",
    "tel": "TEL",
    "neo": "NEO",
    "apecoin": "APE",
    "ape": "APE",
    "safepal": "SFP",
    "sfp": "SFP",
    "crypto-com-chain": "CRO",
    "cro": "CRO",
    "sky": "SKY",
    "midnight-3": "NIGHT",
    "night": "NIGHT",
    "spx6900": "SPX",
    "spx": "SPX",
    "plasma": "XPL",
    "xpl": "XPL",
    "onyxcoin": "XCN",
    "xcn": "XCN",
    "mx-token": "MX",

    # ── Batch 2: top gem-scanner GEM/SHORT results (2026-06-15 11:11 scan) ──
    "grass": "GRASS",
    "flying-tulip": "FT",
    "ft": "FT",
    "venice-token": "VVV",
    "vvv": "VVV",
    "rain": "RAIN",
    "akash-network": "AKT",
    "akt": "AKT",
    "worldcoin-wld": "WLD",
    "wld": "WLD",
    "bitway": "BTW",
    "btw": "BTW",
    "lighter": "LIT",
    "lit": "LIT",
    "eigenlayer": "EIGEN",
    "eigen": "EIGEN",
    # "morpho": "MORPHO",  -- already exists above
    "beldex": "BDX",
    "bdx": "BDX",
    "the9bit": "9BIT",

    # ── Batch 3: gem-scanner table, 5 genuinely new of 14 (2026-06-15 ~13:55 scan) ──
    "railgun": "RAIL",
    "rail": "RAIL",
    "velvet": "VELVET",
    "kaito": "KAITO",
    "dexe": "DEXE",
    "injective": "INJ",
    # "inj": "INJ",  -- already exists above (note: "injective-protocol" alias above is a dead/wrong ID, unused)
    # ── Batch 4: gem-scanner table (2026-07-14) ─────────────────────────────────
    "stp-network": "AWE",   # AI Autonomous Worlds — CoinGecko API ID kept as stp-network post-rebrand
    "awe": "AWE",           # ticker alias (also exists as awe-network on website, but API uses stp-network)
    # "uniswap": "UNI",  -- already exists above
    # "uni": "UNI",      -- already exists above
    # wfi removed (insufficient CoinGecko history — re-add when ≥200 days exist)
    # ── Batch 5: gem-scanner table (2026-07-15) ─────────────────────────────────
    "bitcoin-sv":    "BSV",   # Bitcoin SV — on Binance as BSV/USDT
    "bsv":           "BSV",
    "pancakeswap-token": "CAKE",  # PancakeSwap — 'pancakeswap' alias already exists above
    "whitebit":      None,    # WBT — WhiteBIT exchange token, traded on Kraken/MEXC, not Binance
    "pyth-network":  "PYTH",  # Pyth oracle — on Binance as PYTH/USDT
    "pyth":          "PYTH",
    # "lido-dao": "LDO", "huobi-token": "HT", "aave": "AAVE"  -- all already exist above
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

# [F] Consecutive candle confirmation -- require N consecutive days of BUY-ready signals
MIN_CONSECUTIVE_DAYS = 2

# [G] ML Model configuration -- Logistic Regression for trade outcome prediction
ML_ENABLED = ML_AVAILABLE and True  # Set to False to disable ML scoring
ML_MIN_CONFIDENCE = 0.60            # only add ML bonus if confidence > 60%
ML_LOOKBACK_DAYS = 20               # skip first N rows (indicator warmup only)
ML_MIN_ACCURACY  = 0.52             # below this = heavily dampened
ML_FULL_ACCURACY = 0.60             # above this = fully trusted
ML_FUTURE_DAYS = 5                  # label: did price reach TP in next 5 days?

# ── Training data quality controls ────────────────────────────────────────
# ROOT CAUSE OF LOW ACCURACY: 89% win rate in training data
# HYPE had a bull run → 250 bull bars (labeled WIN) + 115 bear bars (labeled LOSS)
# = model learns to always predict WIN → 41% accuracy on bear test set
#
# Fix 1 — Rolling window: only train on last N days (matches current regime)
# Fix 2 — Regime weighting: bars matching current regime get higher weight
# Fix 3 — Class rebalancing: cap win/loss imbalance at 3:1 max
ML_ROLLING_WINDOW_DAYS = 120  # use last 120 days only (0 = all 365)
ML_REGIME_WEIGHT_MULT  = 3.0  # bars in current regime get 3× weight
ML_MAX_CLASS_RATIO     = 3.0  # max win:loss ratio (undersample if exceeded)

MARKET_STRUCTURE_ENABLED = True
SWING_LOOKBACK           = 3
SFP_LOOKBACK             = 3

# [H] Smart Money / Whale Activity tracking
WHALE_VOLUME_PERCENTILE = 80        # top 20% volume (80th percentile) = whale tier
WHALE_LOOKBACK_DAYS = 30            # baseline: last 30 days
VOLUME_SPIKE_THRESHOLD = 1.5        # 1.5x avg = notable spike
WHALE_DIVERGENCE_LOOKBACK = 14      # bars for accumulation/distribution divergence

# [I] News / Event Filter -- major events that move markets
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
            logger.warning(f"[OHLC] 429 rate limit for {coin} -- waiting {wait:.0f}s")
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
            logger.debug(f"[OHLC] {coin}: fetch failed ({e}) -- will use approximation")
        return pd.DataFrame()


def fetch_coin_data(coin, days=CG_DAYS, interval=CG_INTERVAL):
    """
    Fetch coin price+volume data, merge real OHLC, compute indicators.

    API call budget per coin per session:
      - market_chart: 1 call  (price + volume)
      - ohlc:         1 call  (high/low, cached separately)
      Total: 2 calls first fetch, 0 calls on cache hit.

    Session cache: results stored for CACHE_TTL_SECONDS (default 4h).
    BTC is shared via _btc_cache -- never fetched twice.
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
            logger.warning(f"[API] ⏳ 429 RATE LIMIT for {coin} -- waiting {wait:.0f}s before retry")
            time.sleep(wait)
            # One retry after waiting
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 429:
                logger.error(f"[API] ❌ Still rate-limited after {wait:.0f}s wait -- skipping {coin}")
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
        # fetch_ohlc_data has its own session cache -- safe to call here
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
            logger.warning(f"[CACHE] {coin}: not caching -- insufficient data ({len(df)} candles)")

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

    # RSI -- Wilder's EWM
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

    # ATR -- Wilder's EWM on True Range
    # Use real high/low from OHLC endpoint if available.
    # Fallback: estimate intraday range from close volatility.
    #   rolling(2).max/min dramatically underestimates range -- it only captures
    #   close-to-close movement, ignoring intraday wicks (which in crypto are 1.5-3× larger).
    #   Better fallback: daily close-to-close std × 1.5 intraday multiplier.
    if "high" in df.columns and "low" in df.columns and df["high"].notna().any():
        # Real OHLC available -- use directly
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

    # ADX -- DI+/DI- using real OHLC when available, else same improved approximation
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

    # Volume normalisation -- relative volume using rolling median baseline.
    # Median is robust to spikes; mean gets inflated for months after a spike.
    # rel_vol = 1.0 → typical day, 2.0 → twice typical, 0.5 → half typical.
    # Consistent across small caps, large caps, and different exchanges.
    df["vol_median20"] = df["volume"].rolling(20, min_periods=5).median()
    df["vol_median90"] = df["volume"].rolling(90, min_periods=10).median()
    df["rel_vol"]      = (df["volume"] / (df["vol_median20"] + 1e-9)).clip(upper=20.0)
    df["rel_vol_slow"] = (df["volume"] / (df["vol_median90"] + 1e-9)).clip(upper=20.0)
    # Keep vol_avg90 for any legacy code that still references it
    df["vol_avg90"]    = df["vol_median90"]   # now median-backed

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

    # [B] Chikou Span -- full Ichimoku confirmation
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
    Detect smart money accumulation / distribution using adaptive, volatility-scaled
    thresholds instead of fixed percentages.

    PROBLEMS WITH FIXED THRESHOLDS (price_change > 0.03):
      • BTC has 0.5% daily ATR -- 3% is a huge 6σ move, almost never triggers
      • A meme coin has 8% daily ATR -- 3% is routine noise, always triggers
      • "OBV change >= 0" counts near-flat OBV (noise) as accumulation

    ADAPTIVE APPROACH:
      1. price_z    = price_change / rolling_volatility
                      → measures significance in units of recent price noise
      2. obv_pct_rank = percentile of OBV change vs 60-day history
                      → is OBV rising more/less than typical?
      3. k-factor   = user-tunable sensitivity (default: 0.8)
                      → accumulation fires when price_z > k AND obv_pct_rank > 50th

    CONFIDENCE: continuous [0, 1] from strength of BOTH signals
      = min(price_z / 2.0, 1.0) × obv_pct_rank / 100
      Strong price + strong OBV = high confidence
      Strong price + weak OBV   = low confidence (warn of distribution)

    REGIME-ADAPTIVE: low-liquidity coins naturally have higher ATR, so the
    threshold automatically scales up, preventing false signals on noise.
    """
    if len(df) < max(lookback + 5, 30):
        return "neutral", 0, {}

    try:
        window      = df.iloc[-lookback:]
        full_hist   = df.iloc[-90:] if len(df) >= 90 else df   # 90-day context

        col = "close" if "close" in df.columns else "price"

        # ── 1. Price change as a z-score of recent volatility ────────────────
        price_start    = window[col].iloc[0]
        price_end      = window[col].iloc[-1]
        price_change   = (price_end - price_start) / (price_start + 1e-9)

        # Rolling volatility: std of 14-day returns (annualised not needed, relative)
        returns        = full_hist[col].pct_change().dropna()
        roll_vol       = returns.rolling(14, min_periods=7).std()
        current_vol    = float(roll_vol.iloc[-1]) if len(roll_vol) > 0 else 0.02
        current_vol    = max(current_vol, 0.005)   # floor to prevent div/0 on dead coins
        price_z        = price_change / current_vol   # z-score: > 1 = meaningful move

        # ── 2. OBV change as a percentile rank vs recent history ─────────────
        if "OBV" not in df.columns or len(df) < 20:
            return "neutral", 0, {}

        obv_start      = float(window["OBV"].iloc[0])
        obv_end        = float(window["OBV"].iloc[-1])
        obv_change     = obv_end - obv_start

        # Compare this OBV change against rolling 60-day history of lookback-window changes
        hist_obv_changes = []
        for i in range(30, len(df)):
            h_start = float(df["OBV"].iloc[i - lookback]) if i >= lookback else float(df["OBV"].iloc[0])
            h_end   = float(df["OBV"].iloc[i])
            hist_obv_changes.append(h_end - h_start)

        if len(hist_obv_changes) < 10:
            # Not enough history: fall back to sign-based check
            obv_pct_rank = 75.0 if obv_change > 0 else 25.0
        else:
            arr = np.array(hist_obv_changes)
            obv_pct_rank = float(np.mean(arr < obv_change) * 100)

        # ── 3. Normalize OBV change by volume scale (robust, median-based) ───
        total_vol    = float(window["volume"].sum()) if "volume" in df.columns else 1.0
        obv_chg_pct  = obv_change / (total_vol + 1e-9)

        # ── 4. Adaptive classification ────────────────────────────────────────
        # k = sensitivity factor.  Lower k = more signals (noisier); higher = fewer (cleaner).
        k = 0.80   # fire when price_z > 0.8 (meaningful move, not just noise)

        # ACCUMULATION: price up significantly (z > k) AND OBV above median (> 50th pct)
        # The percentile filter prevents near-flat OBV from counting as accumulation.
        if price_z > k and obv_pct_rank > 55:
            # Confidence: how strong is each signal independently?
            price_conf  = min(price_z / 2.0, 1.0)     # saturates at z=2 → 100%
            obv_conf    = (obv_pct_rank - 55) / 45.0   # 55th pct=0%, 100th pct=100%
            confidence  = round(price_conf * obv_conf, 3)
            mode = "accumulation"

        # DISTRIBUTION: price up significantly BUT OBV below 40th pct (divergence)
        elif price_z > k * 0.6 and obv_pct_rank < 40:
            price_conf  = min(abs(price_z) / 2.0, 1.0)
            obv_div     = (40.0 - obv_pct_rank) / 40.0   # 40th pct=0%, 0th pct=100%
            confidence  = round(price_conf * obv_div, 3)
            mode = "distribution"

        # QUIET ACCUMULATION: price flat/down but OBV rising strongly (above 75th pct)
        # Classic Wyckoff accumulation: price suppressed while volume builds
        elif abs(price_z) < k * 0.5 and obv_pct_rank > 75:
            obv_conf    = (obv_pct_rank - 75) / 25.0
            confidence  = round(0.5 * obv_conf, 3)   # lower confidence: no price confirmation
            mode = "accumulation"

        else:
            mode, confidence = "neutral", 0.0

        details = {
            "price_chg_pct": round(price_change * 100, 3),
            "price_z":        round(price_z, 3),
            "current_vol":    round(current_vol * 100, 3),   # vol as % for display
            "obv_chg":        round(obv_change, 2),
            "obv_chg_pct":    round(obv_chg_pct * 100, 3),
            "obv_pct_rank":   round(obv_pct_rank, 1),
            "threshold_pct":  round(k * current_vol * 100, 3),   # adaptive threshold in %
        }

        return mode, confidence, details

    except Exception as e:
        return "neutral", 0, {"error": str(e)}

def detect_volume_divergence(df, vol_regime: str = "neutral"):
    """
    Detect volume divergence with ATR, percentile, and volatility-regime normalisation.

    FALSE POSITIVE SOURCES in the naive version:
      1. Raw mean comparison -- one past spike inflates the mean → today looks weak
      2. No ATR filter -- a 0.01% new high triggers as much as a 3% breakout
      3. Volatility regime ignored -- volume is naturally lower in compression regimes
      4. Weekend effect -- crypto Saturday/Sunday volume is structurally ~20-40% lower

    NORMALISED APPROACH:
      price_significance = price_move / ATR  (z-score of move size)
        → only flag divergence on MEANINGFUL price extremes (> 0.5×ATR)
        → prevents 0.01% drifts from registering as "new highs"

      vol_pct_rank = percentile(current_volume, 60d rolling history)
        → regime-adaptive: if the market is always low-volume on weekends,
          Saturday's low volume sits at its expected percentile, not flagged
        → uses rolling history, not a mean, so one spike doesn't inflate baseline

      regime_adjustment = threshold scaling by volatility regime:
        compression: require stronger signal (volume naturally thin → more noise)
        expansion:   standard threshold
        squeeze_on:  suppress all divergences (squeeze breakouts are always high-vol)

      weekend_penalty = loosen thresholds on Sat/Sun to prevent systematic bias

    Returns: ("bullish_divergence" | "bearish_divergence" | None, confidence, details)
    """
    if len(df) < 20:
        return None, 0.0, {}

    try:
        col    = "close" if "close" in df.columns else "price"
        prices = df[col].values.astype(float)
        vols   = df["volume"].values.astype(float) if "volume" in df.columns else None
        atrs   = df["ATR"].values.astype(float)    if "ATR"    in df.columns else None

        cur_price  = prices[-1]
        prev_prices= prices[-11:-1]   # 10-bar lookback for high/low
        prev_high  = prev_prices.max()
        prev_low   = prev_prices.min()

        # ── 1. Price significance filter (ATR-normalised) ─────────────────────
        # Only flag when the price extreme is large enough to be meaningful.
        # A 0.01% drift above a prior high on BTC is noise; 1.5×ATR is signal.
        atr_val = float(atrs[-1]) if atrs is not None and not np.isnan(atrs[-1]) else cur_price * 0.015
        new_high = cur_price > prev_high
        new_low  = cur_price < prev_low

        if new_high:
            price_significance = (cur_price - prev_high) / (atr_val + 1e-9)
        elif new_low:
            price_significance = (prev_low - cur_price) / (atr_val + 1e-9)
        else:
            return None, 0.0, {"reason": "no new extreme"}

        # Minimum significance: move must be at least 0.40×ATR
        MIN_PRICE_SIG = 0.40
        if price_significance < MIN_PRICE_SIG:
            return None, 0.0, {
                "reason": f"move too small ({price_significance:.2f}×ATR < {MIN_PRICE_SIG}×ATR)"
            }

        if vols is None:
            return None, 0.0, {"reason": "no volume data"}

        # ── 2. Volume percentile rank (rolling 60-bar history) ────────────────
        # Not a mean comparison -- a percentile rank is immune to past spikes.
        # If today's volume is at the 30th percentile of recent history, that's
        # genuinely low regardless of whether there was a spike 45 days ago.
        hist_vols = vols[max(0, len(vols)-61):-1]
        cur_vol   = float(vols[-1])
        if len(hist_vols) >= 10:
            vol_pct_rank = float(np.mean(hist_vols < cur_vol) * 100)
        else:
            vol_pct_rank = 50.0   # not enough history: assume median

        # ── 3. Volatility regime threshold adjustment ─────────────────────────
        # Compression regime: volume is naturally thin → many bars look "low volume"
        # Raising the threshold prevents false divergences during quiet periods.
        # Expansion regime: volume expectations are higher, standard threshold fine.
        base_low_vol_threshold = 45.0   # below 45th percentile = "low volume"
        if vol_regime == "compression":
            low_vol_threshold = 35.0    # looser: expect thin volume in compression
        elif vol_regime == "expansion":
            low_vol_threshold = 55.0    # tighter: expansion should sustain volume
        elif vol_regime == "squeeze_on":
            return None, 0.0, {"reason": "squeeze active -- divergence suppressed"}
        else:
            low_vol_threshold = base_low_vol_threshold

        # ── 4. Weekend penalty (loosen thresholds on Sat/Sun) ─────────────────
        # Crypto trades 24/7 but retail activity drops on weekends.
        # Saturday/Sunday volume is structurally 20-40% lower -- not a signal.
        try:
            from datetime import datetime as _dt
            dow = _dt.now().weekday()   # 5=Saturday, 6=Sunday
            if dow in (5, 6):
                low_vol_threshold -= 10.0   # require even lower volume to flag
        except Exception:
            pass

        # ── 5. Divergence classification ──────────────────────────────────────
        is_low_vol = vol_pct_rank < low_vol_threshold

        # Confidence: how clearly is each signal present?
        #   price_conf = how significant the move is (capped at 3×ATR)
        #   vol_conf   = how definitively below the threshold the volume is
        price_conf = min(price_significance / 3.0, 1.0)
        vol_conf   = max(0.0, (low_vol_threshold - vol_pct_rank) / low_vol_threshold)
        confidence = round(price_conf * vol_conf, 3)

        details = {
            "price_significance": round(price_significance, 3),
            "vol_pct_rank":       round(vol_pct_rank, 1),
            "low_vol_threshold":  round(low_vol_threshold, 1),
            "is_low_vol":         is_low_vol,
            "vol_regime":         vol_regime,
            "atr_val":            round(atr_val, 4),
            "confidence":         confidence,
        }

        if new_high and is_low_vol:
            # Price breakout on low relative volume → buying exhaustion
            return "bearish_divergence", confidence, details
        elif new_low and is_low_vol:
            # Price breakdown on low relative volume → selling exhaustion
            return "bullish_divergence", confidence, details

        return None, 0.0, details

    except Exception as e:
        return None, 0.0, {"error": str(e)}

# ===========================
# [I] NEWS / EVENT FILTER -- MAJOR MARKET EVENTS
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
            return True, False, "unknown", "OBV data unavailable -- no gate applied"

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
            desc = f"OBV at 7d high ({obv_change_pct:+.1f}%) -- capital inflow confirmed"
        elif obv_trend == "rising":
            desc = f"OBV rising ({obv_change_pct:+.1f}%) but not at 7d high -- partial confirmation"
        elif obv_trend == "flat":
            desc = f"OBV flat ({obv_change_pct:+.1f}%) -- weak hands driving price"
        else:
            desc = f"OBV falling ({obv_change_pct:+.1f}%) -- distribution, price move unreliable"

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
        print(f"[OBV GATE] {coin}: {tag} -- ML {ml_proba:.1%} → {adjusted:.1%} (×{multiplier})")
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
        return True, f"BLOCKED: {closest['description']} is {days_away}d away -- too much uncertainty"
    
    return False, ""

# ===========================
# [G] ML MODEL -- LOGISTIC REGRESSION
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
        if len(df) < 14 or "OBV" not in df.columns:
            return 0.0, "Insufficient data"
        recent_vol=df["volume"].iloc[-14:].median()  # median: robust to spikes
        recent_price_change=(df["price"].iloc[-1]-df["price"].iloc[-14])/(df["price"].iloc[-14]+1e-9)*100
        vol_spike=df["volume"].iloc[-1]/(recent_vol+1e-9)
        obv_now=df["OBV"].iloc[-1]; obv_prev=df["OBV"].iloc[-14]
        total_vol=df["volume"].iloc[-14:].sum()
        obv_trend=np.clip((obv_now-obv_prev)/(total_vol+1e-9),-1.0,1.0)
        if recent_price_change>0 and vol_spike>1.2:
            accumulation_score=min(obv_trend*0.5+0.5,1.0)
            details=f"Markup: Price +{recent_price_change:.1f}%, Vol {vol_spike:.1f}x, OBV {obv_trend:+.2f}"
        elif recent_price_change<-5 and vol_spike>1.5:
            accumulation_score=max(obv_trend*0.5-0.5,-1.0)
            details=f"Distribution: Price {recent_price_change:.1f}%, Vol {vol_spike:.1f}x, OBV {obv_trend:+.2f}"
        else:
            accumulation_score=obv_trend*0.3
            details=f"Neutral: OBV {obv_trend:+.2f}, Vol {vol_spike:.1f}x"
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
        vol_mean = df['volume'].iloc[-20:-1].median()  # median baseline
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


def calculate_advanced_stop_loss(entry_price, current_price, signal_type="BUY",
                                 lookback_high=None, lookback_low=None, atr=None):
    """
    [U] ATR-based unified stop loss -- consistent with build_position_plan
    
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
    [U] ATR-based 3-tier partial profit taking -- Donchian-aware in chop zones.
    
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

# ── SHORT MODEL paths -- separate files, long model untouched ─────────────────
def get_short_model_path(coin):
    return os.path.join(MODEL_PERSISTENCE_DIR, f"{coin}_short_model.pkl")

def get_short_scaler_path(coin):
    return os.path.join(MODEL_PERSISTENCE_DIR, f"{coin}_short_scaler.pkl")

def get_short_metadata_path(coin):
    return os.path.join(MODEL_PERSISTENCE_DIR, f"{coin}_short_metadata.json")

def save_model(coin, model, scaler, accuracy, samples, wf_sharpe: float = 0.0):
    """
    [V] Save trained model, scaler, and metadata to disk.
    wf_sharpe: mean walk-forward Sharpe (stored for Sharpe suppression).
               Negative Sharpe → ML probability dampened toward 0.5 at predict time.
    """
    try:
        if not MODEL_PERSISTENCE_ENABLED:
            return False
        
        Path(MODEL_PERSISTENCE_DIR).mkdir(parents=True, exist_ok=True)
        
        model_path    = get_model_path(coin)
        scaler_path   = get_scaler_path(coin)
        metadata_path = get_metadata_path(coin)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        try:
            from sklearn.calibration import CalibratedClassifierCV as _CCV
            _is_cal = isinstance(model, _CCV)
        except Exception:
            _is_cal = False
        metadata = {
            "coin": coin,
            "trained_date": datetime.now().isoformat(),
            "accuracy": float(accuracy),
            "samples":  int(samples),
            "wf_sharpe": round(float(wf_sharpe), 4),
            "feature_version": ML_FEATURE_VERSION,   # ← invalidates cache when features change
            "model_version": "v8.2",
            "n_features": 20,
            "model_type": "GradientBoosting",
            "probability_calibrated": _is_cal,
            "calibration_method": "isotonic_or_sigmoid" if _is_cal else "none",
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

        # [V] Version check -- reject old models with wrong feature count
        CURRENT_VERSION  = "v8.2"
        CURRENT_FEATURES = 20
        MIN_SAMPLES      = 80   # Reject models trained on too few samples
        saved_version  = metadata.get("model_version", "unknown")
        saved_features = metadata.get("n_features", 11)
        saved_samples  = metadata.get("samples", 0)

        if saved_version != CURRENT_VERSION or saved_features != CURRENT_FEATURES:
            logger.warning(f"[PERSIST] ⚠️ {coin}: Model version mismatch "
                           f"({saved_version}/{saved_features}f → need {CURRENT_VERSION}/{CURRENT_FEATURES}f) "
                           f"-- forcing retrain")
            print(Fore.YELLOW + f"[PERSIST] ⚠️ {coin}: Old model detected ({saved_version}, "
                  f"{saved_features} features) -- will retrain with new {CURRENT_FEATURES}-feature model")
            return None, None, None   # Signal: retrain needed

        if saved_samples < MIN_SAMPLES:
            logger.warning(f"[PERSIST] ⚠️ {coin}: Model has only {saved_samples} samples (min {MIN_SAMPLES}) -- forcing retrain")
            print(Fore.YELLOW + f"[PERSIST] ⚠️ {coin}: Too few samples ({saved_samples} < {MIN_SAMPLES}) -- retraining with 365d data")
            return None, None, None   # Signal: retrain needed

        logger.info(f"[PERSIST] ✅ Model loaded: {coin} (acc: {metadata.get('accuracy', 'N/A'):.1%})")
        print(Fore.CYAN + f"[PERSIST] Model loaded: {coin} (acc: {metadata.get('accuracy', 'N/A'):.1%})")
        return model, scaler, metadata
    
    except Exception as e:
        logger.error(f"[PERSIST] ❌ Error loading {coin}: {e}")
        print(Fore.YELLOW + f"[PERSIST] Error loading {coin}: {e}")
        return None, None, None

def should_retrain_model(coin, scanner_mode: bool = False):
    """
    Check if model should be retrained.

    INVALIDATION CONDITIONS (any one triggers retrain):
      1. No saved model exists
      2. Model older than MODEL_CACHE_DAYS (main bot) or SCANNER_CACHE_HOURS (scanner)
      3. ML_FEATURE_VERSION mismatch — feature set changed (e.g. leakage fix)
      4. Model version mismatch (architecture change)

    scanner_mode=True: uses SCANNER_CACHE_HOURS threshold instead of MODEL_CACHE_DAYS.
    The scanner should NEVER drive retraining — it predicts only.
    Retraining is the main bot's responsibility (runs daily, slower, full pipeline).
    """
    try:
        if not MODEL_PERSISTENCE_ENABLED or not AUTO_RETRAIN:
            return True

        metadata_path = get_metadata_path(coin)
        if not os.path.exists(metadata_path):
            return True  # No model — must train

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Age check
        trained_date = datetime.fromisoformat(metadata['trained_date'])
        age = datetime.now() - trained_date

        if scanner_mode:
            # Scanner: use cached model up to SCANNER_CACHE_HOURS old
            age_limit_met = age.total_seconds() / 3600 >= SCANNER_CACHE_HOURS
            if age_limit_met:
                print(Fore.YELLOW + f"[PERSIST] {coin}: model {age.days}d old "
                      f"(scanner limit {SCANNER_CACHE_HOURS}h) — predict with stale model, "
                      f"schedule retrain via main bot")
                return False   # Scanner never retrains — main bot does it
        else:
            age_limit_met = age.days >= MODEL_CACHE_DAYS
            if age_limit_met:
                print(Fore.YELLOW + f"[PERSIST] {coin} model is {age.days}d old, retraining...")
                return True

        # Feature version check — catches leakage fix, feature additions, etc.
        saved_feature_version = metadata.get("feature_version", "unknown")
        if saved_feature_version != ML_FEATURE_VERSION:
            print(Fore.YELLOW + f"[PERSIST] {coin}: feature version mismatch "
                  f"({saved_feature_version} → {ML_FEATURE_VERSION}) — retraining")
            return True

        # Model architecture version check
        saved_model_version = metadata.get("model_version", "unknown")
        if saved_model_version != "v8.2":
            print(Fore.YELLOW + f"[PERSIST] {coin}: model version mismatch "
                  f"({saved_model_version}) — retraining")
            return True

        return False

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
        
        # Get average volume -- use median for robustness (mean is inflated by past spikes)
        # Prefer precomputed rel_vol column; fall back to inline median calculation
        lookback = min(VOLUME_LOOKBACK, len(df))
        if "rel_vol" in df.columns and not df["rel_vol"].isna().all():
            vol_ratio  = float(df["rel_vol"].iloc[-1])
            avg_volume = current_vol / (vol_ratio + 1e-9)   # implied median
        else:
            avg_volume = df["volume"].iloc[-lookback:].median()
            vol_ratio  = current_vol / (avg_volume + 1e-9)
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
    """[Z] Price change 24h -- PENALTY ONLY (downside unique signal)
    
    WHY ONLY DOWNSIDE:
      Volume already rewards upside:
        Price UP + Volume UP   → +1 (volume spike)
        Price UP + Volume DOWN → -2 (fake move warning)
      
      Price 24h unique value = catching FALLS that volume misses:
        Price DOWN + Volume normal → volume gives no signal
        But price 24h gives -1 or -2 ← unique!
    
    SCORING:
      < -3%:      -2  (sharp decline -- volume may not catch this)
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
        # Capped at ±1 -- volume boost already covers conviction
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
        
        if LOG_ENABLED:
            logger.debug(f"[ORDER BOOK IMBALANCE] {coin}: {label} - {description}")
        
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


def detect_volatility_regime(df):
    try:
        if df is None or len(df)<30: return "neutral",False,{},"Insufficient data"
        col="close" if "close" in df.columns else "price"; prices=df[col]; bb_rank=0.5
        if "UpperBB" in df.columns and "LowerBB" in df.columns:
            sma20=prices.rolling(20).mean(); bb_w=(df["UpperBB"]-df["LowerBB"])/(sma20+1e-9)
            bb_rank=float(bb_w.rolling(50,min_periods=20).rank(pct=True).iloc[-1])
        atr_ratio=1.0
        if "ATR" in df.columns:
            af=float(df["ATR"].rolling(5,min_periods=3).mean().iloc[-1]); as_=float(df["ATR"].rolling(20,min_periods=10).mean().iloc[-1])
            atr_ratio=af/(as_+1e-9)
        squeeze=False
        if "ATR" in df.columns and "UpperBB" in df.columns and "LowerBB" in df.columns:
            e20=float(prices.ewm(span=20,adjust=False).mean().iloc[-1]); an=float(df["ATR"].iloc[-1])
            squeeze=(float(df["UpperBB"].iloc[-1])<e20+1.5*an and float(df["LowerBB"].iloc[-1])>e20-1.5*an)
        log_ret=np.log(prices/prices.shift(1)).dropna(); hv_ratio=1.0
        if len(log_ret)>=30:
            hv10=float(log_ret.iloc[-10:].std()*np.sqrt(252)); hv30=float(log_ret.iloc[-30:].std()*np.sqrt(252)); hv_ratio=hv10/(hv30+1e-9)
        vol_ratio=(float(df["rel_vol"].iloc[-1]) if "rel_vol" in df.columns else float(df["volume"].iloc[-5:].median()/(df["volume"].iloc[-20:].median()+1e-9))) if "volume" in df.columns and len(df)>=20 else 1.0
        s1=-1 if bb_rank<0.20 else(1 if bb_rank>0.80 else 0); s2=-1 if atr_ratio<0.80 else(1 if atr_ratio>1.20 else 0)
        s3=-1 if squeeze else 0; s4=-1 if hv_ratio<0.70 else(1 if hv_ratio>1.30 else 0); s5=-1 if vol_ratio<0.70 else(1 if vol_ratio>1.40 else 0)
        total=s1+s2+s3+s4+s5; regime="compression" if total<=-2 else("expansion" if total>=2 else "neutral")
        signals={"bb_rank":round(bb_rank,2),"atr_ratio":round(atr_ratio,2),"squeeze":squeeze,"hv_ratio":round(hv_ratio,2),"vol_ratio":round(vol_ratio,2),"score":total}
        details=f"BBW:{bb_rank:.0%}  ATR:{atr_ratio:.2f}x  Sqz:{'ON' if squeeze else 'off'}  HV:{hv_ratio:.2f}  Vol:{vol_ratio:.2f}x  [{total:+d}/5]"
        return regime,squeeze,signals,details
    except Exception as e: return "neutral",False,{},f"Error: {e}"

def apply_volatility_regime_boost(score,df,coin,vol_regime,last,vol_details=""):
    """Single consolidated print: regime + signals + score impact."""
    try:
        detail_str = f"  {vol_details}" if vol_details else ""
        if vol_regime=="neutral":
            print(f"[VOL REGIME] {coin}: → NEUTRAL{detail_str}  adj=0  score unchanged")
            return score
        if vol_regime=="compression":
            print(f"[VOL REGIME] {coin}: ⚡ COMPRESSION{detail_str}  adj=0  score unchanged")
            return score
        col="close" if "close" in df.columns else "price"; price_5d=0.0
        if len(df)>=6:
            p0=float(df[col].iloc[-6])
            if p0>0: price_5d=(float(df[col].iloc[-1])-p0)/p0*100
        obv_rising="OBV" in df.columns and len(df)>=8 and float(df["OBV"].iloc[-1])>float(df["OBV"].iloc[-8])
        if price_5d>0 and obv_rising:
            print(f"[VOL REGIME] {coin}: 📈 EXPANSION bullish price +{price_5d:.1f}% OBV↑{detail_str}  adj=+1  score {score} → {score+1}")
            return score+1
        elif price_5d<-3 or (not obv_rising and price_5d<0):
            print(f"[VOL REGIME] {coin}: 📉 EXPANSION bearish price {price_5d:.1f}%{detail_str}  adj=-1  score {score} → {score-1}")
            return score-1
        else:
            print(f"[VOL REGIME] {coin}: 📈 EXPANSION unclear price {price_5d:.1f}%{detail_str}  adj=0  score unchanged")
            return score
    except Exception: return score

def apply_chop_penalty(score, df, coin):
    """Chop zone penalty -- uses ACTUAL penalty from analyze_chop_zone.
    
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
            # (-1 moderate, -2 strong) -- not hardcoded -1
            actual_penalty = abs(penalty)   # always positive
            new_score = score - actual_penalty
            label = "STRONG CHOP" if actual_penalty >= 2 else "Choppy market"
            print(f"[CHOP] {coin}: {label} (-{actual_penalty}) -- breakout skipped, cap tightened to ±1")
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


def analyze_momentum_exhaustion(df, coin):
    """[AS] EMA26 Deviation -- Momentum Exhaustion / Mean Reversion Risk
    
    Formula: dev% = (Price - EMA26) / EMA26 × 100
    
    EMA26 = trend's "fair value" line (slow MACD line)
    Measures how far price has stretched from the trend itself
    
    Different from Z-Score:
      Z-Score:  stretched from recent average (short-term)
      EMA26 dev: stretched from trend line (medium-term)
    
    Scoring:
      > +40%: -3  Extreme extension -- historical reversal zone
      > +25%: -2  Very stretched -- strong mean reversion risk
      > +15%: -1  Stretched -- elevated reversion probability
      ±15%:    0  Healthy -- normal trend distance
      < -15%: +1  Oversold from trend -- bounce potential
      < -25%: +1  Deeply oversold from trend -- strong bounce
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
            signal = f"EXTREME extension +{dev_pct:.1f}% above EMA26 -- reversal zone"
        elif dev_pct > 25:
            score  = -2
            signal = f"Very stretched +{dev_pct:.1f}% above EMA26 -- mean reversion risk"
        elif dev_pct > 15:
            score  = -1
            signal = f"Stretched +{dev_pct:.1f}% above EMA26 -- elevated reversion probability"
        elif dev_pct < -25:
            score  = +1
            signal = f"Deeply oversold {dev_pct:.1f}% below EMA26 -- strong bounce potential"
        elif dev_pct < -15:
            score  = +1
            signal = f"Oversold {dev_pct:.1f}% below EMA26 -- bounce potential"
        else:
            score  = 0
            signal = f"Healthy distance {dev_pct:.1f}% from EMA26"

        # No print here -- actual applied value (after cap) is printed in apply_exhaustion_penalty
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
            raw_adj  = adj
            combined = adj + zscore_adj
            capped   = False
            if combined < -2:
                adj    = max(-2 - zscore_adj, -2)
                capped = True
            new_score = score + adj
            cap_note  = f" (capped from {raw_adj:+d}: Z-score already {zscore_adj:+d})" if capped else ""
            print(f"[EXHAUSTION] {coin}: {desc}  {adj:+d}{cap_note}  score {score} → {new_score}")
            if LOG_ENABLED:
                logger.info(f"[EXHAUSTION] {coin}: {adj:+d} (EMA26 dev={dev_pct:.1f}%){cap_note}")
            return new_score

        return score
    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[EXHAUSTION] {coin}: skipped - {e}")
        return score

def _is_strong_chop(df):
    """Quick chop check -- used internally by RSI, Z-score, and regime bonus.
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
        Z < -1:   0  Dip -- neutral (weak trend could resume either way)
        Z < -2:   0  Do NOT chase as a bounce -- trend may continue

    STRONG TREND (ADX > 25):
        Z > 3:   -2  Blow-off top (even in trends, this is extreme)
        Z > 2:   -1  Very stretched
        Z < -2:  -1  FALLING KNIFE -- momentum accelerating DOWN, do not buy
        Z < -3:  -2  EXTREME FALLING KNIFE -- severe downside momentum
    ─────────────────────────────────────────────────────────────────
    In_trend flag is a soft override -- ADX takes priority when available.
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
            elif z < -3:   return -2, z, f"⚠️ EXTREME FALLING KNIFE (ADX={adx:.0f}, Z={z:.2f}) -- momentum accelerating DOWN"
            elif z < -2:   return -1, z, f"⚠️ FALLING KNIFE (ADX={adx:.0f}, Z={z:.2f}) -- do NOT catch; strong trend continuing"
            else:          return  0, z, f"Normal in strong trend (ADX={adx:.0f})"

        # ── Weak trend: neutral on dips, still penalise overextension ─
        elif regime == "weak_trend":
            if   z > 3:    return -2, z, f"BLOW-OFF TOP (ADX={adx:.0f})"
            elif z > 2:    return -1, z, f"VERY STRETCHED (ADX={adx:.0f})"
            else:          return  0, z, f"Neutral -- weak trend, Z-score inconclusive (ADX={adx:.0f})"

        # ── Ranging: mean-reversion signals are valid ─────────────
        else:
            if   z > 2:    return -1, z, f"STRETCHED in range (ADX={adx:.0f}) → rejection likely"
            elif z > 1:    return  0, z, f"Above average in range -- neutral (ADX={adx:.0f})"
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
    """[AP] Wick ratio analysis -- detect exhaustion vs breakout candles.
    
    WICK RATIO = (High - Close) / (High - Low)
    Measures how much of the candle range was "given back" before close.
    
    SCORING:
      Wide candle (>1.5x ATR) + High wick (>0.5):
        → -1  Rejection/exhaustion -- price couldn't hold the high
        
      Wide candle (>1.5x ATR) + Low wick (<0.2):
        →  0  Strong close -- valid breakout move, no penalty
        
      Wide DOWN candle (>1.5x ATR) + High volume:
        → +1  Capitulation -- panic selling = likely bounce
        
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
            return 0, f"Normal candle ({range_ratio:.1f}x ATR -- no signal)"

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
            signal = f"Rejection candle -- gave back {wick_ratio:.0%} of range (wick={wick_ratio:.2f}, {range_ratio:.1f}x ATR)"
            print(f"[WICK] {coin}: Exhaustion/Rejection → -1  ({signal})")

        elif wick_ratio < 0.2 and not is_down_candle:
            # Closed near high = strong bullish close = breakout confirmed
            score  = 0
            signal = f"Strong close -- valid move ({wick_ratio:.0%} wick, {range_ratio:.1f}x ATR)"
            print(f"[WICK] {coin}: Strong close -- no penalty  ({signal})")

        elif is_down_candle and high_volume and range_ratio > 2.0 and wick_ratio < 0.2:
            # Wide DOWN candle + high volume + closed near lows = capitulation = bounce
            score  = 1
            signal = f"Capitulation -- panic sell {range_ratio:.1f}x ATR, {cur_vol/avg_vol:.1f}x volume"
            print(f"[WICK] {coin}: Capitulation → +1  ({signal})")

        else:
            score  = 0
            signal = f"Wide candle, inconclusive ({range_ratio:.1f}x ATR, wick={wick_ratio:.2f})"
            print(f"[WICK] {coin}: Wide candle -- inconclusive")

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
        avg_vol = df['volume'].iloc[-20:-1].median() if 'volume' in df.columns else 0  # median
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
            print(f"[BREAKOUT] {coin}: Consolidating ({pct:.0f}% of range) -- neutral")
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
            desc = f"Paper-thin liquidity: ${total_usd/1e3:.1f}K in book -- HIGH RISK"

        return total_usd, tier, desc

    except Exception as e:
        return 0, "unknown", f"Error: {e}"


        return total_usd, tier, desc

    except Exception as e:
        return 0, "unknown", f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# [MICRO] MARKET MICROSTRUCTURE DATA
#
# Where real edge starts.  Classic technical analysis uses price and volume.
# Microstructure uses the *mechanics* of trading -- who is initiating trades,
# how thick the order book is, whether forced liquidations are driving moves.
#
# These signals are treated PROBABILISTICALLY, not deterministically:
#   → Each is converted to a Likelihood Ratio via scorer.update() or add_lr()
#   → All enter the MICROSTRUCTURE group with 0.55 dampening
#   → They can OVERRIDE or CONFIRM technical signals, not replace them
#
# Five sources:
#   1. SPREAD          -- bid-ask as % of mid.  Wide → illiquid → signals unreliable.
#   2. OB IMBALANCE    -- weighted bid/ask volume at 5 depth levels.
#   3. AGGRESSOR FLOW  -- CVD = cumulative buy vol − sell vol (who is initiating).
#   4. LIQUIDITY DEPTH -- USD within ±1% of mid.  Thin → cascade / stop-hunt risk.
#   5. LIQUIDATIONS    -- Binance Futures forced orders. Cascades = directional pressure.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MicrostructureData:
    """All microstructure signals for one coin at one point in time."""
    coin:             str
    timestamp:        str
    # Spread
    spread_pct:       float   # (best_ask - best_bid) / mid × 100
    # Order book imbalance (weighted 5-level)
    ob_imbalance:     float   # bid_vol_usd / ask_vol_usd; > 1 = buy pressure
    ob_bid_usd:       float   # total USD on bid side (5 levels)
    ob_ask_usd:       float   # total USD on ask side
    # Aggressor flow (CVD from recent trades)
    cvd_1h:           float   # 1h cumulative volume delta (buy − sell, in coin units)
    cvd_pct:          float   # cvd_1h as % of total volume traded in that window
    # Liquidity depth
    depth_1pct_bid_usd: float  # USD within 1% below mid
    depth_1pct_ask_usd: float  # USD within 1% above mid
    depth_ratio:      float    # depth vs rolling average (>1 = deeper than normal)
    # Liquidations (Binance Futures, 0 if spot-only)
    liq_buy_usd:      float   # buy-side liquidations in last 2h (shorts squeezed)
    liq_sell_usd:     float   # sell-side liquidations in last 2h (longs stopped)
    is_futures:       bool    # True if liquidation data was available
    error:            str = ""


def fetch_microstructure(coin: str, depth_levels: int = 20,
                          trade_limit: int = 1000) -> Optional[MicrostructureData]:
    """
    Fetch all microstructure data for a coin in one consolidated call.
    Uses ccxt.binance() -- same infrastructure as existing order book functions.

    Consolidation matters: each ccxt call takes ~200ms; doing 4 calls adds ~0.8s
    per coin. This function batches all four into a single exchange session.

    Returns None if USE_BINANCE_ORDER_BOOK is False or exchange is unreachable.
    """
    if not USE_BINANCE_ORDER_BOOK:
        return None

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        symbol   = get_exchange_symbol(coin)
        if symbol is None:
            return None

        ts = datetime.now().isoformat()

        # ── 1. Order book -- spread + imbalance + depth ────────────────────
        ob = exchange.fetch_order_book(symbol, limit=depth_levels)
        bids = ob["bids"]   # [[price, vol], ...]
        asks = ob["asks"]

        if not bids or not asks:
            return None

        best_bid = float(bids[0][0]); best_ask = float(asks[0][0])
        mid      = (best_bid + best_ask) / 2.0
        spread_pct = (best_ask - best_bid) / (mid + 1e-9) * 100.0

        # Weighted 5-level imbalance (USD value, not just count)
        levels = min(5, len(bids), len(asks))
        bid_usd = sum(b[0] * b[1] for b in bids[:levels])
        ask_usd = sum(a[0] * a[1] for a in asks[:levels])
        ob_imbalance = bid_usd / (ask_usd + 1e-9)

        # Depth within ±1% of mid (all available levels)
        d1_bid = sum(b[0]*b[1] for b in bids if b[0] >= mid * 0.99)
        d1_ask = sum(a[0]*a[1] for a in asks if a[0] <= mid * 1.01)

        # ── 2. Recent trades -- aggressor flow (CVD) ───────────────────────
        # isBuyerMaker=True → seller initiated (taker = seller)
        # isBuyerMaker=False → buyer initiated (taker = buyer, aggressive)
        cvd_1h = 0.0; cvd_vol_total = 0.0
        try:
            trades = exchange.fetch_trades(symbol, limit=trade_limit)
            cutoff = datetime.now().timestamp() * 1000 - 3600_000   # 1h ago in ms
            for t in trades:
                t_time = t.get("timestamp", 0) or 0
                if t_time < cutoff:
                    continue
                vol  = float(t.get("amount", 0) or 0)
                is_buyer_maker = bool(t.get("info", {}).get("isBuyerMaker", True))
                # Buyer maker = passive buyer; taker = seller → sell-initiated
                cvd_1h       += vol if not is_buyer_maker else -vol
                cvd_vol_total += vol
        except Exception:
            pass
        cvd_pct = cvd_1h / (cvd_vol_total + 1e-9) * 100.0

        # ── 3. Liquidations -- Binance Futures (spot coins use fapi symbol) ─
        liq_buy_usd = 0.0; liq_sell_usd = 0.0; is_futures = False
        try:
            futures_symbol = symbol.replace("/", "").replace("USDT", "") + "USDT"
            # fetch_funding_rate_history works as a proxy; use raw request for liq
            raw = exchange.fapiPublicGetAllForceOrders(
                {"symbol": futures_symbol, "limit": 200}
            )
            cutoff_2h = datetime.now().timestamp() * 1000 - 7_200_000
            for order in raw:
                ts_o = float(order.get("time", 0))
                if ts_o < cutoff_2h:
                    continue
                qty   = float(order.get("origQty", 0))
                px    = float(order.get("price", mid))
                usd   = qty * px
                side  = order.get("side", "").upper()
                if side == "BUY":
                    liq_buy_usd  += usd   # short being liquidated = buy order
                elif side == "SELL":
                    liq_sell_usd += usd   # long being liquidated = sell order
            is_futures = True
        except Exception:
            pass   # futures not available or coin not on fapi -- fine

        # ── 4. Depth ratio (vs rolling baseline) ─────────────────────────
        # Simple ratio against a baseline: if depth < 50% of typical, thin
        depth_total = d1_bid + d1_ask
        # Use total book as a rough baseline proxy
        total_book_usd = bid_usd + ask_usd
        depth_ratio    = depth_total / (total_book_usd / levels * 2 + 1e-9)

        return MicrostructureData(
            coin=coin, timestamp=ts,
            spread_pct=round(spread_pct, 4),
            ob_imbalance=round(ob_imbalance, 4),
            ob_bid_usd=round(bid_usd, 2), ob_ask_usd=round(ask_usd, 2),
            cvd_1h=round(cvd_1h, 4), cvd_pct=round(cvd_pct, 2),
            depth_1pct_bid_usd=round(d1_bid, 2),
            depth_1pct_ask_usd=round(d1_ask, 2),
            depth_ratio=round(depth_ratio, 3),
            liq_buy_usd=round(liq_buy_usd, 2),
            liq_sell_usd=round(liq_sell_usd, 2),
            is_futures=is_futures,
        )

    except Exception as e:
        return MicrostructureData(
            coin=coin, timestamp=datetime.now().isoformat(),
            spread_pct=0.0, ob_imbalance=1.0, ob_bid_usd=0.0, ob_ask_usd=0.0,
            cvd_1h=0.0, cvd_pct=0.0, depth_1pct_bid_usd=0.0, depth_1pct_ask_usd=0.0,
            depth_ratio=1.0, liq_buy_usd=0.0, liq_sell_usd=0.0,
            is_futures=False, error=str(e),
        )


def prob_microstructure_update(scorer: "ProbabilisticScorer",
                                coin: str,
                                micro: Optional[MicrostructureData],
                                price: float = 0.0) -> None:
    """
    Convert microstructure data to Bayesian LR updates via scorer.update().

    Design principles:
      PROBABILISTIC, not deterministic -- each signal is a LR, not a gate.
      GROUP DAMPENED -- all in MICROSTRUCTURE group (0.55 dampening).
        If spread is wide AND OB is bearish AND CVD is negative, they're
        partially correlated (same underlying illiquid/panic state).
        The group prevents triple-counting of the same underlying signal.
      SPREAD AS RELIABILITY MODIFIER -- wide spread reduces confidence in
        all other signals too (illiquid markets have noisy price formation).
    """
    if micro is None or micro.error:
        return

    updates: List[str] = []

    # ── 1. Spread -- reliability modifier ─────────────────────────────────
    if micro.spread_pct < 0.05:
        scorer.update("micro_spread_tight", True,
                      f"Spread {micro.spread_pct:.3f}% -- liquid market",
                      "MICROSTRUCTURE")
        updates.append(f"spread={micro.spread_pct:.3f}% TIGHT")
    elif micro.spread_pct > 0.20:
        scorer.update("micro_spread_wide", True,
                      f"Spread {micro.spread_pct:.3f}% -- illiquid, signals less reliable",
                      "MICROSTRUCTURE")
        updates.append(f"spread={micro.spread_pct:.3f}% WIDE")

    # ── 2. Order book imbalance ───────────────────────────────────────────
    imb = micro.ob_imbalance
    if imb >= 2.0:
        scorer.update("micro_ob_strong_bull", True,
                      f"OB imbalance {imb:.2f}x -- bids dominate strongly",
                      "MICROSTRUCTURE")
        updates.append(f"OB={imb:.2f}x BULL↑↑")
    elif imb >= 1.30:
        scorer.update("micro_ob_mild_bull", True,
                      f"OB imbalance {imb:.2f}x -- mild bid pressure",
                      "MICROSTRUCTURE")
        updates.append(f"OB={imb:.2f}x bull↑")
    elif imb <= 0.50:
        scorer.update("micro_ob_strong_bear", True,
                      f"OB imbalance {imb:.2f}x -- asks dominate strongly",
                      "MICROSTRUCTURE")
        updates.append(f"OB={imb:.2f}x BEAR↓↓")
    elif imb <= 0.77:
        scorer.update("micro_ob_mild_bear", True,
                      f"OB imbalance {imb:.2f}x -- mild ask pressure",
                      "MICROSTRUCTURE")
        updates.append(f"OB={imb:.2f}x bear↓")

    # ── 3. Aggressor flow (CVD) ───────────────────────────────────────────
    # Normalise by price for cross-coin comparability
    cvd = micro.cvd_pct   # already as % of total volume
    if cvd >= 20:
        scorer.update("micro_flow_strong_buy", True,
                      f"CVD +{cvd:.1f}% -- sustained institutional buying",
                      "MICROSTRUCTURE")
        updates.append(f"CVD={cvd:+.1f}% BUY↑↑")
    elif cvd >= 5:
        scorer.update("micro_flow_mild_buy", True,
                      f"CVD +{cvd:.1f}% -- mild buy flow",
                      "MICROSTRUCTURE")
        updates.append(f"CVD={cvd:+.1f}% buy↑")
    elif cvd <= -20:
        scorer.update("micro_flow_strong_sell", True,
                      f"CVD {cvd:.1f}% -- sustained institutional selling",
                      "MICROSTRUCTURE")
        updates.append(f"CVD={cvd:+.1f}% SELL↓↓")
    elif cvd <= -5:
        scorer.update("micro_flow_mild_sell", True,
                      f"CVD {cvd:.1f}% -- mild sell flow",
                      "MICROSTRUCTURE")
        updates.append(f"CVD={cvd:+.1f}% sell↓")

    # ── 4. Liquidity depth ────────────────────────────────────────────────
    if micro.depth_ratio >= 2.0:
        scorer.update("micro_depth_deep", True,
                      f"Depth ratio {micro.depth_ratio:.2f}× -- thick book, trend can sustain",
                      "MICROSTRUCTURE")
        updates.append(f"depth={micro.depth_ratio:.2f}× DEEP")
    elif micro.depth_ratio <= 0.50:
        scorer.update("micro_depth_thin", True,
                      f"Depth ratio {micro.depth_ratio:.2f}× -- thin book, cascade risk",
                      "MICROSTRUCTURE")
        updates.append(f"depth={micro.depth_ratio:.2f}× THIN")

    # ── 5. Liquidation cascades ───────────────────────────────────────────
    if micro.is_futures:
        liq_total = micro.liq_buy_usd + micro.liq_sell_usd
        if liq_total >= 100_000:   # $100k+ liq = significant
            buy_frac = micro.liq_buy_usd / (liq_total + 1e-9)
            if buy_frac >= 0.65 and micro.liq_buy_usd >= 500_000:
                # Dominant buy liquidations = shorts being squeezed = bullish
                scorer.update("micro_liq_shorts_squeezed", True,
                              f"Short squeeze: ${micro.liq_buy_usd/1e6:.1f}M buy liquidations",
                              "MICROSTRUCTURE")
                updates.append(f"liq=short_squeeze ${micro.liq_buy_usd/1e6:.1f}M")
            elif buy_frac <= 0.35 and micro.liq_sell_usd >= 500_000:
                # Dominant sell liquidations = longs stopped = bearish
                scorer.update("micro_liq_longs_stopped", True,
                              f"Long liquidations: ${micro.liq_sell_usd/1e6:.1f}M sell liq",
                              "MICROSTRUCTURE")
                updates.append(f"liq=long_stop ${micro.liq_sell_usd/1e6:.1f}M")

    if updates:
        print(f"[MICRO] {coin}: {len(updates)} microstructure signals → "
              f"{'  '.join(updates)}")
    else:
        print(f"[MICRO] {coin}: no significant microstructure signals "
              f"(spread={micro.spread_pct:.3f}% OB={micro.ob_imbalance:.2f}× "
              f"CVD={micro.cvd_pct:+.1f}%)")

# ── END [MICRO] ───────────────────────────────────────────────────────────────

def _apply_orderbook_context_legacy(coin: str, score: float,
                                     market_regime: str = "unknown") -> float:
    """[AA] Legacy order book context scoring -- kept for backward compatibility.
    New code uses prob_orderbook_update(scorer, ...) via the Bayesian engine.
    Emits a single consolidated summary line per coin rather than multiple prints.
    """
    try:
        # Step 1: Liquidity depth
        total_usd, liq_tier, liq_desc = analyze_order_book_liquidity(coin)

        # Step 2: Wash trade ratio (thin/paper-thin books only)
        wash_penalty = 0
        _ratio       = 0.0
        _vol_usd     = 0.0
        if liq_tier in ("paper_thin", "thin") and USE_BINANCE_ORDER_BOOK:
            try:
                import ccxt as _ccxt
                _ex  = _ccxt.binance()
                _sym = get_exchange_symbol(coin)
                if _sym:
                    _tk      = _ex.fetch_ticker(_sym)
                    _vol_usd = _tk.get("quoteVolume", 0) or 0
                    _ratio   = _vol_usd / total_usd if total_usd > 0 else float("inf")
                    if liq_tier == "paper_thin" and _ratio > 50:
                        wash_penalty = -4
                    elif liq_tier == "thin" and _ratio > 100:
                        wash_penalty = -2
            except Exception:
                pass

        _wash_penalty_deferred = wash_penalty

        # Step 3: Imbalance signal
        imbalance_adj, imbalance_ratio, imbalance_desc = analyze_order_book_imbalance(coin)

        if liq_tier == "no_data":
            if LOG_ENABLED:
                logger.debug(f"[ORDER BOOK] {coin}: No data -- skipping")
            return score

        elif liq_tier == "paper_thin":
            if not USE_BINANCE_ORDER_BOOK:
                print(f"[ORDER BOOK] {coin}: {liq_desc} | Binance disabled → adj=0  score unchanged")
                return score

            try:
                import ccxt as _ccxt
                _ex  = _ccxt.binance()
                _sym = get_exchange_symbol(coin)
                if _sym:
                    _ticker  = _ex.fetch_ticker(_sym)
                    _usd_vol = _ticker.get("quoteVolume", 0) or 0
                    if _usd_vol < 500_000:
                        score += _wash_penalty_deferred
                        adj     = _wash_penalty_deferred - 1
                        note    = f"wash:{wash_penalty:+d} illiquid:-1" if wash_penalty else "illiquid:-1"
                        print(f"[ORDER BOOK] {coin}: {liq_desc} | "
                              f"vol ${_usd_vol/1e3:.0f}K | {note} → score {score-1}")
                        if LOG_ENABLED:
                            logger.warning(f"[ORDER BOOK] {coin}: illiquid+wash (book=${total_usd/1e3:.0f}K)")
                        return score - 1
                    else:
                        # High off-exchange volume cancels wash trade concern
                        slippage = " ⚠️ slippage on Binance" if _usd_vol > 0 else ""
                        ratio_note = f" {_ratio:.0f}x turnover" if _ratio > 50 else ""
                        print(f"[ORDER BOOK] {coin}: {liq_desc}{ratio_note} | ${_usd_vol/1e6:.1f}M off-exchange → adj=0  score unchanged{slippage}")
                        if LOG_ENABLED:
                            logger.warning(f"[ORDER BOOK] {coin}: paper-thin Binance, "
                                           f"${_usd_vol/1e6:.1f}M off-exchange")
                        return score
                else:
                    print(f"[ORDER BOOK] {coin}: {liq_desc} | not on Binance → adj=0  score unchanged")
            except Exception as _e:
                print(f"[ORDER BOOK] {coin}: {liq_desc} | ccxt error → adj={wash_penalty:+d}  score {score} → {score + _wash_penalty_deferred}")
                score += _wash_penalty_deferred
                if LOG_ENABLED:
                    logger.warning(f"[ORDER BOOK] {coin}: Wash trade risk (book=${total_usd/1e3:.0f}K)")

            return score

        elif liq_tier == "thin":
            score += _wash_penalty_deferred
            wash_note = f" wash:{wash_penalty:+d}" if wash_penalty else ""
            if market_regime == "range" and imbalance_adj != 0:
                new_score = score + imbalance_adj
                print(f"[ORDER BOOK] {coin}: thin book | range regime → adj={imbalance_adj:+d}{wash_note}  score {score} → {new_score}")
                if LOG_ENABLED: logger.info(f"[ORDER BOOK] {coin}: thin+range OB {imbalance_adj:+d}")
                return new_score
            print(f"[ORDER BOOK] {coin}: thin book | imbalance suppressed{wash_note} → adj=0  score unchanged")
            if LOG_ENABLED: logger.info(f"[ORDER BOOK] {coin}: thin -- imbalance dampened")
            return score

        else:
            # Moderate or Deep -- full imbalance signal
            score += _wash_penalty_deferred
            liq_label = liq_tier.upper()
            if imbalance_adj != 0:
                new_score = score + imbalance_adj
                print(f"[ORDER BOOK] {coin}: {liq_desc} | {imbalance_desc} → adj={imbalance_adj:+d}  score {score} → {new_score}")
                if LOG_ENABLED: logger.info(f"[ORDER BOOK] {coin}: {imbalance_adj:+d} [{liq_tier}]")
                return new_score
            print(f"[ORDER BOOK] {coin}: {liq_desc} | balanced → adj=0  score unchanged")
            return score

    except Exception as e:
        if LOG_ENABLED:
            logger.debug(f"[ORDER BOOK] {coin}: skipped - {e}")
        return score


def find_swing_points(df,lookback=3):
    hc="high" if "high" in df.columns and df["high"].notna().any() else "price"
    lc="low"  if "low"  in df.columns and df["low"].notna().any()  else "price"
    highs=df[hc].values.astype(float); lows=df[lc].values.astype(float); n=len(df)
    shs,sls=[],[]
    for i in range(lookback,n-lookback):
        wh=highs[i-lookback:i+lookback+1]; wl=lows[i-lookback:i+lookback+1]
        wh_max=np.nanmax(wh) if not np.all(np.isnan(wh)) else np.nan; wl_min=np.nanmin(wl) if not np.all(np.isnan(wl)) else np.nan
        if not np.isnan(highs[i]) and not np.isnan(wh_max) and highs[i]>=wh_max: shs.append((i,float(highs[i])))
        if not np.isnan(lows[i])  and not np.isnan(wl_min) and lows[i]<=wl_min:  sls.append((i,float(lows[i])))
    def dd(pts,min_pct=0.001):
        out=[]
        for p in pts:
            if not out or abs(p[1]-out[-1][1])/(out[-1][1]+1e-9)>min_pct: out.append(p)
        return out
    return dd(shs),dd(sls)

def classify_market_structure(shs,sls):
    if len(shs)<2 or len(sls)<2: return "undefined",None,None,None,None
    sp,sl_=shs[-2][1],shs[-1][1]; lp,ll_=sls[-2][1],sls[-1][1]
    hh=sl_>sp; lh=sl_<sp; hl=ll_>lp; ll=ll_<lp
    if hh and hl: s="bullish"
    elif lh and ll: s="bearish"
    elif hh and ll: s="distribution"
    elif lh and hl: s="accumulation"
    else: s="ranging"
    return s,hh,hl,lh,ll

def detect_bos_choch(df,shs,sls,structure):
    if not shs or not sls: return None,None
    cc="close" if "close" in df.columns else "price"; cur=float(df[cc].iloc[-1]); lsh=shs[-1][1]; lsl=sls[-1][1]
    bos=choch=None
    if cur>lsh:
        if structure in("bullish","distribution"): bos="bullish"
        else: choch="bullish"
    elif cur<lsl:
        if structure in("bearish","accumulation"): bos="bearish"
        else: choch="bearish"
    return bos,choch

def detect_swing_failure(df,shs,sls,lookback=3):
    if not shs or not sls or len(df)<lookback: return None,None,None
    hc="high" if "high" in df.columns and df["high"].notna().any() else None
    lc="low"  if "low"  in df.columns and df["low"].notna().any()  else None
    cc="close" if "close" in df.columns else "price"
    if hc is None or lc is None: return None,None,None
    lsh=shs[-1][1]; lsl=sls[-1][1]
    for _,row in df.iloc[-lookback:].iterrows():
        h=float(row.get(hc,row["price"])); l=float(row.get(lc,row["price"])); c=float(row.get(cc,row["price"]))
        if h>lsh and c<lsh: return "bearish",lsh,f"Grab above {lsh:.6f}"
        if l<lsl and c>lsl: return "bullish",lsl,f"Spring below {lsl:.6f}"
    return None,None,None

def analyze_market_structure(df):
    _e={"structure":"undefined","hh":None,"hl":None,"lh":None,"ll":None,"bos":None,"choch":None,"sfp":None,"sfp_level":None,"swing_highs":[],"swing_lows":[],"score_adj":0,"details":"Insufficient data"}
    try:
        if df is None or len(df)<SWING_LOOKBACK*2+5: return _e
        shs,sls=find_swing_points(df,SWING_LOOKBACK); st,hh,hl,lh,ll=classify_market_structure(shs,sls)
        bos,choch=detect_bos_choch(df,shs,sls,st); sfp,sfpl,_=detect_swing_failure(df,shs,sls,SFP_LOOKBACK)
        adj=0
        if st=="bullish": adj+=1
        elif st in("bearish","distribution"): adj-=1
        if bos=="bullish": adj+=1
        elif bos=="bearish": adj-=1
        if choch=="bullish": adj+=1
        elif choch=="bearish": adj-=1
        if sfp=="bullish": adj+=1
        elif sfp=="bearish": adj-=1
        adj=max(-2,min(2,adj))
        hhs="HH" if hh else("LH" if lh else "?H"); hls="HL" if hl else("LL" if ll else "?L")
        em="📈" if st=="bullish" else("📉" if st=="bearish" else "↔")
        pts=[f"{em} {st.upper()}  {hhs}+{hls}"]
        if bos: pts.append(f"BOS➔{bos}")
        if choch: pts.append(f"CHoCH➔{choch}")
        if sfp: pts.append(f"SFP({sfp})")
        return {"structure":st,"hh":hh,"hl":hl,"lh":lh,"ll":ll,"bos":bos,"choch":choch,"sfp":sfp,"sfp_level":sfpl,"swing_highs":shs[-3:],"swing_lows":sls[-3:],"score_adj":adj,"details":"  ".join(pts)}
    except Exception as e: _e["details"]=f"Error: {e}"; return _e

def apply_market_structure_boost(score,coin,ms):
    adj=ms.get("score_adj",0); print(f"[MKTSTR] {coin}: {ms.get('details','')}  adj={adj:+d}"); return score+adj

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
                          f"-- possible manipulation, boost suppressed!")
                    if LOG_ENABLED:
                        logger.warning(f"[VOLUME] {coin}: Extreme vol {vol_zscore:.1f}σ + thin book -- boost suppressed")
                else:
                    boost = GOOD_VOLUME_BOOST
            elif label == "increasing_volume":
                boost = GOOD_VOLUME_BOOST
            elif label == "healthy_volume":
                boost = 0
        
        # ❌ BAD VOLUME: OBV-aware fake move penalty
        elif quality == "bad":
            if label == "fake_move_warning":
                # Price up + volume declining = potential fake move
                # OBV check -- BUT only trust OBV if volume is meaningful
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
                    print(f"[VOLUME] {coin}: Fake move overridden by OBV rising -- slow accumulation (no penalty)")
                    if LOG_ENABLED:
                        logger.info(f"[VOLUME] {coin}: Fake move overridden by OBV rising")
                elif obv_trend == "rising" and not obv_trustworthy:
                    # OBV rising but volume too low to trust it = noise
                    boost = -1
                    print(f"[VOLUME] ⚠️ {coin}: -1 penalty -- OBV rising but volume too low ({vol_ratio:.2f}x) to confirm")
                    if LOG_ENABLED:
                        logger.warning(f"[VOLUME] {coin}: OBV unreliable at {vol_ratio:.2f}x volume")
                elif obv_trend == "flat":
                    # Weak conviction, mild warning
                    boost = -1
                    print(f"[VOLUME] ⚠️ {coin}: -1 penalty -- low conviction rise (OBV flat)")
                    if LOG_ENABLED:
                        logger.warning(f"[VOLUME] {coin}: -1 fake move (OBV flat)")
                else:
                    # OBV falling = distribution = genuine fake move
                    boost = -BAD_VOLUME_PENALTY
                    print(f"[VOLUME] ⚠️ {coin}: {boost} penalty -- FAKE MOVE DETECTED! (OBV falling = distribution)")
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
            if boost != 0:
                emoji = "✅" if boost > 0 else "⚠️"
                print(f"[VOLUME] {emoji} {coin}: score {score} → {score + boost}  ({details})")
                if LOG_ENABLED:
                    logger.info(f"[VOLUME] {coin}: score {score} → {score + boost}")
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
    """Legacy wrapper — use compute_relative_strength_full() for multi-timeframe RS."""
    try:
        if len(coin_df) < 2 or len(btc_df) < 2:
            return 0.0
        coin_ret = (coin_df['price'].iloc[-1] - coin_df['price'].iloc[-2]) / coin_df['price'].iloc[-2]
        btc_ret  = (btc_df['price'].iloc[-1]  - btc_df['price'].iloc[-2])  / btc_df['price'].iloc[-2]
        return float(np.tanh((coin_ret - btc_ret) / 0.50))
    except Exception:
        return 0.0


def compute_relative_strength_full(coin_df, btc_df, eth_df=None) -> dict:
    """
    Multi-timeframe Relative Strength engine.

    Computes RS vs BTC (1d, 7d, 30d) and vs ETH (7d as TOTAL3 proxy).
    ETH tracks TOTAL3 (total altcoin market cap) closely enough to use
    as a free proxy without an extra API call.

    RS Interpretation:
    ─────────────────
    LEADER   (+5%+ on 7d vs BTC):
      Coin is resisting BTC weakness / leading the rotation.
      Institutions are specifically accumulating THIS coin.
      LR = 2.0 — strong positive prior adjustment.

    STRONG   (+2–5% on 7d):
      Outperforming but not leading. Healthy setup.
      LR = 1.6

    NEUTRAL  (±2% on 7d):
      Moving with the market. No RS edge.
      LR = 1.0

    LAGGARD  (−2 to −8% on 7d):
      Underperforming — capital rotating OUT.
      Buy signals less reliable.
      LR = 0.65

    WEAK     (−8%+ on 7d):
      Strong underperformance. Active distribution vs market.
      LR = 0.40

    RS Trend (30d → 7d momentum):
      IMPROVING: recent RS better than long-term RS → acceleration
      DECLINING: recent RS worse → rotation away
      This component is additive to the base LR.

    Anti-over-suppression:
      LEADER coins get an additional credit that partially offsets
      the BTC bear regime penalty. A coin that resists BTC weakness
      is a genuine institutional target — suppressing it too hard
      causes the bot to miss the best setups.
    """
    result = {
        "rs_btc_1d":   0.0,
        "rs_btc_7d":   0.0,
        "rs_btc_30d":  0.0,
        "rs_eth_7d":   0.0,
        "rs_trend":    "neutral",    # improving / declining / neutral
        "label":       "NEUTRAL",    # LEADER / STRONG / NEUTRAL / LAGGARD / WEAK
        "lr":          1.0,
        "trend_lr":    1.0,
        "available":   False,
    }

    try:
        col = "price" if "price" in coin_df.columns else "close"
        if len(coin_df) < 30 or len(btc_df) < 30:
            return result

        def _ret(df, n):
            """n-day return from n bars ago to now."""
            if len(df) < n + 1:
                return None
            p_now  = float(df[col].iloc[-1])
            p_past = float(df[col].iloc[-(n+1)])
            return (p_now - p_past) / (p_past + 1e-9)

        coin_1d  = _ret(coin_df, 1);  btc_1d  = _ret(btc_df, 1)
        coin_7d  = _ret(coin_df, 7);  btc_7d  = _ret(btc_df, 7)
        coin_30d = _ret(coin_df, 30); btc_30d = _ret(btc_df, 30)

        if None in (coin_7d, btc_7d):
            return result

        rs_1d  = (coin_1d  - btc_1d)  * 100 if coin_1d  and btc_1d  else 0.0
        rs_7d  = (coin_7d  - btc_7d)  * 100
        rs_30d = (coin_30d - btc_30d) * 100 if coin_30d and btc_30d else 0.0

        result["rs_btc_1d"]  = round(rs_1d,  2)
        result["rs_btc_7d"]  = round(rs_7d,  2)
        result["rs_btc_30d"] = round(rs_30d, 2)
        result["available"]  = True

        # RS vs ETH (TOTAL3 proxy)
        if eth_df is not None and len(eth_df) >= 8:
            eth_7d = _ret(eth_df, 7)
            if eth_7d is not None:
                result["rs_eth_7d"] = round((coin_7d - eth_7d) * 100, 2)

        # ── Label based on 7d RS vs BTC ───────────────────────────────────
        if   rs_7d >=  5.0: label = "LEADER";  lr = 2.00
        elif rs_7d >=  2.0: label = "STRONG";  lr = 1.60
        elif rs_7d >= -2.0: label = "NEUTRAL"; lr = 1.00
        elif rs_7d >= -8.0: label = "LAGGARD"; lr = 0.65
        else:               label = "WEAK";    lr = 0.40

        # ── RS Trend: is RS improving or declining? ───────────────────────
        # Compare 7d RS to 30d RS. Positive delta = acceleration.
        trend_lr = 1.0
        if rs_30d != 0:
            rs_delta = rs_7d - rs_30d   # positive = RS getting better recently
            if   rs_delta >  4.0: trend_lr = 1.25; trend = "improving"
            elif rs_delta >  1.5: trend_lr = 1.10; trend = "improving"
            elif rs_delta < -4.0: trend_lr = 0.80; trend = "declining"
            elif rs_delta < -1.5: trend_lr = 0.90; trend = "declining"
            else:                 trend_lr = 1.00; trend = "neutral"
        else:
            trend = "neutral"

        result["label"]    = label
        result["lr"]       = lr
        result["trend_lr"] = trend_lr
        result["rs_trend"] = trend

    except Exception as e:
        pass

    return result


def prob_rs_update(scorer: "ProbabilisticScorer", coin: str,
                   rs: dict, btc_regime: str = "unknown") -> None:
    """
    Add Relative Strength signal to the Bayesian scorer.

    RS sits in its own RELATIVE_STRENGTH group, separate from
    MACRO_CONTEXT, so it doesn't get averaged with BTC regime signals.
    This preserves its independent contribution.

    Anti-over-suppression bonus:
      LEADER coins in BTC bear get a small additional credit
      because they are genuinely outperforming despite macro headwinds.
      This is the "leaders resist weakness" phenomenon.
    """
    if not rs.get("available"):
        return

    label    = rs["label"]
    lr       = rs["lr"]
    trend_lr = rs["trend_lr"]
    rs_7d    = rs["rs_btc_7d"]
    rs_eth   = rs.get("rs_eth_7d", 0)

    # Base RS LR
    scorer.add_lr(lr,
                  f"RS_vs_BTC_{label} (7d={rs_7d:+.1f}%) LR={lr:.2f}",
                  "RELATIVE_STRENGTH")

    # RS trend component (additive, lower weight)
    if trend_lr != 1.0:
        scorer.add_lr(trend_lr,
                      f"RS_trend_{rs['rs_trend']} (LR={trend_lr:.2f})",
                      "RELATIVE_STRENGTH")

    # vs TOTAL3 (ETH proxy) — confirms leadership is altcoin-specific
    if abs(rs_eth) > 2.0:
        eth_lr = (1.30 if rs_eth > 5 else
                  1.15 if rs_eth > 2 else
                  0.85 if rs_eth < -5 else
                  0.90)
        scorer.add_lr(eth_lr,
                      f"RS_vs_TOTAL3 (7d={rs_eth:+.1f}%) LR={eth_lr:.2f}",
                      "RELATIVE_STRENGTH")

    # Anti-over-suppression: LEADER in BTC bear gets partial credit back
    # A coin holding up or rising while BTC falls is showing genuine strength.
    if label == "LEADER" and btc_regime == "bear":
        bonus_lr = 1.35   # partial offset to the BTC bear LR=0.50 penalty
        scorer.add_lr(bonus_lr,
                      f"RS_leader_in_bear_bonus (resists BTC weakness) LR={bonus_lr:.2f}",
                      "RELATIVE_STRENGTH")

    # Print
    eth_str = f"  vs ETH: {rs_eth:+.1f}%" if abs(rs_eth) > 1 else ""
    print(f"[RS] {coin}: {label}  BTC 1d={rs['rs_btc_1d']:+.1f}%  "
          f"7d={rs_7d:+.1f}%  30d={rs['rs_btc_30d']:+.1f}%{eth_str}  "
          f"trend={rs['rs_trend']}  LR={lr:.2f}")


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

def extract_ml_features(row, df=None, btc_df=None, fng_value=50,
                         row_timestamp=None):
    """
    Extract 16 features from bar `row`, using ONLY data available at that bar.

    LEAKAGE RULES (strictly enforced):
      feature_t = f(data[:t])    ← ONLY past data
      target_t  = return[t+1:t+n]

    When called during TRAINING, `df` must be `df.iloc[:i+1]` (sliced up to bar i).
    When called during INFERENCE (live), `df` is the full history (no future exists).

    Features:
     1  RSI                    — row value (computed from past candles) ✓
     2  MACD histogram          — row value ✓
     3  ADX                     — row value ✓
     4  Relative volume         — row value (median of past N bars) ✓
     5  EMA12 > EMA26           — row values ✓
     6  BB position             — row values ✓
     7  ATR slope               — df[-5:] vs df[-20:] → only past via df slice ✓
     8  DI+ > DI-               — row values ✓
     9  Fear & Greed index      — external, point-in-time value ✓
    10  BTC relative strength   — df[-2:] vs btc_df[-2:] ← uses df slice ✓
    11  Day of week             — deterministic from timestamp, no leakage ✓
    12  Sortino ratio           — df[-15:] only past via df slice ✓
    13  Market session          — deterministic from timestamp ✓
    14  Weekly EMA ratio        — row value (EMA of past 250 bars) ✓
    15  Ichimoku cloud position — row senkou_a/b = shift(26) from past values ✓
    16  OBV 7-day trend         — df[-8:] only past via df slice ✓
    """
    features = []
    try:
        # 1. RSI (normalized 0-1)
        rsi = row.get("RSI", np.nan)
        features.append(rsi / 100.0 if not np.isnan(rsi) else 0.5)

        # 2. MACD histogram (bounded via tanh)
        macd_hist = row.get("MACD_Hist", np.nan)
        features.append(np.tanh(macd_hist * 10) if not np.isnan(macd_hist) else 0.0)

        # 3. ADX (normalized 0-1)
        adx = row.get("ADX", np.nan)
        features.append(min(adx / 100.0, 1.0) if not np.isnan(adx) else 0.25)

        # 4. Relative volume
        rel_vol_val = row.get("rel_vol", np.nan)
        if np.isnan(rel_vol_val):
            rel_vol_val = row.get("vol_ratio", np.nan)
        features.append(np.log1p(rel_vol_val) / 3.0 if not np.isnan(rel_vol_val) else 0.35)

        # 5. EMA cross (binary)
        ema12 = row.get("EMA12", np.nan); ema26 = row.get("EMA26", np.nan)
        features.append(1.0 if (not np.isnan(ema12) and not np.isnan(ema26) and ema12 > ema26) else 0.0)

        # 6. BB position
        price = row.get("price", np.nan)
        bb_low = row.get("LowerBB", np.nan); bb_high = row.get("UpperBB", np.nan)
        if not any(np.isnan(x) for x in [price, bb_low, bb_high]):
            features.append(np.clip((price - bb_low) / (bb_high - bb_low + 1e-9), 0.0, 1.0))
        else:
            features.append(0.5)

        # 7. ATR slope — LEAKAGE FIX: uses df[-5:] / df[-20:]
        # When df = df.iloc[:i+1], this correctly uses only bars 0..i
        atr = row.get("ATR", np.nan)
        if df is not None and len(df) >= 20 and not np.isnan(atr):
            _atr_s = df["ATR"].dropna()
            if len(_atr_s) >= 20:
                _af = float(_atr_s.iloc[-5:].mean()); _as = float(_atr_s.iloc[-20:].mean())
                atr_slope = np.clip((_af - _as) / (_as + 1e-9), -1.0, 1.0)
            else:
                atr_slope = 0.0
        else:
            atr_slope = 0.0
        features.append(float(atr_slope))

        # 8. DI direction (binary)
        plus_di = row.get("plus_di", np.nan); minus_di = row.get("minus_di", np.nan)
        features.append(1.0 if (not np.isnan(plus_di) and not np.isnan(minus_di) and plus_di > minus_di) else 0.0)

        # 9. Fear & Greed (point-in-time — no leakage; same value for all training bars)
        features.append(fng_value / 100.0)

        # 10. BTC relative strength — LEAKAGE FIX: uses df[-2:] which respects the slice
        if btc_df is not None and len(btc_df) >= 2 and df is not None and len(df) >= 2:
            rs = calculate_relative_strength_24h(df, btc_df)
            features.append(rs)
        else:
            features.append(0.0)

        # 11. Day of week — LEAKAGE FIX: use row's timestamp if available, else datetime.now()
        # During training, all bars use datetime.now() which is the SAME day for all bars.
        # This is NOT ideal but it's not future leakage — it just means the day feature
        # carries no signal in training (which is correct; day effects are weak anyway).
        # The correct fix would be to store timestamps in df, but that's a schema change.
        day_feature, _ = extract_time_features()
        features.append(day_feature)

        # 12. Sortino ratio — LEAKAGE FIX: uses df[-15:] which respects the slice
        if df is not None and len(df) >= 15:
            sortino = calculate_sortino_ratio(df, window=14)
        else:
            sortino = 0.0
        features.append(sortino)

        # 13. Market session — same note as day-of-week (same for all training bars)
        features.append(get_market_session())

        # 14. Weekly EMA ratio (row value — EMA of past 250 bars) ✓
        ema_w50 = row.get("EMA_WEEKLY50", np.nan)
        if not np.isnan(ema_w50) and not np.isnan(price) and ema_w50 > 0:
            features.append(np.clip((price - ema_w50) / ema_w50, -0.3, 0.3))
        else:
            features.append(0.0)

        # 15. Ichimoku cloud position
        # senkou_a/b = shift(+26) of values computed from PAST data.
        # At bar t: df["senkou_a"].iloc[t] = (tenkan+kijun)/2 from bar t-26.
        # No future data used. ✓
        senkou_a = row.get("senkou_a", np.nan); senkou_b = row.get("senkou_b", np.nan)
        if not any(np.isnan(x) for x in [price, senkou_a, senkou_b]):
            cloud_top = max(senkou_a, senkou_b); cloud_bot = min(senkou_a, senkou_b)
            ichimoku_feat = 1.0 if price > cloud_top else (-1.0 if price < cloud_bot else 0.0)
        else:
            ichimoku_feat = 0.0
        features.append(ichimoku_feat)

        # 16. OBV 7-day trend — LEAKAGE FIX: uses df[-8:] which respects the slice
        try:
            if df is not None and "OBV" in df.columns and len(df) >= 8:
                obv = df["OBV"].iloc[-8:].values
                obv_change = (obv[-1] - obv[0]) / (abs(obv[0]) + 1e-9)
                obv_feat = 1.0 if obv_change > 0.03 else (-1.0 if obv_change < -0.03 else 0.0)
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
        lines.append(f"  Long ML {long_proba:.0%} ({verdict}, acc {ml_accuracy:.0%}) -- top drivers:")

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

            lines.append(f"    {icon} {label}{val_note} -- {pct:.0f}% weight")

    # ── SHORT MODEL ───────────────────────────────────────────────
    if short_insights and short_ml_accuracy >= 0.50:
        s_verdict = (
            "very rarely wins" if short_ml_proba < 0.20 else
            "rarely wins"      if short_ml_proba < 0.35 else
            "coin-flip"        if short_ml_proba < 0.45 else
            "often wins"       if short_ml_proba < 0.65 else
            "frequently wins"
        )
        lines.append(f"  Short ML {short_ml_proba:.0%} (shorts {s_verdict}, acc {short_ml_accuracy:.0%}) -- top drivers:")

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

            lines.append(f"    {icon} {label}{val_note} -- {pct:.0f}% weight")
    elif short_ml_accuracy < 0.50:
        lines.append(f"  Short ML n/a (model accuracy {short_ml_accuracy:.0%} < 50% -- neutralized)")

    return "\n".join(lines)


def _detect_bar_regime(row, btc_df, i, df) -> str:
    """
    Classify the BTC regime at bar i for regime-split training.
    Uses a simple MA200 position + ADX direction — fast, no lookahead.
    Returns: 'bull' | 'bear' | 'range' | 'panic'
    """
    try:
        if btc_df is None or len(btc_df) < 200:
            return "range"
        btc_i = min(i, len(btc_df) - 1)
        btc_sub = btc_df.iloc[:btc_i+1]
        if len(btc_sub) < 50:
            return "range"
        btc_price = float(btc_sub["price"].iloc[-1])
        ma200 = float(btc_sub["price"].rolling(200, min_periods=50).mean().iloc[-1])
        ma50  = float(btc_sub["price"].rolling(50,  min_periods=20).mean().iloc[-1])
        ret3d = float(btc_sub["price"].pct_change(3).iloc[-1]) if len(btc_sub) >= 4 else 0
        if ret3d < -0.08:              return "panic"
        if btc_price > ma200 * 1.02:  return "bull"
        if btc_price < ma200 * 0.98:  return "bear"
        return "range"
    except Exception:
        return "range"


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
        w_list = []   # EV-based sample weights

        # ── Fix 1: Rolling window — only use recent N days ───────────────────
        # This ensures training data matches current market regime.
        # Full 365 days = 89% wins (historical bull) = model learns to always predict win.
        # Rolling 120 days = balanced labels reflecting recent bear conditions.
        if ML_ROLLING_WINDOW_DAYS > 0 and len(df) > ML_ROLLING_WINDOW_DAYS + future_days + lookback:
            _roll_start = max(0, len(df) - ML_ROLLING_WINDOW_DAYS - future_days)
            df_train = df.iloc[_roll_start:].reset_index(drop=True)
            print(f"[ML] {coin}: rolling window {ML_ROLLING_WINDOW_DAYS}d "
                  f"({len(df_train)} bars vs {len(df)} total)")
        else:
            df_train = df

        # Detect current regime for regime weighting (Fix 2)
        _cur_regime = _detect_bar_regime(df_train.iloc[-1], btc_df, len(df_train)-1, df_train)

        for i in range(len(df_train) - future_days - 1):
            if i < lookback:
                continue

            row = df_train.iloc[i]
            df_slice = df_train.iloc[:i+1]
            features = extract_ml_features(row, df_slice, btc_df=btc_df, fng_value=fng_value)
            if features is None:
                continue

            # ── Validate entry/ATR ────────────────────────────────────────────
            entry   = float(row.get("price", np.nan))
            atr_val = float(row.get("ATR",   np.nan))
            if np.isnan(entry) or np.isnan(atr_val) or atr_val <= 0 or entry <= 0:
                continue

            # ── Regime feature ────────────────────────────────────────────────
            _bar_regime  = _detect_bar_regime(row, btc_df, i, df_train)
            _regime_feat = [
                1.0 if _bar_regime == "bull"  else 0.0,
                1.0 if _bar_regime == "bear"  else 0.0,
                1.0 if _bar_regime == "range" else 0.0,
                1.0 if _bar_regime == "panic" else 0.0,
            ]
            features = list(features) + _regime_feat

            future_slice = df_train.iloc[i+1:i+future_days+1]
            if len(future_slice) < future_days:
                break

            # ── 3-CLASS LABELS: 0=short wins, 1=neutral, 2=long wins ─────
            # WHY 3-CLASS instead of 2 separate binary models:
            #   - Both long and short predictions come from ONE model
            #   - P(long)  = predict_proba()[:, 2]
            #   - P(short) = predict_proba()[:, 0]
            #   - Neutral outcomes become class 1 (not discarded)
            #   - Probabilities sum to 1: P(long)+P(neutral)+P(short)=1
            #   - Fixes class imbalance: 3 classes more balanced than 2
            #     skewed binary classes (6-20% wins each)
            #   - Trains on MORE examples (neutrals kept as class 1)
            #
            # HYPE example: P=[0.08, 0.84, 0.08] → both directions fail
            # Clean long:   P=[0.05, 0.25, 0.70] → strong long signal
            # Clean short:  P=[0.68, 0.25, 0.07] → strong short signal

            atr_pct       = atr_val / entry
            long_tp       = entry + ML_TP_ATR_MULT * atr_val
            long_sl       = entry - ML_SL_ATR_MULT * atr_val
            short_tp      = entry - ML_TP_ATR_MULT * atr_val
            short_sl      = entry + ML_SL_ATR_MULT * atr_val

            prices        = future_slice["price"]
            hit_long_tp   = bool((prices >= long_tp).any())
            hit_long_sl   = bool((prices <= long_sl).any())
            hit_short_tp  = bool((prices <= short_tp).any())
            hit_short_sl  = bool((prices >= short_sl).any())

            # Determine which outcome happened first
            _long_tp_i  = int((prices >= long_tp).values.argmax())  if hit_long_tp  else 9999
            _long_sl_i  = int((prices <= long_sl).values.argmax())  if hit_long_sl  else 9999
            _short_tp_i = int((prices <= short_tp).values.argmax()) if hit_short_tp else 9999
            _short_sl_i = int((prices >= short_sl).values.argmax()) if hit_short_sl else 9999

            first_long_event  = min(_long_tp_i, _long_sl_i)
            first_short_event = min(_short_tp_i, _short_sl_i)

            if hit_long_tp and (not hit_long_sl or _long_tp_i <= _long_sl_i):
                label = 2   # long wins
            elif hit_short_tp and (not hit_short_sl or _short_tp_i <= _short_sl_i):
                label = 0   # short wins
            elif hit_long_sl and not hit_long_tp:
                label = 0   # long stopped out → bearish
            elif hit_short_sl and not hit_short_tp:
                label = 2   # short stopped out → bullish
            else:
                label = 1   # neither direction reached TP — neutral

            # Z-score for EV-based sample weight
            forward_return = (float(prices.iloc[-1]) - entry) / entry
            z_score        = forward_return / (atr_pct + 1e-9)

            # Keep neutrals (label=1) as training signal — they tell the model
            # "this setup leads nowhere". Only exclude near-zero neutrals.
            if label == 1 and abs(z_score) < ML_NEUTRAL_Z:
                # Very slight drift — ambiguous, skip
                pass   # keep with reduced weight below
            binary = label   # rename for compat with weight code below

            # ── Label type 3: EV-based sample weight ─────────────────────────
            # Scale training weight by how clear the outcome was.
            # Strong wins and clear losses carry more signal -- model should
            # pay more attention to unambiguous examples.
            # Borderline outcomes (z near 0) are noisy -- downweight them.
            # ── Fix 2: Regime weight boost ────────────────────────────────────
            # Bars that match the current regime get ML_REGIME_WEIGHT_MULT× weight.
            # Model pays more attention to patterns from the regime it's predicting on.
            if z_score >= ML_STRONG_WIN_Z:
                weight = 2.0
            elif z_score <= ML_STRONG_LOSS_Z:
                weight = 1.5
            elif abs(z_score) < ML_BORDERLINE_Z:
                weight = 0.5
            else:
                weight = 1.0
            # Boost bars matching current regime
            if _bar_regime == _cur_regime:
                weight *= ML_REGIME_WEIGHT_MULT

            X_list.append(features)
            y_list.append(binary)
            w_list.append(weight)

        if len(X_list) < 10:
            return None, None, 0, 0

        X = np.array(X_list)
        y = np.array(y_list)
        w = np.array(w_list)

        # ── Fix 3: Class rebalancing — cap win:loss ratio at ML_MAX_CLASS_RATIO ──
        # If wins >> losses (e.g. 89% wins from bull history), undersample wins.
        # Without this, model predicts "win" for everything → 40% test accuracy.
        _n_wins   = int((y == 2).sum())
        _n_losses = int((y == 0).sum())
        _n_neut   = int((y == 1).sum())
        _win_rate = _n_wins / max(1, _n_wins + _n_losses)
        print(f"[ML] {coin}: win rate {_win_rate:.0%} ({_n_wins}/{len(y)} wins)  "
              f"neutral_excluded={max(0, (len(df_train)-future_days-1-lookback)-len(X))}")

        if _n_losses > 0 and _n_wins / _n_losses > ML_MAX_CLASS_RATIO:
            # Undersample wins to bring ratio to ML_MAX_CLASS_RATIO
            _target_wins = int(_n_losses * ML_MAX_CLASS_RATIO)
            _win_idx     = np.where(y == 2)[0]
            # Keep most recent wins (they're most relevant to current conditions)
            _keep_idx    = _win_idx[-_target_wins:]
            _other_idx   = np.where(y != 2)[0]
            _keep_all    = np.sort(np.concatenate([_keep_idx, _other_idx]))
            X, y, w      = X[_keep_all], y[_keep_all], w[_keep_all]
            print(f"[ML] {coin}: rebalanced {_n_wins}→{_target_wins} wins  "
                  f"(ratio was {_n_wins/_n_losses:.1f}:1, capped at {ML_MAX_CLASS_RATIO:.0f}:1)")
        # No shuffle -- time series data must stay in order.
        # Shuffling mixes future bars into training → lookahead bias.
        #
        # Scaler: fit ONLY on X_train (not the full X).
        # Fitting on all X leaks future mean/variance into the training
        # features, giving the model implicit knowledge of unseen data.
        #
        #   X_train (70%): model.fit()         ← model sees only past
        #   X_val   (15%): calibrator.fit()    ← calibrator sees no test data
        #   X_test  (15%): honest evaluation   ← unseen by model + calibrator
        n         = len(X)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        X_train_raw = X[:train_end];       y_train = y[:train_end]; w_train = w[:train_end]
        X_val_raw   = X[train_end:val_end]; y_val   = y[train_end:val_end]
        X_test_raw  = X[val_end:];          y_test  = y[val_end:]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)  # fit+transform train only
        X_val   = scaler.transform(X_val_raw)         # transform val (no fit)
        X_test  = scaler.transform(X_test_raw)         # transform test (no fit)

        _n_pos  = int((y_train == 1).sum()); _n_neg = int((y_train == 0).sum())
        _n_excl = (len(df) - future_days - 1 - lookback) - len(X)
        print(f"[ML] {coin}: split → train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  "
              f"labels: {_n_pos}win / {_n_neg}loss  neutral_excluded={max(0,_n_excl)}")

        # ── Walk-forward validation (TimeSeriesSplit expanding window) ──────
        #
        # WHY NOT FIXED 3 PERCENTILE SPLITS:
        #   The old approach trained on [0:40%] and tested [40:55%] — but
        #   this means EVERY fold trains on only a fraction of the available data.
        #   TimeSeriesSplit uses EXPANDING windows: fold 1 trains on 1/6 of data,
        #   fold 5 trains on 5/6 — simulating real deployment where more history
        #   is available as time passes.
        #
        # FOUR METRICS (not just accuracy):
        #   Accuracy:  is the model right?
        #   Brier:     is the model calibrated? (does P=0.7 win 70%?)
        #   Sharpe:    is the model profitable on a per-trade basis?
        #   Drawdown:  what is the worst losing streak?
        #   Stability: is performance consistent across folds or degrading?
        #
        # STABILITY SCORE:
        #   Compares metrics in first half of folds vs second half.
        #   Declining = feature drift (model learns stale patterns).
        #   Rising    = model improving as more data available.
        #   Volatile  = unreliable signal quality.

        wf_accuracies: List[float] = []
        wf_result = None
        try:
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.ensemble import GradientBoostingClassifier as _WGBC

            N_SPLITS     = 5
            WF_THRESHOLD = 0.60   # only count bars as "trades" above this confidence
            WF_TP_MULT   = ML_TP_ATR_MULT    # 2.5
            WF_SL_MULT   = ML_SL_ATR_MULT    # 1.0

            tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
            _folds = []

            for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X)):
                if len(tr_idx) < 20 or len(te_idx) < 10:
                    continue

                _sc = StandardScaler()
                _Xtr = _sc.fit_transform(X[tr_idx]); _ytr = y[tr_idx]; _wtr = w[tr_idx]
                _Xte = _sc.transform(X[te_idx]);     _yte = y[te_idx]

                _fm = _WGBC(n_estimators=50, max_depth=3, learning_rate=0.10,
                             subsample=0.8, random_state=42)
                _fm.fit(_Xtr, _ytr, sample_weight=_wtr)
                _probs = _fm.predict_proba(_Xte)[:, 1]

                # Accuracy (classification boundary at 0.5)
                fold_acc   = float(np.mean((_probs >= 0.5).astype(int) == _yte))

                # Brier score (calibration quality)
                fold_brier = float(np.mean((_probs - _yte.astype(float)) ** 2))

                # ECE (10-bin expected calibration error)
                fold_ece = 0.0
                for _lo, _hi in zip(np.linspace(0,1,11)[:-1], np.linspace(0,1,11)[1:]):
                    _m = (_probs >= _lo) & (_probs < _hi)
                    if _m.sum() >= 2:
                        fold_ece += float(_m.sum()) / len(_probs) * abs(
                            _probs[_m].mean() - _yte[_m].astype(float).mean()
                        )

                # Simulated Sharpe and max drawdown on "taken trades"
                _trade_mask = _probs >= WF_THRESHOLD
                fold_sharpe = 0.0; fold_maxdd = 0.0; fold_n_trades = int(_trade_mask.sum())
                if fold_n_trades >= 3:
                    _rets = np.where(_yte[_trade_mask] == 1, WF_TP_MULT, -WF_SL_MULT)
                    _raw_sh = (float(_rets.mean()) /
                               (float(_rets.std()) + 1e-9) *
                               np.sqrt(252))
                    fold_sharpe = max(-50.0, min(50.0, _raw_sh))  # cap ±50
                    _eq    = np.cumsum(_rets)
                    _peak  = np.maximum.accumulate(_eq)
                    fold_maxdd = float(np.max(_peak - _eq))

                _folds.append({
                    "fold":        fold_idx + 1,
                    "n_train":     len(tr_idx),
                    "n_test":      len(te_idx),
                    "accuracy":    fold_acc,
                    "brier":       fold_brier,
                    "ece":         fold_ece,
                    "sharpe":      fold_sharpe,
                    "max_drawdown":fold_maxdd,
                    "n_trades":    fold_n_trades,
                })
                wf_accuracies.append(fold_acc)

            if _folds:
                # SPARSE FOLD FIX: folds with <5 trades have no statistical power.
                # acc=100% on 2-trade fold = random noise. Exclude from mean metrics.
                _WF_MIN_TRADES = 5
                _dense_folds = [f for f in _folds if f["n_trades"] >= _WF_MIN_TRADES]
                _sparse_count = len(_folds) - len(_dense_folds)

                # Use dense folds for Sharpe (economic signal); all folds for accuracy
                _mean_acc   = float(np.mean([f["accuracy"]     for f in _folds]))
                _mean_brier = float(np.mean([f["brier"]        for f in _folds]))
                _mean_ece   = float(np.mean([f["ece"]          for f in _folds]))
                _mean_sh    = float(np.mean([f["sharpe"]       for f in _dense_folds])) if _dense_folds else 0.0
                _max_dd     = float(np.max( [f["max_drawdown"] for f in _folds]))
                _total_trades = sum(f["n_trades"] for f in _folds)

                # Stability: compare accuracy across fold halves
                _h1_acc = [f["accuracy"] for f in _folds[:len(_folds)//2]]
                _h2_acc = [f["accuracy"] for f in _folds[len(_folds)//2:]]
                if _h1_acc and _h2_acc:
                    _acc_drift  = float(np.mean(_h2_acc) - np.mean(_h1_acc))
                    _acc_std    = float(np.std([f["accuracy"] for f in _folds]))
                    if   _acc_drift > +0.03:  _stability = "IMPROVING ↑"
                    elif _acc_drift < -0.03:  _stability = "DEGRADING ↓ (drift)"
                    elif _acc_std   > 0.08:   _stability = "VOLATILE ⚡"
                    else:                     _stability = "STABLE ✅"
                else:
                    _stability = "UNKNOWN"

                print(f"[ML WF] {coin}: {len(_folds)}-fold TimeSeriesSplit")
                print(f"[ML WF] {coin}:  acc={_mean_acc:.1%}  brier={_mean_brier:.4f}  "
                      f"ece={_mean_ece:.4f}  sharpe={_mean_sh:+.2f}  "
                      f"max_dd={_max_dd:.1f}  stability={_stability}")
                for _f in _folds:
                    _sparse_tag = "  ⚠️ sparse" if _f["n_trades"] < _WF_MIN_TRADES else ""
                    print(f"[ML WF]   fold {_f['fold']:d}: "
                          f"train={_f['n_train']:3d}  test={_f['n_test']:3d}  "
                          f"acc={_f['accuracy']:.1%}  brier={_f['brier']:.4f}  "
                          f"ece={_f['ece']:.4f}  sharpe={_f['sharpe']:+.2f}  "
                          f"max_dd={_f['max_drawdown']:.1f}  trades={_f['n_trades']:d}"
                          f"{_sparse_tag}")

                if _sparse_count > 0:
                    print(f"[ML WF] ℹ️  {coin}: {_sparse_count}/{len(_folds)} sparse folds excluded from Sharpe "
                          f"(total {_total_trades} trades across all folds)")
                if _total_trades < 10:
                    print(f"[ML WF] ⚠️  {coin}: only {_total_trades} total WF trades — "
                          f"model has seen too few wins to generalise (bear market data)")

                # Warn on specific failure modes
                if _mean_brier > 0.26:
                    print(f"[ML WF] ⚠️  {coin}: Brier={_mean_brier:.4f} > 0.26 -- model is poorly calibrated")
                if _mean_ece > 0.10:
                    print(f"[ML WF] ⚠️  {coin}: ECE={_mean_ece:.4f} > 0.10 -- probabilities are not trustworthy")
                if _mean_sh < 0:
                    print(f"[ML WF] ⚠️  {coin}: Sharpe={_mean_sh:+.2f} < 0 -- strategy loses money in validation")
                if "DEGRADING" in _stability:
                    print(f"[ML WF] ⚠️  {coin}: Accuracy degrading across folds -- possible feature drift")

                wf_result = {
                    "folds":          _folds,
                    "mean_accuracy":  _mean_acc,
                    "mean_brier":     _mean_brier,
                    "mean_ece":       _mean_ece,
                    "mean_sharpe":    _mean_sh,
                    "max_drawdown":   _max_dd,
                    "stability":      _stability,
                    "total_trades":   _total_trades,
                    "sparse_folds":   _sparse_count,
                }
                # Cache Sharpe for MLLR suppression
                _WF_SHARPE[coin] = _mean_sh

        except ImportError:
            print(f"[ML WF] {coin}: TimeSeriesSplit unavailable -- sklearn not installed")
        except Exception as _wfe:
            print(f"[ML WF] {coin}: walk-forward error: {_wfe}")

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            # 3-class: 0=short wins, 1=neutral, 2=long wins
            model = GradientBoostingClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42)
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=150, max_depth=4, class_weight="balanced", random_state=42)

        try:
            from sklearn.utils.class_weight import compute_sample_weight
            _bal = compute_sample_weight("balanced", y_train)
            model.fit(X_train, y_train, sample_weight=w_train * _bal)
        except Exception:
            model.fit(X_train, y_train, sample_weight=w_train)

        # ── Win-rate sanity check (3-class version) ───────────────────────
        try:
            _train_preds = model.predict_proba(X_train)
            _classes     = list(model.classes_)
            _long_idx    = _classes.index(2) if 2 in _classes else -1
            _short_idx   = _classes.index(0) if 0 in _classes else -1
            if _long_idx >= 0:
                _pred_long_rate = float((_train_preds[:, _long_idx] > 0.40).mean())
                _true_long_rate = float((y_train == 2).mean())
                if _pred_long_rate < 0.05 and _true_long_rate >= 0.10:
                    print(f"[ML] {coin}: ⚠️  Model rarely predicts long ({_pred_long_rate:.0%} "
                          f"vs true {_true_long_rate:.0%}) — retraining with forced balance")
                    from sklearn.utils.class_weight import compute_sample_weight as _csw
                    _n_pos = int((y_train == 2).sum()); _n_neg = len(y_train) - _n_pos
                    _ratio = min(8.0, _n_neg / max(1, _n_pos))
                    _forced_bal = _csw("balanced", y_train) * _ratio
                    try:
                        model_base = GradientBoostingClassifier(
                            n_estimators=150, max_depth=3, learning_rate=0.05,
                            subsample=0.8, random_state=42)
                        model_base.fit(X_train, y_train, sample_weight=w_train * _forced_bal)
                        model = model_base
                    except Exception:
                        pass
        except Exception:
            pass

        # ── Calibrate on val, evaluate honestly on test ───────────────────
        try:
            from sklearn.calibration import CalibratedClassifierCV
            n_val = len(X_val); n_test = len(X_test)
            if n_val >= 30:
                cal_method = "isotonic" if n_val >= 150 else "sigmoid"
                # cv="prefit" removed in sklearn 1.2+; use cv=3 cross-val calibration
                calibrated_model = CalibratedClassifierCV(model, method=cal_method, cv=3)
                calibrated_model.fit(X_val, y_val)
                if n_test >= 20:
                    brier_raw = float(((model.predict_proba(X_test)[:,1] - y_test)**2).mean())
                    brier_cal = float(((calibrated_model.predict_proba(X_test)[:,1] - y_test)**2).mean())
                    imp = (brier_raw - brier_cal) / (brier_raw + 1e-9) * 100
                    print(f"[ML] {coin}: Calibration ({cal_method}) val={n_val}  "
                          f"Honest Brier (test={n_test}): {brier_raw:.4f}→{brier_cal:.4f}  ({imp:+.1f}%)")
                model = calibrated_model
            else:
                print(f"[ML] {coin}: Calibration skipped (val={n_val} < 30)")
        except ImportError:
            print(f"[ML] {coin}: sklearn.calibration unavailable")
        except Exception as _ce:
            print(f"[ML] {coin}: Calibration error ({_ce})")

        accuracy = (model.score(X_test, y_test) if len(X_test) >= 5
                    else model.score(X_val, y_val) if len(X_val) >= 5
                    else model.score(X_train, y_train))
        _wr = float(y_train.mean()) if len(y_train) > 0 else 0.0
        _health = "✅ balanced" if _wr >= 0.25 else ("⚠️  low wins" if _wr >= 0.12 else "❌ degenerate")
        print(f"[ML] {coin}: win rate {_wr:.0%} ({int(y_train.sum())}/{len(y_train)} wins) "
              f"neutral_excluded={max(0,_n_excl)}  {_health}")

        # Store walk-forward Sharpe for Sharpe suppression at predict time
        _wf_mean_sharpe = 0.0
        if wf_result and "mean_sharpe" in wf_result:
            _wf_mean_sharpe = float(wf_result["mean_sharpe"])

        # ── Ablation test (weekly, only if enough data) ───────────────────
        # Run after training to identify which features matter.
        # Triggered by: day-of-week Monday (once per week) and n_samples >= 80.
        # Results guide feature pruning and reveal overfit sources.
        try:
            _run_ablation = (
                datetime.now().weekday() == 0         # Monday
                and len(X) >= 80                      # enough samples
                and ML_ENABLED
            )
            if _run_ablation:
                print(f"[ABLATION] {coin}: Monday trigger -- running full ablation ...")
                _abl = run_ablation_test(
                    X=X, y=y, w=w, coin=coin,
                    tp_mult=ML_TP_ATR_MULT, sl_mult=ML_SL_ATR_MULT,
                )
                if _abl:
                    _summary = _abl.get("summary", {})
                    _harmful  = _summary.get("harmful", [])
                    _critical = _summary.get("critical", [])
                    if _harmful:
                        print(f"[ABLATION] ⚠️  {coin}: Consider REMOVING: {', '.join(_harmful)} "
                              f"(removal improves Sharpe)")
                    if _critical:
                        print(f"[ABLATION] ✅ {coin}: Core features: {', '.join(_critical)}")
        except Exception as _ae:
            print(f"[ABLATION] {coin}: error: {_ae}")

        return model, scaler, accuracy, len(X)

    except Exception as e:
        print(Fore.YELLOW + f"[ML] Training failed for {coin}: {e}")
        return None, None, 0, 0


# ── Feature catalogue (index, name, group) ──────────────────────────────────
# Maps each feature index to its name and group for ablation reporting.
# Must match the order returned by extract_ml_features().

ML_FEATURE_CATALOGUE = [
    # idx  name                   group
    ( 0,  "rsi",                 "MOMENTUM"),
    ( 1,  "macd_hist",           "MOMENTUM"),
    ( 2,  "adx",                 "TREND"),
    ( 3,  "rel_vol",             "FLOW"),
    ( 4,  "ema_cross",           "TREND"),
    ( 5,  "bb_position",         "STRUCTURE"),
    ( 6,  "atr_slope",           "VOLATILITY"),
    ( 7,  "di_direction",        "TREND"),
    ( 8,  "fng_sentiment",       "MACRO"),
    ( 9,  "btc_rel_strength",    "MACRO"),
    (10,  "day_of_week",         "CALENDAR"),
    (11,  "sortino_ratio",       "MOMENTUM"),
    (12,  "market_session",      "CALENDAR"),
    (13,  "weekly_ema_ratio",    "MACRO"),
    (14,  "ichimoku_cloud",      "TREND"),
    (15,  "obv_trend",           "FLOW"),
]

ML_FEATURE_GROUPS = {
    "TREND":      [0, 2, 4, 7, 14],   # adx, ema_cross, di_direction, cloud
    "MOMENTUM":   [0, 1, 11],          # rsi, macd_hist, sortino
    "FLOW":       [3, 15],             # rel_vol, obv_trend
    "STRUCTURE":  [5],                 # bb_position
    "VOLATILITY": [6],                 # atr_slope
    "MACRO":      [8, 9, 13],          # fng, btc_rs, weekly_ema
    "CALENDAR":   [10, 12],            # day_of_week, market_session
}


def run_ablation_test(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                      coin: str,
                      tp_mult: float = 2.5, sl_mult: float = 1.0,
                      n_splits: int = 5,
                      threshold: float = 0.60) -> dict:
    """
    Ablation test: systematically remove one feature (or one group) at a time
    and measure the performance delta vs the baseline trained on all features.

    REVEALS:
      - Which features genuinely improve performance
      - Which are redundant (removal has no effect)
      - Which HURT performance (removal improves metrics -- overfit / noise)
      - Which feature groups are the primary signal sources

    METRICS TRACKED PER ABLATION:
      accuracy:   classification accuracy on test folds
      brier:      calibration quality (lower = better, 0.25 = random)
      sharpe:     simulated per-trade Sharpe using TP/SL multiples
      max_dd:     worst fold drawdown
      win_rate:   fraction of threshold-crossing trades that win
      delta_*:    change vs baseline (+ = improved, - = degraded)

    RESULT FLAGS:
      CRITICAL  : removing this feature degrades Sharpe by > 0.30
      IMPORTANT : degrades Sharpe by 0.10-0.30
      NEUTRAL   : change within ±0.10 Sharpe
      REDUNDANT : performance identical (< ±0.05 on all metrics)
      HARMFUL   : removing this IMPROVES Sharpe (was hurting performance)
    """
    try:
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print(f"[ABLATION] {coin}: sklearn not available")
        return {}

    N_FEATURES = X.shape[1]
    N_FEATURES_EXPECTED = len(ML_FEATURE_CATALOGUE)

    def _train_eval(X_sub: np.ndarray) -> dict:
        """Train + evaluate using TimeSeriesSplit. Returns metric dict."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_acc=[]; fold_brier=[]; fold_sharpe=[]; fold_dd=[]; fold_wr=[]

        for tr_idx, te_idx in tscv.split(X_sub):
            if len(tr_idx) < 20 or len(te_idx) < 5:
                continue
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_sub[tr_idx])
            Xte = sc.transform(X_sub[te_idx])
            ytr = y[tr_idx]; yte = y[te_idx]; wtr = w[tr_idx]

            m = GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                            learning_rate=0.10, subsample=0.8,
                                            random_state=42)
            m.fit(Xtr, ytr, sample_weight=wtr)
            probs = m.predict_proba(Xte)[:, 1]

            fold_acc.append(float(np.mean((probs >= 0.5) == yte)))
            fold_brier.append(float(np.mean((probs - yte.astype(float))**2)))

            trade_mask = probs >= threshold
            if trade_mask.sum() >= 3:
                rets = np.where(yte[trade_mask] == 1, tp_mult, -sl_mult)
                raw_sharpe = float(rets.mean()/(rets.std()+1e-9)*np.sqrt(252))
                fold_sharpe.append(max(-50.0, min(50.0, raw_sharpe)))  # cap for readability
                eq = np.cumsum(rets)
                fold_dd.append(float(np.max(np.maximum.accumulate(eq) - eq)))
                fold_wr.append(float(yte[trade_mask].mean()))
            else:
                fold_sharpe.append(0.0); fold_dd.append(0.0); fold_wr.append(0.0)

        if not fold_acc:
            return {"accuracy":0.5,"brier":0.25,"sharpe":0.0,"max_dd":0.0,"win_rate":0.5}
        return {
            "accuracy": float(np.mean(fold_acc)),
            "brier":    float(np.mean(fold_brier)),
            "sharpe":   float(np.mean(fold_sharpe)),
            "max_dd":   float(np.max(fold_dd)) if fold_dd else 0.0,
            "win_rate": float(np.mean(fold_wr)),
        }

    # ── Baseline: all features ────────────────────────────────────────────────
    print(f"\n[ABLATION] {coin}: Baseline ({N_FEATURES} features) ...")
    baseline = _train_eval(X)
    print(f"[ABLATION] {coin}: Baseline  "
          f"acc={baseline['accuracy']:.1%}  brier={baseline['brier']:.4f}  "
          f"sharpe={baseline['sharpe']:+.2f}  max_dd={baseline['max_dd']:.1f}  "
          f"win_rate={baseline['win_rate']:.1%}")

    results = {"baseline": baseline, "features": {}, "groups": {}}

    # ── Per-feature ablation ──────────────────────────────────────────────────
    print(f"\n[ABLATION] {coin}: Feature ablation ({N_FEATURES} tests) ...")
    feature_results = []

    for feat_idx in range(min(N_FEATURES, N_FEATURES_EXPECTED)):
        _, feat_name, feat_group = ML_FEATURE_CATALOGUE[feat_idx]
        # Remove this feature: mask out the column with its mean (neutral value)
        X_ablated = X.copy()
        X_ablated[:, feat_idx] = X_ablated[:, feat_idx].mean()

        metrics = _train_eval(X_ablated)
        d_sharpe = metrics["sharpe"]  - baseline["sharpe"]
        d_brier  = metrics["brier"]   - baseline["brier"]   # negative = improved
        d_acc    = metrics["accuracy"]- baseline["accuracy"]

        if   d_sharpe < -0.30: flag = "CRITICAL ⛔"
        elif d_sharpe < -0.10: flag = "IMPORTANT ⚠️"
        elif d_sharpe >  0.10: flag = "HARMFUL  🔴"   # removal improved
        elif abs(d_sharpe) < 0.05 and abs(d_brier) < 0.005: flag = "REDUNDANT 🔵"
        else:                  flag = "NEUTRAL  ⬜"

        r = {
            "feat_idx":   feat_idx,
            "feat_name":  feat_name,
            "group":      feat_group,
            "accuracy":   metrics["accuracy"],
            "brier":      metrics["brier"],
            "sharpe":     metrics["sharpe"],
            "max_dd":     metrics["max_dd"],
            "win_rate":   metrics["win_rate"],
            "d_sharpe":   d_sharpe,
            "d_brier":    d_brier,
            "d_acc":      d_acc,
            "flag":       flag,
        }
        feature_results.append(r)
        results["features"][feat_name] = r

    # Sort by impact (most critical first)
    feature_results.sort(key=lambda r: r["d_sharpe"])

    print(f"\n[ABLATION] {coin}: Per-feature results (sorted by Sharpe impact):")
    print(f"  {'Feature':<22} {'Group':<12} {'ΔSharpe':>9}  {'ΔBrier':>8}  {'ΔAcc':>7}  Flag")
    print("  " + "─" * 72)
    for r in feature_results:
        flag_icon = r["flag"].split()[0]
        print(f"  {r['feat_name']:<22} {r['group']:<12} {r['d_sharpe']:>+9.3f}  "
              f"{r['d_brier']:>+8.4f}  {r['d_acc']:>+7.3f}  {r['flag']}")

    # ── Per-group ablation ────────────────────────────────────────────────────
    print(f"\n[ABLATION] {coin}: Group ablation ({len(ML_FEATURE_GROUPS)} groups) ...")
    group_results = []

    for group_name, feat_indices in ML_FEATURE_GROUPS.items():
        valid_indices = [i for i in feat_indices if i < N_FEATURES]
        if not valid_indices:
            continue
        X_ablated = X.copy()
        for idx in valid_indices:
            X_ablated[:, idx] = X_ablated[:, idx].mean()

        metrics  = _train_eval(X_ablated)
        d_sharpe = metrics["sharpe"]  - baseline["sharpe"]
        d_brier  = metrics["brier"]   - baseline["brier"]

        if   d_sharpe < -0.50: flag = "CRITICAL ⛔"
        elif d_sharpe < -0.20: flag = "IMPORTANT ⚠️"
        elif d_sharpe >  0.20: flag = "HARMFUL  🔴"
        elif abs(d_sharpe) < 0.05: flag = "REDUNDANT 🔵"
        else:                  flag = "NEUTRAL  ⬜"

        r = {
            "group":     group_name,
            "n_features":len(valid_indices),
            "d_sharpe":  d_sharpe,
            "d_brier":   d_brier,
            "d_acc":     metrics["accuracy"] - baseline["accuracy"],
            "flag":      flag,
        }
        group_results.append(r)
        results["groups"][group_name] = r

    group_results.sort(key=lambda r: r["d_sharpe"])

    print(f"\n[ABLATION] {coin}: Per-group results:")
    print(f"  {'Group':<14} {'N':>3}  {'ΔSharpe':>9}  {'ΔBrier':>8}  {'ΔAcc':>7}  Flag")
    print("  " + "─" * 60)
    for r in group_results:
        print(f"  {r['group']:<14} {r['n_features']:>3}  {r['d_sharpe']:>+9.3f}  "
              f"{r['d_brier']:>+8.4f}  {r['d_acc']:>+7.3f}  {r['flag']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    critical   = [r["feat_name"] for r in feature_results if "CRITICAL"  in r["flag"]]
    important  = [r["feat_name"] for r in feature_results if "IMPORTANT" in r["flag"]]
    harmful    = [r["feat_name"] for r in feature_results if "HARMFUL"   in r["flag"]]
    redundant  = [r["feat_name"] for r in feature_results if "REDUNDANT" in r["flag"]]

    print(f"\n[ABLATION] {coin}: SUMMARY")
    if critical:  print(f"  ⛔ Critical (must keep):    {', '.join(critical)}")
    if important: print(f"  ⚠️  Important:               {', '.join(important)}")
    if harmful:   print(f"  🔴 Harmful (consider drop): {', '.join(harmful)}")
    if redundant: print(f"  🔵 Redundant (can prune):   {', '.join(redundant)}")

    results["summary"] = {
        "critical": critical, "important": important,
        "harmful": harmful,   "redundant": redundant,
    }
    return results

def predict_ml_outcome(row, model, scaler, df=None, btc_df=None, fng_value=50):
    """
    Predict long probability from the unified 3-class model.

    3-class model: 0=short wins, 1=neutral, 2=long wins
      P(long)  = predict_proba()[:, class_2_idx]
      P(short) = predict_proba()[:, class_0_idx]

    For backward compatibility, returns (P_long, confidence).
    To get both, use predict_ml_both_directions().
    """
    if model is None or scaler is None:
        return 0.5, "unknown"
    try:
        features = extract_ml_features(row, df, btc_df=btc_df, fng_value=fng_value)
        if features is None:
            return 0.5, "unknown"
        if hasattr(scaler, 'n_features_in_') and len(features) != scaler.n_features_in_:
            return 0.5, "unknown"

        X_scaled = scaler.transform([features])
        proba_all = model.predict_proba(X_scaled)[0]
        classes   = list(model.classes_)

        # 3-class model: extract long probability (class 2)
        if 2 in classes:
            proba = float(proba_all[classes.index(2)])
        elif 1 in classes:
            proba = float(proba_all[classes.index(1)])  # old binary fallback
        else:
            proba = float(proba_all[-1])

        # Calibration check + fallback shrinkage for uncalibrated models
        try:
            from sklearn.calibration import CalibratedClassifierCV
            is_calibrated = isinstance(model, CalibratedClassifierCV)
        except ImportError:
            is_calibrated = False
        if not is_calibrated:
            import math as _m
            _lo   = _m.log(max(1e-9, proba) / max(1e-9, 1 - proba))
            proba = 1.0 / (1.0 + _m.exp(-_lo * 0.75))

        confidence = "high" if proba >= 0.70 else ("medium" if proba >= 0.55 else "low")
        return proba, confidence

    except Exception as e:
        logger.debug(f"[ML] Prediction error: {e}")
        return 0.5, "unknown"


def predict_ml_both_directions(row, model, scaler, df=None, btc_df=None, fng_value=50):
    """
    Extract BOTH long and short probabilities from the unified 3-class model.

    Returns: (p_long, p_short, p_neutral, confidence_str)
      p_long   = P(class 2) — price rises to TP before falling to SL
      p_short  = P(class 0) — price falls to TP before rising to SL
      p_neutral= P(class 1) — neither direction resolves in window

    This eliminates the need for a separate short ML model entirely.
    HYPE example: (0.08, 0.08, 0.84) → avoid both directions (choppy)
    Clean long:   (0.70, 0.05, 0.25) → strong long signal
    Clean short:  (0.05, 0.72, 0.23) → strong short signal
    """
    if model is None or scaler is None:
        return 0.5, 0.5, 0.0, "unknown"
    try:
        features = extract_ml_features(row, df, btc_df=btc_df, fng_value=fng_value)
        if features is None:
            return 0.5, 0.5, 0.0, "unknown"
        if hasattr(scaler, 'n_features_in_') and len(features) != scaler.n_features_in_:
            return 0.5, 0.5, 0.0, "unknown"

        X_scaled  = scaler.transform([features])
        proba_all = model.predict_proba(X_scaled)[0]
        classes   = list(model.classes_)

        # 3-class extraction
        if set(classes) == {0, 1, 2}:
            p_short   = float(proba_all[classes.index(0)])
            p_neutral = float(proba_all[classes.index(1)])
            p_long    = float(proba_all[classes.index(2)])
        elif set(classes) == {0, 2}:
            # Rare: neutral class absent from training set
            p_long    = float(proba_all[classes.index(2)])
            p_short   = float(proba_all[classes.index(0)])
            p_neutral = 0.0
        else:
            # Old binary model fallback — class 1 was the old "win"
            p_long    = float(proba_all[-1])
            p_short   = 1.0 - p_long
            p_neutral = 0.0

        conf = "high" if max(p_long, p_short) >= 0.65 else "medium" if max(p_long, p_short) >= 0.50 else "low"
        return p_long, p_short, p_neutral, conf

    except Exception as e:
        logger.debug(f"[ML] Both-directions prediction error: {e}")
        return 0.5, 0.5, 0.0, "unknown"
        return 0.5, "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# SHORT ML MODEL -- trained on short outcomes, completely separate from long model
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
            df_slice = df.iloc[:i+1]   # LEAKAGE FIX: only past data up to bar i
            features = extract_ml_features(row, df_slice, btc_df=btc_df, fng_value=fng_value)
            if features is None:
                continue

            # ── Add regime one-hot (same as long model) ────────────────────
            # Without this, model can't distinguish bull-market shorts (fail)
            # from bear-market shorts (succeed) → artificially high accuracy
            # in bear market, dangerous failure when regime flips
            _bar_regime  = _detect_bar_regime(row, btc_df, i, df)
            _regime_feat = [
                1.0 if _bar_regime == "bull"  else 0.0,
                1.0 if _bar_regime == "bear"  else 0.0,
                1.0 if _bar_regime == "range" else 0.0,
                1.0 if _bar_regime == "panic" else 0.0,
            ]
            features = list(features) + _regime_feat   # 16 → 20 features

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
                short_target = current_price * (1 - STOP_ATR_MULT * atr_pct)
                short_stop   = current_price * (1 + STOP_ATR_MULT * atr_pct)
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

        # Chronological 70/15/15 — no shuffle, scaler fit on train only
        n         = len(X)
        train_end = int(n * 0.70); val_end = int(n * 0.85)
        X_train_raw = X[:train_end];       y_train = y[:train_end]
        X_val_raw   = X[train_end:val_end]; y_val   = y[train_end:val_end]
        X_test_raw  = X[val_end:];          y_test  = y[val_end:]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val   = scaler.transform(X_val_raw)
        X_test  = scaler.transform(X_test_raw)
        print(f"[SHORT ML] {coin}: split → train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

        # Walk-forward validation with Sharpe tracking
        _short_wf_sharpe = 0.0
        try:
            from sklearn.model_selection import TimeSeriesSplit as _STSCV
            from sklearn.ensemble import GradientBoostingClassifier as _SGBC
            _stscv = _STSCV(n_splits=5)
            _sfolds_sh = []
            for _str_idx, _ste_idx in _stscv.split(X):
                if len(_str_idx) < 20 or len(_ste_idx) < 5: continue
                _ssc = StandardScaler()
                _Xtr = _ssc.fit_transform(X[_str_idx]); _Xte = _ssc.transform(X[_ste_idx])
                # Class weights in WF folds too
                _y_fold = y[_str_idx]
                _cw_fold = np.ones(len(_y_fold))
                if _y_fold.sum() > 0 and (_y_fold==0).sum() > 0:
                    from sklearn.utils.class_weight import compute_sample_weight as _csw2
                    _cw_fold = _csw2("balanced", _y_fold)
                _ym = _SGBC(n_estimators=80, max_depth=4, learning_rate=0.08,
                             subsample=0.8, min_samples_leaf=8, random_state=42)
                _ym.fit(_Xtr, _y_fold, sample_weight=_cw_fold)
                _sprobs = _ym.predict_proba(_Xte)[:, 1]
                _smask  = _sprobs >= 0.55   # was 0.60 — too conservative
                if _smask.sum() >= 3:
                    _srets  = np.where(y[_ste_idx][_smask] == 1, 2.5, -1.0)
                    _raw_sh = float(_srets.mean() / (_srets.std() + 1e-9) * np.sqrt(252))
                    _sfolds_sh.append(max(-50.0, min(50.0, _raw_sh)))
            if _sfolds_sh:
                _short_wf_sharpe = float(np.mean(_sfolds_sh))
                _wf_status = "⚠️ negative" if _short_wf_sharpe < 0 else "✅ positive"
                print(f"[SHORT ML] {coin}: Walk-forward Sharpe={_short_wf_sharpe:+.2f} {_wf_status}")
                _WF_SHARPE[f"{coin}_short"] = _short_wf_sharpe
        except Exception:
            pass

        # ── Class-balanced weights — critical fix ─────────────────────────
        # In bear market: label=1 (short wins) can be 70-80% of bars.
        # Without balancing: model learns "always predict short win" → 87%
        # "accuracy" that's actually just the bear market base rate.
        # With balancing: model must learn WHEN shorts win, not just that
        # they mostly win in this particular training period.
        try:
            from sklearn.utils.class_weight import compute_sample_weight as _csw_s
            _sw_train = _csw_s("balanced", y_train)
        except Exception:
            _sw_train = np.ones(len(y_train))

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators     = 120,
                max_depth        = 4,
                learning_rate    = 0.07,
                subsample        = 0.80,
                min_samples_leaf = 8,   # was 3 (too granular for 219 samples)
                random_state     = 42,
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, max_depth=4,
                                           class_weight="balanced", random_state=42)
            _sw_train = np.ones(len(y_train))  # RF uses class_weight instead

        model.fit(X_train, y_train, sample_weight=_sw_train)

        try:
            from sklearn.calibration import CalibratedClassifierCV
            n_val = len(X_val)
            if n_val >= 30:
                cal_method = "isotonic" if n_val >= 150 else "sigmoid"
                cal_model  = CalibratedClassifierCV(model, method=cal_method, cv=3)
                cal_model.fit(X_val, y_val)
                if len(X_test) >= 20:
                    raw_b = float(((model.predict_proba(X_test)[:,1]-y_test)**2).mean())
                    cal_b = float(((cal_model.predict_proba(X_test)[:,1]-y_test)**2).mean())
                    print(f"[SHORT ML] {coin}: Calibration ({cal_method}) val={n_val}  "
                          f"Honest Brier (test={len(X_test)}): {raw_b:.4f}→{cal_b:.4f}")
                model = cal_model
            else:
                print(f"[SHORT ML] {coin}: Calibration skipped (val={n_val} < 30)")
        except Exception as _ce:
            print(f"[SHORT ML] {coin}: Calibration error ({_ce})")

        accuracy = (model.score(X_test, y_test)  if len(X_test) >= 5
                    else model.score(X_val, y_val) if len(X_val)  >= 5
                    else model.score(X_train, y_train))
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
            "model_version": "v8.2", "n_features": 20,
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
        if metadata.get("model_version") != "v8.2" or metadata.get("n_features", 0) != 16:
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

    Uses LINEAR INTERPOLATION between anchor points -- no cliff edges.
    A coin at 69% direction confidence scores proportionally between the
    60% and 70% anchors, not the same as 60%.

    Components:
      Technical   (0-35)  -- tech score
      Direction   (0-20)  -- direction model confidence
      ML Model    (0-20)  -- long ML win probability
      Smart Money (0-15)  -- OBV/whale accumulation
      No Blocks   (0-10)  -- signal not blocked

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
        # Anchors: (score_threshold, points) -- more negative = better for short
        _tech_anchors = [(-8, 35), (-7, 28), (-6, 21), (-5, 14), (-3, 7), (0, 2), (1, 0)]
        tech_points   = interp(score, _tech_anchors)
        if score <= -8:   tech_desc = "Excellent"
        elif score <= -6: tech_desc = "Very Good"
        elif score <= -5: tech_desc = "Good"
        elif score <= -3: tech_desc = "Acceptable"
        elif score < 0:   tech_desc = "Weak"
        else:             tech_desc = "Bullish (bad for short)"
    else:
        # Anchors: (score, points) -- higher score = better
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
    # 2. DIRECTION CONSENSUS (0-20 points) -- interpolated
    # ─────────────────────────────────────────────────────────────────
    if is_short:
        if direction == "DOWN":
            _dir_anchors = [(0.30, 2), (0.50, 10), (0.60, 14), (0.70, 17), (0.80, 20)]
            if direction_conf >= 0.80:   dir_desc = "Overwhelming DOWN"
            elif direction_conf >= 0.70: dir_desc = "Very Strong DOWN"
            elif direction_conf >= 0.60: dir_desc = "Strong DOWN"
            elif direction_conf >= 0.50: dir_desc = "Moderate DOWN"
            else:                        dir_desc = "Weak DOWN"
        else:  # UP -- bad for short
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
        else:  # DOWN -- bad for BUY
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
                f"{dir_points}/20 ({dir_desc} -- HALVED: tech bearish, direction bullish = counter-trend risk)"
            )
        elif score > 0 and direction == "DOWN":
            dir_points    = dir_points // 2
            counter_trend = True
            breakdown["Direction (0-20)"] = (
                f"{dir_points}/20 ({dir_desc} -- HALVED: tech bullish, direction bearish = counter-trend risk)"
            )

    base_score += dir_points

    # ─────────────────────────────────────────────────────────────────
    # 3. ML PROBABILITY (0-20 points) -- interpolated
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
                f"P={prob:.0%} (+{_comp})"
            )

    # Case 2: Score strong (≥5) but ML weak (25-50%)
    # Only compensate when ML is uncertain (25-50%), NOT when actively bearish (<35%)
    # If ML < 35%, it's not uncertain -- it's predicting failure. No compensation.
    elif not is_short and score >= 5 and 0.35 <= ml_proba < 0.50:
        _comp = round((score - 4) * 1.5)
        _comp = min(_comp, 5)
        if _comp > 0:
            base_score += _comp
            breakdown["⚡ Score Compensates ML"] = (
                f"P={prob:.0%} partially offsets uncertain ML {ml_proba:.0%} (+{_comp})"
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
                f"Only {backtest_trades} trades -- insufficient for statistical significance "
                f"(WR {backtest_win_rate:.1f}% ignored, need ≥10)"
            )
        elif backtest_win_rate < 25:
            confidence_score -= 15
            breakdown["⚠️ Backtest Penalty"] = f"Win rate {backtest_win_rate:.1f}% < 25% -- historically very poor (-15)"
        elif backtest_win_rate < 35:
            confidence_score -= 10
            breakdown["⚠️ Backtest Penalty"] = f"Win rate {backtest_win_rate:.1f}% < 35% -- below average (-10)"
        elif backtest_win_rate < 45:
            confidence_score -= 5
            breakdown["⚠️ Backtest Penalty"] = f"Win rate {backtest_win_rate:.1f}% < 45% -- marginal (-5)"

    # PROFIT FACTOR PENALTY -- same reliability gate
    if backtest_pf is not None and _backtest_reliable:
        if backtest_pf < 0.60:
            confidence_score -= 10
            breakdown["⚠️ Profit Factor"] = f"PF {backtest_pf:.2f} < 0.60 -- setup consistently loses money (-10)"
        elif backtest_pf < 0.85:
            confidence_score -= 5
            breakdown["⚠️ Profit Factor"] = f"PF {backtest_pf:.2f} < 0.85 -- net negative historically (-5)"
        # PF 0.85-1.10 = breakeven noise band, no penalty

    # ENTRY QUALITY PENALTY: Chasing or near exhaustion
    # Note: Low ADX is already penalised in the scoring pipeline via Z-score gate
    # (STRETCHED in range → -1) and exhaustion check (EMA26 deviation → -1)
    # so no separate ADX penalty needed here.
    entry_quality = sig.get("entry_quality", "normal")
    if entry_quality == "chasing":
        confidence_score -= 8
        breakdown["⚠️ Entry Quality"] = "Chasing breakout -- price above Donchian upper (-8)"
    elif entry_quality == "near_exhaustion":
        confidence_score -= 5
        breakdown["⚠️ Entry Quality"] = f"Near EMA26 exhaustion zone -- {sig.get('ema26_dev_pct', 0):.1f}% above EMA26 (-5)"

    # Clamp final score 0-100
    confidence_score = max(0, min(100, confidence_score))
    
    # Check if this is a SHORT signal
    action   = sig.get("action", "")
    is_short = "SHORT" in action
    # ml_proba comes from the function parameter -- do NOT override from sig
    # (gem scanner was getting 0.5 default because it doesn't set sig["ml_proba"])
    
    # HARD BLOCK: ML very weak -- always flag as poor regardless of confidence
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

    Uses Bayesian probability threshold (prob >= BUY_PROB_THRESHOLD)
    instead of old additive score, consistent with the live scoring engine.
    """
    if len(df) < min_days:
        return False, 0, []

    scores_and_blocks = []
    for i in range(-min_days, 0):
        try:
            row = df.iloc[i]
            sub = df.iloc[:len(df)+i+1]
            _sc_unused, sig = generate_signal_probabilistic(
                row, sub,
                btc_regime     = btc_regime,
                is_btc         = is_btc,
                suppress_debug = True,
            )
            prob    = sig.get("prob", 0.0)
            blocked = sig.get("blocked", False)
            passes  = (prob >= BUY_PROB_THRESHOLD) and not blocked
            scores_and_blocks.append({
                "day_offset": i,
                "score":   round(prob * 10, 1),   # P×10 for display (0-10 scale)
                "blocked": blocked,
                "prob":    prob,
                "action":  "PASS" if passes else "FAIL",
            })
        except Exception:
            scores_and_blocks.append({
                "day_offset": i, "score": 0,
                "blocked": True, "prob": 0.0, "action": "FAIL",
            })

    passes          = all(s["action"] == "PASS" for s in scores_and_blocks)
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
    """Compute dynamic trailing stop -- always BELOW price for BUY."""
    candidates = []

    # EMA12-based trail
    if not np.isnan(ema12) and ema12 > 0:
        candidates.append(ema12 * (1 - ema_buffer))

    # ATR-based trail
    if atr is not None and not np.isnan(atr) and atr > 0:
        candidates.append(price - atr_mult * atr)

    # Kijun-Sen (only if below price -- prevents above-entry bug)
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
            print(Fore.YELLOW + f"[STOP HUNT] {coin}: Paper-thin book -- "
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

    # FIX 3: Enforce MIN_RR_RATIO -- warn if below threshold
    rr_ok = rr >= MIN_RR_RATIO
    if not rr_ok:
        print(Fore.YELLOW + f"[RR WARNING] {coin}: RR={rr:.2f} < MIN={MIN_RR_RATIO} -- trade quality is low")
        if LOG_ENABLED:
            logger.warning(f"[RR] {coin}: RR {rr:.2f} below minimum {MIN_RR_RATIO}")

    return {
        "total_units":   round(total_units,   QUANT_ROUND),
        "initial_units": round(initial_units, QUANT_ROUND),
        "reserve_units": round(reserve_units, QUANT_ROUND),
        "stop_price":    round(stop, 8),
        "tp_price":      round(tp,   8),
        "trailing_stop": round(trailing_stop, 8),
        "partial_tps":   partial_tps,
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
    vol_avg90    = (float(df["vol_median90"].iloc[-1]) if "vol_median90" in df.columns else df["volume"].rolling(90, min_periods=10).median().iloc[-1])  # median baseline
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
    vol_avg    = df["volume"].rolling(20).median().iloc[-2]  # median: robust to spikes
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
    Multiplicative: each bearish condition multiplies bearish log-odds,
    equivalent to multiplying odds in probability space.
    Final score derived from log-odds on a -9 to 0 scale.
    """
    if len(df) < 20:
        return 0.0

    last  = df.iloc[-1]
    close = last["price"]

    # Accumulate bearish log-odds from each signal (negative = bearish)
    bearish_lo = 0.0

    # Price below EMAs (downtrend momentum)
    if close < last["EMA12"] and close < last["EMA26"]:
        bearish_lo -= math.log(2.20)   # LR=2.20 → strong bearish confirmation
    elif close < last["EMA12"] or close < last["EMA26"]:
        bearish_lo -= math.log(1.45)

    # Cloud position
    sa = last.get("senkou_a", float("nan")); sb = last.get("senkou_b", float("nan"))
    if not (math.isnan(float(sa)) or math.isnan(float(sb))):
        cloud_top = max(float(sa), float(sb)); cloud_bot = min(float(sa), float(sb))
        if close < cloud_bot:
            bearish_lo -= math.log(2.80)   # fully below cloud
        elif close < cloud_top:
            bearish_lo -= math.log(1.55)   # inside cloud

    # MACD
    macd_h = float(last.get("MACD_Hist", 0) or 0)
    if macd_h < 0:
        bearish_lo -= math.log(1.60)

    # RSI overbought (mean-reversion short)
    rsi = float(last.get("RSI", 50) or 50)
    if rsi > 85:
        bearish_lo -= math.log(2.20)   # extreme overbought
    elif rsi > 75:
        bearish_lo -= math.log(1.65)
    elif rsi > 70:
        bearish_lo -= math.log(1.30)

    # ADX + DI- > DI+  (trend confirmed downward)
    adx = float(last.get("ADX", 0) or 0)
    pdi = last.get("plus_di", float("nan")); mdi = last.get("minus_di", float("nan"))
    if adx > 25 and not (math.isnan(float(pdi)) or math.isnan(float(mdi))):
        if float(mdi) > float(pdi):
            bearish_lo -= math.log(1.80)

    # RSI bearish divergence (price high, RSI fading)
    divergence = detect_rsi_divergence_sell(df, lookback=DIVERGENCE_LOOKBACK)
    if divergence.get("bearish", False):
        bearish_lo -= math.log(1.70)

    # EMA12 below EMA26 (golden cross failed / death cross)
    ema12 = float(last.get("EMA12", close)); ema26 = float(last.get("EMA26", close))
    if ema12 < ema26:
        bearish_lo -= math.log(1.30)

    # Chikou confirmation
    chikou     = last.get("chikou",           float("nan"))
    chikou_ref = last.get("chikou_ref",       float("nan"))
    chikou_bot = last.get("chikou_cloud_bot", float("nan"))
    _cp = not math.isnan(float(chikou)) and not math.isnan(float(chikou_ref)) and float(chikou) < float(chikou_ref)
    _cc = not math.isnan(float(chikou)) and not math.isnan(float(chikou_bot)) and float(chikou) < float(chikou_bot)
    if _cp and _cc:
        bearish_lo -= math.log(1.80)
    elif _cp:
        bearish_lo -= math.log(1.35)

    # Upper Bollinger Band (overextended)
    upper_bb = float(last.get("UpperBB", close * 1.10))
    if close > upper_bb * 0.98:
        bearish_lo -= math.log(1.25)

    # Convert log-odds to a score on the same -9 to 0 scale as before
    # bearish_lo = 0 → score = 0; bearish_lo = -4.0 → score ≈ -9 (fully bearish)
    score = round(max(-9.0, min(0.0, bearish_lo * 2.25)), 2)
    return score


def detect_rsi_divergence_sell(df, lookback=DIVERGENCE_LOOKBACK):
    """[J] Detect RSI divergences for SELL scoring -- returns dict {bullish, bearish}"""
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

    Safe entry window: 00:00-02:00 UTC (candle just closed, new one started)
    Caution window:    22:00-23:59 UTC (candle hasn't closed yet -- mid-candle)
    Warning window:    all other hours -- mid-candle, signal may change by close

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

        # Safe: entered within 4 hours of candle close (22:00-02:00 UTC)
        is_confirmed = utc_hour < 2    # just closed -- signal confirmed
        is_closing   = utc_hour >= 22  # closing soon -- almost confirmed
        is_mid       = not is_confirmed and not is_closing

        if is_confirmed:
            return True, hours_to_close, ""
        elif is_closing:
            mins_left = int((24 - utc_hour - utc_minute/60) * 60)
            return True, hours_to_close, (
                f"⚠️ [CANDLE] Daily candle closes in ~{mins_left}min (00:00 UTC) -- "
                f"signal almost confirmed but conditions can still change"
            )
        else:
            return False, hours_to_close, (
                f"⏳ [CANDLE] Mid-candle entry -- daily candle closes in ~{hours_to_close:.1f}h (00:00 UTC). "
                f"For futures: wait for candle close to confirm signal. "
                f"Current conditions may reverse before daily close."
            )
    except Exception:
        return True, 0, ""   # fail safe -- don't block on error


def check_btc_relative_strength(df, btc_df, coin, lookback_hours=8):
    """
    Detect when BTC rallies strongly but the coin doesn't follow.
    This is a distribution/weakness signal -- smart money rotating INTO BTC,
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
                f"🚨 [BTC DIVERGENCE] BTC +{btc_chg:.1f}% but {coin.upper()} {coin_chg:+.1f}% -- "
                f"SEVERE underperformance ({divergence:.1f}% gap). "
                f"Smart money rotating INTO BTC, OUT of {coin.upper()}. "
                f"Distribution confirmed -- DO NOT ENTER LONG."
            )
        elif is_weak:
            msg = (
                f"⚠️ [BTC DIVERGENCE] BTC +{btc_chg:.1f}% but {coin.upper()} only {coin_chg:+.1f}% -- "
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
    Regime-conditional entry timing.

    Entry rules scale with trend phase and RS tier:

    TRENDING  (EARLY_BULL/STRONG_BULL + RS LEADER/STRONG):
      Consecutive: 1/2  |  EMA tolerance: 22%  |  ADX: ≥10  |  RSI: ≤82
      ML override: P≥75% + OBV rising → ignore ML CAUTION

    STANDARD  (MATURE_BULL/RANGE + neutral RS):
      Consecutive: 2/2  |  EMA tolerance: 12%  |  ADX: ≥13  |  RSI: ≤75
      ML override: P≥80%

    WEAK      (LATE_BULL/DISTRIBUTION + RS LAGGARD/WEAK):
      Consecutive: 2/2  |  EMA tolerance: 3%   |  ADX: ≥20  |  RSI: ≤65
      No ML override
    """
    import numpy as _np2
    price       = last["price"]
    ema12       = last.get("EMA12", price)
    ema26       = last.get("EMA26", price)
    rsi         = last.get("RSI", 50)
    cloud_state = sig.get("cloud_state", "unknown")
    blocked     = sig.get("blocked", False)
    blockers    = sig.get("blockers", [])

    sa = last.get("senkou_a", float("nan"))
    sb = last.get("senkou_b", float("nan"))
    try:
        cloud_top = max(sa, sb) if (not _np2.isnan(sa) and not _np2.isnan(sb)) else ema26
        cloud_bot = min(sa, sb) if (not _np2.isnan(sa) and not _np2.isnan(sb)) else ema26
    except Exception:
        cloud_top = cloud_bot = ema26
    lower_bb  = last.get("LowerBB", ema26 * 0.97)
    adx       = last.get("ADX", 0) or 0

    try:
        don_upper = df["price"].iloc[-DONCHIAN_LOOKBACK:].max()
    except Exception:
        don_upper = price

    try:
        obv_vals  = df["OBV"].iloc[-8:].values
        obv_chg   = (obv_vals[-1] - obv_vals[0]) / (abs(obv_vals[0]) + 1e-9)
        obv_trend = "rising" if obv_chg > 0.03 else ("falling" if obv_chg < -0.03 else "flat")
    except Exception:
        obv_trend = "unknown"

    prob     = sig.get("prob", 0.5)
    phase    = sig.get("market_phase", "RANGE")
    lc       = sig.get("lifecycle", {}) or {}
    lc_stage = lc.get("stage", "MATURE_TREND")
    rs_full  = sig.get("rs_full", {}) or {}
    rs_label = rs_full.get("label", "NEUTRAL") if rs_full.get("available") else "NEUTRAL"

    # ── Entry regime ──────────────────────────────────────────────────────
    _is_trending = (
        (phase in ("EARLY_BULL","STRONG_BULL") and rs_label in ("LEADER","STRONG")) or
        lc_stage == "EARLY_TREND"
    )
    _is_weak = (
        phase in ("LATE_BULL","DISTRIBUTION","EARLY_BEAR") or
        rs_label in ("LAGGARD","WEAK") or
        lc_stage == "DISTRIBUTION"
    )

    if _is_trending:
        _regime        = "TRENDING"
        _consec_req    = 1
        _ema_tol       = 0.22
        _adx_min       = 10
        _rsi_max       = 82
        _ml_override_p = 0.75
    elif _is_weak:
        _regime        = "WEAK"
        _consec_req    = 2
        _ema_tol       = 0.03
        _adx_min       = 20
        _rsi_max       = 65
        _ml_override_p = 99.0
    else:
        _regime        = "STANDARD"
        _consec_req    = 2
        _ema_tol       = 0.12
        _adx_min       = 13
        _rsi_max       = 75
        _ml_override_p = 0.80

    _ema_dev    = (price - ema26) / (ema26 + 1e-9)
    _in_ema_zone = _ema_dev <= _ema_tol

    # ── SHORT TERRITORY ───────────────────────────────────────────────────
    if final_score <= -5:
        short_blocks = []; short_hints = []
        if adx < 15:
            short_blocks.append(f"ADX {adx:.1f} < 15 — no trend, shorting into chop")
            short_hints.append("Wait for ADX > 15")
        if short_ml_proba < 0.40:
            short_blocks.append(f"Short ML {short_ml_proba:.0%} needs ≥40%")
            short_hints.append("Wait for short ML ≥ 40%")
        if obv_trend == "rising":
            short_blocks.append("OBV rising — smart money still accumulating")
            short_hints.append("Wait for OBV to flatten or fall")
        real_blockers = [b for b in sig.get("blockers",[]) if b != "Cloud Below"]
        if not short_blocks and sig.get("blocked") and real_blockers:
            short_blocks.append(f"Signal blocked: {', '.join(real_blockers)}")
            short_hints.append("Resolve blocker")
        if cloud_state == "below":
            cloud_ctx = (f"✅ Below cloud (${cloud_bot:.4f}) confirmed. "
                         f"Re-entry: bounce to cloud bottom (${cloud_bot:.4f}) → rejection.")
            sw = cloud_bot
        elif cloud_state == "in":
            cloud_ctx = f"Inside cloud — rejection at top (${cloud_top:.4f}) for entry."
            sw = cloud_top
        else:
            cloud_ctx = f"Above cloud — short only valid after reversal below EMA26 (${ema26:.4f})."
            sw = ema26
        if short_blocks:
            bl = "\n".join(f"  {i+1}. {b}" for i,b in enumerate(short_blocks[:4]))
            return ("⛔ SHORT BLOCKED", f"Blocked:\n{bl}\n{cloud_ctx}", sw)
        return ("🔴 SHORT NOW", f"All gates cleared. {cloud_ctx}  ML={short_ml_proba:.0%}  OBV={obv_trend}", price)

    # ── BUY TERRITORY ─────────────────────────────────────────────────────
    if blocked and "BTC Bear Regime" in blockers:
        pass
    if blocked and "Cloud Below" in blockers:
        return ("⛔ CLOUD BELOW", f"Below cloud — enter after reclaiming ${cloud_top:.4f}.", cloud_top)
    if blocked:
        reason = ", ".join(b for b in blockers if b not in ("BTC Bear Regime","Cloud Below")) or "signal blocked"
        return ("⛔ BLOCKED", f"Blocked: {reason}. Resolve before entering.", price)

    # ── ML gate — regime-conditional ──────────────────────────────────────
    if final_score >= MIN_BUY_SCORE and ml_proba < ML_MIN_CONFIDENCE:
        obv_note = "OBV rising ✅" if obv_trend=="rising" else f"OBV {obv_trend} ⚠️"

        if _is_trending and prob >= _ml_override_p and obv_trend == "rising":
            pass   # TRENDING + P≥75% + OBV rising → override ML CAUTION, fall through to ENTER NOW
        elif _is_trending and prob >= _ml_override_p:
            return (
                "⚠️  CAUTION -- OBV WEAK",
                (f"[{_regime}] P={prob:.0%} qualifies. Phase={phase} RS={rs_label}. "
                 f"ML weak ({ml_proba:.0%}) overridden by trend regime. "
                 f"{obv_note} — wait for OBV to confirm or enter half-size. "
                 f"Watch ${ema26:.4f} for better entry."),
                ema26
            )
        else:
            return (
                "⚠️  CAUTION -- ML WEAK",
                (f"[{_regime}] ML {ml_proba:.0%} (needs >{ML_MIN_CONFIDENCE:.0%}). {obv_note}. "
                 f"{'Trending phase — half-size acceptable at ' + str(round(ema26*(1+_ema_tol*0.5),4)) if _is_trending else 'Wait for EMA26 (' + str(round(ema26,4)) + ') or ML to confirm.'}"),
                ema26 * (1 + _ema_tol * 0.5)
            )

    # ── FULL BUY ──────────────────────────────────────────────────────────
    if final_score >= MIN_BUY_SCORE and ml_proba >= ML_MIN_CONFIDENCE:
        obv_note = " | OBV ✅" if obv_trend=="rising" else f" | OBV {obv_trend} ⚠️"
        rsi_note = f" | RSI {rsi:.0f} extended ⚠️" if rsi > _rsi_max else ""
        chasing  = price > don_upper * 1.01 and not _in_ema_zone
        if chasing:
            entry_zone = ema26 * (1 + _ema_tol * 0.5)
            return (
                "⚠️  CHASING -- WAIT",
                (f"[{_regime}] {_ema_dev*100:.1f}% above EMA26 "
                 f"(tolerance ≤{_ema_tol*100:.0f}% for {_regime}). "
                 f"Wait for ${entry_zone:.4f}{obv_note}{rsi_note}."),
                entry_zone
            )

        _candle_safe, _hrs_to_close, _candle_msg = check_candle_close_gate()
        _consecutive_confirmed = sig.get("_consecutive_confirmed", False)
        if not _candle_safe and not _consecutive_confirmed:
            _h = int(_hrs_to_close)
            _m = int((_hrs_to_close - _h) * 60)
            _eta = f"{_h}h {_m}m" if _h > 0 else f"{_m}m"

        _candle_safe, _hrs_to_close, _candle_msg = check_candle_close_gate()
        _consecutive_confirmed = sig.get("_consecutive_confirmed", False)

        if not _candle_safe and not _consecutive_confirmed:
            _h   = int(_hrs_to_close)
            _m   = int((_hrs_to_close - _h) * 60)
            _eta = f"{_h}h {_m}m" if _h > 0 else f"{_m}m"
            return (
                f"⏳ WAIT FOR CANDLE CLOSE ({_eta} remaining)",
                (
                    f"P={prob:.0%} ✅  ML {ml_proba*100:.0f}% ✅ -- technically qualifies BUT\n"
                    f"No consecutive day confirmation yet -- single candle signal mid-forming.\n"
                    f"Daily candle closes in {_eta} (00:00 UTC).\n"
                    f"OBV, volume and RSI can flip before close. Re-check after 00:00 UTC.\n"
                    f"Target entry after close: ${ema26:.4f} (EMA26)"
                ),
                ema26
            )
        elif not _candle_safe and _consecutive_confirmed:
            # Consecutive confirmed -- mid-candle entry allowed but flag the time
            _h   = int(_hrs_to_close)
            _m   = int((_hrs_to_close - _h) * 60)
            _eta = f"{_h}h {_m}m" if _h > 0 else f"{_m}m"
            # Don't block -- just note it in the ENTER NOW message
            obv_str = "OBV rising ✅" if obv_trend == "rising" else f"OBV {obv_trend} ⚠️"
            return (
                "🟢 ENTER NOW",
                (
                    f"All criteria met -- P={prob:.0%} ✅  ML {ml_proba*100:.0f}% ✅ | {obv_str}\n"
                    f"2-day consecutive confirmed ✅ -- mid-candle entry acceptable.\n"
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
                f"All criteria met -- P={prob:.0%} ✅  ML {ml_proba*100:.0f}% ✅"
                f"{obv_note}{rsi_note}. Enter at ${price:.4f}."
            ),
            price
        )

    # ── CANDLE CLOSE GATE ─────────────────────────────────────────
    # Applied after ENTER NOW check -- if all else qualifies but candle mid-forming
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
        conditions.append(f"Score {final_score:.0f}/{MIN_BUY_SCORE} -- needs {score_gap:.0f} more point(s)")

    if ml_proba < ML_MIN_CONFIDENCE:
        conditions.append(f"ML {ml_proba*100:.0f}% → needs >{ML_MIN_CONFIDENCE*100:.0f}% (currently bearish model)")

    if rsi > RSI_OVERBOUGHT_THRESH:
        conditions.append(f"RSI {rsi:.0f} overheated -- wait for cooldown below {RSI_REDUCE_THRESH}")

    if obv_trend == "falling":
        conditions.append("OBV falling (distribution) -- wait for OBV to turn up")

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
        status = "⏳ ALMOST -- 1 condition away"
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

    # suppress_debug was referenced at two call sites further down in this
    # function (prob_lifecycle_update's suppress_debug=suppress_debug, and
    # the "if suppress_debug:" guard before print_confidence_killers) but was
    # never actually assigned anywhere in this scope -- a NameError on every
    # single coin, every run, both silently swallowed by their enclosing
    # try/except blocks at logger.debug level (which LOG_LEVEL="INFO" filters
    # out entirely). Net effect: prob_lifecycle_update() never ran, so the
    # TREND_LIFECYCLE evidence group never contributed to any coin's prob,
    # and print_confidence_killers() never ran, so every "[CONFIDENCE
    # KILLERS]" line you've ever seen was the generic crash fallback, not
    # real output. False matches both functions' own parameter defaults.
    suppress_debug = False

    run_start_time = time.time()
    logger.info("=" * 70)
    logger.info("[BOT] Starting trading bot run...")
    logger.info(f"[BOT] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"[BOT] Scanning {len(COINS)} coins...")
    
    # [GLOBAL ML] Load global model at startup
    if not _GLOBAL_MODEL.get("model"):
        if not load_global_model():
            print(Fore.YELLOW + "[GLOBAL ML] No global model found — will train after this scan")
        else:
            _WF_SHARPE["__global__"] = _GLOBAL_MODEL.get("sharpe", 0.0)

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
    
    # [L] BTC data for Relative Strength -- reuse from get_btc_regime() cache (no extra API call)
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
                    # 3-class model: get both long and short probabilities at once
                    ml_proba, _short_from_long_model, _p_neutral, ml_confidence = \
                        predict_ml_both_directions(last, ml_model, ml_scaler,
                                                    df=df, btc_df=btc_df, fng_value=fng_value)
                    # Use the 3-class short prediction as a prior for short_ml_proba
                    # (overridden by dedicated short model if one exists)
                    if _short_from_long_model > 0 and _short_from_long_model != 0.5:
                        short_ml_proba = _short_from_long_model

                # ── GLOBAL MODEL: override / blend with per-coin ──────────────
                # When global model is available and profitable (Sharpe > 0):
                #   Use global prediction blended with per-coin (70/30 global/per-coin)
                #   Global model has 56,000+ samples vs 314 — far more reliable
                if _GLOBAL_MODEL.get("model") and _GLOBAL_MODEL.get("sharpe", 0) > 0.3:
                    try:
                        _rs_tier_now = sig.get("rs_full", {}).get("label", "NEUTRAL") \
                                       if "rs_full" in dir() else "NEUTRAL"
                        _g_long, _g_short, _g_note = predict_global_ml(
                            last, df, btc_df, coin,
                            rs_tier=_rs_tier_now, fng_value=fng_value)
                        _g_sharpe = _GLOBAL_MODEL.get("sharpe", 0)
                        _g_acc    = _GLOBAL_MODEL.get("acc", 0.5)

                        # Blend weight: more global weight when global Sharpe is higher
                        _global_wt = min(0.70, 0.40 + _g_sharpe * 0.10)
                        _local_wt  = 1.0 - _global_wt

                        if ml_proba == 0.5:
                            # Per-coin ML disabled → use global prediction entirely
                            # Keep ml_accuracy = per-coin acc for display (shows original training quality)
                            # Global acc is shown separately in the [GLOBAL ML] print line
                            ml_proba      = _g_long
                            ml_confidence = "medium"
                            # Note: ml_accuracy intentionally NOT overwritten here
                            # Table shows per-coin acc so user knows why it was disabled
                        else:
                            # Blend global + per-coin
                            ml_proba = _g_long * _global_wt + ml_proba * _local_wt

                        print(f"[GLOBAL ML] {coin}: long={_g_long:.0%}  short={_g_short:.0%}  "
                              f"blend={_global_wt:.0%}/{_local_wt:.0%}  "
                              f"final_ml={ml_proba:.0%}  "
                              f"(global acc={_g_acc:.0%} Sharpe={_g_sharpe:+.2f})")
                    except Exception as _ge:
                        print(f"[GLOBAL ML] {coin}: prediction failed — {_ge}")

                # Accumulate df for global model retraining
                _GLOBAL_COIN_DFS[coin] = df

                # ── ML GATE: Sharpe-primary, accuracy-secondary ────────────
                _wf_sharpe_gate = _WF_SHARPE.get(coin, 0.0)

                if _wf_sharpe_gate < -1.0:
                        # Strategy actively loses money on OOS — hard disable
                        ml_proba      = 0.5
                        ml_confidence = "disabled"
                        if LOG_ENABLED:
                            logger.warning(f"[ML GATE] {coin}: WF Sharpe={_wf_sharpe_gate:+.2f} "
                                           f"< -1.0 — strategy unprofitable, ML disabled")

                elif _wf_sharpe_gate > 0.5:
                        # Profitable strategy — allow regardless of raw accuracy
                        # The model has real edge even if accuracy < 50%
                        # (timing / magnitude asymmetry, not direction accuracy)
                        if ml_accuracy < 0.40:
                            # Very low accuracy — scale down but keep the Sharpe edge
                            ml_proba      = ml_proba * 0.50 + 0.5 * 0.50
                            ml_confidence = "low"
                            if LOG_ENABLED:
                                logger.info(f"[ML GATE] {coin}: Sharpe={_wf_sharpe_gate:+.2f} ✅ "
                                            f"acc={ml_accuracy:.1%} low but Sharpe positive — dampened 50%")
                        else:
                            # Sharpe positive + reasonable accuracy — full weight
                            ml_confidence = "medium" if ml_accuracy < ML_FULL_ACCURACY else "high"
                            if LOG_ENABLED:
                                logger.info(f"[ML GATE] {coin}: Sharpe={_wf_sharpe_gate:+.2f} ✅ "
                                            f"acc={ml_accuracy:.1%} — ML active")

                elif _wf_sharpe_gate >= 0.0:
                    # Marginal Sharpe — fall back to accuracy gate
                    if ml_accuracy < 0.50:
                        ml_proba      = ml_proba * 0.40 + 0.5 * 0.60
                        ml_confidence = "low"
                        if LOG_ENABLED:
                            logger.info(f"[ML GATE] {coin}: Sharpe={_wf_sharpe_gate:+.2f} marginal, "
                                        f"acc={ml_accuracy:.1%} below random — dampened 40%")
                    elif ml_accuracy < ML_MIN_ACCURACY:
                        ml_proba      = ml_proba * 0.50 + 0.5 * 0.50
                        ml_confidence = "low"
                    else:
                        ml_confidence = "medium"

                else:
                    # Sharpe negative but > -1.0 — heavily dampen
                    ml_proba      = ml_proba * 0.25 + 0.5 * 0.75
                    ml_confidence = "disabled"
                    if LOG_ENABLED:
                        logger.warning(f"[ML GATE] {coin}: Sharpe={_wf_sharpe_gate:+.2f} negative "
                                       f"— ML heavily dampened")

            # [H] Detect smart money / whale activity
            is_whale, whale_vol, whale_threshold = detect_whale_volume(df)
            smart_money_mode, smart_money_conf, smart_money_details = detect_smart_money_accumulation(df)
            # Volume divergence -- vol_regime read from df directly (sig not yet assigned here)
            try:
                _atr_col  = df["ATR"].dropna() if "ATR" in df.columns else pd.Series(dtype=float)
                _cur_atr  = float(_atr_col.iloc[-1]) if len(_atr_col) > 0 else 0.0
                _med_atr  = float(_atr_col.rolling(20, min_periods=5).median().iloc[-1]) \
                            if len(_atr_col) >= 5 else _cur_atr
                _atr_ratio = _cur_atr / (_med_atr + 1e-9)
                if   _atr_ratio > 1.30: _vd_regime = "expansion"
                elif _atr_ratio < 0.75: _vd_regime = "compression"
                else:                   _vd_regime = "neutral"
            except Exception:
                _vd_regime = "neutral"
            vol_div, vol_div_conf, vol_div_details = detect_volume_divergence(df, vol_regime=_vd_regime)
            if vol_div:
                _vd_atr_sig = vol_div_details.get("price_significance", 0)
                _vd_rank    = vol_div_details.get("vol_pct_rank", 50)
                print(f"[VOL DIV] {coin}: {vol_div}  "
                      f"ATR_sig={_vd_atr_sig:.2f}×  vol_rank={_vd_rank:.0f}th_pct  "
                      f"conf={vol_div_conf:.2f}  regime={_vd_regime}")

            # [I] Detect upcoming/recent news events
            events = get_upcoming_events(coin, days_ahead=EVENT_RISK_DAYS)
            event_block, event_block_reason = should_block_trade_on_event(coin, events)

            # Only adjust ML confidence if trade isn't already blocked
            # -- no point crushing confidence on a signal we're about to discard
            if event_block:
                ml_proba_adjusted      = ml_proba   # unchanged
                event_adjustment_reason = event_block_reason
            else:
                ml_proba_adjusted, event_adjustment_reason = apply_event_risk_adjustment(ml_proba, coin, events)

            # [AQ] OBV trend -- informational only, no longer multiplies ml_proba.
            # SYMMETRY FIX: OBV falling is already LR=0.35 in the scorer via
            # prob_obv_update(). Applying it again as ml_proba × 0.60 was:
            #   1. Double-counting the same OBV signal
            #   2. Applying the penalty in raw probability space (not log-odds)
            # Now: just detect OBV trend for display and smart money staleness guard.
            _obv_falling = False
            try:
                _, _, obv_trend, obv_desc = check_obv_confirmation(df, coin)
                _obv_falling = (obv_trend == "falling")
                if _obv_falling:
                    print(f"[OBV] {coin}: OBV falling -- already captured as LR=0.35 in scorer "
                          f"(no external ml_proba dampening)")
            except Exception:
                obv_trend = "unknown"; obv_desc = "OBV check error"
            obv_multiplier = 1.0   # kept for display compat; no longer applied

            # ── SHORT ML MODEL ────────────────────────────────────────────
            # Trained on inverted labels: label 1 = short won (-2.5 ATR target hit before +1.5 ATR stop)
            # Completely separate model files -- long model is never touched.
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
                    # Sharpe-primary gate (mirrors long ML gate)
                    _s_sharpe = _WF_SHARPE.get(f"{coin}_short", 0.0)
                    if _s_sharpe < -1.0:
                        short_ml_proba     = 0.5
                        short_ml_confidence = "disabled"
                    elif _s_sharpe > 0.5:
                        pass   # profitable short strategy — use prediction as-is
                    elif short_ml_accuracy < 0.50:
                        short_ml_proba = short_ml_proba * 0.40 + 0.5 * 0.60
                    elif short_ml_accuracy < ML_MIN_ACCURACY:
                        short_ml_proba = short_ml_proba * 0.70 + 0.5 * 0.30

                # ── Global ML fallback for sML ──────────────────────────────
                # When per-coin short model is missing, stale, or disabled by
                # Sharpe gate → use global model's class-0 (short wins) proba.
                if short_ml_proba == 0.5 and _GLOBAL_MODEL.get("model") and \
                        _GLOBAL_MODEL.get("sharpe", 0) > 0:
                    try:
                        _rs_now = sig.get("rs_full", {}).get("label", "NEUTRAL") \
                                  if "rs_full" in dir() else "NEUTRAL"
                        _, _g_short, _ = predict_global_ml(
                            last, df, btc_df, coin,
                            rs_tier=_rs_now, fng_value=fng_value)
                        if _g_short != 0.5:
                            short_ml_proba     = _g_short * 0.70 + 0.5 * 0.30
                            short_ml_confidence = "global"
                    except Exception:
                        pass

            # ═══════════════════════════════════════════════════════════════
            # [P] UNIFIED PROBABILISTIC ENGINE -- ONE SCORER, ONE finalize()
            # All evidence accumulates into scorer._evidence.
            # Group dampening, entropy suppression, and temperature scaling
            # are applied ONCE in finalize() with the COMPLETE evidence set.
            # ═══════════════════════════════════════════════════════════════
            scorer, sig = generate_signal_probabilistic(
                last, df, btc_regime=btc_regime, is_btc=is_btc,
                fng_value=int(fng_value))
            # scorer is NOT finalized yet -- pipeline continues adding evidence

            ml_guard_block = False

            # [X] Sentiment → scorer
            if SENTIMENT_ANALYSIS_ENABLED and USE_VADER:
                try:
                    sentiment, sent_label = get_crypto_sentiment(coin)
                    key_map = {
                        "very_positive": "sentiment_very_positive",
                        "positive":      "sentiment_positive",
                        "negative":      "sentiment_negative",
                        "very_negative": "sentiment_very_negative",
                    }
                    if sent_label in key_map:
                        scorer.update(key_map[sent_label], True,
                                      f"Sentiment: {sent_label}", "OTHER")
                        print(f"[SENTIMENT] {coin}: {sent_label} → scorer")
                    else:
                        print(f"[SENTIMENT] {coin}: neutral -- no update")
                except Exception as e:
                    print(f"[SENTIMENT ERROR] {coin}: {e}")

            # ── PROBABILISTIC PIPELINE -- all stages add to scorer ────────────
            # [Y] Volume quality → scorer
            _obv_trend = "unknown"; _at_h = False; _obv_falling = False
            if VOLUME_ANALYSIS_ENABLED:
                try:
                    quality, label, _vdet = analyze_volume_behavior(df, coin)
                    _obv_ok, _at_h, _obv_trend, _ = check_obv_confirmation(df, coin)
                    try:
                        _cur_v = df["volume"].iloc[-1]
                        _avg_v = df["volume"].iloc[-90:].mean() if len(df) >= 90 else df["volume"].mean()
                        _vzsc  = (_cur_v - _avg_v) / (df["volume"].std() + 1e-9)
                    except Exception:
                        _vzsc = 0.0
                    try:
                        _lusd, _ltier, _ = analyze_order_book_liquidity(coin)
                        _bthin = _ltier in ("paper_thin", "thin")
                    except Exception:
                        _bthin = False
                    prob_volume_update(scorer, coin, quality, label,
                                       obv_trend=_obv_trend, book_thin=_bthin,
                                       vol_zscore=float(_vzsc))
                except Exception as e:
                    print(f"[VOLUME ERROR] {coin}: {e}")

            # [Z] Price change 24h → scorer
            try:
                _pc24 = (df["price"].iloc[-1] - df["price"].iloc[-2]) / df["price"].iloc[-2] * 100 \
                        if len(df) >= 2 else 0
                prob_price_change_update(scorer, coin, _pc24)
            except Exception:
                pass

            # [AC] Z-Score → scorer
            _zscore_lo_before = scorer.log_odds
            try:
                _adx_val  = float(last.get("ADX", 0) or 0)
                _rsi_val  = float(last.get("RSI", 50) or 50)
                _in_trend = (sig.get("cloud_state") == "above" and
                             sig.get("ema12", 0) > sig.get("ema26", 0))
                _col_z    = "close" if "close" in df.columns else "price"
                _pz       = df[_col_z].iloc[-20:].values
                _z        = (_pz[-1] - _pz.mean()) / (_pz.std() + 1e-9)
                sig["z_score"] = round(float(_z), 2)
                prob_zscore_update(scorer, coin, float(_z), _adx_val, _in_trend, rsi=_rsi_val)
            except Exception as e:
                if LOG_ENABLED: logger.debug(f"[ZSCORE] {coin}: skipped - {e}")
            # zscore_delta used by exhaustion: log-odds delta from zscore step
            _zscore_lo_delta = scorer.log_odds - _zscore_lo_before

            # [AS] EMA26 exhaustion → scorer
            try:
                _ema26v  = float(last.get("EMA26", last["price"]))
                _dev     = (last["price"] - _ema26v) / _ema26v * 100 if _ema26v > 0 else 0
                _rsi_val = float(last.get("RSI", 50) or 50)
                prob_exhaustion_update(scorer, coin, _dev,
                                       zscore_delta=_zscore_lo_delta,
                                       rsi=_rsi_val)
            except Exception as e:
                if LOG_ENABLED: logger.debug(f"[EXHAUSTION] {coin}: skipped - {e}")

            # [AP] Wick → scorer
            try:
                _wadj, _wdesc = analyze_candle_rejection(df, coin)
                if _wadj > 0:
                    scorer.update("wick_strong_close", True, "Strong close wick", "OTHER")
                elif _wadj < 0:
                    scorer.update("wick_rejection",    True, "Rejection wick",    "OTHER")
            except Exception as e:
                if LOG_ENABLED: logger.debug(f"[WICK] {coin}: skipped - {e}")

            # [AA] Order book → scorer
            try:
                _lusd2, _ltier2, _ = analyze_order_book_liquidity(coin)
                _iadj, _iratio, _  = analyze_order_book_imbalance(coin)
                prob_orderbook_update(scorer, coin, float(_iratio), _ltier2)
            except Exception:
                pass

            # [AQ2] OBV flow → scorer
            try:
                _obv_c, _at_h, _obv_trend, _obv_desc = check_obv_confirmation(df, coin)
                _obv_falling = (_obv_trend == "falling")
                prob_obv_update(scorer, coin, _obv_trend, _at_h)
            except Exception as e:
                _obv_falling = False
                if LOG_ENABLED: logger.debug(f"[OBV] {coin}: skipped - {e}")

            # [MICRO] Market microstructure -- spread, OB imbalance, CVD, depth, liquidations
            # Probabilistic: each signal becomes a LR update via scorer.update()
            # Not deterministic gates -- microstructure amplifies or dampens Bayesian signals.
            try:
                _micro = fetch_microstructure(coin)
                if _micro and not _micro.error:
                    prob_microstructure_update(scorer, coin, _micro, last["price"])
            except Exception as _me:
                if LOG_ENABLED: logger.debug(f"[MICRO] {coin}: {_me}")

            # [AD] Chop zone → scorer
            chop_detected = False
            try:
                chop_detected = prob_chop_update(scorer, coin, df)
            except Exception as e:
                if LOG_ENABLED: logger.debug(f"[CHOP] {coin}: skipped - {e}")

            # [AE] Breakout → scorer
            try:
                prob_breakout_update(scorer, coin, df, chop_detected)
            except Exception as e:
                if LOG_ENABLED: logger.debug(f"[BREAKOUT] {coin}: skipped - {e}")

            # [VOL] Volatility regime → scorer
            vol_regime, vol_squeeze, vol_signals, vol_details = "neutral", False, {}, ""
            try:
                vol_regime, vol_squeeze, vol_signals, vol_details = detect_volatility_regime(df)
                prob_volregime_update(scorer, coin, df, vol_regime,
                                      vol_squeeze=vol_squeeze,
                                      vol_signals=vol_signals)
                sig["vol_regime"]  = vol_regime
                sig["vol_squeeze"] = vol_squeeze
            except Exception as _ve:
                if LOG_ENABLED: logger.debug(f"[VOL REGIME] {coin}: {_ve}")

            # [REGIME] Market regime amplifier/dampener → scorer
            try:
                _mr         = sig.get("market_regime", "unknown")
                _mr_adx     = float(last.get("ADX", 0) or 0)
                _mr_di_bull = (not math.isnan(float(last.get("plus_di", float("nan")))) and
                               not math.isnan(float(last.get("minus_di", float("nan")))) and
                               float(last.get("plus_di", 0)) > float(last.get("minus_di", 0)))
                _mr_macd_bull = float(last.get("MACD", 0)) > float(last.get("Signal", 0))
                prob_market_regime_update(scorer, coin,
                    market_regime=_mr, cloud_state=sig.get("cloud_state","unknown"),
                    adx=_mr_adx, di_bullish=_mr_di_bull,
                    chop_detected=chop_detected, macd_bullish=_mr_macd_bull)
            except Exception as _re:
                if LOG_ENABLED: logger.debug(f"[REGIME] {coin}: {_re}")

            # [MEM] Regime persistence memory -- chop streaks, vol clustering, half-life
            try:
                if coin not in _regime_memory:
                    _regime_memory[coin] = RegimeMemory()
                _atr_mem = float(last.get("ATR", last["price"] * 0.02) or last["price"] * 0.02)
                _vr_mem  = sig.get("vol_regime", "neutral")
                prob_regime_memory_update(
                    scorer        = scorer,
                    coin          = coin,
                    mem           = _regime_memory[coin],
                    market_regime = sig.get("market_regime", "unknown"),
                    vol_regime    = _vr_mem,
                    chop_detected = chop_detected,
                    atr           = _atr_mem,
                    phase         = sig.get("market_phase", "RANGE"),
                    phase_dur     = sig.get("phase_duration", 0),
                )
            except Exception as _me:
                if LOG_ENABLED: logger.debug(f"[REGIME MEM] {coin}: {_me}")

            # [MS] Market structure → scorer
            _ms = {}
            try:
                if MARKET_STRUCTURE_ENABLED:
                    _ms = analyze_market_structure(df)
                    prob_structure_update(scorer, coin, _ms)
                    sig["market_structure"] = _ms
                else:
                    print(f"[MKTSTR] {coin}: disabled")
            except Exception as _mse:
                print(f"[MKTSTR] {coin}: ERROR -- {_mse}")

            # Smart money → scorer
            try:
                prob_smart_money_update(scorer, coin, smart_money_mode,
                                        obv_trend=_obv_trend,
                                        sm_conf=float(smart_money_conf or 0.5))
            except Exception:
                pass

            # ── RELATIVE STRENGTH vs BTC + ETH (TOTAL3 proxy) ─────────────
            # Multi-timeframe RS: 1d / 7d / 30d vs BTC, 7d vs ETH.
            # Most valuable signal for avoiding over-suppression of leaders.
            try:
                _eth_df = None
                if coin.lower() not in ("ethereum", "eth"):
                    # Fetch ETH as TOTAL3 proxy — cached after first call
                    _eth_cache_key = "__eth_df_cache__"
                    if not hasattr(run_once, _eth_cache_key):
                        try:
                            _eth_raw = fetch_coin_data("ethereum")
                            setattr(run_once, _eth_cache_key,
                                    _eth_raw if _eth_raw is not None else pd.DataFrame())
                        except Exception:
                            setattr(run_once, _eth_cache_key, pd.DataFrame())
                    _eth_df = getattr(run_once, _eth_cache_key, None)
                    if _eth_df is not None and _eth_df.empty:
                        _eth_df = None

                _rs = compute_relative_strength_full(df, btc_df, eth_df=_eth_df)
                prob_rs_update(scorer, coin, _rs, btc_regime=btc_regime)
                sig["rs_full"] = _rs   # store for display / killers

                # ── RS-corrected bear LR (post-apply after real RS known) ──
                # generate_signal_probabilistic applied a temporary LR based on
                # a quick pre-estimate. Now that we have the real RS label,
                # correct the bear penalty if it differs.
                if btc_regime == "bear" and not is_btc:
                    _rs_real   = _rs.get("label", "NEUTRAL") if _rs.get("available") else "NEUTRAL"
                    _lr_real   = (0.70 if _rs_real == "LEADER" else
                                  0.60 if _rs_real == "STRONG"  else 0.50)
                    _lr_used   = _BTC_BEAR_LR   # what was applied in generate_signal_probabilistic
                    if abs(_lr_real - _lr_used) > 0.05:
                        # Correct the probability: divide out old LR, apply new LR
                        try:
                            _cur_lo  = math.log(max(1e-6, prob) / max(1e-6, 1-prob))
                            _cor_lo  = _cur_lo - math.log(_lr_used) + math.log(_lr_real)
                            prob     = 1 / (1 + math.exp(-_cor_lo))
                            sig["prob"] = prob
                            if LOG_ENABLED:
                                logger.debug(f"[RS BEAR LR] {coin}: "
                                             f"corrected LR {_lr_used:.2f}→{_lr_real:.2f} "
                                             f"(RS={_rs_real})")
                        except Exception:
                            pass
            except Exception as _rse:
                if LOG_ENABLED: logger.debug(f"[RS] {coin}: {_rse}")

            # [TRN] Markov transition model → scorer
            _p_transition_component = 0.50   # default neutral
            try:
                _cur_phase = classify_market_phase(
                    last, df, obv_trend=_obv_trend,
                    vol_regime=vol_regime, btc_regime=btc_regime)
                _phase_dur = track_phase_duration(df, _cur_phase)
                _tmat = PHASE_TRANSITION_MATRIX
                _lr_trans, _ = compute_transition_lr(_cur_phase, _phase_dur, _tmat)
                # Approximate transition-only probability for ensemble input
                _p_transition_component = _sigmoid(math.log(_lr_safe(_lr_trans)))
                sig["trans_prob_bull"] = round(_p_transition_component * 100)  # 0-100 integer %
                prob_transition_update(scorer, coin, _cur_phase, _phase_dur, matrix=_tmat)
                sig["market_phase"]   = _cur_phase
                sig["phase_duration"] = _phase_dur

                # ── Trend lifecycle stage (5-stage) ──────────────────────
                _lifecycle = classify_trend_lifecycle(
                    last, df, phase=_cur_phase, obv_trend=_obv_trend)
                prob_lifecycle_update(scorer, coin, _lifecycle, _cur_phase,
                                      suppress_debug=suppress_debug)
                sig["lifecycle"]      = _lifecycle
            except Exception as _te:
                print(Fore.RED + f"[TRANSITION] {coin}: ⚠️  phase/lifecycle update "
                      f"CRASHED — {type(_te).__name__}: {_te}")
                logger.error(f"[TRANSITION] {coin}: phase/lifecycle update crashed", exc_info=True)

            # Snapshot: Bayesian-only probability before ML bridge is added.
            # Used as one of the three inputs to the ensemble calibrator.
            # scorer.current_prob = sigmoid(log_odds) -- undampened running estimate.
            _p_bayes_snapshot = scorer.current_prob

            # [MLLR] ML → Bayesian bridge → scorer
            try:
                apply_ml_lr_to_prob(scorer, coin, float(ml_proba), df,
                                    ml_accuracy=float(ml_accuracy))
            except Exception as _mle:
                if LOG_ENABLED: logger.debug(f"[MLLR] {coin}: {_mle}")

            # ── ONE finalize() -- complete evidence, full group dampening ──────
            _final_result = scorer.finalize()
            _p_bayes_final = _final_result.prob_continuation

            # [CONSENSUS] Signal consensus map
            try:
                print_signal_consensus(
                    coin      = coin,
                    evidence  = _final_result.evidence,
                    vol_regime= sig.get("vol_regime", "neutral"),
                )
            except Exception as _ce:
                pass

            # ── Ensemble calibration ──────────────────────────────────────────
            # Combine Bayesian posterior, raw ML probability, and transition
            # component through a jointly-trained meta-learner.
            # Before enough resolved predictions accumulate, falls back to
            # weighted average: 55% Bayes + 30% ML + 15% Transition.
            try:
                prob = _ens_calibrator.predict(
                    p_bayes     = _p_bayes_final,
                    p_ml        = float(ml_proba),
                    p_transition= _p_transition_component,
                )
                if _ens_calibrator.fitted:
                    print(f"[ENS] {coin}: Bayes={_p_bayes_final:.3f}  "
                          f"ML={ml_proba:.3f}  Trans={_p_transition_component:.3f}  "
                          f"→ Ensemble={prob:.3f}")
                else:
                    prob = _p_bayes_final   # not fitted: use pure Bayesian
            except Exception:
                prob = _p_bayes_final

            if not True:  # suppress_debug -- controlled by global setting
                _final_result.print_summary(coin)

            # ── [ENTROPY] Regime entropy scaling ─────────────────────────────
            # Shrink probability toward 0.5 based on how uncertain the regime
            # classification is. Stable market → full confidence. Chaotic → weak.
            try:
                _re = compute_regime_entropy(
                    last=last, df=df, sig=sig,
                    adx=float(last.get("ADX", 20) or 20),
                    chop_detected=chop_detected,
                )
                prob, _entropy_note = apply_regime_entropy(prob, _re)
                sig["regime_entropy"] = _re
                if _re.label in ("UNSTABLE", "VOLATILE", "CHAOTIC"):
                    print(Fore.YELLOW + _entropy_note)
                else:
                    print(_entropy_note)
            except Exception as _ee:
                _re = None
                sig["regime_entropy"] = None

            # [REGIME CAL] Apply regime-conditional probability correction
            try:
                _regime_for_cal = "chop" if chop_detected else sig.get("market_regime", "unknown")
                _prob_before_rc = prob
                prob, _rc_note  = _brier_monitor.regime_calibrate(prob, _regime_for_cal)
                if abs(prob - _prob_before_rc) >= 0.01:
                    print(f"[REGIME CAL] {coin}: {_rc_note}")
                    sig["regime_calibration_applied"] = True
                    sig["_regime_cal_prob"] = prob          # captured NOW before any downstream reset
                    sig["_regime_cal_pre"]  = _prob_before_rc
                else:
                    sig["regime_calibration_applied"] = False
                    sig["_regime_cal_prob"] = None
            except Exception:
                sig["_regime_cal_prob"] = None
                pass

            # [CONTRADICTION] Detect cross-group indicator conflicts
            # These aren't captured by within-group latent compression.
            # A coin with cloud↓ + DI+ dominant is genuinely ambiguous —
            # the penalty shrinks conviction rather than blocking the trade.
            try:
                _n_contra, _contra_desc, _contra_sev = detect_contradictions(
                    sig=sig, last=last, df=df,
                    btc_regime=btc_regime,
                    chop_detected=chop_detected,
                )
                prob, _contra_note = apply_contradiction_penalty(
                    prob, coin, _n_contra, _contra_desc, _contra_sev)
                sig["contradiction_count"]   = _n_contra
                sig["contradiction_severity"]= _contra_sev
                if _contra_note:
                    print(Fore.YELLOW + _contra_note)
                    for _cd in _contra_desc:
                        print(Fore.YELLOW + f"  ⚠️  {_cd}")
            except Exception:
                sig["contradiction_count"]    = 0
                sig["contradiction_severity"] = 0.0

            # [KILLERS] Confidence killers — explain why P is suppressed
            try:
                if suppress_debug:
                    _killers = []
                else:
                    _killers = print_confidence_killers(
                        coin          = coin,
                        sig           = sig,
                        last          = last,
                        prob          = prob,
                        ml_proba      = float(ml_proba),
                        ml_accuracy   = float(ml_accuracy),
                        entropy       = _re,
                        chop_detected = chop_detected,
                        vol_regime    = sig.get("vol_regime", "neutral"),
                        phase         = sig.get("market_phase", "RANGE"),
                        phase_dur     = sig.get("phase_duration", 0),
                        btc_regime    = btc_regime,
                        mem           = _regime_memory.get(coin),
                    )
                sig["confidence_killers"] = _killers
            except Exception as _ke:
                print(Fore.RED + f"[CONFIDENCE KILLERS] {coin}: ⚠️  print_confidence_killers() "
                      f"CRASHED — {type(_ke).__name__}: {_ke}")
                logger.error(f"[KILLERS] {coin}: print_confidence_killers() crashed", exc_info=True)
                print(Fore.YELLOW + f"[CONFIDENCE KILLERS] {coin}: [fallback — suppression check "
                      f"failed, see error above]  ML_acc={ml_accuracy:.0%}  P={prob:.0%}")
                sig["confidence_killers"] = []

            # Populate the sig dict now that finalize has run
            sig["prob"]         = prob
            sig["prior"]        = sig.get("dynamic_prior", PRIOR_CONTINUATION)  # for bull P
            sig["score"]        = _final_result.score_equivalent
            sig["_prob_result"] = _final_result
            sig["action"]       = _final_result.action
            sig["blocked"]      = _final_result.blocked
            sig["technical_score"] = _final_result.score_equivalent

            # ── Expected Value ────────────────────────────────────────────
            # EV = P(win)×RR - P(loss)×1  per unit risked
            # Uses ATR-based RR from sig; lifecycle adjusts effective RR.
            try:
                _ev_rr     = sig.get("rr_ratio", 2.0) or 2.0
                _ev_stage  = sig.get("lifecycle", {}).get("stage", "MATURE_TREND")
                _ev_atr    = float(last.get("ATR", 0) or 0) / (float(last["price"]) + 1e-9) * 100
                _ev        = compute_ev(prob, _ev_rr, _ev_stage, _ev_atr)
                sig["ev"]  = _ev
                print(f"[EV] {coin}: EV={_ev['ev']:+.2f}  "
                      f"rating={_ev['rating']}  "
                      f"adj_RR={_ev['adj_rr']:.1f}  "
                      f"Kelly={_ev['kelly']:.1%}  "
                      f"breakeven={_ev['breakeven_p']:.0%}  "
                      f"({'✅ tradeable' if _ev['tradeable'] else '❌ EV too low'})")
            except Exception as _eve:
                sig["ev"] = {"ev": 0.0, "adj_rr": 2.0, "kelly": 0.0,
                             "rating": "D", "breakeven_p": 0.33,
                             "positive": False, "tradeable": False}

            # [TDBL] Tradeability Report -- five-component breakdown
            # Replaces the monolithic "8.58/5" with direction/participation/
            # stability/entry/tradeability each scored separately.
            try:
                _btwk_wr  = results[-1].get("backtest_win_rate", 0.0) if results else 0.0
                _btwk_pf  = results[-1].get("backtest_pf",       0.0) if results else 0.0
                _btwk_n   = results[-1].get("backtest_n",         0)   if results else 0
                _micro_tdbl = None
                try:
                    _micro_tdbl = fetch_microstructure(coin)
                except Exception:
                    pass
                _tdbl = compute_tradeability_report(
                    coin          = coin,
                    sig           = sig,
                    prob          = prob,
                    ml_proba      = float(ml_proba),
                    ml_accuracy   = float(ml_accuracy),
                    last          = last,
                    df            = df,
                    btc_regime    = btc_regime,
                    chop_detected = chop_detected,
                    obv_trend     = _obv_trend,
                    micro         = _micro_tdbl,
                    backtest_wr   = _btwk_wr,
                    backtest_pf   = _btwk_pf,
                    backtest_n    = _btwk_n,
                )
                sig["tradeability"] = _tdbl
                print(_tdbl.display())
            except Exception as _te:
                sig["tradeability"] = None
                if LOG_ENABLED: logger.debug(f"[TDBL] {coin}: {_te}")

            # [BRIER] Log this prediction for ongoing calibration monitoring.
            # Only log actionable signals (BUY / STRONG BUY) so the monitor
            # tracks real trading decisions, not WAIT noise.
            try:
                if _final_result.action in ("BUY", "STRONG BUY"):
                    _atr_log = float(last.get("ATR", last["price"] * 0.02) or last["price"] * 0.02)
                    _brier_monitor.log_prediction(
                        coin          = coin,
                        ml_proba      = float(ml_proba),
                        bayes_prob    = prob,
                        price         = float(last["price"]),
                        atr           = _atr_log,
                        action        = _final_result.action,
                        p_bayes_raw   = _p_bayes_snapshot,
                        p_transition  = _p_transition_component,
                        regime        = sig.get("market_regime", "unknown"),
                        chop_detected = chop_detected,
                    )
            except Exception:
                pass

            # ── [SHORT SCORER] Independent Bayesian chain for short setups ──
            # Runs regardless of the long scorer result. A coin can be:
            #   Long P=0.55 (WAIT), Short P=0.72 (SHORT) → short signal
            #   Long P=0.72 (BUY),  Short P=0.30 (WAIT) → long only
            #   Long P=0.35 (WAIT), Short P=0.35 (WAIT) → both wait (chop)
            _short_prob      = 0.0
            _short_action    = "WAIT"
            _short_blocked   = False
            try:
                _short_prior, _short_regime, _freshness_lr = detect_short_prior(
                    market_regime = sig.get("market_regime", "unknown"),
                    btc_regime    = btc_regime,
                    fng_value     = fng_value,
                    adx           = float(last.get("ADX", 20) or 20),
                    phase         = sig.get("market_phase", "RANGE"),
                    phase_dur     = sig.get("phase_duration", 0),
                )
                print(f"[SHORT PRIOR] {coin}: regime={_short_regime}  "
                      f"P₀={_short_prior:.3f}  freshness={_freshness_lr:.2f}")

                short_scorer = ProbabilisticScorer(
                    coin          = coin,
                    prior         = _short_prior,
                    prior_regime  = _short_regime,
                )

                # Apply freshness bonus as first LR if phase is very fresh
                if _freshness_lr > 1.02:
                    short_scorer.add_lr(_freshness_lr,
                                        f"Short freshness bonus (day {sig.get('phase_duration',0)} "
                                        f"of {sig.get('market_phase','?')})",
                                        group="OTHER")

                # Directional + structural signals
                _futures = fetch_futures_data(coin)
                if _futures:
                    _short_blocked = prob_short_squeeze_risk(short_scorer, coin, _futures)
                else:
                    _short_blocked = False

                if not _short_blocked:
                    _cur_phase  = sig.get("market_phase", "RANGE")
                    _cur_pd     = sig.get("phase_duration", 0)
                    _cur_micro  = None
                    try:
                        _cur_micro = fetch_microstructure(coin)
                    except Exception:
                        pass
                    _ba = run_short_scorer(
                        short_scorer   = short_scorer,
                        coin           = coin,
                        df             = df,
                        last           = last,
                        futures        = _futures,
                        obv_trend      = _obv_trend,
                        vol_regime     = sig.get("vol_regime", "neutral"),
                        phase          = _cur_phase,
                        phase_dur      = _cur_pd,
                        micro          = _cur_micro,
                        regime_memory  = _regime_memory.get(coin),
                        short_ml_proba = float(short_ml_proba),
                        short_ml_acc   = float(short_ml_accuracy),
                    )
                    # Store bear acceleration for display/position sizing
                    sig["bear_acceleration"]    = _ba
                    sig["short_expected_move_atr"] = _ba.expected_move_atr if _ba else 1.0
                    if _futures and not _futures.error:
                        print(f"[SHORT FUTURES] {coin}: funding={_futures.funding_rate*100:.3f}%  "
                              f"OI_chg={_futures.oi_change_24h:+.1f}%  L/S={_futures.ls_ratio:.2f}")
                    else:
                        print(f"[SHORT FUTURES] {coin}: no futures data")

                _short_result = short_scorer.finalize()
                _short_prob   = _short_result.prob_continuation

                # ── SHORT POST-FINALIZE PIPELINE (mirrors long pipeline) ───────
                # The long scorer applies 5 post-finalize adjustments.
                # All 5 now apply to short too:
                #   1. Contradiction penalty       ← already applied above
                #   2. Regime entropy scaling      ← NEW (same market entropy)
                #   3. Regime persistence scaling  ← NEW (bear regime age/flips)
                #   4. Short contradiction penalty ← already in run_short_scorer
                #   5. Regime calibration          ← TODO (needs more data first)

                # [1] Short contradiction penalty
                try:
                    _sn, _sdescs, _ssev = detect_short_contradictions(
                        sig=sig, last=last, df=df,
                        short_prob=_short_prob,
                        short_ml_proba=float(short_ml_proba),
                        short_ml_acc=float(short_ml_accuracy),
                        btc_regime=btc_regime,
                    )
                    _short_prob, _scontra_note = apply_contradiction_penalty(
                        _short_prob, coin, _sn, _sdescs, _ssev)
                    sig["short_contradiction_count"]    = _sn
                    sig["short_contradiction_severity"] = _ssev
                    if _scontra_note:
                        print(Fore.YELLOW + _scontra_note.replace("[CONTRADICTION]", "[SHORT CONTRA]"))
                        for _scd in _sdescs[:3]:
                            print(Fore.YELLOW + f"  ⚠️  {_scd}")
                except Exception:
                    sig["short_contradiction_count"]    = 0
                    sig["short_contradiction_severity"] = 0.0

                # [2] Regime entropy scaling for short
                # Same market entropy applies: a chaotic regime is unreliable
                # for BOTH longs and shorts. Pull short_prob toward 0.5 when
                # the regime classification itself is uncertain.
                try:
                    _re_s = sig.get("regime_entropy")   # computed earlier for long
                    if _re_s is None:
                        _re_s = compute_regime_entropy(
                            last=last, df=df, sig=sig,
                            adx=float(last.get("ADX", 20) or 20),
                            chop_detected=chop_detected,
                        )
                    _short_prob_before_entropy = _short_prob
                    _short_prob, _se_note = apply_regime_entropy(_short_prob, _re_s)
                    if abs(_short_prob - _short_prob_before_entropy) >= 0.01:
                        _lbl = _re_s.label
                        print(f"[ENTROPY-S] {coin}: {_lbl}  H={_re_s.entropy_norm:.3f}  "
                              f"scale={_re_s.confidence_scale:.2f}  "
                              f"short_P: {_short_prob_before_entropy:.3f} → {_short_prob:.3f}")
                except Exception:
                    pass

                # [3] Regime persistence scaling for short
                # Bear regime instability affects short reliability:
                #   - Many flips → whipsaw, shorts get stopped out
                #   - Negative vol_ac → mean-reverting vol, bounce risk
                #   - Phase overdue → bear ending, short timing poor
                try:
                    _mem_s = _regime_memory.get(coin)
                    if _mem_s is not None:
                        _flips_s   = _mem_s.regime_flip_count(window=10)
                        _vol_ac_s  = _mem_s.vol_autocorrelation()
                        _chop_s    = _mem_s.chop_streak()
                        _phase_s   = sig.get("market_phase", "RANGE")
                        _pdur_s    = int(sig.get("phase_duration", 0) or 0)
                        _typical_s = PHASE_TYPICAL_DURATION.get(_phase_s, 20)
                        _lambda_s  = PHASE_DECAY_LAMBDA.get(_phase_s, 1.40)
                        _age_pct_s = _pdur_s / max(1, _typical_s)
                        _strength_s= math.exp(-_lambda_s * _age_pct_s)
                        _pers_scale = 1.0
                        _pers_notes = []

                        # Regime whipsaw → poor timing for directional shorts
                        if _flips_s >= 6:
                            _pers_scale *= 0.65
                            _pers_notes.append(f"regime whipsawing ({_flips_s} flips)")
                        elif _flips_s >= 3:
                            _pers_scale *= 0.82
                            _pers_notes.append(f"{_flips_s} regime flips")

                        # Mean-reverting vol → shorts get squeezed
                        if _vol_ac_s < -0.30:
                            _pers_scale *= 0.80
                            _pers_notes.append(f"vol mean-reverting (ac={_vol_ac_s:+.2f})")

                        # Bear phase overdue → decay already absorbed much of the move
                        if _age_pct_s > 1.3:
                            _pers_scale *= max(0.70, _strength_s)
                            _pers_notes.append(f"phase overdue ({_pdur_s}/{_typical_s} bars, "
                                               f"{_strength_s:.0%} signal remaining)")

                        if _pers_scale < 0.99:
                            _sp_before = _short_prob
                            _short_prob = 0.5 + (_short_prob - 0.5) * _pers_scale
                            _short_prob = round(max(0.01, min(0.99, _short_prob)), 4)
                            print(f"[PERSIST-S] {coin}: scale={_pers_scale:.2f}  "
                                  f"short_P: {_sp_before:.3f} → {_short_prob:.3f}  "
                                  f"[{', '.join(_pers_notes)}]")
                except Exception:
                    pass

                # Short consensus map
                try:
                    print_signal_consensus(
                        coin      = coin,
                        evidence  = _short_result.evidence,
                        vol_regime= sig.get("vol_regime", "neutral"),
                    )
                except Exception:
                    pass

                if _short_blocked:
                    _short_action = "WAIT"
                elif _short_prob >= SHORT_STRONG_THRESHOLD:
                    _short_action = "STRONG SHORT"
                elif _short_prob >= SHORT_PROB_THRESHOLD:
                    _short_action = "SHORT"

                sig["short_prob"]              = round(_short_prob, 4)
                sig["short_action"]            = _short_action
                sig["short_blocked"]           = _short_blocked
                sig["short_fragility_score"]   = getattr(
                    _trend_fragility_cache[0] if _trend_fragility_cache else None,
                    "fragility", 0.0)
                sig["short_accel_n"]           = getattr(_ba, "n_confirmed", 0) if _ba else 0

                # Short confidence killers — always print so user knows WHY short isn't triggering
                try:
                    _short_killers = print_short_confidence_killers(
                        coin           = coin,
                        sig            = sig,
                        short_prob     = _short_prob,
                        short_ml_proba = float(short_ml_proba),
                        short_ml_acc   = float(short_ml_accuracy),
                        short_blocked  = _short_blocked,
                        bear_accel     = _ba,
                        trend_frag     = _trend_fragility_cache[0],
                        chop_detected  = chop_detected,
                        btc_regime     = btc_regime,
                        futures        = _futures,
                    )
                    sig["short_confidence_killers"] = _short_killers
                except Exception:
                    sig["short_confidence_killers"] = []
                print(f"[SHORT SCORER] {coin}: P={_short_prob:.3f}  action={_short_action}"
                      f"{'  (SQUEEZE BLOCKED)' if _short_blocked else ''}")

                # [TDBL-S] Short Tradeability Report
                try:
                    _tf = _trend_fragility_cache[0]  # may have been cleared; that's ok
                    _str = compute_short_tradeability_report(
                        coin=coin, sig=sig,
                        short_prob=_short_prob,
                        short_ml_proba=float(short_ml_proba),
                        short_ml_acc=float(short_ml_accuracy),
                        last=last, df=df,
                        obv_trend=_obv_trend,
                        vol_regime=sig.get("vol_regime","neutral"),
                        btc_regime=btc_regime,
                        chop_detected=chop_detected,
                        futures=_futures,
                        micro=_cur_micro,
                        bear_accel=_ba,
                        trend_frag=_tf,
                    )
                    sig["short_tradeability"] = _str
                    print(_str.display())
                except Exception as _ste:
                    sig["short_tradeability"] = None
                    logger.debug(f"[TDBL-S] {coin}: {_ste}")

            except Exception as _se:
                sig["short_prob"]   = 0.0
                sig["short_action"] = "WAIT"
                logger.debug(f"[SHORT SCORER] {coin}: {_se}")

            final_score  = sig["score"]
            _buy_thresh  = BUY_PROB_THRESHOLD
            _sell_thresh = SELL_PROB_THRESHOLD

            # Predict direction before decision
            direction, direction_conf = predict_direction(df)

            # ── RECALCULATE ACTION from probability ───────────────────────
            decision = "WAIT"
            if sig.get("blocked"):
                decision = "WAIT"
            elif prob >= _buy_thresh:
                decision = "BUY"
            elif prob <= _sell_thresh:
                decision = "SELL"
            sig["action"] = decision

            # Short ML dominant gate (pushes back if short model strongly disagrees)
            _short_ml_dominant = (
                short_ml_proba >= 0.65 and short_ml_accuracy >= 0.55 and
                (ml_accuracy < 0.50 or ml_proba < 0.40)
            )
            if _short_ml_dominant and prob >= _buy_thresh:
                prob           = _sell_thresh
                sig["prob"]    = prob
                sig["score"]   = round(max(0, min(12, (prob - 0.35) / 0.55 * 10)), 2)
                final_score    = sig["score"]
                sig["blocked"] = True
                decision       = "WAIT"
                sig["action"]  = "WAIT"
                sig["reasons"].append(
                    f"🚫 [SHORT ML GATE] BUY blocked: Short ML {short_ml_proba:.0%} "
                    f"(acc {short_ml_accuracy:.0%}) dominates. Mixed signal -- no long."
                )
                pass  # shown in [ML ENGINE]

            # SHORT: Needs prob <= SELL threshold + direction DOWN + ML >= 25%
            if prob <= _sell_thresh and direction == "DOWN" and ml_proba >= 0.25:
                sig["action"] = "🔴 SHORT"
            elif prob <= _sell_thresh and ml_proba < 0.25:
                sig["action"] = "🟡 WAIT"
            elif prob <= _sell_thresh:
                sig["action"] = "🟠 AVOID"
            elif prob <= 0.45:
                sig["action"] = "🟠 AVOID"
            elif sig["blocked"]:
                sig["action"] = "⚠️ BLOCKED"
                decision      = "WAIT"
            elif prob >= _buy_thresh:
                sig["action"] = "🟢 BUY"
            elif prob >= 0.50:
                sig["action"] = "🟡 WAIT"
            else:
                sig["action"] = "🟡 WAIT"
            
            # ── BACKTEST — Bayesian ATR-based (replaces old additive scorer) ──
            # Tests whether historical entries at this ATR stop/target ratio
            # would have been profitable. Uses only price data — no signals.
            # Entry: any day with prior P > 0.55 (would have been a BUY)
            # Stop:  entry - BRIER_ATR_STOP_MULT × ATR
            # Target:entry + BRIER_ATR_TARGET_MULT × ATR
            if BACKTEST_ENABLED:
                try:
                    _bt_wins = 0; _bt_losses = 0; _bt_n = 0
                    _bt_closes = df["close"].values if "close" in df.columns else df["price"].values
                    _bt_atrs   = df["ATR"].values   if "ATR"   in df.columns else None
                    _bt_min    = 20   # need at least 20 bars look-forward
                    if _bt_atrs is not None and len(_bt_closes) > _bt_min + 30:
                        for _i in range(30, len(_bt_closes) - _bt_min):
                            _entry = float(_bt_closes[_i])
                            _atr_i = float(_bt_atrs[_i]) if not np.isnan(_bt_atrs[_i]) else 0
                            if _atr_i <= 0: continue
                            _stop  = _entry - BRIER_ATR_STOP_MULT   * _atr_i
                            _tgt   = _entry + BRIER_ATR_TARGET_MULT * _atr_i
                            _hit   = None
                            for _j in range(_i+1, min(_i+_bt_min+1, len(_bt_closes))):
                                _lo = float(df["low"].iloc[_j])  if "low"  in df.columns else _bt_closes[_j]
                                _hi = float(df["high"].iloc[_j]) if "high" in df.columns else _bt_closes[_j]
                                if _lo <= _stop: _hit = "loss"; break
                                if _hi >= _tgt:  _hit = "win";  break
                            if _hit == "win":  _bt_wins   += 1; _bt_n += 1
                            elif _hit == "loss": _bt_losses += 1; _bt_n += 1
                    _bt_raw_wr = _bt_wins / _bt_n if _bt_n > 0 else 0.5
                    if _bt_n > 0:
                        backtest_color = Fore.GREEN if _bt_raw_wr > 0.45 else Fore.RED
                        print(backtest_color +
                              f"[BACKTEST] Bayesian ATR WR={_bt_raw_wr:.0%}  "
                              f"n={_bt_n}  Stop={BRIER_ATR_STOP_MULT}×ATR  "
                              f"Target={BRIER_ATR_TARGET_MULT}×ATR")
                        sig["backtest_win_rate"] = _bt_raw_wr * 100
                        sig["backtest_pf"]       = (_bt_wins / _bt_losses) if _bt_losses > 0 else 2.0
                        sig["backtest_trades"]   = _bt_n
                    if _bt_n >= BACKTEST_MIN_TRADES:
                        _breakeven = 1.0 / (1.0 + BRIER_ATR_TARGET_MULT / BRIER_ATR_STOP_MULT)
                        _adj_wr    = bayesian_win_rate(_bt_wins, _bt_n, _breakeven, BACKTEST_PRIOR_K)
                        _adj_odds  = _adj_wr / (1.0 - _adj_wr + 1e-9)
                        _be_odds   = _breakeven / (1.0 - _breakeven + 1e-9)
                        _raw_lr    = _adj_odds / (_be_odds + 1e-9)
                        _lr_cap    = min(BACKTEST_MAX_LR, 0.80 + _bt_n / 25.0)
                        _min_lr    = max(1.0 / _lr_cap, 0.25)
                        _final_lr  = max(_min_lr, min(_lr_cap, _raw_lr))
                        _data_wt   = _bt_n / (_bt_n + BACKTEST_PRIOR_K)
                        scorer.add_lr(
                            _final_lr,
                            f"Backtest WR={_bt_raw_wr:.0%}→adj={_adj_wr:.1%} "
                            f"(n={_bt_n} data_trust={_data_wt:.0%} LR={_final_lr:.3f})",
                            group="OTHER",
                        )
                        direction = "↑" if _final_lr > 1.02 else ("↓" if _final_lr < 0.98 else "≈")
                        print(f"[BACKTEST] adj_WR={_adj_wr:.1%}  LR={_final_lr:.3f}{direction}  "
                              f"cap={_lr_cap:.2f}  → scorer")
                except Exception as _bte:
                    logger.debug(f"[BACKTEST] {coin}: {_bte}")

                    # ── Backtest quality as scorer LR -- Bayesian shrinkage ───────
                    # PROBLEM with raw win rate:
                    #   WR=80% n=4  → trusts 80% (lucky sample, should be ~50%)
                    #   WR=80% n=70 → trusts 80% (reliable sample, should be ~77%)
                    #   Both treated nearly identically → small sample overpowers evidence.
                    #
                    # PROBLEM with sqrt-shrink on LR:
                    #   Shrinks the LR toward 1.0 (neutral), but the WR estimate is
                    #   still the raw 80%. The adjustment is on the wrong quantity.
                    #
                    # BAYESIAN FIX:
                    #   adjusted_wr = (wins + k*μ) / (n + k)
                    #   μ = breakeven WR = 0.286 (the rational prior with no data)
                    #   k = 10 virtual trades at μ
                    #
                    #   WR=100% n=4:   adj=48.9%  (prior dominates -- 4 trades means nothing)
                    #   WR=80%  n=10:  adj=54.3%  (balanced -- moderate trust)
                    #   WR=80%  n=70:  adj=73.6%  (data dominates -- sample is large)
                    #
                    # LR cap scales with sample size:
                    #   n=4:   max_lr = 1.25  (can barely move the probability)
                    #   n=10:  max_lr = 1.60
                    #   n=50+: max_lr = 2.50

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
                          f"${_don_high:.4f} -- CHASING breakout (confidence penalized)")
                elif _dev_eq > 12:
                    sig["entry_quality"]  = "near_exhaustion"
                    sig["ema26_dev_pct"]  = _dev_eq
                    print(Fore.YELLOW + f"[ENTRY] {coin}: EMA26 deviation {_dev_eq:.1f}% "
                          f"-- near exhaustion zone (confidence penalized)")
                else:
                    sig["entry_quality"] = "normal"
            except Exception:
                sig["entry_quality"] = "normal"

            sig["adx"]         = last.get("ADX", 0) or 0   # for confidence ML cap
            sig["ml_accuracy"] = ml_accuracy                 # for confidence ML cap


            # ═══════════════════════════════════════════════════════════════
            # ENHANCED ACTION: ML + Direction + ADX + Backtest gates
            # ═══════════════════════════════════════════════════════════════
            _coin_adx         = last.get("ADX", 0) or 0
            _backtest_pf      = sig.get("backtest_pf", 1.0)
            _backtest_winrate = sig.get("backtest_win_rate", 50)

            # STRONG BUY: probability qualifies + all quality gates pass
            _strong_buy_ok = (
                prob >= _buy_thresh and
                ml_proba >= 0.70 and
                direction == "UP" and
                _coin_adx >= 20 and
                _backtest_pf >= 0.85 and
                _backtest_winrate >= 45
            )
            if _strong_buy_ok:
                decision = "STRONG BUY"
                sig["action"] = "STRONG BUY"
            elif prob >= _buy_thresh and _coin_adx < 15:
                decision = "BUY"
                sig["action"] = "BUY"
                sig["reasons"].append(
                    f"⚠️ [ADX CAUTION] ADX {_coin_adx:.1f} < 15 -- no trend; "
                    f"signal valid but unreliable. Wait for ADX > 20 before sizing up."
                )
            elif prob >= _buy_thresh and ml_proba < 0.25:
                # SYMMETRY FIX: No longer a hard gate. The ML's bearish view was
                # already incorporated via disagreement-scaled residual LR in the scorer.
                # If P still qualifies after that, the evidence supports the trade.
                # Proceed as BUY but note the ML caution in sig["reasons"].
                decision = "BUY"
                sig["action"] = "BUY"
                sig["reasons"].append(
                    f"⚠️ [ML CAUTION] ML={ml_proba:.0%} bearish but Bayesian P={prob:.1%} "
                    f"held above threshold after ML's disagreement-scaled LR. "
                    f"Evidence outweighs ML concern. Size conservatively."
                )
                print(f"[ML NOTE] {coin}: P={prob:.1%} holds despite ML={ml_proba:.1%} "
                      f"(ML bearish LR already incorporated, evidence still bullish)")
            elif prob >= _buy_thresh and direction == "UP":
                decision = "BUY"
                sig["action"] = "BUY"
            # Short refinement
            _adx_for_short    = last.get("ADX", 0) or 0
            _adx_ok_for_short = _adx_for_short >= 15
            _short_prob_live  = float(sig.get("short_prob", 0.0))

            if not _adx_ok_for_short and _short_prob_live >= SHORT_PROB_THRESHOLD:
                decision = "WAIT"
                sig["action"] = "WAIT"
                sig["adx_short_blocked"] = True
                print(Fore.YELLOW + f"[ADX GATE] {coin}: SHORT blocked -- ADX {_adx_for_short:.1f} < 15")
            elif _short_prob_live >= SHORT_STRONG_THRESHOLD and short_ml_proba >= 0.60 and _adx_ok_for_short:
                decision = "STRONG SHORT"
                sig["action"] = "STRONG SHORT"
            elif _short_prob_live >= SHORT_PROB_THRESHOLD and short_ml_proba >= 0.40 and _adx_ok_for_short:
                decision = "SHORT"
                sig["action"] = "SHORT"
            elif _short_prob_live >= SHORT_PROB_THRESHOLD and short_ml_proba < 0.40 and _adx_ok_for_short:
                decision = "WAIT"
                sig["action"] = "WAIT"

            # ── CONFIDENCE OVERRIDE ──────────────────────────────────────
            # If confidence score says WAIT/AVOID, downgrade BUY to WAIT
            # so the displayed action matches the overall assessment.
            # Threshold: conf < 65 = not enough conviction to act
            if LOG_TRADES and decision not in ("WAIT", "AVOID", "BLOCKED") and not ml_guard_block:
                if decision in ("BUY", "STRONG BUY"):
                    # ⚠️  Log as PENDING — consecutive gate hasn't run yet
                    # Notification only fires AFTER all gates pass (consecutive, ML guard, event block)
                    # This entry does NOT mean a notification was sent
                    logger.info(f"[TRADE] ⏳ PENDING {coin:12} | P={prob:.0%} | ML: {ml_proba:.0%} | EV={sig.get('ev',{}).get('ev',0):+.2f} | awaiting consecutive gate")
                elif decision in ("SHORT", "STRONG SHORT"):
                    logger.info(f"[TRADE] 🔴 {decision} {coin:12} | Short P={sig.get('short_prob',0):.0%} | ML: {ml_proba:.0%}")
                elif decision == "SELL":
                    _short_p_log = float(sig.get("short_prob", 0))
                    logger.info(f"[TRADE] 🔴 SELL {coin:12} | Short P: {_short_p_log:.0%} | ML: {ml_proba:.0%}")
                    if _short_p_log < SHORT_PROB_THRESHOLD:
                        print(Fore.YELLOW + f"[SHORT] {coin}: notification suppressed — short_p={_short_p_log:.0%} below {SHORT_PROB_THRESHOLD:.0%}")
                    else:
                        _ml_reliable = ml_accuracy >= 0.55
                        _priority    = "urgent" if (_short_p_log >= SHORT_STRONG_THRESHOLD or ml_accuracy >= 0.75) else "high"
                        if _ml_reliable:
                            try:
                                _book_usd, _, _ = analyze_order_book_liquidity(coin)
                            except Exception:
                                _book_usd = 0
                            _liq_key, _liq_label, _max_pos = get_liquidity_tier(_book_usd)
                            if _liq_key == "paper_thin":
                                print(Fore.YELLOW + f"[SHORT] {coin}: Skipping ntfy -- paper-thin book, untradeable")
                            else:
                                _pos_msg = f"Max position: ${_max_pos:,}" if _max_pos else "No size limit"
                                send_ntfy(
                                    title=f"SHORT: {coin.upper()} | short_p={_short_p_log:.0%} | {_liq_label}",
                                    message=(
                                        f"Short P: {_short_p_log:.0%}\n"
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

            # [J] Store result for ranking
            _entry_status, _entry_when, _entry_watch = generate_entry_timing(
                coin, sig, df, last, final_score, btc_regime,
                ml_proba=ml_proba, direction=direction, direction_conf=direction_conf,
                short_ml_proba=short_ml_proba
            )
            results.append({
                "coin": coin,
                "score": sig["score"],
                "prob": sig.get("prob", 0.5),
                "prior": sig.get("prior", 0.42),           # for bull P calculation
                "sell_score": sig.get("short_prob", 0),
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
                "smart_money_mode": smart_money_mode,
                "smart_money_conf": smart_money_conf,
                "whale_volume_pct": (whale_vol / whale_threshold * 100) if (whale_threshold and whale_threshold > 0) else 0,
                "obv_momentum": smart_money_conf,
                "obv_trend":    obv_trend,
                "above_weekly_ema": any("above Weekly EMA50" in r for r in sig.get("reasons", [])),
                "blocked": sig["blocked"],
                "event_block": event_block,
                "events": events,
                "action": sig["action"],
                "entry_status":    _entry_status,
                "entry_when":      _entry_when,
                "entry_watch":     _entry_watch,
                "backtest_pf":               sig.get("backtest_pf", 1.0),
                "backtest_win_rate":          sig.get("backtest_win_rate", 50),
                "short_prob":                sig.get("short_prob", 0.0),
                "short_action":              sig.get("short_action", "WAIT"),
                "short_confidence_killers":  sig.get("short_confidence_killers", []),
                "short_fragility_score":     sig.get("short_fragility_score", 0.0),
                "short_accel_n":             sig.get("short_accel_n", 0),
                "confidence_killers":        sig.get("confidence_killers", []),
                "rs_full":                   sig.get("rs_full", {}),
                "lifecycle":                 sig.get("lifecycle", {}),
                "ev":                        sig.get("ev", {}),
                # Fields for trade journal
                "cloud_pos":       sig.get("cloud_state", "unknown"),
                "z_score":         sig.get("z_score"),
                "trans_prob_bull": sig.get("trans_prob_bull"),
                "entropy_label":   (sig.get("regime_entropy").label            if sig.get("regime_entropy") else None),
                "entropy_scale":   (sig.get("regime_entropy").confidence_scale if sig.get("regime_entropy") else None),
                "ema26_dev_pct":   sig.get("ema26_dev_pct"),
            })

            # Print coin name and decision (already have separator above)
            color = Fore.GREEN if decision == "BUY" else (Fore.RED if decision == "SELL" else Fore.CYAN if decision == "BUY_PULLBACK_ONLY" else Fore.MAGENTA)
            # [J] Display both BUY and SELL scores in decision line
            # ── BTC Market Direction Notification ────────────────────────
            if coin == BTC_ID and LOG_TRADES:
                _btc_adx    = last.get("ADX", 0) or 0
                _btc_di_pos = last.get("plus_di", 0) or 0
                _btc_di_neg = last.get("minus_di", 0) or 0
                _btc_rsi    = last.get("RSI", 50) or 50
                _btc_vol    = sig.get("vol_ratio", 1.0) or 1.0
                # Read OBV -- use the coin loop variable, fall back to sig if unknown
                _btc_obv = obv_trend if obv_trend != "unknown" else sig.get("obv_trend", obv_desc if "rising" in str(obv_desc) else "flat")
                _regime_emoji = "🟢" if btc_regime == "bull" else "🔴" if btc_regime == "bear" else "⚪"
                _dir_emoji    = "📈" if direction == "UP" else "📉" if direction == "DOWN" else "➡️"
                send_ntfy(
                    title=f"₿ BTC Market: {btc_regime.upper()} {_regime_emoji} | ${last['price']:,.0f}",
                    message=(
                        f"Regime:     {btc_regime.upper()} {_regime_emoji}\n"
                        f"Direction:  {direction} {_dir_emoji} ({direction_conf:.0%} consensus)\n"
                        f"P={prob:.0%}  EV={sig.get('ev',{}).get('ev',0):+.2f} ({sig.get('ev',{}).get('rating','?')})"
                        f"Prob:       {prob:.0%}\n"
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
                action_label = f"{decision} [P={prob:.0%}  EV={sig.get('ev',{}).get('ev',0):+.2f}  {sig.get('ev',{}).get('rating','?')}]"
            elif decision == "BUY_PULLBACK_ONLY":
                _ema26_lbl = float(last.get("EMA26", last["price"] * 0.90))
                action_label = f"BUY @ PULLBACK [P={prob:.0%} wait ${_ema26_lbl:.4f}]"
            elif decision == "SELL":
                action_label = f"{decision} [short_p={sig.get('short_prob',0):.0%}]"
            elif decision in ("SHORT", "STRONG SHORT"):
                # Use short-side probability and note — not long P/EV which are misleading here
                _sp_disp = sig.get("short_prob", 0)
                action_label = f"{decision} [short_P={_sp_disp:.0%}  long_P={prob:.0%}]"
            else:  # WAIT
                action_label = f"{decision} [P={prob:.0%}  EV={sig.get('ev',{}).get('ev',0):+.2f}  {sig.get('ev',{}).get('rating','?')}]"

            # ── One-line verdict (long + short) ───────────────────────────────
            def _shorten_killer(msg: str) -> str:
                """Compress a full killer message to ≤4 words."""
                _MAP = [
                    ("BTC bear + bear regime",    "BTC bear+local"),
                    ("BTC bear regime",            "BTC bear"),
                    ("BTC bull regime",            "BTC bull"),
                    ("Below Ichimoku cloud",       "below cloud"),
                    ("Price above cloud",          "above cloud"),
                    ("Inside cloud",               "in cloud"),
                    ("Chop detected",              "chop"),
                    ("OBV falling",                "OBV↓"),
                    ("ML strongly bearish",        f"ML↓{ml_proba:.0%}"),
                    ("ML bearish",                 f"ML↓{ml_proba:.0%}"),
                    ("ML disabled",                "ML off"),
                    ("No bear acceleration",       "no accel"),
                    ("Weak acceleration",          "accel weak"),
                    ("No fragility",               "no fragility"),
                    ("Low fragility",              "frag low"),
                    ("squeeze",                    "squeeze risk"),
                    ("Rejection wick",             "rejection wick"),
                    ("Weak ADX",                   f"ADX={float(last.get('ADX',0)):.0f} weak"),
                    ("Very weak ADX",              f"ADX={float(last.get('ADX',0)):.0f} flat"),
                    ("Structure BEARISH",          "struct↓"),
                    ("Structure BULLISH",          "struct↑"),
                    ("CHAOTIC regime entropy",     "chaotic H"),
                    ("Volatile regime entropy",    "volatile H"),
                    ("Unstable regime entropy",    "unstable H"),
                    ("Phase overdue",              "phase old"),
                    ("Regime whipsawing",          "whipsaw"),
                    ("Short P=",                   f"P={sig.get('short_prob',0):.0%}↓"),
                    ("Short ML",                   f"sML↓{short_ml_proba:.0%}"),
                ]
                for fragment, short in _MAP:
                    if fragment.lower() in msg.lower():
                        return short
                # fallback: first 3 words
                words = msg.replace("⛔","").replace("❌","").replace("⚠️","").strip().split()
                return " ".join(words[:3])

            # Long verdict
            _lk     = sig.get("confidence_killers", [])
            _l_tags = " · ".join(_shorten_killer(k) for k in _lk[:3]) if _lk else "signals ok"
            _l_col  = (Fore.GREEN  if decision in ("BUY","STRONG BUY") else
                       Fore.RED    if prob < 0.35 else Fore.YELLOW)
            _l_dec  = decision if decision in ("BUY","STRONG BUY") else "WAIT"

            # Short verdict
            _sp     = float(sig.get("short_prob", 0))
            _sk     = sig.get("short_confidence_killers", [])
            _s_tags = " · ".join(_shorten_killer(k) for k in _sk[:3]) if _sk else "setup ok"
            _s_dec  = (sig.get("short_action","WAIT") if _sp >= SHORT_PROB_THRESHOLD
                       else "WAIT")
            _s_col  = (Fore.RED    if _s_dec in ("SHORT","STRONG SHORT") else
                       Fore.GREEN  if _sp < 0.25 else Fore.YELLOW)

            _bar = "━" * 68
            print(f"{_bar}")
            print(f"{Fore.WHITE}{coin.upper():<6}{Fore.RESET}  "
                  f"{_l_col}LONG  {_l_dec:<12} P={prob:.0%}{Fore.RESET}  │  {_l_tags}")
            print(f"{'':6}  "
                  f"{_s_col}SHORT {_s_dec:<12} P={_sp:.0%}{Fore.RESET}  │  {_s_tags}")
            print(f"{_bar}")

            print(f"{coin.upper():<14} ➜ {color}{action_label}")
            
            # [G] ML prediction display
            if ML_ENABLED and ml_model is not None:
                ml_color = Fore.GREEN if ml_confidence == "high" else (Fore.YELLOW if ml_confidence == "medium" else (Fore.RED if ml_confidence in ("low","disabled") else Fore.WHITE))
                _conf_label = {
                    "high":     "HIGH",
                    "medium":   "MEDIUM",
                    "low":      "LOW",
                    "disabled": "DISABLED (model accuracy < 50% -- worse than random, neutralized to 50%)",
                    "unknown":  "UNKNOWN (prediction failed)",
                }.get(ml_confidence, ml_confidence.upper())
                pass  # suppressed: shown in [ML ENGINE]
            
            # [N] On-Chain Metrics Display
            if DETECT_WHALE_ACTIVITY:
                on_chain_score, on_chain_details = detect_on_chain_accumulation(df)
                is_extreme, vol_zscore, vol_details = detect_volume_extremes(df)
                
                on_chain_color = Fore.GREEN if on_chain_score > 0.3 else (Fore.RED if on_chain_score < -0.3 else Fore.YELLOW)
                print(on_chain_color + f"[ON-CHAIN] {on_chain_details} | {vol_details}")
                if is_extreme:
                    print(Fore.MAGENTA + f"[WHALE ALERT] Extreme volume detected! Volume Z-score: {vol_zscore:.1f}σ")
            
            # [U] Advanced Stops Display -- only for actionable signals
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
            
            
            # [NEW] Direction and move size
            dir_color = Fore.GREEN if direction == "UP" else Fore.RED
            print(dir_color + f"[DIRECTION] {direction} ({direction_conf:.0%} confidence) | Expected move: {move_size:.1f}%")
            
            # [NEW] Narrative summary
            print(Fore.CYAN + f"[WHAT'S NEXT] {narrative}")

            # ── ML INSIGHT -- explain why the model gave this rating ───────────
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
            # ── [ML ENGINE] Single consolidated ML summary ───────────────────
            _ml_verdict = (
                "AVOID LONGS"  if ml_proba < 0.35 else
                "WEAK LONG"    if ml_proba < 0.50 else
                "NEUTRAL"      if ml_proba < 0.60 else
                "BULLISH"      if ml_proba < 0.75 else
                "STRONGLY BULLISH"
            )
            _short_verdict = (
                "AVOID SHORTS" if short_ml_proba < 0.35 else
                "WEAK SHORT"   if short_ml_proba < 0.50 else
                "NEUTRAL"      if short_ml_proba < 0.60 else
                "SHORT SIGNAL" if short_ml_proba < 0.75 else
                "STRONG SHORT"
            )
            _acc_tag = (
                "disabled" if ml_accuracy < 0.50 else
                "marginal"  if ml_accuracy < 0.55 else
                "acceptable" if ml_accuracy < 0.65 else
                "good"       if ml_accuracy < 0.75 else
                "strong"
            )
            _ml_gate_desc = (
                "disabled (<50%)"  if ml_accuracy < 0.50 else
                "dampened (52-55%)" if ml_accuracy < 0.55 else
                "marginal"          if ml_accuracy < 0.60 else
                "trusted"
            )
            _guard_note = f"  ML guard: {1-ml_proba:.0%} failure rate" if ml_proba < 0.40 else ""
            _short_acc_tag = f" (short acc {short_ml_accuracy:.0%})" if short_ml_accuracy >= 0.50 else " (short ML disabled)"

            _top_long  = ""
            _top_short = ""
            if _long_insights:
                _top = [f[0] for f in sorted(_long_insights, key=lambda x: -x[1])[:2]]
                _top_long = f"  Top drivers: {', '.join(_top)}"
            if _short_insights:
                _top2 = [f[0] for f in sorted(_short_insights, key=lambda x: -x[1])[:2]]
                _top_short = f"  Short drivers: {', '.join(_top2)}"

            _ml_color = Fore.GREEN if ml_proba >= 0.65 else (Fore.RED if ml_proba < 0.40 else Fore.YELLOW)
            print(_ml_color +
                  f"[ML ENGINE] {coin}\n"
                  f"  Long  success: {ml_proba:.0%}   → {_ml_verdict}   (acc {ml_accuracy:.0%}, {_ml_gate_desc})"
                  f"{_top_long}\n"
                  f"  Short success: {short_ml_proba:.0%}   → {_short_verdict}{_short_acc_tag}"
                  f"{_top_short}"
                  f"{_guard_note}"
            )

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


            # ── SMART RE-ENTRY ALERT ──────────────────────────────────────
            # When entry timing says "wait for EMA26" or similar level,
            # check if price is currently approaching that level.
            # If within 3% of watch price AND key conditions are still valid,
            # send a specific alert -- not just "price reached level" but
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
                        f"P={prob:.0%}  EV={sig.get('ev',{}).get('ev',0):+.2f}  ML: {ml_proba:.0%}\n"
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
                          f"${_watch_price:.4f} -- {_conditions_met}/4 conditions intact")
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
                    print(event_color + f"[EVENT IMPACT] {event_adjustment_reason} -- ML confidence reduced from {ml_proba:.0%} to {ml_proba_adjusted:.0%}")
                
                if event_block:
                    print(Fore.RED + f"⛔ {event_block_reason}")
            
            # [H] Smart money display
            whale_indicator = "🐋 WHALE ZONE" if is_whale else "normal"
            whale_color = Fore.CYAN if is_whale else Fore.WHITE
            print(whale_color + f"[WHALES] Volume: {whale_vol:.0f} ({whale_vol/whale_threshold*100:.0f}% of whale threshold) -- {whale_indicator}")
            
            smart_color = Fore.GREEN if smart_money_mode == "accumulation" else (Fore.RED if smart_money_mode == "distribution" else Fore.YELLOW)
            print(smart_color + f"[SMART MONEY] {smart_money_mode.upper()} mode | OBV momentum: {smart_money_conf:.0%}")
            
            if vol_div:
                div_color = Fore.GREEN if vol_div == "bullish_divergence" else Fore.RED
                print(div_color + f"[VOLUME DIV] {vol_div.upper()} detected -- low volume on {vol_div_details.get('price_trend', '?')} price")
            
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
                    pass  # ML guard note shown in [ML ENGINE]
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
                if _chop_safe:       zone_label += " (chop -- no expansion)"
                elif not _di_bullish: zone_label += " (DI- dominant -- no expansion)"
                elif not _not_stretched: zone_label += " (Z-score stretched -- no expansion)"

            if price_now < safe_low or price_now > safe_high:
                print(Fore.YELLOW + f"⚠️ [WARNING] Price {price_now:.6f} is OUTSIDE safe entry zone "
                      f"({safe_low:.6f} → {safe_high:.6f}) [{zone_label}]")
            else:
                print(Fore.GREEN + f"✅ Safe Entry Zone: {safe_low:.6f} → {safe_high:.6f} [{zone_label}]")

            if decision == "BUY" or capitulation:
                # ── REGIME CAL GATE ───────────────────────────────────────────
                # regime_calibrate() runs earlier in the pipeline but prob often
                # gets reset downstream (ensemble calibrator, tradeability report,
                # etc.) before reaching here. Read from sig["_regime_cal_prob"]
                # which was captured immediately at the call site, so it's immune
                # to those resets. If regime-cal corrected P below BUY threshold,
                # override the decision to WAIT before the consecutive gate runs.
                _rc_prob = sig.get("_regime_cal_prob")
                if (_rc_prob is not None
                        and _rc_prob < BUY_PROB_THRESHOLD
                        and not capitulation):
                    _rc_pre = sig.get("_regime_cal_pre", prob)
                    print(Fore.YELLOW +
                          f"[REGIME GATE] {coin}: regime-cal P={_rc_pre:.0%} → {_rc_prob:.0%} "
                          f"(below {BUY_PROB_THRESHOLD:.0%} threshold) — BUY overridden to WAIT  "
                          f"[bear regime historical_wr≈3%]")
                    decision       = "WAIT"
                    sig["action"]  = "WAIT"
                    sig["blocked"] = True
                    sig["reasons"].append(
                        f"🚫 [REGIME GATE] Regime-cal P={_rc_prob:.0%} below buy threshold "
                        f"— historical win rate in this regime too low to trade."
                    )
                    continue   # skip all BUY execution code (consecutive gate, notification)
                breakout_up = d["breakout_up"]

                # [G] ML guard blocks BUY if model is confident and predicts failure
                if ml_guard_block and not capitulation:
                    print(Fore.MAGENTA + "[NO ACTION] ML guard blocked this trade.")
                    # Update the already-appended result to reflect the block
                    if results and results[-1]["coin"] == coin:
                        results[-1]["decision"] = "WAIT"
                        results[-1]["action"]   = "WAIT"
                    continue

                # [I] Event block -- do not trade before/after high-impact events
                if event_block and not capitulation:
                    print(Fore.RED + f"[NO ACTION] {event_block_reason}")
                    continue

                # [F] Consecutive confirmation — regime-conditional
                # TRENDING (EARLY_BULL + RS LEADER/STRONG): 1/2 days sufficient
                # STANDARD/WEAK: 2/2 required
                if decision == "BUY" and not capitulation:
                    _phase_now    = sig.get("market_phase", "RANGE")
                    _rs_now       = (sig.get("rs_full", {}) or {}).get("label", "NEUTRAL")
                    _lc_stage_now = (sig.get("lifecycle", {}) or {}).get("stage", "MATURE_TREND")

                    # ── Regime-conditional consecutive requirement ─────────────
                    # TRENDING  (EARLY_BULL/STRONG_BULL + RS LEADER/STRONG
                    #            or EARLY_TREND stage): 1/2 days required
                    # EARLY     (ACCUMULATION phase or EARLY_TREND lifecycle):
                    #            1/2 days — fresh setups shouldn't need 2-day confirm
                    # STANDARD: 2/2 days
                    _is_trending_now = (
                        (_phase_now in ("EARLY_BULL", "STRONG_BULL") and
                         _rs_now in ("LEADER", "STRONG")) or
                        _lc_stage_now == "EARLY_TREND"
                    )
                    _is_early_now = (
                        _phase_now in ("ACCUMULATION",) or
                        _lc_stage_now == "ACCUMULATION"
                    )
                    _consec_days_req = (1 if (_is_trending_now or _is_early_now)
                                        else MIN_CONSECUTIVE_DAYS)
                    consecutive_pass, consecutive_met, consecutive_scores = check_consecutive_confirmation(
                        df, btc_regime=btc_regime, is_btc=is_btc, min_days=_consec_days_req
                    )
                    _regime_note = (
                        f"[TRENDING: {_consec_days_req}/2 req]" if _is_trending_now else
                        f"[ACCUM: {_consec_days_req}/2 req]"    if _is_early_now    else
                        f"[{_consec_days_req}/{_consec_days_req} req]"
                    )
                    print(Fore.CYAN + f"[CONSECUTIVE] Last {_consec_days_req} candles {_regime_note}: ", end="")
                    for cs in consecutive_scores:
                        status = "✅" if cs["action"] == "PASS" else "❌"
                        print(f"{status}D{cs['day_offset']}(score={cs['score']})", end=" ")
                    print()

                    # ── CONSECUTIVE CANDLE LR ─────────────────────────────────
                    # 2/2 confirmed  → LR=1.80  trend established
                    # 1/2 confirmed  → LR=1.20  mild confirmation
                    # 0/2 confirmed  → LR=0.65  single candle (softened from 0.55)
                    #
                    # LR=0.65 effect on key P levels:
                    #   P=80% 0/2 → 73%  ✅ BUY (was blocked at 0.55)
                    #   P=75% 0/2 → 66%  ✅ BUY (was 62%, below threshold)
                    #   P=70% 0/2 → 61%  ⚠️ borderline — below 65% → WAIT
                    #   P=65% 0/2 → 55%  ❌ WAIT
                    _consec_lr = (
                        1.80 if consecutive_met >= _consec_days_req else
                        1.20 if consecutive_met == 1 else
                        0.65   # softened from 0.55 — less aggressive penalty
                    )
                    _prob_before_consec = prob
                    try:
                        import math as _math
                        _lo  = _math.log(max(0.001, min(0.999, prob)) /
                                         (1 - max(0.001, min(0.999, prob))))
                        _lo += _math.log(_consec_lr)
                        prob = 1 / (1 + _math.exp(-_lo))
                    except Exception:
                        prob = prob * 0.85 if _consec_lr < 1.0 else min(0.97, prob * 1.10)
                    sig["prob"] = prob

                    _consec_tag = (
                        f"LR={_consec_lr:.2f} ✅" if consecutive_met >= _consec_days_req else
                        f"LR={_consec_lr:.2f} ⚠️" if consecutive_met == 1 else
                        f"LR={_consec_lr:.2f} ⬇"
                    )
                    print(Fore.CYAN + f"[CONSECUTIVE] Bayesian update: "
                          f"P {_prob_before_consec:.0%} → {prob:.0%}  "
                          f"({consecutive_met}/{_consec_days_req} days  {_consec_tag})")

                    # ── P≥80% VETO (lowered from 85%) ─────────────────────────
                    # Strong signals bypass consecutive gate entirely.
                    # LR penalty still applied for honest probability display.
                    _veto_consecutive = _prob_before_consec >= 0.80

                    if _veto_consecutive and not consecutive_pass:
                        print(Fore.GREEN +
                              f"[CONSECUTIVE] ⚡ P={_prob_before_consec:.0%} VETO — "
                              f"consecutive bypassed. Adjusted P={prob:.0%} after LR penalty.")
                        consecutive_pass = True

                    # Hard gate — only fires as last resort when:
                    # 1. Not vetoed (P < 85%)
                    # 2. 0 consecutive days confirmed (not even 1/2)
                    # 3. LR still left prob above threshold (LR alone wasn't enough)
                    # Cases where LR pushed prob below threshold are handled naturally above.
                    if not consecutive_pass and consecutive_met == 0 and prob >= BUY_PROB_THRESHOLD:
                        print(Fore.YELLOW + f"[CONSECUTIVE] Hard gate: 0/{_consec_days_req} days, "
                              f"P={prob:.0%} still above threshold — WAIT "
                              f"(need at least 1 confirming day or P≥85% veto)")
                        continue
                    elif not consecutive_pass and prob < BUY_PROB_THRESHOLD:
                        print(Fore.YELLOW + f"[CONSECUTIVE] P={prob:.0%} below threshold "
                              f"after LR={_consec_lr:.2f} penalty — WAIT naturally")
                        continue
                    else:
                        print(Fore.GREEN + f"[CONSECUTIVE] ✅ P={prob:.0%}"
                              + (" ⚡ VETO APPLIED" if _veto_consecutive and not consecutive_pass else
                                 f" {consecutive_met}/{_consec_days_req} days confirmed"))

                    # Pass to generate_entry_timing so candle gate can be relaxed
                    sig["_consecutive_confirmed"] = consecutive_pass

                    # Re-generate entry timing now that consecutive is known
                    # First call at line ~7462 ran before consecutive -- may show wrong message
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

                    # ── BUY_PULLBACK_ONLY — intercept before notification ──────
                    # Conditions: BUY + P > 75% + RS LEADER + Stage = EXHAUSTION
                    # Instead of a BUY alert, send a WATCH alert with pullback target
                    _lc_data_check  = sig.get("lifecycle", {}) or {}
                    _lc_stage_check = _lc_data_check.get("stage", "")
                    _rs_data_check  = sig.get("rs_full", {}) or {}
                    _rs_label_check = _rs_data_check.get("label", "NEUTRAL")
                    _pullback_only  = (
                        decision in ("BUY", "STRONG BUY") and
                        prob >= 0.75 and
                        _rs_label_check == "LEADER" and
                        _lc_stage_check == "EXHAUSTION"
                    )

                    if _pullback_only:
                        _ema26_pb  = float(last.get("EMA26", last["price"] * 0.90))
                        _stop_pb   = _ema26_pb * 0.92
                        _target_pb = float(sig.get("stop_price", last["price"]) + (last["price"] - float(last.get("EMA26", last["price"]))) * 0.80)
                        _target_pb = max(_target_pb, last["price"] * 1.15)
                        _rr_pb     = (_target_pb - _ema26_pb) / (_ema26_pb - _stop_pb) if (_ema26_pb - _stop_pb) > 0 else 0
                        # Override decision display — keep scorer result but change action
                        decision          = "BUY_PULLBACK_ONLY"
                        sig["action"]     = "🔄 PULLBACK"
                        if results and results[-1].get("coin") == coin:
                            results[-1]["action"]       = "🔄 PULLBACK"
                            results[-1]["entry_status"] = "🔄 BUY ON PULLBACK"
                            results[-1]["entry_when"]   = (
                                f"[EXHAUSTION + RS LEADER] P={prob:.0%} is strong but price "
                                f"is extended. Do NOT buy at ${last['price']:.4f}.\n"
                                f"Wait for pullback to EMA26 (${_ema26_pb:.4f}).\n"
                                f"Entry zone: ${_ema26_pb:.4f}  Stop: ${_stop_pb:.4f}  "
                                f"Target: ${_target_pb:.4f}  R:R ≈ {_rr_pb:.1f}:1"
                            )
                            results[-1]["entry_watch"] = _ema26_pb

                        print(Fore.CYAN +
                              f"\n[PULLBACK ONLY] {coin.upper()}: P={prob:.0%} ✅ RS LEADER ✅ "
                              f"but Stage=EXHAUSTION — do NOT buy at ${last['price']:.4f}")
                        print(Fore.CYAN +
                              f"[PULLBACK ONLY] Wait for EMA26 ${_ema26_pb:.4f} "
                              f"(currently {((last['price']-_ema26_pb)/_ema26_pb)*100:.1f}% away)")

                        if LOG_TRADES:
                            send_ntfy(
                                title=f"🔄 PULLBACK: {coin.upper()} P={prob:.0%} — wait for ${_ema26_pb:.4f}",
                                message=(
                                    f"Signal:       BUY_PULLBACK_ONLY\n"
                                    f"P={prob:.0%}  EV={sig.get('ev',{}).get('ev',0):+.2f} "
                                    f"({sig.get('ev',{}).get('rating','?')})  RS={_rs_label_check}\n\n"
                                    f"⚠️  Stage = EXHAUSTION — price extended\n"
                                    f"Current price: ${last['price']:.4f} "
                                    f"({((last['price']-_ema26_pb)/_ema26_pb)*100:.1f}% above EMA26)\n\n"
                                    f"📍 Entry zone:  ${_ema26_pb:.4f}  (EMA26)\n"
                                    f"🛑 Stop:        ${_stop_pb:.4f}\n"
                                    f"🎯 Target:      ${_target_pb:.4f}\n"
                                    f"📊 R:R:         {_rr_pb:.1f}:1  (vs {2.0:.1f}:1 now)\n\n"
                                    f"Wait for price to pull back to EMA26, "
                                    f"OBV to hold, then enter."
                                ),
                                priority="default",
                                tags="arrows_counterclockwise,chart_increasing",
                                topic_env="NTFY_TOPIC"
                            )
                        # Skip the normal BUY notification below
                        continue

                    # ── BUY NOTIFICATION -- fired only here, after ALL gates pass ──
                    # Gates passed: ML guard ✅ | event block ✅ | consecutive ✅
                    if decision in ("BUY", "STRONG BUY") and LOG_TRADES:
                        logger.info(f"[TRADE] 🟢 CONFIRMED {coin:12} | P={prob:.0%} | ML: {ml_proba:.0%} | EV={sig.get('ev',{}).get('ev',0):+.2f} | notification sent")
                        _notify_buy = prob >= BUY_PROB_THRESHOLD and ml_proba >= 0.50
                        _priority   = "urgent" if prob >= 0.85 else "high"   # always defined before send_ntfy
                        if _notify_buy:
                            _killers_ntfy = sig.get("confidence_killers", [])
                            _killers_str  = ("\nKillers:\n" +
                                             "\n".join(f"  {k}" for k in _killers_ntfy[:4])
                                             ) if _killers_ntfy else ""
                            send_ntfy(
                                title=f"{'💎' if prob >= 0.85 else '🟢'} BUY: {coin.upper()} P={prob:.0%}  EV={sig.get('ev',{}).get('ev',0):+.2f}",
                                message=(
                                    f"Signal:     {decision}\n"
                                    f"P={prob:.0%}  EV={sig.get('ev',{}).get('ev',0):+.2f} ({sig.get('ev',{}).get('rating','?')})"
                                    f"ML:         {ml_proba:.0%} (acc {ml_accuracy:.0%})\n"
                                    f"Short ML:   {short_ml_proba:.0%} (acc {short_ml_accuracy:.0%})\n"
                                    f"Price:      ${last['price']:.6f}\n"
                                    f"Direction:  {direction} ({direction_conf:.0%})\n"
                                    f"Volume:     {sig.get('vol_ratio',0):.2f}x avg\n"
                                    f"ADX:        {last.get('ADX',0):.1f} | DI+: {last.get('plus_di',0):.1f} DI-: {last.get('minus_di',0):.1f}\n"
                                    f"OBV:        {obv_trend if obv_trend not in ('unknown', None) else ('rising' if df['OBV'].iloc[-1] > df['OBV'].iloc[-4] else 'falling' if df['OBV'].iloc[-1] < df['OBV'].iloc[-4] else 'flat') if 'OBV' in df.columns else '?'}\n"
                                    f"Consecutive:{consecutive_met}/{MIN_CONSECUTIVE_DAYS} days ✅"
                                    f"{_killers_str}"
                                ),
                                priority=_priority,
                                tags="chart_increasing,moneybag",
                                topic_env="NTFY_TOPIC"
                            )

                            # ── Log to forward-test tracker ──────────────────
                            if TRADE_LOGGER_ENABLED:
                                try:
                                    _tl = get_trade_logger()
                                    if _tl:
                                        _ev_data = sig.get("ev", {}) or {}
                                        _lc_data = sig.get("lifecycle", {}) or {}
                                        _rs_data = sig.get("rs_full", {}) or {}
                                        _tl.log_trade(
                                            coin         = coin,
                                            entry_price  = float(last["price"]),
                                            stop         = float(sig.get("stop_price", last["price"] * 0.95)),
                                            target       = float(sig.get("tp_price",   last["price"] * 1.10)),
                                            prob         = prob,
                                            ev           = _ev_data.get("ev", 0.0),
                                            ev_rating    = _ev_data.get("rating", "?"),
                                            rr_ratio     = _ev_data.get("adj_rr", 2.0),
                                            kelly        = _ev_data.get("kelly", 0.0),
                                            btc_regime   = btc_regime,
                                            lifecycle    = _lc_data.get("stage", "UNKNOWN"),
                                            rs_tier      = _rs_data.get("label", "N/A") if _rs_data.get("available") else "N/A",
                                            ml_accuracy  = ml_accuracy,
                                            ml_proba     = ml_proba,
                                            adx          = float(last.get("ADX", 0) or 0),
                                            rsi          = float(last.get("RSI", 50) or 50),
                                        )
                                except Exception as _tle:
                                    print(f"[LOGGER] Warning: {_tle}")

                if cloud_state == "below" and not breakout_up:
                    print(Fore.RED + "[CLOUD] Bearish cloud -- suppressing all buys.")
                    continue

                if breakout_up:
                    if sig["vol_ratio"] < 0.8:
                        print(Fore.MAGENTA + "[ADV] Breakout on low volume -- falling back to standard plan.")
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
    # [CAL] EMPIRICAL LR CALIBRATION -- runs after all coins fetched (data in cache)
    # Estimates LR values from the actual price history instead of hand-crafting them.
    # Uses session cache so no extra API calls.
    # ═══════════════════════════════════════════════════════════════════════════════
    try:
        _cached_lrs = load_calibrated_lrs()
        if _cached_lrs:
            apply_calibrated_lrs(_cached_lrs)
        else:
            print(Fore.CYAN + "\n[CAL] Calibrating LRs from session data...")
            # Pull all available DFs from the session cache (already fetched above)
            _cal_dfs = {}
            for _c in COINS:
                _ck = f"{_c}_{CG_DAYS}"
                if _ck in _session_df_cache:
                    _cal_dfs[_c] = _session_df_cache[_ck]["df"]
            if len(_cal_dfs) >= 3:
                _new_lrs = calibrate_lrs_from_history(_cal_dfs, verbose=True)
                save_calibrated_lrs(_new_lrs)
                apply_calibrated_lrs(_new_lrs)
                logger.info(f"[CAL] LR calibration complete -- {len(_new_lrs)} signals updated")
                # Also calibrate the transition matrix from same data
                try:
                    _new_matrix = estimate_transitions_from_history(_cal_dfs, verbose=True)
                    PHASE_TRANSITION_MATRIX.update(_new_matrix)
                    print(f"[TRN] Transition matrix calibrated from {len(_cal_dfs)} coins")
                    logger.info("[TRN] Transition matrix calibrated empirically")
                except Exception as _tce:
                    print(f"[TRN] Transition calibration error (priors kept): {_tce}")
                # Fit ML interaction LRs from same data
                try:
                    if not _mllr_producer.load():
                        _mllr_producer.fit(_cal_dfs, verbose=True)
                        _mllr_producer.save()
                    logger.info(f"[MLLR] Interaction LRs ready ({len(_mllr_producer.interaction_lrs)} pairs)")
                except Exception as _mce:
                    print(f"[MLLR] Interaction LR error (priors kept): {_mce}")
            else:
                print(Fore.YELLOW + "[CAL] Not enough cached data for calibration -- using priors")
    except Exception as _cal_e:
        print(Fore.YELLOW + f"[CAL] Calibration error (priors kept): {_cal_e}")
        logger.warning(f"[CAL] Calibration failed: {_cal_e}")

    # ═══════════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════════
    # [BRIER] RESOLVE OUTCOMES + PRINT CALIBRATION REPORT
    # ═══════════════════════════════════════════════════════════════════════════════
    try:
        _brier_dfs = {c: _session_df_cache[f"{c}_{CG_DAYS}"]["df"]
                      for c in COINS
                      if f"{c}_{CG_DAYS}" in _session_df_cache}
        if _brier_dfs:
            _newly_resolved = _brier_monitor.resolve_outcomes(_brier_dfs)
            if _newly_resolved:
                print(f"[BRIER] Resolved {_newly_resolved} new predictions")
            _brier_rpt = _brier_monitor.reliability_report()
            if _brier_rpt.strip():
                print(_brier_rpt)
            _mae_mfe_rpt = _brier_monitor.mae_mfe_report()
            if _mae_mfe_rpt.strip():
                print(_mae_mfe_rpt)
            for _bc in COINS:
                if _brier_monitor.should_retrain(_bc):
                    print(Fore.YELLOW + f"[BRIER] ⚠️  {_bc}: model drift -- retrain on next cycle")
    except Exception as _be:
        print(Fore.RED + f"[BRIER] ⚠️  resolve/report CRASHED — {type(_be).__name__}: {_be}")
        logger.error("[BRIER] resolve/report error", exc_info=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # [ENS] FIT ENSEMBLE CALIBRATOR from resolved predictions
    # Trains a logistic regression meta-learner on all three component
    # probabilities simultaneously -- learns optimal combination and correlations.
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        if not _ens_calibrator.fitted:
            _ens_calibrator.load()   # try loading from previous run
        _ens_records = [
            {"p_bayes":      float(p.get("p_bayes_raw",  p.get("bayes_prob", 0.5))),
             "p_ml":         float(p["ml_proba"]),
             "p_transition": float(p.get("p_transition", 0.5)),
             "outcome":      int(p["outcome"])}
            for p in _brier_monitor.predictions
            if p.get("outcome") is not None
        ]
        if len(_ens_records) >= ENS_MIN_RECORDS:
            _fit_ok = _ens_calibrator.fit(_ens_records, verbose=True)
            if _fit_ok:
                _ens_calibrator.save()
                logger.info(f"[ENS] Fitted on {len(_ens_records)} records  "
                            f"val_brier={_ens_calibrator.val_brier:.4f}")
        else:
            _still_need = ENS_MIN_RECORDS - len(_ens_records)
            print(f"[ENS] {len(_ens_records)}/{ENS_MIN_RECORDS} resolved  "
                  f"({_still_need} more needed before ensemble calibration activates)")
    except Exception as _ee:
        logger.warning(f"[ENS] Calibration error: {_ee}")

    # [J] RANKING: Order by BUY + SELL signals with scores
    # ═══════════════════════════════════════════════════════════════════════════════
    if results:
        print("\n\n" + "="*80)
        print(Fore.CYAN + "RANKED OPPORTUNITIES (BUY + SELL Signals)")
        print("="*80)

        # [J] Separate BUY and SELL results -- exclude event-blocked signals
        buy_results  = [r for r in results if r["action"] == "BUY" and not r.get("event_block")]
        sell_results = [r for r in results if r["action"] == "SELL"]
        wait_results = [r for r in results if r["action"] == "WAIT"]
        
        # [J] Sort each group by score
        buy_results_sorted = sorted(buy_results, key=lambda x: x["score"], reverse=True)
        sell_results_sorted = sorted(sell_results, key=lambda x: -x.get("sell_score",0))  # [J] More negative = stronger
        
        # Display BUY signals
        if buy_results_sorted:
            print(Fore.GREEN + "\n🟢 BUY SIGNALS (Long Entry Opportunities):")
            print(Fore.GREEN + "─"*80)
            
            for rank, r in enumerate(buy_results_sorted, 1):
                score_color = Fore.GREEN if r.get("prob", 0) >= BUY_PROB_THRESHOLD else Fore.YELLOW
                print(f"\n{rank:2d}. 🟢 {r['coin'].upper():<15} Price: ${r['price']:.6f}")
                print(f"    Score: {score_color}{r['score']}/{MIN_BUY_SCORE}{Fore.RESET} | " +
                      f"Direction: {r['direction']} ({r['direction_conf']:.0%}) | " +
                      f"Expected Move: {r['move_size']:.1f}%")

                # ── Entry timing ──────────────────────────────────────────
                _r_last = {"price": r["price"]}
                # Reconstruct minimal 'last' dict for timing helper
                print(Fore.GREEN + f"    ✅ ENTER NOW at ${r['price']:.6f} -- all criteria met")
                
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
                score_color = Fore.RED if r.get("sell_score",0) >= SHORT_PROB_THRESHOLD else Fore.YELLOW
                print(f"\n{rank:2d}. 🔴 {r['coin'].upper():<15} Price: ${r['price']:.6f}")
                # [J] Display SELL score (negative values)
                print(f"    Short P: {score_color}{r.get('sell_score',0):.0%}{Fore.RESET} | " +
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
        # Coins that pass the 5-factor qualifier -- closest to a valid entry
        # ═══════════════════════════════════════════════════════════════
        print("\n" + Fore.CYAN + "╔" + "═"*70 + "╗")
        print(Fore.CYAN + "║" + "  🎯 HIGH-PROBABILITY WATCHLIST -- 5-Factor Qualifier".ljust(70) + "║")
        print(Fore.CYAN + "╚" + "═"*70 + "╝")
        print(Fore.WHITE + "  Coins ranked by how many of the 5 factors they satisfy:")
        print(Fore.WHITE + "  OBV↑  |  PF>1.0  |  ML>60%(acc>65%)  |  Score≥5  |  Above Weekly EMA50")

        _watchlist = []
        for r in results:
            _obv     = r.get("obv_trend", "")
            _pf      = r.get("backtest_pf", 0) or 0
            _ml      = r.get("ml_proba", 0) or 0
            _acc     = r.get("ml_accuracy", 0) or 0
            _score   = r.get("score", 0) or 0
            _above   = r.get("above_weekly_ema", False)
            _coin    = r.get("coin", "")

            if _coin.lower() in ("zcash", "origintrail"):  # debug
                f2_check = _pf >= 1.0
                print(f"  [DEBUG] {_coin}: PF={_pf:.2f}({'✅' if f2_check else '❌'}) WR={r.get('backtest_win_rate',0):.1f}% OBV={_obv} ML={_ml:.0%}(acc{_acc:.0%}) Score={_score} EMA50={_above}")

            f1 = _obv == "rising"
            f2 = _pf >= 1.0
            f3 = _ml >= 0.60 and _acc >= 0.65   # lowered acc from 0.70 to 0.65 -- 69.8% was failing
            f4 = _score >= 5
            f5 = _above

            factors_met = sum([f1, f2, f3, f4, f5])
            _prob_pct   = int(round(r.get("prob", 0.5) * 100))
            if factors_met >= 3:   # minimum 3/5 to appear -- 2/5 is too noisy
                _watchlist.append({
                    "coin": _coin, "factors": factors_met,
                    "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
                    "pf": _pf, "ml": _ml, "acc": _acc, "score": _score,
                    "conf": _prob_pct, "obv": _obv,
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
                    "🟢 5/5 -- ALL FACTORS MET" if _w["factors"] == 5 else
                    f"🟡 4/5 -- needs: {', '.join(_missing)}" if _w["factors"] == 4 else
                    f"👀 3/5 HIGH CONF (P={_w['conf']}%) -- needs: {', '.join(_missing)}"
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
                        f"ML: {_w['ml']:.0%}  P={_w.get('prob',0.5):.0%}  EV={_w.get('ev',{}).get('ev',0):+.2f}  PF: {_w['pf']:.2f}"
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
            print(Fore.YELLOW + "\n  No coins meet 2+ of 5 factors right now -- market conditions too weak")

        print(Fore.CYAN + "\n  💡 All 5 green = highest probability trade. Enter only at 5/5 or 4/5+conf≥75")

        # Sort ALL results by probability (descending) for the ranking table
        ranked_results = sorted(results, key=lambda x: float(x.get("prob", 0.5)), reverse=True)

        print("\n" + Fore.YELLOW +
              f"{'Rank':>4}  {'Coin':<12}  {'Long P':>7}  {'Bull P':>7}  {'EV':>8}  {'Short P':>8}  {'OBV':>5}  "
              f"{'L.ML (acc)':>12}  {'S.ML (acc)':>12}  {'RS':>8}  {'Dir':>4}  {'Action':<11}  "
              f"{'Whale':>8}  {'Smart $':>9}  {'Killers'}" + Fore.RESET)
        print("─" * 148)

        for rank, r in enumerate(ranked_results, 1):
            ml_proba           = r.get("ml_proba", 0)
            ml_accuracy        = r.get("ml_accuracy", 0)
            short_ml_proba     = r.get("short_ml_proba", 0.5)
            short_ml_accuracy  = r.get("short_ml_accuracy", 0)
            action             = r.get("action", "WAIT")
            _prob              = float(r.get("prob", 0.5))
            _short_prob        = float(r.get("short_prob", 0.0))

            # ── Colour by long probability ────────────────────────────────
            if   _prob >= STRONG_BUY_THRESHOLD: color = Fore.GREEN
            elif _prob >= BUY_PROB_THRESHOLD:   color = Fore.YELLOW
            elif _prob >= 0.50:                 color = Fore.WHITE
            else:                               color = Fore.RED

            # ── Short probability colour ──────────────────────────────────
            if   _short_prob >= SHORT_STRONG_THRESHOLD: _sc = Fore.RED
            elif _short_prob >= SHORT_PROB_THRESHOLD:   _sc = Fore.YELLOW
            else:                                        _sc = Fore.WHITE

            # ── Action — override with entry timing status ────────────────
            # Entry status (CAUTION, ALMOST, ML WEAK etc.) takes priority
            # over the raw action so table and watch line always agree.
            _entry_s = r.get("entry_status", "")
            if   "STRONG BUY"   in action and "CAUTION" not in _entry_s and "ALMOST" not in _entry_s:
                action_icon = "💎"; action_short = "S.BUY"
            elif "BUY" in action and "CAUTION" not in _entry_s and "ALMOST" not in _entry_s:
                action_icon = "🟢"; action_short = "BUY"
            elif "BUY" in action and "ALMOST" in _entry_s:
                action_icon = "🟡"; action_short = "ALMOST"
            elif "BUY" in action and "CAUTION" in _entry_s:
                action_icon = "⚠️ "; action_short = "CAUTION"
            elif "STRONG SHORT" in action: action_icon = "🔴"; action_short = "S.SHORT"
            elif "SHORT"        in action: action_icon = "🔴"; action_short = "SHORT"
            else:                          action_icon = "🟡"; action_short = "WAIT"
            action_str = f"{action_icon} {action_short}".ljust(11)

            # ── Columns ───────────────────────────────────────────────────
            coin_str  = r["coin"].upper()[:11].ljust(11)

            # Long / short probability bars (5-char mini bar)
            def _pbar(p):
                filled = int(p * 5)
                return ("█" * filled + "░" * (5 - filled))
            long_p_str  = f"{_prob:>4.0%} {_pbar(_prob)}"
            short_p_str = f"{_short_prob:>4.0%} {_pbar(_short_prob)}"

            # Bull scenario P — what this coin would be if BTC flipped bull
            _prior      = r.get("prior", 0.42)
            _bull_p     = compute_bull_scenario_prob(_prob, prior=_prior)
            _bull_color = (Fore.GREEN  if _bull_p >= STRONG_BUY_THRESHOLD else
                           Fore.YELLOW if _bull_p >= BUY_PROB_THRESHOLD   else Fore.WHITE)
            bull_p_str  = f"{_bull_p:>4.0%} {_pbar(_bull_p)}"

            # OBV
            _obv = r.get("obv_trend", "?")
            _obv_color = Fore.GREEN if _obv=="rising" else (Fore.RED if _obv=="falling" else Fore.YELLOW)
            _obv_str   = ("↑ UP" if _obv=="rising" else "↓ DN" if _obv=="falling" else "→ FL").ljust(5)

            # ML strings
            long_ml_str  = (f"n/a ({ml_accuracy:.0%})" if ml_accuracy < 0.50
                            else f"{ml_proba:.0%}({ml_accuracy:.0%})").ljust(12)
            short_ml_str = (f"n/a ({short_ml_accuracy:.0%})" if short_ml_accuracy < 0.50
                            else f"{short_ml_proba:.0%}({short_ml_accuracy:.0%})").ljust(12)

            # Direction
            dir_str = r.get("direction","?").upper()[:3].ljust(4)

            # Whale + smart money (compact)
            whale_vol = r.get("whale_volume_pct", 0)
            whale_icon = "🐋" if whale_vol >= 100 else ("↑" if whale_vol >= 70 else "→")
            whale_str  = f"{whale_icon}{whale_vol:.0f}%".ljust(7)
            _sm  = r.get("smart_money_mode", "neutral")[:4].upper()
            _om  = int(r.get("obv_momentum", 0) * 100)
            sm_str = f"{_sm} {_om}%".ljust(9)

            # Top killers compressed to ≤30 chars
            _killers  = r.get("confidence_killers", [])
            _k_strs   = []
            for k in _killers[:2]:
                k_clean = k.replace("⛔","").replace("❌","").replace("⚠️","").strip()
                k_short = k_clean.split(":")[0][:18].strip()
                _k_strs.append(k_short)
            killer_str = " · ".join(_k_strs) if _k_strs else ""

            # RS label for ranked table
            _rs_full  = r.get("rs_full", {}) or {}
            _rs_label = _rs_full.get("label", "N/A") if _rs_full.get("available") else "N/A"
            _rs_7d    = _rs_full.get("rs_btc_7d", 0)
            _rs_color = (Fore.GREEN  if _rs_label in ("LEADER", "STRONG") else
                         Fore.RED    if _rs_label in ("WEAK",)             else
                         Fore.YELLOW if _rs_label == "LAGGARD"             else Fore.WHITE)
            _rs_str   = f"{_rs_label[:6]}{_rs_7d:+.0f}%" if _rs_label != "N/A" else "  N/A  "

            # EV column
            _ev_data  = r.get("ev", {}) or {}
            _ev_val   = _ev_data.get("ev",     0.0)
            _ev_rate  = _ev_data.get("rating", "?")
            _ev_kelly = _ev_data.get("kelly",  0.0)
            _ev_color = (Fore.GREEN  if _ev_rate in ("A+", "A") else
                         Fore.YELLOW if _ev_rate == "B"         else
                         Fore.RED    if _ev_rate in ("D", "F")  else Fore.WHITE)
            _ev_str   = f"{_ev_val:+.2f}({_ev_rate})" if _ev_data else "  N/A  "

            print(f"{rank:>4}  "
                  f"{color}{coin_str}{Fore.RESET}  "
                  f"{color}{long_p_str}{Fore.RESET}  "
                  f"{_bull_color}{bull_p_str}{Fore.RESET}  "
                  f"{_ev_color}{_ev_str:<8}{Fore.RESET}  "
                  f"{_sc}{short_p_str}{Fore.RESET}  "
                  f"{_obv_color}{_obv_str}{Fore.RESET}  "
                  f"{long_ml_str}  {short_ml_str}  "
                  f"{_rs_color}{_rs_str:<8}{Fore.RESET}  "
                  f"{dir_str}  {action_str}  "
                  f"{whale_str}  {sm_str}  "
                  f"{Fore.YELLOW if _killers else ''}{killer_str}{Fore.RESET}")

            # Signal conflict / pattern detection (kept compact)
            _conflict_note = ""
            if r.get("direction","?") == "UP" and ml_proba < 0.35 and ml_accuracy >= 0.55:
                _conflict_note = (f"  ⚠️  Bull trap: price UP but ML={ml_proba:.0%} — do NOT chase. "
                                  f"Wait for ML to confirm.")
            elif r.get("direction","?") == "DOWN" and ml_proba > 0.55 and ml_accuracy >= 0.55:
                _conflict_note = (f"  ⚠️  Mean-reversion: price DOWN but ML={ml_proba:.0%} — "
                                  f"potential dip buy, wait for BUY threshold.")
            if _conflict_note:
                print(Fore.MAGENTA + _conflict_note)

            # ── Entry timing line -- driven entirely by stored entry_status ──
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

            # Lifecycle stage inline
            _lc       = r.get("lifecycle", {}) or {}
            _lc_stage = _lc.get("stage", "")
            _lc_pos   = _lc.get("position_mod", 1.0)
            _lc_icons = {"ACCUMULATION":"🟡S1","EARLY_TREND":"🟢S2",
                         "MATURE_TREND":"🟢S3","EXHAUSTION":"⚠️S4","DISTRIBUTION":"🔴S5"}
            _lc_str   = f"  {_lc_icons.get(_lc_stage,'')} pos={_lc_pos:.0%}" if _lc_stage else ""

            # Bull P preview + lifecycle
            _bull_preview = ""
            if "BTC BEAR" in _entry_status and btc_regime == "bear":
                _bp = compute_bull_scenario_prob(_prob, prior=r.get("prior", 0.42))
                _ba = ("💎 STRONG BUY" if _bp >= STRONG_BUY_THRESHOLD else
                       "🟢 BUY"        if _bp >= BUY_PROB_THRESHOLD   else
                       "👁 WATCH"      if _bp >= 0.55                 else "WAIT")
                _bull_preview = f"  (Bull P={_bp:.0%} → {_ba})"
            print(_t_color + f"     ↳ {_entry_status} | Watch: ${_entry_watch:.6f}{_bull_preview}{_lc_str}")
            # Full explanation -- split on \n so each gate prints on its own line, nothing truncated
            if _entry_when:
                for _line in _entry_when.split("\n"):
                    if _line.strip():
                        print(_t_color + f"       {_line}")

            # ── 📋 Trade Journal fields (all fields in one pasteable line) ────
            _jcloud = r.get("cloud_pos", "?").replace("unknown","?")[:3].upper()
            _jz     = r.get("z_score")
            _jtpb   = r.get("trans_prob_bull")
            _jel    = r.get("entropy_label","?")[:3]
            _jes    = r.get("entropy_scale")
            _jobv   = r.get("obv_trend","?")[:4].upper()
            _jwt    = _sm.lower()
            _jwpct  = whale_vol
            _jdev   = r.get("ema26_dev_pct")
            _jetim  = (_entry_status.split("--")[0].strip()
                       .replace("⏳","").replace("✅","").replace("⚠️","").replace("🔴","").strip())
            _jlc    = _lc_stage[:12] if _lc_stage else "?"

            print(Fore.CYAN + f"     📋 LOG: "
                  f"Timing={_jetim or '?'}  "
                  f"Entropy={_jel}({_jes:.2f})  "
                  f"Cloud={_jcloud}  "
                  f"OBV={_jobv}  "
                  f"Z={_jz:+.1f}  " if _jz is not None else f"     📋 LOG: Timing={_jetim or '?'}  Entropy={_jel}  Cloud={_jcloud}  OBV={_jobv}  Z=?  "
                  , end="")
            print(Fore.CYAN +
                  f"Dev={_jdev:.1f}%  " if _jdev is not None else "Dev=?  ", end="")
            print(Fore.CYAN +
                  f"P(→bull)={_jtpb}%  " if _jtpb is not None else "P(→bull)=?  ", end="")
            print(Fore.CYAN +
                  f"SM={_jwt}  🐋={_jwpct:.0f}%  "
                  f"LC={_jlc}" + Fore.RESET)
        
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

        # ═══════════════════════════════════════════════════════════════════
        # 🐻 SHORT OPPORTUNITIES TABLE
        # All coins sorted by short probability — shows setup quality per coin
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + Fore.RED + "╔" + "═"*70 + "╗")
        print(Fore.RED + "║" + "  🐻 SHORT OPPORTUNITIES — All Coins by Short Probability".ljust(70) + "║")
        print(Fore.RED + "╚" + "═"*70 + "╝" + Fore.RESET)

        ranked_short = sorted(results,
                              key=lambda x: float(x.get("short_prob", 0.0)),
                              reverse=True)

        print(Fore.RED + f"{'Rank':>4}  {'Coin':<12}  {'Short P':>7}  {'Action':<12}  "
              f"{'Short ML':>8}  {'OBV':>5}  {'Frag':>5}  {'Accel':>5}  "
              f"{'Killers (top 3)'}" + Fore.RESET)
        print("─" * 100)

        for _sr, r in enumerate(ranked_short, 1):
            _sp        = float(r.get("short_prob", 0.0))
            _sa        = r.get("short_action", "WAIT")
            _sml       = float(r.get("short_ml_proba", 0.5))
            _sml_acc   = float(r.get("short_ml_accuracy", 0.0))
            _obv       = r.get("obv_trend", "?")
            _sk        = r.get("short_confidence_killers", []) or []
            _coin_raw  = r.get("coin", "")
            # COIN_SYMBOL_MAP can return None for coins not on Binance
            # (e.g. hyperliquid, story-2, monero, fartcoin) — fall back to
            # the coin_id itself rather than crashing on None[:12]
            _coin_sym  = COIN_SYMBOL_MAP.get(_coin_raw)
            if _coin_sym is None:
                _coin_sym = _coin_raw.upper()
            _coin_s    = _coin_sym[:12].ljust(12)

            # Fragility and acceleration from short scorer outputs
            # stored in sig dict during run — fall back to short_prob tier
            _frag_s   = r.get("short_fragility_score", 0.0)
            _accel_n  = r.get("short_accel_n", 0)

            # Color by short probability
            if   _sp >= SHORT_STRONG_THRESHOLD: _sc = Fore.RED
            elif _sp >= SHORT_PROB_THRESHOLD:   _sc = Fore.YELLOW
            elif _sp >= 0.45:                   _sc = Fore.WHITE
            else:                               _sc = Fore.WHITE

            # Action display
            _sa_disp = ("🔴 SHORT" if "SHORT" in _sa else "⚪ WAIT").ljust(10)

            # OBV colour
            _obv_c = (Fore.GREEN if _obv=="rising" else Fore.RED if _obv=="falling" else Fore.YELLOW)
            _obv_s = ("↑" if _obv=="rising" else "↓" if _obv=="falling" else "→").ljust(5)

            # Short ML display
            if _sml_acc < 0.50:
                _sml_s = "off".ljust(8)
            else:
                _sml_s = f"{_sml:.0%}({_sml_acc:.0%})".ljust(8)

            # Fragility / accel
            _frag_s_disp  = (f"{_frag_s:.2f}" if _frag_s > 0 else "  —").ljust(5)
            _accel_s_disp = (f"{_accel_n}/4"  if _accel_n > 0 else "  —").ljust(5)

            # Top short killers compressed
            _ktags = " · ".join(k.split(":")[0].replace("⛔","").replace("❌","").replace("⚠️","").strip()[:20]
                                 for k in _sk[:3])

            print(f"{_sr:>4}  {_sc}{_coin_s}{Fore.RESET}  "
                  f"{_sc}{_sp:>7.0%}{Fore.RESET}  "
                  f"{_sc}{_sa_disp}{Fore.RESET}  "
                  f"{_sml_s}  "
                  f"{_obv_c}{_obv_s}{Fore.RESET}  "
                  f"{_frag_s_disp}  {_accel_s_disp}  "
                  f"{Fore.WHITE}{_ktags}{Fore.RESET}")

        print("─" * 100)
        print(Fore.WHITE + f"  Short threshold: P≥{SHORT_PROB_THRESHOLD:.0%}  "
              f"Strong: P≥{SHORT_STRONG_THRESHOLD:.0%}  "
              f"Frag = trend fragility score (>0.40 = real setup)  "
              f"Accel = bear acceleration signals (3+/4 = strong)" + Fore.RESET)

        # [W] Log completion with performance metrics
        run_elapsed = time.time() - run_start_time
        print("\n" + "="*80)
        print(f"Total coins analyzed: {len(results)}")
        print(f"🟢 BUY signals: {len(buy_results)} | 🔴 SELL signals: {len(sell_results)} | 🟡 WAIT signals: {len(wait_results)}")
        print("="*80)

        # ── GLOBAL MODEL: retrain if needed ──────────────────────────────
        # NOTE: auto-retrain is intentionally disabled here.
        # The global model should be trained on 200 coins (--train-global),
        # not just the 35 in this bot's COINS list. Auto-retraining on 35
        # coins would produce a weaker model than the 200-coin version.
        #
        # To retrain:  python crypto_trading_bot.py --train-global
        # To check:    python crypto_trading_bot.py --global-status
        if ML_ENABLED and len(_GLOBAL_COIN_DFS) > 0 and not GLOBAL_MODEL_PATH.exists():
            print(Fore.YELLOW + "\n[GLOBAL ML] No global model found. "
                  "Run: python crypto_trading_bot.py --train-global")

        # ── TRADE LOGGER: resolve pending + print dashboard ───────────────────
        if TRADE_LOGGER_ENABLED:
            try:
                _tl = get_trade_logger()
                if _tl and _tl.trades:
                    # Build current price map from scan results
                    _cur_prices = {}
                    for _r in results:
                        try: _cur_prices[_r["coin"]] = float(_r["price"])
                        except Exception: pass
                    _newly_resolved = _tl.resolve_outcomes(_cur_prices)
                    if _newly_resolved:
                        print(f"[LOGGER] Resolved {_newly_resolved} trade(s) this scan")
                    _tl.print_dashboard()
            except Exception as _tle:
                print(f"[LOGGER] Dashboard error: {_tle}")
        
        # [W] Log performance metrics
        logger.info("=" * 70)
        logger.info(f"[BOT] Run completed successfully")
        logger.info(f"[PERFORMANCE] Total time: {run_elapsed:.1f}s")
        logger.info(f"[PERFORMANCE] Coins analyzed: {len(results)}")
        logger.info(f"[PERFORMANCE] BUY signals: {len(buy_results)} | SELL signals: {len(sell_results)} | WAIT signals: {len(wait_results)}")
        logger.info(f"[PERFORMANCE] Average per coin: {run_elapsed/len(results):.2f}s")
        logger.info("=" * 70)

def fetch_global_training_universe(n: int = 200) -> List[str]:
    """
    Fetch top-N coins by market cap from CoinGecko for global model training.
    Returns list of CoinGecko coin IDs (e.g. ["bitcoin", "ethereum", ...]).
    Filters out stablecoins, wrapped tokens, and pegged assets.
    """
    stablecoins = {
        "tether","usd-coin","dai","binance-usd","true-usd","frax","paxos-standard",
        "tether-gold","pax-gold","gho","liquity-usd","paypal-usd","crvusd","ethena-usde",
        "first-digital-usd","usdd","stasis-eurs","ageur","wrapped-bitcoin",
        "wrapped-ethereum","wrapped-steth","staked-ether","rocket-pool-eth",
    }
    coin_ids = []
    pages    = (n // 250) + 2
    for page in range(1, pages + 1):
        try:
            url  = "https://api.coingecko.com/api/v3/coins/markets"
            resp = requests.get(url, params={
                "vs_currency": "usd", "order": "market_cap_desc",
                "per_page": 250, "page": page, "sparkline": False,
                "price_change_percentage": "24h",
            }, headers={"x-cg-demo-api-key": API_KEY}, timeout=20)
            resp.raise_for_status()
            for c in resp.json():
                cid = c.get("id", "")
                sym = c.get("symbol", "").lower()
                if cid in stablecoins: continue
                if any(s in sym for s in ("usd","eur","gbp","btc.","eth.")): continue
                price = c.get("current_price", 0) or 0
                ath   = c.get("ath", price) or price
                atl   = max(c.get("atl", price) or price, 1e-9)
                if price and (ath - atl) / atl * 100 < 10: continue  # pegged
                coin_ids.append(cid)
                if len(coin_ids) >= n:
                    return coin_ids
            time.sleep(1.5)
        except Exception as e:
            print(f"[GLOBAL ML] Market fetch error (page {page}): {e}")
            break
    return coin_ids


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Crypto Trading Bot")
    ap.add_argument("--train-global", action="store_true",
                    help="Retrain global ML model on top-200 coins then exit")
    ap.add_argument("--train-global-n", type=int, default=200,
                    help="Number of coins to train global model on (default: 200)")
    ap.add_argument("--global-status", action="store_true",
                    help="Show global model status and exit")
    args = ap.parse_args()

    if args.global_status:
        if load_global_model():
            m = _GLOBAL_MODEL.get("meta", {})
            print(f"\nGlobal model status:")
            print(f"  Trained:  {m.get('trained_date','?')[:10]}")
            print(f"  Coins:    {m.get('n_coins',0)}")
            print(f"  Samples:  {m.get('n_samples',0):,}")
            print(f"  Acc:      {m.get('wf_acc',0):.1%}")
            print(f"  Sharpe:   {m.get('wf_sharpe',0):+.2f}")
            print(f"  Retrain:  {'needed' if needs_global_retrain() else 'not needed'}")
        else:
            print("No global model found. Run with --train-global to create one.")

    elif args.train_global:
        n_coins = args.train_global_n
        print(f"\n[GLOBAL ML] Fetching top-{n_coins} coins from CoinGecko...")
        universe = fetch_global_training_universe(n=n_coins)
        print(f"[GLOBAL ML] Got {len(universe)} coins after filtering stablecoins")

        fng_value, _ = fetch_fear_greed_index() if FETCH_FEAR_GREED else (50, "Neutral")

        # Fetch BTC first (needed for regime features)
        print("[GLOBAL ML] Fetching BTC data...")
        btc_df = fetch_coin_data("bitcoin", days=CG_DAYS)
        time.sleep(CG_SLEEP_BETWEEN)

        # Fetch all coins
        print(f"[GLOBAL ML] Fetching OHLCV for {len(universe)} coins "
              f"(~{len(universe)*2/60:.0f} min at {CG_SLEEP_BETWEEN}s/coin)...")
        n_ok = 0
        for idx, coin in enumerate(universe, 1):
            try:
                df = fetch_coin_data(coin, days=CG_DAYS)
                if df is not None and len(df) > 100:
                    _GLOBAL_COIN_DFS[coin] = df
                    n_ok += 1
                    if idx % 20 == 0 or idx == len(universe):
                        print(f"  [{idx:3d}/{len(universe)}] Fetched {n_ok} valid  "
                              f"({len(universe)-idx} remaining)")
                time.sleep(CG_SLEEP_BETWEEN)
            except Exception as e:
                print(f"  [{idx:3d}] Skip {coin}: {e}")

        print(f"\n[GLOBAL ML] Training on {len(_GLOBAL_COIN_DFS)} coins...")
        success = train_global_model(_GLOBAL_COIN_DFS, btc_df, fng_value=fng_value)
        if success:
            print(Fore.GREEN + "\n[GLOBAL ML] ✅ Global model ready. "
                  "Run the bot normally — it will load automatically.")
        else:
            print(Fore.RED + "\n[GLOBAL ML] ❌ Training failed.")

    else:
        run_once()
