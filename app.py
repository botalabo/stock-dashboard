"""
==========================================================
  Stock Dashboard — Flask Backend
  GET  /              index.html 配信
  GET  /api/tickers   監視銘柄一覧 + 現在価格・変動率
  GET  /api/detail/<symbol>  銘柄詳細 + ファンダメンタルズスコア
  GET  /api/chart/<symbol>   ローソク足 OHLCV データ
  GET  /api/alerts/<symbol>  モッククジラアラート
  GET  /api/watchlist        ウォッチリスト取得
  POST /api/watchlist        ウォッチリスト追加/削除
==========================================================
"""

import hashlib
import json
import logging
import os
import threading
import time
import webbrowser
from pathlib import Path

import yfinance as yf
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file

# ============================================================
# .env 読み込み
# ============================================================
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stock-dashboard")

# ============================================================
# 設定
# ============================================================
def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


_DATA_DIR = Path(_env("DATA_DIR", "")) if _env("DATA_DIR") else Path(__file__).parent
_DATA_DIR.mkdir(parents=True, exist_ok=True)

PORT = int(_env("PORT", "5000"))

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD",
]

# ============================================================
# ウォッチリスト永続化
# ============================================================
_WATCHLIST_FILE = _DATA_DIR / "watchlist.json"


def _load_watchlist() -> list[str]:
    if _WATCHLIST_FILE.exists():
        try:
            data = json.loads(_WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _save_watchlist(wl: list[str]) -> None:
    _WATCHLIST_FILE.write_text(
        json.dumps(wl, ensure_ascii=False), encoding="utf-8"
    )


watchlist: list[str] = _load_watchlist()

# ============================================================
# インメモリキャッシュ (TTL ベース)
# ============================================================
_cache: dict[str, tuple] = {}  # key -> (data, timestamp)


def _get_cached(key: str, fetch_fn, ttl: int = 300):
    """TTL 秒以内ならキャッシュを返す。超えたら fetch_fn() で取得して更新。"""
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < ttl:
            return data
    try:
        data = fetch_fn()
        _cache[key] = (data, now)
        return data
    except Exception as e:
        log.warning("Cache fetch error for %s: %s", key, e)
        if key in _cache:
            return _cache[key][0]
        return None


# ============================================================
# yfinance データ取得
# ============================================================
def _fetch_ticker_info(symbol: str) -> dict | None:
    """企業情報 + ファンダメンタルズ指標を取得する。"""
    try:
        t = yf.Ticker(symbol)
        info = t.info
        if not info or not info.get("shortName"):
            log.warning("[%s] info empty or no shortName", symbol)
            return None
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        log.info("[%s] OK - %s price=$%s", symbol, info.get("shortName"), price)
        return info
    except Exception as e:
        log.warning("[%s] info fetch FAILED: %s", symbol, e)
    # フォールバック: fast_info + history で最低限のデータを構築
    try:
        t = yf.Ticker(symbol)
        fi = t.fast_info
        price = getattr(fi, "last_price", 0) or 0
        prev = getattr(fi, "previous_close", 0) or 0
        if price > 0:
            log.info("[%s] FALLBACK fast_info price=$%.2f", symbol, price)
            return {
                "shortName": symbol,
                "currentPrice": round(price, 2),
                "previousClose": round(prev, 2),
                "regularMarketPreviousClose": round(prev, 2),
            }
    except Exception as e2:
        log.warning("[%s] fallback also failed: %s", symbol, e2)
    return None


def _fetch_chart_data(symbol: str, period: str) -> list[dict] | None:
    """ローソク足 OHLCV を取得する。"""
    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}
    if period not in valid_periods:
        period = "3mo"
    try:
        t = yf.Ticker(symbol)
        interval = "1d"
        if period == "1mo":
            interval = "1h"
        df = t.history(period=period, interval=interval)
        if df.empty:
            return []
        records = []
        for idx, row in df.iterrows():
            ts = int(idx.timestamp())
            records.append({
                "time": ts,
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
            })
        return records
    except Exception as e:
        log.warning("Chart fetch error for %s: %s", symbol, e)
    return None


# ============================================================
# ファンダメンタルズスコアリング
# ============================================================
def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _fmt_num(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def _score_profitability(info: dict) -> float:
    scores = []
    pm = info.get("profitMargins")
    if pm is not None:
        scores.append(_clamp(pm * 100, 0, 20))
    roe = info.get("returnOnEquity")
    if roe is not None:
        scores.append(_clamp(roe * 50, 0, 20))
    gm = info.get("grossMargins")
    if gm is not None:
        scores.append(_clamp(gm * 40, 0, 20))
    return sum(scores) / len(scores) if scores else 10


def _score_growth(info: dict) -> float:
    scores = []
    rg = info.get("revenueGrowth")
    if rg is not None:
        scores.append(_clamp((rg + 0.1) * 40, 0, 20))
    eg = info.get("earningsGrowth")
    if eg is not None:
        scores.append(_clamp((eg + 0.1) * 40, 0, 20))
    return sum(scores) / len(scores) if scores else 10


def _score_health(info: dict) -> float:
    scores = []
    dte = info.get("debtToEquity")
    if dte is not None:
        scores.append(_clamp(20 - dte / 10, 0, 20))
    cr = info.get("currentRatio")
    if cr is not None:
        scores.append(_clamp(cr * 8, 0, 20))
    return sum(scores) / len(scores) if scores else 10


def _score_value(info: dict) -> float:
    scores = []
    fpe = info.get("forwardPE") or info.get("forwardEps")
    if isinstance(fpe, (int, float)) and fpe > 0:
        scores.append(_clamp(20 - fpe / 3, 0, 20))
    ptb = info.get("priceToBook")
    if ptb is not None and ptb > 0:
        scores.append(_clamp(20 - ptb * 2, 0, 20))
    return sum(scores) / len(scores) if scores else 10


def _score_momentum(info: dict) -> float:
    scores = []
    w52 = info.get("52WeekChange")
    if w52 is not None:
        scores.append(_clamp((w52 + 0.5) * 20, 0, 20))
    sr = info.get("shortRatio")
    if sr is not None:
        scores.append(_clamp(20 - sr * 3, 0, 20))
    return sum(scores) / len(scores) if scores else 10


def compute_fundamentals(info: dict) -> dict:
    categories = {
        "profitability": (_score_profitability(info), 0.25),
        "growth":        (_score_growth(info),        0.25),
        "health":        (_score_health(info),        0.20),
        "value":         (_score_value(info),         0.15),
        "momentum":      (_score_momentum(info),      0.15),
    }
    total = 0
    details = {}
    for key, (raw_score, weight) in categories.items():
        scaled = (raw_score / 20) * 100
        details[key] = round(scaled, 1)
        total += scaled * weight

    total = round(total, 1)
    if total >= 80:
        grade = "A"
    elif total >= 60:
        grade = "B"
    elif total >= 40:
        grade = "C"
    elif total >= 20:
        grade = "D"
    else:
        grade = "E"

    # スコア根拠の元データ
    reasons = {
        "profitability": {
            "利益率": _fmt_pct(info.get("profitMargins")),
            "ROE": _fmt_pct(info.get("returnOnEquity")),
            "粗利率": _fmt_pct(info.get("grossMargins")),
        },
        "growth": {
            "売上成長率": _fmt_pct(info.get("revenueGrowth")),
            "利益成長率": _fmt_pct(info.get("earningsGrowth")),
        },
        "health": {
            "D/E比率": _fmt_num(info.get("debtToEquity")),
            "流動比率": _fmt_num(info.get("currentRatio")),
        },
        "value": {
            "P/E (予想)": _fmt_num(info.get("forwardPE")),
            "P/B": _fmt_num(info.get("priceToBook")),
        },
        "momentum": {
            "52週変動": _fmt_pct(info.get("52WeekChange")),
            "空売比率": _fmt_num(info.get("shortRatio")),
        },
    }

    return {"score": total, "grade": grade, "categories": details, "reasons": reasons}


# ============================================================
# モッククジラアラート生成
# ============================================================
ALERT_TYPES = [
    "大口ブロック取引",
    "異常オプション活動",
    "ダークプール取引",
    "インサイダー取引",
]


def _generate_mock_alerts(symbol: str) -> list[dict]:
    seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    alerts = []
    now = int(time.time())
    day_sec = 86400

    for d in range(30):
        day_seed = seed ^ (d * 7919)
        count = day_seed % 4
        for i in range(count):
            item_seed = day_seed ^ (i * 1013)
            alert_type = ALERT_TYPES[item_seed % len(ALERT_TYPES)]
            ts = now - (d * day_sec) - ((item_seed % 28800) + 34200)

            if alert_type == "大口ブロック取引":
                shares = ((item_seed % 50) + 5) * 1000
                price_base = (item_seed % 300) + 50
                desc = f"{shares:,}株 @ ${price_base}.{item_seed % 100:02d}"
                impact = "HIGH" if shares >= 30000 else "MEDIUM"
            elif alert_type == "異常オプション活動":
                vol = ((item_seed % 100) + 10) * 100
                strike = (item_seed % 200) + 100
                cp = "Call" if item_seed % 2 == 0 else "Put"
                desc = f"{vol:,} {cp}契約, Strike ${strike}"
                impact = "HIGH" if vol >= 5000 else "MEDIUM"
            elif alert_type == "ダークプール取引":
                shares = ((item_seed % 80) + 10) * 1000
                desc = f"ダークプール経由 {shares:,}株"
                impact = "MEDIUM"
            else:
                action = "買い" if item_seed % 3 != 0 else "売り"
                shares = ((item_seed % 20) + 1) * 1000
                desc = f"インサイダー{action}: {shares:,}株"
                impact = "LOW" if shares < 10000 else "MEDIUM"

            alerts.append({
                "timestamp": ts,
                "type": alert_type,
                "description": desc,
                "impact": impact,
            })

    alerts.sort(key=lambda a: a["timestamp"], reverse=True)
    return alerts


# ============================================================
# Flask アプリ
# ============================================================
app = Flask(__name__)

_HTML_FILE = Path(__file__).parent / "index.html"


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
def serve_index():
    return send_file(_HTML_FILE, mimetype="text/html")


def _extract_price_from_info(symbol: str, info: dict) -> dict:
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", 0)
    prev = info.get("regularMarketPreviousClose") or info.get("previousClose", 0)
    change = 0
    change_pct = 0
    if price and prev and prev > 0:
        change = round(price - prev, 2)
        change_pct = round((change / prev) * 100, 2)
    return {
        "symbol": symbol,
        "price": round(price, 2) if price else 0,
        "change": change,
        "changePct": change_pct,
    }


@app.route("/api/tickers")
def get_tickers():
    results = []
    for i, sym in enumerate(DEFAULT_TICKERS):
        info = _get_cached(
            f"info:{sym}",
            lambda s=sym: _fetch_ticker_info(s),
            ttl=300,
        )
        if info:
            pd = _extract_price_from_info(sym, info)
            fund = compute_fundamentals(info)
            results.append({
                **pd,
                "name": info.get("shortName", sym),
                "grade": fund["grade"],
            })
        else:
            results.append({"symbol": sym, "name": sym, "price": 0, "change": 0, "changePct": 0, "grade": ""})
        if f"info:{sym}" not in _cache and i < len(DEFAULT_TICKERS) - 1:
            time.sleep(0.5)
    return jsonify({"tickers": results, "watchlist": watchlist})


@app.route("/api/detail/<symbol>")
def get_detail(symbol):
    symbol = symbol.upper()
    info = _get_cached(
        f"info:{symbol}",
        lambda: _fetch_ticker_info(symbol),
        ttl=600,
    )
    if not info:
        return jsonify({"error": "Ticker not found"}), 404

    fund = compute_fundamentals(info)
    pd = _extract_price_from_info(symbol, info)

    detail = {
        "symbol": symbol,
        "name": info.get("shortName", symbol),
        "longName": info.get("longName", ""),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "marketCap": info.get("marketCap", 0),
        "enterpriseValue": info.get("enterpriseValue", 0),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "priceToBook": info.get("priceToBook"),
        "dividendYield": info.get("dividendYield"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "averageVolume": info.get("averageVolume"),
        "beta": info.get("beta"),
        "profitMargins": info.get("profitMargins"),
        "returnOnEquity": info.get("returnOnEquity"),
        "revenueGrowth": info.get("revenueGrowth"),
        "earningsGrowth": info.get("earningsGrowth"),
        "debtToEquity": info.get("debtToEquity"),
        "currentRatio": info.get("currentRatio"),
        "longBusinessSummary": (info.get("longBusinessSummary") or "")[:300],
        "fundamentals": fund,
        **pd,
    }

    return jsonify(detail)


@app.route("/api/chart/<symbol>")
def get_chart(symbol):
    symbol = symbol.upper()
    period = request.args.get("period", "3mo")
    data = _get_cached(
        f"chart:{symbol}:{period}",
        lambda: _fetch_chart_data(symbol, period),
        ttl=300,
    )
    if data is None:
        return jsonify({"error": "Chart data not available"}), 404
    return jsonify({"symbol": symbol, "period": period, "data": data})


@app.route("/api/alerts/<symbol>")
def get_alerts(symbol):
    symbol = symbol.upper()
    alerts = _get_cached(
        f"alerts:{symbol}",
        lambda: _generate_mock_alerts(symbol),
        ttl=600,
    )
    return jsonify({"symbol": symbol, "alerts": alerts or []})


@app.route("/api/watchlist", methods=["GET"])
def get_watchlist():
    return jsonify({"watchlist": watchlist})


@app.route("/api/watchlist", methods=["POST"])
def update_watchlist():
    global watchlist
    body = request.get_json()
    symbol = body.get("symbol", "").upper()
    action = body.get("action", "add")

    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    if action == "add":
        if symbol not in watchlist:
            watchlist.append(symbol)
    elif action == "remove":
        watchlist = [s for s in watchlist if s != symbol]

    _save_watchlist(watchlist)
    return jsonify({"watchlist": watchlist})


# ============================================================
# エントリポイント
# ============================================================
if __name__ == "__main__":
    url = f"http://localhost:{PORT}/"
    print("=" * 50)
    print("  Stock Dashboard")
    print(f"  {url}")
    print("=" * 50)
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
