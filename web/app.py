#!/usr/bin/env python3
"""
FastAPI Web Application for Stock Pattern Prediction.

A web UI for the A-share stock pattern prediction system.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
import pandas as pd
import logging

# Import existing modules
from data.fetcher import AshareFetcher
from data.cache_manager import CacheManager
from core.landmark_detector import multi_timeframe_analysis, zigzag_detector
from core.sequence_extractor import extract_sequence
from data.stock_db import get_stock_db
from core.multi_layer_detector import MultiLayerDetector, LayerFilterParams
import find_pattern

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Pattern Predictor",
    description="A-share stock pattern prediction using time-based sequence analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates (use absolute paths)
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Custom StaticFiles with cache control
from starlette.staticfiles import StaticFiles
from starlette.responses import Response

class CachedStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            # Add cache control for JS files - no caching during development
            if path.endswith('.js'):
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'
        return response

app.mount("/static", CachedStaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize data fetcher and cache
fetcher = AshareFetcher()
cache = CacheManager("stock_cache.db")
stock_db = get_stock_db()


# === HTML Page Routes ===

@app.get("/")
async def home(request: Request):
    """Home page with stock search."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/analysis/{symbol}")
async def analysis(request: Request, symbol: str):
    """Analysis page with detailed chart."""
    return templates.TemplateResponse(
        "analysis.html",
        {"request": request, "symbol": symbol}
    )


@app.get("/history")
async def history(request: Request):
    """History page with validation (generic)."""
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/history/{symbol}")
async def history_symbol(request: Request, symbol: str):
    """History page with validation (symbol-specific)."""
    return templates.TemplateResponse(
        "history.html",
        {"request": request, "symbol": symbol}
    )


@app.get("/about")
async def about(request: Request):
    """About page."""
    return templates.TemplateResponse("about.html", {"request": request})


# === JSON API Routes ===

@app.get("/api/predict")
async def api_predict(
    symbol: str = Query(..., description="Stock code (6 digits)"),
    landmark_type: str = Query("low", description="Landmark type: low or high"),
    threshold: float = Query(0.05, description="ZigZag threshold (0.05 = 5%)")
):
    """
    Get prediction for a stock.

    Returns:
        JSON with predicted week, date range, confidence, and pattern.
    """
    try:
        # Normalize symbol
        symbol = symbol.strip().zfill(6)

        # Get data (use cache if available)
        if cache.has_data(symbol, 'week'):
            weekly_df = cache.get_data(symbol, 'week')
            daily_df = cache.get_data(symbol, 'day')
        else:
            weekly_df = fetcher.get_weekly_data(symbol, count=1000)
            daily_df = fetcher.get_daily_data(symbol, count=1000)
            if not weekly_df.empty:
                cache.save_data(symbol, weekly_df, 'week')
            if not daily_df.empty:
                cache.save_data(symbol, daily_df, 'day')

        if weekly_df.empty or daily_df.empty:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

        # Detect landmarks - use zigzag_detector directly for all landmarks
        # Then apply multi-filter confirmation
        weekly_prices = weekly_df['收盘'] if '收盘' in weekly_df.columns else weekly_df['close']
        weekly_landmarks = zigzag_detector(weekly_prices, threshold=threshold)

        if not weekly_landmarks:
            raise HTTPException(
                status_code=400,
                detail=f"No landmarks found for {symbol} with threshold {threshold}"
            )

        # Filter by type
        type_landmarks = [lm for lm in weekly_landmarks if lm.get('type') == landmark_type]

        if len(type_landmarks) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient {landmark_type} landmarks (need 3+, got {len(type_landmarks)})"
            )

        # Extract week indices directly from zigzag landmarks
        sequence = [int(lm['index']) for lm in type_landmarks]  # Convert to Python int

        if len(sequence) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient sequence length (need 3+, got {len(sequence)})"
            )

        # Get current week
        current_week = len(weekly_df) - 1

        # Calculate all combinations to find future predictions
        from collections import Counter
        all_results = []
        for num_terms in range(2, 4):
            all_results.extend(find_pattern.calculate_combinations(sequence, num_terms))

        # Get unique values greater than current week (future predictions)
        all_values = [v for _, v in all_results]
        unique_values = sorted(set(all_values))
        future_values = [v for v in unique_values if v > current_week]

        if not future_values:
            raise HTTPException(status_code=400, detail="No future predictions found")

        # Get the smallest future value (next prediction)
        predicted_week_index = future_values[0]

        # Find ALL expressions and frequency for this prediction
        # Group by pattern type (A+B, A-B, A+B-C, etc.)
        all_expressions = []
        pattern_types = {
            'A+B': [],      # Only addition, 2 terms
            'A-B': [],      # Only subtraction, 2 terms
            'A+B+C': [],    # Only addition, 3 terms
            'A+B-C': [],    # Mixed add/subtract
            'A-B-C': [],    # Only subtraction, 3 terms
            'Other': []
        }

        for e, v in all_results:
            if v == predicted_week_index:
                all_expressions.append(e)
                # Categorize by pattern type
                expr_clean = e.replace(' ', '')
                has_plus = '+' in expr_clean
                has_minus = '-' in expr_clean
                plus_count = expr_clean.count('+')
                minus_count = expr_clean.count('-')

                if has_plus and not has_minus:
                    if plus_count == 1:
                        pattern_types['A+B'].append(e)
                    else:
                        pattern_types['A+B+C'].append(e)
                elif has_minus and not has_plus:
                    if minus_count == 1:
                        pattern_types['A-B'].append(e)
                    else:
                        pattern_types['A-B-C'].append(e)
                elif has_plus and has_minus:
                    pattern_types['A+B-C'].append(e)
                else:
                    pattern_types['Other'].append(e)

        # Build pattern expressions summary
        pattern_expressions = {}
        for ptype, exprs in pattern_types.items():
            if exprs:
                # Deduplicate and limit examples
                unique_exprs = list(dict.fromkeys(exprs))  # Preserve order, deduplicate
                pattern_expressions[ptype] = {
                    'count': len(unique_exprs),
                    'examples': unique_exprs[:10]  # Show up to 10 examples per type
                }

        freq = Counter(v for _, v in all_results)[predicted_week_index]
        expr = all_expressions[0] if all_expressions else None  # Keep first example for backward compat

        # Convert to date range
        try:
            # For past weeks, use actual index; for future weeks, extrapolate
            if predicted_week_index < len(weekly_df):
                predicted_week_date = weekly_df.index[predicted_week_index]
            else:
                # Extrapolate future date from last known date
                weeks_ahead = predicted_week_index - (len(weekly_df) - 1)
                predicted_week_date = weekly_df.index[-1] + pd.Timedelta(weeks=weeks_ahead)
            week_start = predicted_week_date - pd.Timedelta(days=6)
            week_end = predicted_week_date
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Date calculation error: {str(e)}")

        # Calculate weeks until prediction
        last_week_index = len(weekly_df) - 1
        weeks_to_prediction = int(predicted_week_index - last_week_index)

        # Get current price
        current_price = float(daily_df['收盘'].iloc[-1])

        # Confidence level
        confidence = "HIGH" if freq >= 3 else "MEDIUM" if freq >= 2 else "LOW"

        # Get stock name
        stock_name = stock_db.get_name(symbol) or symbol

        return {
            "symbol": symbol,
            "name": stock_name,
            "landmark_type": landmark_type,
            "current_price": current_price,
            "predicted_week": int(predicted_week_index),
            "predicted_week_start": week_start.strftime("%Y-%m-%d"),
            "predicted_week_end": week_end.strftime("%Y-%m-%d"),
            "weeks_to_prediction": weeks_to_prediction,
            "pattern_expression": expr,
            "pattern_frequency": int(freq),
            "pattern_expressions": pattern_expressions,  # NEW: All pattern types
            "confidence": confidence,
            "sequence": sequence,
            "current_week": int(last_week_index)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /api/predict for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyze")
async def api_analyze(
    symbol: str = Query(..., description="Stock code (6 digits)"),
    threshold: float = Query(0.05, description="ZigZag threshold"),
    include_secondary: bool = Query(False, description="Include secondary/minor landmarks")
):
    """
    Get full analysis for a stock.

    Returns:
        JSON with landmarks, sequence, predictions, and chart data.
    """
    try:
        # Normalize symbol
        symbol = symbol.strip().zfill(6)

        # Get data
        if cache.has_data(symbol, 'week'):
            weekly_df = cache.get_data(symbol, 'week')
            daily_df = cache.get_data(symbol, 'day')
        else:
            weekly_df = fetcher.get_weekly_data(symbol, count=1000)
            daily_df = fetcher.get_daily_data(symbol, count=1000)
            if not weekly_df.empty:
                cache.save_data(symbol, weekly_df, 'week')
            if not daily_df.empty:
                cache.save_data(symbol, daily_df, 'day')

        if weekly_df.empty or daily_df.empty:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

        # Detect landmarks using multi-layer detector with optimized defaults
        weekly_prices = weekly_df['收盘'] if '收盘' in weekly_df.columns else weekly_df['close']

        # Filter out invalid prices (negative, zero, or NaN)
        valid_mask = (weekly_prices > 0) & (~weekly_prices.isna())
        if valid_mask.sum() < len(weekly_df):
            logger.warning(f"{symbol}: Filtered {(~valid_mask).sum()} invalid price points")
            # Apply the same filter to the dataframe
            weekly_df = weekly_df[valid_mask].copy()
            weekly_prices = weekly_df['收盘'] if '收盘' in weekly_df.columns else weekly_df['close']

        if len(weekly_prices) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient valid data for {symbol}. Only {len(weekly_prices)} weeks of valid price data."
            )

        # Use multi-layer detector with optimized parameters
        params = LayerFilterParams(
            layer1_threshold=0.10,  # 10% threshold
            layer2_min_frequency=2,
            layer2_min_deviation_pct=0.20,  # 20% deviation
            layer3_min_trend_strength=0.50,  # 0.50 trend strength
            layer4_min_weeks_same_type=10,
            layer4_min_weeks_alternating=4,
            show_layer_outputs=False
        )

        detector = MultiLayerDetector(params)
        weekly_landmarks, stats = detector.detect(weekly_prices, debug=False)

        # Convert stats to native Python types
        stats_py = {
            'layer1_count': int(stats['layer1_count']),
            'layer2_count': int(stats['layer2_count']),
            'layer3_count': int(stats['layer3_count']),
            'layer4_count': int(stats['layer4_count']),
            'final_count': int(stats['final_count'])
        }

        # Get all landmarks for chart - use weekly dates to match chart data
        # Need to map filtered indices back to original dataframe indices
        all_landmarks = []
        valid_indices = weekly_df.index.tolist()

        for lm in weekly_landmarks:
            # lm['index'] is the index in the filtered series
            landmark_date = weekly_df.index[lm['index']]
            landmark_data = {
                "date": landmark_date.strftime("%Y-%m-%d"),
                "price": float(lm['price']),
                "type": str(lm.get('type', 'unknown')),
                "weekly_index": int(lm['index']),
                "confidence": float(lm.get('confidence', 0.5)),
                "reasons": []
            }
            # Add optional fields if present
            if 'layer' in lm:
                landmark_data['layer'] = int(lm['layer'])
            if 'zigzag_pct' in lm:
                landmark_data['zigzag_pct'] = float(lm['zigzag_pct'])
            if 'deviation' in lm:
                landmark_data['deviation'] = float(lm['deviation'])
            all_landmarks.append(landmark_data)

        # Chart data - use filtered weekly data
        chart_data = []
        for date, row in weekly_df.iterrows():
            open_price = float(row['开盘'] if '开盘' in row else row.get('open', row['close']))
            close_price = float(row['收盘'] if '收盘' in row else row['close'])
            high_price = float(row['最高'] if '最高' in row else row.get('high', row['close']))
            low_price = float(row['最低'] if '最低' in row else row.get('low', row['close']))
            volume = float(row['成交量'] if '成交量' in row else row.get('volume', 0))

            # Only add valid data points (all prices must be positive)
            if open_price > 0 and close_price > 0 and high_price > 0 and low_price > 0:
                chart_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": open_price,
                    "close": close_price,
                    "high": high_price,
                    "low": low_price,
                    "volume": volume
                })

        # Current info
        current_price = float(daily_df['收盘'].iloc[-1])
        last_week = len(weekly_df) - 1

        # Get stock name
        stock_name = stock_db.get_name(symbol) or symbol

        # Calculate filter rate safely (handle division by zero)
        if stats_py['layer1_count'] > 0:
            filter_rate = f"{(1 - stats_py['final_count']/stats_py['layer1_count'])*100:.1f}%"
        else:
            filter_rate = "N/A"

        return {
            "symbol": symbol,
            "name": stock_name,
            "current_price": current_price,
            "last_week": last_week,
            "last_date": daily_df.index[-1].strftime("%Y-%m-%d"),
            "landmarks": all_landmarks,
            "chart_data": chart_data,
            "threshold": threshold,
            "multi_layer_stats": {
                "layer1_count": stats_py['layer1_count'],
                "layer2_count": stats_py['layer2_count'],
                "layer3_count": stats_py['layer3_count'],
                "layer4_count": stats_py['layer4_count'],
                "final_count": stats_py['final_count'],
                "filter_rate": filter_rate
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /api/analyze for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/search")
async def api_stocks_search(
    q: str = Query(..., min_length=1, description="Search query (code, pinyin, or name)"),
    limit: int = Query(10, description="Maximum results to return")
):
    """
    Search for stocks by code, pinyin abbreviation, or name.

    Supports:
    - Stock codes: 601933, 000001
    - Pinyin abbreviations: jd, ynsh, wly
    - Partial pinyin: yun, zhang
    - Chinese names: 京东, 渝农商行

    Returns:
        List of matching stocks with code, name, and exchange.
    """
    try:
        query = q.strip()

        # If empty query, return empty
        if not query:
            return {"results": []}

        # Use stock database to search
        results = stock_db.search(query, limit=limit)

        return {"results": results}

    except Exception as e:
        logger.exception(f"Error in /api/stocks/search: {e}")
        return {"results": []}


@app.post("/api/multi-layer-detect")
async def api_multi_layer_detect(request: Request):
    """
    Multi-layer landmark detection with configurable parameters.

    This endpoint allows real-time adjustment of filter parameters
    and returns landmarks detected through the 4-layer filtering system.

    Request body:
        {
            "symbol": "300364",
            "layer1_threshold": 8,
            "layer2_min_freq": 2,
            "layer2_min_dev": 15,
            "layer3_trend_str": 50,
            "layer4_same_type": 10,
            "layer4_alt": 6
        }

    Returns:
        JSON with landmarks, statistics, and layer breakdown.
    """
    try:
        import base64
        from io import BytesIO
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        data = await request.json()
        symbol = data.get('symbol', '').strip().zfill(6)

        # Get parameters (optimized defaults based on user testing)
        layer1_threshold = float(data.get('layer1_threshold', 10)) / 100
        layer2_min_freq = int(data.get('layer2_min_freq', 2))
        layer2_min_dev = float(data.get('layer2_min_dev', 20)) / 100
        layer3_trend_str = float(data.get('layer3_trend_str', 50)) / 100
        layer4_same_type = int(data.get('layer4_same_type', 10))
        layer4_alt = int(data.get('layer4_alt', 4))

        # Get stock data
        if cache.has_data(symbol, 'week'):
            weekly_df = cache.get_data(symbol, 'week')
        else:
            weekly_df = fetcher.get_weekly_data(symbol, count=1000)
            if not weekly_df.empty:
                cache.save_data(symbol, weekly_df, 'week')

        if weekly_df.empty:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

        # Extract price series
        weekly_prices = weekly_df['收盘'] if '收盘' in weekly_df.columns else weekly_df['close']

        # Apply multi-layer detection
        params = LayerFilterParams(
            layer1_threshold=layer1_threshold,
            layer2_min_frequency=layer2_min_freq,
            layer2_min_deviation_pct=layer2_min_dev,
            layer3_min_trend_strength=layer3_trend_str,
            layer4_min_weeks_same_type=layer4_same_type,
            layer4_min_weeks_alternating=layer4_alt
        )

        detector = MultiLayerDetector(params)
        landmarks, stats = detector.detect(weekly_prices, debug=False)

        # Convert landmarks to API format
        landmarks_data = []
        for lm in landmarks:
            landmark_date = weekly_df.index[lm['index']]
            landmarks_data.append({
                "date": landmark_date.strftime("%Y-%m-%d"),
                "price": float(lm['price']),
                "type": lm['type'],
                "weekly_index": int(lm['index']),
                "layer": int(lm.get('layer', 1)),
                "confidence": float(lm.get('confidence', 0))
            })

        # Generate visualization
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Price with landmarks
        ax = axes[0]
        ax.plot(weekly_prices.index, weekly_prices.values, 'b-', linewidth=1.5, alpha=0.6, label='价格')

        layer_colors = {
            1: '#ff6b6b',
            2: '#ffd93d',
            3: '#6bcb77',
            4: '#4d96ff'
        }

        for lm in landmarks:
            color = layer_colors.get(lm.get('layer', 1), '#999999')
            marker = '^' if lm['type'] == 'high' else 'v'
            ax.scatter(weekly_prices.index[lm['index']], lm['price'],
                      c=color, s=150, marker=marker,
                      edgecolors='black', linewidths=2, zorder=5, alpha=0.9)
            ax.annotate(f"L{lm.get('layer', 1)}",
                      (weekly_prices.index[lm['index']], lm['price']),
                      xytext=(0, 10 if lm['type'] == 'high' else -15),
                      textcoords='offset points',
                      fontsize=8, ha='center', fontweight='bold')

        ax.set_title(f'{symbol} - 多层地标点检测', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('价格')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Layer statistics
        ax = axes[1]
        layers = ['第一层\nZigZag', '第二层\n统计', '第三层\n趋势', '第四层\n间隔']
        counts = [
            stats['layer1_count'],
            stats['layer2_count'],
            stats['layer3_count'],
            stats['layer4_count']
        ]
        colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']

        bars = ax.bar(layers, counts, color=colors)
        ax.set_ylabel('地标点数量')
        ax.set_title('各层过滤统计', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        plt.tight_layout()

        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Get stock name
        stock_name = stock_db.get_name(symbol) or symbol

        return {
            "symbol": symbol,
            "name": stock_name,
            "plot": img_base64,
            "stats": stats,
            "landmarks": landmarks_data,
            "total_price_points": len(weekly_prices)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /api/multi-layer-detect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Health Check ===

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "stock-predictor"}


# === Run Server ===

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("🚀 Stock Pattern Predictor Web Server")
    print("="*60)
    print(f"Server: http://localhost:8000")
    print(f"API Docs: http://localhost:8000/docs")
    print("="*60)
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
