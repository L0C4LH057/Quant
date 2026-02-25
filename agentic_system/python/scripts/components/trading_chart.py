"""
Professional Trading Chart Component

TradingView-style charts with:
- Candlestick/Line/Area charts
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Drawing tools (trendlines, horizontals, fibonacci)
- Crosshair with price tracking
- Real-time updates
"""

import streamlit.components.v1 as components
import json
from typing import List, Dict, Optional
import pandas as pd


def render_trading_chart(
    ohlcv_data: List[Dict],
    symbol: str = "EURUSD",
    height: int = 600,
    theme: str = "dark",
    show_volume: bool = True,
    indicators: Optional[List[str]] = None,
    bridge_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeframe: str = "H1",
) -> None:
    """
    Render a professional TradingView-style chart with real-time streaming.
    
    Args:
        ohlcv_data: List of dicts with time, open, high, low, close, volume
        symbol: Symbol name for display
        height: Chart height in pixels
        theme: 'dark' or 'light'
        show_volume: Show volume bars
        indicators: List of indicators to show ['SMA20', 'EMA50', 'RSI', 'MACD', 'BB']
        bridge_url: MT5 bridge URL for real-time streaming
        api_key: API key for MT5 bridge
        timeframe: Chart timeframe (M1, M5, H1, etc.)
    """
    
    if not ohlcv_data:
        components.html(f"""
        <div style="display:flex;align-items:center;justify-content:center;height:{height}px;
                    background:#161b22;color:#8b949e;font-family:system-ui;">
            <div style="text-align:center;">
                <p style="font-size:24px;">📊 No Data Available</p>
                <p>Select a valid symbol from your broker</p>
            </div>
        </div>
        """, height=height)
        return
    
    # Convert data to JSON
    candle_data = json.dumps(ohlcv_data)
    indicators_list = json.dumps(indicators or [])
    
    # Calculate indicators if needed
    df = pd.DataFrame(ohlcv_data)
    
    # SMA calculations
    sma20_data = []
    sma50_data = []
    ema20_data = []
    
    if len(df) >= 20:
        df['sma20'] = df['close'].rolling(window=20).mean()
        sma20_data = df[['time', 'sma20']].dropna().rename(columns={'sma20': 'value'}).to_dict('records')
    
    if len(df) >= 50:
        df['sma50'] = df['close'].rolling(window=50).mean()
        sma50_data = df[['time', 'sma50']].dropna().rename(columns={'sma50': 'value'}).to_dict('records')
    
    if len(df) >= 20:
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        ema20_data = df[['time', 'ema20']].dropna().rename(columns={'ema20': 'value'}).to_dict('records')
    
    # Bollinger Bands
    bb_upper = []
    bb_lower = []
    if len(df) >= 20:
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
        bb_upper = df[['time', 'bb_upper']].dropna().rename(columns={'bb_upper': 'value'}).to_dict('records')
        bb_lower = df[['time', 'bb_lower']].dropna().rename(columns={'bb_lower': 'value'}).to_dict('records')
    
    # RSI calculation
    rsi_data = []
    if len(df) >= 14:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        rsi_data = df[['time', 'rsi']].dropna().rename(columns={'rsi': 'value'}).to_dict('records')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                background: {'#0d1117' if theme == 'dark' else '#ffffff'}; 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            #container {{ 
                width: 100%; 
                height: {height}px;
                position: relative;
            }}
            #chart {{ width: 100%; height: {'75%' if indicators and ('RSI' in indicators or 'MACD' in indicators) else '85%'}; }}
            #rsi-chart {{ width: 100%; height: 15%; border-top: 1px solid #30363d; }}
            
            .toolbar {{
                display: flex;
                gap: 8px;
                padding: 8px 12px;
                background: {'#21262d' if theme == 'dark' else '#f6f8fa'};
                border-bottom: 1px solid #30363d;
                align-items: center;
                flex-wrap: wrap;
            }}
            .toolbar-group {{
                display: flex;
                gap: 4px;
                padding-right: 12px;
                border-right: 1px solid #30363d;
            }}
            .toolbar-group:last-child {{ border-right: none; }}
            .toolbar button {{
                background: {'#30363d' if theme == 'dark' else '#e1e4e8'};
                border: 1px solid {'#484f58' if theme == 'dark' else '#d0d7de'};
                color: {'#c9d1d9' if theme == 'dark' else '#24292f'};
                padding: 6px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.2s;
            }}
            .toolbar button:hover {{
                background: {'#484f58' if theme == 'dark' else '#d0d7de'};
            }}
            .toolbar button.active {{
                background: #238636;
                border-color: #238636;
                color: white;
            }}
            .toolbar select {{
                background: {'#30363d' if theme == 'dark' else '#e1e4e8'};
                border: 1px solid {'#484f58' if theme == 'dark' else '#d0d7de'};
                color: {'#c9d1d9' if theme == 'dark' else '#24292f'};
                padding: 6px 10px;
                border-radius: 6px;
                font-size: 12px;
            }}
            .price-display {{
                position: absolute;
                top: 60px;
                left: 16px;
                z-index: 100;
                color: {'#c9d1d9' if theme == 'dark' else '#24292f'};
                font-size: 13px;
            }}
            .price-display .symbol {{ font-size: 18px; font-weight: 600; }}
            .price-display .ohlc {{ margin-top: 4px; opacity: 0.8; }}
            .legend {{
                position: absolute;
                top: 120px;
                left: 16px;
                z-index: 100;
                font-size: 11px;
            }}
            .legend-item {{
                display: inline-block;
                margin-right: 12px;
                padding: 2px 6px;
                border-radius: 4px;
                background: rgba(0,0,0,0.3);
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <div class="toolbar">
                <div class="toolbar-group">
                    <button id="btn-candle" class="active" onclick="setChartType('candle')">🕯️ Candle</button>
                    <button id="btn-line" onclick="setChartType('line')">📈 Line</button>
                    <button id="btn-area" onclick="setChartType('area')">📊 Area</button>
                </div>
                <div class="toolbar-group">
                    <button id="btn-sma20" onclick="toggleIndicator('sma20')">SMA 20</button>
                    <button id="btn-sma50" onclick="toggleIndicator('sma50')">SMA 50</button>
                    <button id="btn-ema20" onclick="toggleIndicator('ema20')">EMA 20</button>
                    <button id="btn-bb" onclick="toggleIndicator('bb')">BB</button>
                </div>
                <div class="toolbar-group">
                    <button id="btn-crosshair" onclick="toggleCrosshair()">➕ Crosshair</button>
                    <button onclick="chart.timeScale().fitContent()">🔍 Fit</button>
                    <button onclick="chart.timeScale().scrollToRealTime()">⏩ Now</button>
                </div>
                <div class="toolbar-group">
                    <button id="btn-draw-line" onclick="setDrawMode('line')">📏 Line</button>
                    <button id="btn-draw-hline" onclick="setDrawMode('hline')">➖ H-Line</button>
                    <button onclick="clearDrawings()">🗑️ Clear</button>
                </div>
            </div>
            
            <div class="price-display">
                <div class="symbol">{symbol}</div>
                <div class="ohlc" id="ohlc-display">Hover over chart</div>
            </div>
            
            <div class="legend" id="legend"></div>
            
            <div id="chart"></div>
            <div id="rsi-chart"></div>
        </div>
        
        <script>
            const candleData = {candle_data};
            const sma20Data = {json.dumps(sma20_data)};
            const sma50Data = {json.dumps(sma50_data)};
            const ema20Data = {json.dumps(ema20_data)};
            const bbUpperData = {json.dumps(bb_upper)};
            const bbLowerData = {json.dumps(bb_lower)};
            const rsiData = {json.dumps(rsi_data)};
            const showIndicators = {indicators_list};
            
            // Theme colors
            const colors = {{
                background: '{'#0d1117' if theme == 'dark' else '#ffffff'}',
                text: '{'#c9d1d9' if theme == 'dark' else '#24292f'}',
                grid: '{'#21262d' if theme == 'dark' else '#f0f0f0'}',
                up: '#3fb950',
                down: '#f85149',
                sma20: '#2196F3',
                sma50: '#FF9800',
                ema20: '#9C27B0',
                bbUpper: '#00BCD4',
                bbLower: '#00BCD4',
            }};
            
            // Create main chart
            const chartContainer = document.getElementById('chart');
            const chart = LightweightCharts.createChart(chartContainer, {{
                layout: {{
                    background: {{ type: 'solid', color: colors.background }},
                    textColor: colors.text,
                }},
                grid: {{
                    vertLines: {{ color: colors.grid }},
                    horzLines: {{ color: colors.grid }},
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                    vertLine: {{
                        width: 1,
                        color: '#758696',
                        style: LightweightCharts.LineStyle.Dashed,
                        labelBackgroundColor: '#2962FF',
                    }},
                    horzLine: {{
                        width: 1,
                        color: '#758696',
                        style: LightweightCharts.LineStyle.Dashed,
                        labelBackgroundColor: '#2962FF',
                    }},
                }},
                timeScale: {{
                    timeVisible: true,
                    secondsVisible: false,
                    borderColor: colors.grid,
                }},
                rightPriceScale: {{
                    borderColor: colors.grid,
                }},
            }});
            
            // Candlestick series
            let candleSeries = chart.addCandlestickSeries({{
                upColor: colors.up,
                downColor: colors.down,
                borderUpColor: colors.up,
                borderDownColor: colors.down,
                wickUpColor: colors.up,
                wickDownColor: colors.down,
            }});
            candleSeries.setData(candleData);
            
            // Volume series
            let volumeSeries = null;
            if ({str(show_volume).lower()}) {{
                volumeSeries = chart.addHistogramSeries({{
                    color: '#26a69a',
                    priceFormat: {{ type: 'volume' }},
                    priceScaleId: '',
                    scaleMargins: {{ top: 0.85, bottom: 0 }},
                }});
                volumeSeries.setData(candleData.map(d => ({{
                    time: d.time,
                    value: d.volume,
                    color: d.close >= d.open ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)',
                }})));
            }}
            
            // Indicator series (hidden by default)
            const indicators = {{}};
            
            indicators.sma20 = chart.addLineSeries({{
                color: colors.sma20,
                lineWidth: 2,
                title: 'SMA 20',
            }});
            indicators.sma20.setData(sma20Data);
            indicators.sma20.applyOptions({{ visible: showIndicators.includes('SMA20') }});
            
            indicators.sma50 = chart.addLineSeries({{
                color: colors.sma50,
                lineWidth: 2,
                title: 'SMA 50',
            }});
            indicators.sma50.setData(sma50Data);
            indicators.sma50.applyOptions({{ visible: showIndicators.includes('SMA50') }});
            
            indicators.ema20 = chart.addLineSeries({{
                color: colors.ema20,
                lineWidth: 2,
                title: 'EMA 20',
            }});
            indicators.ema20.setData(ema20Data);
            indicators.ema20.applyOptions({{ visible: showIndicators.includes('EMA20') }});
            
            indicators.bbUpper = chart.addLineSeries({{
                color: colors.bbUpper,
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                title: 'BB Upper',
            }});
            indicators.bbUpper.setData(bbUpperData);
            indicators.bbUpper.applyOptions({{ visible: showIndicators.includes('BB') }});
            
            indicators.bbLower = chart.addLineSeries({{
                color: colors.bbLower,
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                title: 'BB Lower',
            }});
            indicators.bbLower.setData(bbLowerData);
            indicators.bbLower.applyOptions({{ visible: showIndicators.includes('BB') }});
            
            // RSI Chart (if data available)
            let rsiChart = null;
            let rsiSeries = null;
            if (rsiData.length > 0 && showIndicators.includes('RSI')) {{
                const rsiContainer = document.getElementById('rsi-chart');
                rsiContainer.style.display = 'block';
                
                rsiChart = LightweightCharts.createChart(rsiContainer, {{
                    layout: {{
                        background: {{ type: 'solid', color: colors.background }},
                        textColor: colors.text,
                    }},
                    grid: {{
                        vertLines: {{ color: colors.grid }},
                        horzLines: {{ color: colors.grid }},
                    }},
                    timeScale: {{ visible: false }},
                }});
                
                rsiSeries = rsiChart.addLineSeries({{
                    color: '#f59e0b',
                    lineWidth: 2,
                    priceScaleId: 'right',
                }});
                rsiSeries.setData(rsiData);
                
                // Sync timescales
                chart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                    if (range) rsiChart.timeScale().setVisibleLogicalRange(range);
                }});
            }}
            
            // Fit content
            chart.timeScale().fitContent();
            
            // OHLC display on hover
            chart.subscribeCrosshairMove(param => {{
                if (param.time) {{
                    const data = param.seriesData.get(candleSeries);
                    if (data) {{
                        document.getElementById('ohlc-display').innerHTML = 
                            `O: ${{data.open?.toFixed(5) || '-'}} H: ${{data.high?.toFixed(5) || '-'}} ` +
                            `L: ${{data.low?.toFixed(5) || '-'}} C: ${{data.close?.toFixed(5) || '-'}}`;
                    }}
                }}
            }});
            
            // Toggle indicator visibility
            function toggleIndicator(name) {{
                const btn = document.getElementById('btn-' + name);
                const series = indicators[name];
                
                if (name === 'bb') {{
                    const visible = !indicators.bbUpper.options().visible;
                    indicators.bbUpper.applyOptions({{ visible }});
                    indicators.bbLower.applyOptions({{ visible }});
                    btn.classList.toggle('active', visible);
                }} else if (series) {{
                    const visible = !series.options().visible;
                    series.applyOptions({{ visible }});
                    btn.classList.toggle('active', visible);
                }}
                updateLegend();
            }}
            
            // Chart type switching
            let currentType = 'candle';
            let lineSeries = null;
            let areaSeries = null;
            
            function setChartType(type) {{
                document.querySelectorAll('.toolbar-group:first-child button').forEach(b => b.classList.remove('active'));
                document.getElementById('btn-' + type).classList.add('active');
                
                if (type === 'candle') {{
                    candleSeries.applyOptions({{ visible: true }});
                    if (lineSeries) lineSeries.applyOptions({{ visible: false }});
                    if (areaSeries) areaSeries.applyOptions({{ visible: false }});
                }} else if (type === 'line') {{
                    candleSeries.applyOptions({{ visible: false }});
                    if (!lineSeries) {{
                        lineSeries = chart.addLineSeries({{
                            color: '#2962FF',
                            lineWidth: 2,
                        }});
                        lineSeries.setData(candleData.map(d => ({{ time: d.time, value: d.close }})));
                    }}
                    lineSeries.applyOptions({{ visible: true }});
                    if (areaSeries) areaSeries.applyOptions({{ visible: false }});
                }} else if (type === 'area') {{
                    candleSeries.applyOptions({{ visible: false }});
                    if (lineSeries) lineSeries.applyOptions({{ visible: false }});
                    if (!areaSeries) {{
                        areaSeries = chart.addAreaSeries({{
                            topColor: 'rgba(41, 98, 255, 0.5)',
                            bottomColor: 'rgba(41, 98, 255, 0.0)',
                            lineColor: '#2962FF',
                            lineWidth: 2,
                        }});
                        areaSeries.setData(candleData.map(d => ({{ time: d.time, value: d.close }})));
                    }}
                    areaSeries.applyOptions({{ visible: true }});
                }}
                currentType = type;
            }}
            
            // Crosshair toggle
            let crosshairEnabled = true;
            function toggleCrosshair() {{
                crosshairEnabled = !crosshairEnabled;
                chart.applyOptions({{
                    crosshair: {{
                        mode: crosshairEnabled ? 
                            LightweightCharts.CrosshairMode.Normal : 
                            LightweightCharts.CrosshairMode.Hidden,
                    }},
                }});
                document.getElementById('btn-crosshair').classList.toggle('active', crosshairEnabled);
            }}
            
            // Drawing tools (simplified - stores as markers)
            let drawMode = null;
            let drawings = [];
            
            function setDrawMode(mode) {{
                drawMode = drawMode === mode ? null : mode;
                document.getElementById('btn-draw-line').classList.toggle('active', drawMode === 'line');
                document.getElementById('btn-draw-hline').classList.toggle('active', drawMode === 'hline');
            }}
            
            function clearDrawings() {{
                drawings = [];
                candleSeries.setMarkers([]);
            }}
            
            // Add horizontal line on click when in hline mode
            chartContainer.addEventListener('click', (e) => {{
                if (drawMode === 'hline') {{
                    const price = candleSeries.coordinateToPrice(e.offsetY - 60);
                    if (price) {{
                        const priceLine = candleSeries.createPriceLine({{
                            price: price,
                            color: '#f59e0b',
                            lineWidth: 1,
                            lineStyle: LightweightCharts.LineStyle.Dashed,
                            axisLabelVisible: true,
                        }});
                        drawings.push(priceLine);
                    }}
                }}
            }});
            
            // Update legend
            function updateLegend() {{
                let legend = '';
                if (indicators.sma20.options().visible) 
                    legend += `<span class="legend-item" style="color:${{colors.sma20}}">SMA 20</span>`;
                if (indicators.sma50.options().visible) 
                    legend += `<span class="legend-item" style="color:${{colors.sma50}}">SMA 50</span>`;
                if (indicators.ema20.options().visible) 
                    legend += `<span class="legend-item" style="color:${{colors.ema20}}">EMA 20</span>`;
                if (indicators.bbUpper.options().visible) 
                    legend += `<span class="legend-item" style="color:${{colors.bbUpper}}">Bollinger Bands</span>`;
                document.getElementById('legend').innerHTML = legend;
            }}
            
            // Responsive resize
            new ResizeObserver(() => {{
                chart.applyOptions({{ width: chartContainer.clientWidth }});
                if (rsiChart) rsiChart.applyOptions({{ width: chartContainer.clientWidth }});
            }}).observe(chartContainer);
            
            // ========== REAL-TIME STREAMING ==========
            const bridgeUrl = '{bridge_url or ""}';
            const apiKey = '{api_key or ""}';
            const streamSymbol = '{symbol}';
            const streamTimeframe = '{timeframe}';
            
            // Timeframe to seconds mapping
            const tfSeconds = {{
                'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800,
                'H1': 3600, 'H4': 14400, 'D1': 86400, 'W1': 604800
            }};
            const candleSeconds = tfSeconds[streamTimeframe] || 3600;
            
            // Live price display
            let livePrice = null;
            let lastCandle = candleData.length > 0 ? {{ ...candleData[candleData.length - 1] }} : null;
            
            // Create live price indicator
            const livePriceDiv = document.createElement('div');
            livePriceDiv.id = 'live-price';
            livePriceDiv.style.cssText = `
                position: absolute;
                top: 60px;
                right: 80px;
                z-index: 100;
                font-size: 24px;
                font-weight: bold;
                font-family: 'Courier New', monospace;
                color: #c9d1d9;
                background: rgba(0,0,0,0.5);
                padding: 8px 16px;
                border-radius: 8px;
            `;
            document.getElementById('container').appendChild(livePriceDiv);
            
            // Streaming status indicator
            const streamStatus = document.createElement('div');
            streamStatus.id = 'stream-status';
            streamStatus.style.cssText = `
                position: absolute;
                top: 100px;
                right: 80px;
                z-index: 100;
                font-size: 11px;
                color: #3fb950;
            `;
            streamStatus.innerHTML = '🟢 Live';
            document.getElementById('container').appendChild(streamStatus);
            
            // Fetch latest quote and update chart
            async function fetchAndUpdatePrice() {{
                if (!bridgeUrl) return;
                
                try {{
                    const response = await fetch(
                        `${{bridgeUrl}}/quote/${{streamSymbol}}`,
                        {{ headers: {{ 'X-API-Key': apiKey }} }}
                    );
                    
                    if (response.ok) {{
                        const quote = await response.json();
                        const bid = quote.bid || quote.close;
                        const ask = quote.ask;
                        const mid = ask ? (bid + ask) / 2 : bid;
                        
                        if (mid && mid > 0) {{
                            livePrice = mid;
                            
                            // Update live price display
                            const priceColor = lastCandle && mid >= lastCandle.open ? '#3fb950' : '#f85149';
                            livePriceDiv.innerHTML = `<span style="color:${{priceColor}}">${{mid.toFixed(5)}}</span>`;
                            
                            // Update current candle
                            if (lastCandle) {{
                                const now = Math.floor(Date.now() / 1000);
                                const candleStart = Math.floor(now / candleSeconds) * candleSeconds;
                                
                                // If we're in a new candle period
                                if (candleStart > lastCandle.time) {{
                                    // Close the old candle and start new one
                                    lastCandle = {{
                                        time: candleStart,
                                        open: mid,
                                        high: mid,
                                        low: mid,
                                        close: mid,
                                        volume: 0
                                    }};
                                }} else {{
                                    // Update current candle
                                    lastCandle.close = mid;
                                    lastCandle.high = Math.max(lastCandle.high, mid);
                                    lastCandle.low = Math.min(lastCandle.low, mid);
                                }}
                                
                                // Apply update to chart
                                candleSeries.update(lastCandle);
                                
                                // Update volume if available
                                if (volumeSeries) {{
                                    volumeSeries.update({{
                                        time: lastCandle.time,
                                        value: lastCandle.volume || 0,
                                        color: lastCandle.close >= lastCandle.open 
                                            ? 'rgba(63, 185, 80, 0.3)' 
                                            : 'rgba(248, 81, 73, 0.3)'
                                    }});
                                }}
                            }}
                            
                            streamStatus.innerHTML = '🟢 Live';
                            streamStatus.style.color = '#3fb950';
                        }}
                    }} else {{
                        streamStatus.innerHTML = '🟡 Reconnecting...';
                        streamStatus.style.color = '#f59e0b';
                    }}
                }} catch (error) {{
                    console.log('Streaming error:', error);
                    streamStatus.innerHTML = '🔴 Disconnected';
                    streamStatus.style.color = '#f85149';
                }}
            }}
            
            // Start real-time streaming (update every 1 second)
            if (bridgeUrl) {{
                setInterval(fetchAndUpdatePrice, 1000);
                fetchAndUpdatePrice(); // Initial fetch
            }}
        </script>
    </body>
    </html>
    """
    
    components.html(html_content, height=height + 50)

