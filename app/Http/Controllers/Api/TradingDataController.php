<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Log;
use App\Models\TradingAccount;
use App\Services\Trading\TradingServiceFactory;
use Carbon\Carbon;

class TradingDataController extends Controller
{
    protected $tradingFactory;

    public function __construct(TradingServiceFactory $tradingFactory)
    {
        $this->tradingFactory = $tradingFactory;
    }

    /**
     * Fetch historical candle data for a symbol.
     * Route: GET /api/trading/history/{symbol}
     */
    protected function generateMockCandles($currentPrice = 2035.00, $timeframe = '1H')
    {
        // ... (Mock Fallback kept for safety, but primary logic is below) ...
        return $this->generateRandomWalk($currentPrice, $timeframe);
    }

    /**
     * Fetch Real Data from Yahoo Finance (Public Endpoint)
     * Note: This is a workaround to provide real charts without a paid Market Data subscription.
     */
    public function getHistory(Request $request, string $symbol)
    {
        try {
            $tf = $request->query('timeframe', '1H');
            
            // 1. Map Symbol to Yahoo Ticker
            $yahooSymbol = match(strtoupper($symbol)) {
                'XAUUSD', 'GOLD' => 'GC=F', // Gold Futures (Close enough for visual)
                'BTCUSD', 'BTC' => 'BTC-USD',
                'EURUSD' => 'EURUSD=X',
                'GBPUSD' => 'GBPUSD=X',
                'USDJPY' => 'USDJPY=X',
                default => 'BTC-USD'
            };

            // 2. Map Timeframe to Yahoo Interval/Range
            $params = match($tf) {
                '1M' => ['interval' => '1m', 'range' => '1d'],   // Intraday
                '15M' => ['interval' => '15m', 'range' => '5d'],
                '1H' => ['interval' => '60m', 'range' => '1mo'], // Good availability
                '4H' => ['interval' => '60m', 'range' => '3mo'], // Yahoo doesn't support 4h, well aggregate or just show 1h
                '1D' => ['interval' => '1d', 'range' => '2y'],
                default => ['interval' => '60m', 'range' => '1mo']
            };

            $url = "https://query1.finance.yahoo.com/v8/finance/chart/{$yahooSymbol}?interval={$params['interval']}&range={$params['range']}";

            // 3. Fetch Data
            $response = \Illuminate\Support\Facades\Http::get($url);
            
            if ($response->successful()) {
                $raw = $response->json();
                $result = $raw['chart']['result'][0] ?? null;
                
                if ($result) {
                    $timestamps = $result['timestamp'] ?? [];
                    $quote = $result['indicators']['quote'][0] ?? [];
                    
                    $opens = $quote['open'] ?? [];
                    $highs = $quote['high'] ?? [];
                    $lows = $quote['low'] ?? [];
                    $closes = $quote['close'] ?? [];
                    
                    $candles = [];
                    foreach ($timestamps as $i => $time) {
                        // Skip incomplete candles
                        if (!isset($opens[$i]) || !isset($closes[$i])) continue;
                        
                        $candles[] = [
                            'time' => $time,
                            'open' => round($opens[$i], 5),
                            'high' => round($highs[$i], 5),
                            'low' => round($lows[$i], 5),
                            'close' => round($closes[$i], 5),
                        ];
                    }

                    // Sort valid candles
                   // $candles = collect($candles)->sortBy('time')->values()->all();

                    return response()->json([
                        'success' => true,
                        'source' => 'yahoo_finance',
                        'data' => $candles
                    ]);
                }
            }
            
            // Fallback if Yahoo fails
            return response()->json([
                'success' => true, 
                'source' => 'mock_fallback_network_error',
                'data' => $this->generateRandomWalk(2000, $tf)
            ]);

        } catch (\Exception $e) {
            Log::error("Yahoo Data Error: " . $e->getMessage());
            return response()->json(['success' => false, 'message' => $e->getMessage()], 500);
        }
    }

    protected function generateRandomWalk($currentPrice, $timeframe) {
        $data = [];
        $secondsPerCandle = match($timeframe) {
            '1M' => 60,
            '15M' => 900,
            '4H' => 14400,
            '1D' => 86400,
            default => 3600
        };
        $time = Carbon::now()->subSeconds($secondsPerCandle * 100)->timestamp;
        $price = $currentPrice;

        for ($i = 0; $i < 100; $i++) {
            $volatility = $price * 0.002;
            $change = (rand(-100, 100) / 100) * $volatility;
            $open = $price;
            $close = $open + $change;
            $high = max($open, $close) + $volatility/2;
            $low = min($open, $close) - $volatility/2;

            $data[] = [
                'time' => $time,
                'open' => round($open, 5),
                'high' => round($high, 5),
                'low' => round($low, 5),
                'close' => round($close, 5),
            ];
            $price = $close;
            $time += $secondsPerCandle;
        }
        return $data;
    }
}

// Add simple Carbon Macro for subCandles if needed or just logic above
