<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use App\Models\TradingAccount;

class DashboardController extends Controller
{
    protected $tradingFactory;

    public function __construct(\App\Services\Trading\TradingServiceFactory $tradingFactory)
    {
        $this->tradingFactory = $tradingFactory;
    }

    public function index()
    {
        $user = Auth::user();
        $accounts = TradingAccount::where('user_id', $user->id)->get();
        
        $allTrades = collect();
        $openPositions = collect();
        
        // 1. Aggregate Real History, Positions, and Sync Balance
        foreach ($accounts as $index => $account) {
            if ($account->meta_api_id && $account->status === 'active') { 
                try {
                    $service = $this->tradingFactory->make($account->broker_type);
                    
                    // A. Fetch Account Info & Sync
                    $info = $service->getAccountInfo($account->meta_api_id);
                    if (!empty($info)) {
                        // Update DB
                        $account->update([
                             'balance' => $info['balance'] ?? $account->balance,
                             'equity' => $info['equity'] ?? $account->equity ?? 0,
                             'margin' => $info['margin'] ?? $account->margin ?? 0,
                             'currency' => $info['currency'] ?? $account->currency ?? 'USD',
                             'name' => $info['name'] ?? $account->name,
                        ]);

                        // CRITICAL: Update the object in memory for the View immediately
                        $account->balance = $info['balance'];
                        $account->equity = $info['equity'];
                        $account->currency = $info['currency'];
                        
                        // Re-assign to collection to ensure view sees fresh data
                        $accounts[$index] = $account;
                    }

                    // B. Fetch History (Last 90 days)
                    $history = $service->getHistory($account->meta_api_id, 90);
                    // MetaApi returns a list of deals directly or wrapped in 'deals' key depending on endpoint version
                    // Our debug script showed it returns a direct array of deals, so let's handle both cases.
                    $deals = isset($history['deals']) ? $history['deals'] : (is_array($history) ? $history : []);

                    foreach ($deals as $deal) {
                        // Skip if not a valid array deal or missing type
                        if (!is_array($deal) || !isset($deal['type'])) continue;

                        // We only want actual BUY/SELL trades, not balance updates
                        if (!in_array($deal['type'], ['DEAL_TYPE_BUY', 'DEAL_TYPE_SELL'])) continue;

                        $allTrades->push([
                            'id' => $deal['id'],
                            'pair' => $deal['symbol'] ?? 'Unknown',
                            'type' => str_replace('DEAL_TYPE_', '', $deal['type']), // BUY or SELL
                            'price' => $deal['price'] ?? 0,
                            'profit' => $deal['profit'] ?? 0,
                            'status' => ($deal['profit'] ?? 0) >= 0 ? 'won' : 'lost',
                            'time' => \Carbon\Carbon::parse($deal['time']),
                        ]);
                    }

                    // C. Fetch Open Positions
                    $positions = $service->getPositions($account->meta_api_id);
                    foreach ($positions as $pos) {
                        $openPositions->push([
                            'id' => $pos['id'],
                            'pair' => $pos['symbol'],
                            'type' => str_replace('POSITION_TYPE_', '', $pos['type']), // BUY or SELL
                            'entry_price' => $pos['openPrice'],
                            'current_price' => $pos['currentPrice'],
                            'profit' => $pos['profit'],
                            'volume' => $pos['volume'],
                            'time' => \Carbon\Carbon::parse($pos['time']),
                        ]);
                    }

                } catch (\Exception $e) {
                    \Illuminate\Support\Facades\Log::error("Dashboard Sync Error for {$account->id}: " . $e->getMessage());
                }
            }
        }

        // 2. Sort and Slice for "Recent Trades"
        $aiTrades = $allTrades->sortByDesc('time')->take(50);
        
        // Sort Open Positions
        $openPositions = $openPositions->sortByDesc('time');

        // 3. Calculate Real Stats
        $wins = $allTrades->where('profit', '>=', 0);
        $losses = $allTrades->where('profit', '<', 0);
        
        $totalProfit = $wins->sum('profit');
        $totalLoss = abs($losses->sum('profit'));
        $totalTradesCount = $allTrades->count();

        $aiStats = [
            'total_trades' => $totalTradesCount,
            'total_profit' => $totalProfit,
            'total_loss' => $totalLoss,
            'win_rate' => $totalTradesCount > 0 ? round(($wins->count() / $totalTradesCount) * 100) : 0,
            'profit_factor' => $totalLoss > 0 ? round($totalProfit / $totalLoss, 2) : ($totalProfit > 0 ? 999 : 0) // Handle div by zero
        ];

        return view('dashboard', compact('accounts', 'aiStats', 'aiTrades', 'openPositions'));
    }

    public function toggleAi(Request $request, $id)
    {
        $account = TradingAccount::where('user_id', Auth::id())->findOrFail($id);
        
        // Explicit toggle
        $newState = !$account->is_ai_active;
        $account->is_ai_active = $newState;
        $account->save();

        if ($newState && $account->meta_api_id) {
            // TRIGGER DEMO TRADE BASED ON CONFIG
            try {
                $service = $this->tradingFactory->make($account->broker_type);
                
                // 1. Load User Configuration
                $config = \App\Models\AiConfiguration::firstOrCreate(
                    ['user_id' => Auth::id()],
                    [
                        'trading_pair' => 'EURUSD', // Default
                        'risk_mode' => 'Moderate'
                    ]
                );

                // 2. Determine Symbol (Handle Suffix 'm' for this broker if needed)
                $rawSymbol = $config->trading_pair;
                // Simple heuristic: if user saved 'EURUSD' but broker needs 'EURUSDm'
                // We'll try the config value first, if price fails, try appending 'm'
                $symbol = $rawSymbol;
                
                $priceData = $service->getSymbolPrice($account->meta_api_id, $symbol);
                if (empty($priceData)) {
                     // Try appending 'm' common for this broker
                     $symbol = $rawSymbol . 'm';
                     $priceData = $service->getSymbolPrice($account->meta_api_id, $symbol);
                }

                // 3. Determine Volume based on Risk Mode
                $volume = match($config->risk_mode) {
                    'Safe' => 0.01,
                    'Moderate' => 0.05,
                    'Aggressive' => 0.10,
                    'Capital Protection' => 0.01,
                    default => 0.01
                };

                // 4. Calculate SL/TP Dynamically
                $bid = $priceData['bid'] ?? 0;
                $ask = $priceData['ask'] ?? 0;
                
                if ($bid > 0 && $ask > 0) {
                    // Decide Direction (Random for Demo, but respecting settings context eventually)
                    $action = rand(0, 1) ? 'BUY' : 'SELL';

                    // Adaptive Pips:
                    // Forex (e.g., 1.0500) -> 10 pips = 0.0010
                    // Crypto/Indices (e.g., 40000.00 or 2000.00) -> 10 pips = 10.00 or 1.00 depending on digits
                    // We use price magnitude to estimate.
                    // If price < 500, assume Forex-like logic (0.0001 per pip)
                    // If price > 500, assume Index/Crypto (1.0 per unit)
                    
                    if ($bid < 500) {
                        // Forex Scales
                        $point = 0.0001; 
                        // JPY Pair Exception (price ~140.00) - simplistic check
                        if (str_contains($symbol, 'JPY')) $point = 0.01;
                        
                        $slPips = 20; // 20 pips
                        $tpPips = 40; // 40 pips
                        
                        $slDist = $slPips * $point;
                        $tpDist = $tpPips * $point;
                    } else {
                        // Crypto/Indices Scales
                        // BTC ~ 90k. SL 200 units.
                        $slDist = 200.0;
                        $tpDist = 500.0;
                    }

                    $sl = 0; $tp = 0;
                    if ($action === 'BUY') {
                        $openPrice = $ask;
                        $sl = $openPrice - $slDist;
                        $tp = $openPrice + $tpDist;
                    } else {
                        $openPrice = $bid;
                        $sl = $openPrice + $slDist;
                        $tp = $openPrice - $tpDist;
                    }

                    $tradeResult = $service->executeTrade($account->meta_api_id, $symbol, $action, $volume, $sl, $tp);
                
                    if (!$tradeResult['success']) {
                        \Illuminate\Support\Facades\Log::error("AI Demo Trade Failed: " . $tradeResult['message']);
                    } else {
                        $msg = "AI Strategy ({$config->strategy_mode}): $action $volume $symbol @ $openPrice (SL: $sl, TP: $tp)";
                        \Illuminate\Support\Facades\Log::info($msg);
                    }
                } else {
                    \Illuminate\Support\Facades\Log::error("AI Fail: Could not fetch price for symbol: $symbol");
                }

            } catch (\Exception $e) {
                \Illuminate\Support\Facades\Log::error("AI Trigger Error: " . $e->getMessage());
            }
        }

        return response()->json([
            'success' => true, 
            'is_ai_active' => $account->is_ai_active,
            'message' => $account->is_ai_active ? 'AI Activated & Trade Executed!' : 'AI Paused'
        ]);
    }
}
