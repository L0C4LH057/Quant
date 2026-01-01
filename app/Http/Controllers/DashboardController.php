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
        $totalReads = 0;

        // 1. Aggregate Real History
        foreach ($accounts as $account) {
            if ($account->meta_api_id && $account->status === 'active') { // Only active accounts
                try {
                    $service = $this->tradingFactory->make($account->broker_type);
                    // Fetch last 30 days
                    $history = $service->getHistory($account->meta_api_id, 30);
                    
                    if (isset($history['deals']) && is_array($history['deals'])) {
                         foreach ($history['deals'] as $deal) {
                            // Filter out balance operations if needed, usually type starts with DEAL_TYPE_BALANCE
                            if (!in_array($deal['type'], ['DEAL_TYPE_BUY', 'DEAL_TYPE_SELL'])) continue;

                            $allTrades->push([
                                'id' => $deal['id'],
                                'pair' => $deal['symbol'],
                                'type' => str_replace('DEAL_TYPE_', '', $deal['type']), // BUY or SELL
                                'price' => $deal['price'],
                                'profit' => $deal['profit'],
                                'status' => $deal['profit'] >= 0 ? 'won' : 'lost',
                                'time' => \Carbon\Carbon::parse($deal['time']),
                            ]);
                         }
                    }
                } catch (\Exception $e) {
                    // Log error but continue
                }
            }
        }

        // 2. Sort and Slice for "Recent Trades"
        $aiTrades = $allTrades->sortByDesc('time')->take(10);

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

        return view('dashboard', compact('accounts', 'aiStats', 'aiTrades'));
    }

    public function toggleAi(Request $request, $id)
    {
        $account = TradingAccount::where('user_id', Auth::id())->findOrFail($id);
        
        // Explicit toggle
        $newState = !$account->is_ai_active;
        $account->is_ai_active = $newState;
        $account->save();

        return response()->json([
            'success' => true, 
            'is_ai_active' => $account->is_ai_active,
            'message' => $account->is_ai_active ? 'AI Activated for ' . ($account->login ?? 'Account') : 'AI Paused for ' . ($account->login ?? 'Account')
        ]);
    }
}
