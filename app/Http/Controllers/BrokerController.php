<?php

namespace App\Http\Controllers;

use App\Models\TradingAccount;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Log;

class BrokerController extends Controller
{
    protected $tradingFactory;

    public function __construct(\App\Services\Trading\TradingServiceFactory $tradingFactory)
    {
        $this->tradingFactory = $tradingFactory;
    }

    /**
     * Display connected accounts.
     */
    public function index()
    {
        $accounts = TradingAccount::where('user_id', Auth::id())->latest()->get();

        // Sync latest data from Broker
        foreach ($accounts as $account) {
            if ($account->status === 'active' && $account->meta_api_id) {
                try {
                    $service = $this->tradingFactory->make($account->broker_type);
                    $info = $service->getAccountInfo($account->meta_api_id);

                    if (!empty($info)) {
                        $account->update([
                            'balance' => $info['balance'] ?? $account->balance,
                            'equity' => $info['equity'] ?? $account->equity ?? 0,
                            'margin' => $info['margin'] ?? $account->margin ?? 0,
                            'currency' => $info['currency'] ?? $account->currency ?? 'USD',
                            'name' => $info['name'] ?? $account->name,
                        ]);
                    }
                } catch (\Exception $e) {
                    Log::error("Failed to sync account {$account->id}: " . $e->getMessage());
                }
            }
        }
        
        // Refetch to ensure view gets updated values
        $accounts = TradingAccount::where('user_id', Auth::id())->latest()->get();
        return view('accounts.index', compact('accounts'));
    }

    /**
     * Handle broker connection.
     */
    public function connect(Request $request)
    {
        $request->validate([
            'type' => 'required|in:mt4,mt5,deriv,crypto',
            'login_id' => 'required', 
            'password' => 'required|string',
            'server' => 'nullable|string',
        ]);

        // Attempt connection via Factory
        try {
            $service = $this->tradingFactory->make($request->type);
            $result = $service->connect(
                $request->login_id,
                $request->password,
                $request->input('server', 'default'),
                $request->type
            );
        } catch (\Exception $e) {
             return back()->withErrors(['login_id' => $e->getMessage()]);
        }

        if (!$result['success']) {
            return back()->withErrors(['login_id' => 'Connection failed: ' . $result['message']]);
        }

        // Create Account (Store encrypted password automatically via model cast)
        TradingAccount::create([
            'user_id' => Auth::id(),
            'name' => $result['data']['name'] ?? 'Quant User ' . $request->login_id,
            'broker_type' => $request->type,
            'login_id' => $request->login_id,
            'server' => $request->input('server') ?? 'N/A',
            'password' => $request->password, 
            'status' => 'active',
            'meta_api_id' => $result['id'],
            'balance' => $result['data']['balance'] ?? 0.00,
            'equity' => $result['data']['equity'] ?? 0.00,
            'margin' => $result['data']['margin'] ?? 0.00,
            'currency' => $result['data']['currency'] ?? 'USD',
        ]);

        return redirect()->route('accounts.index')->with('status', 'Account connected successfully (Secured).');
    }
    /**
     * Force sync an account.
     */
    public function sync($id)
    {
        $account = TradingAccount::where('user_id', Auth::id())->findOrFail($id);
        
        try {
            if (!$account->meta_api_id) {
                return back()->withErrors(['sync' => 'Account is not connected to MetaApi correctly.']);
            }
            
            $service = $this->tradingFactory->make($account->broker_type);
            $info = $service->getAccountInfo($account->meta_api_id);
            
            if (empty($info)) {
                 return back()->withErrors(['sync' => 'Sync failed. Broker returned no data. Check if account is deployed.']);
            }
            
            $updateData = [
                'balance' => $info['balance'] ?? $account->balance,
                'equity' => $info['equity'] ?? $account->equity ?? 0,
                'margin' => $info['margin'] ?? $account->margin ?? 0,
                'currency' => $info['currency'] ?? $account->currency ?? 'USD',
                'name' => $info['name'] ?? $account->name,
            ];

            $account->update($updateData);

            return back()->with('status', 'Account synchronized successfully. Equity: ' . $updateData['equity']);

        } catch (\Exception $e) {
            Log::error("Manual Sync Failed: " . $e->getMessage());
            return back()->withErrors(['sync' => 'Sync error: ' . $e->getMessage()]);
        }
    }
}
