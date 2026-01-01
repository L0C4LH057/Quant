<?php

namespace App\Services\Trading;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class MetaApiService implements TradingServiceInterface
{
    protected $token;
    protected $baseUrl = 'https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai';

    public function __construct()
    {
        $this->token = config('services.meta_api.token');
        $this->baseUrl = config('services.meta_api.account_management_url', 'https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai');
    }

    public function connect(string $login, string $password, string $server, string $platform): array
    {
        Log::info("MetaApi Connect Attempt: Login=$login, Server=$server, Platform=$platform");
        
        // Debug Config
        if (empty($this->token)) {
            Log::warning("MetaApi: Token is EMPTY. Falling back to simulation.");
            return [
                'success' => true,
                'id' => 'mock_' . uniqid(),
                'message' => 'Connected (Simulated - No Token Found)',
                'data' => [
                    'connectionStatus' => 'CONNECTED',
                    'balance' => rand(1000, 50000)
                ]
            ];
        }

        Log::info("MetaApi: Token found (" . substr($this->token, 0, 10) . "...). Attempting real connection...");

        // 2. Real API implementation
        try {
            $response = Http::withoutVerifying()->withHeaders([
                'auth-token' => $this->token,
                'Content-Type' => 'application/json'
            ])->post("{$this->baseUrl}/users/current/accounts", [
                'name' => "Quant User " . $login,
                'login' => $login,
                'password' => $password,
                'server' => $server,
                'platform' => $platform,
                'magic' => 1000,
                'quoteStreamingIntervalInSeconds' => 2.5,
                'reliability' => 'regular'
            ]);

            if ($response->successful()) {
                $data = $response->json();
                return [
                    'success' => true,
                    'id' => $data['id'],
                    'message' => 'Account provisioned successfully',
                    'data' => $data
                ];
            }

            Log::error("MetaApi API Error: " . $response->body());

            return [
                'success' => false,
                'message' => $response->json()['message'] ?? 'Failed to connect to MetaApi (See logs)'
            ];

        } catch (\Exception $e) {
            Log::error("MetaApi Connect Error: " . $e->getMessage());
            return [
                'success' => false,
                'message' => 'System error connecting to broker.'
            ];
        }
    }

    public function getAccountInfo(string $accountId): array
    {
        try {
            $url = "{$this->getTradeUrl()}/users/current/accounts/{$accountId}/information";
            
            $response = Http::withoutVerifying()->withHeaders([
                'auth-token' => $this->token,
            ])->get($url);

            if ($response->successful()) {
                $data = $response->json();
                return [
                    'name' => $data['name'] ?? null,
                    'balance' => $data['balance'] ?? 0.00,
                    'equity' => $data['equity'] ?? 0.00,
                    'margin' => $data['margin'] ?? 0.00,
                    'freeMargin' => $data['freeMargin'] ?? 0.00,
                    'leverage' => $data['leverage'] ?? 100,
                    'currency' => $data['currency'] ?? 'USD',
                ];
            }

            Log::warning("MetaApi: Failed to fetch account info for {$accountId}. Status: " . $response->status());
            return []; // Return empty to indicate failure/no data

        } catch (\Exception $e) {
            Log::error("MetaApi Info Error: " . $e->getMessage());
            return [];
        }
    }

    public function getHistory(string $accountId, int $days = 30): array
    {
        try {
            $startTime = now()->subDays($days)->toIso8601String();
            $endTime = now()->addHours(1)->toIso8601String(); // slight buffer
            
            $url = "{$this->getTradeUrl()}/users/current/accounts/{$accountId}/history-deals/time/{$startTime}/{$endTime}";
            
            $response = Http::withoutVerifying()->withHeaders([
                'auth-token' => $this->token,
            ])->get($url);

            if ($response->successful()) {
                return $response->json();
            }
            
            Log::warning("MetaApi History Fail: " . $response->body());
            return [];

        } catch (\Exception $e) {
            Log::error("MetaApi History Error: " . $e->getMessage());
            return [];
        }
    }

    public function deployStrategy(string $accountId, string $strategyId): bool
    {
        return true;
    }

    public function getPositions(string $accountId): array
    {
        try {
            $url = "{$this->getTradeUrl()}/users/current/accounts/{$accountId}/positions";
            
            $response = Http::withoutVerifying()->withHeaders([
                'auth-token' => $this->token,
            ])->get($url);

            if ($response->successful()) {
                return $response->json();
            }
            
            Log::warning("MetaApi Positions Fail: " . $response->body());
            return [];

        } catch (\Exception $e) {
            Log::error("MetaApi Positions Error: " . $e->getMessage());
            return [];
        }
    }

    // --- Trading Implementation ---

    protected function getTradeUrl(): string
    {
        // Switch from provisioning to client API subdomain
        return str_replace('mt-provisioning-api-v1', 'mt-client-api-v1', $this->baseUrl);
    }

    public function executeTrade(string $accountId, string $symbol, string $action, float $volume, float $stopLoss = 0, float $takeProfit = 0): array
    {
        return $this->placeOrder($accountId, [
            'symbol' => $symbol,
            'actionType' => $action === 'BUY' ? 'ORDER_TYPE_BUY' : 'ORDER_TYPE_SELL',
            'volume' => $volume,
            'stopLoss' => $stopLoss,
            'takeProfit' => $takeProfit
        ]);
    }

    public function placeLimitOrder(string $accountId, string $symbol, string $action, float $volume, float $price, float $stopLoss = 0, float $takeProfit = 0): array
    {
        // Map common actions to MetaApi types (e.g. BUY_LIMIT, SELL_STOP)
        // For simplicity, assuming caller passes valid MetaApi Action Type or we keep it simple here.
        // Let's assume $action is passed as parameter like 'ORDER_TYPE_BUY_LIMIT'
        return $this->placeOrder($accountId, [
            'symbol' => $symbol,
            'actionType' => $action, 
            'volume' => $volume,
            'openPrice' => $price,
            'stopLoss' => $stopLoss,
            'takeProfit' => $takeProfit
        ]);
    }

    protected function placeOrder(string $accountId, array $payload): array
    {
        try {
            Log::info("MetaApi: Placing Order on $accountId", $payload);
            
            $url = "{$this->getTradeUrl()}/users/current/accounts/{$accountId}/trade";
            
            $response = Http::withoutVerifying()->withHeaders([
                'auth-token' => $this->token,
                'Content-Type' => 'application/json'
            ])->post($url, $payload);

            if ($response->successful()) {
                $data = $response->json();
                return [
                    'success' => true,
                    'id' => $data['orderId'] ?? $data['positionId'] ?? 'unknown',
                    'message' => 'Order executed successfully',
                    'data' => $data
                ];
            }

            Log::error("MetaApi Order Fail: " . $response->body());
            return ['success' => false, 'message' => $response->json()['message'] ?? 'Order failed'];

        } catch (\Exception $e) {
            Log::error("MetaApi Error: " . $e->getMessage());
            return ['success' => false, 'message' => 'System error placing order'];
        }
    }

    public function modifyTrade(string $accountId, string $tradeId, float $stopLoss, float $takeProfit): array
    {
        try {
            $url = "{$this->getTradeUrl()}/users/current/accounts/{$accountId}/orders/{$tradeId}";
            
            $response = Http::withoutVerifying()->withHeaders([
                'auth-token' => $this->token,
            ])->put($url, [
                'actionType' => 'ORDER_MODIFY', 
                'stopLoss' => $stopLoss,
                'takeProfit' => $takeProfit
            ]);

            return $response->successful() 
                ? ['success' => true, 'message' => 'Trade modified']
                : ['success' => false, 'message' => 'Modify failed: ' . $response->body()];

        } catch (\Exception $e) {
            return ['success' => false, 'message' => $e->getMessage()];
        }
    }

    public function closeTrade(string $accountId, string $tradeId): array
    {
        try {
            // Note: Close often uses a separate endpoint or 'ORDER_CLOSE' action
            // Using the MetaApi "close position" pattern usually
            $url = "{$this->getTradeUrl()}/users/current/accounts/{$accountId}/positions/{$tradeId}/close";
            
            $response = Http::withoutVerifying()->withHeaders([
                'auth-token' => $this->token,
            ])->post($url, []);

             return $response->successful() 
                ? ['success' => true, 'message' => 'Trade closed']
                : ['success' => false, 'message' => 'Close failed: ' . $response->body()];

        } catch (\Exception $e) {
             return ['success' => false, 'message' => $e->getMessage()];
        }
    }
}
