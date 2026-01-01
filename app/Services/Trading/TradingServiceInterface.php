<?php

namespace App\Services\Trading;

interface TradingServiceInterface
{
    /**
     * Connect to a trading account (verify credentials).
     * 
     * @param string $login
     * @param string $password
     * @param string $server
     * @param string $platform (mt4/mt5)
     * @return array {'id' => string, 'success' => bool, 'message' => string}
     */
    public function connect(string $login, string $password, string $server, string $platform): array;

    /**
     * Get real-time account information (balance, equity).
     */
    public function getAccountInfo(string $accountId): array;

    /**
     * Get historical trades.
     */
    public function getHistory(string $accountId, int $days = 30): array;
    
    /**
     * Get currently open positions.
     */
    public function getPositions(string $accountId): array;

    /**
     * Deploy the AI Strategy (Ea/Signal) to the account.
     */
    public function deployStrategy(string $accountId, string $strategyId): bool;

    // --- Trading Methods ---

    /**
     * Execute a market order.
     * @return array {'success' => bool, 'id' => string, 'message' => string}
     */
    public function executeTrade(string $accountId, string $symbol, string $action, float $volume, float $stopLoss = 0, float $takeProfit = 0): array;

    /**
     * Place a pending limit/stop order.
     */
    public function placeLimitOrder(string $accountId, string $symbol, string $action, float $volume, float $price, float $stopLoss = 0, float $takeProfit = 0): array;

    /**
     * Modify an existing trade/order.
     */
    public function modifyTrade(string $accountId, string $tradeId, float $stopLoss, float $takeProfit): array;

    /**
     * Close an existing trade/order.
     */
    public function closeTrade(string $accountId, string $tradeId): array;
}
