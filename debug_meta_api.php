<?php

use Illuminate\Support\Facades\Http;
use App\Models\TradingAccount;

require __DIR__.'/vendor/autoload.php';
$app = require_once __DIR__.'/bootstrap/app.php';
$kernel = $app->make(Illuminate\Contracts\Console\Kernel::class);
$kernel->bootstrap();

$accountId = 'e6993a89-53ec-4311-96c4-62a910fddbd8'; 
$token = config('services.meta_api.token');
// Use the config value exactly as is
$provisioningBaseUrl = config('services.meta_api.account_management_url');

echo "\n--- DEBUG START ---\n";
echo "Account ID: $accountId\n";
echo "Token Prefix: " . substr($token, 0, 5) . "...\n";
echo "Provisioning URL: $provisioningBaseUrl\n";

// 1. Check Provisioning API (Metadata)
echo "\n[1] Fetching Metadata from Provisioning API...\n";
$metaUrl = "$provisioningBaseUrl/users/current/accounts/$accountId";
try {
    $response = Http::withoutVerifying()->withHeaders(['auth-token' => $token])->get($metaUrl);
    echo "Status: " . $response->status() . "\n";
    if ($response->successful()) {
        $data = $response->json();
        echo "Region: " . ($data['region'] ?? 'N/A') . "\n";
        echo "Connection Status: " . ($data['connectionStatus'] ?? 'N/A') . "\n";
        echo "State: " . ($data['state'] ?? 'N/A') . "\n";
        
        $region = $data['region'] ?? 'new-york';
    } else {
        echo "FAIL: " . $response->body() . "\n";
        exit;
    }
} catch (\Exception $e) {
    echo "EXCEPTION: " . $e->getMessage() . "\n";
    exit;
}

// 2. Comprehensive Connectivity Test
echo "\n[2] Connectivity Matrix...\n";
$region = $data['region'] ?? 'london'; 

$domains = [
    'Standard' => "mt-client-api-v1.{$region}.agiliumtrade.ai",
    'Double'   => "mt-client-api-v1.{$region}.agiliumtrade.agiliumtrade.ai",
    'NoRegion-Std' => "mt-client-api-v1.agiliumtrade.ai",
    'NoRegion-Dbl' => "mt-client-api-v1.agiliumtrade.agiliumtrade.ai",
    'G2-Std' => "mt-client-api-v1.{$region}.g2.agiliumtrade.ai",
];

foreach ($domains as $label => $host) {
    echo "\n--- $label ($host) ---\n";
    
    // DNS Check
    $ip = gethostbyname($host);
    echo "DNS: " . ($ip == $host ? "FAIL" : $ip) . "\n";
    
    if ($ip != $host) {
        $url = "https://{$host}/users/current/accounts/$accountId/information";
        try {
            $response = Http::withoutVerifying()->timeout(5)->withHeaders(['auth-token' => $token])->get($url);
            echo "HTTP: " . $response->status() . "\n";
            if ($response->successful()) {
                echo "SUCCESS! >>> " . $url . "\n";
                echo "Balance: " . ($response->json()['balance'] ?? 'N/A') . "\n";
            }
        } catch (\Exception $e) {
            echo "Err: " . $e->getMessage() . "\n";
        }
    }
}

echo "\n--- DEBUG END ---\n";
