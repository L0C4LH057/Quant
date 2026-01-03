<?php
// Debug script to check why trade filtering is failing
$token = trim(str_replace('META_API_TOKEN=', '', shell_exec('grep META_API_TOKEN .env')));
$accountId = 'e6993a89-53ec-4311-96c4-62a910fddbd8';

// Testing History Endpoint
$startTime = date('Y-m-d\TH:i:s.000\Z', strtotime('-90 days'));
$endTime = date('Y-m-d\TH:i:s.000\Z', strtotime('+1 hour'));

$url = 'https://mt-client-api-v1.london.agiliumtrade.ai/users/current/accounts/'.$accountId.'/history-deals/time/'.$startTime.'/'.$endTime;

$ch = curl_init($url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false);
curl_setopt($ch, CURLOPT_HTTPHEADER, ['auth-token: ' . $token]);
$resp = curl_exec($ch);
curl_close($ch);

$data = json_decode($resp, true);

echo "Total Deals Found: " . count($data) . "\n";

foreach ($data as $deal) {
  echo "Deal ID: " . $deal['id'] . " | Type: " . $deal['type'] . " | Symbol: " . ($deal['symbol'] ?? 'N/A') . " | Profit: " . $deal['profit'] . "\n";
}
