<?php
$token = trim(str_replace('META_API_TOKEN=', '', shell_exec('grep META_API_TOKEN .env')));
$accountId = 'e6993a89-53ec-4311-96c4-62a910fddbd8';

// Testing History Endpoint
$startTime = date('Y-m-d\TH:i:s.000\Z', strtotime('-90 days'));
$endTime = date('Y-m-d\TH:i:s.000\Z', strtotime('+1 hour'));

$url = 'https://mt-client-api-v1.london.agiliumtrade.ai/users/current/accounts/'.$accountId.'/history-deals/time/'.$startTime.'/'.$endTime;
echo "URL: $url\n";

$ch = curl_init($url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false);
curl_setopt($ch, CURLOPT_HTTPHEADER, ['auth-token: ' . $token]);
$resp = curl_exec($ch);
$code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

echo 'History Code: ' . $code . PHP_EOL;
echo 'History Resp Snippet: ' . substr($resp, 0, 500) . PHP_EOL;
