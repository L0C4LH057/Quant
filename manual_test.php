<?php
$token = trim(str_replace('META_API_TOKEN=', '', shell_exec('grep META_API_TOKEN .env')));
$accountId = 'e6993a89-53ec-4311-96c4-62a910fddbd8';

// 1. Provisioning Check
$provUrl = 'https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts/' . $accountId;
echo "Provisioning URL: $provUrl\n";

$ch = curl_init($provUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false);
curl_setopt($ch, CURLOPT_HTTPHEADER, ['auth-token: ' . $token]);
$response = curl_exec($ch);
$info = curl_getinfo($ch);
curl_close($ch);

echo "Prov Code: " . $info['http_code'] . "\n";
echo "Prov Resp: " . $response . "\n";

if ($info['http_code'] == 200) {
    $data = json_decode($response, true);
    $region = $data['region'];
    echo "Region: $region\n";

    // 2. Client Check (Standard)
    $clientUrl = "https://mt-client-api-v1.$region.agiliumtrade.ai/users/current/accounts/$accountId/account-information";
    echo "Client URL: $clientUrl\n";

    $ch2 = curl_init($clientUrl);
    curl_setopt($ch2, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch2, CURLOPT_SSL_VERIFYPEER, false);
    curl_setopt($ch2, CURLOPT_HTTPHEADER, ['auth-token: ' . $token]);
    $resp2 = curl_exec($ch2);
    $info2 = curl_getinfo($ch2);
    curl_close($ch2);

    echo "Client Code: " . $info2['http_code'] . "\n";
    echo "Client Resp: " . $resp2 . "\n";
}
