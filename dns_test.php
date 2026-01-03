<?php
$domains = [
    'mt-provisioning-api-v1.agiliumtrade.ai',
    'mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai',
    'mt-client-api-v1.london.agiliumtrade.ai',
    'mt-client-api-v1.london.agiliumtrade.agiliumtrade.ai'
];

foreach ($domains as $domain) {
    if (gethostbyname($domain) !== $domain) {
        echo "[√] RESOLVED: $domain\n";
    } else {
        echo "[X] FAILED:   $domain\n";
    }
}
