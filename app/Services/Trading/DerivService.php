<?php

namespace App\Services\Trading;

use Illuminate\Support\Facades\Log;

class DerivService implements TradingServiceInterface
{
    protected $appId = 1089; // Public default App ID for testing
    protected $endpoint = 'ssl://ws.binaryws.com:443';

    public function connect(string $login, string $password, string $server, string $platform): array
    {
        $token = $password;
        Log::info("Deriv Connect Attempt: Login=$login");

        try {
            $host = 'ws.binaryws.com';
            $port = 443;
            $path = '/websockets/v3?app_id=' . $this->appId;

            // 1. Open TCP/SSL Connection
            $socket = stream_socket_client("ssl://{$host}:{$port}", $errno, $errstr, 5);
            
            if (!$socket) {
                throw new \Exception("Could not connect to Deriv WS: $errstr ($errno)");
            }

            // 2. Perform WebSocket Handshake
            $key = base64_encode(random_bytes(16));
            $headers = "GET {$path} HTTP/1.1\r\n" .
                       "Host: {$host}\r\n" .
                       "Upgrade: websocket\r\n" .
                       "Connection: Upgrade\r\n" .
                       "Sec-WebSocket-Key: {$key}\r\n" .
                       "Sec-WebSocket-Version: 13\r\n\r\n";
            
            fwrite($socket, $headers);

            // 3. Read Handshake Response
            $responseHeader = fread($socket, 4096);
            if (!str_contains($responseHeader, '101 Switching Protocols')) {
                Log::error("Deriv Handshake Failed: " . $responseHeader);
                return ['success' => false, 'message' => 'Deriv Handshake Failed (See logs)'];
            }

            // 4. Send Authorize Request (Frame Masked)
            $payload = json_encode(['authorize' => $token]);
            $this->writeFrame($socket, $payload);

            // 5. Read Frame
            $responsePayload = $this->readFrame($socket);
            fclose($socket);

            Log::info("Deriv Response: " . $responsePayload);

            $data = json_decode($responsePayload, true);

            if (isset($data['error'])) {
                 return [
                    'success' => false,
                    'message' => $data['error']['message'] ?? 'Invalid Deriv Token'
                ];
            }

            if (isset($data['authorize'])) {
                return [
                    'success' => true,
                    'id' => 'deriv_' . $data['authorize']['loginid'],
                    'message' => 'Deriv Account Connected Successfully',
                    'data' => [
                        'connectionStatus' => 'CONNECTED',
                        'balance' => $data['authorize']['balance'] ?? 0.00,
                        'currency' => $data['authorize']['currency'] ?? 'USD',
                        'email' => $data['authorize']['email'] ?? ''
                    ]
                ];
            }

            return ['success' => false, 'message' => 'Unknown response from Deriv'];

        } catch (\Exception $e) {
            Log::error("Deriv Error: " . $e->getMessage());
            return ['success' => false, 'message' => 'System error connecting to Deriv.'];
        }
    }

    /**
     * Write a Masked Text Frame (Client -> Server)
     */
    protected function writeFrame($socket, $payload)
    {
        $len = strlen($payload);
        $frame = chr(0x81); // FIN + Text

        // Client frames must be masked (bit 7 set)
        if ($len < 126) {
            $frame .= chr(0x80 | $len);
        } elseif ($len < 65536) {
            $frame .= chr(0x80 | 126) . pack('n', $len);
        } else {
            $frame .= chr(0x80 | 127) . pack('J', $len);
        }

        $maskKey = random_bytes(4);
        $frame .= $maskKey;

        // Mask Payload
        for ($i = 0; $i < $len; $i++) {
            $payload[$i] = chr(ord($payload[$i]) ^ ord($maskKey[$i % 4]));
        }
        $frame .= $payload;

        fwrite($socket, $frame);
    }

    /**
     * Read an Unmasked Text Frame (Server -> Client)
     */
    protected function readFrame($socket)
    {
        // 1. Read first 2 bytes (FIN/Opcode + Len)
        $header = fread($socket, 2);
        if (!$header) return '';

        $len = ord($header[1]) & 0x7F;

        if ($len === 126) {
            $extended = fread($socket, 2);
            $len = unpack('n', $extended)[1];
        } elseif ($len === 127) {
            $extended = fread($socket, 8);
            $len = unpack('J', $extended)[1];
        }

        // Server frames are NOT masked
        return fread($socket, $len);
    }

    public function getAccountInfo(string $accountId): array
    {
        return [
             'balance' => 0.00,
             'equity' => 0.00,
             'margin' => 0.00
        ];
    }

    public function getHistory(string $accountId, int $days = 30): array
    {
        return [];
    }

    public function getPositions(string $accountId): array
    {
        return [];
    }

    public function deployStrategy(string $accountId, string $strategyId): bool
    {
        return true;
    }

    public function executeTrade(string $accountId, string $symbol, string $action, float $volume, float $stopLoss = 0, float $takeProfit = 0): array
    {
        return ['success' => false, 'message' => 'Deriv Trading Not Yet Implemented'];
    }

    public function placeLimitOrder(string $accountId, string $symbol, string $action, float $volume, float $price, float $stopLoss = 0, float $takeProfit = 0): array
    {
         return ['success' => false, 'message' => 'Deriv Limit Orders Not Yet Implemented'];
    }

    public function modifyTrade(string $accountId, string $tradeId, float $stopLoss, float $takeProfit): array
    {
         return ['success' => false, 'message' => 'Deriv Modify Not Yet Implemented'];
    }

    public function closeTrade(string $accountId, string $tradeId): array
    {
         return ['success' => false, 'message' => 'Deriv Close Not Yet Implemented'];
    }
}
