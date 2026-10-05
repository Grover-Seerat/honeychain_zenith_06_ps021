# HoneyChain production blockchain anchor

HoneyChain keeps a local SHA-256 hash-linked ledger for deterministic verification and can additionally anchor every ledger block to a public EVM blockchain.

## What gets anchored
Every important level creates a HoneyChain block: Hive registration, IoT reading/CSV import, Health Analysis, Hive Source, Harvest, Processing, Packaging, QR generation and other traceability events.

## Deployment
1. Deploy `HoneyChainTraceability.sol` to an EVM network (Sepolia is suitable for a demo).
2. Run `deploy.py` with:
   - `HONEYCHAIN_BLOCKCHAIN_RPC_URL`
   - `HONEYCHAIN_BLOCKCHAIN_PRIVATE_KEY`
3. Put the returned contract address in `HONEYCHAIN_BLOCKCHAIN_CONTRACT_ADDRESS`.
4. Set `HONEYCHAIN_BLOCKCHAIN_EXPLORER_URL` to the explorer base for the selected network.
5. Restart the backend.

No application API token is required by HoneyChain itself. A hosted RPC provider may require an API key in its RPC URL. Never put a private key in frontend code.

## IoT deployment
The private LAN address `10.192.35.56` cannot be reached by a cloud deployment. For production, configure the device/ESP32 to POST every 15 seconds to:
`POST https://YOUR-BACKEND/api/iot/readings`
with JSON `{ "hive_id": "HC-001", "timestamp": "...", "temperature": 34.1, "humidity": 62.4 }` and optional `X-Device-Key`.

For local development, HoneyChain can still pull from `HONEYCHAIN_IOT_URL`.
