# PipFlow AI — Implementation Plan & Change Log

> Generated from the System Review Report.  Every item below has been
> **implemented** in this session.

---

## Summary

| Category | Count | Status |
|----------|------:|--------|
| Bugs fixed | 11 | ✅ Done |
| Security issues resolved | 3 | ✅ Done |
| Feature gaps filled | 8 | ✅ Done |
| Upgrades applied | 9 | ✅ Done |
| Code quality / hygiene | 3 | ✅ Done |
| **Total** | **34** | **All complete** |

---

## 1  Bug Fixes

### BUG-01 — `_rl_signal_node` uses fake constant-price data
**File:** `src/agents/orchestration/workflow.py`  
**Problem:** The RL signal node generated a DataFrame filled with a single constant price, zeroing out every technical indicator.  
**Fix:** The node now reads `state["market"]["ohlcv_df"]` (real OHLCV history).  A synthetic fallback with realistic noise is still available when no data is supplied, accompanied by a WARNING log.

### BUG-02 — `AgentState` dataclass incompatible with LangGraph
**File:** `src/agents/orchestration/state.py`  
**Problem:** LangGraph ≥ 0.2 requires `TypedDict` for node states; the `@dataclass`-based state caused silent channel-merge failures.  
**Fix:** Rewrote `MarketState`, `PortfolioState`, and `AgentState` as `TypedDict(total=False)`.  Added an `Annotated[List, operator.add]` reducer on the `messages` field.  Helper functions `create_initial_state()` and `state_to_dict()` were added.

### BUG-01+02 adaptation — `workflow.py` TypedDict migration
**File:** `src/agents/orchestration/workflow.py`  
All node methods now access state via `.get()` dict patterns, return `Dict[str, Any]`, and the `run()` entry point uses `create_initial_state()`.

### BUG-03 — Backtest endpoint returns stub `{"status": "TODO"}`
**File:** `src/api/routes/backtest.py`  
**Fix:** Fully implemented `execute_backtest()` as an async background task.  Routes now accept `BacktestRequest`, create a `BacktestJobStatus`, enqueue a `BackgroundTask`, and return the job ID immediately.

### BUG-04 — CORS allows `*` origins
**File:** `src/api/main.py`  
**Fix:** Replaced `allow_origins=["*"]` with `CORS_ORIGIN` env var (comma-separated list).  Defaults to `["http://localhost:3000"]`.

### BUG-05 — API key comparison via `!=` (timing-attack vulnerable)
**File:** `src/api/main.py`  
**Fix:** Switched to `hmac.compare_digest()`.

### BUG-06 — MT5 client created per request
**File:** `src/api/routes/trading.py`  
**Fix:** `get_mt5()` is now a module-level singleton initialised once.

### BUG-07 — `DiscreteTradingEnv.__init__` drops `reward_type` / `position_change_penalty`
**File:** `src/environments/discrete_trading_env.py`  
**Fix:** Added both parameters and forwarded them to `super().__init__()`.

### BUG-08 — Backtest jobs not persisted
**File:** `src/api/routes/backtest.py`  
**Fix:** Jobs saved as JSON in `config.backtest_results_path / "jobs"`. On restart, completed jobs are re-loadable via `GET /backtest/jobs`.

### BUG-09 — `DataPreprocessor._normalize()` silently no-ops when unfitted
**File:** `src/data/preprocessor.py`  
**Fix:** Added a guard that raises `RuntimeError("Normalization params not fitted …")` when `fit=False` and `_normalization_params` is empty.

### BUG-11 — `RLTraceWrapper` uses synchronous `requests` inside async pipeline
**File:** `src/agents/rl_agents/rl_trace_wrapper.py`  
**Fix:** Replaced `import requests` with `httpx.AsyncClient`.  All public methods are now `async def`.  Added `_get_client()` lazy init and `close()` cleanup.

---

## 2  Security Fixes

### SEC-01 — Hardcoded default API key `"pipflow-dev-key-change-me"`
**Files:** `src/api/routes/trading.py`  
**Fix:** Default changed to empty string with a WARNING log.  Production must set `MT5_BRIDGE_API_KEY`.

### SEC-02 — Prompt injection via unsanitised symbol names
**Files:** `src/utils/validators.py`, `src/llm/prompts.py`  
**Fix:** Added a strict regex allowlist `^[A-Z0-9][A-Z0-9.\-=/_^]{0,19}$` in `validate_symbol()`.  `format_market_analysis_prompt()` now validates before interpolation.

### SEC-05 — No rate limiting on the API
**File:** `src/api/main.py`  
**Fix:** Integrated `slowapi` with a configurable `RATE_LIMIT` env var (default `60/minute`).  Graceful degradation if slowapi is not installed.

---

## 3  Feature Gaps Filled

### GAP-01 — No model registry / versioning
**File:** `src/models/registry.py` *(new)*  
`ModelRegistry` with filesystem backend.  Supports `register()`, `promote()`, `load()`, `list_models()`, `list_versions()`.  Metadata stored as JSON under `storage/rl_models/registry/`.

### GAP-02 — Only DeepSeek LLM provider
**File:** `src/llm/provider.py`  
Added `AnthropicProvider` (Claude) implementing `BaseLLMProvider`.  `LLMProviderFactory.create("anthropic")` is now supported.

### GAP-04 — No real-time market data stream
**File:** `src/data/stream.py` *(new)*  
`MarketStream` class with persistent WebSocket connection, subscribe/unsubscribe, auto-reconnect with configurable back-off.

### GAP-05 — No order lifecycle management
**File:** `src/data/brokers/order_manager.py` *(new)*  
`OrderManager` with idempotency keys, JSON journal persistence, SL/TP distance-to-price conversion, and exponential-backoff retry.

### GAP-06 — No data quality validation
**File:** `src/data/fetcher.py`  
Added `MarketDataFetcher.check_quality()` static method.  Checks: missing values, zero prices, >20 % spikes, negative volume, high < low.

### GAP-07 — Health endpoint reports hardcoded `ok`
**File:** `src/api/routes/health.py`  
Rewrote to check Redis, model files, and LLM key presence.  Returns `{ "status": "healthy" | "degraded", "checks": { … } }`.

### GAP-11 — No alerting / notifications
**File:** `src/utils/alerting.py` *(new)*  
`AlertManager` dispatches alerts to Webhook (Slack/Discord) + structured log.  Supports severity levels and cooldown-based deduplication.

### GAP-12 — No circuit breaker for external calls
**File:** `src/utils/circuit_breaker.py` *(new)*  
Async `CircuitBreaker` implementing the CLOSED → OPEN → HALF_OPEN pattern.  Usable as a context manager or decorator.

### GAP-13 — Hardcoded `periods_per_year=252`
**File:** `src/backtesting/metrics.py`  
Added `infer_periods_per_year()` that auto-detects the frequency from a `DatetimeIndex` (1 min → monthly).

---

## 4  Upgrades

### UPGRADE-01+02 — Bump LangChain & core
**File:** `requirements.txt`  
`langchain>=0.1.0` → `>=0.3.0`, `langchain-core` similarly.

### UPGRADE-03 — VecNormalize observation wrapping
**File:** `src/training/trainer.py`  
Added `use_vec_normalize: bool = True` to `TrainingConfig`.  `_create_envs()` wraps the vectorised environment in `VecNormalize`.

### UPGRADE-04 — Raise default training timesteps
**File:** `src/training/trainer.py`  
`total_timesteps`: 100 K → 500 K.  `eval_freq`: 5 000 → 10 000.

### UPGRADE-05 — FinBERT auto-detect
**File:** `src/features/sentiment_analyzer.py`  
Default `model` changed from `"vader"` to `"auto"` — tries `import torch` and uses FinBERT when available.

### UPGRADE-06 — Half-Kelly position sizing guard
**File:** `src/agents/specialized/risk_management_agent.py`  
Variable renamed for clarity; added `if stop_distance > 0 else 0` guard.

### UPGRADE-07 — Async data fetch
**File:** `src/data/fetcher.py`  
Added `afetch()` and `afetch_multiple()` using `asyncio.to_thread`.

### UPGRADE-08 — Cache TTL
**File:** `src/data/fetcher.py`  
Constructor accepts `cache_ttl`.  `_load_from_cache()` checks file age (24 h daily, 1 h sub-daily).

### UPGRADE-09 — `pyproject.toml` build system
**File:** `pyproject.toml`  
Added `[build-system]` table (`requires = ["setuptools>=68.0", "wheel"]`).

### UPGRADE-12 — DeepSeek model configurable
**File:** `src/llm/provider.py`  
Model reads from `DEEPSEEK_MODEL` env var with fallback to `DEFAULT_MODEL`.

---

## 5  Code Quality & Hygiene

| Item | Detail |
|------|--------|
| `.gitignore` updated | Added `trace_out.*`, `storage/news_sentiment/`, `storage/orders/`, `.ipynb_checkpoints/` |
| `.env.example` created | Documents all env vars the system reads |
| New dependencies | `slowapi>=0.1.9`, `typing_extensions>=4.9.0`, `pyarrow>=14.0.0`, `websockets` (optional for `MarketStream`) |

---

## New Files Created

| File | Purpose |
|------|---------|
| `src/models/registry.py` | Model versioning & lifecycle |
| `src/data/stream.py` | Real-time WebSocket market data |
| `src/data/brokers/order_manager.py` | Order lifecycle & journal |
| `src/utils/alerting.py` | Alert dispatch (webhook + log) |
| `src/utils/circuit_breaker.py` | Circuit breaker pattern |
| `.env.example` | Env var documentation |

---

## Files Modified

`state.py` · `workflow.py` · `main.py` · `backtest.py` · `trading.py` ·
`discrete_trading_env.py` · `preprocessor.py` · `rl_trace_wrapper.py` ·
`validators.py` · `prompts.py` · `health.py` · `provider.py` ·
`llm/__init__.py` · `models/__init__.py` · `data/__init__.py` ·
`data/brokers/__init__.py` · `utils/__init__.py` · `fetcher.py` ·
`metrics.py` · `risk_management_agent.py` · `trainer.py` ·
`sentiment_analyzer.py` · `requirements.txt` · `pyproject.toml` ·
`.gitignore`

---

*End of implementation plan.*
