# 🚀 Samvaad Voice RAG Latency Benchmark Report

- **Execution Timestamp (UTC):** `2026-08-22T19:14:33.983295+00:00`
- **Total Benchmark Queries:** `25`
- **Platform:** `Windows-11-10.0.26200-SP0` (Intel64 Family 6 Model 140 Stepping 2, GenuineIntel)
- **SLA Target:** `< 200ms`
- **Benchmark Status:** ❌ **EXCEEDED BUDGET** (`P50 = 3235.99ms`)

## 📊 Per-Stage Latency Percentiles (Milliseconds)

| Pipeline Stage | P50 (Median) | P70 | P90 | P95 | P100 (Max) | Mean | Target Budget |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Guardrail Safety Check | `0.17` | `0.18` | `0.31` | `0.44` | `0.49` | `0.20` | < 5ms |
| 2. Dense Query Embedding | `44.03` | `47.24` | `158.92` | `257.40` | `380.81` | `74.91` | < 15ms |
| 3. FAISS Vector Search | `0.80` | `0.94` | `1.44` | `2.45` | `2.84` | `1.01` | < 5ms |
| 4. Fast Hybrid Reranker | `1.05` | `1.22` | `2.01` | `2.59` | `25.47` | `2.15` | < 5ms |
| 📌 Total Retrieval Time | `46.84` | `54.55` | `160.73` | `259.86` | `384.81` | `78.30` | < 30ms |
| 5. LLM Inference (TTFT) | `811.19` | `6781.17` | `14588.38` | `17035.48` | `21226.31` | `5091.23` | < 80ms |
| 6. Grounding Verification | `0.19` | `0.26` | `3.35` | `5.87` | `7.04` | `1.08` | < 10ms |
| 7. Voice TTS Synthesis | `1767.90` | `1892.84` | `2257.14` | `2804.37` | `2945.71` | `1849.60` | < 50ms |
| ⚡ **END-TO-END PIPELINE** | `3235.99` | `8659.08` | `16454.30` | `18773.80` | `23131.94` | `7020.46` | **< 200ms** |

## ⚙️ Measured Components

| Stage | Implementation |
| :--- | :--- |
| embedder | `HuggingFaceEmbedder` |
| llm | `ModelOrchestrationHarness` |
| tts | `SarvamTTS` |

---
### 🎯 Findings

1. **End-to-end P50 is `3235.99ms`** against a `200ms` target (over budget). P95 is `18773.80ms`, P100 `23131.94ms`.
2. **Local compute is not the bottleneck.** Retrieval P50 is `46.84ms` (embedding + FAISS + rerank).
3. **Third-party API calls dominate.** LLM P50 `811.19ms` + TTS P50 `1767.90ms` = `2579.09ms`, about 80% of end-to-end latency. These are network-bound and not reducible by local optimization.
4. **Tail latency is driven by LLM generation variance.** LLM P100 `21226.31ms` is 26x its P50 of `811.19ms`. On a reasoning-class model most completion tokens are spent deliberating rather than on the answer text, so a short reply can still be slow and the spread between a fast and a slow response is large.
5. **The `200ms` target is not reachable while the LLM and TTS are remote API calls.** Retrieval, guardrail, and grounding together are `47.20ms` at P50, so the budget is only meaningful as a local-compute target. Meeting it end to end would require on-device or streamed generation.
