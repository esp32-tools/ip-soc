# esp32-tools/ip-soc

Remote / IP-connected **SOTA Battery State of Charge & System-ID** library (`ip soc code` + `ip soc history`), written in zero-dependency TypeScript ESM (runs identically in browsers / IndexedDB and Node.js with **0 bytes** of ESP32 flash, RAM, or CPU).

Designed as the remote half of a complementary **2×2 architecture** paired with **[`esp32-tools/od-soc`](https://github.com/esp32-tools/od-soc)**:

| | **On-Device (`esp32-tools/od-soc`)** | **Remote / IP (`esp32-tools/ip-soc`)** |
| :--- | :--- | :--- |
| **Algorithm** | **`od soc code`** — **455 B** Xtensa code flash, **16 B** RTC RAM, **0 FPU**; always on; **sole writer** to on-device flash | **`ip soc code`** — 4-state 2RC + Plett hysteresis AEKF (MLE adaptive $R$, Joseph SPD) + non-causal RTS smoother + cell-ID; runs **only when connected**; pushes calibrated 52 B `CellParams` to `od soc code` |
| **History** | **`od soc history`** — Always-on hypercompressed **8 B/event** ring (RTC → 4–8 KB flash) | **`ip soc history`** — High-res `SOC1` stream + spillover drain + 8 B offline gap-fill stored on remote SSD / IndexedDB |

## Modules

- `src/protocol.ts` — Bit-exact codecs shared with `od-soc`: 52-byte `CellParams` (CRC-16/CCITT-FALSE), 8-byte `OdEvent` (log-dt 1 s..18.7 days), and `SOC1` stream frames.
- `src/cellModel.ts` — 2RC + Plett 1-state hysteresis ECM, Fritsch–Carlson monotone PCHIP OCV spline ($C^1$ analytic $d    text{OCV}/dz$), Arrhenius thermal resistance scaling.
- `src/ipEstimator.ts` — **`ip soc code`**: 4-state `[SOC, V1, V2, h]` Joseph-form Adaptive EKF with sliding-window MLE $R$ adaptation and Mahalanobis/glitch gating.
- `src/rtsSmoother.ts` — Non-causal Rauch–Tung–Striebel backward-forward smoother exploiting diagonal $F_k$.
- `src/cellIdentifier.ts` — Excitation-gated $R_0$ regression, RTS-endpoint SOH ($Q_{\max}$) identification, and 2RC → 1RC projection emitting signed 52-byte `CellParams`.
- `src/ipHistory.ts` — **`ip soc history`**: unified deduplicated timeline (`MemoryHistoryStore` & browser `IndexedDbHistoryStore`) merging live `SOC1`, drained spillover, and 8-byte `od soc history` gap-fills.
- `src/syncController.ts` — High-level `IpSocSyncController` deciding when calibrated parameters improved enough to inform `od-soc`.

## Run Tests (Zero Dependencies, Node ≥ 22.6)

```bash
npm test
```
