// Non-causal Rauch-Tung-Striebel (RTS) backward-forward smoother over `ip soc history`.
// Exploits diagonal F_k to refine historical SOC trajectories beyond any causal filter,
// producing ultra-clean endpoints for CellIdentifier SOH / R0 / 1RC fitting.

import type { FilterStepTrace } from "./ipEstimator.ts";

export interface SmoothedPoint {
  tsMs: number;
  socPct: number;
  socSigmaPct: number;
  v1V: number;
  v2V: number;
  h: number;
}

// Gauss-Jordan 4x4 matrix inverse (with small ridge regularization for safety).
function invert4x4(m: number[]): number[] | null {
  const a = new Array<number>(32).fill(0);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      a[r * 8 + c] = m[r * 4 + c] + (r === c ? 1e-10 : 0);
      a[r * 8 + 4 + c] = r === c ? 1 : 0;
    }
  }
  for (let col = 0; col < 4; col++) {
    let pivot = col;
    for (let r = col + 1; r < 4; r++) {
      if (Math.abs(a[r * 8 + col]) > Math.abs(a[pivot * 8 + col])) pivot = r;
    }
    if (Math.abs(a[pivot * 8 + col]) < 1e-14) return null;
    if (pivot !== col) {
      for (let c = 0; c < 8; c++) {
        const tmp = a[col * 8 + c];
        a[col * 8 + c] = a[pivot * 8 + c];
        a[pivot * 8 + c] = tmp;
      }
    }
    const div = a[col * 8 + col];
    for (let c = 0; c < 8; c++) a[col * 8 + c] /= div;
    for (let r = 0; r < 4; r++) {
      if (r === col) continue;
      const f = a[r * 8 + col];
      for (let c = 0; c < 8; c++) a[r * 8 + c] -= f * a[col * 8 + c];
    }
  }
  const inv = new Array<number>(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) inv[r * 4 + c] = a[r * 8 + 4 + c];
  }
  return inv;
}

export function runRtsSmoother(trace: FilterStepTrace[]): SmoothedPoint[] {
  const n = trace.length;
  if (n === 0) return [];

  const xs = trace.map((t) => [...t.xFilt] as [number, number, number, number]);
  const Ps = trace.map((t) => t.pFilt.slice());

  for (let k = n - 2; k >= 0; k--) {
    const next = trace[k + 1];
    const pPredInv = invert4x4(next.pPred);
    if (!pPredInv) continue;

    // G_k = P_filt[k] * F_{k+1}^T * (P_pred[k+1])^{-1}  (F_{k+1} is diagonal!)
    const PF = new Array<number>(16);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        PF[r * 4 + c] = trace[k].pFilt[r * 4 + c] * next.fDiag[c];
      }
    }
    const G = new Array<number>(16).fill(0);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        let sum = 0;
        for (let j = 0; j < 4; j++) sum += PF[r * 4 + j] * pPredInv[j * 4 + c];
        G[r * 4 + c] = sum;
      }
    }

    // x_s[k] = x_filt[k] + G_k * (x_s[k+1] - x_pred[k+1])
    const dx = [
      xs[k + 1][0] - next.xPred[0],
      xs[k + 1][1] - next.xPred[1],
      xs[k + 1][2] - next.xPred[2],
      xs[k + 1][3] - next.xPred[3],
    ];
    for (let r = 0; r < 4; r++) {
      let corr = 0;
      for (let c = 0; c < 4; c++) corr += G[r * 4 + c] * dx[c];
      xs[k][r] = trace[k].xFilt[r] + corr;
    }
    xs[k][0] = Math.max(0, Math.min(1, xs[k][0]));

    // P_s[k] = P_filt[k] + G_k * (P_s[k+1] - P_pred[k+1]) * G_k^T
    const dP = new Array<number>(16);
    for (let i = 0; i < 16; i++) dP[i] = Ps[k + 1][i] - next.pPred[i];
    const GdP = new Array<number>(16).fill(0);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        let sum = 0;
        for (let j = 0; j < 4; j++) sum += G[r * 4 + j] * dP[j * 4 + c];
        GdP[r * 4 + c] = sum;
      }
    }
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        let sum = trace[k].pFilt[r * 4 + c];
        for (let j = 0; j < 4; j++) sum += GdP[r * 4 + j] * G[c * 4 + j];
        Ps[k][r * 4 + c] = sum;
      }
    }
    Ps[k][0] = Math.max(1e-6, Ps[k][0]);
  }

  return trace.map((t, i) => ({
    tsMs: t.tsMs,
    socPct: xs[i][0] * 100,
    socSigmaPct: Math.sqrt(Math.max(1e-6, Ps[i][0])) * 100,
    v1V: xs[i][1],
    v2V: xs[i][2],
    h: xs[i][3],
  }));
}
