// 2RC + Plett 1-state Hysteresis Equivalent Circuit Model + C1-smooth PCHIP OCV spline
// + Arrhenius temperature scaling (for `ip soc code`).

import { OCV_POINTS } from "./protocol.ts";

export interface IpCellModelParams {
  ocvV: number[];        // 17-point OCV curve in Volts
  qMaxMah: number;
  r0Ohm25C: number;
  r1Ohm25C: number;
  tau1S: number;
  r2Ohm25C: number;
  tau2S: number;
  hystMaxV: number;      // Plett hysteresis amplitude (V)
  hystGamma: number;     // Plett hysteresis decay rate
  arrheniusK: number;    // Thermal resistance slope (K)
  chgEff: number;        // Charging coulomb efficiency (~0.995)
  vFullV: number;
  iTermMa: number;
}

export const M5STICKC_PLUS_2RC_MODEL: IpCellModelParams = {
  ocvV: [
    3.300, 3.625, 3.700, 3.726, 3.750, 3.775, 3.795, 3.816, 3.840,
    3.860, 3.890, 3.940, 3.980, 4.035, 4.095, 4.140, 4.200,
  ],
  qMaxMah: 120.0,
  r0Ohm25C: 0.28,
  r1Ohm25C: 0.10,
  tau1S: 12.0,
  r2Ohm25C: 0.08,
  tau2S: 120.0,
  hystMaxV: 0.018,
  hystGamma: 35.0,
  arrheniusK: 1800.0,
  chgEff: 0.995,
  vFullV: 4.165,
  iTermMa: 10.0,
};

// Monotone Cubic Hermite (Fritsch-Carlson PCHIP) interpolator over uniform 17-point grid.
// Guarantees monotonicity and C1-continuous dOCV/dz (zero EKF Jacobian spikes).
export class PchipOcvSpline {
  private readonly y: number[];
  private readonly m: number[]; // nodal derivatives dV/dz
  private readonly h = 1 / (OCV_POINTS - 1);

  constructor(ocvV: number[]) {
    if (ocvV.length !== OCV_POINTS) {
      throw new Error(`Expected ${OCV_POINTS} OCV points`);
    }
    this.y = ocvV.slice();
    const n = OCV_POINTS;
    const d = new Array<number>(n - 1);
    for (let i = 0; i < n - 1; i++) {
      d[i] = (this.y[i + 1] - this.y[i]) / this.h;
    }
    const m = new Array<number>(n).fill(0);
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for (let i = 1; i < n - 1; i++) {
      if (d[i - 1] <= 0 || d[i] <= 0) {
        m[i] = 0;
      } else {
        // Harmonic mean preserves monotonicity on uniform grid.
        m[i] = (2 * d[i - 1] * d[i]) / (d[i - 1] + d[i]);
      }
    }
    this.m = m;
  }

  eval(zIn: number): { ocvV: number; docvDz: number } {
    const z = Math.max(0, Math.min(1, zIn));
    const u = z / this.h;
    const i = Math.min(OCV_POINTS - 2, Math.floor(u));
    const t = u - i;
    const t2 = t * t;
    const t3 = t2 * t;

    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;

    const y0 = this.y[i];
    const y1 = this.y[i + 1];
    const m0 = this.m[i] * this.h;
    const m1 = this.m[i + 1] * this.h;

    const ocvV = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1;

    // Exact analytic derivative d/dt divided by h
    const dh00 = 6 * t2 - 6 * t;
    const dh10 = 3 * t2 - 4 * t + 1;
    const dh01 = -6 * t2 + 6 * t;
    const dh11 = 3 * t2 - 2 * t;
    const docvDz = Math.max(0.02, (dh00 * y0 + dh10 * m0 + dh01 * y1 + dh11 * m1) / this.h);
    return { ocvV, docvDz };
  }

  socFromOcv(vTarget: number): number {
    if (vTarget <= this.y[0]) return 0;
    if (vTarget >= this.y[OCV_POINTS - 1]) return 1;
    let lo = 0;
    let hi = 1;
    for (let it = 0; it < 20; it++) {
      const mid = 0.5 * (lo + hi);
      if (this.eval(mid).ocvV < vTarget) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  }
}

export function arrheniusScale(tempC: number, arrheniusK: number): number {
  const tk = Math.max(-10, Math.min(60, tempC)) + 273.15;
  return Math.exp(arrheniusK * (1 / tk - 1 / 298.15));
}
