// `ip soc code`: 4-state 2RC + Plett Hysteresis Adaptive Joseph-EKF (AEKF).
// Runs ONLY on the connected remote (browser/Node) for 0 bytes of ESP32 flash/RAM/CPU.
// Features:
//  - 4-state vector x = [z, v1, v2, h]
//  - Diagonal F_k = diag(1, a1, a2, ah)
//  - Full 4x4 Joseph-form covariance update (guaranteed SPD)
//  - Sliding-window Maximum Likelihood Estimation (MLE) adaptive measurement variance R
//  - Mahalanobis outlier gating + hard glitch rejection
//  - Records FilterStepTrace for non-causal RTS backward smoothing

import {
  arrheniusScale,
  M5STICKC_PLUS_2RC_MODEL,
  PchipOcvSpline,
  type IpCellModelParams,
} from "./cellModel.ts";
import { SampleFlags } from "./protocol.ts";

export interface IpSample {
  tsMs: number;
  vV: number;
  iMa: number;
  dqMah?: number;     // Hardware coulomb delta if available
  dtS: number;
  tempC?: number;
  flags?: number;
}

export interface FilterStepTrace {
  tsMs: number;
  dtS: number;
  iMa: number;
  vV: number;
  tempC: number;
  dqMah: number;
  xPred: [number, number, number, number];
  pPred: number[];    // 16-element row-major 4x4
  xFilt: [number, number, number, number];
  pFilt: number[];    // 16-element row-major 4x4
  fDiag: [number, number, number, number];
  innovationV: number;
  rEffV2: number;
  outlierRejected: boolean;
}

export interface IpEstimate {
  socPct: number;
  socSigmaPct: number;
  v1V: number;
  v2V: number;
  hystV: number;
  innovationV: number;
  rAdaptV2: number;
  outlierRejected: boolean;
}

export class IpSocEstimator {
  private model: IpCellModelParams;
  private spline: PchipOcvSpline;
  private x: [number, number, number, number] = [0.5, 0, 0, 0];
  private P: number[] = new Array(16).fill(0);
  private iPrevMa = 0;
  private innovWin: number[] = [];
  private readonly winSize = 24;
  private rBaseV2 = 2.25e-4; // (15 mV)^2

  constructor(model: IpCellModelParams = M5STICKC_PLUS_2RC_MODEL) {
    this.model = { ...model, ocvV: model.ocvV.slice() };
    this.spline = new PchipOcvSpline(this.model.ocvV);
    this.seed(0.5, 0.2);
  }

  setModel(model: IpCellModelParams): void {
    this.model = { ...model, ocvV: model.ocvV.slice() };
    this.spline = new PchipOcvSpline(this.model.ocvV);
  }

  getModel(): IpCellModelParams {
    return { ...this.model, ocvV: this.model.ocvV.slice() };
  }

  seed(soc01: number, sigmaSoc = 0.15): void {
    this.x = [Math.max(0, Math.min(1, soc01)), 0, 0, 0];
    this.P.fill(0);
    this.P[0] = sigmaSoc * sigmaSoc;
    this.P[5] = 4e-4;  // (20 mV)^2
    this.P[10] = 4e-4; // (20 mV)^2
    this.P[15] = 0.25; // hysteresis state in [-1, 1]
    this.innovWin = [];
  }

  seedFromVi(vV: number, iMa: number, tempC = 25): void {
    const kr = arrheniusScale(tempC, this.model.arrheniusK);
    const rTot = (this.model.r0Ohm25C + this.model.r1Ohm25C + this.model.r2Ohm25C) * kr;
    const vOcv = vV - (iMa * 1e-3) * rTot;
    const sigma = Math.abs(iMa) > 10 ? 0.15 : 0.04;
    this.seed(this.spline.socFromOcv(vOcv), sigma);
    this.iPrevMa = iMa;
  }

  step(s: IpSample, recordTrace?: FilterStepTrace[]): IpEstimate {
    const dt = Math.max(0, s.dtS);
    const tempC = s.tempC ?? 25;
    const kr = arrheniusScale(tempC, this.model.arrheniusK);
    const flags = s.flags ?? 0;

    const iNowA = s.iMa * 1e-3;
    const hasDq = s.dqMah !== undefined && Number.isFinite(s.dqMah);
    const dqMah = hasDq ? s.dqMah! : 0.5 * (this.iPrevMa + s.iMa) * (dt / 3600);
    const iAvgA = hasDq && dt > 0 ? (dqMah * 3.6) / dt : 0.5 * (this.iPrevMa + s.iMa) * 1e-3;

    // 1. State Predict (exact closed-form exponential for any dt)
    const eff = dqMah > 0 ? this.model.chgEff : 1.0;
    const dz = (eff * dqMah) / this.model.qMaxMah;
    const zPred = Math.max(0, Math.min(1, this.x[0] + dz));

    const a1 = dt > 0 ? Math.exp(-dt / this.model.tau1S) : 1;
    const a2 = dt > 0 ? Math.exp(-dt / this.model.tau2S) : 1;
    const v1Pred = a1 * this.x[1] + (1 - a1) * (this.model.r1Ohm25C * kr) * iAvgA;
    const v2Pred = a2 * this.x[2] + (1 - a2) * (this.model.r2Ohm25C * kr) * iAvgA;

    const ah = Math.exp(-this.model.hystGamma * Math.abs(dz));
    const sgn = iAvgA > 5e-4 ? 1 : iAvgA < -5e-4 ? -1 : 0;
    const hPred = sgn !== 0 ? ah * this.x[3] + (1 - ah) * sgn : this.x[3];

    const xPred: [number, number, number, number] = [zPred, v1Pred, v2Pred, hPred];
    const fDiag: [number, number, number, number] = [1, a1, a2, sgn !== 0 ? ah : 1];

    // 2. Covariance Predict P^- = F P F^T + Q (exploiting diagonal F!)
    const qDiag = [
      1.5e-8 * dt + 1.5e-4 * dz * dz,
      2.5e-5 * (1 - a1 * a1),
      2.5e-5 * (1 - a2 * a2),
      0.01 * (1 - fDiag[3] * fDiag[3]) + 1e-6 * dt,
    ];
    const pPred = new Array<number>(16);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        pPred[r * 4 + c] = fDiag[r] * this.P[r * 4 + c] * fDiag[c] + (r === c ? qDiag[r] : 0);
      }
    }

    // 3. Measurement Model & Innovation
    const { ocvV, docvDz } = this.spline.eval(zPred);
    const vPred = ocvV + v1Pred + v2Pred + this.model.hystMaxV * hPred + iNowA * (this.model.r0Ohm25C * kr);
    const innov = s.vV - vPred;
    const H = [docvDz, 1, 1, this.model.hystMaxV];

    // HP = P^- H^T (4x1) and hph = H P^- H^T (scalar)
    const ph = [0, 0, 0, 0];
    let hph = 0;
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let c = 0; c < 4; c++) sum += pPred[r * 4 + c] * H[c];
      ph[r] = sum;
      hph += H[r] * sum;
    }

    // Sliding-window MLE adaptive measurement variance R
    const rNom = this.rBaseV2 + 0.025 * iNowA * iNowA;
    let rAdapt = rNom;
    if (this.innovWin.length >= 6) {
      let mse = 0;
      for (const e2 of this.innovWin) mse += e2;
      mse /= this.innovWin.length;
      const rMle = Math.max(rNom * 0.5, mse - hph);
      rAdapt = Math.min(rNom * 8.0, Math.max(rNom * 0.5, 0.7 * rNom + 0.3 * rMle));
    }

    const sNom = hph + rAdapt;
    const nis = (innov * innov) / sNom;

    // Hard glitch gate (> 250 mV) + Mahalanobis soft inflation (> 3.5 sigma)
    let outlierRejected = false;
    let xFilt: [number, number, number, number] = [...xPred];
    let pFilt = pPred.slice();
    let rEff = rAdapt;

    if (Math.abs(innov) > 0.25) {
      outlierRejected = true;
    } else {
      if (nis > 12.25) {
        rEff = rAdapt * (nis / 12.25);
      } else {
        this.innovWin.push(innov * innov);
        if (this.innovWin.length > this.winSize) this.innovWin.shift();
      }
      const sEff = hph + rEff;
      const K = [ph[0] / sEff, ph[1] / sEff, ph[2] / sEff, ph[3] / sEff];

      xFilt = [
        Math.max(0, Math.min(1, xPred[0] + K[0] * innov)),
        Math.max(-0.5, Math.min(0.5, xPred[1] + K[1] * innov)),
        Math.max(-0.5, Math.min(0.5, xPred[2] + K[2] * innov)),
        Math.max(-1, Math.min(1, xPred[3] + K[3] * innov)),
      ];

      // Exact 4x4 Joseph-form update: P = (I - K H) P^- (I - K H)^T + K R K^T
      const A = new Array<number>(16);
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          A[r * 4 + c] = (r === c ? 1 : 0) - K[r] * H[c];
        }
      }
      const AP = new Array<number>(16).fill(0);
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          let sum = 0;
          for (let j = 0; j < 4; j++) sum += A[r * 4 + j] * pPred[j * 4 + c];
          AP[r * 4 + c] = sum;
        }
      }
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          let sum = K[r] * K[c] * rEff;
          for (let j = 0; j < 4; j++) sum += AP[r * 4 + j] * A[c * 4 + j];
          pFilt[r * 4 + c] = sum;
        }
      }
      // Symmetrize & floor diagonals for numerical hygiene
      for (let r = 0; r < 4; r++) {
        for (let c = r + 1; c < 4; c++) {
          const avg = 0.5 * (pFilt[r * 4 + c] + pFilt[c * 4 + r]);
          pFilt[r * 4 + c] = avg;
          pFilt[c * 4 + r] = avg;
        }
      }
      pFilt[0] = Math.max(1e-6, pFilt[0]);
      pFilt[5] = Math.max(1e-7, pFilt[5]);
      pFilt[10] = Math.max(1e-7, pFilt[10]);
      pFilt[15] = Math.max(1e-5, pFilt[15]);
    }

    // 4. CV Charge-Termination Anchor
    const cvDone =
      (flags & SampleFlags.CHARGE_DONE) !== 0 ||
      ((flags & SampleFlags.CHARGING) !== 0 &&
        s.vV >= this.model.vFullV &&
        s.iMa >= 0 &&
        s.iMa <= this.model.iTermMa);
    if (cvDone) {
      xFilt[0] = 1.0;
      xFilt[1] = 0;
      xFilt[2] = 0;
      pFilt[0] = 2e-5;
    }

    this.x = xFilt;
    this.P = pFilt;
    this.iPrevMa = s.iMa;

    if (recordTrace) {
      recordTrace.push({
        tsMs: s.tsMs,
        dtS: dt,
        iMa: s.iMa,
        vV: s.vV,
        tempC,
        dqMah,
        xPred,
        pPred,
        xFilt: [...xFilt],
        pFilt: pFilt.slice(),
        fDiag,
        innovationV: innov,
        rEffV2: rEff,
        outlierRejected,
      });
    }

    return {
      socPct: xFilt[0] * 100,
      socSigmaPct: Math.sqrt(pFilt[0]) * 100,
      v1V: xFilt[1],
      v2V: xFilt[2],
      hystV: this.model.hystMaxV * xFilt[3],
      innovationV: innov,
      rAdaptV2: rEff,
      outlierRejected,
    };
  }
}
