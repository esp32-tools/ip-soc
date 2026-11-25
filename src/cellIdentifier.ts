// System Identification (`ip soc code` -> 52-byte `CellParams` for `od soc code`):
// Fits aged R0 (25 C), usable Q_max (SOH), and optimal 1RC projection (r1, tau) from
// `ip soc history` + RTS-smoothed trajectories, for 0 MCU cost.

import { arrheniusScale, PchipOcvSpline, type IpCellModelParams } from "./cellModel.ts";
import type { FilterStepTrace } from "./ipEstimator.ts";
import {
  encodeCellParams,
  ParamFlags,
  type CellParams,
} from "./protocol.ts";
import type { SmoothedPoint } from "./rtsSmoother.ts";

export interface IdentificationResult {
  r0Ohm25C: number;
  qMaxMah: number;
  sohPct: number;
  r0Samples: number;
  qMaxArcs: number;
  confidence: number;       // 0..1
  params: CellParams;
  paramsBlob: Uint8Array;   // Ready to push over IP -> od-soc
}

function median(values: number[]): number {
  const s = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? 0.5 * (s[mid - 1] + s[mid]) : s[mid];
}

export function identifyCellForOdSoc(
  trace: FilterStepTrace[],
  smoothed: SmoothedPoint[],
  baseModel: IpCellModelParams,
  nominalQMaxMah: number,
): IdentificationResult {
  const spline = new PchipOcvSpline(baseModel.ocvV);

  // 1. Excitation-gated ohmic resistance R0 (25 C) from current steps |dI| >= 30 mA
  const r0Obs: number[] = [];
  for (let k = 1; k < trace.length; k++) {
    const prev = trace[k - 1];
    const curr = trace[k];
    if (curr.outlierRejected || prev.outlierRejected) continue;
    if (curr.dtS <= 0 || curr.dtS > 15) continue;
    const zMid = 0.5 * (smoothed[k - 1].socPct + smoothed[k].socPct) * 0.01;
    if (zMid < 0.15 || zMid > 0.90) continue;

    const diA = (curr.iMa - prev.iMa) * 1e-3;
    if (Math.abs(diA) < 0.03) continue;

    const dOcv =
      spline.eval(smoothed[k].socPct * 0.01).ocvV -
      spline.eval(smoothed[k - 1].socPct * 0.01).ocvV;
    const dRc =
      (smoothed[k].v1V + smoothed[k].v2V) - (smoothed[k - 1].v1V + smoothed[k - 1].v2V);
    const dvOhmic = (curr.vV - prev.vV) - dOcv - dRc;
    const kr = arrheniusScale(curr.tempC, baseModel.arrheniusK);
    const r0 = dvOhmic / (diA * kr);
    if (r0 > baseModel.r0Ohm25C * 0.35 && r0 < baseModel.r0Ohm25C * 4.5) {
      r0Obs.push(r0);
    }
  }

  const r0Est = r0Obs.length >= 4 ? median(r0Obs) : baseModel.r0Ohm25C;

  // 2. Non-circular usable capacity Q_max (SOH) via OCV-implied SOC vs. cumulative Coulomb
  //    regression (using newly identified r0Est + smoothed RC/hysteresis states, zero dependence
  //    on baseModel.qMaxMah!).
  const zPts: number[] = [];
  const qPts: number[] = [];
  let qCum = 0;
  // Open-loop RC + hysteresis driven purely by measured current (cannot absorb Q_max error!).
  let v1Open = 0;
  let v2Open = 0;
  let hOpen = 0;
  for (let k = 0; k < trace.length; k++) {
    const t = trace[k];
    const dq = t.dqMah;
    qCum += dq > 0 ? dq * baseModel.chgEff : dq;
    const kr = arrheniusScale(t.tempC, baseModel.arrheniusK);
    const iA = t.iMa * 1e-3;
    const dt = Math.max(0, t.dtS);
    const a1 = dt > 0 ? Math.exp(-dt / baseModel.tau1S) : 1;
    const a2 = dt > 0 ? Math.exp(-dt / baseModel.tau2S) : 1;
    v1Open = a1 * v1Open + (1 - a1) * (baseModel.r1Ohm25C * kr) * iA;
    v2Open = a2 * v2Open + (1 - a2) * (baseModel.r2Ohm25C * kr) * iA;
    const ah = Math.exp(-baseModel.hystGamma * (Math.abs(dq) / nominalQMaxMah));
    const sgn = t.iMa > 0.5 ? 1 : t.iMa < -0.5 ? -1 : 0;
    if (sgn !== 0) hOpen = ah * hOpen + (1 - ah) * sgn;

    if (t.outlierRejected || k === 0) continue;
    const vOcvEst =
      t.vV -
      iA * (r0Est * kr) -
      v1Open -
      v2Open -
      baseModel.hystMaxV * hOpen;
    const zOcv = spline.socFromOcv(vOcvEst);
    if (zOcv >= 0.12 && zOcv <= 0.92) {
      zPts.push(zOcv);
      qPts.push(qCum);
    }
  }

  let qMaxEst = baseModel.qMaxMah;
  let qMaxArcs = 0;
  if (zPts.length >= 20) {
    let zMin = zPts[0];
    let zMax = zPts[0];
    let zMean = 0;
    let qMean = 0;
    for (let i = 0; i < zPts.length; i++) {
      if (zPts[i] < zMin) zMin = zPts[i];
      if (zPts[i] > zMax) zMax = zPts[i];
      zMean += zPts[i];
      qMean += qPts[i];
    }
    zMean /= zPts.length;
    qMean /= qPts.length;
    if (zMax - zMin >= 0.20) {
      let cov = 0;
      let varZ = 0;
      for (let i = 0; i < zPts.length; i++) {
        const dz = zPts[i] - zMean;
        cov += dz * (qPts[i] - qMean);
        varZ += dz * dz;
      }
      const qReg = cov / Math.max(1e-9, varZ);
      if (qReg >= nominalQMaxMah * 0.45 && qReg <= nominalQMaxMah * 1.15) {
        qMaxEst = qReg;
        qMaxArcs = Math.max(1, Math.floor((zMax - zMin) / 0.25));
      }
    }
  }

  // 3. Project 2RC parameters cleanly onto od-soc's 1RC model (energy-matched r1 & tau).
  const r1Proj = baseModel.r1Ohm25C + baseModel.r2Ohm25C * 0.75;
  const tauProj =
    (baseModel.r1Ohm25C * baseModel.tau1S + baseModel.r2Ohm25C * baseModel.tau2S * 0.5) /
    Math.max(1e-3, r1Proj);

  const r0Conf = Math.min(1, r0Obs.length / 12);
  const qConf = Math.min(1, qMaxArcs / 2);
  const confidence = 0.5 * r0Conf + 0.5 * qConf;

  const params: CellParams = {
    flags: ParamFlags.TEMP_COMP | (confidence >= 0.4 ? ParamFlags.CALIBRATED : 0),
    ocvMv: baseModel.ocvV.map((v) => Math.round(v * 1000)),
    qMaxDmah: Math.max(10, Math.round(qMaxEst * 10)),
    r0Mohm: Math.max(10, Math.round(r0Est * 1000)),
    r1Mohm: Math.max(5, Math.round(r1Proj * 1000)),
    tauS: Math.max(5, Math.round(tauProj)),
    vFullMv: Math.round(baseModel.vFullV * 1000),
    iTermMa: Math.round(baseModel.iTermMa),
    chgEffSub900: Math.max(0, Math.min(100, Math.round((baseModel.chgEff - 0.9) * 1000))),
  };

  return {
    r0Ohm25C: r0Est,
    qMaxMah: qMaxEst,
    sohPct: (qMaxEst / nominalQMaxMah) * 100,
    r0Samples: r0Obs.length,
    qMaxArcs,
    confidence,
    params,
    paramsBlob: encodeCellParams(params),
  };
}
