// High-level remote sync controller bridging `ip soc code` + `ip soc history` -> `od soc code`.
// Runs ONLY when remote is connected; informs `od soc code` with 52-byte CellParams blobs
// only when calibration materially improves (minimizing NVS flash writes on the robot).

import { identifyCellForOdSoc, type IdentificationResult } from "./cellIdentifier.ts";
import { M5STICKC_PLUS_2RC_MODEL, type IpCellModelParams } from "./cellModel.ts";
import { IpSocEstimator, type FilterStepTrace, type IpEstimate } from "./ipEstimator.ts";
import { IpSocHistory } from "./ipHistory.ts";
import { ParamFlags, type CellParams } from "./protocol.ts";
import { runRtsSmoother, type SmoothedPoint } from "./rtsSmoother.ts";

export interface SyncAnalysis {
  latestEstimate: IpEstimate | null;
  smoothed: SmoothedPoint[];
  identification: IdentificationResult | null;
  shouldPushToDevice: boolean;
  paramsBlobToPush: Uint8Array | null;
}

export class IpSocSyncController {
  readonly history: IpSocHistory;
  private model: IpCellModelParams;
  private readonly nominalQMaxMah: number;
  private lastPushed: CellParams | null = null;

  constructor(
    model: IpCellModelParams = M5STICKC_PLUS_2RC_MODEL,
    nominalQMaxMah = model.qMaxMah,
    history = new IpSocHistory(),
  ) {
    this.model = { ...model, ocvV: model.ocvV.slice() };
    this.nominalQMaxMah = nominalQMaxMah;
    this.history = history;
  }

  // Replays `ip soc history` through 4-state AEKF + RTS smoother + CellIdentifier,
  // and decides whether `od soc code` should be informed with updated 52-byte CellParams.
  analyzeAndRecommend(currentDeviceParams?: CellParams): SyncAnalysis {
    const timeline = this.history.getTimeline();
    if (timeline.length === 0) {
      return {
        latestEstimate: null,
        smoothed: [],
        identification: null,
        shouldPushToDevice: false,
        paramsBlobToPush: null,
      };
    }

    const est = new IpSocEstimator(this.model);
    const first = timeline[0];
    if (first.odSocPct !== null) {
      est.seed(first.odSocPct * 0.01, 0.12);
    } else {
      est.seedFromVi(first.vV, first.iMa, first.tempC);
    }

    const trace: FilterStepTrace[] = [];
    let latest: IpEstimate | null = null;
    for (let i = 0; i < timeline.length; i++) {
      const cur = timeline[i];
      const dtS = i === 0 ? 0 : Math.max(0, (cur.tsMs - timeline[i - 1].tsMs) * 1e-3);
      latest = est.step(
        {
          tsMs: cur.tsMs,
          vV: cur.vV,
          iMa: cur.iMa,
          dqMah: cur.dqMah,
          dtS,
          tempC: cur.tempC,
          flags: cur.sampleFlags,
        },
        trace,
      );
    }

    const smoothed = runRtsSmoother(trace);
    const id = identifyCellForOdSoc(trace, smoothed, this.model, this.nominalQMaxMah);

    const ref = currentDeviceParams ?? this.lastPushed;
    let shouldPush = false;
    if (id.confidence >= 0.4) {
      if (!ref || (ref.flags & ParamFlags.CALIBRATED) === 0) {
        shouldPush = true;
      } else {
        const qRel = Math.abs(id.params.qMaxDmah - ref.qMaxDmah) / Math.max(1, ref.qMaxDmah);
        const rRel = Math.abs(id.params.r0Mohm - ref.r0Mohm) / Math.max(1, ref.r0Mohm);
        shouldPush = qRel >= 0.025 || rRel >= 0.08;
      }
    }

    if (shouldPush) {
      this.lastPushed = id.params;
      this.model = {
        ...this.model,
        qMaxMah: id.qMaxMah,
        r0Ohm25C: id.r0Ohm25C,
      };
    }

    return {
      latestEstimate: latest,
      smoothed,
      identification: id,
      shouldPushToDevice: shouldPush,
      paramsBlobToPush: shouldPush ? id.paramsBlob : null,
    };
  }
}
