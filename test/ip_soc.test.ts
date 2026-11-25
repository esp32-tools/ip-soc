import test from "node:test";
import assert from "node:assert/strict";

import {
  arrheniusScale,
  decodeCellParams,
  decodeDtLog,
  decodeIpFrame,
  encodeCellParams,
  encodeDtLog,
  encodeIpFrame,
  encodeTempHalfC,
  EventType,
  identifyCellForOdSoc,
  IpSocEstimator,
  IpSocHistory,
  IpSocSyncController,
  M5STICKC_PLUS_2RC_MODEL,
  M5STICKC_PLUS_DEFAULT_PARAMS,
  MemoryHistoryStore,
  packOdEvent,
  PchipOcvSpline,
  runRtsSmoother,
  SampleFlags,
  unpackOdEvent,
  type FilterStepTrace,
  type IpCellModelParams,
} from "../src/index.ts";

// Synthetic 2RC + Plett hysteresis truth plant (matches od-soc/sim/plant.hpp)
class TruthPlant2RC {
  private z: number;
  private v1 = 0;
  private v2 = 0;
  private h = 0;
  private readonly m: IpCellModelParams;
  private readonly spline: PchipOcvSpline;

  constructor(model: IpCellModelParams, initialSoc: number) {
    this.m = model;
    this.z = initialSoc;
    this.spline = new PchipOcvSpline(model.ocvV);
  }

  step(iMa: number, dtS: number, tempC = 25): { vV: number; dqMah: number; trueSocPct: number } {
    const iA = iMa * 1e-3;
    const dqMah = iMa * (dtS / 3600);
    const eff = dqMah > 0 ? this.m.chgEff : 1;
    this.z = Math.max(0, Math.min(1, this.z + (eff * dqMah) / this.m.qMaxMah));

    const kr = arrheniusScale(tempC, this.m.arrheniusK);
    const a1 = Math.exp(-dtS / this.m.tau1S);
    const a2 = Math.exp(-dtS / this.m.tau2S);
    this.v1 = a1 * this.v1 + (1 - a1) * (this.m.r1Ohm25C * kr) * iA;
    this.v2 = a2 * this.v2 + (1 - a2) * (this.m.r2Ohm25C * kr) * iA;

    const ah = Math.exp(-this.m.hystGamma * (Math.abs(dqMah) / this.m.qMaxMah));
    const sgn = iMa > 0.5 ? 1 : iMa < -0.5 ? -1 : 0;
    if (sgn !== 0) this.h = ah * this.h + (1 - ah) * sgn;

    const { ocvV } = this.spline.eval(this.z);
    const vV = ocvV + this.v1 + this.v2 + this.m.hystMaxV * this.h + iA * (this.m.r0Ohm25C * kr);
    return { vV, dqMah, trueSocPct: this.z * 100 };
  }
}

test("protocol: 52-byte CellParams, 8-byte OdEvent, and SOC1 frame roundtrip", () => {
  const blob = encodeCellParams(M5STICKC_PLUS_DEFAULT_PARAMS);
  assert.equal(blob.length, 52);
  const decoded = decodeCellParams(blob);
  assert.ok(decoded);
  assert.equal(decoded.qMaxDmah, 1200);
  assert.equal(decoded.r0Mohm, 280);
  assert.deepEqual(decoded.ocvMv, M5STICKC_PLUS_DEFAULT_PARAMS.ocvMv);

  // Flip a bit -> CRC16 must reject
  const bad = blob.slice();
  bad[40] ^= 1;
  assert.equal(decodeCellParams(bad), null);

  // Log-dt bounds
  assert.equal(decodeDtLog(encodeDtLog(59)), 59);
  assert.equal(decodeDtLog(encodeDtLog(300)), 300);
  assert.equal(decodeDtLog(encodeDtLog(3600)), 3600);
  assert.ok(decodeDtLog(encodeDtLog(1_500_000)) >= 1_499_000);

  // 8-byte OdEvent exact roundtrip (matches od-soc C++ test vector)
  const raw = packOdEvent({
    dtS: 300,
    type: EventType.CHG_DONE,
    vMv: 4195,
    iMa: -123,
    dqQmah: -205,
    socHalfPct: 167,
    tempHalfC: encodeTempHalfC(34.5),
  });
  const ev = unpackOdEvent(raw);
  assert.equal(ev.dtS, 300);
  assert.equal(ev.type, EventType.CHG_DONE);
  assert.equal(ev.vMv, 4195);
  assert.equal(ev.iMa, -123);
  assert.equal(ev.dqQmah, -205);
  assert.equal(ev.socHalfPct, 167);

  // SOC1 frame encode/decode + CRC protection
  const frame = encodeIpFrame(42, 0x01, [
    {
      tsMs: 10000,
      vMv: 3850,
      iMa: -95,
      dqUah: -132,
      tempDc: 295,
      eventType: EventType.PERIODIC,
      sampleFlags: SampleFlags.HAS_DQ,
      odSocHalfPct: 102,
    },
  ]);
  assert.equal(frame.length, 12 + 16 + 2);
  const parsed = decodeIpFrame(frame);
  assert.ok(parsed);
  assert.equal(parsed.seq, 42);
  assert.equal(parsed.records[0].vMv, 3850);
  assert.equal(parsed.records[0].dqUah, -132);
  frame[15] ^= 0xff;
  assert.equal(decodeIpFrame(frame), null);
});

test("ip soc code: 4-state AEKF + RTS smoother achieves < 0.7% RMSE and outperforms causal filter", () => {
  const plant = new TruthPlant2RC(M5STICKC_PLUS_2RC_MODEL, 0.88);
  const est = new IpSocEstimator(M5STICKC_PLUS_2RC_MODEL);
  // Deliberately start 20% wrong to verify fast AEKF convergence + RTS backward correction!
  est.seed(0.68, 0.20);

  const trace: FilterStepTrace[] = [];
  const truth: number[] = [];
  const filtErr2: number[] = [];

  for (let k = 0; k < 360; k++) {
    let iMa = -95;
    if (k % 12 === 0) iMa = -240;
    else if (k % 20 >= 16) iMa = -10;
    const out = plant.step(iMa, 5, 30);
    // Inject a single -400 mV glitch at k = 150
    const vMeas = k === 150 ? out.vV - 0.4 : out.vV;
    const e = est.step(
      {
        tsMs: (k + 1) * 5000,
        vV: vMeas,
        iMa,
        dqMah: out.dqMah,
        dtS: 5,
        tempC: 30,
        flags: SampleFlags.HAS_DQ,
      },
      trace,
    );
    if (k === 150) assert.equal(e.outlierRejected, true);
    truth.push(out.trueSocPct);
    filtErr2.push((e.socPct - out.trueSocPct) ** 2);
  }

  const smoothed = runRtsSmoother(trace);
  const steadyFiltRmse = Math.sqrt(
    filtErr2.slice(40).reduce((a, b) => a + b, 0) / (filtErr2.length - 40),
  );
  const fullFiltRmse = Math.sqrt(filtErr2.reduce((a, b) => a + b, 0) / filtErr2.length);
  const fullRtsRmse = Math.sqrt(
    smoothed.reduce((acc, s, i) => acc + (s.socPct - truth[i]) ** 2, 0) / smoothed.length,
  );

  console.log(
    `  [bench] ip-soc 4-state AEKF steady RMSE = ${steadyFiltRmse.toFixed(3)}% | full causal = ${fullFiltRmse.toFixed(2)}% -> RTS smoother = ${fullRtsRmse.toFixed(2)}%`,
  );
  assert.ok(steadyFiltRmse < 0.7, `steady RMSE too high: ${steadyFiltRmse}`);
  assert.ok(fullRtsRmse < fullFiltRmse, "RTS smoother should beat causal filter over full trajectory");
});

test("CellIdentifier + IpSocSyncController recovers aged R0 and SOH -> pushes 52-byte CellParams", async () => {
  // Real physical cell has aged: Q_max = 92 mAh (76.7% SOH), R0 = 0.48 Ohm (1.7x nominal)!
  const agedTruthModel: IpCellModelParams = {
    ...M5STICKC_PLUS_2RC_MODEL,
    qMaxMah: 92.0,
    r0Ohm25C: 0.48,
  };
  const plant = new TruthPlant2RC(agedTruthModel, 0.90);

  const store = new MemoryHistoryStore();
  const ctrl = new IpSocSyncController(M5STICKC_PLUS_2RC_MODEL, 120.0);

  // Build 1) an offline gap from od-soc 8-byte ring, plus 2) live SOC1 discharge stream
  const ringBuf = new Uint8Array(16);
  const dv = new DataView(ringBuf.buffer);
  dv.setBigUint64(
    0,
    packOdEvent({
      dtS: 0,
      type: EventType.BOOT,
      vMv: 4100,
      iMa: -5,
      dqQmah: 0,
      socHalfPct: 180,
      tempHalfC: encodeTempHalfC(25),
    }),
    true,
  );
  dv.setBigUint64(
    8,
    packOdEvent({
      dtS: 600,
      type: EventType.WAKE,
      vMv: 4095,
      iMa: -5,
      dqQmah: -2,
      socHalfPct: 179,
      tempHalfC: encodeTempHalfC(25),
    }),
    true,
  );
  const merged = ctrl.history.mergeOdHistoryRing(ringBuf, 600_000);
  assert.equal(merged, 2);

  // Stream 240 live 5-second steps (~20 min, discharging ~45% SOC with load steps for R0 observability)
  const records = [];
  for (let k = 0; k < 240; k++) {
    const iMa = k % 10 < 5 ? -150 : -50; // 100 mA current steps excite R0 cleanly!
    const out = plant.step(iMa, 5, 28);
    records.push({
      tsMs: 605_000 + k * 5000,
      vMv: Math.round(out.vV * 1000),
      iMa,
      dqUah: Math.round(out.dqMah * 1000),
      tempDc: 280,
      eventType: EventType.PERIODIC,
      sampleFlags: SampleFlags.HAS_DQ,
      odSocHalfPct: Math.round(out.trueSocPct * 2),
    });
  }
  const frame = encodeIpFrame(1, 0, records);
  const ing = ctrl.history.ingestIpFrame(frame, "ip_live");
  assert.ok(ing && ing.added === 240);

  await ctrl.history.saveTo(store, "balax-01");
  const reloaded = await store.load("balax-01");
  assert.equal(reloaded.length, 242);

  const analysis = ctrl.analyzeAndRecommend(M5STICKC_PLUS_DEFAULT_PARAMS);
  assert.ok(analysis.identification);
  console.log(
    `  [bench] Aged cell ID (truth R0=0.480, Q=92.0): est R0=${analysis.identification.r0Ohm25C.toFixed(3)} Ohm, Q=${analysis.identification.qMaxMah.toFixed(1)} mAh (SOH=${analysis.identification.sohPct.toFixed(1)}%, conf=${analysis.identification.confidence.toFixed(2)})`,
  );
  assert.ok(
    Math.abs(analysis.identification.r0Ohm25C - 0.48) < 0.05,
    `R0 estimate off: ${analysis.identification.r0Ohm25C}`,
  );
  assert.ok(
    Math.abs(analysis.identification.qMaxMah - 92.0) < 8.0,
    `Q_max estimate off: ${analysis.identification.qMaxMah}`,
  );
  assert.equal(analysis.shouldPushToDevice, true);
  assert.ok(analysis.paramsBlobToPush && analysis.paramsBlobToPush.length === 52);
  assert.ok(decodeCellParams(analysis.paramsBlobToPush) !== null);
});
