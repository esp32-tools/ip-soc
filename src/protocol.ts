// Bit-exact binary wire contracts shared with esp32-tools/od-soc:
//  - CellParams (52 bytes LE, CRC-16/CCITT-FALSE): pushed by ip-soc -> od-soc
//  - OdEvent    ( 8 bytes LE, 64-bit packed):      always-on `od soc history` ring
//  - SOC1 Frame (12 B header + N*16 B + 2 B CRC):  high-res `ip soc history` stream / spillover

export const OCV_POINTS = 17;
export const PARAMS_BYTES = 52;
export const PARAMS_MAGIC_0 = 0x4f; // 'O'
export const PARAMS_MAGIC_1 = 0x53; // 'S'
export const PARAMS_VERSION = 1;

export const ParamFlags = {
  CALIBRATED: 1 << 0,
  TEMP_COMP: 1 << 1,
} as const;

export const SampleFlags = {
  HAS_DQ: 1 << 0,
  CHARGE_DONE: 1 << 1,
  CHARGING: 1 << 2,
} as const;

export const EventType = {
  PERIODIC: 0,
  BOOT: 1,
  SHUTDOWN: 2,
  USB_IN: 3,
  USB_OUT: 4,
  CHG_START: 5,
  CHG_DONE: 6,
  SLEEP_WIFI: 7,
  SLEEP_USB: 8,
  WAKE: 9,
  ARM: 10,
  DISARM: 11,
  LOW_BAT: 12,
  CAL_APPLIED: 13,
  THERMAL: 14,
  FAULT: 15,
} as const;
export type EventTypeValue = (typeof EventType)[keyof typeof EventType];

export interface CellParams {
  flags: number;
  ocvMv: number[];        // 17 points: 0%, 6.25%, ..., 100%
  qMaxDmah: number;       // 0.1 mAh units (1200 = 120.0 mAh)
  r0Mohm: number;         // mOhm @ 25 C
  r1Mohm: number;         // mOhm @ 25 C
  tauS: number;           // seconds
  vFullMv: number;        // mV
  iTermMa: number;        // mA
  chgEffSub900: number;   // eff = (900 + val) / 1000
}

export interface UnpackedOdEvent {
  dtS: number;
  type: EventTypeValue;
  vMv: number;
  iMa: number;
  dqQmah: number;       // 0.25 mAh units
  socHalfPct: number;   // 0..200 (0.5% steps), 255 = unknown
  tempHalfC: number;    // (tempC + 20) * 2, 255 = unknown
}

export interface IpRecord {
  tsMs: number;
  vMv: number;
  iMa: number;
  dqUah: number;
  tempDc: number;       // 0.1 C
  eventType: EventTypeValue;
  sampleFlags: number;
  odSocHalfPct: number;
}

export const M5STICKC_PLUS_DEFAULT_PARAMS: CellParams = {
  flags: ParamFlags.TEMP_COMP,
  ocvMv: [
    3300, 3625, 3700, 3726, 3750, 3775, 3795, 3816, 3840,
    3860, 3890, 3940, 3980, 4035, 4095, 4140, 4200,
  ],
  qMaxDmah: 1200,
  r0Mohm: 280,
  r1Mohm: 140,
  tauS: 35,
  vFullMv: 4165,
  iTermMa: 10,
  chgEffSub900: 95,
};

export function crc16Ccitt(data: Uint8Array, len = data.length): number {
  let crc = 0xffff;
  for (let i = 0; i < len; i++) {
    crc ^= data[i] << 8;
    for (let b = 0; b < 8; b++) {
      crc = (crc & 0x8000) !== 0 ? ((crc << 1) ^ 0x1021) & 0xffff : (crc << 1) & 0xffff;
    }
  }
  return crc;
}

export function encodeCellParams(p: CellParams): Uint8Array {
  if (p.ocvMv.length !== OCV_POINTS) {
    throw new Error(`ocvMv must have exactly ${OCV_POINTS} points`);
  }
  const buf = new Uint8Array(PARAMS_BYTES);
  const dv = new DataView(buf.buffer);
  buf[0] = PARAMS_MAGIC_0;
  buf[1] = PARAMS_MAGIC_1;
  buf[2] = PARAMS_VERSION;
  buf[3] = p.flags & 0xff;
  for (let i = 0; i < OCV_POINTS; i++) {
    dv.setUint16(4 + i * 2, Math.round(p.ocvMv[i]) & 0xffff, true);
  }
  dv.setUint16(38, Math.round(p.qMaxDmah) & 0xffff, true);
  dv.setUint16(40, Math.round(p.r0Mohm) & 0xffff, true);
  dv.setUint16(42, Math.round(p.r1Mohm) & 0xffff, true);
  dv.setUint16(44, Math.round(p.tauS) & 0xffff, true);
  dv.setUint16(46, Math.round(p.vFullMv) & 0xffff, true);
  buf[48] = Math.round(p.iTermMa) & 0xff;
  buf[49] = Math.round(p.chgEffSub900) & 0xff;
  const crc = crc16Ccitt(buf, PARAMS_BYTES - 2);
  dv.setUint16(50, crc, true);
  return buf;
}

export function decodeCellParams(buf: Uint8Array): CellParams | null {
  if (buf.length < PARAMS_BYTES) return null;
  if (buf[0] !== PARAMS_MAGIC_0 || buf[1] !== PARAMS_MAGIC_1 || buf[2] !== PARAMS_VERSION) {
    return null;
  }
  const dv = new DataView(buf.buffer, buf.byteOffset, PARAMS_BYTES);
  const expected = crc16Ccitt(buf.subarray(0, PARAMS_BYTES - 2));
  if (expected !== dv.getUint16(50, true)) return null;

  const ocvMv: number[] = [];
  for (let i = 0; i < OCV_POINTS; i++) {
    const v = dv.getUint16(4 + i * 2, true);
    if (i > 0 && v < ocvMv[i - 1]) return null;
    ocvMv.push(v);
  }
  const qMaxDmah = dv.getUint16(38, true);
  if (qMaxDmah < 10) return null;

  return {
    flags: buf[3],
    ocvMv,
    qMaxDmah,
    r0Mohm: dv.getUint16(40, true),
    r1Mohm: dv.getUint16(42, true),
    tauS: dv.getUint16(44, true),
    vFullMv: dv.getUint16(46, true),
    iTermMa: buf[48],
    chgEffSub900: buf[49],
  };
}

export function encodeDtLog(dtS: number): number {
  const d = Math.max(0, Math.floor(dtS));
  if (d < 60) return d;
  if (d < 600) return 60 + Math.floor((d - 60) / 10);
  if (d < 86640) return 114 + Math.floor((d - 600) / 60);
  return Math.min(4095, 1548 + Math.floor((d - 86640) / 600));
}

export function decodeDtLog(code: number): number {
  const c = code & 0x0fff;
  if (c < 60) return c;
  if (c < 114) return 60 + (c - 60) * 10;
  if (c < 1548) return 600 + (c - 114) * 60;
  return 86640 + (c - 1548) * 600;
}

export function encodeTempHalfC(tempC: number): number {
  return Math.max(0, Math.min(254, Math.round((tempC + 20) * 2)));
}

export function decodeTempHalfC(code: number): number {
  return code * 0.5 - 20;
}

const ERASED_EVENT = 0xffffffffffffffffn;

export function packOdEvent(ev: UnpackedOdEvent): bigint {
  const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.trunc(x)));
  const dtBits = BigInt(encodeDtLog(ev.dtS) & 0x0fff);
  const evBits = BigInt(ev.type & 0x0f);
  const vBits = BigInt(clamp(ev.vMv - 2500, 0, 2047) & 0x7ff);
  const iBits = BigInt(clamp(ev.iMa, -1024, 1023) & 0x7ff);
  const dqBits = BigInt(clamp(ev.dqQmah, -512, 511) & 0x3ff);
  const socBits = BigInt(ev.socHalfPct & 0xff);
  const tBits = BigInt(ev.tempHalfC & 0xff);

  let raw =
    dtBits |
    (evBits << 12n) |
    (vBits << 16n) |
    (iBits << 27n) |
    (dqBits << 38n) |
    (socBits << 48n) |
    (tBits << 56n);
  if (raw === ERASED_EVENT) raw ^= 1n << 56n;
  return raw;
}

export function unpackOdEvent(raw: bigint): UnpackedOdEvent {
  const dtCode = Number(raw & 0x0fffn);
  const type = Number((raw >> 12n) & 0x0fn) as EventTypeValue;
  const vMv = 2500 + Number((raw >> 16n) & 0x7ffn);

  let iSigned = Number((raw >> 27n) & 0x7ffn);
  if ((iSigned & 0x400) !== 0) iSigned -= 0x800;

  let dqSigned = Number((raw >> 38n) & 0x3ffn);
  if ((dqSigned & 0x200) !== 0) dqSigned -= 0x400;

  return {
    dtS: decodeDtLog(dtCode),
    type,
    vMv,
    iMa: iSigned,
    dqQmah: dqSigned,
    socHalfPct: Number((raw >> 48n) & 0xffn),
    tempHalfC: Number((raw >> 56n) & 0xffn),
  };
}

export function unpackOdEventBytes(bytes: Uint8Array): UnpackedOdEvent[] {
  const out: UnpackedOdEvent[] = [];
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let off = 0; off + 8 <= bytes.byteLength; off += 8) {
    const raw = dv.getBigUint64(off, true);
    if (raw === ERASED_EVENT) continue;
    out.push(unpackOdEvent(raw));
  }
  return out;
}

export interface DecodedIpFrame {
  seq: number;
  frameFlags: number;
  records: IpRecord[];
}

export function encodeIpFrame(seq: number, frameFlags: number, records: IpRecord[]): Uint8Array {
  const total = 12 + records.length * 16 + 2;
  const buf = new Uint8Array(total);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x53; // 'S'
  buf[1] = 0x4f; // 'O'
  buf[2] = 0x43; // 'C'
  buf[3] = 0x31; // '1'
  dv.setUint32(4, seq >>> 0, true);
  dv.setUint16(8, records.length & 0xffff, true);
  buf[10] = frameFlags & 0xff;
  buf[11] = 0;

  let off = 12;
  for (const r of records) {
    dv.setUint32(off + 0, r.tsMs >>> 0, true);
    dv.setUint16(off + 4, Math.round(r.vMv) & 0xffff, true);
    dv.setInt16(off + 6, Math.round(r.iMa), true);
    dv.setInt32(off + 8, Math.round(r.dqUah), true);
    dv.setInt16(off + 12, Math.round(r.tempDc), true);
    buf[off + 14] = (r.eventType & 0x0f) | ((r.sampleFlags & 0x0f) << 4);
    buf[off + 15] = r.odSocHalfPct & 0xff;
    off += 16;
  }
  const crc = crc16Ccitt(buf, total - 2);
  dv.setUint16(total - 2, crc, true);
  return buf;
}

export function decodeIpFrame(buf: Uint8Array): DecodedIpFrame | null {
  if (buf.length < 14) return null;
  if (buf[0] !== 0x53 || buf[1] !== 0x4f || buf[2] !== 0x43 || buf[3] !== 0x31) return null;
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const count = dv.getUint16(8, true);
  const expected = 12 + count * 16 + 2;
  if (buf.length < expected) return null;
  if (crc16Ccitt(buf.subarray(0, expected - 2)) !== dv.getUint16(expected - 2, true)) {
    return null;
  }
  const records: IpRecord[] = [];
  let off = 12;
  for (let i = 0; i < count; i++) {
    const ef = buf[off + 14];
    records.push({
      tsMs: dv.getUint32(off + 0, true),
      vMv: dv.getUint16(off + 4, true),
      iMa: dv.getInt16(off + 6, true),
      dqUah: dv.getInt32(off + 8, true),
      tempDc: dv.getInt16(off + 12, true),
      eventType: (ef & 0x0f) as EventTypeValue,
      sampleFlags: (ef >> 4) & 0x0f,
      odSocHalfPct: buf[off + 15],
    });
    off += 16;
  }
  return {
    seq: dv.getUint32(4, true),
    frameFlags: buf[10],
    records,
  };
}
