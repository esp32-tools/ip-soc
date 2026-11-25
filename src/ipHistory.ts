// `ip soc history`: High-resolution remote battery history store.
// Merges live `SOC1` stream records, drained spillover frames, and hypercompressed 8-byte
// `od soc history` gap-fill records into one deduplicated timeline stored on the remote (RAM/IndexedDB).

import {
  decodeIpFrame,
  decodeTempHalfC,
  EventType,
  SampleFlags,
  unpackOdEventBytes,
  type EventTypeValue,
  type IpRecord,
} from "./protocol.ts";

export interface TimelineEntry {
  tsMs: number;
  vV: number;
  iMa: number;
  dqMah: number;
  tempC: number;
  eventType: EventTypeValue;
  sampleFlags: number;
  odSocPct: number | null;
  source: "ip_live" | "ip_spillover" | "od_ring";
}

export interface HistoryStorageAdapter {
  load(deviceId: string): Promise<TimelineEntry[]>;
  save(deviceId: string, entries: TimelineEntry[]): Promise<void>;
  clear(deviceId: string): Promise<void>;
}

export class MemoryHistoryStore implements HistoryStorageAdapter {
  private readonly map = new Map<string, TimelineEntry[]>();
  async load(deviceId: string): Promise<TimelineEntry[]> {
    return (this.map.get(deviceId) ?? []).slice();
  }
  async save(deviceId: string, entries: TimelineEntry[]): Promise<void> {
    this.map.set(deviceId, entries.slice());
  }
  async clear(deviceId: string): Promise<void> {
    this.map.delete(deviceId);
  }
}

// Zero-dependency browser IndexedDB adapter for balax-remote / PWAs.
export class IndexedDbHistoryStore implements HistoryStorageAdapter {
  private readonly dbName: string;
  private readonly storeName = "ip_soc_history";

  constructor(dbName = "esp32_tools_ip_soc") {
    this.dbName = dbName;
  }

  private open(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, 1);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(this.storeName)) {
          req.result.createObjectStore(this.storeName);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async load(deviceId: string): Promise<TimelineEntry[]> {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, "readonly");
      const req = tx.objectStore(this.storeName).get(deviceId);
      req.onsuccess = () => resolve((req.result as TimelineEntry[] | undefined) ?? []);
      req.onerror = () => reject(req.error);
    });
  }

  async save(deviceId: string, entries: TimelineEntry[]): Promise<void> {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, "readwrite");
      tx.objectStore(this.storeName).put(entries, deviceId);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async clear(deviceId: string): Promise<void> {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, "readwrite");
      tx.objectStore(this.storeName).delete(deviceId);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
}

export class IpSocHistory {
  private entries: TimelineEntry[] = [];
  private lastAckSeq = 0;
  private readonly maxEntries: number;

  constructor(maxEntries = 100_000) {
    this.maxEntries = maxEntries;
  }

  ingestIpFrame(frameBytes: Uint8Array, source: "ip_live" | "ip_spillover" = "ip_live"): {
    seq: number;
    added: number;
  } | null {
    const decoded = decodeIpFrame(frameBytes);
    if (!decoded) return null;
    const added = this.ingestIpRecords(decoded.records, source);
    if (decoded.seq > this.lastAckSeq) this.lastAckSeq = decoded.seq;
    return { seq: decoded.seq, added };
  }

  ingestIpRecords(records: IpRecord[], source: "ip_live" | "ip_spillover" = "ip_live"): number {
    let added = 0;
    const existingTs = new Set(this.entries.map((e) => e.tsMs));
    for (const r of records) {
      if (existingTs.has(r.tsMs)) continue;
      existingTs.add(r.tsMs);
      this.entries.push({
        tsMs: r.tsMs,
        vV: r.vMv * 1e-3,
        iMa: r.iMa,
        dqMah: r.dqUah * 1e-3,
        tempC: r.tempDc * 0.1,
        eventType: r.eventType,
        sampleFlags: r.sampleFlags,
        odSocPct: r.odSocHalfPct <= 200 ? r.odSocHalfPct * 0.5 : null,
        source,
      });
      added++;
    }
    if (added > 0) this.sortAndTrim();
    return added;
  }

  // Reconstructs offline gaps from `od soc history`'s chronological 8-byte ring,
  // anchored to `anchorEndTsMs` (the device timestamp of the newest event in the ring).
  mergeOdHistoryRing(ringBytes: Uint8Array, anchorEndTsMs: number): number {
    const events = unpackOdEventBytes(ringBytes);
    if (events.length === 0) return 0;

    // Walk backwards from anchorEndTsMs using each event's log-quantized dtS.
    const timestamps = new Array<number>(events.length);
    timestamps[events.length - 1] = anchorEndTsMs;
    for (let i = events.length - 2; i >= 0; i--) {
      timestamps[i] = Math.max(0, timestamps[i + 1] - events[i + 1].dtS * 1000);
    }

    let added = 0;
    for (let i = 0; i < events.length; i++) {
      const ts = timestamps[i];
      // Do not overwrite high-resolution IP entries within +/- 2 s.
      const covered = this.entries.some(
        (e) => e.source !== "od_ring" && Math.abs(e.tsMs - ts) <= 2000,
      );
      if (covered) continue;
      const ev = events[i];
      this.entries.push({
        tsMs: ts,
        vV: ev.vMv * 1e-3,
        iMa: ev.iMa,
        dqMah: ev.dqQmah * 0.25,
        tempC: ev.tempHalfC <= 254 ? decodeTempHalfC(ev.tempHalfC) : 25,
        eventType: ev.type,
        sampleFlags:
          SampleFlags.HAS_DQ |
          (ev.type === EventType.CHG_DONE ? SampleFlags.CHARGE_DONE : 0) |
          (ev.iMa > 5 ? SampleFlags.CHARGING : 0),
        odSocPct: ev.socHalfPct <= 200 ? ev.socHalfPct * 0.5 : null,
        source: "od_ring",
      });
      added++;
    }
    if (added > 0) this.sortAndTrim();
    return added;
  }

  getTimeline(): readonly TimelineEntry[] {
    return this.entries;
  }

  getLastAckSeq(): number {
    return this.lastAckSeq;
  }

  async loadFrom(store: HistoryStorageAdapter, deviceId: string): Promise<void> {
    this.entries = await store.load(deviceId);
    this.sortAndTrim();
  }

  async saveTo(store: HistoryStorageAdapter, deviceId: string): Promise<void> {
    await store.save(deviceId, this.entries);
  }

  private sortAndTrim(): void {
    this.entries.sort((a, b) => a.tsMs - b.tsMs);
    if (this.entries.length > this.maxEntries) {
      this.entries = this.entries.slice(this.entries.length - this.maxEntries);
    }
  }
}
