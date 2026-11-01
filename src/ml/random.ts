/** Mulberry32: the data, initialization and shuffle each use independent streams. */
export class Random {
  constructor(private state: number) {}
  next(): number {
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  normal(): number {
    return Math.sqrt(-2 * Math.log(Math.max(this.next(), 1e-12))) * Math.cos(2 * Math.PI * this.next());
  }
  shuffle<T>(values: T[]): T[] {
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
  }
}
