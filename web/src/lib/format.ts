// Pure display helpers, safe to import from browser scripts. Keep data loading out of this file:
// src/lib/data.ts pulls every JSON file into its module, and that must never reach the browser.

/** "Sat 17 Oct, 11:00 UK time" - FPL deadlines are set in UK time, so that's the clock shown. */
export function ukTime(iso: string): string {
  const d = new Date(iso);
  const part = (o: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat("en-GB", { timeZone: "Europe/London", ...o }).format(d);
  return `${part({ weekday: "short" })} ${part({ day: "numeric" })} ${part({ month: "short" })}, ${part({ hour: "2-digit", minute: "2-digit", hour12: false })} UK time`;
}

export const signed = (v: number, digits = 1) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(digits);
