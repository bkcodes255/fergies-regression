-- Deadline reminder idempotency log (src/notify/deadline_scheduler.py).
--
-- Kept as its own file rather than folded into schema.sql for now - there's a concurrent
-- session actively editing schema.sql for the injury-signal feature (2026-08-26), and this
-- table has no dependency on that work. TODO: fold this into schema.sql once that lands.
--
-- GitHub Actions runs the scheduler on a cron (every ~15 min); a tier ('T-24h'/'T-3h'/'T-30m')
-- is "due" once now crosses into its window before a gameweek's deadline. Without this table,
-- every run inside a window would re-fire the same Telegram message. One row per
-- (season, event_id, tier) actually sent - checked before sending, inserted after.
CREATE TABLE IF NOT EXISTS reminder_log (
    season      TEXT NOT NULL,
    event_id    INTEGER NOT NULL,
    tier        TEXT NOT NULL,
    sent_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (season, event_id, tier)
);
