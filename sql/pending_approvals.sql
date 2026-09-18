-- Telegram approval workflow (src/notify/approval_bot.py) - the piece that closes the gap
-- between deadline_scheduler.py's recommendation-only reminders and actually submitting the
-- move to FPL. Kept as its own file for the same reason reminder_log.sql is - no dependency on
-- concurrent schema work elsewhere.
--
-- One row per (season, event_id, kind) - kind is 'transfer'/'captain'/'squad', each approved
-- independently (Brian's explicit choice: 3 separate buttons, not one "approve everything").
-- plan_json snapshots exactly what would be submitted, captured when the approval request is
-- sent - execute_* functions re-fetch live prices/squad state at execution time rather than
-- trusting this snapshot for anything money-sensitive (prices can move between request and
-- decision), but it's what gets shown to Brian and is enough to reconstruct intent if the
-- process needs to be debugged after the fact.
--
-- status transitions: proposed -> approved|rejected|executing|executed|failed. 'executing' is a
-- transient claim state (UPDATE ... WHERE status='proposed' RETURNING ...) so a button-tap and
-- the T-30m auto-submit timeout can never both execute the same row - whichever UPDATE lands
-- first wins the claim, the other sees 0 rows updated and backs off.
CREATE TABLE IF NOT EXISTS pending_approvals (
    season          TEXT NOT NULL,
    event_id        INTEGER NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('transfer', 'captain', 'squad')),
    plan_json       JSONB NOT NULL,
    status          TEXT NOT NULL DEFAULT 'proposed'
                        CHECK (status IN ('proposed', 'approved', 'rejected', 'executing', 'executed', 'failed')),
    telegram_message_id  BIGINT,
    result_detail   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at      TIMESTAMPTZ,
    PRIMARY KEY (season, event_id, kind)
);

-- Telegram getUpdates() offset, persisted so a poll from a fresh GitHub Actions runner picks up
-- exactly where the last one left off instead of either missing button-taps or reprocessing old
-- ones. Single-row table (id is always 1) rather than a bare key-value settings table - this is
-- the only piece of cross-run state the bot needs right now.
CREATE TABLE IF NOT EXISTS telegram_update_offset (
    id              INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    last_update_id  BIGINT NOT NULL DEFAULT 0
);
INSERT INTO telegram_update_offset (id, last_update_id) VALUES (1, 0) ON CONFLICT (id) DO NOTHING;
