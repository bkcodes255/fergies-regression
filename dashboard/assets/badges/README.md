# Team crest images

Drop a small image file here per team, named `<TEAM_SHORT_NAME>.<ext>` (case-insensitive),
matching the `short_name` values in the `teams` table (e.g. `LIV.png`, `ARS.png`, `MCI.svg`).
Supported extensions: `.png`, `.svg`, `.jpg`, `.jpeg`, `.webp`.

`dashboard/app.py`'s `_crest_data_uri()` picks these up automatically — no code change needed.
A team with no file here just keeps the current colored badge (no broken-image icon).

Keep each file small (a few KB) — they're base64-embedded directly into every rendered tile,
not served as static files, since the dashboard's `st.iframe` tiles don't use Streamlit's
static-file serving (not enabled for this app).

Real club crests are trademarked assets — make sure you have the right to use whatever you put
here before committing it.
