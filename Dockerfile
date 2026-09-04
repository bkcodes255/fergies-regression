# Runs the Telegram instruction bot (src/notify/telegram_bot.py) as a long-lived process -
# see that module's docstring for why it needs to stay up continuously, unlike the
# GitHub-Actions-cron pieces (deadline_scheduler.py, ingestion) which don't. Not used for those;
# this image is scoped to exactly what the bot's import graph touches.
FROM python:3.12-slim

WORKDIR /app

# Only what src/notify/telegram_bot.py, src/notify/bot_commands.py, src/fpl_write/client.py,
# src/recommendations/transfers.py and src/notify/deadline_scheduler.py's imports actually need -
# matches the deliberately-trimmed install list in .github/workflows/deadline_reminders.yml
# rather than the full requirements.txt (matplotlib/scikit-learn/streamlit/etc. would just slow
# every image build and cold start down for nothing here).
RUN pip install --no-cache-dir \
    pandas psycopg2-binary sqlalchemy python-telegram-bot[job-queue] python-dotenv rapidfuzz requests

COPY config/ ./config/
COPY src/ ./src/

# Real secrets (DATABASE_URL, FPL_ENTRY_ID, FPL_API_AUTHORIZATION, FPL_SESSION_COOKIE,
# TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) are injected as real environment variables by the host
# (e.g. `fly secrets set ...` - see fly.toml) - never baked into the image or committed as a
# .env file here.
CMD ["python", "-m", "src.notify.telegram_bot"]
