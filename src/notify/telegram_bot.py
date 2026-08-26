"""Telegram bot — Step 1: connectivity only. Confirms the bot token works and captures your
chat_id (needed to send you proactive deadline notifications later, since the bot can't message
you first without knowing where to send it). No deadline scheduling, no LLM negotiation, no FPL
writes here yet — this just proves the wire works, same staged-verification approach used for
the FPL auth/write path.

Setup:
  1. In Telegram, message @BotFather -> /newbot -> follow the prompts -> copy the token it gives you.
  2. Add it to .env as TELEGRAM_BOT_TOKEN (see .env.example).
  3. Run this script, then message your new bot anything in Telegram.
  4. It'll reply and print your chat_id here - add that to .env as TELEGRAM_CHAT_ID.

Run directly:
    python -m src.notify.telegram_bot
"""
from __future__ import annotations

import asyncio

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import settings


async def send_message(text: str) -> None:
    """Proactively sends a message to TELEGRAM_CHAT_ID - the bot speaking first, not replying.
    This is the primitive the deadline scheduler will call; verifying it works on its own
    (separate from the reply-only handlers above) confirms the bot can actually initiate
    contact, not just respond."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID not set in .env.")
    bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    await bot.send_message(chat_id=settings.TELEGRAM_CHAT_ID, text=text)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    print(f"Received /start from chat_id={chat_id}")
    await update.message.reply_text(
        "Fergie's Regression bot is connected.\n\n"
        f"Your chat_id is {chat_id} - add it to .env as TELEGRAM_CHAT_ID."
    )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    text = update.message.text
    print(f"Received message from chat_id={chat_id}: {text!r}")
    await update.message.reply_text(f"Got it (chat_id={chat_id}): {text!r}")


def run() -> None:
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env - see .env.example.")

    app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    print("Bot starting (long polling) - go message it in Telegram now. Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    run()
