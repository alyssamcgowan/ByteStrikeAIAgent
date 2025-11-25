import os
import re
import asyncio
from dotenv import load_dotenv

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler

from agent import agent 

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

app = AsyncApp(token=SLACK_BOT_TOKEN)


# ---------------------------
# Utility: async-safe cleaning
# ---------------------------
def clean_input(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<@[^>]+>", "", text).strip()


# ---------------------------
# Event: @mention
# ---------------------------
@app.event("app_mention")
async def handle_mention(event, say, logger):
    cleaned = clean_input(event.get("text", ""))

    if not cleaned:
        await say("Hi! What would you like to know?")
        return

    try:
        reply = await agent(cleaned)      # <-- IMPORTANT
        await say(reply)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        await say("Sorry, something went wrong.")


# ---------------------------
# Event: Direct Messages
# ---------------------------
@app.event("message")
async def handle_dm(event, say, logger):

    # ignore bot messages
    if event.get("bot_id"):
        return

    if event.get("channel_type") != "im":
        return

    cleaned = event.get("text", "").strip()

    if not cleaned:
        await say("How can I help?", thread_ts=event.get("ts"))
        return

    try:
        reply = await agent(cleaned)
        await say(reply, thread_ts=event.get("ts"))
    except Exception as e:
        logger.error(f"Agent error: {e}")
        await say("Sorry, something went wrong.", thread_ts=event.get("ts"))


# ---------------------------
# Main entry
# ---------------------------
async def main():
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
