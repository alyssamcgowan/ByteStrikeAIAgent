import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import re

from agent import agent  # your existing function

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

app = App(token=SLACK_BOT_TOKEN)

@app.event("app_mention")
def handle_mention(event, say, logger):
    print("=== APP MENTION RECEIVED ===")
    print(event)
    print("=============================")

    user_input = event.get("text", "")

    # Remove the bot mention text like <@U12345>
    cleaned_input = re.sub(r"<@[^>]+>", "", user_input).strip()

    print(f"Processing mention query: '{cleaned_input}'")

    if not cleaned_input:
        say("Hi there! What would you like to know about ByteStrike?")
        return

    try:
        response = agent(cleaned_input)
        print(f"Agent returned: {response}")
        say(response)
    except Exception as e:
        print(f"Agent error: {str(e)}")
        say("I'm having trouble processing your request right now. Please try again.")


# --- Respond to Direct Messages (DMs) ---
@app.event("message")
def handle_dm(event, say, logger):
    # Ignore messages that:
    # (1) Came from a bot, including itself
    # (2) Are not direct messages ("im")
    if event.get("bot_id"):
        return
    if event.get("channel_type") != "im":
        return

    print("=== DIRECT MESSAGE RECEIVED ===")
    print(event)
    print("===============================")

    user_input = event.get("text", "").strip()
    print(f"Processing DM query: '{user_input}'")

    if not user_input:
        say("How can I help?", thread_ts=event.get("ts"))
        return

    try:
        response = agent(user_input)
        print(f"Agent returned: {response}")
        say(response, thread_ts=event.get("ts"))
    except Exception as e:
        print(f"Agent error: {str(e)}")
        say("I'm having trouble processing your request right now. Please try again.")

if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
