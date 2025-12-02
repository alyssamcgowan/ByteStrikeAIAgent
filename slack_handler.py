# slack_handler.py
import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import re
from agent import agent, chat_history

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

app = App(token=SLACK_BOT_TOKEN)

@app.event("app_mention")
def handle_mention(event, say, logger):
    print("=== APP MENTION RECEIVED ===")
    user_input = event.get("text", "")
    user_id = event.get("user", "unknown")

    # Remove the bot mention text like <@U12345>
    cleaned_input = re.sub(r"<@[^>]+>", "", user_input).strip()

    print(f"Processing mention query from user {user_id}: '{cleaned_input}'")

    if not cleaned_input:
        say("Hi there! What would you like to know about ByteStrike?")
        return

    try:
        response = agent(cleaned_input, user_id)
        print(f"Agent returned: {response[:200]}...")
        say(response)
    except Exception as e:
        print(f"Agent error: {str(e)}")
        say("I'm having trouble processing your request right now. Please try again.")

@app.event("message")
def handle_dm(event, say, logger):
    # Ignore messages that:
    # (1) Came from a bot, including itself
    # (2) Are not direct messages ("im")
    # (3) ignore edited messages
    if event.get("bot_id"):
        return
    if event.get("channel_type") != "im":
        return
    if event.get("subtype") == "message_changed":
        return

    print("=== DIRECT MESSAGE RECEIVED ===")
    user_input = event.get("text", "").strip()
    user_id = event.get("user", "unknown")
    print(f"Processing DM query from user {user_id}: '{user_input}'")

    if not user_input:
        say("How can I help?", thread_ts=event.get("ts"))
        return

    try:
        response = agent(user_input, user_id)
        print(f"Agent returned: {response[:200]}...")
        say(response, thread_ts=event.get("ts"))
    except Exception as e:
        print(f"Agent error: {str(e)}")
        say("I'm having trouble processing your request right now. Please try again.")

if __name__ == "__main__":
    print("Starting Slack bot in socket mode")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()