import os
import re
import asyncio
from dotenv import load_dotenv

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler

# Assuming your agent is now an async function that returns a stream (AsyncIterator)
from agent import stream_agent as agent

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
    # Removed the asyncio.to_thread call, as re.sub is fast enough not to block
    return re.sub(r"<@[^>]+>", "", text).strip()


# ---------------------------
# Core Streaming Function
# ---------------------------
async def process_and_stream(text_input, say, client, channel_id, thread_ts=None, logger=None):
    """Handles the RAG call and streams the response back to Slack."""
    
    # 1. Post an initial placeholder message
    # We use 'say' here to get the ts (timestamp) and channel_id for updates
    placeholder = await say(
        text="Thinking...",
        thread_ts=thread_ts
    )
    
    # Extract the necessary IDs
    ts = placeholder.get("ts")
    
    full_response = ""
    # The agent must now return an async generator/iterator of text chunks
    chunk_stream = agent(text_input) 
    
    # Track the number of updates to prevent hitting rate limits
    # Slack allows about 1 message update per second.
    update_count = 0
    
    try:
        # 2. Iterate over the stream of chunks
        async for chunk in chunk_stream:
            if not chunk:
                continue

            full_response += chunk
            
            # 3. Update the message periodically
            # We update on every 10th chunk or if the text is long enough for the first time
            # Adjust the update logic to balance responsiveness and rate limits.
            if len(full_response) % 10 == 0 or len(full_response) < 50 and len(full_response) > 0:
                # Use the Slack client to call the chat.update API
                # This call overwrites the previous message content
                await client.chat_update(
                    channel=channel_id,
                    ts=ts,
                    text=f"{full_response} ",
                )
                await asyncio.sleep(0.1) # small pause to yield control
                update_count += 1

        # 4. Final Update: Remove the thinking indicator and post the complete, final text
        if full_response:
            await client.chat_update(
                channel=channel_id,
                ts=ts,
                text=full_response, # Final text, no writing emoji
            )
        else:
            await client.chat_update(
                channel=channel_id,
                ts=ts,
                text="The agent did not return a response.",
            )

    except Exception as e:
        if logger:
            logger.error(f"Agent streaming error: {e}")
        # Final update to show an error
        await client.chat_update(
            channel=channel_id,
            ts=ts,
            text=f"Sorry, an error occurred during generation: {e}",
        )


# ---------------------------
# Event: @mention
# ---------------------------
# Pass 'client' for chat.update and 'channel_id' for thread ID
@app.event("app_mention")
async def handle_mention(event, say, client, logger):
    cleaned = clean_input(event.get("text", ""))
    channel_id = event["channel"]
    thread_ts = event.get("ts", event.get("event_ts")) # Use main timestamp for the thread

    if not cleaned:
        await say("Hi! What would you like to know?")
        return

    # Call the new streaming function
    await process_and_stream(
        text_input=cleaned,
        say=say,
        client=client,
        channel_id=channel_id,
        thread_ts=thread_ts,
        logger=logger
    )


# ---------------------------
# Event: Direct Messages
# ---------------------------
@app.event("message")
async def handle_dm(event, say, client, logger):

    # ignore bot messages
    if event.get("bot_id"):
        return

    if event.get("channel_type") != "im":
        return

    cleaned = event.get("text", "").strip()
    channel_id = event["channel"]
    thread_ts = event.get("ts")

    if not cleaned:
        await say("How can I help?", thread_ts=thread_ts)
        return

    # Call the new streaming function
    await process_and_stream(
        text_input=cleaned,
        say=say,
        client=client,
        channel_id=channel_id,
        thread_ts=thread_ts,
        logger=logger
    )


# ---------------------------
# Main entry
# ---------------------------
async def main():
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())