import os
import json
import re
from pathlib import Path

INPUT_DIR = "slack"   
OUTPUT_DIR = "cleanedSlack"  
OUTPUT_FORMAT = "txt"            

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    #clean slack message text for embedding
    if not text:
        return ""

    text = re.sub(r"<@[A-Z0-9]+>", "", text)
    text = re.sub(r"<(http[^>|]+)(\|[^>]+)?>", r"\1", text)
    text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def consolidate_slack_export(input_dir: str, output_dir: str, format: str = "txt"):
    all_documents = []

    #traverse channel folders
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".json"):
                continue

            file_path = os.path.join(root, file)
            channel = Path(root).name

            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    messages = json.load(f)
                except Exception:
                    print("Error reading:", file_path)
                    continue

            for m in messages:
                text = clean_text(m.get("text", ""))
                user = m.get("user", "")
                ts = m.get("ts", "")

                if not text:
                    continue

                doc = {
                    "channel": channel,
                    "timestamp": ts,
                    "user": user,
                    "text": text
                }

                all_documents.append(doc)

    if format == "txt":
        #one .txt file per channel
        channel_docs = {}
        for doc in all_documents:
            channel_docs.setdefault(doc["channel"], []).append(doc)

        for channel, docs in channel_docs.items():
            out_path = os.path.join(output_dir, f"{channel}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                for d in docs:
                    f.write(f"[{d['timestamp']}] {d['user']}: {d['text']}\n")

        print("TXT export complete.")
        return


    if format == "jsonl":
        out_path = os.path.join(output_dir, "slack_messages.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for d in all_documents:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        print("JSONL export complete.")
        return


if __name__ == "__main__":
    consolidate_slack_export(INPUT_DIR, OUTPUT_DIR, OUTPUT_FORMAT)
