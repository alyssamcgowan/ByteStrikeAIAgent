import mailbox
from bs4 import BeautifulSoup
import re
import json

def extract_text_from_email(msg):
    text_parts = []
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", "")).lower()
            
            if "attachment" in content_disposition:
                continue
            
            if content_type == "text/plain":
                try:
                    text_parts.append(part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore'))
                except Exception:
                    continue
            elif content_type == "text/html" and not text_parts:
                html = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
                soup = BeautifulSoup(html, "html.parser")
                text_parts.append(soup.get_text())
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            text_parts.append(msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore'))
        elif content_type == "text/html":
            html = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')
            soup = BeautifulSoup(html, "html.parser")
            text_parts.append(soup.get_text())

    full_text = "\n".join(text_parts)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    return full_text


def clean_mbox(input_path, output_path="cleaned_emails.json"):
    mbox = mailbox.mbox(input_path)
    cleaned_data = []

    for msg in mbox:
        try:
            sender = msg.get("From", "")
            to = msg.get("To", "")
            cc = msg.get("Cc", "")
            bcc = msg.get("Bcc", "")
            body = extract_text_from_email(msg)
            
            cleaned_data.append({
                "from": sender,
                "to": to,
                "cc": cc,
                "bcc": bcc,
                "body": body
            })
        except Exception as e:
            print(f"Error processing message: {e}")
            continue

    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"Cleaned data saved to {output_path} ({len(cleaned_data)} emails)")



if __name__ == "__main__":
    input_file = "emails/mail.mbox"
    clean_mbox(input_file)
