import re

def clean_text(text: str) -> str:
    """
    Removes HTML tags, Markdown image tags, and other noise from text.
    """
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove Markdown image tags and base64 data
    text = re.sub(r'!\\[.*?\\]\\(data:image\\/[a-zA-Z]+;base64,.*?\\)', ' ', text)
    # Remove any remaining markdown image tags
    text = re.sub(r'!\\[.*?\\]\\(.*?\\)', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\\s+', ' ', text).strip()
    return text
