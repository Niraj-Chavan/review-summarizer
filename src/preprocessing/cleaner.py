import re
import emoji
import langid

class TextCleaner:
    def __init__(self):
        # A simple dictionary for common slang/abbreviations
        self.slang_dict = {
            "u": "you", "ur": "your", "idk": "I don't know",
            "imo": "in my opinion", "imho": "in my humble opinion",
            "gr8": "great", "tbh": "to be honest", "lol": "laughing out loud",
            "omg": "oh my god", "smh": "shaking my head", "btw": "by the way"
        }

    def _replace_slang(self, text: str) -> str:
        words = text.split()
        # Case-insensitive replacement
        replaced = [self.slang_dict.get(w.lower(), w) for w in words]
        return " ".join(replaced)

    def clean_text(self, text: str) -> str:
        """
        Cleans the review text by removing HTML tags, URLs, 
        demojizing emojis (to preserve meaning), and handling common slang.
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Convert emojis to text (e.g. 👍 -> :thumbs_up:)
        text = emoji.demojize(text)
        
        # Replace common slang
        text = self._replace_slang(text)
        
        # Replace multiple spaces/newlines with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the text using langid.
        Returns ISO 639-1 code (e.g., 'en', 'es').
        """
        if not text or len(text.strip()) < 3:
            return "unknown"
            
        try:
            lang, _ = langid.classify(text)
            return lang
        except Exception:
            return "unknown"
