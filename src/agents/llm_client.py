import os
import json
import logging
from typing import Dict, Any, Optional
import ollama
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.mode = os.getenv("OLLAMA_MODE", "local").lower()
        self.local_model = os.getenv("OLLAMA_MODEL_LOCAL", "llama3.1:8b")
        self.cloud_model = os.getenv("OLLAMA_MODEL_CLOUD", "llama3.1:70b-cloud")
        
        # 4 modes: local, cloud, auto, hybrid (auto falls back to local if cloud unauthenticated)
        if self.mode == "cloud":
            self.model = self.cloud_model
        elif self.mode in ("auto", "hybrid"):
            # hybrid/auto: prefer cloud if signed in, else local
            try:
                import pathlib
                cfg = pathlib.Path.home() / ".ollama" / "config.json"
                has_cloud = cfg.exists() and "ollama.com" in cfg.read_text()
            except Exception:
                has_cloud = False
            self.model = self.cloud_model if has_cloud else self.local_model
            self.mode = "cloud" if has_cloud else "local"
        else:
            self.model = self.local_model
            self.mode = "local"
            
        # The ollama python client uses the local daemon, which handles cloud proxying.
        self.client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    def chat(self, messages: list[dict], expect_json: bool = False) -> str:
        """
        Send a chat request to the configured Ollama model.
        """
        logger.info(f"Ollama request | Mode: {self.mode} | Model: {self.model} | JSON: {expect_json}")
        
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if expect_json:
            kwargs["format"] = "json"
            
        try:
            # Note: Timeout handling in the python client is typically managed at the httpx client level.
            # We log the mode to indicate latency expectations.
            response = self.client.chat(**kwargs)
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise

    def chat_structured(self, messages: list[dict]) -> Dict[str, Any]:
        """
        Gets structured output with a validate-and-retry loop.
        """
        raw_response = self.chat(messages, expect_json=True)
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            logger.warning("JSON decode failed. Retrying with stricter instructions.")
            
            retry_messages = messages.copy()
            retry_messages.append({
                "role": "user",
                "content": "The previous output was invalid JSON. Return ONLY valid JSON, no markdown, no explanation."
            })
            
            retry_response = self.chat(retry_messages, expect_json=True)
            try:
                return json.loads(retry_response)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode failed on retry. Raw: {retry_response}")
                raise ValueError("LLM failed to return valid JSON after retry.") from e
