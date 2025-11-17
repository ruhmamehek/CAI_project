"""LLM client abstraction for different providers."""

import logging
from typing import Optional, List
from abc import ABC, abstractmethod

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, images: Optional[List[bytes]] = None, image_mime_types: Optional[List[str]] = None) -> str:
        """Generate text from prompt."""
        pass

class GeminiClient(LLMClient):
    """Gemini LLM client."""
    
    def __init__(self, config: LLMConfig):
        """Initialize Gemini client."""
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai package not installed. Install with: pip install google-genai")
        
        if not config.api_key:
            raise ValueError("Gemini API key not provided")
        
        self.client = genai.Client(api_key=config.api_key)
        self.model_name = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        
        logger.info(f"Initialized Gemini client with model: {self.model_name}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, images: Optional[List[bytes]] = None, image_mime_types: Optional[List[str]] = None) -> str:
        """Generate response using Gemini with retry logic for rate limits."""
        import time
        
        max_retries = 3
        retry_delay = 5
        
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        from google.genai import types
        contents = [full_prompt]
        if images and image_mime_types:
            for img_bytes, mime_type in zip(images, image_mime_types):
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    } if self.max_tokens else {
                        "temperature": self.temperature,
                    }
                )
                
                return response.text.strip()
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        if "retry in" in error_str.lower():
                            import re
                            delay_match = re.search(r'retry in ([\d.]+)s', error_str, re.IGNORECASE)
                            if delay_match:
                                retry_delay = float(delay_match.group(1)) + 1
                        
                        logger.warning(f"Rate limit error (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay:.1f} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        logger.error(f"Rate limit error after {max_retries} attempts")
                        raise ValueError(f"Gemini API rate limit exceeded. Please try again later. Error: {error_str}")
                else:
                    raise


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create LLM client based on provider."""
    provider = config.provider.lower()
    
    if provider == "gemini":
        return GeminiClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
