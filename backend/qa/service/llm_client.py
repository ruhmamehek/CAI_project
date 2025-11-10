"""LLM client abstraction for different providers."""

import logging
from typing import Optional
from abc import ABC, abstractmethod

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            Generated text
        """
        pass

class GeminiClient(LLMClient):
    """Gemini LLM client."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Gemini client.
        
        Args:
            config: LLM configuration
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
        
        if not config.api_key:
            raise ValueError("Gemini API key not provided")
        
        genai.configure(api_key=config.api_key)
        self.genai = genai
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        
        logger.info(f"Initialized Gemini client with model: {self.model}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using Gemini."""
        # Build the full prompt with system message if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Generate content using Gemini
        model = self.genai.GenerativeModel(
            model_name=self.model,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }
        )
        
        response = model.generate_content(full_prompt)
        return response.text.strip()
        
# class OpenAIClient(LLMClient):
#     """OpenAI LLM client."""
    
#     def __init__(self, config: LLMConfig):
#         """
#         Initialize OpenAI client.
        
#         Args:
#             config: LLM configuration
#         """
#         try:
#             from openai import OpenAI
#         except ImportError:
#             raise ImportError("openai package not installed. Install with: pip install openai")
        
#         if not config.api_key:
#             raise ValueError("OpenAI API key not provided")
        
#         self.client = OpenAI(api_key=config.api_key)
#         self.model = config.model
#         self.temperature = config.temperature
#         self.max_tokens = config.max_tokens
        
#         logger.info(f"Initialized OpenAI client with model: {self.model}")
    
#     def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
#         """Generate response using OpenAI."""
#         messages = []
        
#         if system_prompt:
#             messages.append({"role": "system", "content": system_prompt})
        
#         messages.append({"role": "user", "content": prompt})
        
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=self.temperature,
#             max_tokens=self.max_tokens
#         )
        
#         return response.choices[0].message.content.strip()


# class AnthropicClient(LLMClient):
#     """Anthropic LLM client."""
    
#     def __init__(self, config: LLMConfig):
#         """
#         Initialize Anthropic client.
        
#         Args:
#             config: LLM configuration
#         """
#         try:
#             from anthropic import Anthropic
#         except ImportError:
#             raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        
#         if not config.api_key:
#             raise ValueError("Anthropic API key not provided")
        
#         self.client = Anthropic(api_key=config.api_key)
#         self.model = config.model
#         self.temperature = config.temperature
#         self.max_tokens = config.max_tokens
        
#         logger.info(f"Initialized Anthropic client with model: {self.model}")
    
    # def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
    #     """Generate response using Anthropic."""
    #     messages = [{"role": "user", "content": prompt}]
        
    #     response = self.client.messages.create(
    #         model=self.model,
    #         max_tokens=self.max_tokens,
    #         temperature=self.temperature,
    #         messages=messages,
    #         system=system_prompt if system_prompt else None
    #     )
        
    #     return response.content[0].text.strip()


# class OllamaClient(LLMClient):
#     """Ollama LLM client (OpenAI-compatible API)."""
    
#     def __init__(self, config: LLMConfig):
#         """
#         Initialize Ollama client.
        
#         Args:
#             config: LLM configuration
#         """
#         try:
#             from openai import OpenAI
#         except ImportError:
#             raise ImportError("openai package not installed for Ollama. Install with: pip install openai")
        
#         base_url = config.base_url or 'http://localhost:11434/v1'
#         api_key = config.api_key or 'ollama'
        
#         self.client = OpenAI(base_url=base_url, api_key=api_key)
#         self.model = config.model
#         self.temperature = config.temperature
#         self.max_tokens = config.max_tokens
        
#         logger.info(f"Initialized Ollama client with model: {self.model} at {base_url}")
    
#     def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
#         """Generate response using Ollama."""
#         messages = []
        
#         if system_prompt:
#             messages.append({"role": "system", "content": system_prompt})
        
#         messages.append({"role": "user", "content": prompt})
        
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=self.temperature,
#             max_tokens=self.max_tokens
#         )
        
#         return response.choices[0].message.content.strip()


def create_llm_client(config: LLMConfig) -> LLMClient:
    """
    Create LLM client based on provider.
    
    Args:
        config: LLM configuration
        
    Returns:
        LLMClient instance
    """
    provider = config.provider.lower()
    
    if provider == "gemini":
        return GeminiClient(config)
    elif provider == "openai":
        return OpenAIClient(config)
    elif provider == "anthropic":
        return AnthropicClient(config)
    elif provider == "ollama":
        return OllamaClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

