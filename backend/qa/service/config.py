"""Configuration management for RAG service."""

import os
import yaml
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChromaDBConfig:
    """ChromaDB configuration."""
    api_key: str
    tenant: str
    database: str
    collection_name: str = "sec_filings"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    
    @classmethod
    def from_env(cls, collection_name: Optional[str] = None, embedding_model: Optional[str] = None) -> 'ChromaDBConfig':
        """Create ChromaDBConfig from environment variables."""
        api_key = os.getenv('CHROMA_API_KEY')
        tenant = os.getenv('CHROMA_TENANT')
        database = os.getenv('CHROMA_DATABASE')
        
        if not all([api_key, tenant, database]):
            raise ValueError(
                "Missing ChromaDB credentials. Set environment variables: "
                "CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE"
            )
        
        return cls(
            api_key=api_key,
            tenant=tenant,
            database=database,
            collection_name=collection_name or "sec_filings",
            embedding_model=embedding_model or "BAAI/bge-base-en-v1.5"
        )


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str  # "openai", "anthropic", "ollama"
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None  # None means no limit (model's maximum)
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create LLMConfig from environment variables."""
        provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
        # If LLM_MAX_TOKENS is not set or empty, use None (no limit)
        max_tokens_str = os.getenv('LLM_MAX_TOKENS', '').strip()
        max_tokens = int(max_tokens_str) if max_tokens_str else None
        temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))
        
        if provider == "gemini":
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            return cls(
                provider=provider,
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature
            )

        # elif provider == "openai":
        #     api_key = os.getenv('OPENAI_API_KEY')
        #     if not api_key:
        #         raise ValueError("OPENAI_API_KEY not set")
        #     return cls(provider=provider, model=model, api_key=api_key)
        
        # elif provider == "anthropic":
        #     api_key = os.getenv('ANTHROPIC_API_KEY')
        #     if not api_key:
        #         raise ValueError("ANTHROPIC_API_KEY not set")
        #     return cls(provider=provider, model=model, api_key=api_key)
        
        # elif provider == "ollama":
        #     base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
        #     api_key = os.getenv('OLLAMA_API_KEY', 'ollama')
        #     return cls(provider=provider, model=model, api_key=api_key, base_url=base_url)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


@dataclass
class RAGConfig:
    """RAG service configuration."""
    chroma: ChromaDBConfig
    llm: LLMConfig
    top_k: int = 20
    max_context_length: int = 50000
    rerank_max_length: int = 512  # Max length for cross-encoder reranking (512, 1024, etc.)
    enable_reranking: bool = True  # Whether to enable reranking
    
    @classmethod
    def from_file(cls, config_path: str = "config.yaml") -> 'RAGConfig':
        """Load configuration from YAML file and environment variables."""
        config_file = Path(config_path)
        
        # Load YAML config if it exists
        if config_file.exists():
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
        else:
            yaml_config = {}
        
        # Get retrieval settings from config file
        retrieval_config = yaml_config.get('retrieval', {})
        dense_config = retrieval_config.get('dense', {})
        
        top_k = dense_config.get('top_k', 20)
        embedding_model = dense_config.get('model_name', 'BAAI/bge-base-en-v1.5')
        
        # Get reranking settings (with fallback to env vars)
        rerank_config = yaml_config.get('reranking', {})
        rerank_max_length = rerank_config.get('max_length', int(os.getenv('RERANK_MAX_LENGTH', '512')))
        enable_reranking = rerank_config.get('enable', os.getenv('ENABLE_RERANKING', 'false').lower() == 'true')
        
        # Get LLM settings from config file (with fallback to env vars)
        llm_config = yaml_config.get('llm', {})
        # If max_tokens is None, empty, or not specified, allow unlimited (None)
        llm_max_tokens_raw = llm_config.get('max_tokens', os.getenv('LLM_MAX_TOKENS', '').strip())
        if llm_max_tokens_raw is None or (isinstance(llm_max_tokens_raw, str) and not llm_max_tokens_raw.strip()):
            llm_max_tokens = None
        else:
            llm_max_tokens = int(llm_max_tokens_raw)
        llm_temperature = llm_config.get('temperature', float(os.getenv('LLM_TEMPERATURE', '0.1')))
        
        # Create config objects
        chroma = ChromaDBConfig.from_env(
            collection_name="sec_filings",
            embedding_model=embedding_model
        )
        llm = LLMConfig.from_env()
        # Override max_tokens and temperature if specified in config file
        llm.max_tokens = llm_max_tokens
        llm.temperature = llm_temperature
        
        return cls(
            chroma=chroma,
            llm=llm,
            top_k=top_k,
            rerank_max_length=rerank_max_length,
            enable_reranking=enable_reranking
        )
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create RAGConfig from environment variables only."""
        chroma = ChromaDBConfig.from_env()
        llm = LLMConfig.from_env()
        top_k = int(os.getenv('RAG_TOP_K', '20'))
        rerank_max_length = int(os.getenv('RERANK_MAX_LENGTH', '512'))
        enable_reranking = os.getenv('ENABLE_RERANKING', 'true').lower() == 'true'
        
        return cls(
            chroma=chroma,
            llm=llm,
            top_k=top_k,
            rerank_max_length=rerank_max_length,
            enable_reranking=enable_reranking
        )


def load_config(config_path: Optional[str] = None) -> RAGConfig:
    """
    Load RAG configuration.
    
    Args:
        config_path: Path to config YAML file (optional)
        
    Returns:
        RAGConfig instance
        
    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist
    """
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            return RAGConfig.from_file(config_path)
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Config file not found: {config_path}. Using environment variables.")
            return RAGConfig.from_env()
    else:
        return RAGConfig.from_env()

