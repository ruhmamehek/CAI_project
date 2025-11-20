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
    collection_name: str = "isaac_test_filings"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    
    @classmethod
    def from_env(cls, collection_name: Optional[str] = None, embedding_model: Optional[str] = None) -> 'ChromaDBConfig':
        """Create ChromaDBConfig from environment variables.
        
        Note: collection_name should be provided from config.yaml, not environment variables.
        """
        api_key = os.getenv('CHROMA_API_KEY')
        tenant = os.getenv('CHROMA_TENANT')
        database = os.getenv('CHROMA_DATABASE')
        
        if not all([api_key, tenant, database]):
            raise ValueError(
                "Missing ChromaDB credentials. Set environment variables: "
                "CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE"
            )
        
        if not collection_name:
            raise ValueError(
                "Collection name must be provided. It should be read from config.yaml, "
                "not from environment variables."
            )
        
        return cls(
            api_key=api_key,
            tenant=tenant,
            database=database,
            collection_name=collection_name,
            embedding_model=embedding_model or "BAAI/bge-base-en-v1.5"
        )


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create LLMConfig from environment variables."""
        provider = os.getenv('LLM_PROVIDER', 'gemini').lower()
        model = os.getenv('LLM_MODEL', 'gemini-2.5-flash')
        max_tokens_str = os.getenv('LLM_MAX_TOKENS', '').strip()
        max_tokens = int(max_tokens_str) if max_tokens_str else None
        temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))
        
        if provider == "gemini":
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not set. Please set it as an environment variable or in a .env file.\n"
                    "You can create a .env file in the project root or backend/qa/ directory with:\n"
                    "  GEMINI_API_KEY=your_gemini_api_key_here"
                )
            return cls(
                provider=provider,
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature
            )
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
    enable_verification: bool = False  # Whether to enable verification
    
    @classmethod
    def from_file(cls, config_path: str = "config.yaml") -> 'RAGConfig':
        """Load configuration from YAML file and environment variables."""
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
        else:
            yaml_config = {}
        
        retrieval_config = yaml_config.get('retrieval', {})
        dense_config = retrieval_config.get('dense', {})
        
        top_k = dense_config.get('top_k', 20)
        embedding_model = dense_config.get('model_name', 'BAAI/bge-base-en-v1.5')
        
        rerank_config = yaml_config.get('reranking', {})
        rerank_max_length = rerank_config.get('max_length', int(os.getenv('RERANK_MAX_LENGTH', '512')))
        enable_reranking = rerank_config.get('enable', os.getenv('ENABLE_RERANKING', 'false').lower() == 'true')
        
        llm_config = yaml_config.get('llm', {})
        llm_max_tokens_raw = llm_config.get('max_tokens', os.getenv('LLM_MAX_TOKENS', '').strip())
        if llm_max_tokens_raw is None or (isinstance(llm_max_tokens_raw, str) and not llm_max_tokens_raw.strip()):
            llm_max_tokens = None
        else:
            llm_max_tokens = int(llm_max_tokens_raw)
        llm_temperature = llm_config.get('temperature', float(os.getenv('LLM_TEMPERATURE', '0.1')))
        
        verification_config = yaml_config.get('verification', {})
        enable_verification = verification_config.get('enable', os.getenv('ENABLE_VERIFICATION', 'true').lower() == 'true')
        
        # Get collection name from YAML config (required)
        import logging
        logger = logging.getLogger(__name__)
        
        chroma_config = yaml_config.get('chroma', {})
        if not chroma_config:
            logger.error(f"Config file structure: {list(yaml_config.keys())}")
            raise ValueError(
                "No 'chroma' section found in config.yaml. "
                "Please add a 'chroma' section with 'collection_name' to your config.yaml file."
            )
        
        collection_name = chroma_config.get('collection_name')
        
        # Handle empty string or None
        if not collection_name or (isinstance(collection_name, str) and not collection_name.strip()):
            logger.error(f"Chroma config keys: {list(chroma_config.keys())}")
            logger.error(f"Collection name value: {repr(collection_name)}")
            raise ValueError(
                "Collection name not found or empty in config.yaml. "
                "Please ensure 'chroma.collection_name' is set to a non-empty value in your config.yaml file."
            )
        
        # Strip whitespace if it's a string
        if isinstance(collection_name, str):
            collection_name = collection_name.strip()
        
        # Log which collection name is being used
        logger.info(f"Using ChromaDB collection name from config.yaml: {collection_name}")
        
        chroma = ChromaDBConfig.from_env(
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        llm = LLMConfig.from_env()
        llm.max_tokens = llm_max_tokens
        llm.temperature = llm_temperature
        
        return cls(
            chroma=chroma,
            llm=llm,
            top_k=top_k,
            rerank_max_length=rerank_max_length,
            enable_reranking=enable_reranking,
            enable_verification=enable_verification
        )
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create RAGConfig from environment variables only."""
        # Try to read collection name from default config.yaml first
        import logging
        logger = logging.getLogger(__name__)
        
        collection_name = None
        default_config_path = Path("config.yaml")
        if default_config_path.exists():
            try:
                with open(default_config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                chroma_config = yaml_config.get('chroma', {})
                collection_name = chroma_config.get('collection_name')
                if collection_name:
                    logger.info(f"Using ChromaDB collection name from config.yaml: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not read collection name from config.yaml: {e}")
        
        if not collection_name:
            raise ValueError(
                "Collection name not found. Please ensure 'chroma.collection_name' is set in config.yaml. "
                "The from_env() method requires a config.yaml file with the collection name specified."
            )
        
        chroma = ChromaDBConfig.from_env(collection_name=collection_name)
        llm = LLMConfig.from_env()
        top_k = int(os.getenv('RAG_TOP_K', '20'))
        rerank_max_length = int(os.getenv('RERANK_MAX_LENGTH', '512'))
        enable_reranking = os.getenv('ENABLE_RERANKING', 'true').lower() == 'true'
        enable_verification = os.getenv('ENABLE_VERIFICATION', 'true').lower() == 'true'
        
        return cls(
            chroma=chroma,
            llm=llm,
            top_k=top_k,
            rerank_max_length=rerank_max_length,
            enable_reranking=enable_reranking,
            enable_verification=enable_verification
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

