"""Answer generation with citations from retrieved chunks."""

from typing import List, Dict
import logging
import re

try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    raise ImportError("transformers package required. Install with: pip install transformers")

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate answers from retrieved chunks with citations."""
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: Device to run model on ("auto", "cuda", "mps", or "cpu")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = self._get_device(device)
        self._load_local_model(model_name)
    
    def _load_local_model(self, model_name: str):
        """Load a local HuggingFace model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required for local models. Install with: pip install transformers")
        
        logger.info(f"Loading generator model: {model_name} on {self.device}")
        
        device_id = 0 if self.device == "cuda" else -1
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        if self.device == "cuda":
            try:
                import accelerate
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=dtype, device_map="auto",
                    trust_remote_code=True, low_cpu_mem_usage=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.pipeline = pipeline("text-generation", model=self.model, 
                                        tokenizer=self.tokenizer, torch_dtype=dtype)
                logger.info("Model loaded with device_map='auto'")
                return
            except (ImportError, Exception) as e:
                logger.warning(f"Optimized loading failed ({e}), using standard pipeline")
        
        self.pipeline = pipeline(
            "text-generation", model=model_name, device=device_id,
            torch_dtype=dtype, trust_remote_code=True,
            model_kwargs={"low_cpu_mem_usage": True}
        )
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _get_device(self, device_preference: str) -> str:
        """Get the best available device."""
        if device_preference == "auto":
            if TRANSFORMERS_AVAILABLE:
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
            return "cpu"
        return device_preference
    
    def _format_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Format prompt with retrieved chunks."""
        context = "\n\n".join([
            f"[{i+1}] {chunk['text'][:500]}..." if len(chunk['text']) > 500 else f"[{i+1}] {chunk['text']}"
            for i, chunk in enumerate(chunks[:5])
        ])

        # TO DO: PROMPT OPTIMIZATION
        
        prompt = f"""Based on the following context from SEC filings, provide a concise answer to the question. Cite specific chunks using [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def generate(self, query: str, chunks: List[Dict]) -> Dict[str, any]:
        """
        Generate answer from query and retrieved chunks.
        
        Args:
            query: User question
            chunks: Retrieved chunks with metadata
            
        Returns:
            Dict with 'answer', 'citations', and 'chunks_used'
        """
        if not chunks:
            return {
                "answer": "I couldn't find relevant information to answer this question.",
                "citations": [],
                "chunks_used": []
            }
        
        prompt = self._format_prompt(query, chunks)
        
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                return_full_text=False
            )
            answer = outputs[0]['generated_text'].strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            answer = "I encountered an error generating an answer. Please try again."
        
        citations = self._extract_citations(answer)
        chunks_used = [chunks[i-1] for i in citations if 1 <= i <= len(chunks)]
        
        return {
            "answer": answer,
            "citations": citations,
            "chunks_used": chunks_used,
            "all_chunks": chunks[:5]
        }
    
    def _extract_citations(self, text: str) -> List[int]:
        """Extract citation numbers from answer text."""
        citations = re.findall(r'\[(\d+)\]', text)
        return [int(c) for c in citations if c.isdigit()]