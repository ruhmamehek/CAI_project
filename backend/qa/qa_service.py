"""Flask service for one-turn RAG queries and responses."""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from service.utils import get_project_root, is_path_safe, get_image_mime_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env_paths = [
    Path(__file__).parent.parent.parent.resolve() / '.env',
    Path(__file__).parent.parent.resolve() / '.env',
    Path(__file__).parent.resolve() / '.env',
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break
else:
    load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).parent))

from service import RAGService, load_config, QueryRequest

app = Flask(__name__)
CORS(app)

rag_service: RAGService = None


def init_rag_service(config_path: Optional[str] = None):
    """Initialize global RAG service from config file."""
    global rag_service
    
    # Default to config.yaml in the same directory as this file (backend/qa/config.yaml)
    if config_path is None:
        config_path = str(Path(__file__).parent / 'config.yaml')
    
    try:
        config = load_config(config_path)
        rag_service = RAGService(config)
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}", exc_info=True)
        raise


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "rag-qa"})


@app.route('/query', methods=['POST'])
def query():
    """Handle RAG query."""
    if rag_service is None:
        return jsonify({"error": "RAG service not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        query_request = QueryRequest.from_dict(data)
        query_request.validate()
        
        enable_verification = data.get('enable_verification', True)
        
        # Check if auto filter determination should be enabled (default: True)
        # If filters are explicitly provided, auto-determination is disabled
        auto_determine_filters = data.get('auto_determine_filters', query_request.filters is None)
        
        # Execute RAG pipeline
        response = rag_service.query(
            query=query_request.query,
            filters=query_request.filters,
            top_k=query_request.top_k,
            enable_verification=enable_verification,
            auto_determine_filters=auto_determine_filters
        )
        
        return jsonify(response.to_dict())
    
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/retrieve', methods=['POST'])
def retrieve():
    """Retrieve chunks without generating response."""
    if rag_service is None:
        return jsonify({"error": "RAG service not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        query_text = data.get('query')
        if not query_text:
            return jsonify({"error": "Missing 'query' field in request"}), 400
        
        filters = data.get('filters')
        top_k = data.get('top_k')
        
        chunks = rag_service.retrieve(query=query_text, filters=filters, top_k=top_k)
        
        chunks_dict = [
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": chunk.score,
                "chunk_id": chunk.chunk_id
            }
            for chunk in chunks
        ]
        
        return jsonify({
            "chunks": chunks_dict,
            "num_chunks": len(chunks)
        })
    
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/retrieve/rerank', methods=['POST'])
def retrieve_with_rerank():
    """Retrieve chunks with reranking: initially retrieve 5*k, then narrow down to k after reranking."""
    if rag_service is None:
        return jsonify({"error": "RAG service not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        query_text = data.get('query')
        if not query_text:
            return jsonify({"error": "Missing 'query' field in request"}), 400
        
        filters = data.get('filters')
        top_k = data.get('top_k') or rag_service.config.top_k
        
        # Retrieve 5*k chunks initially
        initial_k = 2 * top_k
        chunks = rag_service.retrieve(query=query_text, filters=filters, top_k=initial_k)
        logger.info(f"Retrieved {len(chunks)} chunks (5*{top_k}={initial_k}) for reranking")
        
        # Rerank and narrow down to k
        chunks = rag_service.rerank_chunks(query=query_text, chunks=chunks, top_k=top_k)
        logger.info(f"Reranked to {len(chunks)} chunks")
        
        chunks_dict = [
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": chunk.score,
                "chunk_id": chunk.chunk_id
            }
            for chunk in chunks
        ]
        
        return jsonify({
            "chunks": chunks_dict,
            "num_chunks": len(chunks),
            "initial_retrieved": initial_k
        })
    
    except Exception as e:
        logger.error(f"Error retrieving chunks with reranking: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/retrieve/filter-rerank', methods=['POST'])
def retrieve_with_filter_rerank():
    """Retrieve chunks with filter reasoning and reranking: determine filters, retrieve 5*k, then narrow down to k after reranking."""
    if rag_service is None:
        return jsonify({"error": "RAG service not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        query_text = data.get('query')
        if not query_text:
            return jsonify({"error": "Missing 'query' field in request"}), 400
        
        top_k = data.get('top_k') or rag_service.config.top_k
        
        # Determine filters using filter reasoning
        filters, filter_reasoning = rag_service.determine_filters(query_text)
        logger.info(f"Auto-determined filters: {filters}, reasoning: {filter_reasoning}")
        
        # Retrieve 5*k chunks initially with determined filters
        initial_k = 5 * top_k
        chunks = rag_service.retrieve(query=query_text, filters=filters, top_k=initial_k)
        logger.info(f"Retrieved {len(chunks)} chunks (5*{top_k}={initial_k}) with filters for reranking")
        
        # Rerank and narrow down to k
        chunks = rag_service.rerank_chunks(query=query_text, chunks=chunks, top_k=top_k)
        logger.info(f"Reranked to {len(chunks)} chunks")
        
        chunks_dict = [
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": chunk.score,
                "chunk_id": chunk.chunk_id
            }
            for chunk in chunks
        ]
        
        response = {
            "chunks": chunks_dict,
            "num_chunks": len(chunks),
            "initial_retrieved": initial_k,
            "applied_filters": filters,
            "filter_reasoning": filter_reasoning
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error retrieving chunks with filter reasoning and reranking: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/retrieve/filter', methods=['POST'])
def retrieve_with_filter_reasoning():
    """Retrieve chunks with filter reasoning only (no reranking)."""
    if rag_service is None:
        return jsonify({"error": "RAG service not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        query_text = data.get('query')
        if not query_text:
            return jsonify({"error": "Missing 'query' field in request"}), 400
        
        top_k = data.get('top_k') or rag_service.config.top_k
        
        # Determine filters using filter reasoning
        filters, filter_reasoning = rag_service.determine_filters(query_text)
        logger.info(f"Auto-determined filters (no rerank): {filters}, reasoning: {filter_reasoning}")
        
        chunks = rag_service.retrieve(query=query_text, filters=filters, top_k=top_k)
        logger.info(f"Retrieved {len(chunks)} chunks with filters (no rerank)")
        
        chunks_dict = [
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": chunk.score,
                "chunk_id": chunk.chunk_id
            }
            for chunk in chunks
        ]
        
        return jsonify({
            "chunks": chunks_dict,
            "num_chunks": len(chunks),
            "applied_filters": filters,
            "filter_reasoning": filter_reasoning
        })
    
    except Exception as e:
        logger.error(f"Error retrieving chunks with filter reasoning: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/collection/info', methods=['GET'])
def collection_info():
    """Get information about the ChromaDB collection."""
    if rag_service is None:
        return jsonify({"error": "RAG service not initialized"}), 500
    
    try:
        info = rag_service.get_collection_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting collection info: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/images/<path:image_path>', methods=['GET'])
def serve_image(image_path):
    """Serve images from the data directory."""
    try:
        project_root = get_project_root()
        full_path = project_root / image_path
        
        if not is_path_safe(full_path, project_root):
            logger.warning(f"Attempted to access path outside project root: {image_path}")
            return jsonify({"error": "Invalid image path"}), 403
        
        if not full_path.exists():
            logger.warning(f"Image not found: {full_path}")
            return jsonify({"error": "Image not found"}), 404
        
        mime_type = get_image_mime_type(full_path.suffix)
        if not mime_type:
            return jsonify({"error": "Not an image file"}), 400
        
        return send_file(str(full_path), mimetype=mime_type)
    
    except Exception as e:
        logger.error(f"Error serving image {image_path}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Default to config.yaml in the same directory as this file (backend/qa/config.yaml)
    default_config = Path(__file__).parent / 'config.yaml'
    config_path = os.getenv('CONFIG_PATH', str(default_config))
    init_rag_service(config_path)
    
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
