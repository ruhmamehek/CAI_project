"""Flask service for one-turn RAG queries and responses."""

import os
import sys
import logging
from pathlib import Path
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


def init_rag_service(config_path: str = "config.yaml"):
    """Initialize global RAG service from config file."""
    global rag_service
    
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
        
        response = rag_service.query(
            query=query_request.query,
            filters=query_request.filters,
            top_k=query_request.top_k,
            enable_verification=enable_verification
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
    default_config = Path(__file__).parent.parent.parent / 'config.yaml'
    config_path = os.getenv('CONFIG_PATH', str(default_config))
    init_rag_service(config_path)
    
    port = int(os.getenv('PORT', 8888))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
