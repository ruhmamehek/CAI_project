"""Flask service for one-turn RAG queries and responses."""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv() 

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from service import RAGService, load_config, QueryRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all routes (allows frontend to communicate with backend)
CORS(app)

# Global RAG service instance
rag_service: RAGService = None


def init_rag_service(config_path: str = "config.yaml"):
    """
    Initialize global RAG service from config file.
    
    Args:
        config_path: Path to config YAML file
    """
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
        
        # Create request model
        query_request = QueryRequest.from_dict(data)
        query_request.validate()
        
        # Check if verification should be enabled (default: True)
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
    """Retrieve chunks without generating response (for debugging)."""
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
        
        # Retrieve chunks
        chunks = rag_service.retrieve(query=query_text, filters=filters, top_k=top_k)
        
        # Convert chunks to dictionaries
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


if __name__ == '__main__':
    # Initialize RAG service
    # Look for config.yaml in project root (2 levels up from this file)
    default_config = Path(__file__).parent.parent.parent / 'config.yaml'
    config_path = os.getenv('CONFIG_PATH', str(default_config))
    init_rag_service(config_path)
    
    # Run Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
