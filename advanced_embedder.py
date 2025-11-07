"""
Advanced Embedding Manager for Dating Match POC
Supports multiple embedding models and providers for comparison analysis.
"""

from openai import OpenAI
import voyageai
import pymongo
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    OPENAI = "openai"
    VOYAGE = "voyage"

@dataclass
class EmbeddingConfig:
    provider: EmbeddingProvider
    model: str
    dimensions: int
    description: str
    
class AdvancedEmbeddingManager:
    """
    Manages multiple embedding models for comparing match outcomes.
    Supports OpenAI and VoyageAI models.
    """
    
    SUPPORTED_MODELS = {
        "openai_ada_002": EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-ada-002", 
            dimensions=1536,
            description="OpenAI Ada-002 - General purpose, reliable baseline"
        ),
        "openai_3_small": EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            dimensions=1536,
            description="OpenAI v3 Small - Improved performance over Ada-002"
        ),
        "openai_3_large": EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-large",
            dimensions=3072,
            description="OpenAI v3 Large - Highest quality OpenAI embeddings"
        ),
        "voyage_3_large": EmbeddingConfig(
            provider=EmbeddingProvider.VOYAGE,
            model="voyage-3-large",
            dimensions=1024,
            description="VoyageAI v3 Large - Optimized for semantic similarity"
        ),
        "voyage_3": EmbeddingConfig(
            provider=EmbeddingProvider.VOYAGE,
            model="voyage-3",
            dimensions=1024,
            description="VoyageAI v3 - Balanced performance and speed"
        )
    }
    
    def __init__(self, openai_key: str = "", voyage_key: str = "", 
                 mongodb_uri: str = "", db_name: str = "DateFinder", 
                 collection_name: str = "Singles"):
        """
        Initialize the advanced embedding manager.
        """
        self.openai_key = openai_key
        self.voyage_key = voyage_key
        
        # Initialize API clients
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
        else:
            self.openai_client = None
        if voyage_key:
            self.voyage_client = voyageai.Client(api_key=voyage_key)
        else:
            self.voyage_client = None
            
        # Initialize MongoDB connection
        if mongodb_uri:
            self.mongo_client = pymongo.MongoClient(mongodb_uri)
            self.collection = self.mongo_client[db_name][collection_name]
        else:
            self.mongo_client = None
            self.collection = None
            
    def get_embedding(self, text: str, model_name: str) -> List[float]:
        """
        Get embedding for text using specified model.
        
        Args:
            text: Text to embed
            model_name: Name of the model to use (key in SUPPORTED_MODELS)
            
        Returns:
            List of embedding values
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available: {list(self.SUPPORTED_MODELS.keys())}")
            
        config = self.SUPPORTED_MODELS[model_name]
        
        try:
            if config.provider == EmbeddingProvider.OPENAI:
                return self._get_openai_embedding(text, config.model)
            elif config.provider == EmbeddingProvider.VOYAGE:
                return self._get_voyage_embedding(text, config.model)
        except Exception as e:
            logger.error(f"Error getting embedding with {model_name}: {e}")
            raise
            
    def _get_openai_embedding(self, text: str, model: str) -> List[float]:
        """Get embedding from OpenAI."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        response = self.openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
        
    def _get_voyage_embedding(self, text: str, model: str) -> List[float]:
        """Get embedding from VoyageAI."""
        if not self.voyage_client:
            raise ValueError("VoyageAI client not initialized")
            
        response = self.voyage_client.embed(
            texts=[text],
            model=model
        )
        return response.embeddings[0]
        
    def embed_all_profiles(self, model_names: List[str], 
                          force_recompute: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Generate embeddings for all profiles using specified models.
        
        Args:
            model_names: List of model names to use
            force_recompute: Whether to recompute existing embeddings
            
        Returns:
            Dictionary with stats on embedding generation
        """
        if not self.collection:
            raise ValueError("MongoDB collection not initialized")
            
        stats = {}
        
        for model_name in model_names:
            logger.info(f"Starting embedding generation with {model_name}")
            model_stats = {"processed": 0, "skipped": 0, "errors": 0}
            
            # Create field name for this model
            embedding_field = f"embedding_{model_name}"
            
            # Find documents that need embedding
            query = {embedding_field: {"$exists": False}} if not force_recompute else {}
            documents = list(self.collection.find(query))
            
            logger.info(f"Found {len(documents)} documents to process with {model_name}")
            
            for doc in documents:
                try:
                    user_id = doc.get("user_id")
                    bio = doc.get("bio", "")
                    
                    if not bio:
                        logger.warning(f"Skipping {user_id}, no bio")
                        model_stats["skipped"] += 1
                        continue
                        
                    # Get embedding
                    embedding = self.get_embedding(bio, model_name)
                    
                    # Store embedding in database
                    self.collection.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {embedding_field: embedding}}
                    )
                    
                    model_stats["processed"] += 1
                    logger.info(f"✅ Embedded {user_id} with {model_name}")
                    
                    # Small delay to respect rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing {user_id} with {model_name}: {e}")
                    model_stats["errors"] += 1
                    
            stats[model_name] = model_stats
            logger.info(f"Completed {model_name}: {model_stats}")
            
        return stats
        
    def compare_embeddings(self, text1: str, text2: str, 
                          model_names: List[str]) -> Dict[str, float]:
        """
        Compare similarity of two texts across different models.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            model_names: List of models to use for comparison
            
        Returns:
            Dictionary mapping model names to cosine similarities
        """
        similarities = {}
        
        for model_name in model_names:
            try:
                emb1 = self.get_embedding(text1, model_name)
                emb2 = self.get_embedding(text2, model_name)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(emb1, emb2)
                similarities[model_name] = similarity
                
            except Exception as e:
                logger.error(f"Error comparing with {model_name}: {e}")
                similarities[model_name] = 0.0
                
        return similarities
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics on embeddings stored in the database."""
        if not self.collection:
            raise ValueError("MongoDB collection not initialized")
            
        stats = {}
        total_docs = self.collection.count_documents({})
        
        for model_name in self.SUPPORTED_MODELS.keys():
            embedding_field = f"embedding_{model_name}"
            embedded_count = self.collection.count_documents({
                embedding_field: {"$exists": True}
            })
            
            stats[model_name] = {
                "embedded": embedded_count,
                "total": total_docs,
                "percentage": (embedded_count / total_docs * 100) if total_docs > 0 else 0
            }
            
        return stats
        
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their configurations."""
        return {
            name: {
                "provider": config.provider.value,
                "model": config.model,
                "dimensions": config.dimensions,
                "description": config.description
            }
            for name, config in self.SUPPORTED_MODELS.items()
        }

if __name__ == "__main__":
    # Demo usage
    print("🚀 Advanced Embedding Manager Demo")
    print("=" * 50)
    
    # Initialize manager (keys would come from environment in real usage)
    manager = AdvancedEmbeddingManager()
    
    # List available models
    models = manager.list_available_models()
    print(f"📋 Available Models ({len(models)}):")
    for name, info in models.items():
        print(f"  • {name}: {info['description']}")
        print(f"    Provider: {info['provider']}, Dims: {info['dimensions']}")
    
    print("\\n✨ Ready for embedding generation and comparison!")