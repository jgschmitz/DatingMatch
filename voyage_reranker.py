"""
VoyageAI Reranker for Dating Match POC
Proper implementation of VoyageAI reranking for match optimization.
"""

import voyageai
import pymongo
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoyageReranker:
    """
    VoyageAI Reranker for improving match quality by reordering initial results.
    """
    
    def __init__(self, voyage_key: str, mongodb_uri: str = "", 
                 db_name: str = "DateFinder", collection_name: str = "Singles"):
        """Initialize the VoyageAI reranker."""
        self.voyage_client = voyageai.Client(api_key=voyage_key)
        
        # Initialize MongoDB connection if provided
        if mongodb_uri:
            self.mongo_client = pymongo.MongoClient(mongodb_uri)
            self.collection = self.mongo_client[db_name][collection_name]
        else:
            self.mongo_client = None
            self.collection = None
    
    def rerank_matches(self, query_text: str, documents: List[Dict[str, Any]], 
                      model: str = "rerank-2.5", top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Rerank match results using VoyageAI reranker.
        
        Args:
            query_text: The query text (user's bio/preferences)
            documents: List of potential matches with their data
            model: Reranker model to use
            top_k: Number of top results to return
            
        Returns:
            List of reranked documents with relevance scores
        """
        if not documents:
            return []
            
        # Prepare documents for reranking (extract text content)
        doc_texts = []
        for doc in documents:
            # Create a comprehensive text representation of each potential match
            doc_text = self._create_match_text(doc)
            doc_texts.append(doc_text)
        
        try:
            # Use VoyageAI reranker
            rerank_result = self.voyage_client.rerank(
                query=query_text,
                documents=doc_texts,
                model=model,
                top_k=min(top_k, len(documents))
            )
            
            # Reorder original documents based on reranking results
            reranked_docs = []
            for result in rerank_result.results:
                original_doc = documents[result.index].copy()
                original_doc['rerank_score'] = result.relevance_score
                original_doc['rerank_index'] = result.index
                reranked_docs.append(original_doc)
                
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Return original documents if reranking fails
            return documents[:top_k]
    
    def _create_match_text(self, match_doc: Dict[str, Any]) -> str:
        """
        Create a comprehensive text representation of a potential match.
        """
        text_parts = []
        
        # Add bio (most important)
        if 'bio' in match_doc:
            text_parts.append(f"Bio: {match_doc['bio']}")
        
        # Add interests
        if 'interests' in match_doc:
            interests = match_doc['interests']
            if isinstance(interests, list):
                text_parts.append(f"Interests: {', '.join(interests)}")
            else:
                text_parts.append(f"Interests: {interests}")
        
        # Add basic demographics
        if 'age' in match_doc:
            text_parts.append(f"Age: {match_doc['age']}")
        
        if 'location' in match_doc:
            text_parts.append(f"Location: {match_doc['location']}")
            
        # Add name for context
        if 'name' in match_doc:
            text_parts.append(f"Name: {match_doc['name']}")
        
        return " | ".join(text_parts)
    
    def enhanced_match_with_rerank(self, user_id: str, embedding_field: str = "profileEmbedding",
                                  initial_limit: int = 20, final_top_k: int = 5,
                                  rerank_model: str = "rerank-2.5") -> List[Dict[str, Any]]:
        """
        Perform enhanced matching: vector search + reranking.
        
        Args:
            user_id: ID of the user seeking matches
            embedding_field: Field containing the embeddings
            initial_limit: Number of initial vector search results
            final_top_k: Number of final reranked results
            rerank_model: Reranker model to use
            
        Returns:
            List of reranked top matches
        """
        if not self.collection:
            raise ValueError("MongoDB collection not initialized")
        
        # Get query user
        query_user = self.collection.find_one({"user_id": user_id})
        if not query_user:
            raise ValueError(f"User {user_id} not found")
        
        query_embedding = query_user.get(embedding_field)
        if not query_embedding:
            raise ValueError(f"No embedding found for user {user_id} in field {embedding_field}")
        
        # Determine target gender
        target_gender = "female" if query_user["gender"] == "male" else "male"
        location = query_user.get("location")
        
        # Build vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "match_index",
                    "path": embedding_field,
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": initial_limit,
                    "filter": {
                        "gender": target_gender
                    }
                }
            },
            {
                "$project": {
                    "user_id": 1,
                    "name": 1,
                    "gender": 1,
                    "age": 1,
                    "location": 1,
                    "bio": 1,
                    "interests": 1,
                    "vector_score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        # Add location filter if specified
        if location:
            pipeline[0]["$vectorSearch"]["filter"]["location"] = location
        
        # Execute vector search
        initial_matches = list(self.collection.aggregate(pipeline))
        
        if not initial_matches:
            logger.info("No initial matches found")
            return []
        
        logger.info(f"Found {len(initial_matches)} initial matches, reranking to top {final_top_k}")
        
        # Create query text for reranking
        query_text = self._create_query_text(query_user)
        
        # Rerank the results
        reranked_matches = self.rerank_matches(
            query_text=query_text,
            documents=initial_matches,
            model=rerank_model,
            top_k=final_top_k
        )
        
        return reranked_matches
    
    def _create_query_text(self, user_doc: Dict[str, Any]) -> str:
        """
        Create query text from user profile for reranking.
        """
        query_parts = []
        
        if 'bio' in user_doc:
            query_parts.append(user_doc['bio'])
        
        if 'interests' in user_doc:
            interests = user_doc['interests']
            if isinstance(interests, list):
                query_parts.append(f"Interested in: {', '.join(interests)}")
            else:
                query_parts.append(f"Interested in: {interests}")
        
        return " ".join(query_parts)
    
    def compare_with_without_rerank(self, user_id: str, embedding_field: str = "profileEmbedding",
                                   top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compare results with and without reranking.
        
        Returns:
            Dictionary with 'vector_only' and 'with_rerank' results
        """
        if not self.collection:
            raise ValueError("MongoDB collection not initialized")
        
        # Get vector-only results
        query_user = self.collection.find_one({"user_id": user_id})
        if not query_user:
            raise ValueError(f"User {user_id} not found")
        
        query_embedding = query_user.get(embedding_field)
        target_gender = "female" if query_user["gender"] == "male" else "male"
        location = query_user.get("location")
        
        # Vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "match_index",
                    "path": embedding_field,
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": top_k,
                    "filter": {"gender": target_gender}
                }
            },
            {
                "$project": {
                    "user_id": 1,
                    "name": 1,
                    "gender": 1,
                    "age": 1,
                    "location": 1,
                    "bio": 1,
                    "interests": 1,
                    "vector_score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        if location:
            pipeline[0]["$vectorSearch"]["filter"]["location"] = location
        
        vector_only_results = list(self.collection.aggregate(pipeline))
        
        # Get reranked results
        reranked_results = self.enhanced_match_with_rerank(
            user_id=user_id,
            embedding_field=embedding_field,
            initial_limit=20,
            final_top_k=top_k
        )
        
        return {
            'vector_only': vector_only_results,
            'with_rerank': reranked_results
        }

def run_rerank_demo(voyage_key: str, user_id: str = "u001"):
    """
    Demo function showing reranking in action.
    """
    print("🚀 VoyageAI Reranker Demo")
    print("=" * 40)
    
    # Initialize reranker (MongoDB not required for basic demo)
    reranker = VoyageReranker(voyage_key=voyage_key)
    
    # Sample documents for demo
    query = "Software engineer who loves board games and good coffee"
    
    sample_docs = [
        {
            "name": "Alice",
            "bio": "Marketing exec who loves yoga and indie films",
            "interests": ["yoga", "movies", "travel"],
            "age": 29,
            "location": "New York"
        },
        {
            "name": "Sarah", 
            "bio": "Software developer who enjoys chess and craft coffee",
            "interests": ["programming", "chess", "coffee"],
            "age": 27,
            "location": "San Francisco"
        },
        {
            "name": "Emma",
            "bio": "Data scientist who enjoys board games and live jazz",
            "interests": ["data", "games", "music"],
            "age": 30,
            "location": "Seattle"
        }
    ]
    
    print(f"Query: {query}")
    print(f"\\nReranking {len(sample_docs)} potential matches...")
    
    # Rerank with different models
    for model in ["rerank-2.5", "rerank-2"]:
        print(f"\\n--- Using {model} ---")
        try:
            reranked = reranker.rerank_matches(
                query_text=query,
                documents=sample_docs,
                model=model,
                top_k=3
            )
            
            for i, match in enumerate(reranked, 1):
                score = match.get('rerank_score', 0)
                print(f"{i}. {match['name']} (Score: {score:.3f})")
                print(f"   {match['bio']}")
                
        except Exception as e:
            print(f"Error with {model}: {e}")
    
    print("\\n✨ Reranking demo complete!")

if __name__ == "__main__":
    # Demo would require a VoyageAI key
    print("VoyageAI Reranker ready!")
    print("Use run_rerank_demo(voyage_key) to test reranking")