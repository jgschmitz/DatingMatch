"""
Enhanced Dating Match POC Demo
Demonstrates the impact of different embedding models, reranking, and business logic.
Addresses customer feedback on understanding general-purpose embedding models.
"""

import json
import time
from typing import Dict, List, Any
import numpy as np
from advanced_embedder import AdvancedEmbeddingManager
from voyage_reranker import VoyageReranker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchAnalytics:
    """Analytics for comparing match outcomes across different approaches."""
    
    def __init__(self):
        self.results = {}
    
    def calculate_match_diversity(self, matches: List[Dict]) -> float:
        """Calculate diversity of matches based on interests overlap."""
        if len(matches) < 2:
            return 0.0
        
        all_interests = []
        for match in matches:
            if 'interests' in match and isinstance(match['interests'], list):
                all_interests.extend(match['interests'])
        
        unique_interests = len(set(all_interests))
        total_interests = len(all_interests)
        
        return unique_interests / total_interests if total_interests > 0 else 0.0
    
    def analyze_age_distribution(self, matches: List[Dict]) -> Dict[str, float]:
        """Analyze age distribution in matches."""
        ages = [match.get('age', 0) for match in matches if 'age' in match]
        if not ages:
            return {"mean": 0, "std": 0, "range": 0}
        
        return {
            "mean": np.mean(ages),
            "std": np.std(ages),
            "range": max(ages) - min(ages)
        }

class EnhancedMatchDemo:
    """
    Main demo class showcasing enhanced matching capabilities.
    """
    
    def __init__(self, openai_key: str = "", voyage_key: str = "", 
                 mongodb_uri: str = ""):
        """Initialize the demo with API keys."""
        self.openai_key = openai_key
        self.voyage_key = voyage_key
        self.mongodb_uri = mongodb_uri
        
        # Initialize components
        if openai_key or voyage_key:
            self.embedder = AdvancedEmbeddingManager(
                openai_key=openai_key,
                voyage_key=voyage_key,
                mongodb_uri=mongodb_uri
            )
        else:
            self.embedder = None
            
        if voyage_key:
            self.reranker = VoyageReranker(
                voyage_key=voyage_key,
                mongodb_uri=mongodb_uri
            )
        else:
            self.reranker = None
            
        self.analytics = MatchAnalytics()
    
    def demo_model_comparison(self, test_user_id: str = "u001") -> Dict[str, Any]:
        """
        Demo: How different embedding models impact match outcomes.
        Addresses customer feedback point (i).
        """
        print("\\n🧠 DEMO 1: Embedding Model Impact Analysis")
        print("=" * 60)
        
        if not self.embedder:
            print("⚠️ No embedding manager available (missing API keys)")
            return {}
        
        results = {}
        
        # Test models to compare
        models_to_test = [
            "openai_ada_002",
            "openai_3_small", 
            "voyage_3_large"
        ]
        
        print(f"Comparing embedding models for user: {test_user_id}")
        print(f"Models: {models_to_test}")
        
        # For each model, get top matches (simulated since we need MongoDB)
        for model in models_to_test:
            print(f"\\n--- Testing {model} ---")
            
            config = self.embedder.SUPPORTED_MODELS.get(model)
            if config:
                print(f"📋 {config.description}")
                print(f"   Provider: {config.provider.value}")
                print(f"   Dimensions: {config.dimensions}")
                
                # In a real scenario, this would query the database
                # For demo purposes, we'll simulate different results
                simulated_matches = self._simulate_matches_for_model(model)
                
                # Analyze results
                diversity = self.analytics.calculate_match_diversity(simulated_matches)
                age_stats = self.analytics.analyze_age_distribution(simulated_matches)
                
                results[model] = {
                    "matches": simulated_matches,
                    "diversity_score": diversity,
                    "age_distribution": age_stats,
                    "model_config": {
                        "provider": config.provider.value,
                        "dimensions": config.dimensions,
                        "description": config.description
                    }
                }
                
                print(f"   📊 Diversity Score: {diversity:.3f}")
                print(f"   👥 Age Range: {age_stats['range']:.1f} years")
                
        return results
    
    def demo_reranking_impact(self, test_query: str = None) -> Dict[str, Any]:
        """
        Demo: How blending vector, filters, and reranking impacts outcomes.
        Addresses customer feedback points (ii) and (iii).
        """
        print("\\n🔄 DEMO 2: Reranking & Business Logic Impact")
        print("=" * 60)
        
        if not self.reranker:
            print("⚠️ No reranker available (missing VoyageAI key)")
            return {}
        
        # Use default query if none provided
        if not test_query:
            test_query = "Software engineer who loves board games and good coffee"
        
        print(f"Query: {test_query}")
        
        # Sample candidate matches (simulated database results)
        candidates = [
            {
                "name": "Alice",
                "bio": "Marketing exec who loves yoga and indie films",
                "interests": ["yoga", "movies", "travel"],
                "age": 29,
                "location": "New York",
                "vector_score": 0.85
            },
            {
                "name": "Sarah",
                "bio": "Software developer who enjoys chess and craft coffee", 
                "interests": ["programming", "chess", "coffee"],
                "age": 27,
                "location": "San Francisco",
                "vector_score": 0.82
            },
            {
                "name": "Emma",
                "bio": "Data scientist who enjoys board games and live jazz",
                "interests": ["data", "games", "music"],
                "age": 30,
                "location": "Seattle",
                "vector_score": 0.88
            },
            {
                "name": "Lisa",
                "bio": "Designer who loves photography and hiking",
                "interests": ["design", "photography", "hiking"],
                "age": 26,
                "location": "Portland",
                "vector_score": 0.79
            },
            {
                "name": "Maya",
                "bio": "Teacher who enjoys reading and board game nights",
                "interests": ["books", "games", "education"],
                "age": 28,
                "location": "Austin",
                "vector_score": 0.84
            }
        ]
        
        print(f"\\nInitial candidates (Vector Search): {len(candidates)}")
        for i, candidate in enumerate(candidates, 1):
            print(f"{i}. {candidate['name']} - Score: {candidate['vector_score']:.3f}")
            print(f"   {candidate['bio']}")
        
        # Apply reranking
        print("\\n--- Applying VoyageAI Reranking ---")
        try:
            reranked = self.reranker.rerank_matches(
                query_text=test_query,
                documents=candidates,
                model="rerank-2.5",
                top_k=3
            )
            
            print("\\nReranked Results:")
            for i, match in enumerate(reranked, 1):
                vector_score = match.get('vector_score', 0)
                rerank_score = match.get('rerank_score', 0)
                print(f"{i}. {match['name']}")
                print(f"   Vector: {vector_score:.3f} → Rerank: {rerank_score:.3f}")
                print(f"   {match['bio']}")
                
            # Calculate rank changes
            rank_changes = self._analyze_rank_changes(candidates, reranked)
            
            return {
                "original_results": candidates,
                "reranked_results": reranked,
                "rank_changes": rank_changes,
                "rerank_model": "rerank-2.5"
            }
            
        except Exception as e:
            print(f"❌ Reranking failed: {e}")
            return {"error": str(e)}
    
    def demo_business_logic_boosting(self) -> Dict[str, Any]:
        """
        Demo: Business logic and boosting strategies.
        Shows how additional factors can influence match quality.
        """
        print("\\n💼 DEMO 3: Business Logic & Boosting")
        print("=" * 60)
        
        # Sample user profile
        user_profile = {
            "name": "Alex",
            "age": 30,
            "location": "New York",
            "interests": ["coding", "board games", "coffee"],
            "bio": "Software engineer who loves board games and good coffee"
        }
        
        # Sample matches with various attributes
        matches = [
            {
                "name": "Sarah",
                "age": 27,
                "location": "New York",  # Same location
                "interests": ["programming", "chess", "coffee"],  # Shared interest: coffee
                "bio": "Software developer who enjoys chess and craft coffee",
                "base_score": 0.82
            },
            {
                "name": "Emma", 
                "age": 30,  # Same age
                "location": "Seattle",
                "interests": ["data", "games", "music"],  # Shared interest: games
                "bio": "Data scientist who enjoys board games and live jazz",
                "base_score": 0.88
            },
            {
                "name": "Alice",
                "age": 35,  # Age difference: 5 years
                "location": "Los Angeles",
                "interests": ["yoga", "movies", "travel"],  # No shared interests
                "bio": "Marketing exec who loves yoga and indie films",
                "base_score": 0.85
            }
        ]
        
        print("User Profile:")
        print(f"  {user_profile['name']}, {user_profile['age']}, {user_profile['location']}")
        print(f"  Interests: {user_profile['interests']}")
        
        print("\\nApplying Business Logic Boosters:")
        
        boosted_matches = []
        for match in matches:
            boosted_score = match['base_score']
            boosts = []
            
            # Age proximity boost
            age_diff = abs(user_profile['age'] - match['age'])
            if age_diff <= 3:
                age_boost = 0.05
                boosted_score += age_boost
                boosts.append(f"Age (+{age_boost:.2f})")
            
            # Location proximity boost
            if user_profile['location'] == match['location']:
                location_boost = 0.08
                boosted_score += location_boost
                boosts.append(f"Location (+{location_boost:.2f})")
            
            # Shared interests boost
            shared_interests = set(user_profile['interests']) & set(match['interests'])
            if shared_interests:
                interest_boost = len(shared_interests) * 0.03
                boosted_score += interest_boost
                boosts.append(f"Interests (+{interest_boost:.2f})")
            
            match_result = {
                **match,
                "boosted_score": boosted_score,
                "boosts_applied": boosts,
                "shared_interests": list(shared_interests) if shared_interests else []
            }
            boosted_matches.append(match_result)
        
        # Sort by boosted score
        boosted_matches.sort(key=lambda x: x['boosted_score'], reverse=True)
        
        print("\\nFinal Rankings with Business Logic:")
        for i, match in enumerate(boosted_matches, 1):
            print(f"{i}. {match['name']}")
            print(f"   Base: {match['base_score']:.3f} → Boosted: {match['boosted_score']:.3f}")
            print(f"   Boosts: {', '.join(match['boosts_applied']) if match['boosts_applied'] else 'None'}")
            if match['shared_interests']:
                print(f"   Shared: {match['shared_interests']}")
            print()
        
        return {
            "user_profile": user_profile,
            "boosted_matches": boosted_matches,
            "boost_strategies": ["Age proximity", "Location match", "Shared interests"]
        }
    
    def demo_evaluation_framework(self) -> Dict[str, Any]:
        """
        Demo: Match evaluation and labeling system.
        Shows how to assess and adjust match quality.
        """
        print("\\n📊 DEMO 4: Match Evaluation & Labeling Framework")
        print("=" * 60)
        
        # Sample match scenarios with different quality levels
        match_scenarios = [
            {
                "match_id": "m001",
                "user_query": "Love hiking and outdoor adventures",
                "match_profile": "Nature enthusiast who hikes every weekend",
                "similarity_score": 0.92,
                "predicted_quality": "High",
                "evaluation_metrics": {
                    "semantic_relevance": 0.95,
                    "interest_overlap": 0.90,
                    "personality_compatibility": 0.88
                }
            },
            {
                "match_id": "m002", 
                "user_query": "Software engineer who codes in Python",
                "match_profile": "Marketing manager who loves social events",
                "similarity_score": 0.65,
                "predicted_quality": "Low",
                "evaluation_metrics": {
                    "semantic_relevance": 0.45,
                    "interest_overlap": 0.20,
                    "personality_compatibility": 0.70
                }
            },
            {
                "match_id": "m003",
                "user_query": "Creative person who loves art and music",
                "match_profile": "Graphic designer and weekend DJ",
                "similarity_score": 0.88,
                "predicted_quality": "High",
                "evaluation_metrics": {
                    "semantic_relevance": 0.85,
                    "interest_overlap": 0.95,
                    "personality_compatibility": 0.82
                }
            }
        ]
        
        print("Match Quality Evaluation:")
        
        for scenario in match_scenarios:
            print(f"\\n--- Match {scenario['match_id']} ---")
            print(f"Query: {scenario['user_query']}")
            print(f"Match: {scenario['match_profile']}")
            print(f"Overall Score: {scenario['similarity_score']:.3f}")
            print(f"Predicted Quality: {scenario['predicted_quality']}")
            
            metrics = scenario['evaluation_metrics']
            print("Detailed Metrics:")
            print(f"  • Semantic Relevance: {metrics['semantic_relevance']:.3f}")
            print(f"  • Interest Overlap: {metrics['interest_overlap']:.3f}")
            print(f"  • Personality Compatibility: {metrics['personality_compatibility']:.3f}")
            
            # Generate quality label
            avg_metric = np.mean(list(metrics.values()))
            if avg_metric >= 0.8:
                quality_label = "Excellent Match"
            elif avg_metric >= 0.6:
                quality_label = "Good Match"
            else:
                quality_label = "Poor Match"
            
            print(f"  → Quality Label: {quality_label}")
        
        return {
            "evaluation_scenarios": match_scenarios,
            "quality_thresholds": {
                "excellent": 0.8,
                "good": 0.6,
                "poor": 0.0
            }
        }
    
    def _simulate_matches_for_model(self, model_name: str) -> List[Dict[str, Any]]:
        """Simulate different match results for different embedding models."""
        base_matches = [
            {"name": "Alice", "bio": "Marketing exec", "interests": ["yoga", "films"], "age": 29},
            {"name": "Sarah", "bio": "Software developer", "interests": ["coding", "coffee"], "age": 27},
            {"name": "Emma", "bio": "Data scientist", "interests": ["games", "jazz"], "age": 30}
        ]
        
        # Simulate model-specific variations
        if "openai" in model_name:
            # OpenAI models might focus more on semantic similarity
            return base_matches[:2] + [{"name": "Lisa", "bio": "Creative writer", "interests": ["books", "coffee"], "age": 28}]
        elif "voyage" in model_name:
            # Voyage models might have different ranking preferences  
            return [base_matches[1], base_matches[0], base_matches[2]]
        else:
            return base_matches
    
    def _analyze_rank_changes(self, original: List[Dict], reranked: List[Dict]) -> List[Dict]:
        """Analyze how rankings changed after reranking."""
        changes = []
        
        # Create mapping of original positions
        original_positions = {doc['name']: i for i, doc in enumerate(original)}
        
        for new_pos, doc in enumerate(reranked):
            name = doc['name']
            old_pos = original_positions.get(name, -1)
            
            if old_pos != -1:
                position_change = old_pos - new_pos  # Positive = moved up
                changes.append({
                    "name": name,
                    "old_position": old_pos + 1,  # 1-indexed
                    "new_position": new_pos + 1,
                    "position_change": position_change
                })
        
        return changes
    
    def run_full_demo(self) -> Dict[str, Any]:
        """Run all demos and return comprehensive results."""
        print("🚀 ENHANCED DATING MATCH POC - FULL DEMO")
        print("=" * 80)
        print("Addressing customer feedback on embedding models and match optimization")
        print("=" * 80)
        
        all_results = {}
        
        # Run all demos
        try:
            all_results["model_comparison"] = self.demo_model_comparison()
            all_results["reranking_impact"] = self.demo_reranking_impact() 
            all_results["business_logic"] = self.demo_business_logic_boosting()
            all_results["evaluation_framework"] = self.demo_evaluation_framework()
            
            print("\\n✅ DEMO COMPLETE!")
            print("\\n📋 Summary of POC Capabilities:")
            print("  ✓ Multiple embedding model comparison (OpenAI, VoyageAI)")
            print("  ✓ VoyageAI reranking for improved relevance")
            print("  ✓ Business logic boosting (age, location, interests)")
            print("  ✓ Match evaluation and quality labeling")
            print("  ✓ Analytics for understanding match outcomes")
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            all_results["error"] = str(e)
        
        return all_results

def main():
    """Main function to run the enhanced matching demo."""
    print("To run full demo with API keys:")
    print("demo = EnhancedMatchDemo(openai_key='your_key', voyage_key='your_key')")
    print("results = demo.run_full_demo()")
    print()
    print("Running limited demo without API keys...")
    
    # Run demo without API keys (limited functionality)
    demo = EnhancedMatchDemo()
    results = demo.run_full_demo()
    
    return results

if __name__ == "__main__":
    results = main()