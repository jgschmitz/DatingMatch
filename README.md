## 💖 DateFinder Matchmaking Demo (MongoDB + OpenAI Vector Search)
This Jupyter notebook walks you through a simple dating match scenario using:

MongoDB Atlas Vector Search

OpenAI Embeddings

Python

1. ✅ Prerequisites
MongoDB Atlas cluster (Vector Search enabled)
OpenAI API Key
Load Singles.js into Atlas
Create Vector Index (Index is included)

3. Install libraries:
```
pip install pymongo openai
```
3. 🔗 Connect to MongoDB
```
import pymongo

MONGODB_URI = "your_mongodb_uri"
DB_NAME = "DateFinder"
COLLECTION_NAME = "Singles"

mongo_client = pymongo.MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]
```
3. 🤖 Generate Embeddings with OpenAI
```
import openai

openai.api_key = "your_openai_api_key"

def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-3-small")
    return response['data'][0]['embedding']
```
4. 🧠 Embed All Profiles
```
for doc in collection.find({"profileEmbedding": {"$exists": False}}):
    embedding = get_embedding(doc["bio"])
    collection.update_one({"_id": doc["_id"]}, {"$set": {"profileEmbedding": embedding}})
```

5. 🔍 Find Matches Using Atlas Vector Search

```
query_user = collection.find_one({"user_id": "u001"})
query_embedding = query_user["profileEmbedding"]
target_gender = "female"
location = query_user["location"]

pipeline = [
    {
        "$vectorSearch": {
            "index": "match_index",
            "path": "profileEmbedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": 3,
            "filter": {
                "gender": target_gender,
                "location": location
            }
        }
    },
    {
        "$project": {
            "name": 1,
            "gender": 1,
            "location": 1,
            "bio": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]
matches = list(collection.aggregate(pipeline))
for match in matches:
    print(f"- {match['name']} ({match['gender']}), {match['location']}")
    print(f"  📖 {match['bio']}")
    print(f"  🔢 Score: {round(match['score'], 3)}\n")
```
Results should render like this!

### 🔍 Finding Matches for `u001`...

💘 **Top Matches for Alex (male):**

- **Alice** *(female)* — *New York*  
  📖 Marketing exec who loves yoga and indie films.  
  🔢 **Score:** `0.926`

- **Samantha** *(female)* — *Chicago*  
  📖 Cardio nurse who hikes on weekends and loves dogs.  
  🔢 **Score:** `0.913`

- **Emma** *(female)* — *Seattle*  
  📖 Data scientist who enjoys board games and live jazz.  
  🔢 **Score:** `0.902`

    
    
✅ Done!
You’ve now run vector search matchmaking in MongoDB! 🎉

