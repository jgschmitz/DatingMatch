import pymongo
import openai
import numpy as np

# === CONFIG ===
openai.api_key = "" your openAI key
MONGODB_URI = "" your MongoDB Atlas connection string
DB_NAME = "DateFinder"
COLLECTION_NAME = "Singles"
INDEX_NAME = "match_index"

# === SETUP ===
mongo_client = pymongo.MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

# === INPUT ===
user_id = "u001"  # change this to test others
print(f"\n🔍 Finding matches for {user_id}...\n")

# === GET QUERY USER ===
query_user = collection.find_one({"user_id": user_id})
if not query_user:
    print("❌ User not found.")
    exit()

query_embedding = query_user.get("profileEmbedding")
if not query_embedding:
    print("❌ No embedding found for user.")
    exit()

target_gender = "female" if query_user["gender"] == "male" else "male"
location = query_user.get("location", "New York")

# === VECTOR SEARCH PIPELINE ===
query_pipeline = [
    {
        "$vectorSearch": {
            "index": INDEX_NAME,
            "path": "profileEmbedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": 5,
            "filter": {
                "gender": target_gender,
                "location": location
            }
        }
    },
    {
        "$project": {
            "user_id": 1,
            "name": 1,
            "gender": 1,
            "location": 1,
            "bio": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]

# === RUN AGGREGATION ===
results = list(collection.aggregate(query_pipeline))

if not results:
    print("⚠️ No matches found.")
else:
    print(f"💘 Top matches for {query_user['name']} ({query_user['gender']}):\n")
    for match in results:
        print(f"- {match['name']} ({match['gender']}), {match['location']}")
        print(f"  📖 {match['bio']}")
        print(f"  🔢 Score: {round(match['score'], 3)}\n")

