import pymongo
import openai

# === CONFIGURATION ===
MONGODB_URI = "your_connection_string"
DB_NAME = "DateFinder"
COLLECTION_NAME = "Singles"
OPENAI_KEY = "your_openai_key"
EMBEDDING_MODEL = "text-embedding-ada-002"

# === SETUP ===
openai.api_key = OPENAI_KEY
mongo_client = pymongo.MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

# === FUNCTION ===
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response['data'][0]['embedding']

# === CLI INPUT ===
user_input = input("🔍 Describe your ideal match (e.g. 'loves hiking and music'): ").strip()
if not user_input:
    print("❌ No input provided.")
    exit()

query_vector = get_embedding(user_input)

# === VECTOR SEARCH PIPELINE ===
pipeline = [
    {
        "$vectorSearch": {
            "index": "match_index",
            "path": "profileEmbedding",
            "queryVector": query_vector,
            "numCandidates": 100,
            "limit": 5
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

# === RUN AND PRINT MATCHES ===
matches = list(collection.aggregate(pipeline))
print("\n💘 Top matches:\n")
for match in matches:
    print(f"- {match['name']} ({match['gender']}), {match['location']}")
    print(f"  📖 {match['bio']}")
    print(f"  🔢 Score: {round(match['score'], 3)}\n")

