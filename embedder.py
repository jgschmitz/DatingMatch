import openai
import pymongo

# === CONFIGURATION ===
MONGODB_URI = "" # mongodb Atlas connection string 
DB_NAME = "DateFinder"
COLLECTION_NAME = "Singles"
OPENAI_KEY = ""  # your actual key
EMBEDDING_MODEL = "text-embedding-ada-002"

# === SETUP ===
openai.api_key = OPENAI_KEY
mongo_client = pymongo.MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

# === PROCESS ===
documents = list(collection.find({}))
for doc in documents:
    user_id = doc.get("user_id")
    bio = doc.get("bio", "")

    if not bio:
        print(f"⚠️ Skipping {user_id}, no bio.")
        continue

    response = openai.Embedding.create(
        input=bio,
        model=EMBEDDING_MODEL
    )
    embedding = response["data"][0]["embedding"]

    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"profileEmbedding": embedding}}
    )
    print(f"✅ Embedded: {user_id}")

print("🎉 All done.")
