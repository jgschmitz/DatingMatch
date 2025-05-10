results = list(collection.find({
    "bio": {"$regex": "hiking", "$options": "i"}
}))
for r in results:
    print(f"{r['name']} ({r['location']}) — {r['bio']}")
