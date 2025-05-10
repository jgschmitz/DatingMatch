results = list(collection.find({
    "gender": "female",
    "location": "Chicago"
}))
for r in results:
    print(r["name"], "-", r["bio"])
