results = list(collection.find({
    "gender": "male",
    "location": "Chicago"
}))
for r in results:
    print(r["name"], "-", r["bio"])
