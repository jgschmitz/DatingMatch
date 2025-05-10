{
  "fields": [
    {
      "type": "vector",
      "path": "profileEmbedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "gender"
    },
    {
      "type": "filter",
      "path": "location"
    }
  ]
}
