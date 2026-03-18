import chromadb

# Connect to your local database folder
client = chromadb.PersistentClient(path="./chroma_db")

# Get your document collection (Chroma usually defaults to 'langchain')
collection = client.get_collection("langchain")

# Fetch and print the first 2 chunks of data, including the embeddings!
data = collection.peek(limit=2)
print(data)
