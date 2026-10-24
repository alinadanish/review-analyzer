from sentence_transformers import SentenceTransformer

documents = [
    "My order arrived three days late.",
    "The package came quickly and was easy to track.",
    "The material feels sturdy and well made.",
    "The product broke after only two days.",
    "Good value for the price.",
]

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# convert each document into an embedding
document_embeddings = model.encode(
    documents,
    normalize_embeddings=True,
)

def search(query):
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    #convert the user query into an embedding 
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    #an empty list to hold each document's similarity score.
    similarities = []
    ranked_results = []
    #compare the query with each document embedding
    for index, document_embedding in enumerate(document_embeddings):
        # Because the embeddings are normalized, @ calculates cosine similarity.
        similarity = float(document_embedding @ query_embedding)
        ranked_results.append((similarity, documents[index]))
        # Add this document's similarity score to our list.
        similarities.append(similarity)

    # Find the position of the highest similarity score.
    # best_index = similarities.index(max(similarities))
    # # Use that position to return the most relevant document.
    # return documents[best_index]
    ranked_results.sort(key=lambda result: result[0], reverse=True)
    return ranked_results[:3]


# This part runs when we start this file directly with Python.
if __name__ == "__main__":
    # Ask the user to type a query.
    user_query = input("Enter a search query: ")

    # Search the documents and save the best match.
    ranked_results = search(user_query)

    # Show the most relevant document.
    print(f"top 3 matching docs: {ranked_results}")