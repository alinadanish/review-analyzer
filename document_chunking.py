import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

documents = [
    {
        "department": "Shipping",
        "text": """
        Standard delivery usually takes three to five business days.
        Express delivery is available for some addresses and usually takes
        one to two business days. You can view tracking information from the
        order page after the package has shipped. Delivery estimates may
        change during holidays or severe weather. If a package is delayed,
        check the tracking page first. Contact support if tracking has not
        updated for several days.
        """,
    },
    {
        "department": "Customer Support",
        "text": """
        If a product arrives damaged or defective, contact customer support
        within seven days. Include your order number and a description of the
        problem. Photos of the damage can help the support team review your
        request more quickly. Keep the packaging until support has responded,
        because they may need information from the label. You can also contact
        support to ask about a return or refund.
        """,
    },
    {
        "department": "Human Resources",
        "text": """
        New employees should complete their onboarding forms during their
        first week. The human resources team can answer questions about
        benefits, workplace policies, and time off. Employees should submit
        vacation requests through the staff portal in advance. For payroll
        questions, contact the payroll team with your employee number.
        """,
    },
]

test_cases = [
    {
        "query": "How long does delivery take and how can I track my package?",
        "relevant_ids": {0, 1},
    },
    {
        "query": "What should I do if my order is delayed during holidays?",
        "relevant_ids": {1, 2},
    },
    {
        "query": "How do I return a damaged product and get a refund?",
        "relevant_ids": {3, 4, 5},
    },
    {
        "query": "What information and photos should I send about a defective item?",
        "relevant_ids": {3, 4},
    },
    {
        "query": "How do employees request vacation and ask about benefits?",
        "relevant_ids": {6, 7},
    },
]

def split_into_chunks(text, chunk_size=35, overlap=10):
    if chunk_size < 1:
        raise ValueError("chunk size mst atleast be 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")

    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    start =0 

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += step_size

    return chunks

chunks = []
for source_document in documents:
    document_chunks = split_into_chunks(source_document["text"])
    for chunk_text in document_chunks:
        chunks.append(
            {
                "id": len(chunks),
                "text": chunk_text,
                "department": source_document["department"]
            }
        )

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_texts = [chunk["text"] for chunk in chunks]
chunk_embeddings = model.encode(
    chunk_texts,
    normalize_embeddings=True,
)

# FAISS expects vectors in NumPy's 32-bit float format.
chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)

embedding_size = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(embedding_size)
faiss_index.add(chunk_embeddings)

def search_faiss(query, department=None, top_k=3):
    if not query.strip():
        raise ValueError("seach query cant be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )
    query_embedding = np.asarray([query_embedding], dtype=np.float32)

    scores, indexes = faiss_index.search(query_embedding, len(chunks))
    ranked_results = []

    for score, index in zip(scores[0], indexes[0]):
        if index == -1:
            continue

        chunk = chunks[index]

        if department is None or (
            chunk["department"].casefold() == department.strip().casefold()
        ):
            ranked_results.append((float(score), chunk))

    return ranked_results[:top_k]

def search_cosine(query, department, top_k=3):
    if not query.strip():
        raise ValueError("Search query cannot be empty.")
    if not department.strip():
        raise ValueError("Department cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    ranked_results = []

    # Compare the query with each chunk, keeping only the requested department.
    for index, chunk in enumerate(chunks):
        if chunk["department"].lower() == department.lower():
            score = float(chunk_embeddings[index] @ query_embedding)
            ranked_results.append((score, chunk))

    # Sort manually so the closest match appears first.
    ranked_results.sort(key=lambda result: result[0], reverse=True)
    return ranked_results[:top_k]

def precision_at_k(retrieved_ids, relevant_ids, k=3): #what fraction of the top k results were relevant?
    relevant_retrieved = sum(
        chunk_id in relevant_ids
        for chunk_id in retrieved_ids[:k]
    )
    return relevant_retrieved / k

def recall_at_k(retrieved_ids, relevant_ids, k=3): #what fraction of the relevant chunks did we find?
    if not relevant_ids:
        raise ValueError("Each test query must have at least one relevant chunk ID.")
    relevant_retrieved = sum(
        chunk_id in relevant_ids
        for chunk_id in retrieved_ids[:k]
    )
    return relevant_retrieved / len(relevant_ids)

if __name__ == "__main__":
    for chunk in chunks:
        print(f"\nID: {chunk['id']} | Department: {chunk['department']}")
        print(chunk["text"])

    for test_case in test_cases:
        results = search_faiss(test_case["query"], top_k=3)
        retrieved_ids = [chunk["id"] for score, chunk in results]
        relevant_ids = test_case["relevant_ids"]

        precision = precision_at_k(retrieved_ids, relevant_ids, k=3)
        recall = recall_at_k(retrieved_ids, relevant_ids, k=3)

        print(f"\nQuery: {test_case['query']}")
        print(f"Retrieved chunk IDs: {retrieved_ids}")
        print(f"Relevant chunk IDs:  {sorted(relevant_ids)}")
        print(f"Precision@3: {precision:.2f}")
        print(f"Recall@3:    {recall:.2f}")


    # user_query = input("What do you want to search for? ")
    # requested_department = input("Which department? ")

    # faiss_results = search_faiss(
    #     user_query,
    #     requested_department,
    #     top_k=3,
    # )
    # cosine_results = search_cosine(
    #     user_query,
    #     requested_department,
    #     top_k=3,
    # )

    # # Show FAISS results, including their original department metadata.
    # print("\nFAISS results:")
    # for rank, (score, chunk) in enumerate(faiss_results, start=1):
    #     print(f"\n{rank}. Similarity: {score:.3f}")
    #     print(f"Department: {chunk['department']}")
    #     print(chunk["text"])

    # # Show the manual cosine results so you can compare their ranking.
    # print("\nManual cosine results:")
    # for rank, (score, chunk) in enumerate(cosine_results, start=1):
    #     print(f"\n{rank}. Similarity: {score:.3f}")
    #     print(f"Department: {chunk['department']}")
    #     print(chunk["text"])