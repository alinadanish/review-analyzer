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

def search(query, department, top_k=3):
    if not query.strip():
        raise ValueError("seach query cant be empty")
    if not department.strip():
        raise ValueError("Department cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # First filter by department. Similarity search will only use these indexes.
    matching_indexes = []
    for index, chunk in enumerate(chunks):
        if chunk["department"].lower() == department.lower():
            matching_indexes.append(index)

    if not matching_indexes:
        return []
    
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    ranked_results = []
    for index in matching_indexes:
        similarity = float(chunk_embeddings[index] @ query_embedding)
        ranked_results.append((similarity, chunks[index]))
    ranked_results.sort(key=lambda result: result[0], reverse=True)
    results = ranked_results[:top_k]
    for score, chunk in results:
        if chunk["department"].lower() != department.lower():
            raise RuntimeError("Search returned a chunk from the wrong department.")

    return results

if __name__ == "__main__":
    user_query = input("What do you want to search for? ")
    requested_department = input("Which department? ")

    results = search(user_query, requested_department, top_k=3)

    if not results:
        print(f"No chunks found for department: {requested_department}")
    else:
        print(f"\nTop matches from {requested_department}:")
        for rank, (score, chunk) in enumerate(results, start=1):
            print(f"\n{rank}. Similarity: {score:.3f}")
            print(f"Department: {chunk['department']}")
            print(chunk["text"])