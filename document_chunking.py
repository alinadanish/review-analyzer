import faiss
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from gemini_analyzer import client
from io import BytesIO
from threading import RLock
from pypdf import PdfReader
import time
import json 
import logging

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

INPUT_USD_PER_MILLION_TOKENS = 0.0  # Replace with the model's current input rate
OUTPUT_USD_PER_MILLION_TOKENS = 0.0  # Replace with the model's current output rate

logger = logging.getLogger(__name__)

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

def add_pdf_document(filename: str, pdf_bytes: bytes) -> dict:
    global chunk_embeddings

    reader = PdfReader(BytesIO(pdf_bytes))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    new_chunks = []

    # Split each page separately so the page citation stays accurate.
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()

        for text in split_into_chunks(page_text):
            new_chunks.append({
                "text": text,
                "department": "Uploaded PDF",
                "filename": filename,
                "page_number": page_number,
            })

    if not new_chunks:
        raise ValueError("The PDF contains no extractable text.")

    new_embeddings = model.encode(
        [chunk["text"] for chunk in new_chunks],
        normalize_embeddings=True,
    )
    new_embeddings = np.asarray(new_embeddings, dtype=np.float32)

    with _index_lock:
        first_chunk_id = len(chunks)
        indexed_chunks = [
            {"id": first_chunk_id + offset, **chunk}
            for offset, chunk in enumerate(new_chunks)
        ]
        updated_embeddings = np.concatenate(
            (chunk_embeddings, new_embeddings),
            axis=0,
        )

        faiss_index.add(new_embeddings)
        chunks.extend(indexed_chunks)
        chunk_embeddings = updated_embeddings

    return {
        "filename": filename,
        "page_count": len(reader.pages),
        "chunk_count": len(indexed_chunks),
    }
    
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
_index_lock = RLock()

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

    with _index_lock:
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

def answer_question(question):
    if not question.strip():
        raise ValueError("Question cant be empty")

    #ask faiss for chunks most similar to the question
    retrieval_started = time.perf_counter()
    retrieved_results = search_faiss(question, top_k=3)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    
    logger.info(
        "%s",
        json.dumps({
            "event": "retrieval_completed",
            "retrieval_ms": round(retrieval_ms, 2),
            "retrieved_count": len(retrieved_results),
        }),
    )
    
    #if no context, dont make gemini guess 
    if not retrieved_results:
        return {
            "answer": "i dont have enough information to answer this right now",
            "sources": [],
            "metadata": {
                "retrieval_ms": retrieval_ms,
                "llm_ms": None,
            },
        }
    
    best_score = max(score for score, _ in retrieved_results)

    MIN_SIMILARITY = 0.05
    if best_score < MIN_SIMILARITY:
        return {
            "answer": "I don't have enough information to answer this right now.",
            "sources": [],
            "metadata": {
                "retrieval_ms": retrieval_ms,
                "llm_ms": None
            }
        }
    
    #build readable context
    context_parts= []
    for score, chunk in retrieved_results:
        source_details = ""
        if "filename" in chunk:
            source_details = (
                f" | File: {chunk['filename']} | Page: {chunk['page_number']}"
            )
            
    for score, chunk in retrieved_results:
        context_parts.append(
            f"[Chunk {chunk['id']} | Department: {chunk['department']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the question using only the context below.
If the context does not contain enough information to answer, say:
"I don't have enough information in the documents to answer that."
Do not guess or add facts from outside the context.
For factual claims, cite the supporting chunk using its ID, like [Chunk 2].
Context:
{context}
Question:
{question}
"""
    
    llm_started = time.perf_counter()
    try:
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt
        )
    except Exception as exc:
        llm_ms = (time.perf_counter() - llm_started) * 1000
        logger.error(
            "%s",
            json.dumps({
                "event": "llm_failed",
                "llm_ms": round(llm_ms, 2),
                "error_type": type(exc).__name__,
            }),
        )
        return{
            "answer": None,
            "sources": [],
            "error": "i coudnt generate an answewr, please try later",
            "metadata": {
                "retrieval_ms": retrieval_ms,
                "llm_ms": llm_ms
            }
        }
    
    llm_ms = (time.perf_counter() - llm_started) * 1000
    if not response.text or not response.text.strip():
        return {
            "answer": None,
            "sources": [],
            "error": "Gemini returned an empty response. Please try again later.",
            "metadata": {
                "retrieval_ms": retrieval_ms,
                "llm_ms": llm_ms
            }
        }
    
    
    usage = getattr(response, "usage_metadata", None)

    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None) 
    
    estimated_cost_usd = None
    if input_tokens is not None and output_tokens is not None:
        estimated_cost_usd = (
            input_tokens * INPUT_USD_PER_MILLION_TOKENS
            + output_tokens * OUTPUT_USD_PER_MILLION_TOKENS
        ) / 1_000_000
        
    logger.info(
        "%s",
        json.dumps({
            "event": "ask_completed",
            "retrieval_ms": round(retrieval_ms, 2),
            "llm_ms": round(llm_ms, 2),
            "retrieved_count": len(retrieved_results),
            "slower_stage": (
                "retrieval" if retrieval_ms > llm_ms else "llm"
                
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        }),
    )
    
    sources = []
    for score, chunk in retrieved_results:
        source = {
            "chunk_id": chunk["id"],
            "department": chunk["department"],
            "similarity": float(score),
            "text": chunk["text"],
        }

        if "filename" in chunk:
            source["filename"] = chunk["filename"]
            source["page_number"] = chunk["page_number"]

        sources.append(source) 

    return {
        "answer": response.text.strip(),
        "sources": sources,
        "metadata": {
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    }

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'quit' to exit): ").strip()

        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not question: 
            print("enter a question")
            continue
    
        result = answer_question(question)

        if result.get("error"):
            print(f"\n{result['error']}")
            continue

        print(f"\nAnswer: {result['answer']}")
        print("Sources:")

        for source in result["sources"]:
            print(
                f"- Chunk {source['chunk_id']} "
                f"({source['department']}), "
                f"similarity {source['similarity']:.3f}"
            )
            
    #----FOR FAISSS
    # for chunk in chunks:
    #     print(f"\nID: {chunk['id']} | Department: {chunk['department']}")
    #     print(chunk["text"])

    # for test_case in test_cases:
    #     results = search_faiss(test_case["query"], top_k=3)
    #     retrieved_ids = [chunk["id"] for score, chunk in results]
    #     relevant_ids = test_case["relevant_ids"]

    #     precision = precision_at_k(retrieved_ids, relevant_ids, k=3)
    #     recall = recall_at_k(retrieved_ids, relevant_ids, k=3)

    #     print(f"\nQuery: {test_case['query']}")
    #     print(f"Retrieved chunk IDs: {retrieved_ids}")
    #     print(f"Relevant chunk IDs:  {sorted(relevant_ids)}")
    #     print(f"Precision@3: {precision:.2f}")
    #     print(f"Recall@3:    {recall:.2f}")

    #-----FOR DOC-CHUNKING
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