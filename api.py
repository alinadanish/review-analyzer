from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
from uuid import uuid4

from pypdf.errors import PdfReadError
from starlette.concurrency import run_in_threadpool

from app import analyze_sentiment
from database import initialize_database, save_analysis, get_analyses, save_document, get_documents
from document_chunking import answer_question, add_pdf_document

from gemini_analyzer import analyze_with_gemini
from google.genai.errors import APIError
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
app = FastAPI()
UPLOAD_DIR = Path(__file__).with_name("saved_pdfs")
initialize_database()

@app.on_event("startup")
def reload_uploaded_documents():
    for document in get_documents():
        pdf_path = Path(document["file_path"])
        
        if not pdf_path.is_file():
            raise RuntimeError(
                f"Saved pdf is missing: {document['file_name']}"
            )
            
        add_pdf_document(
            document["file_name"],
            pdf_path.read_bytes()
        )

class ReviewRequest(BaseModel):
    review: str

class BatchReviewRequest(BaseModel):
    reviews: list[str] = Field(min_length=1, max_length=5)

def run_gemini_analysis(review: str):
    try:
        return analyze_with_gemini(review)
    except APIError as error:
        if error.code == 503:
            raise HTTPException(
                status_code=503,
                detail="Gemini is temporarily unavailable. Please try again shortly.",
            ) from error

        raise HTTPException(
            status_code=502,
            detail="Gemini could not complete the analysis.",
        ) from error


@app.post("/analyze")
def analyze_review(request: ReviewRequest):
    if not request.review.strip():
        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty.",
        )

    result = run_gemini_analysis(request.review)
    save_analysis(request.review, result, "gemini")
    return result


@app.post("/analyze/rules")
def analyze_with_rules(request: ReviewRequest):
    if not request.review.strip():
        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty.",
        )

    result = analyze_sentiment(request.review)
    save_analysis(request.review, result, "rules")
    return result
    
@app.post("/analyze/batch")
def analyze_batch(request: BatchReviewRequest):
    if any(not review.strip() for review in request.reviews):
        raise HTTPException(
            status_code=400,
            detail="Reviews cannot be empty.",
        )

    results = []
    for review in request.reviews:
        analysis = run_gemini_analysis(review)
        results.append({"review": review, "analysis": analysis})

    for item in results:
        save_analysis(item["review"], item["analysis"], "gemini")

    return {"results": results}

@app.get("/analyses")
def get_saved_analyses():
    return {"analyses": get_analyses()}

class AskRequest(BaseModel):
    question: str = Field(min_length=1)

@app.post("/ask")
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Q cant be empty"
        )
    result = answer_question(question)

    if result.get("error"):
        raise HTTPException(
            status_code=502,
            detail="Ans generation failed"
        )
    
    sources = result["sources"]
    best_similarity = max(
        (source["similarity"] for source in sources),
        default=None
    )
    return{
        "answer": result["answer"],
        "sources": sources,
        "metadata": {
            "source_count": len(sources),
            "best_similarity": best_similarity,
            **result.get("metadata", {})
        }
    }

@app.post("/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    original_filename = Path(file.filename or "").name

    if not original_filename or Path(original_filename).suffix.casefold() != ".pdf":
        raise HTTPException(
            status_code=415,
            detail="Upload a PDF file.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOAD_DIR / f"{uuid4().hex}.pdf"
    
    try:
        pdf_bytes = await file.read()
        saved_path.write_bytes(pdf_bytes)
        result = await run_in_threadpool(
            add_pdf_document,
            original_filename,
            pdf_bytes,
        )
        save_document(original_filename, str(saved_path))
        
    except PdfReadError as error:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a readable PDF.",
        ) from error
    except ValueError as error:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
    finally:
        await file.close()

    return {
        "message": "PDF indexed successfully.",
        **result,
    }
        
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).with_name("static"), html=True),
    name="frontend",
)