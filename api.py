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

from gemini_analyzer import analyze_with_gemini_async
from google.genai.errors import APIError
import logging
import hmac
import os
import asyncio

from dotenv import load_dotenv
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
app = FastAPI()
UPLOAD_DIR = Path(__file__).with_name("saved_pdfs")
initialize_database()

load_dotenv()

alice_token = os.getenv("DEMO_ALICE_TOKEN")
bob_token = os.getenv("DEMO_BOB_TOKEN")

if not alice_token or not bob_token:
    raise RuntimeError("Both demo user tokens must be configured.")

demo_tokens = [
    ("alice", alice_token),
    ("bob", bob_token),
]

bearer_scheme = HTTPBearer(auto_error=False)

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    supplied_token = credentials.credentials
    for username, expected_token in demo_tokens:
        if hmac.compare_digest(supplied_token, expected_token):
            return username

    raise HTTPException(
        status_code=401,
        detail="Invalid bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
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
            pdf_path.read_bytes(),
            document["owner_id"]
        )

class ReviewRequest(BaseModel):
    review: str

class BatchReviewRequest(BaseModel):
    reviews: list[str] = Field(min_length=1, max_length=5)

async def run_gemini_analysis_async(review: str):
    try:
        return await analyze_with_gemini_async(review)
    except APIError as error:
        logging.error(
            "Gemini API failure for analysis: type=%s code=%s message=%s",
            type(error).__name__,
            error.code,
            error.message,
        )

        if error.code == 429:
            raise HTTPException(
                status_code=429,
                detail="Gemini rate limit reached. Please try again shortly.",
                headers={"Retry-After": "30"},
            ) from error

        if error.code == 503:
            raise HTTPException(
                status_code=503,
                detail="Gemini is temporarily unavailable. Please try again shortly.",
            ) from error

        raise HTTPException(
            status_code=502,
            detail="Gemini could not complete the analysis.",
        ) from error
    except TimeoutError as error:
        logging.error("Gemini analysis timed out")
        raise HTTPException(
            status_code=504,
            detail="Gemini timed out. Please try again.",
        ) from error
    except Exception as error:
        logging.exception("Unexpected Gemini analysis failure: %s", type(error).__name__)
        raise HTTPException(
            status_code=502,
            detail="Gemini could not complete the analysis.",
        ) from error


@app.post("/analyze")
async def analyze_review(request: ReviewRequest):
    if not request.review.strip():
        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty.",
        )

    try:
        result = await asyncio.wait_for(
            run_gemini_analysis_async(request.review),
            timeout=45.0,
        )
    except asyncio.TimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="Gemini timed out. Please try again.",
        ) from error
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
async def analyze_batch(request: BatchReviewRequest):
    if any(not review.strip() for review in request.reviews):
        raise HTTPException(
            status_code=400,
            detail="Reviews cannot be empty.",
        )

    semaphore = asyncio.Semaphore(1)

    async def analyze_one(review: str):
        async with semaphore:
            analysis = await run_gemini_analysis_async(review)
            return {"review": review, "analysis": analysis}

    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(analyze_one(review) for review in request.reviews)
            ),
            timeout=60.0,
        )
    except asyncio.TimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="Batch analysis timed out. Please try fewer reviews.",
        ) from error

    for item in results:
        save_analysis(item["review"], item["analysis"], "gemini")

    return {"results": results}

@app.get("/analyses")
def get_saved_analyses():
    return {"analyses": get_analyses()}

class AskRequest(BaseModel):
    question: str = Field(min_length=1)

@app.post("/ask")
def ask(request: AskRequest, current_user: str = Depends(get_current_user)):
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Q cant be empty"
        )
    result = answer_question(question, current_user)

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
async def upload_document(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
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
            current_user
        )
        save_document(original_filename, str(saved_path), current_user)
        
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