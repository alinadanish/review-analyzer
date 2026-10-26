from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path

from app import analyze_sentiment
from database import initialize_database, save_analysis, get_analyses
from document_chunking import answer_question

from gemini_analyzer import analyze_with_gemini
from google.genai.errors import APIError

app = FastAPI()
initialize_database()

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
            "best_similarity": best_similarity
        }
    }

app.mount(
    "/",
    StaticFiles(directory=Path(__file__).with_name("static"), html=True),
    name="frontend",
)