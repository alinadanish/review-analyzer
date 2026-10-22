from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import analyze_sentiment

app = FastAPI()

class ReviewRequest(BaseModel):
    review: str

@app.post("/analyze")
def analyze_review(request: ReviewRequest):
    if not request.review.strip():
        raise HTTPException(
            status_code=400,
            detail="Review cant be empty"
        )

    return analyze_sentiment(request.review)