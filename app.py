positive_words = {"good", "great", "love", "excellent", "amazing"}
negative_words = {"bad", "poor", "hate", "awful", "terrible", "late"}

theme_keywords = {
    "quality": {"quality", "durable", "broken", "defective", "material"},
    "delivery": {"delivery", "shipping", "arrived", "late", "package"},
    "price": {"price", "cost", "expensive", "cheap", "value"},
}

def analyze_sentiment(review):
    words = [
        word.strip(".,!?;:()\"'")
        for word in review.lower().split()
    ]
    positive_count = sum(word in positive_words for word in words)
    negative_count = sum(word in negative_words for word in words)

    score = positive_count - negative_count

    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else: 
        label = "neutral"
    
    theme_scores = {
        theme: sum(word in keywords for word in words)
        for theme, keywords in theme_keywords.items()
    }

    theme = max(theme_scores, key=theme_scores.get)

    if theme_scores[theme] == 0:
        theme = "general"
    return {"label": label, "score": score, "theme": theme}
if __name__ == "__main__":
    review = input("Enter a review: ")

    if not review.strip():
        print("Please enter a review")
    else:
        result = analyze_sentiment(review)
        print(f"Label: {result['label']}")
        print(f"Score: {result['score']}")
        print(f"Theme: {result['theme']}")