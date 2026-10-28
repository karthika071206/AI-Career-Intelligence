"""Safe baseline predictor used when a trained career model is not packaged."""
CAREER_KEYWORDS = {
    "Software Developer": ["python", "java", "c", "c++", "javascript", "html", "css", "dsa", "programming"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "pandas", "statistics", "data analysis"],
    "Data Scientist": ["python", "pandas", "numpy", "statistics", "machine learning", "data science"],
    "AI / ML Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "ai"],
    "Backend Developer": ["python", "java", "sql", "flask", "django", "api", "git"],
    "Frontend Developer": ["html", "css", "javascript", "react", "ui"],
    "Cybersecurity Analyst": ["networking", "linux", "security", "cybersecurity"],
    "Product Manager": ["communication", "leadership", "project management", "analytics"],
}
def predict_careers(skills, top_k=5, **kwargs):
    text = str(skills or "").casefold()
    results = []
    for career, keywords in CAREER_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text]
        score = round(100 * len(matches) / max(1, len(keywords)), 2)
        results.append({"career": career, "score": score, "matched": matches})
    return sorted(results, key=lambda x: x["score"], reverse=True)[:max(1, int(top_k))]
