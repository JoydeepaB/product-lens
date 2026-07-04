import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


class ProductLensEngine:
    def __init__(self, products_path="data/products.json", reviews_path="data/reviews.json"):
        with open(products_path) as f:
            self.products = json.load(f)
        with open(reviews_path) as f:
            self.reviews = json.load(f)

        self.products_df = pd.DataFrame(self.products)
        self.reviews_df = pd.DataFrame(self.reviews)

        self._build_recommender()

    # ---------- Recommendation engine (content-based, TF-IDF + cosine similarity) ----------
    def _build_recommender(self):
        corpus = (self.products_df["name"] + " " + self.products_df["category"] + " " + self.products_df["description"])
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def get_recommendations(self, product_id, top_n=4):
        idx_series = self.products_df.index[self.products_df["id"] == product_id]
        if len(idx_series) == 0:
            return []
        idx = idx_series[0]

        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        recommendations = []
        for i, score in scores:
            row = self.products_df.iloc[i]
            recommendations.append({
                "id": int(row["id"]),
                "name": row["name"],
                "category": row["category"],
                "price": int(row["price"]),
                "similarity": round(float(score), 3),
            })
        return recommendations

    # ---------- Fake / spam review detection (rule-based scoring, no external labels needed) ----------
    def analyze_reviews(self, product_id):
        product_reviews = self.reviews_df[self.reviews_df["product_id"] == product_id].copy()
        if product_reviews.empty:
            return []

        # Signal 1: same-day burst — many reviews posted the same date is a spam signal
        date_counts = product_reviews["date"].value_counts()
        burst_dates = set(date_counts[date_counts >= 3].index)

        # Signal 2: near-duplicate / generic text reused across reviewers
        text_counts = Counter(product_reviews["text"].str.lower().str.strip())

        # Signal 3: excessive punctuation / hype words
        hype_words = {"amazing", "best", "awesome", "superb", "perfect", "excellent", "buy now", "hurry", "must buy"}

        results = []
        for _, r in product_reviews.iterrows():
            score = 0
            reasons = []

            if r["date"] in burst_dates:
                score += 2
                reasons.append("posted in a same-day review burst")

            if text_counts[r["text"].lower().strip()] >= 3:
                score += 2
                reasons.append("near-identical text reused across multiple reviews")

            text_lower = r["text"].lower()
            hype_hits = sum(1 for w in hype_words if w in text_lower)
            if hype_hits >= 2:
                score += 2
                reasons.append("excessive hype/marketing language")

            if len(r["text"].split()) <= 3 and r["rating"] == 5:
                score += 1
                reasons.append("very short 5-star text with no detail")

            if text_lower.count("!") >= 2:
                score += 1
                reasons.append("excessive exclamation marks")

            is_suspicious = score >= 3
            sentiment = self._simple_sentiment(r["text"], r["rating"])

            results.append({
                "id": int(r["id"]),
                "reviewer": r["reviewer"],
                "rating": int(r["rating"]),
                "text": r["text"],
                "date": r["date"],
                "suspicion_score": score,
                "is_suspicious": bool(is_suspicious),
                "reasons": reasons,
                "sentiment": sentiment,
            })

        results.sort(key=lambda x: x["suspicion_score"], reverse=True)
        return results

    def _simple_sentiment(self, text, rating):
        positive_words = {"good", "great", "excellent", "happy", "satisfied", "recommend", "value", "well", "nice", "glad", "better"}
        negative_words = {"disappointed", "average", "delayed", "damaged", "stopped", "return", "off", "worse", "issue"}

        text_lower = text.lower()
        pos_hits = sum(1 for w in positive_words if w in text_lower)
        neg_hits = sum(1 for w in negative_words if w in text_lower)

        if pos_hits > neg_hits and rating >= 4:
            return "positive"
        elif neg_hits > pos_hits or rating <= 2:
            return "negative"
        return "neutral"

    # ---------- Aggregate stats for a product ----------
    def get_review_summary(self, product_id):
        analyzed = self.analyze_reviews(product_id)
        if not analyzed:
            return {"total": 0, "suspicious_count": 0, "avg_rating": 0, "sentiment_breakdown": {}}

        total = len(analyzed)
        suspicious_count = sum(1 for r in analyzed if r["is_suspicious"])
        avg_rating = round(sum(r["rating"] for r in analyzed) / total, 2)

        sentiment_breakdown = Counter(r["sentiment"] for r in analyzed)

        # rating average excluding suspicious reviews — a more "honest" score
        genuine = [r for r in analyzed if not r["is_suspicious"]]
        genuine_avg = round(sum(r["rating"] for r in genuine) / len(genuine), 2) if genuine else avg_rating

        return {
            "total": total,
            "suspicious_count": suspicious_count,
            "avg_rating": avg_rating,
            "genuine_avg_rating": genuine_avg,
            "sentiment_breakdown": dict(sentiment_breakdown),
        }

    def get_all_products(self, category=None, search=None):
        df = self.products_df
        if category:
            df = df[df["category"] == category]
        if search:
            mask = df["name"].str.lower().str.contains(search.lower()) | df["description"].str.lower().str.contains(search.lower())
            df = df[mask]
        return df.to_dict(orient="records")

    def get_product(self, product_id):
        row = self.products_df[self.products_df["id"] == product_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_categories(self):
        return sorted(self.products_df["category"].unique().tolist())
