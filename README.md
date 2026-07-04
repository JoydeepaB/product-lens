# ProductLens — E-Commerce Product Intelligence Platform

**Live Demo:** https://product-lens-cvvv.onrender.com

AI-powered product recommendation engine + fake review detector for e-commerce platforms.

**Stack:** Flask · pandas · scikit-learn (TF-IDF + cosine similarity)

---

## Features

### 1. **Content-Based Product Recommendations**
- TF-IDF vectorization on product names, categories, and descriptions
- Cosine similarity to identify semantically similar products
- Shows match percentage for each recommendation
- Works across all 40 products in 4 categories

### 2. **Fake Review Detection Engine**
- **Rule-based pattern analysis** flagging suspicious reviews across 6 dimensions:
  - Same-day review bursts (multiple reviews posted same day)
  - Near-duplicate/reused text across reviewers
  - Excessive hype language ("amazing", "best", "awesome", "buy now", etc.)
  - Very short 5-star reviews with no substantive detail
  - Repetitive punctuation (multiple "!!!")
  - **Suspicion score** ≥ 3 marks review as potentially fake
- Provides red flags explaining *why* each review is flagged
- No external labels needed — works with pattern detection alone

### 3. **Sentiment Analysis**
- Classifies each review as positive/negative/neutral
- Calculates "genuine average rating" after filtering suspicious reviews
- More honest review metrics for decision-making

### 4. **Smart Dashboard**
- Search/filter products by category and keyword
- View review summary: total count, avg rating, % suspicious, sentiment breakdown
- Click any product to see recommendations and analyzed reviews
- Filter reviews by type (All / Suspicious / Genuine)

---

## Deployment (Render)

**Already live at:** https://product-lens-cvvv.onrender.com

**Data:** 40 synthetic products across 4 categories (Electronics, Fashion, Home, Beauty) + 250 reviews (mix of genuine + planted spam patterns)

---

## How It Works

### Recommendation Engine
1. Vectorize product descriptions using TF-IDF (removes stopwords)
2. Compute cosine similarity between all product pairs
3. For a given product, return top 4 most similar products by score

### Fake Review Detection
Each review gets scored on:
- **Same-day burst** (+2): Posted on a day with 3+ other reviews
- **Text reuse** (+2): Identical/near-identical text appears 3+ times
- **Hype language** (+2): Contains 2+ hype words ("amazing", "best", "awesome", etc.)
- **Short 5-star** (+1): ≤3 words AND 5-star rating
- **Excessive punctuation** (+1): 2+ exclamation marks

**Score ≥ 3** = suspicious. Red flags explain which patterns triggered.

### Sentiment Analysis
- Count positive words (good, great, happy, recommend, etc.)
- Count negative words (disappointed, damaged, stopped, return, etc.)
- If positive > negative AND rating ≥4 → positive
- If negative > positive OR rating ≤2 → negative
- Otherwise → neutral

---

## Known Limitations

- **Recommendation is content-based only** — doesn't use collaborative filtering (v2 could add user interaction data)
- **Fake detection is pattern-based, not ML-trained** — designed to catch obvious spam without labeled training data
- **Data is synthetic** — 40 products and 250 reviews are procedurally generated for demo; a real system would ingest live marketplace data
- **No user accounts/login** — stateless single-session app; real platforms would track user behavior for personalization

---

## The Problem

E-commerce platforms face two critical challenges:

1. **Product Discovery at Scale** — Customers struggle to find similar products when browsing. Recommendation systems need to surface relevant alternatives without relying on user interaction history (which may not exist for new products).

2. **Review Authenticity Crisis** — Fake/spam reviews inflate ratings and distort customer perception. Platforms lose credibility when sellers manipulate reviews with bursts of low-effort praise or repetitive text.

ProductLens demonstrates how **content-based recommendations** and **pattern-based spam detection** solve these problems without requiring massive labeled datasets or complex ML infrastructure.

---
