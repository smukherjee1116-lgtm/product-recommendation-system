# 💄 Amazon Beauty Product Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://amazon-beauty-recommender.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

> An end-to-end product recommendation system built on 2 million real Amazon Beauty ratings — combining Collaborative Filtering, Content-Based Filtering and a Hybrid model — deployed as an interactive Streamlit dashboard.

🔗 **Live Demo:** https://amazon-beauty-recommender.streamlit.app/

---

## 📌 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Models Built](#models-built)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Key Insights](#key-insights)
- [Resume Bullet](#resume-bullet)

---

## 🎯 Problem Statement

Amazon sells millions of Beauty products. Without personalised recommendations:
- Customers get overwhelmed and leave without buying
- Long-tail products never get discovered
- Amazon loses revenue on niche inventory

**Goal:** Build a recommendation system that surfaces the right product to the right user at the right time — using real Amazon ratings data.

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| Source | [Amazon Beauty Ratings — Kaggle](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings) |
| Raw ratings | 2,023,070 |
| Raw users | 1,210,271 |
| Raw products | 249,274 |
| After cleaning | 394,908 ratings |
| Active users | 52,204 |
| Active products | 57,289 |
| Sparsity | 99.99% |

**Cleaning strategy:** Kept only users with 5+ ratings and products with 5+ ratings to reduce noise and improve model quality.

---

## 🏗️ Project Architecture
Raw Data (2M ratings)
↓
Data Cleaning
(394k ratings)
↓
┌───────────────────────────┐
│                           │
SVD Model          Content-Based Model
(Collaborative)      (TF-IDF + Cosine)
│                           │
└──────────┬────────────────┘
↓
Hybrid Model
(60% SVD + 40% Content)
↓
Streamlit Dashboard
(Live Public URL)

---

## 🤖 Models Built

### 1. Collaborative Filtering — SVD
- Decomposes the 52k × 57k user-item matrix using Truncated SVD
- Learns 50 latent factors representing hidden user taste dimensions
- Recommends products based on behaviour of similar users

### 2. Content-Based Filtering — TF-IDF
- Builds a rich text profile for each product from rating statistics
- Converts profiles to TF-IDF vectors
- Finds similar products using cosine similarity
- Solves cold start problem partially

### 3. Sentence Transformer Embeddings
- Uses `all-MiniLM-L6-v2` from HuggingFace
- Generates 384-dimensional semantic embeddings for product descriptions
- Achieves similarity scores of 0.94+ vs TF-IDF's 0.47

### 4. Hybrid Model
- Combines SVD (60%) and Content-Based (40%) using weighted blending
- Normalises scores to 0-1 scale before combining
- Products endorsed by both models receive a natural score boost
- Production-ready architecture

---

## 📊 Results

| Metric | SVD | Content-Based | Hybrid |
|--------|-----|---------------|--------|
| Catalog Coverage | 0.49% | 1.92% | 1.63% |
| Unique Products Recommended | 283 | 1,098 | 931 |
| Avg Rating of Recommendations | 4.27 | 4.48 | 4.39 |
| Cold Start Handling | ❌ | ⚠️ | ⚠️ |
| Popularity Bias | High | Low | Medium |
| Production Ready | ❌ | ⚠️ | ✅ |

**Key finding:** SVD alone misses 99.5% of the product catalog due to popularity bias. The Hybrid model discovers 3.3x more products while maintaining high recommendation quality.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data processing | Pandas, NumPy, SQL |
| Machine Learning | scikit-learn, SciPy |
| NLP | TF-IDF, Sentence Transformers |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | GitHub (daily commits) |

---

## 📁 Project Structure
product-recommendation-system/
├── data/
│   ├── ratings_Beauty.csv          # Raw dataset
│   └── ratings_clean.csv           # Cleaned dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA + data loading
│   ├── 02_collaborative_filtering.ipynb  # SVD model
│   ├── 03_model_evaluation.ipynb   # RMSE + coverage metrics
│   ├── 04_deep_eda.ipynb           # Deep EDA + insights
│   ├── 05_content_based_filtering.ipynb  # TF-IDF model
│   ├── 06_sentence_transformers.ipynb    # Semantic embeddings
│   ├── 07_hybrid_model.ipynb       # Hybrid recommendation
│   ├── 08_model_comparison.ipynb   # Full comparison + radar chart
│   └── 09_data_story.ipynb         # Data narrative
├── src/
│   ├── svd_model.pkl               # Trained SVD model
│   ├── matrix_reduced.pkl          # Reduced user-item matrix
│   └── encoders.pkl                # User/product encoders
├── dashboard/
│   └── app.py                      # Streamlit app
├── outputs/
│   └── data_story.md               # Full data story
├── requirements.txt
└── README.md

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/smukherjee1116-lgtm/product-recommendation-system.git
cd product-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python -m streamlit run dashboard/app.py
```

---

## 💡 Key Insights

### Data Insights
- Rating activity was flat 2000–2010 then **exploded post-2012** — mirroring smartphone-driven online shopping
- **62% of all ratings are 5 stars** — strong positive bias in Beauty products
- Average user rates only **7.6 products** — most users are casual, one-time reviewers
- Median product has only **3 ratings** — most products are niche

### Model Insights
- SVD captures user taste well but suffers from severe **popularity bias** (0.49% coverage)
- Content-based achieves best **catalog coverage** (1.92%) and highest quality recs (4.48 avg rating)
- Hybrid is the **production-ready choice** — balancing popularity, diversity and quality

### Business Insights
- SVD alone misses **99.5% of the product catalog**
- Hybrid discovers **3.3x more products** than SVD alone
- More catalog coverage = more long-tail products sold = higher revenue

---

## 📝 Resume Bullet

> Built an end-to-end product recommendation system on 2M+ Amazon Beauty ratings using SVD matrix factorisation, TF-IDF content filtering and a hybrid weighted blending model — achieving 3.3x improvement in catalog coverage (0.49% → 1.63%) over the SVD baseline. Deployed as an interactive Streamlit dashboard with live public URL.

---

## 🔮 Future Improvements
- Add implicit feedback (clicks, time spent) for richer user signals
- Implement FAISS for approximate nearest neighbour search at scale
- Add A/B testing framework to measure real business impact
- Fine-tune hybrid weights using learning-to-rank (LambdaMART)
- Add real product metadata (titles, categories, brand) for richer content profiles

---

*Built in 15 days · Daily GitHub commits · Real Amazon data · 3 models · 1 live dashboard*
