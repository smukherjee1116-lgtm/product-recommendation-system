import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Beauty Recommender",
    page_icon="💄",
    layout="wide"
)

# ─── Load data and models ──────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Users\Lenovo\product-recommendation-system\data\ratings_clean.csv')
    df['user_encoded'] = pd.Categorical(df['user_id']).codes
    df['product_encoded'] = pd.Categorical(df['product_id']).codes
    return df

@st.cache_resource
def load_models():
    with open(r'C:\Users\Lenovo\product-recommendation-system\src\svd_model.pkl', 'rb') as f:
        svd = pickle.load(f)
    with open(r'C:\Users\Lenovo\product-recommendation-system\src\matrix_reduced.pkl', 'rb') as f:
        matrix_reduced = pickle.load(f)
    with open(r'C:\Users\Lenovo\product-recommendation-system\src\encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return svd, matrix_reduced, encoders

@st.cache_resource
def build_tfidf(_df):
    product_profiles = _df.groupby('product_id').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean'),
        rating_std=('rating', 'std')
    ).reset_index()
    product_profiles['rating_std'] = product_profiles['rating_std'].fillna(0)

    def create_description(row):
        quality = "excellent highly praised" if row['avg_rating'] >= 4.5 else \
                  "good well received" if row['avg_rating'] >= 4.0 else \
                  "average mixed reviews" if row['avg_rating'] >= 3.0 else \
                  "poor low rated"
        popularity = "very popular bestseller" if row['num_ratings'] >= 100 else \
                     "popular" if row['num_ratings'] >= 50 else \
                     "moderately popular" if row['num_ratings'] >= 20 else \
                     "niche product"
        return f"{quality} {popularity} product with {row['num_ratings']} reviews and {row['avg_rating']:.1f} average rating"

    product_profiles['description'] = product_profiles.apply(create_description, axis=1)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(product_profiles['description'])
    product_idx_map = pd.Series(product_profiles.index, index=product_profiles['product_id'])
    return product_profiles, tfidf_matrix, product_idx_map

# ─── Recommendation functions ──────────────────────────────
def get_svd_recommendations(user_id, df, matrix_reduced, svd, encoders, n=10):
    user_encoder = encoders['user_encoder']
    product_decoder = encoders['product_decoder']
    if user_id not in user_encoder:
        return pd.DataFrame()
    user_idx = user_encoder[user_id]
    user_vector = matrix_reduced[user_idx]
    scores = np.dot(user_vector, svd.components_)
    rated = df[df['user_id'] == user_id]['product_encoded'].values
    scores[rated] = 0
    top_indices = np.argsort(scores)[::-1][:n]
    return pd.DataFrame({
        'product_id': [product_decoder[i] for i in top_indices],
        'score': scores[top_indices],
        'model': 'SVD'
    })

def get_content_recommendations(user_id, df, product_profiles, tfidf_matrix, product_idx_map, n=10):
    user_rated = df[(df['user_id'] == user_id) & (df['rating'] >= 4)]['product_id'].values
    if len(user_rated) == 0:
        return pd.DataFrame()
    all_scores = np.zeros(len(product_profiles))
    for product in user_rated[:5]:
        if product not in product_idx_map:
            continue
        idx = product_idx_map[product]
        sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        all_scores += sims
    rated_products = df[df['user_id'] == user_id]['product_id'].values
    for p in rated_products:
        if p in product_idx_map:
            all_scores[product_idx_map[p]] = 0
    top_indices = np.argsort(all_scores)[::-1][:n]
    return pd.DataFrame({
        'product_id': product_profiles['product_id'].iloc[top_indices].values,
        'score': all_scores[top_indices],
        'model': 'Content'
    })

def get_hybrid_recommendations(user_id, df, matrix_reduced, svd, encoders,
                                product_profiles, tfidf_matrix, product_idx_map, n=10):
    svd_recs = get_svd_recommendations(user_id, df, matrix_reduced, svd, encoders, n=30)
    content_recs = get_content_recommendations(user_id, df, product_profiles, tfidf_matrix, product_idx_map, n=30)
    if svd_recs.empty and content_recs.empty:
        return pd.DataFrame()
    def normalise(series):
        if series.max() == series.min():
            return series * 0 + 1
        return (series - series.min()) / (series.max() - series.min())
    if not svd_recs.empty:
        svd_recs['weighted_score'] = normalise(svd_recs['score']) * 0.6
    if not content_recs.empty:
        content_recs['weighted_score'] = normalise(content_recs['score']) * 0.4
    combined = pd.concat([svd_recs, content_recs])
    hybrid = combined.groupby('product_id').agg(
        total_score=('weighted_score', 'sum'),
        appeared_in=('model', lambda x: ' + '.join(sorted(set(x))))
    ).reset_index()
    hybrid = hybrid.sort_values('total_score', ascending=False).head(n)
    hybrid = hybrid.merge(
        product_profiles[['product_id', 'avg_rating', 'num_ratings']],
        on='product_id', how='left'
    )
    return hybrid

# ─── Load everything ───────────────────────────────────────
df = load_data()
svd, matrix_reduced, encoders = load_models()
product_profiles, tfidf_matrix, product_idx_map = build_tfidf(df)

# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)
st.sidebar.title("💄 Beauty Recommender")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Get Recommendations", "📊 Model Comparison", "📈 Data Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with:**")
st.sidebar.markdown("SVD · TF-IDF · Sentence Transformers")
st.sidebar.markdown("**Dataset:** Amazon Beauty Ratings")
st.sidebar.markdown("**Records:** 394,908 ratings")

# ─── Home page ─────────────────────────────────────────────
if page == "🏠 Home":
    st.title("💄 Amazon Beauty Product Recommendation System")
    st.markdown("### Built with real Amazon data · 3 models · 394k ratings")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ratings", "394,908")
    col2.metric("Unique Users", "52,204")
    col3.metric("Unique Products", "57,289")
    col4.metric("Models Built", "3")

    st.markdown("---")
    st.markdown("## How it works")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🤝 Collaborative Filtering")
        st.markdown("Finds users similar to you and recommends what they liked. Uses SVD matrix factorisation on 52k users.")
    with col2:
        st.markdown("### 📦 Content-Based")
        st.markdown("Finds products similar to ones you already loved. Uses TF-IDF and cosine similarity on product profiles.")
    with col3:
        st.markdown("### 🔀 Hybrid Model")
        st.markdown("Combines both approaches with weighted blending. 60% SVD + 40% Content. Production-ready.")

    st.markdown("---")
    st.markdown("## Key findings")
    col1, col2, col3 = st.columns(3)
    col1.metric("SVD Coverage", "0.49%", "-popularity bias")
    col2.metric("Content Coverage", "1.92%", "+best diversity")
    col3.metric("Hybrid Coverage", "1.63%", "+best balance")
# ─── Get Recommendations page ──────────────────────────────
elif page == "🔍 Get Recommendations":
    st.title("🔍 Get Product Recommendations")
    st.markdown("Enter a User ID to get personalised recommendations")
    st.markdown("---")

    # Sample users for demo
    sample_users = df['user_id'].unique()[:5].tolist()

    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.text_input(
            "Enter User ID",
            value=sample_users[0],
            help="Try one of the sample users"
        )
    with col2:
        model_choice = st.selectbox(
            "Select Model",
            ["Hybrid (Recommended)", "SVD", "Content-Based"]
        )

    st.markdown("**Sample User IDs to try:**")
    st.code(" | ".join(sample_users))

    n_recs = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("🚀 Get Recommendations", type="primary"):
        with st.spinner("Finding best products for you..."):

            # Show user history
            user_history = df[df['user_id'] == user_id][['product_id', 'rating']].sort_values('rating', ascending=False)

            if user_history.empty:
                st.error("User not found! Please try a different User ID.")
            else:
                st.markdown(f"### 📋 User {user_id} has rated {len(user_history)} products")
                st.dataframe(user_history.head(5), use_container_width=True)

                st.markdown("---")

                # Get recommendations based on model choice
                if model_choice == "SVD":
                    recs = get_svd_recommendations(user_id, df, matrix_reduced, svd, encoders, n=n_recs)
                    if not recs.empty:
                        recs = recs.merge(product_profiles[['product_id', 'avg_rating', 'num_ratings']], on='product_id', how='left')
                        recs = recs.rename(columns={'score': 'total_score', 'model': 'appeared_in'})

                elif model_choice == "Content-Based":
                    recs = get_content_recommendations(user_id, df, product_profiles, tfidf_matrix, product_idx_map, n=n_recs)
                    if not recs.empty:
                        recs = recs.merge(product_profiles[['product_id', 'avg_rating', 'num_ratings']], on='product_id', how='left')
                        recs = recs.rename(columns={'score': 'total_score', 'model': 'appeared_in'})

                else:
                    recs = get_hybrid_recommendations(user_id, df, matrix_reduced, svd, encoders,
                                                      product_profiles, tfidf_matrix, product_idx_map, n=n_recs)

                if recs.empty:
                    st.warning("No recommendations found for this user.")
                else:
                    st.markdown(f"### 🎯 Top {n_recs} Recommendations — {model_choice}")
                    st.dataframe(recs[['product_id', 'total_score', 'avg_rating', 'num_ratings']].round(3), use_container_width=True)

                    # Chart
                    fig = px.bar(recs.head(10),
                                 x='total_score',
                                 y='product_id',
                                 orientation='h',
                                 color='total_score',
                                 color_continuous_scale='purples',
                                 title=f'Top 10 recommendations — {model_choice}')
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

# ─── Model Comparison page ─────────────────────────────────
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("How do our 3 models compare against each other?")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("SVD Coverage", "0.49%", "-popularity bias")
    col2.metric("Content Coverage", "1.92%", "+best diversity")
    col3.metric("Hybrid Coverage", "1.63%", "+best balance")

    st.markdown("---")

    # Comparison table
    comparison = pd.DataFrame({
        'Metric': ['Catalog Coverage %', 'Unique Products', 'Avg Rating', 'Cold Start', 'Popularity Bias'],
        'SVD': ['0.49%', '283', '4.27', '❌', 'High'],
        'Content-Based': ['1.92%', '1,098', '4.48', '⚠️', 'Low'],
        'Hybrid': ['1.63%', '931', '4.39', '⚠️', 'Medium']
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Coverage chart
    fig = go.Figure()
    models = ['SVD', 'Content-Based', 'Hybrid']
    coverages = [0.49, 1.92, 1.63]
    colors = ['#7F77DD', '#1D9E75', '#D85A30']

    fig.add_trace(go.Bar(
        x=models, y=coverages,
        marker_color=colors,
        text=[f'{c}%' for c in coverages],
        textposition='outside'
    ))
    fig.update_layout(title='Catalog Coverage % by Model',
                      yaxis_title='Coverage %')
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    categories = ['Coverage', 'Avg Rating', 'Diversity', 'Cold Start', 'Low Bias']
    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=[0.49, 4.27*20, 283/1098*100, 0, 20],
        theta=categories, fill='toself',
        name='SVD', line_color='#7F77DD'
    ))
    fig2.add_trace(go.Scatterpolar(
        r=[1.92, 4.48*20, 100, 50, 80],
        theta=categories, fill='toself',
        name='Content-Based', line_color='#1D9E75'
    ))
    fig2.add_trace(go.Scatterpolar(
        r=[1.63, 4.39*20, 931/1098*100, 50, 50],
        theta=categories, fill='toself',
        name='Hybrid', line_color='#D85A30'
    ))
    fig2.update_layout(title='Model Comparison — Radar Chart')
    st.plotly_chart(fig2, use_container_width=True)

# ─── Data Insights page ────────────────────────────────────
elif page == "📈 Data Insights":
    st.title("📈 Data Insights")
    st.markdown("Key patterns discovered in the Amazon Beauty dataset")
    st.markdown("---")

    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df['year'] = df['date'].dt.year

    col1, col2 = st.columns(2)

    with col1:
        # Ratings per year
        yearly = df.groupby('year').size().reset_index(name='count')
        fig1 = px.line(yearly, x='year', y='count',
                       title='Ratings per year',
                       markers=True,
                       color_discrete_sequence=['#7F77DD'])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Rating distribution
        rating_dist = df['rating'].value_counts().sort_index().reset_index()
        rating_dist.columns = ['rating', 'count']
        fig2 = px.bar(rating_dist, x='rating', y='count',
                      title='Rating distribution',
                      color='count',
                      color_continuous_scale='purples')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Top products
        top_products = df['product_id'].value_counts().head(10).reset_index()
        top_products.columns = ['product_id', 'num_ratings']
        fig3 = px.bar(top_products, x='num_ratings', y='product_id',
                      orientation='h',
                      title='Top 10 most rated products',
                      color='num_ratings',
                      color_continuous_scale='teal')
        fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # User activity
        user_activity = df.groupby('user_id').size().reset_index(name='num_ratings')
        fig4 = px.histogram(user_activity, x='num_ratings',
                            nbins=50,
                            title='User activity distribution',
                            color_discrete_sequence=['#1D9E75'])
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("### Key Insights")
    col1, col2, col3 = st.columns(3)
    col1.info("📅 Rating activity exploded post-2012 — mirroring smartphone-driven online shopping")
    col2.info("⭐ 62% of all ratings are 5 stars — strong positive bias in Beauty products")
    col3.info("👤 Average user rates only 7.6 products — most users are casual reviewers")