import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
from io import BytesIO

# 페이지 설정
st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('amazon.csv')
    
    # 가격 데이터 전처리
    df['discounted_price'] = df['discounted_price'].replace('[\₹,]', '', regex=True).astype(float)
    df['actual_price'] = df['actual_price'].replace('[\₹,]', '', regex=True).astype(float)
    df['discount_percentage'] = df['discount_percentage'].str.rstrip('%').astype(float)
    
    # 평점 데이터 전처리 
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = pd.to_numeric(df['rating_count'].str.replace(',', ''), errors='coerce')
    
    # 결측값 처리
    df = df.fillna(0)
    
    # description이 없는 상품 제거 
    df = df[df['about_product'].notna() & (df['about_product'] != '')]
    
    return df

@st.cache_resource
def get_tfidf_matrices(texts):
    """텍스트 데이터를 TF-IDF 행렬로 변환"""
    tfidf = TfidfVectorizer(
        stop_words='english',
        min_df=2,
        max_features=5000
    )
    return tfidf.fit_transform(texts), tfidf

def create_recommendation_plots(product, recommendations):
    """추천된 상품들의 추천 이유를 시각화"""
    
    # 1. Radar Plot - 유사도 지표 비교
    fig1 = plt.figure(figsize=(10, 6))
    
    # 비교할 지표 설정
    metrics = ['Similarity', 'Price Ratio', 'Rating Ratio', 'Review Count Ratio']
    num_metrics = len(metrics)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 첫점으로 돌아가기 위해
    
    ax = fig1.add_subplot(111, polar=True)
    
    for idx, rec in recommendations.iterrows():
        # 각 지표별 값 계산
        values = [
            rec['similarity_score'],
            min(product['discounted_price'], rec['discounted_price']) / max(product['discounted_price'], rec['discounted_price']),
            min(product['rating'], rec['rating']) / max(product['rating'], rec['rating']) if max(product['rating'], rec['rating']) > 0 else 0,
            min(product['rating_count'], rec['rating_count']) / max(product['rating_count'], rec['rating_count']) if max(product['rating_count'], rec['rating_count']) > 0 else 0
        ]
        values = np.concatenate((values, [values[0]]))  # 첫점으로 돌아가기 위해
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Product {idx}')
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.3, 1))
    plt.title('Similarity Metrics Comparison')

    # 2. Bar Plot - 상품별 주요 키워드 중요도
    fig2 = plt.figure(figsize=(12, 6))
    
    keywords_data = []
    for idx, rec in recommendations.iterrows():
        keywords = rec['key_features'].split(', ')
        for kw in keywords:
            name, score = kw.split(' (')
            score = float(score.rstrip(')'))
            keywords_data.append({
                'Product': f'Product {idx}',
                'Keyword': name,
                'Score': score
            })
    
    keywords_df = pd.DataFrame(keywords_data)
    
    plt.barh(y=np.arange(len(keywords_data)), 
            width=keywords_df['Score'],
            color=plt.cm.tab20(np.linspace(0, 1, len(set(keywords_df['Product'])))))
    
    plt.yticks(np.arange(len(keywords_data)), 
              [f"{row['Product']} - {row['Keyword']}" for _, row in keywords_df.iterrows()])
    
    plt.xlabel('TF-IDF Score')
    plt.title('Keyword Importance by Product')
    
    return fig1, fig2

def get_similar_products(df, product_idx, tfidf_matrix, tfidf_vectorizer, n_recommendations=5):
    """주어진 상품과 유사한 상품들을 찾아 반환"""
    
    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(tfidf_matrix[product_idx:product_idx+1], tfidf_matrix).flatten()
    
    # 상품 인덱스와 유사도 점수 매핑
    sim_scores = list(enumerate(cosine_sim))
    
    # 자기 자신과 중복 상품 제외
    product_name = df.iloc[product_idx]['product_name']
    product_desc = df.iloc[product_idx]['about_product']
    
    filtered_scores = [
        (idx, score) 
        for idx, score in sim_scores 
        if idx != product_idx and 
        df.iloc[idx]['product_name'] != product_name and 
        df.iloc[idx]['about_product'] != product_desc
    ]
    
    # 유사도 기준 정렬
    filtered_scores = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    if not filtered_scores:
        return pd.DataFrame()
    
    # 추천 상품 정보와 유사도 점수 결합
    indices = [i[0] for i in filtered_scores]
    scores = [i[1] for i in filtered_scores]
    
    recommendations = df.iloc[indices].copy()
    recommendations['similarity_score'] = scores
    
    # 중요 키워드 추출
    feature_names = tfidf_vectorizer.get_feature_names_out()
    for idx, row in recommendations.iterrows():
        tfidf_scores = tfidf_matrix[idx].toarray()[0]
        top_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:5]
        keywords_with_scores = [f"{k} ({v:.2f})" for k, v in top_keywords if v > 0]
        recommendations.at[idx, 'key_features'] = ', '.join(keywords_with_scores)
    
    return recommendations

def display_product_card(product, recommendations=None, show_similarity=False):
    """상품 정보를 카드 형태로 표시"""
    
    st.markdown("""
    <style>
    .product-card {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .small-text {
        font-size: 0.85rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            response = requests.get(product['img_link'], timeout=3)
            img = Image.open(BytesIO(response.content))
            st.image(img, width=150)
        except:
            st.info("Image not available")
    
    with col2:
        st.subheader(product['product_name'])
        st.write(f"Price: ₹{product['discounted_price']:.2f}")
        st.write(f"Rating: ⭐ {product['rating']:.1f} ({int(product['rating_count']):,} reviews)")
        st.write(f"Discount: {product['discount_percentage']}%")
        
        if show_similarity:
            st.write(f"Similarity Score: {product['similarity_score']:.3f}")
    
    st.markdown("<p class='small-text'>{}</p>".format(
        product['about_product'][:200] + "..." if len(product['about_product']) > 200 else product['about_product']
    ), unsafe_allow_html=True)
    
    # 추천 상품이 있고 유사도를 표시해야 할 경우 시각화
    if recommendations is not None and show_similarity:
        st.subheader("Recommendation Analysis")
        
        fig1, fig2 = create_recommendation_plots(product, recommendations)
        
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig1)
        with col2:
            st.pyplot(fig2)

def main():
    st.title("Product Recommendation System")
    st.write("Using TF-IDF and Cosine Similarity with Visualization")
    
    try:
        df = load_data()
        tfidf_matrix, vectorizer = get_tfidf_matrices(df['about_product'])
        
        search_query = st.text_input("Search for a product:", key="search")
        if search_query:
            results = df[df['product_name'].str.contains(search_query, case=False, na=False)]
            
            if len(results) == 0:
                st.warning("No products found.")
            else:
                st.subheader(f"Found {len(results)} products")
                
                for idx, product in results.iterrows():
                    st.markdown("---")
                    display_product_card(product)
                    
                    if st.button(f"Show Similar Products", key=f"btn_{idx}"):
                        recommendations = get_similar_products(df, idx, tfidf_matrix, vectorizer)
                        
                        if len(recommendations) > 0:
                            st.subheader("Recommended Products")
                            display_product_card(product, recommendations)
                            
                            for _, rec in recommendations.iterrows():
                                display_product_card(rec, recommendations, show_similarity=True)
                                st.markdown("---")
                        else:
                            st.info("No similar products found.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()