import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load and preprocess data
@st.cache_data 
def load_data():
    df = pd.read_csv('amazon.csv')
    
    ratings_df = df[['user_id', 'product_id', 'rating']].copy()
    ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce')
    ratings_df = ratings_df.dropna()
    
    products_df = df[['product_id', 'product_name', 'category', 'about_product', 'img_link']].copy()
    products_df = products_df.drop_duplicates('product_id')
    
    return ratings_df, products_df

# Create rating matrix
def create_rating_matrix(ratings_df):
    return ratings_df.pivot_table(
        values='rating',
        index='user_id',
        columns='product_id',
        aggfunc='mean'
    )

# Calculate user similarity
def calculate_similarity(rating_matrix):
    matrix_dummy = rating_matrix.fillna(0)
    user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
    return pd.DataFrame(
        user_similarity,
        index=rating_matrix.index,
        columns=rating_matrix.index
    )

# Get recommendation factors for visualization
def get_recommendation_factors(user_id, product_id, rating_matrix, user_similarity, k=20):
    user_ratings = rating_matrix.loc[user_id].dropna()
    user_mean = user_ratings.mean()
    product_ratings = rating_matrix[product_id].dropna()
    product_mean = product_ratings.mean() if not product_ratings.empty else user_mean
    
    similarities = user_similarity.loc[user_id, product_ratings.index]
    top_k_users = similarities.nlargest(k)
    
    similar_users_avg = 0
    if not top_k_users.empty:
        similar_ratings = []
        for other_user in top_k_users.index:
            if rating_matrix.loc[other_user, product_id] == rating_matrix.loc[other_user, product_id]:  # Check for non-NaN
                similar_ratings.append(rating_matrix.loc[other_user, product_id])
        if similar_ratings:
            similar_users_avg = np.mean(similar_ratings)
    
    return {
        'user_mean': user_mean,
        'product_mean': product_mean,
        'similar_users_avg': similar_users_avg,
        'n_ratings': len(product_ratings),
    }

# Predict rating for a product
def predict_rating(user_id, product_id, rating_matrix, user_similarity, k=20):
    if product_id not in rating_matrix.columns:
        return rating_matrix.loc[user_id].mean()
    
    user_ratings = rating_matrix.loc[user_id].dropna()
    user_mean = user_ratings.mean()
    product_ratings = rating_matrix[product_id].dropna()
    
    if len(product_ratings) == 0:
        return user_mean
    
    product_mean = product_ratings.mean()
    similarities = user_similarity.loc[user_id, product_ratings.index]
    top_k_users = similarities.nlargest(k)
    
    if top_k_users.empty:
        return user_mean
        
    # Calculate weighted prediction
    numerator = denominator = 0
    for other_user, sim in top_k_users.items():
        if sim <= 0:
            continue
        other_rating = rating_matrix.loc[other_user, product_id]
        other_mean = rating_matrix.loc[other_user].mean()
        numerator += sim * (other_rating - other_mean)
        denominator += abs(sim)
    
    if denominator == 0:
        base_prediction = user_mean
    else:
        base_prediction = user_mean + (numerator / denominator)
    
    # Combine different factors
    n_ratings = len(product_ratings)
    confidence = min(n_ratings / 20, 1.0)
    final_prediction = (
        0.6 * base_prediction +
        0.3 * product_mean +
        0.1 * user_mean
    )
    
    random_factor = np.random.normal(0, 0.2)
    final_prediction += random_factor
    
    return np.clip(final_prediction, 1, 5)

# Get recommendations for a user
def get_recommendations(user_id, rating_matrix, user_similarity, products_df, n_items=5, k=20):
    user_ratings = rating_matrix.loc[user_id].dropna()
    rated_products = products_df[products_df['product_id'].isin(user_ratings.index)]
    category_scores = rated_products.groupby('category')['product_id'].count()
    
    unrated_products = rating_matrix.loc[user_id][rating_matrix.loc[user_id].isna()].index
    predictions = []
    
    for product_id in unrated_products:
        if product_id not in products_df['product_id'].values:
            continue
            
        predicted_rating = predict_rating(user_id, product_id, rating_matrix, user_similarity, k)
        factors = get_recommendation_factors(user_id, product_id, rating_matrix, user_similarity, k)
        
        if predicted_rating is not None:
            product_info = products_df[products_df['product_id'] == product_id].iloc[0]
            category = product_info['category']
            
            # Category weight calculation
            category_weight = 1.0
            if category in category_scores.index:
                category_weight = 1.0 + (category_scores[category] / category_scores.sum())
            
            final_score = predicted_rating * category_weight
            
            predictions.append({
                **product_info,
                'predicted_rating': predicted_rating,
                'final_score': final_score,
                'category_weight': category_weight,
                **factors
            })
    
    if not predictions:
        return pd.DataFrame()
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values('final_score', ascending=False)
    
    return predictions_df.head(n_items)

# Create recommendation factor visualization
def create_factor_visualization(product_data):
    factors = [
        'predicted_rating',
        'user_mean',
        'product_mean',
        'similar_users_avg',
        'category_weight'
    ]
    
    values = [
        product_data['predicted_rating'],
        product_data['user_mean'],
        product_data['product_mean'],
        product_data['similar_users_avg'],
        product_data['category_weight']
    ]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(factors, values)
    
    # Customize colors
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_title('Recommendation Factors', pad=20)
    ax.set_xlim(0, max(values) * 1.2)
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(v, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    return fig

# Display product information with visualization
def display_product(product, show_prediction=False):
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            try:
                st.image(product['img_link'], width=150)
            except:
                st.write("Image not available")
        
        with col2:
            st.subheader(product['product_name'])
            st.write(f"Category: {product['category']}")
            if show_prediction:
                st.write(f"Predicted Rating: ⭐ {float(product['predicted_rating']):.2f}")
            if pd.notna(product['about_product']):
                st.write(str(product['about_product'])[:200] + "...")
        
        if show_prediction:
            st.pyplot(create_factor_visualization(product))

# Main application
def main():
    st.title("Amazon Product Recommendation System")
    st.write("Using KNN Collaborative Filtering with Bias")
    
    try:
        ratings_df, products_df = load_data()
        
        if len(ratings_df) == 0:
            st.error("No ratings data available")
            return
            
        rating_matrix = create_rating_matrix(ratings_df)
        user_similarity = calculate_similarity(rating_matrix)
        
        st.sidebar.header("User Selection")
        users = sorted(ratings_df['user_id'].unique())
        selected_user = st.sidebar.selectbox("Select User ID:", users)
        
        st.header("User's Rated Products")
        user_ratings = rating_matrix.loc[selected_user].dropna()
        
        if not user_ratings.empty:
            rated_products = products_df[products_df['product_id'].isin(user_ratings.index)]
            for _, product in rated_products.iterrows():
                display_product(product)
                st.markdown("---")
        
        st.sidebar.header("Recommendation Settings")
        n_recommendations = st.sidebar.slider("Number of recommendations:", 1, 10, 5)
        k_neighbors = st.sidebar.slider("Number of neighbors (K):", 5, 50, 20)
        
        if st.sidebar.button("Get Recommendations"):
            st.header("Recommended Products")
            with st.spinner("Finding recommendations..."):
                recommendations = get_recommendations(
                    selected_user,
                    rating_matrix,
                    user_similarity,
                    products_df,
                    n_recommendations,
                    k_neighbors
                )
                
                if not recommendations.empty:
                    for _, product in recommendations.iterrows():
                        display_product(product, show_prediction=True)
                        st.markdown("---")
                else:
                    st.info("Could not find recommendations. Try adjusting the parameters.")
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Error details:", str(e))

if __name__ == "__main__":
    main()