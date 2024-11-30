import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 전처리 함수
def preprocess_data(df):
    # 가격에서 '₹'와 ',' 제거하고 숫자로 변환
    df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['discount_percentage'] = df['discount_percentage'].str.rstrip('%').astype(float)
    
    # rating과 rating_count를 float로 변환
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = pd.to_numeric(df['rating_count'].str.replace(',', ''), errors='coerce')
    
    return df

class MatrixFactorization:
    def __init__(self, ratings_matrix, n_factors=50, learning_rate=0.01, regularization=0.02):
        self.ratings_matrix = ratings_matrix.fillna(0).values
        self.n_users, self.n_items = ratings_matrix.shape
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Initialize matrices
        self.user_factors = np.random.normal(scale=0.1, size=(self.n_users, n_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(self.n_items, n_factors))
        
        # Initialize biases
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.global_bias = np.mean(ratings_matrix[ratings_matrix.notna()])

    def train(self, n_epochs=20):
        for _ in range(n_epochs):
            for u in range(self.n_users):
                for i in range(self.n_items):
                    if self.ratings_matrix[u, i] > 0:
                        # Calculate prediction and error
                        prediction = self._predict(u, i)
                        error = self.ratings_matrix[u, i] - prediction
                        
                        # Update parameters
                        self.user_biases[u] += self.learning_rate * (error - self.regularization * self.user_biases[u])
                        self.item_biases[i] += self.learning_rate * (error - self.regularization * self.item_biases[i])
                        
                        self.user_factors[u] += self.learning_rate * (error * self.item_factors[i] - 
                                                                    self.regularization * self.user_factors[u])
                        self.item_factors[i] += self.learning_rate * (error * self.user_factors[u] - 
                                                                    self.regularization * self.item_factors[i])

    def _predict(self, user_idx, item_idx):
        prediction = self.global_bias
        prediction += self.user_biases[user_idx]
        prediction += self.item_biases[item_idx]
        prediction += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return prediction

    def get_recommendations(self, user_idx, n_recommendations=5):
        predictions = []
        for i in range(self.n_items):
            pred = self._predict(user_idx, i)
            predictions.append(pred)
        return pd.Series(predictions)

def collaborative_filtering(ratings_matrix, user_id, n_recommendations=5):
    # Calculate user similarity matrix
    user_similarity = cosine_similarity(ratings_matrix.fillna(0))
    user_similarity_df = pd.DataFrame(user_similarity, 
                                    index=ratings_matrix.index,
                                    columns=ratings_matrix.index)
    
    # Get target user's similarity scores
    target_user_similarities = user_similarity_df.loc[user_id]
    
    # Calculate weighted ratings
    weighted_ratings = pd.DataFrame(0, index=ratings_matrix.columns, columns=['rating'])
    similarity_sums = pd.Series(0, index=ratings_matrix.columns)
    
    for other_user in ratings_matrix.index:
        if other_user != user_id:
            similarity = target_user_similarities[other_user]
            # Skip users with zero similarity
            if similarity > 0:
                other_user_ratings = ratings_matrix.loc[other_user]
                for item in ratings_matrix.columns:
                    if not pd.isna(other_user_ratings[item]):
                        weighted_ratings.loc[item, 'rating'] += similarity * other_user_ratings[item]
                        similarity_sums[item] += abs(similarity)
    
    # Avoid division by zero
    similarity_sums[similarity_sums == 0] = 1
    predictions = weighted_ratings['rating'] / similarity_sums
    
    return predictions.sort_values(ascending=False).head(n_recommendations)

class HybridRecommender:
    def __init__(self, ratings_matrix, cf_weight=0.7, mf_weight=0.3):
        self.ratings_matrix = ratings_matrix
        self.cf_weight = cf_weight
        self.mf_weight = mf_weight
        self.mf = MatrixFactorization(ratings_matrix)
        
    def train(self):
        self.mf.train()
        
    def get_recommendations(self, user_id, n_recommendations=5):
        # Get CF recommendations
        cf_recommendations = collaborative_filtering(self.ratings_matrix, user_id, n_recommendations)
        
        # Get MF recommendations
        user_idx = self.ratings_matrix.index.get_loc(user_id)
        mf_recommendations = self.mf.get_recommendations(user_idx)
        mf_recommendations.index = self.ratings_matrix.columns
        
        # Combine recommendations
        hybrid_scores = (self.cf_weight * cf_recommendations +
                        self.mf_weight * mf_recommendations)
        
        return hybrid_scores.sort_values(ascending=False).head(n_recommendations)

def create_user_card(df, user_id):
    user_data = df[df['user_id'] == user_id]
    
    st.sidebar.write("### User Information")
    st.sidebar.write(f"User ID: {user_id}")
    st.sidebar.write(f"Number of Reviews: {len(user_data)}")
    
    avg_rating = user_data['rating'].mean()
    st.sidebar.write(f"Average Rating: {avg_rating:.2f}")
    
 

def visualize_recommendation_reason(df, product_id, recommendation_score):
    product = df[df['product_id'] == product_id].iloc[0]
    
    metrics = {
        'Rating': product['rating'] / 5,
        'Price Value': 1 - (product['discounted_price'] / product['actual_price']),
        'Popularity': np.log1p(product['rating_count']) / np.log1p(df['rating_count'].max()),
        'Recommendation Score': recommendation_score
    }
    
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    
    bars = ax.barh(list(metrics.keys()), list(metrics.values()), color=colors)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if not np.isnan(width):  # Check for NaN values
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', 
                    va='center')
    
    ax.set_xlim(0, 1)
    ax.set_title("Recommendation Metrics (Normalized)")
    plt.tight_layout()
    
    return fig

def main():
    st.title("Hybrid Product Recommender System")
    
    df = pd.read_csv('amazon.csv')
    df = preprocess_data(df)
    
    ratings_matrix = pd.pivot_table(df, 
                                  values='rating',
                                  index='user_id',
                                  columns='product_id')
    
    recommender = HybridRecommender(ratings_matrix)
    recommender.train()
    
    users = df['user_id'].unique()
    selected_user = st.selectbox("Select a user:", users)
    
    create_user_card(df, selected_user)
    
    if st.button("Get Recommendations"):
        recommendations = recommender.get_recommendations(selected_user)
        
        st.write("### Recommended Products")
        
        for product_id, score in recommendations.items():
            product = df[df['product_id'] == product_id].iloc[0]
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 2])
            
            # Display image in the first column
            with col1:
                st.image(product['img_link'], width=200)
            
            # Display product info in the second column
            with col2:
                st.write(f"**{product['product_name']}**")
                st.write(f"Price: ₹{product['discounted_price']:.2f} (Original: ₹{product['actual_price']:.2f})")
                st.write(f"Discount: {product['discount_percentage']}%")
                st.write(f"Rating: {product['rating']:.1f} ({int(product['rating_count'])} reviews)")
                st.write(f"Category: {product['category']}")
            
            fig = visualize_recommendation_reason(df, product_id, score/recommendations.max())
            st.pyplot(fig)
            plt.close()
            
            st.write("---")

if __name__ == "__main__":
    main()