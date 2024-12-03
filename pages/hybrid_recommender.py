import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['discount_percentage'] = df['discount_percentage'].str.rstrip('%').astype(float)
    
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = pd.to_numeric(df['rating_count'].str.replace(',', ''), errors='coerce')
    
    return df

class MatrixFactorization:
    def __init__(self, ratings_matrix, n_factors=50, learning_rate=0.01, regularization=0.02):
        self.original_matrix = ratings_matrix.copy()
        self.ratings_matrix = ratings_matrix.fillna(0).values
        self.n_users, self.n_items = ratings_matrix.shape
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        self.user_factors = np.random.normal(scale=0.1, size=(self.n_users, n_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(self.n_items, n_factors))
        
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.global_bias = np.mean(ratings_matrix[ratings_matrix.notna()])

    def train(self, n_epochs=20):
        try:
            for _ in range(n_epochs):
                for u in range(self.n_users):
                    for i in range(self.n_items):
                        if self.ratings_matrix[u, i] > 0:
                            prediction = self._predict(u, i)
                            error = self.ratings_matrix[u, i] - prediction
                            
                            self.user_biases[u] += self.learning_rate * (error - self.regularization * self.user_biases[u])
                            self.item_biases[i] += self.learning_rate * (error - self.regularization * self.item_biases[i])
                            
                            self.user_factors[u] += self.learning_rate * (error * self.item_factors[i] - 
                                                                        self.regularization * self.user_factors[u])
                            self.item_factors[i] += self.learning_rate * (error * self.user_factors[u] - 
                                                                        self.regularization * self.item_factors[i])
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            raise

    def _predict(self, user_idx, item_idx):
        try:
            prediction = self.global_bias
            prediction += self.user_biases[user_idx]
            prediction += self.item_biases[item_idx]
            prediction += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            return np.clip(prediction, 1, 5)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return self.global_bias

    def get_recommendations(self, user_idx, n_recommendations=5):
        try:
            predictions = []
            for i in range(self.n_items):
                pred = self._predict(user_idx, i)
                predictions.append(pred)
            
            return pd.Series(predictions, index=self.original_matrix.columns)
        except Exception as e:
            st.error(f"Recommendation error: {str(e)}")
            return pd.Series([], dtype=float)

def collaborative_filtering(ratings_matrix, user_id, n_recommendations=5):
    try:
        user_similarity = cosine_similarity(ratings_matrix.fillna(0))
        user_similarity_df = pd.DataFrame(user_similarity, 
                                        index=ratings_matrix.index,
                                        columns=ratings_matrix.index)
        
        target_user_similarities = user_similarity_df.loc[user_id]
        
        weighted_ratings = pd.DataFrame(0, index=ratings_matrix.columns, columns=['rating'])
        similarity_sums = pd.Series(0, index=ratings_matrix.columns)
        
        for other_user in ratings_matrix.index:
            if other_user != user_id:
                similarity = target_user_similarities[other_user]
                if similarity > 0:
                    other_user_ratings = ratings_matrix.loc[other_user]
                    for item in ratings_matrix.columns:
                        if not pd.isna(other_user_ratings[item]):
                            weighted_ratings.loc[item, 'rating'] += similarity * other_user_ratings[item]
                            similarity_sums[item] += abs(similarity)
        
        similarity_sums[similarity_sums == 0] = 1
        predictions = weighted_ratings['rating'] / similarity_sums
        
        return predictions.sort_values(ascending=False).head(n_recommendations)
    except Exception as e:
        st.error(f"Collaborative filtering error: {str(e)}")
        return pd.Series([], dtype=float)

class HybridRecommender:
    def __init__(self, ratings_matrix, cf_weight=0.7, mf_weight=0.3):
        self.ratings_matrix = ratings_matrix
        self.cf_weight = cf_weight
        self.mf_weight = mf_weight
        self.mf = MatrixFactorization(ratings_matrix)
        
    def train(self):
        try:
            self.mf.train()
        except Exception as e:
            st.error(f"Training error in hybrid recommender: {str(e)}")
            raise
        
    def get_recommendations(self, user_id, n_recommendations=5):
        try:
            cf_recommendations = collaborative_filtering(self.ratings_matrix, user_id, n_recommendations)
            
            user_idx = self.ratings_matrix.index.get_loc(user_id)
            mf_recommendations = self.mf.get_recommendations(user_idx)
            
            all_items = set(cf_recommendations.index) | set(mf_recommendations.index)
            
            cf_recommendations = cf_recommendations.reindex(all_items, fill_value=0)
            mf_recommendations = mf_recommendations.reindex(all_items, fill_value=0)
            
            hybrid_scores = (self.cf_weight * cf_recommendations +
                           self.mf_weight * mf_recommendations)
            
            return hybrid_scores.sort_values(ascending=False).head(n_recommendations)
        except Exception as e:
            st.error(f"Recommendation error in hybrid recommender: {str(e)}")
            return pd.Series([], dtype=float)

def visualize_recommendation_scores(mf_pred, cf_weight=0.7, mf_weight=0.3):
    plt.figure(figsize=(10, 6))
    
    # Calculate scores
    cf_score = cf_weight * mf_pred
    mf_weighted_score = mf_weight * mf_pred
    hybrid_score = cf_score + mf_weighted_score
    
    # Create bar plot
    scores = [mf_weighted_score, cf_score, hybrid_score]
    labels = ['MF Score\n(30%)', 'CF Score\n(70%)', 'Hybrid\nScore']
    bars = plt.bar(labels, scores, color=['skyblue', 'lightgreen', 'orange'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('Recommendation Scores Composition')
    plt.ylabel('Score Value')
    plt.ylim(0, max(scores) * 1.2)
    
    return plt.gcf()

def visualize_recommendation_process(recommender, df, user_id, product_id):
    try:
        user_idx = recommender.ratings_matrix.index.get_loc(user_id)
        item_idx = recommender.ratings_matrix.columns.get_loc(product_id)
        
        mf_pred = recommender.mf._predict(user_idx, item_idx)
        return visualize_recommendation_scores(mf_pred)
        
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None

def create_user_card(df, user_id):
    try:
        user_data = df[df['user_id'] == user_id]
        
        st.sidebar.write("### User Information")
        st.sidebar.write(f"User ID: {user_id}")
        st.sidebar.write(f"Number of Reviews: {len(user_data)}")
        
        avg_rating = user_data['rating'].mean()
        st.sidebar.write(f"Average Rating: {avg_rating:.2f}")
    except Exception as e:
        st.error(f"Error creating user card: {str(e)}")

def main():
    try:
        st.title("Hybrid Product Recommender System")
        
        df = pd.read_csv('amazon.csv')
        if df.empty:
            st.error("Error: Empty dataset")
            return
            
        df = preprocess_data(df)
        
        ratings_matrix = pd.pivot_table(df, 
                                      values='rating',
                                      index='user_id',
                                      columns='product_id')
        
        if ratings_matrix.empty:
            st.error("Error: No ratings data available")
            return
            
        recommender = HybridRecommender(ratings_matrix)
        recommender.train()
        
        users = df['user_id'].unique()
        if len(users) == 0:
            st.error("Error: No users found")
            return
            
        selected_user = st.selectbox("Select a user:", users)
        
        create_user_card(df, selected_user)
        
        if st.button("Get Recommendations"):
            recommendations = recommender.get_recommendations(selected_user)
            
            if recommendations.empty:
                st.warning("No recommendations found for this user")
                return
                
            st.write("### Recommended Products")
            
            for product_id, score in recommendations.items():
                product = df[df['product_id'] == product_id].iloc[0]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(product['img_link'], width=200)
                
                with col2:
                    st.write(f"**{product['product_name']}**")
                    st.write(f"Price: ₹{product['discounted_price']:.2f} (Original: ₹{product['actual_price']:.2f})")
                    st.write(f"Discount: {product['discount_percentage']}%")
                    st.write(f"Rating: {product['rating']:.1f} ({int(product['rating_count']) if not pd.isna(product['rating_count']) else 0} reviews)")
                    st.write(f"Category: {product['category']}")
                
                fig = visualize_recommendation_process(recommender, df, selected_user, product_id)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close()
                
                st.write("---")
    except Exception as e:
        st.error(f"Main application error: {str(e)}")

if __name__ == "__main__":
    main()
