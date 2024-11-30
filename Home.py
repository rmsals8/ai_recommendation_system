import streamlit as st

st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

st.title("Welcome to Product Recommendation System")
st.write("""
## Choose a recommendation method from the sidebar:

1. **Content Based** - Using product descriptions and categories
2. **KNN Based** - Using user-rating based collaborative filtering
3. **TF-IDF Based** - Using text similarity between product descriptions
""")

# Optional: Add some styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        h1 {
            color: #1f77b4;
        }
    </style>
""", unsafe_allow_html=True)