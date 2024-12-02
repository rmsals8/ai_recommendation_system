import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform

st.set_page_config(
    page_title="상품 추천 시스템",
    page_icon="🛍️",
    layout="wide"
)

def load_sample_data():
    try:
        df = pd.read_csv('amazon.csv')
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating_count'] = pd.to_numeric(df['rating_count'].str.replace(',', ''), errors='coerce')
        return df
    except:
        return None

def set_korean_font():
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    else:  # Linux
        plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

def create_system_overview():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 콘텐츠 기반 추천
        다음 요소를 기반으로 추천:
        - 상품 설명
        - 카테고리
        - 가격대
        - 상품 특성
        
        특징: 유사한 상품 찾기에 최적화
        """)
    
    with col2:
        st.markdown("""
        ### 👥 협업 필터링
        다음 요소를 기반으로 추천:
        - 사용자 평점
        - 구매 이력
        - 사용자 유사도
        - 평점 패턴
        
        특징: 개인화된 추천에 최적화
        """)
    
    with col3:
        st.markdown("""
        ### 🔄 하이브리드 방식
        두 가지 방식의 장점 결합:
        - 콘텐츠 분석
        - 사용자 행동 분석
        - 유사도 측정
        - 행렬 분해
        
        특징: 균형잡힌 추천 제공
        """)

def create_sample_visualizations(df):
    if df is not None:
        set_korean_font()
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 평점 분포")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data=df, x='rating', bins=10)
            plt.title("상품 평점 분포")
            plt.xlabel("평점")
            plt.ylabel("빈도")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### 카테고리 현황")
            category_counts = df['category'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=category_counts.values, y=category_counts.index)
            plt.title("상위 10개 상품 카테고리")
            plt.xlabel("상품 수")
            st.pyplot(fig)
            plt.close()

def main():
    st.title("상품 추천 시스템에 오신 것을 환영합니다")
    
    st.markdown("""
    ## 🎯 나만을 위한 맞춤 상품 추천
    다양한 추천 방식을 활용하여 고객님께 딱 맞는 상품을 추천해드립니다.
    왼쪽 사이드바에서 원하시는 추천 방식을 선택해보세요!
    """)
    
    create_system_overview()
    
    st.markdown("---")
    
    st.markdown("""
    ## 🔍 시스템 작동 방식
    세 가지 강력한 추천 방식을 결합하였습니다:
    """)
    
    tab1, tab2, tab3 = st.tabs(["콘텐츠 기반", "협업 필터링", "하이브리드 방식"])
    
    with tab1:
        st.markdown("""
        ### 콘텐츠 기반 추천
        - TF-IDF를 활용한 상품 설명 분석
        - 특성 유사도 기반 상품 매칭
        - 카테고리 및 가격 관계 고려
        - 유사 상품 검색에 최적화
        """)
    
    with tab2:
        st.markdown("""
        ### 협업 필터링
        - KNN(K-최근접 이웃) 알고리즘 활용
        - 사용자 평점 패턴 분석
        - 유사 사용자 식별
        - 개인화 추천에 최적화
        """)
    
    with tab3:
        st.markdown("""
        ### 하이브리드 추천 시스템
        - 콘텐츠 기반과 협업 필터링 결합
        - 행렬 분해 기법 활용
        - 사용자 선호도와 상품 특성의 균형
        - 가장 종합적인 추천 제공
        """)
    
    st.markdown("---")
    
    st.markdown("## 📈 시스템 분석")
    df = load_sample_data()
    create_sample_visualizations(df)
    
    # 스타일 추가
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        h1 {
            color: #1f77b4;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
        }
        .stMarkdown {
            padding: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
