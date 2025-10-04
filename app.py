import streamlit as st
import joblib
import pandas as pd

# Load model & vectorizer
model = joblib.load('model/logistic_regression_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

# APP CONFIG
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS STYLE
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stDownloadButton button {
        background-color: #FF4B4B;
        color: white;
    }
    .stRadio > div {
        flex-direction: row;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# HEADER
st.title("Sentiment Analysis 💬")
st.markdown("Analyze sentiments from text or CSV/Excel file.")

# SIDEBAR
with st.sidebar:
    st.header("About App")
    st.markdown("""
    - This app performs **Sentiment Analysis** using a Logistic Regression model.
    - Supports **Single text** or **batch CSV/Excel** file.
    - Developed by Karan Singh with project team.
    """)
    st.header("Model Info")
    st.markdown("""
    - Model: Logistic Regression
    - Vectorizer: TF-IDF
    - Accuracy: **78.34%**
    """)

# INPUT METHOD
input_method = st.radio("Choose Input Method:", ("Single Text", "Upload CSV/Excel File"))

# SINGLE TEXT
if input_method == "Single Text":
    text_input = st.text_area("Enter tweet text:", height=150)

    col1, col2 = st.columns(2)
    with col1:
        analyze_button = st.button("Analyze")
    with col2:
        clear_button = st.button("Clear")

    if clear_button:
        st.experimental_rerun()

    if analyze_button and text_input.strip() != "":
        input_vector = tfidf.transform([text_input])
        prediction = model.predict(input_vector)[0]

        sentiment_emojis = {
            'positive': '👍 Positive',
            'neutral': '😐 Neutral',
            'negative': '👎 Negative'
        }

        st.subheader("Prediction Result")
        st.markdown(f"### {sentiment_emojis[prediction]}")

    elif analyze_button and text_input.strip() == "":
        st.warning("Please enter tweet text before clicking Analyze.")

# CSV / EXCEL FILE
elif input_method == "Upload CSV/Excel File":
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Read CSV or Excel
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("File Preview:")
            st.dataframe(df.head())

            if 'text' not in df.columns:
                st.error("File must contain a column named 'text'.")
            else:
                # Predict
                input_vectors = tfidf.transform(df['text'].astype(str))
                predictions = model.predict(input_vectors)

                df['Predicted Sentiment'] = predictions

                # Show sentiment counts
                sentiment_counts = df['Predicted Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']

                st.subheader("Overall Sentiment Counts")
                st.dataframe(sentiment_counts)

                # Show table of results
                st.subheader("Prediction Results")
                st.dataframe(df[['text', 'Predicted Sentiment']])

                # Download button
                csv_download = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_download,
                    file_name='sentiment_results.csv',
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"Error reading file: {e}")
