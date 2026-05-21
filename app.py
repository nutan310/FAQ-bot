
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page config
st.set_page_config(
    page_title="FAQ Semantic Search",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 FAQ Semantic Search App")
st.write("Ask a question and get the most relevant FAQ answer using Sentence Transformers.")

# Load model (cache for performance)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# FAQs
faq_questions = [
    "What is AI?",
    "What is Machine Learning?",
    "How does Deep Learning work?",
    "What is Python used for?"
]

faq_answers = [
    "AI enables machines to mimic human intelligence.",
    "Machine Learning allows systems to learn from data.",
    "Deep Learning uses neural networks with many layers.",
    "Python is widely used in AI, ML, and web development."
]

# User input
query = st.text_input("Ask your question:")

# Button click
if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Finding best answer..."):

            # Generate embeddings
            faq_embeddings = model.encode(faq_questions)
            query_embedding = model.encode([query])

            # Calculate cosine similarity
            scores = cosine_similarity(
                query_embedding,
                faq_embeddings
            )

            # Best match
            best_index = np.argmax(scores)
            similarity_score = scores[0][best_index]

        # Display result
        st.success("Best Match Found!")

        st.subheader("Most Relevant FAQ")
        st.write(faq_questions[best_index])

        st.subheader("Answer")
        st.write(faq_answers[best_index])

        st.subheader("Similarity Score")
        st.write(f"{similarity_score:.4f}")

        