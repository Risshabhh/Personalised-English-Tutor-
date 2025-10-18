# Personalized English Tutor (Gemini + Streamlit)
import os
import streamlit as st
import pdfplumber
import docx
import spacy
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load API Key and Models
load_dotenv()  # Optional: loads GOOGLE_API_KEY from .env

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyBizQv0q-6AnozWIzCd--cIv-Rp7X-FJ74"

if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Please set GOOGLE_API_KEY.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Load spaCy English model
import spacy
nlp = spacy.load("en_core_web_sm")

# Helper Functions
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

def summarize_with_gemini(text, prompt="Summarize this English text:"):
    # Stable Gemini 2.5 Flash for summarization
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(f"{prompt}\n\n{text}")
    return response.text

def give_feedback(user_text):
    # Stable Gemini 2.5 Pro for writing feedback
    model = genai.GenerativeModel("gemini-2.5-pro")
    prompt = f"""
    You are an English tutor. Analyze this student's writing and provide:
    1. Grammar corrections
    2. Vocabulary suggestions
    3. Fluency rating (1-10)
    4. Brief motivational feedback

    Student text:
    {user_text}
    """
    response = model.generate_content(prompt)
    return response.text

def measure_similarity(text1, text2):
    vec = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vec[0:1], vec[1:2])[0][0]
# Streamlit Interface
st.set_page_config(page_title="Personalized English Tutor", layout="wide")

st.title("Personalized English Tutor")
st.markdown("Improve your English with AI-powered feedback, document analysis, and learning recommendations.")

tab1, tab2, tab3 = st.tabs(["* Writing Practice", "* Document Analyzer", "* Chat Tutor"])

# Writing Practice
with tab1:
    st.subheader("Practice Writing")
    user_text = st.text_area("Write something in English:", height=200)
    if st.button("Analyze Writing"):
        if user_text.strip():
            if not GOOGLE_API_KEY:
                st.error("Google API key missing. Cannot generate feedback.")
            else:
                with st.spinner("Analyzing your writing..."):
                    feedback = give_feedback(user_text)
                    st.success("Feedback generated!")
                    st.write(feedback)
        else:
            st.warning("Please enter some text first!")
# Document Analyzer
with tab2:
    st.subheader("Upload and Learn from a Document")
    uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
    if uploaded_file:
        if not GOOGLE_API_KEY:
            st.error("Google API key missing. Cannot summarize document.")
        else:
            with st.spinner("Extracting and summarizing content..."):
                if uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                else:
                    text = extract_text_from_docx(uploaded_file)

                summary = summarize_with_gemini(text)
                st.write("### 📖 Summary")
                st.write(summary)

                st.write("### Key Vocabulary (Extracted with spaCy)")
                doc = nlp(text)
                nouns = [token.text for token in doc if token.pos_ == "NOUN"]
                unique_nouns = list(set(nouns))
                st.write(", ".join(unique_nouns[:20]))
# Chat Tutor (Conversational Mode)
with tab3:
    st.subheader("Chat with your English Tutor")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", placeholder="Ask me about grammar, writing, or vocabulary...")

    if st.button("Send"):
        if user_input.strip():
            if not GOOGLE_API_KEY:
                st.error("Google API key missing. Cannot chat.")
            else:
                model = genai.GenerativeModel("gemini-2.5-pro")
                chat = model.start_chat(history=st.session_state.chat_history)
                response = chat.send_message(user_input)
                st.session_state.chat_history.append({"role": "user", "parts": user_input})
                st.session_state.chat_history.append({"role": "model", "parts": response.text})
                st.write(f"**Tutor:** {response.text}")
        else:
            st.warning("Type something to start chatting!")

    if st.session_state.chat_history:
        st.write("### Chat History")
        for msg in st.session_state.chat_history[-6:]:
            st.write(f"**{msg['role'].capitalize()}:** {msg['parts']}")
