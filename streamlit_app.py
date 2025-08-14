import streamlit as st
from config import supabase
import tempfile
import os
from collections import Counter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
import pandas as pd
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Tutor App", layout="wide", page_icon="📘")

# ---------------- THEME ----------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0 if st.session_state.theme=="light" else 1)
st.session_state.theme = theme_choice

bg_color = "#FFFFFF" if theme_choice=="Light" else "#111111"
text_color = "#111111" if theme_choice=="Light" else "#FFFFFF"
bubble_user = "#D1E8FF" if theme_choice=="Light" else "#2B6CB0"
bubble_ai = "#F1F1F1" if theme_choice=="Light" else "#1A202C"
card_bg = "#F9F9F9" if theme_choice=="Light" else "#1A202C"

# ---------------- SESSION STATE ----------------
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.user_id = None
if "chains" not in st.session_state:
    st.session_state.chains = {}  # {course_name: chain}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}  # {course_name: [(q,a), ...]}
if "pdf_summaries" not in st.session_state:
    st.session_state.pdf_summaries = {}  # {course_name: {file_name: summary}}
if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = {}  # {course_name: {file_name: [questions]}}

# ---------------- LOGIN / SIGNUP ----------------
with st.sidebar:
    st.header("🔐 Login / Signup")
    choice = st.selectbox("Choose", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if choice == "Sign Up":
        if st.button("Create Account"):
            result = supabase.auth.sign_up({"email": email, "password": password})
            if result.user:
                st.success("Account created! Please login.")
            else:
                st.error(result.error.message if result.error else "Signup failed")
    else:
        if st.button("Login"):
            result = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if result.user:
                st.session_state.user = result.user
                st.session_state.user_id = result.user.id
                st.success(f"Logged in as {email}")
            else:
                st.error(result.error.message if result.error else "Login failed")

# ---------------- UPLOAD PDF ----------------
if st.session_state.user:
    st.subheader("📚 Upload a Class PDF")
    course_name = st.text_input("Course Name")
    uploaded_file = st.file_uploader("Upload Notes (PDF)", type=["pdf"])

    if uploaded_file and course_name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        path = f"{st.session_state.user_id}/{course_name}/{uploaded_file.name}"
        with open(tmp_path, "rb") as f:
            supabase.storage.from_("course-files").upload(path, f)
        url = supabase.storage.from_("course-files").get_public_url(path)
        st.success(f"✅ File uploaded! Access it [here]({url['publicUrl']})")

        supabase.table("courses").insert({
            "user_id": st.session_state.user_id,
            "course_name": course_name,
            "file_name": uploaded_file.name,
            "file_path": path,
            "file_url": url['publicUrl']
        }).execute()

        with st.spinner("Processing PDF with AI..."):
            loader = UnstructuredPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            vectorstore = FAISS.from_documents(chunks, embeddings)
            llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            st.session_state.chains[course_name] = chain
            st.session_state.chat_history[course_name] = []

            summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = summary_chain.run(docs)
            if course_name not in st.session_state.pdf_summaries:
                st.session_state.pdf_summaries[course_name] = {}
            st.session_state.pdf_summaries[course_name][uploaded_file.name] = summary
            st.session_state.quiz_history.setdefault(course_name, {})[uploaded_file.name] = []

        st.success(f"✅ AI Tutor ready for course: {course_name}")

# ---------------- MULTI-COURSE CHAT ----------------
if st.session_state.chains:
    st.subheader("🔍 Search Courses")
    search_text = st.text_input("Search courses by name")
    available_courses = list(st.session_state.chains.keys())
    if search_text:
        available_courses = [c for c in available_courses if search_text.lower() in c.lower()]

    if available_courses:
        selected_course = st.selectbox("Choose course", available_courses)
        chain = st.session_state.chains[selected_course]

        # ---------------- COURSE ANALYTICS ----------------
        num_questions = len(st.session_state.chat_history.get(selected_course, []))
        st.info(f"📊 Questions asked in {selected_course}: {num_questions}")

        all_questions = [q for q, _ in st.session_state.chat_history.get(selected_course, [])]
        trending = Counter(all_questions).most_common(5)
        if trending:
            st.subheader("🔥 Trending Topics")
            for topic, count in trending:
                st.markdown(f"- {topic} ({count} times)")

        # ---------------- PDF CARDS ----------------
        st.subheader("📄 Uploaded PDFs")
        course_files = supabase.table("courses").select("*") \
            .eq("user_id", st.session_state.user_id) \
            .eq("course_name", selected_course).execute().data

        for file in course_files:
            with st.expander(f"{file['file_name']}", expanded=False):
                st.markdown(f"[Open PDF]({file['file_url']})")
                summary = st.session_state.pdf_summaries.get(selected_course, {}).get(file['file_name'], "")
                st.markdown(f"**Key Points:** {summary}")
                if st.button(f"Generate Quiz for {file['file_name']}", key=f"quiz-{file['file_name']}"):
                    st.session_state.quiz_history[selected_course][file['file_name']] = [
                        f"Q1: {summary[:50]}... ?",
                        f"Q2: {summary[50:100]}... ?",
                        f"Q3: {summary[100:150]}... ?"
                    ]
                    st.success(f"✅ Quiz generated for {file['file_name']}!")

        # ---------------- DYNAMIC CHAT PANEL ----------------
        st.subheader(f"💬 Ask Your Tutor - {selected_course}")
        chat_input = st.empty()
        chat_container = st.container()

        query = chat_input.text_input("Type a question", key="dynamic_chat_input")
        if query:
            with st.spinner("AI is typing..."):
                result = chain({
                    "question": query,
                    "chat_history": st.session_state.chat_history[selected_course]
                })
                answer = ""
                for char in result["answer"]:
                    answer += char
                    chat_container.markdown(
                        f"<div style='background-color:{bubble_ai};padding:10px;margin:5px;border-radius:10px;'><b>AI:</b> {answer}</div>",
                        unsafe_allow_html=True
                    )
                    time.sleep(0.01)
                st.session_state.chat_history[selected_course].append((query, result["answer"]))
                chat_input.text_input("Type a question", value="", key="dynamic_chat_input")  # Clear input

        # Auto-scroll container
        st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

        # Display full chat history collapsibly
        st.markdown("### Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history[selected_course]):
            with st.expander(f"Message {i+1}"):
                st.markdown(f"<div style='background-color:{bubble_user};padding:10px;margin:5px;border-radius:10px;'><b>You:</b> {q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color:{bubble_ai};padding:10px;margin:5px;border-radius:10px;'><b>AI:</b> {a}</div>", unsafe_allow_html=True)

        # ---------------- EXPORT ----------------
        if st.button("Export Full Data CSV"):
            data = []
            for q, a in st.session_state.chat_history[selected_course]:
                data.append({"Type": "Chat", "Question": q, "Answer": a})
            for fname, summary in st.session_state.pdf_summaries.get(selected_course, {}).items():
                data.append({"Type": "Summary", "Question": f"{fname}", "Answer": summary})
            for fname, quiz in st.session_state.quiz_history.get(selected_course, {}).items():
                for q in quiz:
                    data.append({"Type": "Quiz", "Question": q, "Answer": ""})
            df = pd.DataFrame(data)
            df.to_csv(f"{selected_course}_full_export.csv", index=False)
            st.success(f"✅ Exported full CSV for {selected_course}")
