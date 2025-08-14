import streamlit as st
from config import supabase  # Supabase client we set up
import tempfile

st.set_page_config(page_title="AI Tutor App", layout="wide")
st.title("✅ AI Tutor App")

if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.user_id = None

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

if st.session_state.user:
    st.subheader("📚 Upload a Class PDF")
    course_name = st.text_input("Course Name")
    uploaded_file = st.file_uploader("Upload Notes", type=["pdf"])

    if uploaded_file and course_name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        path = f"{st.session_state.user_id}/{course_name}/{uploaded_file.name}"
        with open(tmp_path, "rb") as f:
            supabase.storage.from_("course-files").upload(path, f)

        url = supabase.storage.from_("course-files").get_public_url(path)
        st.success(f"✅ File uploaded! Access it [here]({url['publicUrl']})")
