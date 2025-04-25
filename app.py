import streamlit as st
import google.generativeai as genai

# ---- Gemini Setup ----
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Replace with your API key
model = genai.GenerativeModel("gemini-pro")

# ---- App UI ----
st.set_page_config(page_title="AI Case Studies App", layout="centered")
st.title("📚 AI Case Studies App")

# ---- User Input ----
subject = st.selectbox("Select a subject", ["Math", "Science", "History", "Literature"])
role = st.radio("Select your role", ["Student", "Teacher"])

# ---- Prompt Construction ----
if st.button("Generate Content"):
    with st.spinner("Generating with Gemini 1.5 Pro..."):
        if role == "Teacher":
            prompt = f"Give 2 real-world case study examples with detailed explanations for a teacher to use in class for the subject: {subject}."
        else:
            prompt = f"Give 2 student-friendly case studies with quiz questions at the end for the subject: {subject}. Make them interactive and easy to understand."

        try:
            response = model.generate_content(prompt)
            st.success("Generated successfully!")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"Error: {e}")
