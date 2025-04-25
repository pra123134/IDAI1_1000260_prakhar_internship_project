import streamlit as st
import google.generativeai as genai

# ✅ API Key Setup
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("⚠️ API Key missing. Please add it in Streamlit → Settings → Secrets.")
    st.stop()

# ✅ AI Response
def ask_ai(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# ✅ Module Topics
modules = {
    "Planning & Coordination": "Event checklists, timelines, and budgeting.",
    "Marketing & Promotion": "Using social media and PR for events.",
    "Risk Management": "Backups, safety, and legal considerations.",
}

# ✅ UI
st.title("🎓 Simple Event Manager (AI-Powered)")
st.sidebar.header("📚 Choose a Module")
choice = st.sidebar.selectbox("Select a Topic", list(modules.keys()))

st.markdown(f"### {choice}")
st.markdown(f"_{modules[choice]}_")

# ✅ Generate content
if st.button("Generate Case Study"):
    case = ask_ai(f"Create a short event management case study on: {choice}")
    st.subheader("📌 Case Study")
    st.write(case)

    st.subheader("💡 Hint")
    st.info(ask_ai(f"Give a hint for solving: {case}"))

    st.subheader("🎯 Strategy")
    st.write(ask_ai(f"How would a professional handle this situation?\n\n{case}"))

    st.subheader("📝 Summary")
    st.markdown(ask_ai(f"Summarize key ideas about: {choice}"))

    st.subheader("🎓 Quiz Question")
    st.markdown(ask_ai(f"Create 1 MCQ on {choice}, with 4 options and mark the correct one."))

    st.subheader("💬 Peer Prompt")
    st.info(ask_ai(f"Write a discussion question about {choice}."))

# ✅ Sidebar Extras
st.sidebar.markdown("---")
st.sidebar.info("🎓 Quiz & 💬 Discussion included after case study is generated.")
