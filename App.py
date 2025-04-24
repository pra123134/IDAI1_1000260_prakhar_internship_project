import streamlit as st
import google.generativeai as genai
import time

# ✅ Secure API Key Setup
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API Key is missing. Add your key to Streamlit Cloud → Settings → Secrets.")
    st.stop()

# 🔁 AI Utility
def get_ai_response(prompt, fallback="⚠️ AI response unavailable. Try again later.", max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            model = genai.GenerativeModel("gemini-2.5-pro")
            response = model.generate_content(prompt)
            return response.text.strip() if hasattr(response, "text") and response.text.strip() else fallback
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                return f"⚠️ Error: {str(e)}\n{fallback}"
            time.sleep(2)  # Wait before retrying to avoid hitting the API too quickly

# 🔁 Scenario Generators with caching
@st.cache_data
def generate_case_study(module_topic):
    return get_ai_response(f"Create a detailed event management case study for the topic: {module_topic}. Include realistic planning, coordination, stakeholder, and risk management challenges.")

@st.cache_data
def generate_hint(scenario):
    return get_ai_response(f"Provide a short hint to handle this event management scenario:\n\n{scenario}")

@st.cache_data
def generate_guidance(scenario):
    prompt = (
        f"Event Management Scenario:\n"
        f"{scenario}\n\n"
        f"You are a professional event manager. Provide:\n"
        f"- Step-by-step strategy to resolve the situation\n"
        f"- Key decisions to make\n"
        f"- Industry best practices\n"
        f"- Common pitfalls\n"
        f"- A critical thinking question for students"
    )
    return get_ai_response(prompt)

@st.cache_data
def generate_summary_notes(topic):
    return get_ai_response(f"Summarize the essential strategies and concepts in event management for the topic: {topic}. Use concise bullet points.")

@st.cache_data
def generate_quiz_question(topic):
    return get_ai_response(f"Create one MCQ quiz question with 4 options and the correct answer clearly marked for the topic: {topic}.")

@st.cache_data
def generate_peer_prompt(topic):
    return get_ai_response(f"Write a peer discussion prompt related to an event management topic: {topic}. It should invite open-ended responses and reflections.")

# ✅ Master Course Structure
modules = {
    "Introduction to Event Management": "Scope, types, and career pathways in event planning and coordination.",
    "Planning & Coordination": "Timelines, checklists, budgeting, and logistics.",
    "Venue & Vendor Selection": "Negotiations, contracts, permits, and location planning.",
    "Marketing & Promotion": "Campaign planning, social media strategies, and public relations.",
    "Stakeholder & Client Communication": "Briefing clients, managing expectations, and post-event feedback.",
    "Risk Management & Contingency Planning": "Emergency preparedness, legal considerations, and backup plans.",
    "Team & Volunteer Management": "Role assignment, team briefing, motivation, and coordination.",
    "Sustainable & Inclusive Events": "Eco-friendly practices and ensuring diversity and accessibility.",
}

# ✅ UI
st.title("🎓 Event Manager Master Course (AI-Enhanced)")
st.sidebar.header("📚 Event Management Modules")
selected_module = st.sidebar.selectbox("Select a Module", list(modules.keys()))

st.subheader(f"📘 {selected_module}")
st.markdown(f"_{modules[selected_module]}_")

# Case Study Generator
if st.button("🎯 Generate Event Case Study"):
    st.session_state.case_study = generate_case_study(selected_module)

if "case_study" in st.session_state:
    st.markdown("---")
    st.subheader("📌 Event Management Case Study")
    st.write(st.session_state.case_study)

    st.subheader("💡 Hint from AI")
    st.info(generate_hint(st.session_state.case_study))

    st.subheader("🧠 AI Strategy Guide")
    st.write(generate_guidance(st.session_state.case_study))

    st.subheader("📝 Reflection Journal")
    user_reflection = st.text_area("How would you handle this scenario? Relate it to real-world experience or theory.", height=150)

    st.subheader("📒 Summary Notes")
    summary = generate_summary_notes(selected_module)
    st.markdown(summary)

    st.subheader("🎓 Certification Quiz")
    quiz = generate_quiz_question(selected_module)
    st.markdown(quiz)

    st.subheader("💬 Peer Discussion Prompt")
    discussion_prompt = generate_peer_prompt(selected_module)
    st.info(discussion_prompt)

# 🚧 Coming Soon
st.sidebar.markdown("---")
st.sidebar.info(""" 
🎓 Certification Quiz: Practice with MCQs and reflections.
💬 Peer Discussion: Invite open-ended insights from classmates.
""")
