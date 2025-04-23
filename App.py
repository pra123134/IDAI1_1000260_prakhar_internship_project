import streamlit as st
import google.generativeai as genai
import tempfile
import pdfkit
import os

# ✅ Secure API Key Setup
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("\u26a0\ufe0f API Key is missing. Add your key to Streamlit Cloud \u2192 Settings \u2192 Secrets.")
    st.stop()

# 🔁 AI Utility
def get_ai_response(prompt, fallback="\u26a0\ufe0f AI response unavailable. Try again later."):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip() if hasattr(response, "text") and response.text.strip() else fallback
    except Exception as e:
        return f"\u26a0\ufe0f Error: {str(e)}\n{fallback}"

# 🔁 Scenario Generators
def generate_case_study(module_topic):
    return get_ai_response(f"Create a detailed event management case study for the topic: {module_topic}. Include realistic planning, coordination, stakeholder, and risk management challenges.")

def generate_hint(scenario):
    return get_ai_response(f"Provide a short hint to handle this event management scenario:\n\n{scenario}")

def generate_guidance(scenario):
    return get_ai_response(f"""
    Event Management Scenario:
    {scenario}

    You are a professional event manager. Provide:
    - Step-by-step strategy to resolve the situation
    - Key decisions to make
    - Industry best practices
    - Common pitfalls
    - A critical thinking question for students
    """)

def generate_summary_notes(topic):
    return get_ai_response(f"Summarize the essential strategies and concepts in event management for the topic: {topic}. Use concise bullet points.")

def generate_quiz_question(topic):
    return get_ai_response(f"Create one MCQ quiz question with 4 options and the correct answer clearly marked for the topic: {topic}.")

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
st.title("\ud83c\udf93 Event Manager Master Course (AI-Enhanced)")
st.sidebar.header("\ud83d\udcda Event Management Modules")
selected_module = st.sidebar.selectbox("Select a Module", list(modules.keys()))

st.subheader(f"\ud83d\udcd8 {selected_module}")
st.markdown(f"_{modules[selected_module]}_")

# Case Study Generator
if st.button("\ud83c\udfaf Generate Event Case Study"):
    st.session_state.case_study = generate_case_study(selected_module)

if "case_study" in st.session_state:
    st.markdown("---")
    st.subheader("\ud83d\udccc Event Management Case Study")
    st.write(st.session_state.case_study)

    st.subheader("\ud83d\udca1 Hint from AI")
    st.info(generate_hint(st.session_state.case_study))

    st.subheader("\ud83e\uddd0 AI Strategy Guide")
    st.write(generate_guidance(st.session_state.case_study))

    st.subheader("\ud83d\udcdd Reflection Journal")
    user_reflection = st.text_area("How would you handle this scenario? Relate it to real-world experience or theory.", height=150)

    st.subheader("\ud83d\udcd2 Summary Notes")
    summary = generate_summary_notes(selected_module)
    st.markdown(summary)

    # PDF Export
    if st.button("\ud83d\udcc5 Export Case Study & Notes as PDF"):
        pdf_content = f"""
        <h1>{selected_module}</h1>
        <h2>Case Study</h2>
        <p>{st.session_state.case_study}</p>
        <h2>Hint</h2>
        <p>{generate_hint(st.session_state.case_study)}</p>
        <h2>Strategy</h2>
        <p>{generate_guidance(st.session_state.case_study)}</p>
        <h2>Reflection</h2>
        <p>{user_reflection}</p>
        <h2>Summary Notes</h2>
        <p>{summary}</p>
        """

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdfkit.from_string(pdf_content, tmpfile.name)
            with open(tmpfile.name, "rb") as file:
                st.download_button(
                    label="\ud83d\udcc4 Download PDF",
                    data=file.read(),
                    file_name="event_case_study.pdf",
                    mime="application/pdf"
                )
            os.remove(tmpfile.name)

    st.subheader("\ud83c\udf93 Certification Quiz")
    quiz = generate_quiz_question(selected_module)
    st.markdown(quiz)

    st.subheader("\ud83d\udcac Peer Discussion Prompt")
    discussion_prompt = generate_peer_prompt(selected_module)
    st.info(discussion_prompt)

# \ud83d\udea7 Coming Soon
st.sidebar.markdown("---")
st.sidebar.info("""
\ud83c\udf93 Certification Quiz: Practice with MCQs and reflections.
\ud83d\udcac Peer Discussion: Invite open-ended insights from classmates.
""")
