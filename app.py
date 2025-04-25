import streamlit as st
import google.generativeai as genai

# ✅ Secure API Key Setup
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("⚠️ API Key is missing. Add your key in Streamlit → Settings → Secrets.")
    st.stop()

# 🔁 AI Utility
def get_ai_response(prompt, fallback="⚠️ AI response unavailable. Try again later."):
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text.strip():
            return response.text.strip()
        else:
            return fallback
    except Exception as e:
        return f"⚠️ Error: {str(e)}\n{fallback}"

# 🔁 Scenario Generators
def generate_case_study(topic):
    return get_ai_response(f"Create a detailed event management case study for the topic: {topic}. Include realistic planning, coordination, stakeholder, and risk management challenges.")

def generate_hint(scenario):
    return get_ai_response(f"Provide a short hint to handle this event management scenario:\n\n{scenario}")

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
st.set_page_config(page_title="Event Manager AI Course", layout="centered")
st.title("🎓 Event Manager Master Course (AI-Enhanced)")

st.sidebar.header("📚 Event Management Modules")
selected_module = st.sidebar.selectbox("Select a Module", list(modules.keys()))

# Sidebar bottom section — ADDED here
st.sidebar.markdown("---")
st.sidebar.info("""
🎓 Certification Quiz: Practice with MCQs and reflections.  
💬 Peer Discussion: Invite open-ended insights from classmates.
""")

# Module Info
st.subheader(f"📘 {selected_module}")
st.markdown(f"_{modules[selected_module]}_")

# ✅ Initialize session state
if "case_study" not in st.session_state:
    st.session_state.case_study = ""

# Case Study Generator
if st.button("🎯 Generate Event Case Study"):
    st.session_state.case_study = generate_case_study(selected_module)

if st.session_state.case_study:
    st.markdown("---")
    st.subheader("📌 Event Management Case Study")
    st.write(st.session_state.case_study)

    st.subheader("💡 Hint from AI")
    st.info(generate_hint(st.session_state.case_study))

    st.subheader("🧠 AI Strategy Guide")
    st.write(generate_guidance(st.session_state.case_study))

    st.subheader("📝 Reflection Journal")
    st.text_area(
        "How would you handle this scenario? Relate it to real-world experience or theory.",
        key="reflection",
        height=150
    )

    st.subheader("📒 Summary Notes")
    st.markdown(generate_summary_notes(selected_module))

    st.subheader("🎓 Certification Quiz")
    st.markdown(generate_quiz_question(selected_module))

    st.subheader("💬 Peer Discussion Prompt")
    st.info(generate_peer_prompt(selected_module))
import streamlit as st
import google.generativeai as genai

# ✅ Secure API Key Setup
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("⚠️ API Key is missing. Add your key in Streamlit → Settings → Secrets.")
    st.stop()

# 🔁 AI Utility
def get_ai_response(prompt, fallback="⚠️ AI response unavailable. Try again later."):
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text.strip():
            return response.text.strip()
        else:
            return fallback
    except Exception as e:
        return f"⚠️ Error: {str(e)}\n{fallback}"

# 🔁 Scenario Generators
def generate_case_study(topic):
    return get_ai_response(f"Create a detailed event management case study for the topic: {topic}. Include realistic planning, coordination, stakeholder, and risk management challenges.")

def generate_hint(scenario):
    return get_ai_response(f"Provide a short hint to handle this event management scenario:\n\n{scenario}")

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
st.set_page_config(page_title="Event Manager AI Course", layout="centered")
st.title("🎓 Event Manager Master Course (AI-Enhanced)")

st.sidebar.header("📚 Event Management Modules")
selected_module = st.sidebar.selectbox("Select a Module", list(modules.keys()))

# Sidebar bottom section — ADDED here
st.sidebar.markdown("---")
st.sidebar.info("""
🎓 Certification Quiz: Practice with MCQs and reflections.  
💬 Peer Discussion: Invite open-ended insights from classmates.
""")

# Module Info
st.subheader(f"📘 {selected_module}")
st.markdown(f"_{modules[selected_module]}_")

# ✅ Initialize session state
if "case_study" not in st.session_state:
    st.session_state.case_study = ""

# Case Study Generator
if st.button("🎯 Generate Event Case Study"):
    st.session_state.case_study = generate_case_study(selected_module)

if st.session_state.case_study:
    st.markdown("---")
    st.subheader("📌 Event Management Case Study")
    st.write(st.session_state.case_study)

    st.subheader("💡 Hint from AI")
    st.info(generate_hint(st.session_state.case_study))

    st.subheader("🧠 AI Strategy Guide")
    st.write(generate_guidance(st.session_state.case_study))

    st.subheader("📝 Reflection Journal")
    st.text_area(
        "How would you handle this scenario? Relate it to real-world experience or theory.",
        key="reflection",
        height=150
    )

    st.subheader("📒 Summary Notes")
    st.markdown(generate_summary_notes(selected_module))

    st.subheader("🎓 Certification Quiz")
    st.markdown(generate_quiz_question(selected_module))

    st.subheader("💬 Peer Discussion Prompt")
    st.info(generate_peer_prompt(selected_module))
