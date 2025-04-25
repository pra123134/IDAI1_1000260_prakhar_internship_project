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
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text.strip():
            return response.text.strip()
        else:
            return fallback
    except Exception as e:
        return f"⚠️ Error: {str(e)}\n{fallback}"

# Function to show the personalized menu
def show_personalized_menu(preferences):
    prompt = f"Suggest a personalized menu for a customer with dietary preferences: {preferences}. Include dishes with ingredients and nutritional information."
    return get_ai_response(prompt)

# Main Page for Restaurant Ordering
def order_page():
    st.title("Smart Restaurant Menu")
    
    # Get customer dietary preferences
    preferences = st.text_input("Enter your dietary preferences (e.g., vegan, gluten-free, low-carb):")
    
    if preferences:
        # Show the personalized menu based on customer preferences
        st.subheader(f"Menu Suggestions for {preferences} Diet:")
        personalized_menu = show_personalized_menu(preferences)
        st.write(personalized_menu)

    # Add functionality for placing the order
    order = st.selectbox("Choose your dish:", ["Pizza", "Burger", "Pasta", "Salad", "Soup"])
    quantity = st.number_input("Quantity", min_value=1, value=1)
    
    if st.button("Place Order"):
        # Placeholder for order processing (could connect to payment API)
        total_cost = quantity * 12.99  # Example cost per dish
        st.write(f"Order Summary: {quantity} x {order} = ${total_cost:.2f}")
        st.write("⚡ Your order is being processed!")
        st.write("Thank you for ordering! 🎉")
        
# Main page to welcome and guide users
def main():
    st.sidebar.title("Restaurant Features")
    menu = ["Order Food", "View Personalized Recommendations"]
    choice = st.sidebar.radio("Select a page", menu)

    if choice == "Order Food":
        order_page()
    elif choice == "View Personalized Recommendations":
        preferences = st.text_input("Enter dietary preferences:")
        if preferences:
            personalized_recommendations = show_personalized_menu(preferences)
            st.write(personalized_recommendations)

if __name__ == "__main__":
    main()
