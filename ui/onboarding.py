import streamlit as st # type: ignore
from db.db import SessionLocal
from db.models import User

def onboarding_form():
    st.title("🌐 Welcome to Personalized AI, Zen-Buddy")
    st.subheader("Please provide your details before we start chatting.")

    with st.form("user_form"):
        name = st.text_input("Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email")
        country = st.text_input("Country")
        state = st.text_input("State")

        submitted = st.form_submit_button("Continue")

    if submitted:
        language = choose_language(country, state)
        save_user(name, phone, email, country, state, language)
        st.success(f"Welcome {name}! We'll chat with you in **{language}**.")
        st.session_state["user_info"] = {
            "name": name,
            "phone": phone,
            "email": email,
            "country": country,
            "state": state,
            "language": language
        }

def choose_language(country, state):
    # Simple language mapping logic
    if country.lower() == "india":
        if state.lower() in ["karnataka"]:
            return "Kannada"
        elif state.lower() in ["tamil nadu"]:
            return "Tamil"
        elif state.lower() in ["maharashtra"]:
            return "Marathi"
        else:
            return "Hindi"
    return "English"

def save_user(name, phone, email, country, state, language):
    session = SessionLocal()
    user = User(name=name, phone=phone, email=email, country=country, state=state, language=language)
    session.add(user)
    session.commit()
    session.close()
