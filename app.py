import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load("E:\PYTHON\projects\SPAM\models\spam_model.pkl")



# Streamlit UI setup
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

st.title("📧 Spam vs Ham Classifier")
st.write("Paste any email or message below, and I'll predict whether it's spam or not with confidence levels.")

# Text input
user_input = st.text_area("Enter the message:", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([user_input])[0]
        probs = model.predict_proba([user_input])[0]  # [ham_prob, spam_prob]

        ham_prob, spam_prob = probs[0], probs[1]

        # Show label with confidence
        if prediction == "spam":
            st.error(f"🚨 Prediction: **SPAM** ({spam_prob*100:.2f}% confident)")
        else:
            st.success(f"✅ Prediction: **HAM** ({ham_prob*100:.2f}% confident)")

        # Display probability bars
        st.subheader("Prediction Probabilities")
        st.progress(int(ham_prob*100))
        st.write(f"HAM: {ham_prob*100:.2f}%")

        st.progress(int(spam_prob*100))
        st.write(f"SPAM: {spam_prob*100:.2f}%")
        st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; font-size: 14px; color: grey;">
        Designed by <b>Harshal</b>
    </div>
    """,
    unsafe_allow_html=True
)