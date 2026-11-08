"""Local SMS classification demo. Run with: python -m streamlit run app.py"""

import json
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="SMS Spam Detector", page_icon="✉️", layout="centered")


@st.cache_resource
def load_model():
    # Load only the trusted artifact bundled with this project.
    return joblib.load(ROOT / "models" / "sms_pipeline.joblib")


EXAMPLES = {
    "Write my own": "",
    "Everyday message": "Hey, are we still meeting for lunch tomorrow?",
    "Prize message": "Congratulations! You have won a FREE cash prize. Call now to claim your reward!",
}


def use_example():
    st.session_state["message"] = EXAMPLES[st.session_state["example"]]


st.title("✉️ SMS Spam Detector")
st.write("Paste an SMS message to check whether the model predicts spam or a legitimate message.")
st.caption("A local Streamlit demo powered by TF-IDF and Logistic Regression.")

try:
    pipeline = load_model()
except (FileNotFoundError, ModuleNotFoundError, ImportError, ValueError, EOFError):
    st.error("The saved model could not be loaded. Install requirements.txt and run python train_model.py, then restart the app.")
    st.stop()

st.selectbox("Start with an example", options=list(EXAMPLES), key="example", on_change=use_example)
with st.form("prediction_form"):
    message = st.text_area("SMS message", key="message", height=150, max_chars=5000,
                           placeholder="Type or paste your message here…")
    submitted = st.form_submit_button("Check message", type="primary")

if submitted:
    text = message.strip()
    if not text:
        st.warning("Enter a message before checking.")
    elif pipeline.named_steps["tfidf"].transform([text]).nnz == 0:
        st.warning("This message contains no terms recognized by the model. Try a longer English SMS message.")
    else:
        prediction = pipeline.predict([text])[0]
        spam_index = list(pipeline.classes_).index("spam")
        spam_score = float(pipeline.predict_proba([text])[0, spam_index])
        if prediction == "spam":
            st.warning("Prediction: Spam")
        else:
            st.success("Prediction: Legitimate message (ham)")
        st.metric("Model spam score", f"{spam_score:.1%}")
        st.caption("This is an uncalibrated model score, not a guarantee that the message is safe or unsafe.")

with st.expander("About this model"):
    st.write("The saved pipeline includes text preprocessing and the classifier. It was trained on the historical UCI SMS Spam Collection; newer spam, other languages, and unfamiliar words may behave differently.")
    st.write("A false positive flags a legitimate message; a false negative misses spam. Always use your judgment when reviewing a message.")
    metadata_path = ROOT / "models" / "model_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        st.write("Held-out test results (spam is the positive class):")
        st.json(metadata["test_metrics"])
    st.caption("Predictions run on your computer without an external prediction API. This app does not save entered messages.")
