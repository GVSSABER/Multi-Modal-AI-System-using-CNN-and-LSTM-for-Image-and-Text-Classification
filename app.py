import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# LOAD CNN MODEL (IMAGE)
# ----------------------------
cnn_model = tf.keras.models.load_model("cnn_model.h5")

cnn_labels = [
    "airplane","car","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# ----------------------------
# LOAD LSTM MODEL (TEXT)
# ----------------------------
lstm_model = tf.keras.models.load_model("text_model.h5")

# Load tokenizer (VERY IMPORTANT)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ----------------------------
# IMAGE PREDICTION FUNCTION
# ----------------------------
def predict_image(img):
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img)
    return cnn_labels[np.argmax(pred)]

# ----------------------------
# TEXT PREDICTION FUNCTION
# ----------------------------
def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=200)

    pred = lstm_model.predict(pad)

    if pred[0][0] > 0.5:
        return "Positive 😊"
    else:
        return "Negative 😞"

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🧠 CNN + LSTM Multi-Modal AI")

menu = st.radio("Choose Input Type:", ["Image", "Text"])

# ----------------------------
# IMAGE SECTION
# ----------------------------
if menu == "Image":
    st.header("🖼️ Image Classification")

    img_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Uploaded Image")

        if st.button("Predict Image"):
            result = predict_image(img)
            st.success("Prediction: " + result)

# ----------------------------
# TEXT SECTION
# ----------------------------
if menu == "Text":
    st.header("📝 Text Sentiment Analysis")

    text = st.text_input("Enter review")

    if text:
        if st.button("Predict Text"):
            result = predict_text(text)
            st.success("Result: " + result)