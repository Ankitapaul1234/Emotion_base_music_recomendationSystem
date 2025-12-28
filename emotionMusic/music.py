import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import os

# =========================
# PATH CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "labels.npy")
EMOTION_PATH = os.path.join(BASE_DIR, "emotion.npy")  # optional, can store last emotion

# =========================
# LOAD MODEL & LABELS
# =========================
model = load_model(MODEL_PATH)
labels = np.load(LABEL_PATH)
print("Labels order:", labels)  # debug: check label order

# =========================
# MEDIAPIPE SETUP
# =========================
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
holistic = mp_holistic.Holistic()
drawing = mp.solutions.drawing_utils

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Emotion Based Music Recommender")
st.title("🎵 Emotion Based Music Recommender")

if "run" not in st.session_state:
    st.session_state["run"] = True

if "latest_emotion" not in st.session_state:
    st.session_state["latest_emotion"] = ""

# =========================
# VIDEO PROCESSOR CLASS
# =========================
class EmotionProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        data = []

        if results.face_landmarks:
            # Face landmarks
            for lm in results.face_landmarks.landmark:
                data.extend([lm.x - results.face_landmarks.landmark[1].x,
                             lm.y - results.face_landmarks.landmark[1].y])

            # Left hand
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    data.extend([lm.x - results.left_hand_landmarks.landmark[8].x,
                                 lm.y - results.left_hand_landmarks.landmark[8].y])
            else:
                data.extend([0.0] * 42)

            # Right hand
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    data.extend([lm.x - results.right_hand_landmarks.landmark[8].x,
                                 lm.y - results.right_hand_landmarks.landmark[8].y])
            else:
                data.extend([0.0] * 42)

            data = np.array(data).reshape(1, -1)

            # Predict probabilities
            pred_probs = model.predict(data, verbose=0)[0]
            pred_index = np.argmax(pred_probs)
            prediction = labels[pred_index]

            # DEBUG: print all probabilities
            print("=== Frame prediction probabilities ===")
            for i, p in enumerate(pred_probs):
                print(f"{labels[i]}: {p:.3f}")
            print(f"Predicted index: {pred_index}, label: {prediction}")

            # Display prediction on frame
            cv2.putText(img, prediction, (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Save current emotion to session_state for real-time use
            st.session_state["latest_emotion"] = prediction

            # Optional: also save to file
            np.save(EMOTION_PATH, np.array([prediction]))

        # Draw landmarks
        drawing.draw_landmarks(img, results.face_landmarks,
                               mp_holistic.FACEMESH_TESSELATION)
        drawing.draw_landmarks(img, results.left_hand_landmarks,
                               mp_hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(img, results.right_hand_landmarks,
                               mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================
# USER INPUTS
# =========================
lang = st.text_input("Language")
singer = st.text_input("Singer")

if lang and singer and st.session_state["run"]:
    webrtc_streamer(
        key="emotion",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=EmotionProcessor,
        async_processing=True
    )

# =========================
# BUTTON ACTION
# =========================
if st.button("🎶 Recommend me songs"):

    # ===== READ LATEST EMOTION FROM session_state =====
    latest_emotion = st.session_state.get("latest_emotion", "")
    if not latest_emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = True
    else:
        # YouTube playlist mapping
        youtube_links = {
            "happy": "https://www.youtube.com/playlist?list=YOUR_HAPPY_PLAYLIST_ID",
            "sad": "https://www.youtube.com/playlist?list=YOUR_SAD_PLAYLIST_ID",
            "angry": "https://www.youtube.com/playlist?list=YOUR_ANGRY_PLAYLIST_ID",
            "neutral": "https://www.youtube.com/playlist?list=YOUR_NEUTRAL_PLAYLIST_ID",
            "prayer": "https://www.youtube.com/playlist?list=YOUR_PRAYER_PLAYLIST_ID"
        }

        url = youtube_links.get(latest_emotion,
                                f"https://www.youtube.com/results?search_query={lang}+{latest_emotion}+song+{singer}")
        webbrowser.open(url)

        # Reset emotion
        st.session_state["latest_emotion"] = ""
        np.save(EMOTION_PATH, np.array([""]))
        st.session_state["run"] = False





