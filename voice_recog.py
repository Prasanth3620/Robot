import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample data
data = {
    'command': [
        "Move to the left", "Turn right", "Step forward", "Go back", "Stop moving",
        "Turn to the right", "Move backward", "Go forward", "Halt", "Move left",
        "Shift to the left", "Step back", "Walk forward", "Reverse", "Don't move",
        "Pivot right", "Slide backward", "March ahead", "Hold position", "Turn left",
        "Lean to the left", "Shift back", "Step ahead", "Retreat", "Cease movement",
        "Swing right", "Step to the rear", "Advance", "Freeze", "Rotate left",
        "Drift left", "Back off", "Stride forward", "Withdraw", "Stay still",
        "Twist right", "Roll backward", "Move straight", "No motion", "Bend left",
        "Veer right", "Take a step back", "Head forward", "Step aside", "Be still",
        "Angle right", "Tread backward", "Proceed forward", "Remain motionless", "Bank left"
    ],
    'action': [
        "left", "right", "forward", "backward", "stop",
        "right", "backward", "forward", "stop", "left",
        "left", "backward", "forward", "backward", "stop",
        "right", "backward", "forward", "stop", "left",
        "left", "backward", "forward", "backward", "stop",
        "right", "backward", "forward", "stop", "left",
        "left", "backward", "forward", "backward", "stop",
        "right", "backward", "forward", "stop", "left",
        "right", "backward", "forward", "left", "stop",
        "right", "backward", "forward", "stop", "left"
    ]
}

df = pd.DataFrame(data)

# Data preprocessing
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['command'])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['action'])

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

# Get the unique classes in y_test
unique_classes = np.unique(y_test)

# Get the corresponding class names
unique_class_names = label_encoder.inverse_transform(unique_classes)

# Generate the classification report
print(classification_report(y_test, y_pred, target_names=unique_class_names))

# Function to record audio using sounddevice
def record_audio(duration=5, fs=44100):
    print("Recording audio for {} seconds...".format(duration))
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until the recording is finished
    return fs, audio

# Function to save the recorded audio to a WAV file
def save_wav(filename, fs, audio):
    wav.write(filename, fs, np.int16(audio * 32767))

# Function to recognize speech from a WAV file using speech_recognition
def recognize_speech_from_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        text = ""
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        return text

# Predict action based on voice command
def predict_action(command):
    command_vector = vectorizer.transform([command])
    action_label = model.predict(command_vector)
    return label_encoder.inverse_transform(action_label)[0]

# Main function to integrate everything
if __name__ == "__main__":
    # Record audio
    fs, audio = record_audio(duration=5)
    audio_filename = 'recorded_audio.wav'
    save_wav(audio_filename, fs, audio)

    # Recognize speech from the recorded audio file
    recognized_text = recognize_speech_from_audio(audio_filename)
    if recognized_text:
        if recognized_text.lower().startswith("hey siri"):
            command = recognized_text[len("Hey Siri"):].strip()  # Remove "Hey Siri" from the command
            action = predict_action(command)
            print(f"Predicted Action: {action}")
        else:
            print("Command ignored. Please start with 'Hey Siri'.")