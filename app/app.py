from flask import Flask, request, jsonify, render_template
import whisper
import openai
import uuid
import json
import os
import sqlite3
from transformers import pipeline
import librosa
import numpy as np
np.int = int
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

app = Flask(__name__)

# Load Whisper model for transcription
model = whisper.load_model("base")

# Set up sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# At the beginning of your app.py
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# SQLite database
DATABASE = "database.db"

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS VoiceResponses (
            id TEXT PRIMARY KEY,
            name TEXT,
            college TEXT,
            passout_year INTEGER,
            transcription TEXT,
            sentiment TEXT,
            feedback TEXT,
            audio_features TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/interview_prep')
def interview_prep():
    return render_template('interview_prep.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    
    print("Files received:", request.files)
    print("Form data received:", request.form)

    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio data uploaded"}), 400

    audio_file = request.files['audio_data']
    name = request.form['name']
    college = request.form['college']
    passout = int(request.form['passout'])

    # Save audio file locally
    file_name = f"{uuid.uuid4()}.wav"
    audio_path = os.path.join("uploads", file_name)
    audio_file.save(audio_path)
     
    # Analyze audio features
    audio_features = analyze_audio(audio_path)

    # Transcribe audio
    transcription_text = transcribe_audio(audio_path)

    # Analyze transcription sentiment
    sentiment = analyze_sentiment(transcription_text)

    # Generate feedback using OpenAI
    feedback = generate_feedback(transcription_text, sentiment)

    # Save to SQLite Database
    save_to_database(name, college, passout, transcription_text, sentiment, feedback, audio_features)

    # Convert audio features dictionary to an HTML table
    audio_features_table = """
    <div style="display: flex; justify-content: center;">
        <table border="1" cellpadding="10" cellspacing="0">
            <tr><th>Feature</th><th>Value</th><th>Interpretation</th></tr>
            <tr><td>Average Pitch</td><td>{:.2f}</td><td>The average frequency of sound, indicating how high or low the sound is.</td></tr>
            <tr><td>Mean Energy</td><td>{:.3f}</td><td>The average amplitude of sound, reflecting overall loudness.</td></tr>
            <tr><td>Energy</td><td>{:.3f}</td><td>Overall intensity of the sound, which can indicate emphasis or volume.</td></tr>
            <tr><td>Spectral Centroid</td><td>{:.2f}</td><td>Center of the sound spectrum, giving a sense of the brightness of the sound.</td></tr>
            <tr><td>Tempo</td><td>{:.2f}</td><td>The speed of the audio, measured in beats per minute.</td></tr>
        </table>
   </div>
   """.format(
        audio_features.get("average_pitch", 0),
        audio_features.get("mean_energy", 0),
        audio_features.get("energy", 0),
        audio_features.get("spectral_centroid", 0),
        audio_features.get("tempo", 0)
    )
    audio_features_table = audio_features_table.replace('\n', '')
    audio_features_table = audio_features_table.replace('"', '')
    return jsonify({
        'transcription': transcription_text,
        'sentiment': sentiment,
        'feedback': feedback,
        'audio_features': audio_features_table  # Send as an HTML table string
    })

def analyze_audio(audio_path):
    # Librosa-based analysis
    y, sr = librosa.load(audio_path, sr=None)
    pitch, _ = librosa.core.piptrack(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    energy = librosa.feature.rms(y=y)
    
    # Convert librosa features to standard Python types
    avg_pitch = float(pitch.mean())
    avg_spectral_centroid = float(spectral_centroid.mean())
    avg_energy = float(energy.mean())
    tempo = float(tempo)


    # pyAudioAnalysis - Emotion Detection
    [Fs, x] = audioBasicIO.readAudioFile(audio_path)
    F, _ = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.05*Fs, 0.025*Fs)
    emotion_features = {
        'mean_energy': float(F[1].mean()),  # Example feature
    }

    audio_features = {
        'average_pitch': avg_pitch,
        'tempo': tempo,
        'spectral_centroid': avg_spectral_centroid,
        'energy': avg_energy,
        'emotion_features': emotion_features
    }
    return audio_features

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

def analyze_sentiment(text):
    analysis = sentiment_analysis(text)
    sentiment = analysis[0]["label"]
    return sentiment

def generate_feedback(transcription_text, sentiment):
    prompt = f"Provide feedback for the following transcription: {transcription_text} with a sentiment of {sentiment}."
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    
    feedback = response['choices'][0]['message']['content']
    return feedback

# Endpoint for Interview Prep to get a response from OpenAI
@app.route('/ask_openai', methods=['POST'])
def ask_openai():
    data = request.get_json()
    question = data.get('question')

    # Use OpenAI API to get the response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a interview preparation assistanthelping to assist for preparing a job at Quantium"},
            {"role": "user", "content": question}
        ],
        max_tokens=100
    )

    answer = response['choices'][0]['message']['content'].strip()
    return jsonify({"answer": answer})

def save_to_database(name, college, passout, transcription, sentiment, feedback, audio_features):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO VoiceResponses (id, name, college, passout_year, transcription, sentiment, feedback, audio_features)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (id, name, college, passout, transcription, sentiment, feedback, json.dumps(audio_features)))
    conn.commit()
    conn.close()

#if __name__ == '__main__':
#    app.run(debug=True, host='0.0.0.0')
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

