import os
import tempfile
import whisper
import yt_dlp
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load Whisper ASR model
model = whisper.load_model("small")  # Using "small" model to optimize memory

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/")
def home():
    return "Video Transcript Summarizer API is running!"

def download_audio(youtube_url):
    """Download audio from YouTube and return the file path."""
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "audio.mp3")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    return output_path

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    youtube_url = data.get("url")

    if not youtube_url:
        return jsonify({"error": "No YouTube URL provided"}), 400

    try:
        audio_path = download_audio(youtube_url)
        result = model.transcribe(audio_path)
        transcript = result["text"]

        summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]

        return jsonify({"transcript": transcript, "summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
