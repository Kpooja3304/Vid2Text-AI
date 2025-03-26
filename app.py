import os
import tempfile
from flask import Flask, render_template, request, jsonify
import yt_dlp
from faster_whisper import WhisperModel
from transformers import pipeline
from deep_translator import GoogleTranslator

app = Flask(__name__)

# --- Language Options ---
LANGUAGES = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Chinese": "zh"
}

# --- Load Whisper Model (Optimized) ---
device = "cpu"  # Optimized for CPU
model_size = "medium"  # Improved accuracy with minimal memory impact
whisper_model = WhisperModel(model_size, device=device, compute_type="float16")

# --- Load Summarization Model ---
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

@app.route("/")
def index():
    return render_template("index.html", languages=LANGUAGES.keys())

@app.route("/process", methods=["POST"])
def process_video():
    try:
        data = request.json
        video_url = data.get("video_url")
        transcript_lang = data.get("transcript_lang")
        summary_lang = data.get("summary_lang")
        summary_format = data.get("summary_format")

        if not video_url:
            return jsonify({"error": "No video URL provided."}), 400

        print("🔄 Extracting audio...")

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_filename = "temp_audio.wav"
            audio_path = os.path.join(temp_dir, audio_filename)

            # --- Step 1: Download and Extract Audio ---
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    downloaded_filename = os.path.join(temp_dir, f"{info['id']}.wav")
                    if os.path.exists(downloaded_filename):
                        os.rename(downloaded_filename, audio_path)
                    else:
                        return jsonify({"error": "Downloaded audio file not found."}), 500
            except Exception as e:
                return jsonify({"error": f"Error extracting audio: {str(e)}"}), 500

            print("✅ Audio extracted successfully!")

            # --- Step 2: Transcribe Audio ---
            try:
                segments, _ = whisper_model.transcribe(audio_path, beam_size=1, language="en")  # Reduced beam_size for efficiency
                transcript_text = " ".join(segment.text for segment in segments).strip()
                if not transcript_text:
                    return jsonify({"error": "No text transcribed from the audio."}), 500
            except Exception as e:
                return jsonify({"error": f"Error during transcription: {str(e)}"}), 500

            print("✅ Transcription completed!")

            # --- Step 3: Translate Transcript ---
            try:
                translated_transcript = GoogleTranslator(source="auto", target=LANGUAGES[transcript_lang]).translate(transcript_text)
            except Exception as e:
                return jsonify({"error": f"Error during translation: {str(e)}"}), 500

            print("✅ Translation completed!")

            # --- Step 4: Summarize Transcript (Optimized) ---
            try:
                def chunk_text(text, max_words=300):  # Reduced chunk size for efficiency
                    words = text.split()
                    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)][:3]  # Max 3 chunks

                chunks = chunk_text(transcript_text)
                summary_text = ""
                for chunk in chunks:
                    summary_result = summarizer(chunk, max_length=120, min_length=50, do_sample=False)
                    summary_text += summary_result[0]["summary_text"] + " "
                
                if summary_format == "Bullet Points":
                    summary_text = "\n".join([f"- {sentence}" for sentence in summary_text.split(". ") if sentence])
                elif summary_format == "Key Highlights":
                    summary_text = "\n".join([f"✔ {sentence}" for sentence in summary_text.split(". ")[:5]])
            except Exception as e:
                return jsonify({"error": f"Error during summarization: {str(e)}"}), 500

            print("✅ Summarization completed!")

            # --- Step 5: Translate Summary ---
            try:
                translated_summary = GoogleTranslator(source="auto", target=LANGUAGES[summary_lang]).translate(summary_text)
            except Exception as e:
                return jsonify({"error": f"Error during summary translation: {str(e)}"}), 500

            print("✅ Summary translation completed!")

            return jsonify({
                "transcript_en": transcript_text,
                "transcript_selected": translated_transcript,
                "summary_en": summary_text,
                "summary_selected": translated_summary
            })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns a port
    app.run(host="0.0.0.0", port=port)
