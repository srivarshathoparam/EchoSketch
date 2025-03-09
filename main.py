from fastapi import FastAPI, File, UploadFile
import whisper
import io
from pydub import AudioSegment

app = FastAPI()
model = whisper.load_model("base")

@app.get("/")
def read_root():
    return {"message": "Audio2Art API is running!"}

@app.post("/speech-to-text/")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        audio = AudioSegment.from_file(io.BytesIO(await file.read()))
        audio = audio.set_frame_rate(16000).set_channels(1)
        temp_filename = "temp_audio.wav"
        audio.export(temp_filename, format="wav")
        result = model.transcribe(temp_filename)
        return {"transcription": result["text"]}
    except Exception as e:
        return {"error": str(e)}
