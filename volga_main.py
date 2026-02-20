import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
from pydub import AudioSegment

app = FastAPI(title="Transcription Pipeline API")

model = WhisperModel("base", device="cpu", compute_type="int8")

def preprocess_audio(input_path: str) -> str:
    output_path = f"processed_{uuid.uuid4()}.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_input = f"temp_{file.filename}"
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        processed_path = preprocess_audio(temp_input)

        segments, info = model.transcribe(processed_path, beam_size=5)
        
        transcript_data = []
        for segment in segments:
            transcript_data.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })

        os.remove(temp_input)
        os.remove(processed_path)

        return {
            "language": info.language,
            "duration": round(info.duration, 2),
            "segments": transcript_data
        }

    except Exception as e:
        if os.path.exists(temp_input): os.remove(temp_input)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)