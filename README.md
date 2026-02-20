Simple Transcription Pipeline

Project Overview
This project is a modular transcription service that converts various audio formats into text with precise timestamps. It is built using Python, FastAPI, and the Faster-Whisper engine.

Key Design Decisions
Faster-Whisper Implementation: I chose faster-whisper over the standard OpenAI version because it utilizes CTranslate2 for 8-bit quantization. This makes it significantly faster and allows it to run efficiently on CPUs without requiring expensive GPUs.

Normalization Gatekeeper: Using pydub and FFmpeg, I implemented a preprocessing step that standardizes all incoming audio to 16kHz Mono. This reduces Word Error Rate (WER) by ensuring the model receives a consistent signal.

Asynchronous-Ready Architecture: The code is structured to be easily extended into a background worker pattern (using Celery/Redis) for handling long-form audio and high concurrency.

Statelessness: The API does not store local state, making it easy to containerize with Docker and scale horizontally across multiple instances.

How to Run
Install FFmpeg on your system.

Install dependencies: pip install fastapi uvicorn faster-whisper pydub python-multipart

Run the server: python volga_main.py

Test the endpoint: Send a POST request to localhost:8000/transcribe with an audio file.
