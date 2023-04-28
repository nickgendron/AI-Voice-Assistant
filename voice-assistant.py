import os
from elevenlabs import generate, play, set_api_key
import whisper
import pyaudio
import io
import wave
import subprocess
import shlex
import openai

# set elevenlabs & OpenAI API key
set_api_key(os.environ.get("ELEVEN_LABS"))
openai.api_key = os.environ.get("OPEN_AI")

# Configuration parameters for PyAudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

p = pyaudio.PyAudio()

# Open a stream to record audio from the default system microphone
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

print("BEGIN SPEAKING")

# Record audio to an in-memory byte stream
frames = io.BytesIO()
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.write(data)
print("END SPEAKING")
p.close(stream)

# Convert the in-memory byte stream to a wave file
frames.seek(0)
with wave.open(frames, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(frames.getvalue())

print("[DEGUG] Processing speech to text")
# Use ffmpeg to convert the wave file to an m4a file
cmd = 'ffmpeg -y -i - -c:a aac -strict -2 output.m4a -hide_banner -loglevel error'
proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE)
proc.communicate(input=frames.getvalue())

# Call to OpenAI Whisper for speech-to-text processing
print("[DEGUG] Processing OpenAI Whisper")
model = whisper.load_model("tiny")
result = whisper.transcribe(model, "output.m4a")
options = whisper.DecodingOptions(language='en', fp16=True)
print(result["text"])

# Call to OpenAI ChatGPT with transcribed text from Whisper
print("[DEBUG] Awaiting ChatGPT response...")
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": result["text"]}
    ],
)

print(completion.choices[0].message.content)
gptResponse = completion.choices[0].message.content

# Call to Elevenlabs for text-to-speech processing
print("[DEBUG] Elevenlabs processing ChatGPT response...")
voice = "Bella"
audio = generate(gptResponse, voice=voice)
play(audio)

