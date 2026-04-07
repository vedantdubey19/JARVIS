import os
import time
import math
import struct
import numpy as np
import pyaudio
import pyttsx3
import speech_recognition as sr
from openai import OpenAI
import pygame
import threading
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# ==============================================================================
# CONFIGURATION & SETUP
# ==============================================================================

# API Keys (Replace with your actual key or load from environment variable)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Audio Config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CLAP_THRESHOLD = 500  # Adjust this depending on your microphone's sensitivity!

# System States
is_listening = False

# Flask & SocketIO Server Initialization
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

@socketio.on('connect')
def test_connect():
    print('Dashboard connected via WebSockets')
    socketio.emit('system_log', {'log': '[SYS] Dashboard connected dynamically.'})

@socketio.on('send_command')
def handle_client_command(data):
    """Directly process a command from dashboard."""
    command = data.get('command', '')
    if command:
        print(f"WEB_CMD: {command}")
        socketio.emit('user_message', {'text': command})
        threading.Thread(target=process_command, args=(command,), daemon=True).start()

# Initialize Text-to-Speech (offline)
engine = pyttsx3.init()
voices = engine.getProperty('voices')
# Set to a recognizable voice (id 0 is often male, 1 is often female depending on OS)
if len(voices) > 0:
    engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 170)

# Initialize global OpenAI client (Lazy initialization in conversational function)
client = None
conversation_history = [
    {"role": "system", "content": "You are JARVIS, a highly capable, concise, and professional AI voice assistant. Keep your answers brief and tailored for text-to-speech output. You manage a user's local system."}
]

# Initialize music mixer
pygame.mixer.init()

# ==============================================================================
# MODULE 1: VOICE OUTPUT (TTS)
# ==============================================================================
def speak(text):
    """Converts text to speech and plays it audibly."""
    print(f"JARVIS: {text}")
    socketio.emit('ai_response', {'text': text})
    engine.say(text)
    engine.runAndWait()

# ==============================================================================
# MODULE 2: AI CONVERSATION ENGINE
# ==============================================================================
def get_ai_response(user_input):
    """Sends the user input to the LLM and returns the response."""
    global client
    if client is None:
        client = OpenAI(api_key=OPENAI_API_KEY)
    
    conversation_history.append({"role": "user", "content": user_input})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Model upgraded for better reasoning
            messages=conversation_history,
            max_tokens=150
        )
        ai_reply = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": ai_reply})
        
        # Keep history short to avoid context bloat
        if len(conversation_history) > 10:
            # Keep system prompt + last 8 messages
            conversation_history[:] = [conversation_history[0]] + conversation_history[-8:]
            
        return ai_reply
    except Exception as e:
        print(f"Error communicating with OpenAI: {e}")
        return "I'm sorry, I am having trouble connecting to my brain."

# ==============================================================================
# MODULE 3: MUSIC CONTROL SYSTEM
# ==============================================================================
def handle_music_command(command):
    """Executes music-related commands using pygame."""
    command = command.lower()
    
    music_folder = "./music" # Ensure this folder exists and has an mp3!
    os.makedirs(music_folder, exist_ok=True)
    
    # We grab the first mp3 in the folder
    songs = [f for f in os.listdir(music_folder) if f.endswith('.mp3')]
    
    if "play" in command:
        if not songs:
            speak("I couldn't find any MP3 files in the local music folder.")
            return True
        song_path = os.path.join(music_folder, songs[0])
        try:
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            speak("Playing music now, sir.")
        except Exception as e:
            speak("There was an error attempting to play the music.")
        return True
        
    elif "pause" in command:
        pygame.mixer.music.pause()
        speak("Music paused.")
        return True
        
    elif "resume" in command:
        pygame.mixer.music.unpause()
        speak("Resuming playback.")
        return True
        
    elif "stop" in command:
        pygame.mixer.music.stop()
        speak("Music stopped.")
        return True

    return False

# ==============================================================================
# MODULE 4: COMMAND HANDLING SYSTEM
# ==============================================================================
def process_command(command):
    """Evaluates the transcribed voice command and delegates it to modules."""
    print(f"You said: {command}")
    command_lower = command.lower()
    
    if "music" not in command_lower and "play" not in command_lower and "time" not in command_lower and "date" not in command_lower:
        socketio.emit('ai_state', {'state': 'thinking'})
    
    # 1. Music Controls
    if any(keyword in command_lower for keyword in ["music", "play", "pause", "resume", "stop"]):
        handled = handle_music_command(command_lower)
        if handled: return

    # 2. Basic utilities
    elif "time" in command_lower:
        current_time = time.strftime("%I:%M %p")
        speak(f"The current time is {current_time}.")
        return

    elif "date" in command_lower:
        current_date = time.strftime("%B %d, %Y")
        speak(f"Today is {current_date}.")
        return

    # 3. Fallback to conversational AI
    else:
        response = get_ai_response(command)
        speak(response)

# ==============================================================================
# MODULE 5: VOICE INPUT (SPEECH RECOGNITION)
# ==============================================================================
def listen_for_command():
    """Listens for a voice command and returns the transcribed text."""
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        # Adjust for ambient noise briefly
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening for your command...")
        
        try:
            # Listen until silence is detected
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Processing speech...")
            # Use Google Web Speech API (Free and no API key required)
            command = recognizer.recognize_google(audio)
            return command
        
        except sr.WaitTimeoutError:
            print("Listening timed out.")
            return None
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return None
        except sr.RequestError:
            print("Could not request results from the speech recognition service.")
            return "Speech recognition service is unreachable."

# ==============================================================================
# MODULE 6: CLAP DETECTION & MAIN LOOP
# ==============================================================================
def get_rms(block):
    """Calculates the Root Mean Square of an audio block."""
    # Convert binary audio data to a numpy array for fast vectorized operations
    indata = np.frombuffer(block, dtype=np.int16).astype(np.float32)
    # Mean of squared values
    return np.sqrt(np.mean(indata**2))

def activate_assistant():
    """Triggers the full assistant sequence when a clap is detected."""
    global is_listening
    if is_listening:
        return
        
    is_listening = True
    socketio.emit('system_log', {'log': '[SYS] Listening for voice command...'})
    speak("Yes sir?")
    
    # Pause the background music if it is playing, so JARVIS can hear better
    was_playing = pygame.mixer.music.get_busy()
    if was_playing:
        pygame.mixer.music.pause()
    
    # Wait and listen
    command = listen_for_command()
    
    if command:
        process_command(command)
        
    # Resume music if it was paused
    if was_playing and not ("stop" in str(command).lower() or "pause" in str(command).lower()):
        pygame.mixer.music.unpause()
        
    print("Going back to sleep. Waiting for a clap...")
    # Add a brief delay so the end of synthesis doesn't trigger the clap detector
    time.sleep(1)
    is_listening = False

def audio_loop():
    """Starts the continuous audio stream monitoring for claps."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
                    
    print("=" * 50)
    print("JARVIS Audio Engine Initialization Complete!")
    print(f"Listening for a loud clap (Threshold: {CLAP_THRESHOLD}).")
    print("=" * 50)

    try:
        while True:
            if is_listening:
                time.sleep(0.1)
                continue
            
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except IOError as e:
                continue
                
            rms = get_rms(data)
            
            if rms > CLAP_THRESHOLD:
                print(f"*** CLAP DETECTED! (Amplitude: {int(rms)}) ***")
                socketio.emit('system_log', {'log': '[SYS] Clap detected. Waking UI...'})
                threading.Thread(target=activate_assistant, daemon=True).start()
                time.sleep(1.0) 

    except Exception as e:
        print(f"\nShutting down Audio: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def main():
    audio_thread = threading.Thread(target=audio_loop, daemon=True)
    audio_thread.start()
    
    print("Starting Flask-SocketIO on port 5000...")
    socketio.run(app, host='0.0.0.0', port=5050, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
