import cv2
import numpy as np
import time
import os
import google.generativeai as genai
from PIL import Image
import speech_recognition as sr
import pyttsx3
import threading
import pvporcupine
import pyaudio
import struct # For audio data conversion

# secret
GOOGLE_API_KEY ="AIzaSyB0XEfsaSyN_2UvViONMwIN7QmQjYPtQ4k"
PICOVOICE_ACCESS_KEY = "DeKpc5vZsc6xSA3ipBNVWgsj/bDcve/7zJdhSwA3qexXx7Wd+Fh9Nw=="

# --- Configuration ---

STREAM_URL = 'http://192.168.40.253:8080/video'

# Google Gemini Config


# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') # Get key from environment

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set. Please set it before running.")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

# Picovoice Porcupine Config
# IMPORTANT: Use environment variables for access keys as well.
# PICOVOICE_ACCESS_KEY = os.getenv('PICOVOICE_ACCESS_KEY') # Get key from environment
if not PICOVOICE_ACCESS_KEY:
    print("Error: PICOVOICE_ACCESS_KEY environment variable not set. Please set it before running.")
    exit()

# Use a built-in keyword first for simplicity, e.g., "porcupine" or "hey siri"
# Or replace with the path to your custom .ppn file:
# KEYWORD_PATHS = ["path/to/your/Hey-Robo_....ppn"]
KEYWORD_PATHS = [pvporcupine.KEYWORD_PATHS["porcupine"]] # Using built-in "porcupine"

# --- State Definitions ---
class State:
    LISTENING_WW = "Listening for Wake Word"
    TRIGGERED = "Wake Word Detected"
    LISTENING_Q = "Listening for Query"
    PROCESSING = "Processing Query"
    SPEAKING = "Speaking Answer"
    ERROR = "Error State"

# --- Globals and Shared State ---
latest_frame = None
llm_answer = "System Initialized. Say 'Porcupine' to activate." # Initial message
current_status = State.LISTENING_WW # Start by listening for wake word

frame_lock = threading.Lock()
state_lock = threading.Lock()

stop_event = threading.Event() # To signal threads to stop

# --- Initialize Services ---
# Gemini Model (same as before)
generation_config = {"temperature": 0.4, "top_p": 1, "top_k": 32, "max_output_tokens": 1024}
safety_settings = [ # Adjust as needed
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- FIX: Change the model_name to a currently supported one ---
# The error indicated 'gemini-pro-vision' is deprecated.
# Use 'gemini-1.5-flash' or 'gemini-1.5-pro'.
try:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # Use the suggested, currently supported model
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    print(f"Successfully initialized model: gemini-1.5-flash")
except Exception as e:
    print(f"Error initializing Gemini model 'gemini-1.5-flash': {e}")
    # Optionally try 'gemini-1.5-pro' as a fallback
    try:
         model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
         print(f"Successfully initialized model: gemini-1.5-pro (as fallback)")
    except Exception as e2:
         print(f"Error initializing Gemini model 'gemini-1.5-pro' either: {e2}")
         print("Failed to initialize any supported Gemini Vision model. Exiting.")
         exit()


# Speech Recognition (will be initialized potentially later or within thread)
r = None
mic = None

# Text-to-Speech
tts_engine = pyttsx3.init()

# Porcupine Wake Word Engine (initialized in its thread)
porcupine = None
pa = None
audio_stream = None

# --- Utility Functions ---
def update_status(new_status):
    global current_status
    with state_lock:
        current_status = new_status
        # Clear the previous status line if using a simple console
        # print(f"\rStatus change: {current_status}", end='', flush=True)
        print(f"Status change: {current_status}") # Debug print for status changes


def get_current_status():
    with state_lock:
        return current_status

# --- Thread Functions ---

def wake_word_listener():
    """Listens for the wake word using Porcupine."""
    global porcupine, pa, audio_stream
    keyword_index = -1 # Default value

    try:
        # Check if stop_event is set before potentially initializing porcupine
        if stop_event.is_set():
            print("Wake word listener received stop signal before init.")
            return

        porcupine = pvporcupine.create(
            access_key=PICOVOICE_ACCESS_KEY,
            keyword_paths=KEYWORD_PATHS # Use the list of keyword file paths
            # You might need to specify model_path explicitly depending on your setup
        )
        print(f"Porcupine Initialized (Sample Rate: {porcupine.sample_rate}, Frame Length: {porcupine.frame_length}). Listening for '{KEYWORD_PATHS[0].split('/')[-1].replace('.ppn','')}'...") # Display keyword name

        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length)

        update_status(State.LISTENING_WW) # Ensure status is correct

        while not stop_event.is_set():
            # Only process audio if in the listening state
            if get_current_status() == State.LISTENING_WW:
                try:
                    pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                    keyword_index = porcupine.process(pcm)

                    if keyword_index >= 0:
                        print(f"\nDetected keyword (index {keyword_index})!") # Add newline after status print
                        # Wake word detected! Transition state
                        update_status(State.TRIGGERED)
                        # Optional: Play a sound?
                        # tts_engine.say("Yes?") # Might interfere with listening immediately
                        # tts_engine.runAndWait()
                        # Now wait for the query processor thread to take over
                        # No need for sleep here, the state change handles the handoff
                except IOError as e:
                    # Handle input overflow/underflow specifically
                    print(f"\nAudio stream read error: {e}")
                    # Might need to re-open stream, but simple sleep might suffice
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    print(f"\nUnexpected error processing audio frame in wake word: {e}")
                    stop_event.set() # Signal main loop to stop on serious error
                    break # Exit loop

            else:
                # If not listening for wake word, pause briefly to yield CPU
                time.sleep(0.01) # Use a smaller sleep for faster reaction

    except pvporcupine.PorcupineActivationError as e:
        print(f"\nPorcupine activation error: {e}")
        update_status(State.ERROR)
    except pvporcupine.PorcupineActivationLimitError:
        print("\nPorcupine activation limit reached (monthly free tier)")
        update_status(State.ERROR)
    except pvporcupine.PorcupineActivationRefusedError:
        print("\nPorcupine activation refused (invalid key?)")
        update_status(State.ERROR)
    except pvporcupine.PorcupineError as e:
        print(f"\nPorcupine error: {e}")
        update_status(State.ERROR)
    except Exception as e:
        print(f"\nError in wake word listener: {e}")
        update_status(State.ERROR)
    finally:
        print("\nWake word listener shutting down...")
        if audio_stream is not None:
            audio_stream.close()
        if pa is not None:
            pa.terminate()
        if porcupine is not None:
            porcupine.delete()
        print("Wake word listener cleanup complete.")


def query_processor():
    """Listens for the user's query after wake word and processes it."""
    global r, mic, llm_answer # Need to modify global answer

    # Initialize SR components here, within the thread,
    # to potentially avoid conflicts with Porcupine's audio stream
    try:
        r = sr.Recognizer()
        r.energy_threshold = 4000 # Adjust as needed, higher means less sensitive
        r.dynamic_energy_threshold = False # Set threshold manually
        r.pause_threshold = 0.8 # Seconds of non-speaking audio before a phrase is considered complete
        r.operation_timeout = 5 # Seconds before a listen() times out waiting for phrase start
        mic = sr.Microphone()
        print("Query processor ready.")
    except Exception as e:
        print(f"Failed to initialize SpeechRecognition: {e}")
        update_status(State.ERROR)
        return # Cannot proceed

    while not stop_event.is_set():
        # Check if we are in the state to listen for a query
        if get_current_status() == State.TRIGGERED:
            update_status(State.LISTENING_Q) # Now actively listening for query
            print("Listening for your query...")
            with mic as source:
                audio = None
                try:
                    # It's sometimes good practice to adjust for ambient noise after wake word
                    # if your environment changes or the mic level might fluctuate.
                    # However, it can add delay. Try adjusting once during init or skip.
                    # r.adjust_for_ambient_noise(source, duration=1) # Adjust duration if needed
                    # print("Adjusted for ambient noise.")

                    audio = r.listen(source, timeout=5, phrase_time_limit=8) # Listen for query
                    print("Query audio captured.")
                except sr.WaitTimeoutError:
                    print("No query speech detected within timeout.")
                    update_status(State.LISTENING_WW) # Go back to listening for wake word
                    continue
                except Exception as e:
                    print(f"Error during query listening: {e}")
                    update_status(State.LISTENING_WW)
                    continue

            # If audio was captured, try to recognize and process
            if audio:
                try:
                    print("Recognizing query...")
                    # Using Google Web Speech API for recognition
                    query_text = r.recognize_google(audio)
                    print(f"Query recognized: {query_text}")

                    update_status(State.PROCESSING)
                    # Process the query with the latest frame
                    frame_to_process = None
                    with frame_lock:
                        # Get the latest frame available without blocking
                        if latest_frame is not None:
                            frame_to_process = latest_frame.copy()

                    if frame_to_process is not None:
                        answer = get_gemini_response(frame_to_process, query_text)
                        with state_lock:
                             llm_answer = answer # Store the new answer
                        update_status(State.SPEAKING) # Trigger TTS
                    else:
                        print("Error: No frame available for processing query.")
                        with state_lock:
                            llm_answer = "Sorry, I couldn't see anything to process right now."
                        update_status(State.SPEAKING) # Speak the error

                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand query audio.")
                    with state_lock:
                         llm_answer = "Sorry, I didn't understand that."
                    update_status(State.SPEAKING) # Speak the error, then return to WW
                except sr.RequestError as e:
                    print(f"Could not request query results from Google; {e}")
                    with state_lock:
                         llm_answer = f"Sorry, there was an issue with the speech recognition service: {e}"
                    update_status(State.SPEAKING) # Speak the error, then return to WW
                except Exception as e:
                    print(f"An unexpected error occurred during query recognition/processing: {e}")
                    with state_lock:
                        llm_answer = f"An internal error occurred: {e}"
                    update_status(State.SPEAKING) # Speak the error, then return to WW
        else:
            # If not triggered, wait briefly
            time.sleep(0.01) # Use a smaller sleep for faster reaction

    print("Query processor shutting down...")


def speak_answer():
    """Speaks the generated answer using TTS."""
    global llm_answer
    while not stop_event.is_set():
        answer_to_speak = None
        status_now = get_current_status()

        if status_now == State.SPEAKING:
            # Get the answer and clear it immediately after speaking starts
            with state_lock:
                answer_to_speak = llm_answer
                llm_answer = "" # Clear the answer once speaking is initiated

            if answer_to_speak:
                print(f"Speaking: {answer_to_speak}")
                try:
                    # Ensure engine properties are set if needed
                    # tts_engine.setProperty('rate', 180) # Adjust speech rate
                    # tts_engine.setProperty('volume', 1.0) # Adjust volume (0.0 to 1.0)
                    tts_engine.say(answer_to_speak)
                    tts_engine.runAndWait() # Blocks this thread until speaking is done
                except Exception as e:
                    print(f"TTS Error: {e}")
                    # Decide how to handle TTS errors - maybe just print and move on
                finally:
                    # IMPORTANT: Transition back to listening for wake word after speaking
                    print("Finished speaking, returning to listen for wake word.")
                    update_status(State.LISTENING_WW)
            else:
                 # If answer is somehow empty when SPEAKING, just go back
                 print("Speaking state detected but no answer text found, returning to listen.")
                 update_status(State.LISTENING_WW)
        else:
            time.sleep(0.01) # Wait if not in speaking state

    print("Speaker thread shutting down...")


# --- Gemini Function (Updated with clearer error handling) ---
def get_gemini_response(image_frame, prompt_text):
    """Sends frame and prompt to Gemini, returns text response."""
    if image_frame is None:
        print("Gemini request received with None frame.")
        return "Error: No image frame available for AI processing."

    try:
        # Convert OpenCV frame (BGR) to PIL Image (RGB)
        rgb_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Prepare the multimodal prompt parts
        prompt_parts = [prompt_text, pil_image]

        print("Sending request to Gemini model...")
        # Call the generate_content method
        response = model.generate_content(prompt_parts, stream=False) # stream=False for simpler handling here

        # Process the response
        if response and response.parts:
             # Concatenate text from all text parts
             response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
             if response_text:
                 return response_text
             else:
                 # Handle case where parts exist but none are text (e.g., only function calls)
                 print("Warning: Gemini response contained parts, but no text parts.")
                 try: print("Feedback:", response.prompt_feedback) # Check feedback for blockage reasons
                 except Exception: pass
                 return "The AI generated a response, but it wasn't text I can speak."

        elif response and response.prompt_feedback:
             # Handle cases where the prompt or response was blocked
             feedback = response.prompt_feedback
             block_reason = feedback.block_reason if hasattr(feedback, 'block_reason') else 'unknown reason'
             safety_ratings = feedback.safety_ratings if hasattr(feedback, 'safety_ratings') else []
             print(f"Warning: Gemini response blocked. Reason: {block_reason}. Safety Ratings: {safety_ratings}")
             return f"My response was blocked due to safety reasons ({block_reason})."
        else:
            # Handle empty or unexpected response structure
            print("Warning: Gemini response was empty or unexpected.")
            return "My response was empty."

    except Exception as e:
        # Catch any exceptions during the API call
        print(f"Error calling Gemini API: {e}")
        return f"Sorry, I encountered an error communicating with the AI model: {e}"


# --- Main Video Loop ---
def main_loop():
    global latest_frame

    print(f"Attempting to open video stream at {STREAM_URL}...")
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print(f"Error: Could not open video stream at {STREAM_URL}")
        stop_event.set() # Signal other threads to stop
        return

    print("Connected to video stream.")
    print(f"System active. Say '{KEYWORD_PATHS[0].split('/')[-1].replace('.ppn','')}' to ask a question.")
    print("Press 'Q' to quit.")

    frame_read_error_count = 0
    MAX_FRAME_READ_ERRORS = 10 # Threshold before attempting reconnect

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            frame_read_error_count += 1
            print(f"Warning: Failed to retrieve frame ({frame_read_error_count}/{MAX_FRAME_READ_ERRORS}).")
            if frame_read_error_count >= MAX_FRAME_READ_ERRORS:
                print("Max frame read errors reached. Attempting to reconnect...")
                cap.release()
                time.sleep(2) # Wait before attempting reconnect
                cap = cv2.VideoCapture(STREAM_URL)
                if cap.isOpened():
                    print("Reconnected to video stream.")
                    frame_read_error_count = 0 # Reset counter on successful reconnect
                else:
                    print("Error: Reconnect failed. Exiting.")
                    stop_event.set() # Signal all threads to stop
                    break # Exit the main loop
            else:
                 # Brief pause on transient error
                 time.sleep(0.1)
            continue # Skip processing if frame wasn't read

        frame_read_error_count = 0 # Reset counter on successful read

        # Store the latest frame safely
        with frame_lock:
            latest_frame = frame.copy()

        # --- Process User Input ---
        # cv2.waitKey needs to be called regularly to process window events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit key pressed. Exiting...")
            stop_event.set() # Signal all threads to stop
            break

        # --- Display Frame and Status ---
        display_frame = None
        with frame_lock:
            # Get a copy of the latest frame for display
            if latest_frame is not None:
                display_frame = latest_frame.copy()

        if display_frame is not None:
            status_now = get_current_status()
            answer_now = ""
            with state_lock: # Get current answer safely for display
                 # Only display the answer text if it's set AND we are in a state where
                 # we expect an answer to be displayed (e.g., processing or finished speaking)
                 # Let's simplify and just show the last set answer or initial message
                 answer_now = llm_answer if llm_answer else "..." # Display initial message or '...'


            # Display Status
            cv2.putText(display_frame, f"Status: {status_now}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display Answer/Message (using text wrapping)
            # ...(Text wrapping logic from previous version)...
            y0, dy = 60, 25
            text_to_display = answer_now
            lines_wrapped = []
            current_line = ""
            words = text_to_display.split(' ')

            for word in words:
                test_line = f"{current_line} {word}".strip()
                # Estimate text width. Use a larger font scale for width calculation if display font is smaller
                (text_width, text_height), baseline = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1) # Using 0.6 for size calc
                # Ensure text doesn't go beyond image width minus padding (e.g., 20 pixels)
                if text_width > display_frame.shape[1] - 20:
                    lines_wrapped.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line: # Add the last line
                 lines_wrapped.append(current_line)


            for i, wrapped_line in enumerate(lines_wrapped):
                 y = y0 + i * dy
                 # Add a background rectangle for readability? Optional.
                 # (text_width_disp, text_height_disp), baseline_disp = cv2.getTextSize(wrapped_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                 # cv2.rectangle(display_frame, (10, y - text_height_disp - baseline_disp), (10 + text_width_disp, y), (0, 0, 0), -1) # Black background
                 cv2.putText(display_frame, wrapped_line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) # Yellow text


            cv2.imshow('Robot View - Say "Porcupine", Press Q to Quit', display_frame)
        else:
             # If display_frame is None (shouldn't happen if capture is good, but safety)
            print("Warning: Display frame is None.")
            time.sleep(0.1)


    # --- Cleanup ---
    print("\nCleaning up main loop...")
    if cap is not None:
       cap.release()
    cv2.destroyAllWindows()
    print("Main loop finished.")

# --- Start Threads ---
if __name__ == "_main_":
    # Ensure essential configurations are set before starting threads
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY is not set. Exiting.")
        exit()
    if not PICOVOICE_ACCESS_KEY:
        print("PICOVOICE_ACCESS_KEY is not set. Exiting.")
        exit()
    if 'model' not in locals() or model is None:
         print("Gemini model failed to initialize. Exiting.")
         exit()


    # Create threads
    # Make threads daemons so they exit when the main thread exits unexpectedly
    # If you need careful shutdown, manage non-daemon threads and joins
    ww_thread = threading.Thread(target=wake_word_listener, daemon=True)
    qp_thread = threading.Thread(target=query_processor, daemon=True)
    speak_thread = threading.Thread(target=speak_answer, daemon=True)

    # Start threads
    print("Starting threads...")
    ww_thread.start()
    qp_thread.start()
    speak_thread.start()
    print("Threads started.")

    # Start the main loop in the main thread
    main_loop()

    # Signal threads to stop (redundant with daemon=True but good practice if changing)
    stop_event.set()

    # Optional: Join threads for clean exit if they were not daemon
    # print("Waiting for threads to finish...")
    # ww_thread.join()
    # qp_thread.join()
    # speak_thread.join()
    # print("Threads finished.")

    print("Application exiting.")