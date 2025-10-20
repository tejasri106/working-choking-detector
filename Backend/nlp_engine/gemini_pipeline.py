from dotenv import load_dotenv
import os
import time
import pyttsx3

# --- Gemini setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

try:
    from google import genai
    client = genai.Client(api_key=api_key)
except ImportError:
    print("❌ 'google-genai' library not installed.")
    client = None

if not api_key:
    print("❌ Gemini API key not found!")
else:
    print("✅ Gemini API loaded.")

# --------------------------------------------
# Choking emergency pipeline
# --------------------------------------------

emergency_context = None
conversation_memory = []

def text_to_speech(text, filename=None):
    """Convert text to speech and play audio (cross-platform)."""
    if filename is None:
        filename = "response.wav"

    print(f"🖥️ SCREEN DISPLAY: {text}")

    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 0.9)
    engine.save_to_file(text, filename)
    engine.runAndWait()

    try:
        if os.name == "nt":  # Windows
            import winsound
            winsound.PlaySound(filename, winsound.SND_FILENAME)
        else:  # macOS/Linux
            os.system(f"afplay {filename}")
    except Exception as e:
        print(f"⚠️ Could not play audio: {e}")

def start_emergency_guidance(emergency_type):
    """Called when choking is detected."""
    global emergency_context, conversation_memory

    emergency_context = {
        "type": emergency_type,
        "current_step": 1,
        "steps_given": []
    }

    conversation_memory = [f"Emergency detected: {emergency_type}"]

    prompt = f"Start first aid instructions for {emergency_type}. Give only the first step clearly."

    first_step = get_gemini_response(prompt)
    conversation_memory.append(f"Gemini: {first_step}")

    text_to_speech(first_step, "emergency_start.wav")

    return {
        "intent": "emergency_started",
        "emergency_type": emergency_type,
        "response": first_step
    }

def process_user_response(user_text):
    """Handles user’s speech or text interaction."""
    global emergency_context, conversation_memory

    if not emergency_context:
        return {"intent": "no_emergency", "response": "No emergency active."}

    conversation_memory.append(f"User: {user_text}")
    recent_context = "\n".join(conversation_memory[-5:])

    prompt = f"""
    Emergency: {emergency_context['type']}
    Conversation: {recent_context}
    User just said: "{user_text}"
    Provide the next choking guidance in 2 sentences.
    """

    gemini_response = get_gemini_response(prompt)
    conversation_memory.append(f"Gemini: {gemini_response}")
    text_to_speech(gemini_response, "response.wav")

    return {
        "intent": "emergency_guidance",
        "emergency_type": emergency_context["type"],
        "response": gemini_response
    }

def get_gemini_response(prompt):
    if not client:
        return "Please continue with emergency first aid procedures."

    try:
        response = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"(Gemini error: {e})"
