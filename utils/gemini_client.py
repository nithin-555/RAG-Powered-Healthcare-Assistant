import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def configure_gemini(api_key=None):
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API Key is missing. Please set it in .env or pass it directly.")
    
    genai.configure(api_key=api_key)

def generate_answer(prompt, model_name="gemini-1.5-flash"):
    """
    Generates an answer using Google Gemini.
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"
