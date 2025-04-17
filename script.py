import streamlit as st # type: ignore
from db.db import init_db
from ui.onboarding import onboarding_form
from chat_engine import load_faiss_index, load_finbert, answer_query, categorize_query
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googletrans import Translator # type: ignore
import google.generativeai as genai # type: ignore
import speech_recognition as sr # type: ignore
from gtts import gTTS # type: ignore
import os
import tempfile
import uuid
import time
from io import BytesIO
import base64
import threading
import concurrent.futures
import queue
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# Initialize DB
init_db()

# Configure Gemini-Pro API Key
genai.configure(api_key="AIzaSyCjeWAsyXA24ercu7XRISggxH0_Fzf68Kw")

# ---------------- Global Model Loading ---------------- #

# Use global variables for models to ensure they're loaded only once for all users
@st.cache_resource
def load_global_models():
    """Load models once and cache them globally for all sessions"""
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    faiss_index = load_faiss_index()
    
    # Try to load structured content too
    try:
        with open("BEAT-article_thebeatmar2025_structured.json", "r") as f:
            structured_content = json.load(f)
    except FileNotFoundError:
        structured_content = None
        
    return tokenizer, model, faiss_index, structured_content

# ---------------- Session Initialization ---------------- #

if 'initialized' not in st.session_state:
    # Load global models
    global_tokenizer, global_model, global_faiss, global_structured_content = load_global_models()
    
    # Initialize session state
    st.session_state.tokenizer = global_tokenizer
    st.session_state.model = global_model
    st.session_state.faiss_index = global_faiss
    st.session_state.structured_content = global_structured_content
    st.session_state.chat_history = []  # Store chat history
    st.session_state.user_info = {}
    st.session_state.language = None
    st.session_state.language_code = 'en'  # Default to English code
    st.session_state.voice_input_text = ""
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.current_page_view = None  # For PDF page viewer
    
    # Initialize Gemini model with better caching strategy
    st.session_state.gemini_chat = genai.GenerativeModel('gemini-1.5-flash-001-tuning').start_chat(
        history=[],
    )
    
    # Optimize speech recognition
    st.session_state.recognizer = sr.Recognizer()
    st.session_state.recognizer.energy_threshold = 300
    st.session_state.recognizer.dynamic_energy_threshold = True
    st.session_state.recognizer.dynamic_energy_adjustment_damping = 0.15
    st.session_state.recognizer.dynamic_energy_ratio = 1.5
    st.session_state.recognizer.pause_threshold = 0.8
    st.session_state.recognizer.non_speaking_duration = 0.5
    
    # Processing queues for concurrent operations
    st.session_state.processing_queue = queue.Queue()
    
    # Mark as initialized
    st.session_state.initialized = True

# Initialize translator - create once per session
@st.cache_resource
def get_translator():
    return Translator()

translator = get_translator()

# ---------------- Voice Processing Functions ---------------- #

def text_to_speech(text):
    """Convert text to speech using gTTS and return audio data"""
    if st.session_state.language == "English":
        try:
            # Create a BytesIO object
            audio_bytes = BytesIO()
            
            # Generate speech with reduced quality for speed
            tts = gTTS(text=text, lang='en', slow=False)
            tts.write_to_fp(audio_bytes)
            
            # Reset the position to the beginning
            audio_bytes.seek(0)
            
            # Return the audio bytes
            return audio_bytes.read()
        except Exception as e:
            st.error(f"TTS Error: {str(e)}")
            return None
    return None

def speech_to_text():
    """Convert speech to text using microphone"""
    try:
        with sr.Microphone() as source:
            # Quick adjustment for ambient noise
            st.session_state.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            st.info("🎤 Listening... Speak now!")
            
            # Shorter timeout for faster response
            audio = st.session_state.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.info("🔍 Processing your speech...")
            
            try:
                # Use Google's speech recognition for better accuracy
                text = st.session_state.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return "Sorry, I couldn't understand what you said."
            except sr.RequestError:
                return "Sorry, speech service is unavailable right now."
    except Exception as e:
        st.error(f"Error with speech recognition: {str(e)}")
        return None

# Voice input handler
def handle_voice_input():
    text = speech_to_text()
    if text and text != "Sorry, I couldn't understand what you said." and text != "Sorry, speech service is unavailable right now.":
        st.session_state.voice_input_text = text

# ---------------- Visual Content Handling ---------------- #

def create_chart_from_data(chart_data, chart_type):
    """Create a matplotlib chart from extracted data"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    try:
        # Parse data from the description
        if "values:" in chart_data.lower():
            data_str = chart_data.lower().split("values:")[1].strip()
            # Extract numbers from data string
            values = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', data_str)]
            
            if chart_type == "bar chart":
                labels = [f"Item {i+1}" for i in range(len(values))]
                ax.bar(labels, values)
                ax.set_title("Reconstructed Bar Chart from Document")
                
            elif chart_type == "line chart":
                ax.plot(range(1, len(values)+1), values, marker='o')
                ax.set_title("Reconstructed Line Chart from Document")
                
            elif chart_type == "pie chart":
                ax.pie(values, labels=[f"{v}%" for v in values], autopct='%1.1f%%')
                ax.set_title("Reconstructed Pie Chart from Document")
                
            else:
                ax.bar(range(len(values)), values)  # Default to bar chart
                ax.set_title(f"Reconstructed {chart_type} from Document")
                
            return fig
        else:
            return None
    except:
        return None

def create_table_from_data(table_data):
    """Create a pandas DataFrame from extracted table data"""
    try:
        # Split by rows
        rows = table_data.strip().split('\n')
        if len(rows) < 2:
            return None
            
        # Create DataFrame
        df = pd.DataFrame([row.split() for row in rows])
        
        # If first row seems like a header, set it as such
        if len(df) > 1:
            df.columns = df.iloc[0]
            df = df[1:]
            
        return df
    except:
        return None

# ---------------- Regional Language Mapping ---------------- #

REGIONAL_LANGUAGES = {
    ("India", "Karnataka"): "Kannada",
    ("India", "Tamil Nadu"): "Tamil",
    ("India", "Maharashtra"): "Marathi",
    ("India", "West Bengal"): "Bengali",
    ("India", "Gujarat"): "Gujarati",
    ("India", "Kerala"): "Malayalam",
    ("India", "Telangana"): "Telugu",
    ("United States", "North Carolina"): "Spanish",
    # Add more as needed
}

# Language code mapping
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Malayalam": "ml",
    "Telugu": "te",
    "Spanish": "es",
    # Add more as needed
}

# ---------------- Optimized Query Processing ---------------- #

def process_query_async(input_text, target_lang_code, target_lang_name):
    """Process user query with parallel processing where possible"""
    # Step 1: Detect input language and translate to English if needed
    detected_lang = translator.detect(input_text).lang
    if detected_lang != 'en':
        input_english = translator.translate(input_text, src=detected_lang, dest='en').text
    else:
        input_english = input_text

    # Step 2: Categorize the query
    query_category = categorize_query(input_english)

    # Step 3: Process using FinBERT → FAISS (optimized with threading)
    finbert_response = answer_query(
        input_english,
        st.session_state.faiss_index,
        st.session_state.tokenizer,
        st.session_state.model
    )

    # Step 4: Create prompt that ensures output matches user's language preference
    prompt = f"""
    Here is the financial context: {finbert_response}

    Please improve this response and return in bullet points.
    
    IMPORTANT: Your response must be in {target_lang_name} language only.
    
    The response should be conversational and helpful.
    
    If the answer refers to any visual elements (charts, graphs, tables, images), please be very descriptive so the user can understand even without seeing the actual visual.
    """

    # Step 5: Generate response with Gemini-Pro
    gemini_reply = st.session_state.gemini_chat.send_message(prompt).text
    
    # Step 6: Verify language and translate if necessary
    detected_output_lang = translator.detect(gemini_reply).lang
    
    # Only translate if the output isn't already in the requested language
    if detected_output_lang != target_lang_code:
        final_response = translator.translate(gemini_reply, src=detected_output_lang, dest=target_lang_code).text
    else:
        final_response = gemini_reply
        
    # Step 7: Check for references to specific pages and visual content
    page_references = re.findall(r'page\s+(\d+)', finbert_response.lower())
    
    result = {
        "text": final_response,
        "category": query_category,
        "pages_referenced": list(set(int(p) for p in page_references)) if page_references else []
    }
    
    return result

# ---------------- UI Components ---------------- #

def render_pdf_page_viewer():
    """Render a simplified PDF page viewer based on referenced pages"""
    if st.session_state.structured_content and st.session_state.current_page_view:
        page_num = st.session_state.current_page_view
        
        # Find this page in structured content
        page_data = None
        for page in st.session_state.structured_content["content"]:
            if page["page_number"] == page_num:
                page_data = page
                break
                
        if page_data:
            st.subheader(f"📄 Page {page_num} Content")
            
            # Display text content
            if page_data["text"]:
                with st.expander("📝 Text Content", expanded=True):
                    st.write(page_data["text"])
            
            # Display tables if any
            if page_data["tables"]:
                with st.expander(f"📊 Tables ({len(page_data['tables'])})", expanded=True):
                    for i, table in enumerate(page_data["tables"]):
                        st.markdown(f"**Table {i+1}:**")
                        table_df = create_table_from_data(table["content"])
                        if table_df is not None:
                            st.dataframe(table_df)
                        else:
                            st.code(table["content"])
            
            # Display charts if any
            if page_data["charts"]:
                with st.expander(f"📈 Charts ({len(page_data['charts'])})", expanded=True):
                    for i, chart in enumerate(page_data["charts"]):
                        st.markdown(f"**Chart {i+1} ({chart['type']}):**")
                        st.markdown(f"*{chart['description']}*")
                        
                        # Try to visualize the chart
                        fig = create_chart_from_data(chart["data"], chart["type"])
                        if fig:
                            st.pyplot(fig)
                        else:
                            st.write(chart["data"])
            
            # Display images if any
            if page_data["images"]:
                with st.expander(f"🖼️ Images ({len(page_data['images'])})", expanded=True):
                    for i, img in enumerate(page_data["images"]):
                        st.markdown(f"**Image {i+1}:**")
                        st.markdown(f"*{img['description']}*")

def render_sidebar():
    """Render the sidebar with document navigation options"""
    with st.sidebar:
        st.header("📑 Document Navigator")
        
        if st.session_state.structured_content:
            total_pages = st.session_state.structured_content["total_pages"]
            
            st.write(f"Document has {total_pages} pages")
            
            # Add page selector
            selected_page = st.number_input(
                "Select a page to view:", 
                min_value=1, 
                max_value=total_pages,
                value=st.session_state.current_page_view if st.session_state.current_page_view else 1
            )
            
            if selected_page != st.session_state.current_page_view:
                st.session_state.current_page_view = selected_page
                st.rerun()
            
            # Add page navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Previous Page") and st.session_state.current_page_view > 1:
                    st.session_state.current_page_view -= 1
                    st.rerun()
            
            with col2:
                if st.button("➡️ Next Page") and st.session_state.current_page_view < total_pages:
                    st.session_state.current_page_view += 1
                    st.rerun()
                    
            # Document content overview
            st.subheader("Document Content")
            
            # Count tables, charts, and images
            tables_count = sum(len(page["tables"]) for page in st.session_state.structured_content["content"])
            charts_count = sum(len(page["charts"]) for page in st.session_state.structured_content["content"])
            images_count = sum(len(page["images"]) for page in st.session_state.structured_content["content"])
            
            st.markdown(f"""
            - 📝 Text content across {total_pages} pages
            - 📊 {tables_count} tables identified
            - 📈 {charts_count} charts/graphs detected
            - 🖼️ {images_count} images found
            """)

# ---------------- Onboarding ---------------- #

if not st.session_state.user_info:
    onboarding_form()
else:
    st.success("✅ You're onboarded!")

    # Render sidebar
    render_sidebar()

    country = st.session_state.user_info.get("country")
    state = st.session_state.user_info.get("state")

    # Language Selection
    language_options = ["English", "Hindi"]
    regional_lang = REGIONAL_LANGUAGES.get((country, state))
    if regional_lang:
        if "," in regional_lang:
            language_options.extend(regional_lang.split(","))
        else:
            language_options.append(regional_lang)

    selected_language = st.selectbox(
        "🌐 Choose your preferred language:", 
        language_options,
        index=language_options.index(st.session_state.language) if st.session_state.language in language_options else 0
    )
    
    # Update language if changed
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.session_state.language_code = LANGUAGE_CODES.get(selected_language, 'en')

    # Main content area
    main_content_col, page_viewer_col = st.columns([2, 1])
    
    with main_content_col:
        # ---------------- Chat Section with Conversation History ---------------- #
        st.header(f"💬 Chat in {st.session_state.language}")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Bot:** {message['content']}")
                    
                    # Add audio player for English responses
                    if st.session_state.language == "English" and message.get("audio"):
                        st.markdown(message["audio"], unsafe_allow_html=True)
                    
                    # If this message references pages, provide page links
                    if message.get("pages_referenced"):
                        pages_text = ", ".join(map(str, message["pages_referenced"]))
                        st.info(f"💡 This answer references content from page(s): {pages_text}")
                        
                        # Allow viewing a specific referenced page
                        if len(message["pages_referenced"]) > 0:
                            for page in message["pages_referenced"]:
                                if st.button(f"View Page {page}", key=f"view_page_{page}_{len(st.session_state.chat_history)}"):
                                    st.session_state.current_page_view = page
                                    st.rerun()

        # Show voice input option only for English
        voice_placeholder = st.empty()
        
        # Voice input button - placed BEFORE text input field
        if st.session_state.language == "English":
            if st.button("🎤 Voice Input", key="voice_button"):
                handle_voice_input()
        
        # Display voice transcription if available
        if st.session_state.voice_input_text:
            voice_placeholder.info(f"You said: {st.session_state.voice_input_text}")
        
        # Text input field - set default value from voice input if available
        if st.session_state.language == "English" and st.session_state.voice_input_text:
            user_input = st.text_input("Ask your financial question:", 
                                    value=st.session_state.voice_input_text,
                                    key=f"input_{len(st.session_state.chat_history)}")
            # Clear voice input after using it
            st.session_state.voice_input_text = ""
        else:
            user_input = st.text_input("Ask your financial question:",
                                    key=f"input_{len(st.session_state.chat_history)}")

        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Force refresh to show user message immediately
            st.rerun()
            
    # PDF Page Viewer (in the side column)
    with page_viewer_col:
        if st.session_state.current_page_view:
            render_pdf_page_viewer()

# Process the latest message if it hasn't been processed yet
if st.session_state.chat_history and len(st.session_state.chat_history) % 2 == 1:
    with st.spinner("Analyzing and Generating Response..."):
        # Get the last user message
        last_user_message = st.session_state.chat_history[-1]["content"]
        
        # Process the query
        response_data = process_query_async(
            last_user_message, 
            st.session_state.language_code, 
            st.session_state.language
        )
        
        # Extract response text and metadata
        response_text = response_data["text"]
        query_category = response_data["category"]
        pages_referenced = response_data["pages_referenced"]
        
        # Prepare bot message
        bot_message = {
            "role": "assistant", 
            "content": response_text,
            "category": query_category,
            "pages_referenced": pages_referenced
        }
        
        # Add text-to-speech for English responses
        if st.session_state.language == "English":
            # Clean up the response for better speech
            speech_text = response_text.replace("•", "").replace("\n", " ").strip()
            
            # Create audio bytes
            audio_bytes = text_to_speech(speech_text)
            
            if audio_bytes:
                # Encode audio bytes as base64
                b64 = base64.b64encode(audio_bytes).decode()
                
                # Create HTML audio element
                audio_player = f'<audio autoplay controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                bot_message["audio"] = f'<div>🔊 Voice Response: {audio_player}</div>'
        
        # Add bot response to chat history
        st.session_state.chat_history.append(bot_message)
        
        # If response references pages, update current view to first referenced page
        if pages_referenced and not st.session_state.current_page_view:
            st.session_state.current_page_view = pages_referenced[0]
        
        # Force refresh to show response
        st.rerun()

# Clear chat button
if st.session_state.chat_history:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Cleanup temp files
for file in os.listdir(st.session_state.temp_dir):
    if file.endswith('.mp3'):
        try:
            os.remove(os.path.join(st.session_state.temp_dir, file))
        except:
            pass