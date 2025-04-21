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
import networkx as nx
from typing import List, Dict, Any, Tuple

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

# ---------------- GraphRAG Implementation ---------------- #

class GraphRAG:
    """A Graph-based RAG approach for more intelligent document queries"""
    
    def __init__(self, document_data):
        self.document_data = document_data
        self.knowledge_graph = self._build_knowledge_graph()
        
    def _build_knowledge_graph(self):
        """Build a knowledge graph from the document data"""
        G = nx.DiGraph()
        
        # Extract entities and relationships from document
        entities = self._extract_entities()
        relationships = self._extract_relationships(entities)
        
        # Add nodes and edges to graph
        for entity in entities:
            G.add_node(entity["id"], 
                      type=entity["type"], 
                      name=entity["name"],
                      attributes=entity.get("attributes", {}))
            
        for rel in relationships:
            G.add_edge(rel["source"], rel["target"], 
                      type=rel["type"],
                      weight=rel.get("weight", 1.0),
                      context=rel.get("context", ""))
            
        return G
    
    def _extract_entities(self):
        """Extract entities from document data"""
        # This would be implemented with NER in production
        # For now, use a manual approach based on document structure
        entities = []
        
        # Extract financial concepts
        financial_terms = [
            "investment", "portfolio", "stocks", "bonds", "ETF",
            "mutual fund", "retirement", "IRA", "401k", "tax",
            "inflation", "interest rate", "dividend", "capital gain",
            "market", "recession", "bull market", "bear market"
        ]
        
        # Create entity objects
        for i, term in enumerate(financial_terms):
            if term.lower() in str(self.document_data).lower():
                entities.append({
                    "id": f"concept_{i}",
                    "type": "financial_concept",
                    "name": term,
                    "attributes": {
                        "frequency": str(self.document_data).lower().count(term.lower())
                    }
                })
        
        return entities
    
    def _extract_relationships(self, entities):
        """Extract relationships between entities"""
        relationships = []
        
        # For each pair of entities, check if they might be related
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    # Check for co-occurrence (simplified)
                    if entity1["attributes"]["frequency"] > 0 and entity2["attributes"]["frequency"] > 0:
                        # Add a relationship if both terms appear in the document
                        relationships.append({
                            "source": entity1["id"],
                            "target": entity2["id"],
                            "type": "co_occurs_with",
                            "weight": min(entity1["attributes"]["frequency"], entity2["attributes"]["frequency"]) / 10.0,
                            "context": "Document co-occurrence"
                        })
        
        return relationships
    
    def query(self, query_text: str) -> List[Dict[str, Any]]:
        """Query the knowledge graph for relevant information"""
        # Extract query entities
        query_terms = set(re.findall(r'\b\w+\b', query_text.lower()))
        
        # Find related nodes in graph
        related_nodes = []
        for node_id in self.knowledge_graph.nodes:
            node_data = self.knowledge_graph.nodes[node_id]
            node_name = node_data["name"].lower()
            
            # Check if any query term is in node name
            if any(term in node_name or node_name in term for term in query_terms):
                # Add this node and its neighbors
                related_nodes.append({
                    "id": node_id,
                    "name": node_data["name"],
                    "type": node_data["type"],
                    "relevance": "direct match"
                })
                
                # Add neighbors (one hop away)
                for neighbor_id in self.knowledge_graph.neighbors(node_id):
                    neighbor_data = self.knowledge_graph.nodes[neighbor_id]
                    edge_data = self.knowledge_graph.get_edge_data(node_id, neighbor_id)
                    
                    related_nodes.append({
                        "id": neighbor_id,
                        "name": neighbor_data["name"],
                        "type": neighbor_data["type"],
                        "relevance": "connected concept",
                        "connection": edge_data["type"],
                        "weight": edge_data["weight"]
                    })
        
        return related_nodes
    
    def enhance_query_response(self, query: str, initial_response: str) -> str:
        """Enhance a query response with knowledge graph insights"""
        # Get related concepts from the knowledge graph
        related_concepts = self.query(query)
        
        if not related_concepts:
            return initial_response
            
        # Format the related concepts for inclusion in prompt
        concepts_text = ""
        for concept in related_concepts[:5]:  # Limit to top 5 concepts
            if concept["relevance"] == "direct match":
                concepts_text += f"- {concept['name']} (directly mentioned)\n"
            else:
                concepts_text += f"- {concept['name']} (related concept, {concept['connection']})\n"
        
        # Create enhanced prompt
        enhanced_prompt = f"""
        Original query: {query}
        
        Initial response: {initial_response}
        
        Related financial concepts from the document:
        {concepts_text}
        
        Please enhance the response based on these related financial concepts.
        Make sure the answer is comprehensive, accurate, and directly addresses the query.
        Only use information from the provided document content.
        Organize the response in clear bullet points.
        """
        
        # Generate enhanced response
        try:
            enhanced_chat = genai.GenerativeModel('gemini-1.5-flash-001-tuning').start_chat(history=[])
            enhanced_response = enhanced_chat.send_message(enhanced_prompt).text
            return enhanced_response
        except Exception as e:
            # Fall back to original response if enhancement fails
            return initial_response

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
    st.session_state.current_page = "chat"  # Track current page
    st.session_state.summary_option = None  # Track selected summary option
    st.session_state.summary_generated = False  # Track if summary has been generated
    st.session_state.doc_summary_option = None  # Track selected document summary option
    st.session_state.doc_summary_generated = False  # Track if document summary has been generated
    
    # Initialize Gemini model with better caching strategy
    st.session_state.gemini_chat = genai.GenerativeModel('gemini-1.5-flash-001-tuning').start_chat(
        history=[],
    )
    
    # Initialize GraphRAG with document content
    if global_structured_content:
        st.session_state.graph_rag = GraphRAG(global_structured_content)
    else:
        # Use raw document text if structured content is not available
        try:
            with open("BEAT-article_thebeatmar2025.txt", "r") as f:
                raw_content = f.read()
                st.session_state.graph_rag = GraphRAG(raw_content)
        except FileNotFoundError:
            st.session_state.graph_rag = None
    
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

# ---------------- Improved Query Processing ---------------- #

def process_query_async(input_text, target_lang_code, target_lang_name):
    """Process user query with GraphRAG enhanced reasoning"""
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

    # Step 4: Create improved prompt that focuses on document content
    prompt = f"""
    CONTEXT FROM DOCUMENT:
    {finbert_response}

    USER QUERY:
    {input_english}
    
    INSTRUCTIONS:
    1. Answer the query ONLY using information from the document.
    2. If the information is not in the document, clearly state that.
    3. Organize your response in clear, easy-to-read bullet points.
    4. Start with the most relevant information.
    5. Include specific facts, figures, and details from the document.
    6. Do not make up information or add external knowledge.
    7. Keep your response concise but comprehensive.
    
    Your response must be in {target_lang_name} language only.
    """

    # Step 5: Generate initial response with Gemini-Pro
    initial_response = st.session_state.gemini_chat.send_message(prompt).text
    
    # Step 6: Enhance response with GraphRAG if available
    if st.session_state.graph_rag:
        enhanced_response = st.session_state.graph_rag.enhance_query_response(
            input_english, 
            initial_response
        )
    else:
        enhanced_response = initial_response
    
    # Step 7: Verify language and translate if necessary
    detected_output_lang = translator.detect(enhanced_response).lang
    
    # Only translate if the output isn't already in the requested language
    if detected_output_lang != target_lang_code:
        final_response = translator.translate(enhanced_response, src=detected_output_lang, dest=target_lang_code).text
    else:
        final_response = enhanced_response
        
    result = {
        "text": final_response,
        "category": query_category
    }
    
    return result

# ---------------- Generate Document Summary Function ---------------- #

def generate_document_summary(summary_type):
    """Generate a summary of the entire document based on specified length"""
    
    # Get document content - prioritize structured content if available
    if st.session_state.structured_content:
        doc_content = json.dumps(st.session_state.structured_content, indent=2)
    else:
        try:
            with open("BEAT-article_thebeatmar2025.txt", "r") as f:
                doc_content = f.read()
        except FileNotFoundError:
            return "Document content not available for summarization."
    
    # Create prompt based on summary type
    if summary_type == "Short Summary":
        prompt = f"""
        Please create a very brief summary (2-3 paragraphs) of the following financial document:
        
        {doc_content}
        
        Focus only on the key financial insights, main topics, and core messages.
        """
    elif summary_type == "Medium Summary":
        prompt = f"""
        Please create a medium-length summary (4-6 paragraphs) of the following financial document:
        
        {doc_content}
        
        Include the main financial topics discussed, key insights, important data points, and overall themes.
        Organize by main sections of the document where appropriate.
        """
    else:  # Long Summary
        prompt = f"""
        Please create a comprehensive summary of the following financial document:
        
        {doc_content}
        
        Include:
        1. An executive summary at the beginning
        2. All major financial topics discussed with detailed explanations
        3. Specific data points, statistics, and figures mentioned
        4. Key recommendations or conclusions
        5. Structure the summary by document sections
        
        Use bullet points and subheadings for clarity. Aim for a comprehensive but readable summary.
        """
    
    # Generate summary with Gemini
    try:
        # Create a separate chat instance for the summary
        summary_chat = genai.GenerativeModel('gemini-1.5-flash-001-tuning').start_chat(history=[])
        summary_response = summary_chat.send_message(prompt).text
        
        # Translate if not in English
        if st.session_state.language != "English":
            summary_response = translator.translate(
                summary_response, 
                src='en', 
                dest=st.session_state.language_code
            ).text
        
        return summary_response
    except Exception as e:
        return f"Failed to generate document summary: {str(e)}"

# ---------------- Generate Chat Summary Function ---------------- #

def generate_summary(summary_type):
    """Generate a summary based on the chat history and specified length"""
    if not st.session_state.chat_history:
        return "No chat history to summarize."
    
    # Extract the relevant parts of chat history for summarization
    conversation = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            conversation.append(f"User: {msg['content']}")
        else:
            conversation.append(f"Assistant: {msg['content']}")
    
    conversation_text = "\n".join(conversation)
    
    # Create prompt based on summary type
    if summary_type == "Short Summary":
        prompt = f"""
        Please create a very brief summary (2-3 sentences) of the following financial conversation:
        
        {conversation_text}
        
        Focus only on the key financial insights and decisions.
        """
    elif summary_type == "Medium Summary":
        prompt = f"""
        Please create a medium-length summary (4-6 sentences) of the following financial conversation:
        
        {conversation_text}
        
        Include the main financial topics discussed, key insights, and any recommendations made.
        """
    else:  # Long Summary
        prompt = f"""
        Please create a comprehensive summary of the following financial conversation:
        
        {conversation_text}
        
        Include all financial topics discussed, detailed insights, specific recommendations, 
        and categorize the information by topic. Structure with bullet points where appropriate.
        """
    
    # Generate summary with Gemini
    try:
        # Create a separate chat instance for the summary to avoid mixing with user conversation
        summary_chat = genai.GenerativeModel('gemini-1.5-flash-001-tuning').start_chat(history=[])
        summary_response = summary_chat.send_message(prompt).text
        
        # Translate if not in English
        if st.session_state.language != "English":
            summary_response = translator.translate(
                summary_response, 
                src='en', 
                dest=st.session_state.language_code
            ).text
        
        return summary_response
    except Exception as e:
        return f"Failed to generate summary: {str(e)}"

# ---------------- Navigation Callback Functions ---------------- #

def nav_to_chat():
    st.session_state.current_page = "chat"
    
def nav_to_summary():
    st.session_state.current_page = "summary"

def nav_to_doc_summary():
    st.session_state.current_page = "doc_summary"
    
def handle_summary_option_change():
    # This gets called when dropdown selection changes
    st.session_state.summary_generated = False
    
def handle_doc_summary_option_change():
    # This gets called when document summary dropdown selection changes
    st.session_state.doc_summary_generated = False
    
def generate_summary_button():
    if st.session_state.summary_option:
        st.session_state.summary_generated = True
    else:
        st.warning("Please select a summary type first")

def generate_doc_summary_button():
    if st.session_state.doc_summary_option:
        st.session_state.doc_summary_generated = True
    else:
        st.warning("Please select a document summary type first")

# ---------------- CSS Styling ---------------- #

st.markdown("""
<style>
    /* Main layout styling */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Custom navigation styling */
    .nav-link {
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        text-decoration: none;
        display: block;
        text-align: left;
        transition: background-color 0.3s;
    }
    
    .nav-link:hover {
        background-color: rgba(49, 51, 63, 0.1);
    }
    
    .nav-link.active {
        background-color: rgba(49, 51, 63, 0.2);
        font-weight: bold;
    }
    
    /* Message styling */
    .user-message, .bot-message {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        max-width: 85%;
    }
    
    .user-message {
        background-color: #e6f7ff;
        margin-left: auto;
        margin-right: 0;
    }
    
    .bot-message {
        background-color: #f0f2f6;
        margin-left: 0;
        margin-right: auto;
    }
    
    /* Summary section styling */
    .summary-section {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f9f9f9;
        margin-top: 1rem;
    }
    
    /* Document summary styling */
    .doc-summary-section {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f7ff;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .generate-btn {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.25rem;
        cursor: pointer;
        font-weight: bold;
    }
    
    .generate-btn:hover {
        background-color: #45a049;
    }
    
    /* Dropdown styling */
    .summary-dropdown {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Onboarding ---------------- #

if not st.session_state.user_info:
    onboarding_form()
else:
    # Set up the layout with sidebar navigation
    with st.sidebar:
        st.title("Financial Assistant")
        
        # User profile information
        st.subheader("Profile")
        st.write(f"👤 **Name:** {st.session_state.user_info.get('name', 'User')}")
        st.write(f"🌎 **Location:** {st.session_state.user_info.get('country', 'Unknown')}, {st.session_state.user_info.get('state', 'Unknown')}")
        
        st.markdown("---")
        
        # Navigation buttons
        st.subheader("Navigation")
        
        # Chat nav button
        chat_active_class = "active" if st.session_state.current_page == "chat" else ""
        st.markdown(f"""
        <div class="nav-link {chat_active_class}" onclick="document.getElementById('chat-btn').click()">
            💬 Chat
        </div>
        """, unsafe_allow_html=True)
        chat_btn = st.button("Chat", key="chat-btn", on_click=nav_to_chat, help="Go to chat interface")
        
        # Chat Summary nav button
        summary_active_class = "active" if st.session_state.current_page == "summary" else ""
        st.markdown(f"""
        <div class="nav-link {summary_active_class}" onclick="document.getElementById('summary-btn').click()">
            📋 Chat Summary
        </div>
        """, unsafe_allow_html=True)
        summary_btn = st.button("Chat Summary", key="summary-btn", on_click=nav_to_summary, help="Generate conversation summaries")
        
        # Document Summary nav button
        doc_summary_active_class = "active" if st.session_state.current_page == "doc_summary" else ""
        st.markdown(f"""
        <div class="nav-link {doc_summary_active_class}" onclick="document.getElementById('doc-summary-btn').click()">
            📚 Document Summary
        </div>
        """, unsafe_allow_html=True)
        doc_summary_btn = st.button("Document Summary", key="doc-summary-btn", on_click=nav_to_doc_summary, help="Generate document summaries")
        
        # Summary options section (only shown on summary page)
        if st.session_state.current_page == "summary":
            st.markdown("---")
            st.subheader("Summary Options")
            
            # Summary type dropdown
            st.session_state.summary_option = st.selectbox(
                "Summary Length:",
                ["Short Summary", "Medium Summary", "Long Summary"],
                key="summary_dropdown",
                on_change=handle_summary_option_change
            )
            
            # Generate button
            generate_btn = st.button(
                "🔄 Generate Summary", 
                key="generate_summary_btn",
                on_click=generate_summary_button,
                help="Create a summary of your conversation"
            )
            
        # Document Summary options section (only shown on doc summary page)
        if st.session_state.current_page == "doc_summary":
            st.markdown("---")
            st.subheader("Document Summary Options")
            
            # Document summary type dropdown
            st.session_state.doc_summary_option = st.selectbox(
                "Summary Length:",
                ["Short Summary", "Medium Summary", "Long Summary"],
                key="doc_summary_dropdown",
                on_change=handle_doc_summary_option_change
            )
            
            # Generate document summary button
            generate_doc_btn = st.button(
                "🔄 Generate Doc Summary", 
                key="generate_doc_summary_btn",
                on_click=generate_doc_summary_button,
                help="Create a summary of the entire document"
            )
        
        st.markdown("---")
        
        # Language Selection
        country = st.session_state.user_info.get("country")
        state = st.session_state.user_info.get("state")

        language_options = ["English", "Hindi"]
        regional_lang = REGIONAL_LANGUAGES.get((country, state))
        if regional_lang:
            if "," in regional_lang:
                language_options.extend(regional_lang.split(","))
            else:
                language_options.append(regional_lang)

        st.subheader("Language Settings")
        selected_language = st.selectbox(
            "🌐 Preferred language:", 
            language_options,
            index=language_options.index(st.session_state.language) if st.session_state.language in language_options else 0
        )
        
        # Update language if changed
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.session_state.language_code = LANGUAGE_CODES.get(selected_language, 'en')

    # Main content area
    if st.session_state.current_page == "chat":
        # Chat interface
        st.header(f"💬 Financial Assistant Chat")
        st.caption(f"Currently conversing in {st.session_state.language}")

        # Display chat history with improved UI
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bot-message">
                        <strong>Assistant:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add audio player for English responses
                    if st.session_state.language == "English" and message.get("audio"):
                        st.markdown(message["audio"], unsafe_allow_html=True)

        # Input section
        st.markdown("---")
        input_col1, input_col2 = st.columns([4, 1])
        
        # Show voice input option only for English
        voice_placeholder = st.empty()
        
        # Voice input button - placed alongside text input
        with input_col2:
            if st.session_state.language == "English":
                if st.button("🎤 Voice", key="voice_button"):
                    handle_voice_input()
        
        # Display voice transcription if available
        if st.session_state.voice_input_text:
            voice_placeholder.info(f"You said: {st.session_state.voice_input_text}")
        
        # Text input field - set default value from voice input if available
        with input_col1:
            if st.session_state.language == "English" and st.session_state.voice_input_text:
                user_input = st.text_input("Ask your financial question:", 
                                    value=st.session_state.voice_input_text,
                                    key=f"input_{len(st.session_state.chat_history)}")
                # Clear voice input after using it
                st.session_state.voice_input_text = ""
            else:
                user_input = st.text_input("Ask your financial question:",
                                    key=f"input_{len(st.session_state.chat_history)}")

        # Process user input
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Force refresh to show user message immediately
            st.rerun()

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
                
                # Prepare bot message
                bot_message = {
                    "role": "assistant", 
                    "content": response_text,
                    "category": query_category
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
                
                # Force refresh to show response
                st.rerun()

        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        with col2:
            if st.session_state.chat_history:
                if st.button("📋 View Chat Summary", on_click=nav_to_summary):
                    pass
        with col3:
            if st.button("📚 View Document Summary", on_click=nav_to_doc_summary):
                pass
    
    elif st.session_state.current_page == "summary":
        # Chat Summary page
        st.header("📋 Conversation Summary")
        
        if not st.session_state.chat_history:
            st.warning("No conversation data available to summarize. Please chat first.")
        else:
            st.info(f"This will generate a {st.session_state.summary_option or 'selected'} summary of your conversation.")
            
            # Show summary if generated
            if st.session_state.summary_generated and st.session_state.summary_option:
                with st.spinner(f"Generating {st.session_state.summary_option}..."):
                    summary_text = generate_summary(st.session_state.summary_option)
                    
                st.markdown("### Summary Result")
                st.markdown(f"""
                <div class="summary-section">
                    {summary_text}
                </div>
                """, unsafe_allow_html=True)
                
                # Export options
                st.markdown("### Export Options")
                export_col1, export_col2 = st.columns(2)
                with export_col1:
                    if st.button("📄 Copy to Clipboard"):
                        # Using JavaScript to copy to clipboard
                        st.markdown(f"""
                        <script>
                            navigator.clipboard.writeText(`{summary_text}`);
                        </script>
                        """, unsafe_allow_html=True)
                        st.success("Summary copied to clipboard!")
                
                with export_col2:
                    if st.button("📱 Share Summary"):
                        # This would normally integrate with sharing functionality
                        st.info("Sharing functionality would be implemented here")
            
            # Show conversation statistics
            st.markdown("### Conversation Statistics")
            total_messages = len(st.session_state.chat_history)
            user_messages = sum(1 for msg in st.session_state.chat_history if msg["role"] == "user")
            bot_messages = sum(1 for msg in st.session_state.chat_history if msg["role"] == "assistant")
            
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            stats_col1.metric("Total Messages", total_messages)
            stats_col2.metric("Your Messages", user_messages)
            stats_col3.metric("Assistant Responses", bot_messages)
            
            # Add a button to return to chat
            st.markdown("---")
            if st.button("💬 Return to Chat", on_click=nav_to_chat):
                pass
    
    elif st.session_state.current_page == "doc_summary":
        # Document Summary page
        st.header("📚 Document Summary")
        
        st.info(f"This will generate a {st.session_state.doc_summary_option or 'selected'} summary of the entire document.")
        
        # Show document summary if generated
        if st.session_state.doc_summary_generated and st.session_state.doc_summary_option:
            with st.spinner(f"Generating {st.session_state.doc_summary_option} of the document..."):
                doc_summary_text = generate_document_summary(st.session_state.doc_summary_option)
                
            st.markdown("### Document Summary Result")
            st.markdown(f"""
            <div class="doc-summary-section">
                {doc_summary_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Export options
            st.markdown("### Export Options")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📄 Copy to Clipboard", key="copy_doc_summary"):
                    # Using JavaScript to copy to clipboard
                    st.markdown(f"""
                    <script>
                        navigator.clipboard.writeText(`{doc_summary_text}`);
                    </script>
                    """, unsafe_allow_html=True)
                    st.success("Document summary copied to clipboard!")
            
            with export_col2:
                if st.button("📱 Share Summary", key="share_doc_summary"):
                    # This would normally integrate with sharing functionality
                    st.info("Sharing functionality would be implemented here")
                    
            with export_col3:
                if st.button("💾 Save as PDF", key="save_doc_summary"):
                    # This would normally integrate with PDF generation
                    st.info("PDF generation would be implemented here")
            
            # Document metadata - if available
            if st.session_state.structured_content:
                try:
                    # Extract metadata if available
                    metadata = st.session_state.structured_content.get("metadata", {})
                    if metadata:
                        st.markdown("### Document Metadata")
                        meta_col1, meta_col2 = st.columns(2)
                        
                        with meta_col1:
                            st.write(f"**Title:** {metadata.get('title', 'N/A')}")
                            st.write(f"**Date:** {metadata.get('date', 'N/A')}")
                            st.write(f"**Author:** {metadata.get('author', 'N/A')}")
                            
                        with meta_col2:
                            st.write(f"**Type:** {metadata.get('type', 'Financial Document')}")
                            st.write(f"**Pages:** {metadata.get('pages', 'N/A')}")
                            st.write(f"**Version:** {metadata.get('version', '1.0')}")
                except:
                    pass
        
        # Add a button to return to chat
        st.markdown("---")
        if st.button("💬 Return to Chat", key="return_from_doc", on_click=nav_to_chat):
            pass

# Cleanup temp files
for file in os.listdir(st.session_state.temp_dir):
    if file.endswith('.mp3'):
        try:
            os.remove(os.path.join(st.session_state.temp_dir, file))
        except:
            pass
