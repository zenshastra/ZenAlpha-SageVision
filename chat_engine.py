import os
import google.generativeai as genai  # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain.vectorstores import FAISS # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain.docstore.document import Document # type: ignore
#from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
#from langchain.document_loaders import PyMuPDFLoader # type: ignore
import json
import re

# SET YOUR GEMINI API KEY
GEMINI_API_KEY = "AIzaSyBMhMP13_-94m6eZDRum-X020Ds7U-fk2I"
genai.configure(api_key=GEMINI_API_KEY)

# Load FAISS index with metadata support
def load_faiss_index(index_path="beat_article_1"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(index_path, embeddings)

# Load structured content
def load_structured_content(json_path="BEAT-article_thebeatmar2025_structured.json"):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Structured content file {json_path} not found.")
        return None

# Load FinBERT model
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

# Perform sentiment classification
def classify_with_finbert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = torch.argmax(logits).item()
    sentiments = ["positive", "negative", "neutral"]
    return sentiments[predicted_class_id]

# Function to check if query is about visual content
def is_visual_query(query):
    visual_keywords = [
        'image', 'picture', 'photo', 'graph', 'chart', 'plot', 'diagram',
        'table', 'figure', 'visualization', 'pie chart', 'bar chart',
        'line graph', 'histogram', 'heatmap', 'heat map', 'scatter plot',
        'visual', 'illustration', 'infographic', 'show me'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in visual_keywords)

# Function to check if query is about numeric data
def is_data_query(query):
    data_keywords = [
        'data', 'numbers', 'statistics', 'percentage', 'value', 'metric',
        'measurement', 'figure', 'quantity', 'amount', 'total', 'average',
        'mean', 'median', 'mode', 'sum', 'maximum', 'minimum', 'highest',
        'lowest', 'trend', 'increase', 'decrease', 'growth', 'decline',
        'compare', 'comparison', 'ratio', 'proportion', 'how many', 'how much'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in data_keywords)

# Function to determine query category for specialized handling
def categorize_query(query):
    if is_visual_query(query):
        return "visual"
    elif is_data_query(query):
        return "data"
    else:
        return "general"

# Process query with context awareness
def process_query_context(query, docs, query_category, structured_content=None):
    """Process query with awareness of different content types"""
    
    # Prepare context from documents
    text_contexts = []
    visual_contexts = []
    data_contexts = []
    
    for doc in docs:
        content_type = doc.metadata.get('content_type', 'text')
        
        if content_type == 'text':
            text_contexts.append(doc.page_content)
        elif content_type in ['chart', 'image']:
            visual_contexts.append(doc.page_content)
        elif content_type == 'table':
            data_contexts.append(doc.page_content)
    
    # Combine contexts based on query category
    if query_category == "visual":
        primary_contexts = visual_contexts
        secondary_contexts = text_contexts + data_contexts
    elif query_category == "data":
        primary_contexts = data_contexts
        secondary_contexts = text_contexts + visual_contexts
    else:
        primary_contexts = text_contexts
        secondary_contexts = visual_contexts + data_contexts
    
    # Prioritize relevant contexts but include all
    combined_context = "\n\n".join(primary_contexts + secondary_contexts)
    
    # If we have structured content and this is a visual/data query, add specific reference
    if structured_content and query_category in ["visual", "data"]:
        # Find relevant pages with visual or data content
        relevant_pages = []
        for page in structured_content["content"]:
            if query_category == "visual" and (page["charts"] or page["images"]):
                relevant_pages.append(page["page_number"])
            elif query_category == "data" and page["tables"]:
                relevant_pages.append(page["page_number"])
        
        if relevant_pages:
            combined_context += f"\n\nThe document contains {query_category} content on pages: {', '.join(map(str, relevant_pages))}"
    
    return combined_context

# Enhance response using Gemini-Pro
def enhance_with_gemini(user_query, context, sentiment, query_category):
    # Configure the Gemini API using your API key
    genai.configure(api_key="AIzaSyBMhMP13_-94m6eZDRum-X020Ds7U-fk2I")  # Replace with your actual key

    # Correct model name: must match the tuned model ID from your console
    model = genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-001-tuning"
    )

    # Create base prompt elements
    query_type_note = ""
    if query_category == "visual":
        query_type_note = """
        This query appears to be about visual content (images, charts, graphs, etc.).
        If your answer refers to visual elements:
        1. Describe what appears in the visuals in detail
        2. Explain the meaning of any charts or graphs
        3. Mention specific data points shown in the visuals
        4. Clearly cite the page number where this visual appears
        """
    elif query_category == "data":
        query_type_note = """
        This query appears to be about numeric data or tabular information.
        If your answer includes data from tables:
        1. Format data clearly using markdown tables if possible
        2. Highlight key trends or important data points
        3. Provide precise numbers rather than approximations
        4. Clearly cite the page number where this data appears
        """

    # Create the enhanced prompt
    prompt = f"""
    You are a highly intelligent Financial assistant. Only answer questions based on the content extracted from the provided PDF/document. Do not use any external or prior knowledge, and do not make assumptions. If the answer is not clearly present in the document, respond with: "The answer is not available in the provided document." Use appropriate emojis based on context.
    I want the Responses to be precise and accurate.
    
    Context:
    {context}

    Sentiment Analysis:
    {sentiment}
    
    {query_type_note}

    User Query:
    {user_query}

    Generate an insightful, helpful response that directly addresses the query and references specific data from the document. Your response should be:
    1. Well-organized with clear bullet points
    2. Include specific numbers and data points from the document when relevant
    3. Explain any trends or patterns visible in the data
    4. Be conversational and helpful in tone
    5. If referring to visual elements, describe them clearly so user can understand without seeing them
    6. Always cite page numbers when referring to specific content
    """

    # Generate response
    response = model.generate_content(prompt)

    return response.text


# Main chatbot logic
def answer_query(user_query, faiss_index, tokenizer, model):
    # Load structured content if available
    structured_content = load_structured_content()
    
    # Categorize query for specialized handling
    query_category = categorize_query(user_query)
    
    # Get sentiment
    sentiment = classify_with_finbert(user_query, tokenizer, model)
    
    # Get relevant documents (increase k for more comprehensive context)
    docs = faiss_index.similarity_search(user_query, k=5)
    
    # Process query with context awareness
    context = process_query_context(user_query, docs, query_category, structured_content)
    
    # Generate enhanced response
    enhanced_response = enhance_with_gemini(user_query, context, sentiment, query_category)
    
    return enhanced_response
