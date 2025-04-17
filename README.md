# 💬 PDF-Aware Financial Chatbot

An intelligent, multilingual chatbot built to answer **financial questions** from a **PDF document**. Supports:
- 📊 Data/table/chart understanding
- 🖼 Visual description generation
- 💡 Financial sentiment detection
- 🌍 Regional language support
- 🔊 Voice input & output

---

## 🚀 Features

- ✅ **Google Gemini-Pro** for AI response generation
- ✅ **FinBERT** sentiment analysis for finance-specific tone detection
- ✅ **Semantic search** over PDFs using **FAISS** & **LangChain**
- ✅ Visual & data-aware response enhancement (e.g. "Show me the pie chart on page 4")
- ✅ **Voice-based interaction**
- ✅ **Regional language support** with translation
- ✅ Reconstructed charts/tables using `matplotlib` and `pandas`

---

## 🛠️ Tech Stack

| Layer | Tools |
|------|-------|
| Frontend | Streamlit |
| AI & NLP | Gemini-Pro, HuggingFace (FinBERT, SentenceTransformers) |
| PDF Processing | PyMuPDF (fitz), PyPDF2 |
| OCR & Vision | Tesseract, OpenCV |
| Vector DB | FAISS |
| Translation & Voice | googletrans, gTTS, SpeechRecognition |

---


