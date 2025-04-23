# 💬 PDF-Aware Fin-Assistant Chatbot

**Zen-Buddy Zenshastra Agentic AI for Market Commentary**

**Fin-Assistant** is an AI-powered financial assistant designed to provide intelligent insights and support for financial data analysis. Leveraging advanced natural language processing and machine learning techniques, Fin-Assistant aims to assist users in navigating complex financial information with ease.

---

## 🚀 Features

- **Natural Language Querying**: Interact with financial data using conversational language.
- **PDF Document Parsing**: Automatically extract and process data from financial PDFs.
- **FAISS Indexing**: Efficient similarity search using Facebook AI Similarity Search (FAISS).
- **User-Friendly Interface**: Intuitive UI for seamless user experience.
- **Data Persistence**: Store and manage user data securely.

---

## 🧰 Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript (details in `ui/` directory)
- **Database**: SQLite (`user_data.db`)
- **Machine Learning**: FAISS for similarity search
- **Document Processing**: Custom PDF parsers

---

## 📂 Project Structure

```
Fin-Assistant/
├── .github/workflows/               # GitHub Actions workflows
├── __pycache__/                     # Compiled Python files
├── db/                              # Database-related files
├── beat_article_1/                  # FAISS index files
├── ui/                              # Frontend UI components
├── BEAT-article_thebeatmar2025.pdf  # Sample financial article
├── app.py                           # Main Flask application
├── chat_engine.py                   # Chatbot engine logic
├── pdf_to_faiss.py                  # PDF to FAISS index converter
├── requirements.txt                 # Python dependencies
├── script.py                        # Utility scripts
└── user_data.db                     # SQLite database file
```

---

## ⚙️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/zenshastra/ZenAlpha-SageVision/
   cd Fin-Assistant
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   python pdf_to_faiss.py
   python chat_engine.py
   streamlit run script.py
   ```

   The application will start on `http://localhost:5000/`.

---

## 📝 Usage

1. **Access the Web Interface**:
   Navigate to `http://localhost:5000/` in your web browser.

2. **Interact with the Chatbot**:
   Use the chat interface to ask financial questions. The chatbot will process your queries and provide relevant information extracted from the indexed documents.

3. **Upload Documents**:
   Use the upload feature to add new financial PDFs. The system will parse and index these documents for future queries.

---

## 📄 Sample Documents

- **BEAT-article_thebeatmar2025.pdf**: An article from "The Beat" magazine, March 2025 edition.

This document is used to demonstrate the application's capabilities in parsing and extracting meaningful information.

---

## 🧪 Testing

To run tests (if available), use the following command:

```bash
python -m unittest discover tests
```

Ensure that you have a `tests/` directory with appropriate test cases.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**: Click on the "Fork" button at the top right of the repository page.
2. **Create a New Branch**:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Make Your Changes**: Implement your feature or fix.
4. **Commit Your Changes**:
   ```bash
   git commit -m "Add YourFeatureName"
   ```
5. **Push to Your Fork**:
   ```bash
   git push origin feature/YourFeatureName
   ```
6. **Create a Pull Request**: Navigate to the original repository and click on "New Pull Request".

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For any inquiries or feedback, please contact [SpinnovaOps](mailto:spinnovaops@example.com).

---

*Note: This README is based on the available information in the `Branch-2` branch of the repository. For more detailed documentation and updates, please refer to the repository directly.*




