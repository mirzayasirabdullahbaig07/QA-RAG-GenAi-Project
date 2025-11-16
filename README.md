# 🎗️ QA-RAG-GenAi Project

An **AI-powered web application** built with **Streamlit** that fetches content from websites and answers questions based on that content using Google GenAI and ChromaDB.

---

## 🚀 Demo
🔗 [Live App on Streamlit](https://app-rag-genai-project-007.streamlit.app/)

---

## 🚀 Video Demo
[qa-rag-genai-demo.webm](https://github.com/user-attachments/assets/fe6264a6-b46c-4925-bdbd-8b427926062e)

---

## 📌 Features
- Fetches and parses website content using a URL.
- Converts website text into **vector embeddings** for semantic search.
- Answers user questions based on website content using **Google GenAI**.
- Interactive, user-friendly **Streamlit interface**.
- Displays results in clear **cards** for easy readability.
- Includes an **About Me sidebar** with portfolio links.

---

## 🔍 Usage
1. Open the app in your browser.
2. Enter a website URL in the input field.
3. Click **Fetch** to extract website content.
4. Enter your question in the input box.
5. Click **Ask AI** to get the answer.
6. View the result:
   - ✅ Answer displayed in context cards
   - 📝 Optionally explore the scraped content

---

## 📊 Dataset / Memory
- No explicit dataset required.
- Website content is converted into **vector embeddings** and stored in **ChromaDB** for semantic search.
- Embeddings allow the app to answer questions based on meaning rather than just keywords.

---

## ⚙️ Tech Stack
- **Python 3.9+**
- **Streamlit** – Frontend web app
- **Requests** – Web content fetching
- **BeautifulSoup** – Web content parsing
- **ChromaDB** – Vector database for embeddings
- **Google GenAI SDK** – LLM for text embedding and generation
- **NumPy & Pandas** – Data processing
- **pysqlite3-binary** – Streamlit Cloud SQLite compatibility

---

## 📸 Screenshots
### 🏠 Home Page
![Home Page](images/home_page.png)  <!-- Add your screenshot -->
<img width="1863" height="806" alt="image" src="https://github.com/user-attachments/assets/614fd1f9-0dc5-49c2-9bef-babf0e8e7fa4" />


### ✅ Answer Result
![Answer Result](images/answer_result.png)  <!-- Add your screenshot -->
<img width="1854" height="764" alt="image" src="https://github.com/user-attachments/assets/aa398165-5a93-4883-8ea8-c4bbe259e0b1" />

### ✅ Answer Result
<img width="1883" height="814" alt="image" src="https://github.com/user-attachments/assets/b40042d5-3df6-401f-bc18-6417cbf85586" />

---

## 👨‍💻 Author
**Mirza Yasir Abdullah Baig**  
🌐 [Kaggle](https://www.kaggle.com/mirzayasirabdullah07/)) 
💼 [LinkedIn](https://www.linkedin.com/in/mirza-yasir-abdullah-baig/)  
💻 [GitHub](https://github.com/mirzayasirabdullahbaig07)

---

## ❤️ Acknowledgements
- [Google GenAI SDK Documentation](https://developers.generativeai.google/)
- [ChromaDB Documentation](https://www.trychroma.com/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- Open-source community for Python ML & AI tools

---

## ⚠️ Disclaimer
This project is for **educational purposes only** and should NOT be used as a substitute for professional advice or factual information verification.
