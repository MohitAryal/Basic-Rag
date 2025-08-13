# Day 1: Basic RAG with LangChain

## Goal
Build a simple Retrieval-Augmented Generation (RAG) pipeline that:
1. Loads a local text file
2. Splits the content into smaller chunks
3. Generates embeddings for each chunk
4. Stores embeddings in a vector database (ChromaDB)
5. Retrieves relevant chunks for a given query
6. Uses an LLM to produce an answer using the retrieved context

-----------------------------------------------------------------------------------------------------------------

## Tech Stack
- **Python 3.10+**
- [LangChain](https://python.langchain.com/) — framework for LLM apps
- [ChromaDB](https://www.trychroma.com/) — local vector database
- [Groq API](https://groq.com/) — LLM
- [Hugging Face](https://huggingface.co/) — models, datasets, and inference APIs
- `python-dotenv` — for managing API keys

-------------------------------------------------------------------------------------------------------------------

## Files
```
1. Basic Rag/
│── sample_document.txt # Sample document for testing
│── main.py # Main script
│── requirements.txt # Python dependencies
```
----------------------------------------------------------------------------------------------------------------------


---

##  How to Run

### Clone the repository and navigate to the folder
```bash
git clone https://github.com/MohitAryal/Basic-Rag.git
```

### Install Dependencies
```pip install -r requirements.txt```


### Add your API Key
Create a .env file in this folder:
```bash 
GROQ_API_KEY=your_api_key_here
```

### Run the script
```bash
python main.py
```


-----------------------------------------------------------------------------------------------------------------------------

# Learnings from the project
- Loading local documents with LangChain

- Splitting large text into chunks for efficient retrieval

- Creating embeddings and storing them in ChromaDB

- Retrieving context and generating answers using an LLM
