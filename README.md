# RAG-based Text Document Retriever

This project demonstrates a **Retrieval-Augmented Generation (RAG)**
pipeline using **LangChain**, **ChromaDB**, **HuggingFace embeddings**,
and **Groq LLM**, specifically designed to work with plain text
documents.\
It creates persistent vector stores for each document and reuses them
across runs, avoiding unnecessary recomputation.

------------------------------------------------------------------------

## Features

-   Loads text documents using `TextLoader`.
-   Generates a **unique collection name** based on the filename
    (ensuring multiple documents can be handled independently).
-   Splits text into chunks with `RecursiveCharacterTextSplitter`.
-   Creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
-   Persists embeddings in **ChromaDB** for reusability.
-   Checks if vector store already exists before re-embedding.
-   Uses **Groq LLM** (`deepseek-r1-distill-llama-70b`) for
    context-based answers.
-   Returns "context not enough" if the answer cannot be found.

------------------------------------------------------------------------

## Project Structure

    project/
    │── chroma/             # Persistent ChromaDB storage
    │── documents/          # Store your text files here
    │── main.py             # RAG pipeline implementation
    │── .env                # Environment variables
    │── requirements.txt    # Dependencies
    │── README_Text.md      # Documentation

------------------------------------------------------------------------

## Installation

``` bash
# Clone this repository
git clone https://github.com/MohitAryal/Document-Rag
cd Document-Rag

# Create and activate virtual environment
python -m venv venv
source venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Environment Setup

Create a `.env` file in the project root with:

``` env
DOCUMENT_PATH=your_file_path.txt
PERSIST_DIRECTORY=your_intended_directory
GROQ_API_KEY=your_groq_api_key
```

------------------------------------------------------------------------

## Example Output

    Query : How many runs did Brevis score against Australia in the 2nd T20i?
    Answer: Dewald Brevis scored 125 runs against Australia in the 2nd T20I.
------------------------------------------------------------------------

## Requirements

-   Python 3.10+
-   LangChain
-   ChromaDB
-   HuggingFace Transformers
-   LangChain Groq
-   dotenv

Install all dependencies via:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------