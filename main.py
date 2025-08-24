from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

document_path = os.getenv('DOCUMENT_PATH')
persist_directory = os.getenv('PERSIST_DIRECTORY')

def get_document_based_collection_name(document_path):
    """Create a unique collection name based on document filename"""
    doc_name = Path(document_path).stem  # filename without extension
    return f"rag_{doc_name}"


def is_vector_store_populated(persist_directory, collection_name):
    """Check if vector store exists"""
    if vector_store._collection.count() == 0:
        return False
    return True


collection_name = get_document_based_collection_name(document_path)
print(f"Using collection: {collection_name}")

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_directory
)

if is_vector_store_populated(persist_directory, collection_name):
    print(f"Vector store for '{document_path}' already exists. Using existing collection...")

else:
    print(f"Creating new vector store for '{document_path}'...")

    loader = TextLoader(document_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vector_store.add_documents(documents=chunks)
    vector_store._client.persist()

    print(f"Created collection with {len(chunks)} chunks")


prompt = ChatPromptTemplate.from_messages([
    ('system', '''
    You are a helpful assistant that uses the context given in triple backticks to provide answers to user queries. 
    - If the context provided is not enough to answer the question, don't just make up an answer, insted say that the context is not enough.
    - Context = ```{context}```
    '''),
    ('human', '{input}')
])

llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, reasoning_format='hidden')

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(combine_docs_chain=combine_docs_chain, retriever=vector_store.as_retriever())

query = 'How many runs did Brevis score against australia in the 2nd T20i?'
response = qa_chain.invoke({'input': query})

print(response['answer'])