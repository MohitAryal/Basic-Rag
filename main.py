from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
import re
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('sample_document.txt')
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store = Chroma(collection_name='basic_rag', embedding_function=embeddings, persist_directory='chroma')
vector_store.add_documents(documents=chunks)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant that uses the context given in triple backticks to provide answers to user queries. ```{context}```'),
    ('human', '{input}')
]
)

retriever = VectorStoreRetriever(vectorstore=vector_store)

llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(combine_docs_chain=combine_docs_chain, retriever=retriever)

query = 'How many runs did Brevis score against australia in the 2nd T20i?'
response = qa_chain.invoke({'input': query})

clean_answer = re.sub(r"<think>.*?</think>", "", response['answer'], flags=re.DOTALL).strip()
print(clean_answer)