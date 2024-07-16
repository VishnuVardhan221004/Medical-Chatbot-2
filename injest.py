# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.schema import Document  # Import the Document class
# from datasets import load_dataset  # Hugging Face datasets

# DATA_PATH = "data/"
# DB_FAISS_PATH = "vectorstores/db_faiss"

# def create_vector_db():
#     # Load documents from PDF files
#     loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
#     documents = loader.load()
    
#     # Load dataset from Hugging Face
#     dataset = load_dataset('lavita/ChatDoctor-HealthCareMagic-100k', split='train[:10%]')  # Example dataset, use appropriate dataset for your use case
    
#     # Inspect the dataset to find the correct key
#     print(dataset.column_names)
    
#     # Convert dataset to LangChain documents
#     # Assuming 'text' is the correct key after inspection, change it if necessary
#     hf_documents = [Document(page_content=sample["input"], metadata={}) for sample in dataset]

#     # Combine documents from PDF and Hugging Face dataset
#     combined_documents = documents + hf_documents

#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(combined_documents)

#     # Create embeddings
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

#     # Create FAISS vector store
#     db = FAISS.from_documents(texts, embeddings)
#     db.save_local(DB_FAISS_PATH)

# if __name__ == '__main__':
#     create_vector_db()





from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document  # Import the Document class
from datasets import load_dataset  # Hugging Face datasets

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

def create_vector_db():
    # Load documents from PDF files
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Load dataset from Hugging Face
    dataset = load_dataset('lavita/ChatDoctor-HealthCareMagic-100k', split='train[:10%]')
    
    # Inspect the dataset to find the correct keys
    print(dataset.column_names)
    
    # Convert dataset to LangChain documents using both 'instruction' and 'input'
    hf_documents = [
        Document(page_content=f"Instruction: {sample['instruction']}\nInput: {sample['input']}", metadata={'output': sample['output']}) 
        for sample in dataset
    ]

    # Combine documents from PDF and Hugging Face dataset
    combined_documents = documents + hf_documents

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(combined_documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    # Create FAISS vector store
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
    create_vector_db()
