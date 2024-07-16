# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.prompts import PromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import CTransformers
# from langchain.chains import RetrievalQA
# import chainlit as cl
# import pickle

# DB_FAISS_PATH = 'vectorstores/db_faiss'

# custom_prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

# def set_custom_prompt():
#     """
#     Prompt template for QA retrieval for each vectorstore
#     """
#     prompt = PromptTemplate(template=custom_prompt_template,
#                             input_variables=['context', 'question'])
#     return prompt

# #Retrieval QA Chain
# def retrieval_qa_chain(llm, prompt, db):
#     qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                        chain_type='stuff',
#                                        retriever=db.as_retriever(search_kwargs={'k': 2}),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={'prompt': prompt}
#                                        )
#     return qa_chain

# #Loading the model
# def load_llm():
#     # Load the locally downloaded model here
#     llm = CTransformers(
#         model = "TheBloke/Llama-2-7B-Chat-GGML",
#         model_type="llama",
#         max_new_tokens = 512,
#         temperature = 0.5
#     )
#     return llm

# #QA Model Function
# def qa_bot():
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                        model_kwargs={'device': 'cpu'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization = True)
#     llm = load_llm()
#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)

#     return qa

# #output function
# def final_result(query):
#     qa_result = qa_bot()
#     response = qa_result({'query': query})
#     return response

# #chainlit code
# @cl.on_chat_start
# async def start():
#     chain = qa_bot()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to Medical ChatBot. What is your query?"
#     await msg.update()

#     cl.user_session.set("chain", chain)

# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain") 
#     cb = cl.AsyncLangchainCallbackHandler(
#         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
#     )
#     cb.answer_reached = True
#     res = await chain.acall(message.content, callbacks=[cb])
#     answer = res["result"]
#     sources = res["source_documents"]

#     if sources:
#         answer += f"\nSources:" + str(sources)
#     else:
#         answer += "\nNo sources found"

#     await cl.Message(content=answer).send()



# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.prompts import PromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import CTransformers
# from langchain.chains import RetrievalQA
# import chainlit as cl
# import pickle

# DB_FAISS_PATH = 'vectorstores/db_faiss'

# custom_prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

# def set_custom_prompt():
#     """
#     Prompt template for QA retrieval for each vectorstore
#     """
#     prompt = PromptTemplate(template=custom_prompt_template,
#                             input_variables=['context', 'question'])
#     return prompt

# #Retrieval QA Chain
# def retrieval_qa_chain(llm, prompt, db):
#     qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                            chain_type='stuff',
#                                            retriever=db.as_retriever(search_kwargs={'k': 2}),
#                                            return_source_documents=True,
#                                            chain_type_kwargs={'prompt': prompt})
#     return qa_chain

# #Loading the model
# def load_llm():
#     # Load the locally downloaded model here
#     llm = CTransformers(
#         model="TheBloke/Llama-2-7B-Chat-GGML",
#         model_type="llama",
#         max_new_tokens=512,
#         temperature=0.5
#     )
#     return llm

# #QA Model Function
# def qa_bot():
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                        model_kwargs={'device': 'cpu'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
#     llm = load_llm()
#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)

#     return qa

# #output function
# def final_result(query):
#     qa_result = qa_bot()
#     response = qa_result({'query': query})
#     return response

# #chainlit code
# @cl.on_chat_start
# async def start():
#     chain = qa_bot()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to Medical ChatBot. What is your query?"
#     await msg.update()

#     cl.user_session.set("chain", chain)

# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")
#     cb = cl.AsyncLangchainCallbackHandler(
#         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
#     )
#     cb.answer_reached = True
#     res = await chain.acall(message.content, callbacks=[cb])
#     answer = res["result"]

#     await cl.Message(content=answer).send()




from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import pickle
from langchain.schema import Document  # Import the Document class
from datasets import load_dataset  # Hugging Face datasets

DATA_PATH = "data/"
DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=1056,
        temperature=0.5
    )
    return llm

#Create Vector Database
def create_vector_db():
    # Load documents from PDF files
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Load dataset from Hugging Face
    dataset = load_dataset('lavita/ChatDoctor-HealthCareMagic-100k', split='train[:10%]')
    
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

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical ChatBot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]

    await cl.Message(content=answer).send()

if __name__ == '__main__':
    create_vector_db()
