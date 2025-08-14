from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from shared.vectorstore_instance import get_vectorstore_client
from dotenv import load_dotenv
from os import getenv
load_dotenv()
file_path = getenv("PDF_FILE_PATH")
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

split_docs = text_splitter.split_documents(docs)

vectorstore_client = get_vectorstore_client()

document_ids = vectorstore_client.add_documents(documents=split_docs)
print(document_ids)