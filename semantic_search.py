from langchain_community.document_loaders import PyPDFLoader
from shared.splitter_instance import get_splitter
from shared.vectorstore_instance import get_vectorstore_client

file_path = "sample_data/nke-10k-2023.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

splitter = get_splitter()

all_splits = splitter.split_documents(docs)

vector_store = get_vectorstore_client()

# uncomment this only to actually embed something, its a bit expensive operation
# ids = vector_store.add_documents(documents=all_splits)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":1},
)

retrieved_docs = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

print(retrieved_docs)