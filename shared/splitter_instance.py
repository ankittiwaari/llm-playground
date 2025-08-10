from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
