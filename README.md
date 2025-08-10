# Learn LangChain: Semantic Search Example

This project demonstrates how to use [LangChain](https://python.langchain.com/) for semantic search on PDF documents, including document splitting and vector-based retrieval.

## Features

- Loads PDF documents using `PyPDFLoader` from `langchain_community`
- Splits documents into smaller chunks for better semantic search
- Stores document chunks in a vector store for efficient retrieval
- Retrieves relevant document chunks based on user queries

## Requirements

- Python 3.8+
- [langchain_community](https://pypi.org/project/langchain-community/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- Additional dependencies for your splitter and vector store (see `shared/` modules)

Install dependencies:

```bash
pip install langchain_community PyPDF2
# Add any other dependencies required by your splitter/vectorstore
```

## Usage

1. Place your PDF file in the `sample_data` directory.
2. Update the `file_path` in `semantic_search.py` if needed.
3. Ensure you have the required modules in the `shared/` directory:
    - `splitter_instance.py` (must provide `get_splitter()`)
    - `vectorstore_instance.py` (must provide `get_vectorstore_client()`)
4. Run the script:

```bash
python semantic_search.py
```

This will:
- Load and split the PDF into chunks
- (Optionally) Embed and add chunks to the vector store
- Retrieve relevant chunks for sample queries and print the results

## Project Structure

```
learn_langchain/
├── sample_data/
│   └── nke-10k-2023.pdf
├── shared/
│   ├── splitter_instance.py
│   └── vectorstore_instance.py
├── semantic_search.py
├── README.md
└── .gitignore
```

## Notes

- Embedding documents can be expensive; the code for adding documents to the vector store is commented out by default.
- You can modify the queries in `semantic_search.py` as needed.

## License

This project is for educational purposes.
