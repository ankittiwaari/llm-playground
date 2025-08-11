# Learn LangChain: Semantic Search & Chatbot Example

This project demonstrates how to use [LangChain](https://python.langchain.com/) for semantic search on PDF documents and includes a chatbot example that answers only in a specified language, using LangGraph and Postgres for checkpointing.

## Features

- Loads PDF documents using `PyPDFLoader` from `langchain_community`
- Splits documents into smaller chunks for better semantic search
- Stores document chunks in a vector store for efficient retrieval
- Retrieves relevant document chunks based on user queries
- **Chatbot**: Responds to user queries in a specified language using LangChain, LangGraph, and Postgres for stateful conversation

## Requirements

- Python 3.8+
- [langchain_community](https://pypi.org/project/langchain-community/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [langgraph](https://pypi.org/project/langgraph/)
- [psycopg2](https://pypi.org/project/psycopg2/) (for Postgres checkpointing)
- Additional dependencies for your splitter and vector store (see `shared/` modules)

Install dependencies:

```bash
pip install langchain_community PyPDF2 langgraph psycopg2
# Add any other dependencies required by your splitter/vectorstore
```

## Usage

### Semantic Search

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

### Chatbot

The `chatbot.py` script demonstrates a chatbot that only answers in a specified language and uses Postgres for checkpointing conversation state.

1. Ensure Postgres is running and accessible with the credentials in `chatbot.py`.
2. Ensure you have the required model setup in `shared/model_instance.py` (must provide `get_model()`).
3. Run the chatbot script:

```bash
python chatbot.py
```

This will:
- Start a conversation with a sample query
- Respond only in the specified language (see the `language` variable in the script)
- Use Postgres to checkpoint conversation state

## Project Structure

```
learn_langchain/
├── sample_data/
│   └── nke-10k-2023.pdf
├── shared/
│   ├── splitter_instance.py
│   ├── vectorstore_instance.py
│   └── model_instance.py
├── semantic_search.py
├── chatbot.py
├── README.md
└── .gitignore
```

## Notes

- Embedding documents can be expensive; the code for adding documents to the vector store is commented out by default.
- You can modify the queries in `semantic_search.py` and the language in `chatbot.py` as needed.
- The chatbot requires a running Postgres instance and the correct connection string.

## License

This project is for educational purposes.
