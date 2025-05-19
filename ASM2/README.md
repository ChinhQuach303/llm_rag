# RAG System for Crawled Data

This project implements a complete Retrieval-Augmented Generation (RAG) system using crawled data from various sources including CMU, Pittsburgh information, sports, food, events, culture, and AI.

## Project Structure

- `data_cleaning.py`: Cleans and processes raw crawled data
- `text_chunker.py`: Chunks cleaned text into manageable segments
- `annotator.py`: Adds annotations to text chunks to enhance retrieval
- `embedding_engine.py`: Generates embeddings and builds a vector store
- `rag_app.py`: Implements the RAG application with LLM integration
- `run_pipeline.py`: Runner script to execute the complete pipeline

## Setup

1. Clone the repository
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo  # or another model
```

## Running the Application

To run the complete pipeline from data cleaning to application launch:

```
python run_pipeline.py
```

To skip certain stages:

```
python run_pipeline.py --skip-data-cleaning --skip-chunking
```

To only run the application (assuming you have already processed the data):

```
python run_pipeline.py --run-app-only
```

## Pipeline Stages

1. **Data Cleaning**: Parses and cleans raw crawled data files
2. **Text Chunking**: Splits cleaned text into manageable chunks with appropriate overlap
3. **Annotation**: Adds metadata like named entities, keywords, and document type
4. **Embedding**: Generates vector embeddings and builds a FAISS index
5. **Application**: Launches a web interface with Gradio for querying the system

## Using the Application

After launching the application:

1. Type your question in the input field
2. Optionally filter by category (CMU, Pittsburgh, food, etc.)
3. Set the number of documents to retrieve
4. Choose whether to display retrieved context
5. Click Submit to get your answer

## Extending the System

- Add new data sources by following the same file format
- Modify prompt templates in `rag_app.py` to improve responses
- Replace OpenAI with another LLM provider by changing the `LLMService` class

## License

This project is licensed under the MIT License - see the LICENSE file for details. 