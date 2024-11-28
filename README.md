# Reinforcement Learning ChatBot

This project implements a Reinforcement Learning ChatBot using various components for document processing, vector storage, and web scraping. The ChatBot utilizes LangChain and Hugging Face models to provide responses based on user queries.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)
- [License](#license)

## Features

- Load and process text documents from a specified directory.
- Create a vector store for efficient retrieval of relevant information.
- Scrape content from web pages and PDFs.
- Utilize a conversational AI model to answer user queries about Reinforcement Learning.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OmarSaad-png/RAG-ChatBot.git
   cd RAG-ChatBot
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the web scraping process to gather content:
   ```bash
   python Scraper.py
   ```
2. Add additional text documents in the `output` directory (Markdown format).
3. Run the document processing script to split documents into chunks:
   ```bash
   python Document_Processing.py
   ```



4. Launch the Streamlit application for the ChatBot:
   ```bash
   streamlit run RL_ChatBot.py
   ```

5. Open your web browser and navigate to `http://localhost:8501` to interact with the ChatBot.

## File Descriptions

- **RL_ChatBot.py**: Main application file that initializes the ChatBot and handles user interactions.
- **Vector_Store1.py**: Contains functions for loading text chunks and creating a vector store using FAISS.
- **Document_Processing.py**: Loads markdown documents, splits them into chunks, and saves them for further processing.
- **Scraper.py**: Scrapes web pages and PDFs, extracting and saving content for the ChatBot to use.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
