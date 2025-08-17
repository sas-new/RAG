# RAG - QA

This repository contains a Retrieval Augmented Generation (RAG) application that allows you to chat with your documents and YouTube videos. It uses Streamlit for the user interface, LangChain for the RAG pipeline, and Google's Gemini model for generation.

## How it Works

The application consists of two main parts:

1.  **Data Ingestion**: A script (`ingest.py`) processes PDF documents and YouTube videos, creates embeddings from their content, and stores them in a ChromaDB vector store.
2.  **Chat Application**: A Streamlit application (`app.py`) provides a chat interface where you can ask questions. The application retrieves relevant information from the vector store and uses it to generate answers with the Gemini model.

## Prerequisites

Before you begin, ensure you have the following installed on your Windows 10 system:

*   **Python 3.11 or higher**: You can download it from the [official Python website](https://www.python.org/downloads/).
*   **Git**: You can download it from the [Git website](https://git-scm.com/downloads).
*   **uv (Python package manager)**: This project uses `uv` for package management.

### Installing uv

Open PowerShell and run the following command to install `uv`:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

To verify the installation, run:

```powershell
uv --version
```

## Setup Instructions

### 1. Clone the Repository

Open your terminal or PowerShell and clone the repository to your local machine:

```bash
git clone https://github.com/akash-balakrishnan-22/RAG.git
cd RAG
```

### 2. Create and Activate the Virtual Environment

Create a virtual environment using `uv`:

```bash
uv venv
```

Activate the environment. On Windows, you need to run:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages using `uv`:

```bash
uv sync
```

### 4. Set Up Your API Key

The application requires a Google API key to use the Gemini model.

1.  Create a file named `.env` in the root of the project directory.
2.  Add your Google API key to the `.env` file as follows:

    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

    You can obtain a Google API key from the [Google AI Studio](https://aistudio.google.com/app/apikey).

## Data Configuration

### 1. Add Your PDF Documents

Place any PDF files you want to chat with into the `data` directory. The ingestion script will automatically process all PDF files in this folder.

### 2. Configure the YouTube Video

To change the YouTube video to be processed, open the `config.py` file and modify the `YOUTUBE_VIDEO_URL` variable:

```python
class Config:
    # ...
    YOUTUBE_VIDEO_URL: str = "https://www.youtube.com/watch?v=your_video_id"
    # ...
```

## Running the Application

### 1. Run the Data Ingestion

Before running the application for the first time, you need to run the data ingestion script. This will process your documents and create the vector store.

```bash
python ingest.py
```

This script will create a `docs/chroma` directory containing the vector store.

### 2. Start the Chat Application

Once the ingestion is complete, you can start the Streamlit application:

```bash
streamlit run app.py
```

This will open a new tab in your web browser with the chat interface.

## Troubleshooting

*   **API Key Errors**: If you get an error related to the API key, make sure you have created the `.env` file correctly and that the key is valid.
*   **ChromaDB Errors**: If you see errors related to ChromaDB, ensure that the `ingest.py` script has been run successfully and that the `docs/chroma` directory exists.
*   **Missing Dependencies**: If you encounter errors about missing modules, make sure you have activated the virtual environment and run `uv sync` to install all the required packages.
