import os

class Config:
    """
    Configuration class to manage file paths and settings for the application.
    """

    # --- YouTube Configuration ---
    # URL of the YouTube video to be processed.
    YOUTUBE_VIDEO_URL: str = "https://www.youtube.com/watch?v=oPn_adlC1Q0"
    
    # Directory where downloaded YouTube audio files will be saved.
    YOUTUBE_AUDIO_SAVE_DIRECTORY: str = "docs/youtube/"

    # --- PDF Configuration ---
    # Directory where source PDF documents are located.
    PDF_SOURCE_DIRECTORY: str = "data"
    # Directory where ChromaDB embeddings will be persisted.
    CHROMA_PERSIST_DIRECTORY: str = "docs/chroma"


    # ---Embedding Model Configuration ---
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Efficient embedding model with good performance
    CHUNK_SIZE = 512  # Optimized chunk size for the BGE model
    CHUNK_OVERLAP = 50  # Reduced overlap to prevent redundancy while maintaining context

    def __init__(self):
        # Ensure the PDF source directory exists upon initialization.
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        print(f"Configuration loaded. PDF documents should be placed in '{self.PDF_SOURCE_DIRECTORY}'.")

# Create a global instance of the Config class for easy access.
# You can import `config` from this module in other files.
config = Config()
