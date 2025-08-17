
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangGraph and Langchain imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

from config import config

load_dotenv()

# --- Configuration and Initialization ---

# Set Google API Key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize embeddings model
def get_embeddings_model():
    """Caches the HuggingFaceEmbeddings model."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

embeddings = get_embeddings_model()

# Initialize Chroma vector store
def get_vector_store(embed_func):
    """Caches the Chroma vector store."""
    try:
        return Chroma(persist_directory=config.CHROMA_PERSIST_DIRECTORY, embedding_function=embed_func)
    except Exception as e:
        st.error(f"Error loading ChromaDB. Make sure '{config.CHROMA_PERSIST_DIRECTORY}' exists and is populated. Error: {e}")
        st.stop() # Stop the app if DB cannot be loaded

vectordb = get_vector_store(embeddings)

# Initialize the Gemini Pro model
def get_chat_model():
    """Caches the Gemini Pro model."""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.1,  # Slight randomness for more natural responses
            max_tokens=1024,  # Increased token limit
            convert_system_message_to_human=True,  # Added to handle system messages properly
            verbose=True  # Added for debugging
        )
    except Exception as e:
        st.error(f"Error initializing Gemini Pro model: {e}")
        raise

model = get_chat_model()

# Define the LangGraph node function
def call_model(state: MessagesState):
    """
    This function defines the 'model' node in the LangGraph workflow.
    It takes the current state (conversation messages) and invokes the LLM.
    """
    system_prompt = (
        "You are an expert assistant for question-answering tasks specialized in technical and academic content. "
        "Analyze the following pieces of retrieved context carefully to answer the question. "
        "If the context contains relevant information, use it to provide a detailed and accurate answer. "
        "If you don't find the specific information in the context, say so clearly. "
        "Keep your answers clear and focused, using the context provided. "
        "If you're unsure about any part of the answer, acknowledge the uncertainty."
    )
    # Prepend the system message to the current conversation history
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    try:
        response = model.invoke(messages)
        # Ensure we have a valid response
        if not response or not response.content:
            return {"messages": [AIMessage(content="I apologize, but I couldn't generate a proper response. Please try rephrasing your question.")]}
        return {"messages": response}
    except Exception as e:
        st.error(f"Error in model response: {str(e)}")
        return {"messages": [AIMessage(content="I encountered an error while processing your question. Please try again.")]}

# Build and compile the LangGraph workflow
def get_langgraph_app():
    """Caches and compiles the LangGraph workflow."""
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    # Add simple in-memory checkpointer for conversation history
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

app = get_langgraph_app()

# --- Streamlit UI Setup ---

st.set_page_config(page_title="Academic QA", layout="centered")
st.title("💬 Academic QA")

# Initialize session state for messages and thread ID
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores all messages for display
if "thread_id" not in st.session_state:
    # Use a fixed thread ID for this session or generate a unique one
    st.session_state.thread_id = "streamlit_chat_session"

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Logic ---

if prompt := st.chat_input("Ask a any question..."):
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1. Retrieve context for the current user question
                docs = vectordb.similarity_search_with_score(prompt, k=5)  # Increased k for more context
                if not docs:
                    st.warning("No relevant context found in the documents. The response might be less accurate.")
                    current_context = ""
                else:
                    _docs = pd.DataFrame(
                        [(prompt, doc[0].page_content, doc[0].metadata.get('source'), doc[0].metadata.get('page'), doc[1]) for doc in docs],
                        columns=['query', 'paragraph', 'document', 'page_number', 'relevant_score']
                    )
                    # Sort by relevance score (lower is better) and take top results
                    _docs = _docs.sort_values('relevant_score').head(3)
                    
                    # Format the context with clear separation
                    context_parts = []
                    for _, row in _docs.iterrows():
                        context_parts.append(f"[From {os.path.basename(row['document'])}, Page {row['page_number']}]:\n{row['paragraph']}")
                    current_context = "\n\n---\n\n".join(context_parts)

                # 2. Construct the HumanMessage for the current turn, including context
                # This message will be appended to the existing conversation history by LangGraph
                current_turn_message = HumanMessage(content=f"Context: {current_context}\n\nQuestion: {prompt}")

                # 3. Invoke the LangGraph app with the new message and the consistent thread_id
                # LangGraph's checkpointer handles loading previous state and appending this message.
                result = app.invoke(
                    {"messages": [current_turn_message]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )

                # Get the AI's response (the last AIMessage in the result)
                ai_response = result['messages'][-1].content

                # Extract source document and page numbers
                source_document = _docs['document'][0] if not _docs.empty and 'document' in _docs.columns else "N/A"
                # Get unique top 3 page numbers and convert to string
                top_three_page_numbers = _docs['page_number'].drop_duplicates().head(3).astype(str).tolist()
                page_numbers_str = ', '.join(top_three_page_numbers) if top_three_page_numbers else "N/A"

                # Format the final response with Markdown for better display
                final_response = f"{ai_response}\n\n**Source Document**: {source_document}\n**Reference Page Numbers**: {page_numbers_str}"
                
                st.markdown(final_response)

                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                error_message = f"An error occurred while processing your request: {str(e)}"
                st.error(error_message)
                
                if "API key" in str(e).lower():
                    st.error("There seems to be an issue with the API key. Please check your .env file.")
                elif "model" in str(e).lower():
                    st.error("There seems to be an issue with the model configuration. Please verify the model name and settings.")
                elif "context" in str(e).lower():
                    st.error("There was an issue retrieving context from the documents. Please verify your document ingestion.")
                
                st.session_state.messages.append({"role": "assistant", "content": "I encountered an error. Please try again."})

# To run this Streamlit app:
# 1. Save the code above as a Python file (e.g., `chatbot_app.py`).
# 2. Make sure you have your 'docs/chroma/' directory correctly set up with your vector store.
# 3. Run from your terminal: `streamlit run chatbot_app.py`
