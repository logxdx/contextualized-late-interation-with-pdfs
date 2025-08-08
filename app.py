import os
import io
import base64
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import streamlit as st

# Import your backend RAG system
from rag import RAG  # Save your backend code as rag_system.py

# ==================
# Streamlit Settings
# ==================
st.set_page_config(page_title="RAG PDF Search", page_icon="📄", layout="wide")

# Load environment variables
load_dotenv(".env")

# ==================
# Sidebar: Settings
# ==================
st.sidebar.header("⚙️ Configuration")

rag_model = st.sidebar.selectbox(
    "RAG Model",
    [
        "vidore/colpali-v1.3",
        "vidore/colSmol-500M",
        "vidore/colSmol-256M",
        "vidore/colqwen2-v1.0",
        "vidore/colqwen2.5-v0.2",
    ],
    index=0,
)
batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=2)
top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=10, value=2)
prefetch_limit = st.sidebar.slider("Prefetch Limit", min_value=1, max_value=50, value=5)

model_name = st.sidebar.text_input("LLM Model Name", value="openai-fast")
api_key = st.sidebar.text_input(
    "API Key", value=os.getenv("POLLINATIONS_API_KEY", ""), type="password"
)
base_url = st.sidebar.text_input(
    "Base URL", value=os.getenv("POLLINATIONS_BASE_URL", "http://localhost:11434/v1")
)

index_path = st.sidebar.text_input("Index Path", value="./index")
collection_name = st.sidebar.text_input("Collection Name", value="rag")

# ==================
# App Title
# ==================
st.title("📄 RAG PDF Search with ColPali / ColQwen")
st.markdown("Upload PDFs, index them, and ask questions using your RAG pipeline.")

# ==================
# Session State
# ==================
if "rag" not in st.session_state:
    st.session_state.rag = RAG(
        rag_model=rag_model, index_path=index_path, collection_name=collection_name
    )

rag = st.session_state.rag

# ==================
# Tabs
# ==================
tab1, tab2 = st.tabs(["📥 Index PDFs", "❓ Ask Questions"])

# ------------------
# Index PDFs Tab
# ------------------
with tab1:
    st.header("📥 Index PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDF Files", type=["pdf"], accept_multiple_files=True
    )

    index_folder_path = st.text_input("Or specify a folder path to index:", "")

    if st.button("Index PDFs"):
        if uploaded_files:
            for pdf_file in uploaded_files:
                pdf_path = Path(f"./temp_{pdf_file.name}")
                with open(pdf_path, "wb") as f:
                    f.write(pdf_file.read())
                st.write(f"Indexing {pdf_file.name}...")
                rag.index_file(pdf_path=pdf_path, batch_size=batch_size)
                os.remove(pdf_path)
            st.success("Uploaded PDFs indexed successfully!")
        elif index_folder_path:
            rag.index_folder(path=Path(index_folder_path), batch_size=batch_size)
            st.success("Folder indexed successfully!")
        else:
            st.warning("Please upload at least one PDF or provide a folder path.")

# ------------------
# Ask Questions Tab
# ------------------
with tab2:
    st.header("❓ Ask a Question")
    images = None

    query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and generating answer..."):
                # Get both the answer and the retrieved image metadata
                answer, images = rag.answer(
                    query=query,
                    top_k=top_k,
                    prefetch_limit=prefetch_limit,
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    with_images=True,
                )

                # Display retrieved context pages
                if images:
                    with st.expander("📑 Retrieved Context Pages"):
                        cols = st.columns(min(len(images), 3))
                        for i, payload in enumerate(images):
                            image_data = base64.b64decode(payload["image"])
                            image = io.BytesIO(image_data)
                            with cols[i % len(cols)]:
                                st.image(
                                    image,
                                    caption=f"{payload['file']} (Page {payload['page_no']})",
                                )

                # Display the generated answer
                st.subheader("💡 Answer")
                st.write(answer)

if st.button("Rerun"):
    st.rerun()


# ==================
# Clean Exit
# ==================
def cleanup():
    rag.close()

