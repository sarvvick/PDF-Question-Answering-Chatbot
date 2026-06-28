import os
import tempfile
import fitz
import camelot
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# CONFIG

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VISION_MODEL = "qwen2.5vl"

# EXTRACT TEXT

def extract_text(pdf_path):

    pdf = fitz.open(pdf_path)

    text = ""

    for page in pdf:
        page_text = page.get_text()

        if page_text:
            text += page_text + "\n"

    pdf.close()

    return text

# EXTRACT IMAGES

def extract_images(pdf_path):

    image_paths = []

    pdf = fitz.open(pdf_path)

    image_dir = "temp_images"

    os.makedirs(image_dir, exist_ok=True)

    for page_num in range(len(pdf)):

        page = pdf[page_num]

        images = page.get_images(full=True)

        for img_idx, img in enumerate(images):

            xref = img[0]

            image = pdf.extract_image(xref)

            image_bytes = image["image"]

            image_path = os.path.join(
                image_dir,
                f"page_{page_num}_{img_idx}.png"
            )

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(image_path)

    pdf.close()

    return image_paths

# IMAGE CAPTIONING

def describe_image(image_path, vision_llm):

    try:

        prompt = f"""
        Analyze this image and describe:

        - Charts
        - Graphs
        - Tables
        - Trends
        - Values
        - Important information

        Image path:
        {image_path}
        """

        response = vision_llm.invoke(prompt)

        return response.content

    except Exception as e:

        return f"Image analysis failed: {e}"

# TABLE EXTRACTION

def extract_tables(pdf_path):

    table_texts = []

    try:

        tables = camelot.read_pdf(
            pdf_path,
            pages="all"
        )

        for table in tables:

            table_texts.append(
                table.df.to_string(index=False)
            )

    except Exception:
        pass

    return table_texts

# BUILD VECTOR STORE

def build_vector_store(pdf_path):

    vision_llm = ChatOllama(
        model=VISION_MODEL,
        temperature=0
    )

    knowledge_base = []

    # TEXT
    
    text = extract_text(pdf_path)

    if text.strip():

        knowledge_base.append(text)
    
    # IMAGES
    
    image_paths = extract_images(pdf_path)

    for image_path in image_paths:

        caption = describe_image(
            image_path,
            vision_llm
        )

        knowledge_base.append(
            f"IMAGE CONTENT:\n{caption}"
        )
    
    # TABLES
    
    table_texts = extract_tables(pdf_path)

    for table in table_texts:

        knowledge_base.append(
            f"TABLE CONTENT:\n{table}"
        )
    
    # CHUNKING
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents(
        knowledge_base
    )
    
    # EMBEDDINGS
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    vector_db = FAISS.from_documents(
        docs,
        embeddings
    )

    return vector_db

# QUESTION ANSWERING

def answer_question(vector_db, question):

    docs = vector_db.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    llm = ChatOllama(
        model=VISION_MODEL,
        temperature=0
    )

    prompt = f"""
    You are a document assistant.

    Answer ONLY from the provided context.

    If the answer is not found,
    say:
    "The information is not present in the document."

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return response.content

# STREAMLIT UI

st.set_page_config(
    page_title="Multimodal PDF RAG",
    layout="wide"
)

st.title("Multimodal PDF RAG")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.read())

        pdf_path = tmp.name

    if "vector_db" not in st.session_state:

        with st.spinner(
            "Processing PDF..."
        ):

            st.session_state.vector_db = (
                build_vector_store(pdf_path)
            )

        st.success(
            "PDF processed successfully!"
        )

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        with st.spinner(
            "Generating answer..."
        ):

            answer = answer_question(
                st.session_state.vector_db,
                question
            )

        st.markdown("### Answer")

        st.write(answer)