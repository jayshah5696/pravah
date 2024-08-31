import streamlit as st
import uuid
import lancedb
import pypdf
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import OpenAIEmbeddings
from ..pravah.retrieval import RetrievalEngine
# Initialize LanceDB connection
db = lancedb.connect("../my_db")

# Create table if it doesn't exist
model = OpenAIEmbeddings(name='text-embedding-3-small',dim=1536)
class Document(LanceModel):
    document_id: str
    filename: str
    chunk: str = model.SourceField()
    vector: Vector(1536) = model.VectorField()
    page_number: int  # Add page number field
    deleted: bool = False

table_mode = "append" if "document_chunks" in db.table_names() else "create"
if table_mode == "create":
    db.create_table("document_chunks", schema=Document)

def chunk_text(text, chunk_size, overlap, chunking_method):
    if chunking_method == 'tokens':
        return RetrievalEngine.chunk_text_by_tokens(text, chunk_size, overlap)
    elif chunking_method == 'text':
        return RetrievalEngine.chunk_text(text, chunk_size, overlap)
    elif chunking_method == 'regex':
        return RetrievalEngine.chunk_regax(text, chunk_size, overlap)

def process_pdf(uploaded_file):
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text_by_page = {}
    for page_number, page in enumerate(pdf_reader.pages, start=1):
        text_by_page[page_number] = page.extract_text()
    return text_by_page

def index_document(document_id, filename, text_by_page, config):
    # Open the LanceDB table
    table = db.open_table("document_chunks")

    # Iterate over each page and its text
    for page_number, text in text_by_page.items():
        # Chunk the text using the chunk_text function
        chunks = chunk_text(text, config.chunk_size, config.overlap, config.chunking_method)

        # Index chunks into LanceDB with page number as metadata
        table.add(
            [
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk": chunk,
                    "page_number": page_number,  # Add page number as metadata
                    "deleted": False
                }
                for chunk in chunks
            ]
        )

def delete_document(document_id):
    table = db.open_table("document_chunks")
    table.update(where={"document_id": document_id}, set={"deleted": True})

def document_management_page(config):
    st.title("Document Management")

    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            document_id = uuid.uuid4()
            filename = uploaded_file.name
            text = process_pdf(uploaded_file)
            index_document(document_id, filename, text, config)
            st.success(f"Indexed {filename}")

    # Display document list
    table = db.open_table("document_chunks")
    documents = table.search("").to_list()
    if documents:
        st.subheader("Uploaded Documents")
        for doc in documents:
            st.write(f"**Filename:** {doc['filename']}, **Deleted:** {doc['deleted']}")

            if not doc['deleted']:
                if st.button(f"Delete {doc['filename']}"):
                    delete_document(doc['document_id'])
                    st.experimental_rerun()  # Refresh the page after deletion
    else:
        st.info("No documents uploaded yet.")