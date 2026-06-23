import streamlit as st

from src.config import Configuration
from src.graph.knowledge_graph import KnowledgeGraph
from src.ingestion.chunks_ingestor import ChunksIngestor
from src.ingestion.embedder import ChunkEmbedder
from src.ingestion.graph_miner import GraphMiner

from pgs.utils import get_configuration_from_env

import os

st.set_page_config(
    page_title="Upload Chunks",
    page_icon="🧩",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    ## 🧩 Ingest Pre-built Chunks into the Graph 🕸️

    Use the box below to upload a **JSONL** file where each line is a chunk object.

    Expected fields per record:
    | Field | Required | Description |
    |---|---|---|
    | `text` | ✅ | Chunk text |
    | `doc_id` | ✅ | Document identifier (groups chunks) |
    | `chunk_id` | ✅ | Original chunk identifier |
    | `index` | recommended | Integer position within doc |
    | `source_path` | optional | Original file path |
    | `source_kind` | optional | File type (e.g. `pdf`) |
    | `n_chunks` | optional | Total chunks in document |

    The pipeline will skip document loading, cleaning, and chunking and run directly:
    **Embed → Extract Graph → Upload to Knowledge Graph**
    """
)

CONF_PATH = f"{os.getcwd()}/configuration.json"

conf = None
st.session_state.setdefault("chunks_ingest_clicked", False)

try:
    conf = Configuration.from_file(CONF_PATH)
except Exception:
    conf = get_configuration_from_env()

if conf:
    input_mode = st.radio(
        "How would you like to provide the chunks file?",
        options=["Upload file", "Local file path"],
        horizontal=True,
    )

    uploaded_file = None
    local_path = None

    if input_mode == "Upload file":
        uploaded_file = st.file_uploader(
            label="Upload JSONL Chunks File",
            type=["jsonl", "json"],
            accept_multiple_files=False,
        )
    else:
        local_path = st.text_input(
            label="Path to a .jsonl file or a folder containing .jsonl files",
            placeholder="/path/to/chunks_data/  or  /path/to/chunks.jsonl",
        )

    ready = uploaded_file is not None or (local_path and local_path.strip())

    if ready:
        st.session_state["chunks_ingest_clicked"] = st.button(
            label="Ingest into Knowledge Graph",
            icon="🧩",
        )

        if st.session_state["chunks_ingest_clicked"]:
            with st.status("Ingesting Chunks...", expanded=True) as status:

                st.write("Loading chunks from file...")
                ingestor = ChunksIngestor()
                if uploaded_file is not None:
                    docs = ingestor.load_from_bytes(uploaded_file.getvalue())
                else:
                    path = local_path.strip()
                    if os.path.isdir(path):
                        docs = ingestor.load_from_folder(path)
                    else:
                        docs = ingestor.load_from_file(path)

                total_chunks = sum(len(d.chunks) for d in docs)
                st.write(f"Found **{total_chunks} chunks** across **{len(docs)} documents**.")

                st.write("Setting up pipeline components...")
                embedder = ChunkEmbedder(conf=conf.embedder_conf)
                graph_miner = GraphMiner(
                    conf=conf.re_model_conf,
                    ontology=conf.database.ontology,
                )
                knowledge_graph = KnowledgeGraph(
                    conf=conf.database,
                    embeddings_model=embedder.embeddings,
                )

                if not knowledge_graph._driver.verify_authentication():
                    st.error("Check your Neo4j configuration!")
                else:
                    st.write("Embedding chunks...")
                    docs = embedder.embed_documents_chunks(docs)

                    st.write("Extracting Knowledge Graph from chunks...")
                    docs = graph_miner.mine_graph_from_docs(docs=docs)

                    st.write("Uploading to Knowledge Graph...")
                    knowledge_graph.add_documents(docs)

                    st.write("Updating communities and centralities...")
                    knowledge_graph.update_centralities_and_communities()

                    status.update(
                        label="Done!",
                        state="complete",
                        expanded=False,
                    )

            if status._current_state == "complete":
                st.success(f"Ingested {total_chunks} chunks from {len(docs)} documents.")
