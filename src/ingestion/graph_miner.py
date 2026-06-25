from src.utils.logger import get_logger
from typing import List, Optional

from src.agents.graph_extractor import GraphExtractor
from src.graph.graph_model import _Graph, Ontology, map_to_lc_graph
from src.config import LLMConf
from src.schema import ProcessedDocument

logger = get_logger(__name__)


class GraphMiner:
    """ Contains methods to mine graphs from a (list of) `ProcessedDocument`."""

    def __init__(self, conf: LLMConf, ontology: Optional[Ontology]=None):
        self.graph_extractor = GraphExtractor(conf=conf, ontology=ontology)

        if self.graph_extractor:
            logger.info(f"GraphMiner initialized.")


    def mine_graph_from_doc_chunks(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Mines a graph from a `ProcessedDocument` instance.
        """
        source_name = doc.filename or "unknown"
        source_format = (doc.metadata or {}).get("source_kind", source_name.rsplit(".", 1)[-1] if "." in source_name else "unknown")

        for chunk in doc.chunks:
            try:
                graph: _Graph = self.graph_extractor.extract_graph(
                    text=chunk.text,
                    source_name=source_name,
                    source_format=source_format
                )

                graph_doc = map_to_lc_graph(graph, source_content=chunk.text)

                chunk.nodes = graph_doc.nodes
                chunk.relationships = graph_doc.relationships

            except Exception as e:
                logger.warning(f"Error while mining graph: {e}")

            logger.info(f"Created a graph representation for {len(doc.chunks)} chunks.")

        return doc


    def mine_graph_from_docs(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """
        Mines graphs from a list of `ProcessedDocument` instances.
        """
        return [self.mine_graph_from_doc_chunks(doc) for doc in docs]
    