from src.utils.logger import get_logger
from typing import Optional

# from langchain_neo4j.graphs.graph_document import Relationship, Node
from langchain.schema import Document

from src.factory.llm import fetch_llm
from src.config import LLMConf
from src.graph.graph_model import Ontology, _Graph
from src.prompts.graph_extractor import get_graph_extractor_prompt


logger = get_logger(__name__)


class GraphExtractor:
    """ Agent able to extract informations in a graph representation format from a given text.
    """

    def __init__(self, conf: LLMConf, ontology: Optional[Ontology]=None):
        self.conf = conf
        self.llm = fetch_llm(conf)
        self.prompt = get_graph_extractor_prompt()


    def extract_graph(self, text: str, source_name: str = "unknown", source_format: str = "unknown") -> _Graph:
        """
        Extracts a graph from a text.
        """

        if self.llm is not None:
            try:
                graph: _Graph = self.llm.with_structured_output(
                    schema=_Graph
                    ).invoke(
                        input=self.prompt.format(
                            input_text=text,
                            source_name=source_name,
                            source_format=source_format
                        )
                    )

                return graph

            except Exception as e:
                logger.warning(f"Error while extracting graph: {e}")