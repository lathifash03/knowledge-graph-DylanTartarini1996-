import time
import traceback

from src.utils.logger import get_logger
from typing import Optional

from src.factory.llm import fetch_llm
from src.config import LLMConf
from src.graph.graph_model import Ontology, _Graph
from src.prompts.graph_extractor import get_graph_extractor_prompt


logger = get_logger(__name__)

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 65  # wait 65s between retries on rate limit


class GraphExtractor:
    """ Agent able to extract informations in a graph representation format from a given text.
    """

    def __init__(self, conf: LLMConf, ontology: Optional[Ontology]=None):
        self.conf = conf
        self.llm = fetch_llm(conf)
        self.prompt = get_graph_extractor_prompt()


    def extract_graph(self, text: str, source_name: str = "unknown", source_format: str = "unknown") -> _Graph:
        """
        Extracts a graph from a text. Retries on rate limit errors.
        """

        if self.llm is None:
            return None

        for attempt in range(1, MAX_RETRIES + 1):
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
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()

                is_permanent = "limit: 0" in error_str or "limit_exceeded" in error_str.lower() and "limit: 0" in error_str

                if is_rate_limit and not is_permanent and attempt < MAX_RETRIES:
                    logger.warning(f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). Waiting {RETRY_WAIT_SECONDS}s...")
                    time.sleep(RETRY_WAIT_SECONDS)
                else:
                    logger.error(f"Error while extracting graph (attempt {attempt}): {e}\n{traceback.format_exc()}")
                    return None