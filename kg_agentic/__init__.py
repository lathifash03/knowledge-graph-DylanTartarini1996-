"""KG-agentic verification layer.

A research add-on to the DylanTartarini1996 knowledge-graph pipeline that adds
temporal metadata, temporal-validity checking, node trust scoring and an
agentic verification loop on top of the existing Neo4j knowledge graph.

Phases
------
1. Temporal metadata migration   -> ``kg_agentic.neo4j_client``
2. Temporal validity check        -> ``kg_agentic.temporal_validity``
3. Node trust scoring             -> ``kg_agentic.node_trust``
4. Agentic verification           -> ``kg_agentic.agentic_verifier``

Evaluation (RAGAS, offline)       -> ``kg_agentic.evaluation``
Entry point / demo                -> ``kg_agentic.main``
"""

__all__ = [
    "config",
    "neo4j_client",
    "temporal_validity",
    "node_trust",
    "agentic_verifier",
    "evaluation",
]
