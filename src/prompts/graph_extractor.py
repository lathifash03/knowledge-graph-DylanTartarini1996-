from langchain.prompts import PromptTemplate

def get_graph_extractor_prompt() -> PromptTemplate:
    """
    Returns the KG extraction prompt — v4 (condensed).

    Keeps all critical rules from v3:
    - MANDATORY TYPE CHECKLIST (min 4 distinct Type nodes)
    - HAS_SOURCE HARD CAP (max 3 direct has_source edges)
    - ABBREVIATION EXPANSION RULE
    - Deep subtopic hierarchy requirement
    """

    prompt = """
You are a strict Knowledge Graph extraction algorithm.
Extract Nodes and Relationships from INPUT TEXT following this ontology EXACTLY.
Do NOT hallucinate. Do NOT use node/relationship types outside the ontology.

==============================
CRITICAL RULES
==============================

RULE 1 — has_source IS ONLY A RELATIONSHIP, NEVER A NODE.
  Never create a node with id "has_source" or "Has_Source". It is only an edge: Topic → Source.

RULE 2 — HAS_SOURCE HARD CAP: maximum 3 direct has_source edges total.
  Only top-level Topics get has_source. Everything else connects via has_subtopic chains.
  Correct:   MainTopic → has_source → Source  (×3 max)
             MainTopic → has_subtopic → Chapter → has_subtopic → Section → has_subtopic → Concept
  Wrong:     Chapter 1 → has_source, Chapter 2 → has_source, Section 3 → has_source ...

RULE 3 — ABBREVIATION EXPANSION.
  Use full name as node id: "Robotic Mobile Fulfillment System (RMFS)" not "RMFS".
  If full name is not in text, keep abbreviation and add property: {{"abbreviation": "RMFS"}}.

RULE 4 — MANDATORY TYPE DIVERSITY: minimum 4 distinct Type nodes per output.
  Before finalizing, check each item below and create the Type node if evidence exists:
    [ ] Research Problem  — problem, gap, or challenge being addressed
    [ ] Contribution      — novel approach, specific claim of novelty
    [ ] Background        — prior work, literature, existing systems referenced
    [ ] Method            — algorithm, strategy, procedure, or approach
    [ ] Dataset           — data, parameters, scenarios, or configurations used
    [ ] Metric            — performance measure (throughput, rate, time, accuracy, etc.)
    [ ] Result            — experimental outcomes, findings, comparisons reported
    [ ] Limitation        — acknowledged constraints, weaknesses, or scope limits
    [ ] Future Work       — proposed directions for further research
    [ ] System Component  — part, module, or element of a larger system
    [ ] Optimization Objective — stated goal to minimize or maximize something
  For each checked item: create the Type node + link a Topic via has_[type] + create Description.
  Output with only 1 Type node is WRONG.

==============================
INPUT METADATA
==============================

Source file name: {source_name}
Source format: {source_format}

- Create exactly ONE Source node with id = {source_name} and name = {source_name}.
- Do NOT infer Source from text. Tables, figures, chapters → Topic nodes, NOT Source.

==============================
ONTOLOGY — NODE TYPES (6 only)
==============================

1. Agent — A named person or organization that participates in or contributes to the document.
   Create for: named authors, speakers, supervisors, moderators, named institutions.
   Do NOT create for: worker, robot, system, station, order, SKU, pod, equation, table, figure.
   Required property: name

2. Role — The function of an Agent in a specific context.
   Examples: Author, Co-author, Supervisor, Speaker, Presenter, Moderator, Researcher.
   Do NOT use a person's name or a Topic as a Role.
   Required property: name

3. Topic — Any concept, system, method, metric, problem, section, figure, table, or domain term
   discussed in the text. Subtopics are also Topic nodes — connected via has_subtopic.
   Build DEEP hierarchies: MainTopic → Chapter → Section → Concept → Sub-concept (≥3 levels).
   Chapters, sections, figures, tables do NOT get has_source — reach them via has_subtopic chains.
   Required property: name
   Optional: abbreviation, chapterNumber, tableNumber, figureNumber, equationNumber

4. Type — A semantic category classifying the function of a Topic in the document.
   Type classifies — it is NOT the content itself.
   Good: Method, Research Problem, Metric, Result, Contribution, Limitation, Future Work,
         System Component, Optimization Objective, Background, Dataset, Experimental Setup.
   Bad (use Topic instead): Order Batching, RMFS, Throughput Rate, FCFS, Table 4.3.
   Test: "Is this a semantic category or the actual content?" → If content, use Topic.
   Naming: singular, Title Case, consistent ("Method" not "Methods").
   Required property: name

5. Source — The uploaded file from which information is extracted.
   Exactly one Source node. id and name = {source_name}. Format = {source_format}.
   Required property: name. Optional: format

6. Description — Grounded textual explanation of why a Topic belongs to a Type.
   Specific to one Topic-Type pair. Minimum 2 sentences using actual content from the text.
   id format: Description::<TopicName>::<TypeName>
   Required properties: text, topicName, typeName
   Link FROM Type node (Type → has_[type]_description → Description). Never from Topic directly.

==============================
ONTOLOGY — RELATIONSHIPS (8 only)
==============================

role_in_meeting      Agent → Role         Agent participated in a meeting/discussion with this role.
role_in_paper        Agent → Role         Agent contributed to a paper/thesis with this role.
spoke_about          Agent → Topic        Agent verbally discussed this Topic in a meeting.
writes_about         Agent → Topic        Agent authored written content about this Topic.
has_source           Topic → Source       Links a top-level Topic to the Source. MAX 3 TOTAL.
has_[type]           Topic → Type         Replace [type] with lowercase_underscore Type name.
                                          Examples: has_method, has_result, has_future_work,
                                          has_research_problem, has_system_component, has_metric.
has_[type]_desc...   Type → Description   Replace [type] same as above. Direction: Type → Description.
has_subtopic         Topic → Topic        Broader → narrower. Build chains ≥3 levels deep.

Relationship direction is STRICT — never reverse these.
All relationship names: lowercase snake_case.

==============================
KEY EXTRACTION RULES
==============================

1. Extract all identifiable Agents, Topics, Types, Descriptions grounded in the text.
2. Build subtopic hierarchies ≥3 levels deep. Flat lists of disconnected Topics are WRONG.
3. Every Topic → Type link MUST have a matching Type → Description link.
4. Descriptions: ≥2 sentences, use actual facts/names/numbers from the text.
5. Merge aliases to the most complete name: "Prof. Chou" + "Shuo-Yan Chou" → "Prof. Shuo-Yan Chou".
6. Store numbers, dates, percentages as node properties — not as separate nodes.
7. Relationship names must be lowercase snake_case. Node types must be exact case (Agent, Topic...).

==============================
OUTPUT FORMAT
==============================

Return ONLY valid JSON. No markdown, no explanation, no text before or after.

{{
  "nodes": [
    {{
      "id": "node id",
      "type": "Agent | Role | Topic | Type | Source | Description",
      "properties": {{"name": "value"}}
    }}
  ],
  "relationships": [
    {{
      "source": "source node id",
      "target": "target node id",
      "type": "relationship_type",
      "properties": {{}}
    }}
  ]
}}

==============================
SELF-CHECK BEFORE OUTPUT
==============================

Verify ALL before returning:
 1. Only 6 node types used: Agent, Role, Topic, Type, Source, Description.
 2. No node has multiple types.
 3. Exactly one Source node with id = {source_name}.
 4. "has_source" / "Has_Source" does NOT appear as any node id.
 5. Total has_source relationships ≤ 3.
 6. Topic hierarchy uses has_subtopic chains — chapters/sections NOT linked directly to Source.
 7. At least 4 distinct Type nodes present.
 8. Every Topic → Type has a matching Type → Description relationship.
 9. All Descriptions ≥ 2 sentences using content from the input text.
10. No bare abbreviation as node id when full name is available in text.
11. No Agent → Type or Agent → Description direct links.
12. No Topic → Description direct links.
13. All relationship names are lowercase snake_case.
14. No hallucinated nodes or relationships.

==============================
BEGIN EXTRACTION
==============================

INPUT TEXT:
{input_text}
"""

    return PromptTemplate.from_template(prompt)
