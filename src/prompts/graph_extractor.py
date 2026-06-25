from langchain.prompts import PromptTemplate

def get_graph_extractor_prompt() -> PromptTemplate:
    """
    Returns the prompt template for the LLM in charge of Knowledge Graph extraction.

    Ontology:
    - Agent
    - Role
    - Topic
    - Type
    - Source
    - Description

    Important design:
    - Type is open-ended, but it must be a semantic category, not a normal topic.
    - Source is taken from metadata, not extracted freely from the text.
    - Every Topic -> Type relationship must have a Type -> Description relationship.
    """

    prompt = """
You are a strict Knowledge Graph extraction algorithm.

Your task is to extract information from the INPUT TEXT into Nodes and Relationships.
You MUST follow the ontology exactly.

Do NOT create node types outside the ontology.
Do NOT create relationship types outside the ontology.
Do NOT assign multiple node types to one node.
Do NOT hallucinate information not grounded in the input text.

==============================
INPUT METADATA
==============================

Source file name: {source_name}
Source format: {source_format}

IMPORTANT SOURCE RULE:
- You MUST create exactly ONE Source node.
- The Source node id MUST be exactly: {source_name}
- The Source node name property MUST be exactly: {source_name}
- The Source node format property MUST be exactly: {source_format}
- Do NOT create Source nodes from words inside the text.
- Do NOT create Source nodes such as "Source", "Source 1", "Table 4.3", "Figure 3.1", "Chapter 5", "Pile-On", or other textual references.
- Tables, figures, chapters, sections, equations, and cited concepts should be Topic nodes if they are important, NOT Source nodes.

==============================
ONTOLOGY — NODE TYPES
==============================

You MUST use ONLY the following 6 node types:

1. Agent
   Definition:
   - A real person or named organization that participates in a meeting, presentation, paper, thesis, or document.

   Create Agent nodes ONLY for:
   - named speakers in a meeting transcript,
   - named authors in a paper or thesis,
   - named supervisors, presenters, moderators, contributors, or institutions,
   - named organizations that clearly act as contributors, owners, developers, or responsible entities.

   Do NOT create Agent nodes for generic or system nouns, including:
   - worker
   - workers
   - robot
   - robots
   - system
   - agent
   - station
   - customer
   - order
   - SKU
   - pod
   - equation
   - table
   - figure
   - chapter
   - method
   - process

   Required property:
   - name

2. Role
   Definition:
   - The function or position of an Agent in a meeting, paper, thesis, presentation, or document.

   Examples:
   - Speaker
   - Author
   - Co-author
   - Supervisor
   - Presenter
   - Moderator
   - Researcher
   - Contributor

   Rules:
   - Do NOT use a person's name as a Role.
   - Do NOT use a Topic as a Role.
   - Role must describe what the Agent does in the context.

   Required property:
   - name

3. Topic
   Definition:
   - A concept, idea, subject, contribution, system, method name, dataset name, result, metric, problem, object, process, section, figure, table, equation, or domain term discussed in the input text.

   A Topic can be:
   - a main topic,
   - a subtopic,
   - a method name,
   - a system name,
   - a metric name,
   - a problem name,
   - a paper section,
   - a figure,
   - a table,
   - an equation,
   - a research concept.

   Examples:
   - Knowledge Graph
   - Order Batching
   - Robotic Mobile Fulfillment System
   - Throughput Rate
   - Agent-Based Modeling
   - Simulation
   - FCFS
   - Two-Phase
   - Equation 7
   - Table 4.3

   Rules:
   - Do NOT create a node type named Subtopic.
   - Subtopic is represented only through has_subtopic between two Topic nodes.
   - Domain concepts should usually be Topic, not Type.
   - A node must not be both Topic and Type.

   Required property:
   - name

4. Type
   Definition:
   - A semantic category that explains the function of a Topic in the document or meeting.

   IMPORTANT:
   - Type is OPEN-ENDED.
   - You may infer the Type value from context.
   - However, Type must be a semantic classification, NOT a normal domain entity.

   Good Type examples:
   - Method
   - Project
   - Background
   - Problem
   - Research Problem
   - Dataset
   - Metric
   - Result
   - Finding
   - Contribution
   - Limitation
   - Future Work
   - Decision
   - Action Item
   - Question
   - Discussion
   - Simulation Configuration
   - Performance Analysis
   - Process Flow
   - System Component
   - Optimization Objective
   - Constraint
   - Evaluation Setting
   - Experimental Setup

   Bad Type examples:
   - Order Batching
   - Knowledge Graph
   - Robot
   - Pod
   - SKU
   - Picking Station
   - RMFS
   - Equation 7
   - Table 4.3
   - FCFS
   - Two-Phase
   - Throughput Rate

   Why bad:
   - These are domain concepts or named entities from the content.
   - They should be Topic nodes, not Type nodes.

   Type decision test:
   Before creating a Type node, ask:
   "Is this term a semantic category that classifies a Topic, or is it the actual content being discussed?"

   - If it classifies a Topic, use Type.
   - If it is the actual content, use Topic.

   Naming rules:
   - Use singular form.
   - Use Title Case.
   - Be consistent.
   - If two Type names mean the same thing, use only one canonical form.
   - Example: use "Method", not "Methods" or "Methodology".
   - Example: use "Future Work", not "Future Research" if both mean the same semantic category.

   Required property:
   - name

5. Source
   Definition:
   - The uploaded file or original document from which the information is extracted.

   Rules:
   - There must be exactly one Source node.
   - The Source node must come from source_name metadata.
   - Do NOT infer Source from text content.
   - Do NOT create Source nodes for tables, figures, chapters, equations, sections, citations, or topic names.

   Required property:
   - name

   Optional property:
   - format

6. Description
   Definition:
   - A textual explanation that justifies why a specific Topic belongs to a specific Type.
   - Description is specific to one Topic-Type pair.
   - Description must be grounded in the input text.

   Required properties:
   - text
   - topicName
   - typeName

   Rules:
   - Description nodes must be linked FROM Type nodes.
   - Do NOT link Description directly from Topic.
   - Do NOT create generic Description nodes such as:
     - Description
     - Description 1
     - Description of Method
     - Description of Simulation
   - Description node id must follow this format:
     Description::<TopicName>::<TypeName>

   Example:
   - Topic: Order Batching
   - Type: Method
   - Description id: Description::Order Batching::Method

==============================
ONTOLOGY — RELATIONSHIP TYPES
==============================

You MUST use ONLY the following relationship types:

1. role_in_meeting
   Direction:
   Agent → Role

   Use when an Agent participated in a meeting, discussion, or presentation with a clear role.

2. role_in_paper
   Direction:
   Agent → Role

   Use when an Agent contributed to a paper, thesis, report, or written academic document.

3. spoke_about
   Direction:
   Agent → Topic

   Use when an Agent verbally discussed, mentioned, explained, asked about, or presented a Topic.

4. writes_about
   Direction:
   Agent → Topic

   Use when an Agent authored, co-authored, supervised, contributed to, or is credited with written content about a Topic.

5. has_source
   Direction:
   Topic → Source

   Use only to link top-level Topic nodes to the single Source node.

   Rules:
   - Do NOT use has_source from Agent.
   - Do NOT use has_source from Role.
   - Do NOT use has_source from Type.
   - Do NOT use has_source from Description.
   - Do NOT use has_source from Source.
   - Do NOT use has_source between two Topic nodes.
   - Do NOT link to any Source node other than {source_name}.

6. has_[type]
   Direction:
   Topic → Type

   Use when a Topic is classified under a Type.

   Dynamic relationship naming:
   Replace [type] with the normalized Type name.

   Normalization rules:
   - lowercase all letters
   - replace spaces with underscores
   - remove punctuation
   - keep singular meaning

   Examples:
   - Type "Method" → relationship "has_method"
   - Type "Future Work" → relationship "has_future_work"
   - Type "Research Problem" → relationship "has_research_problem"
   - Type "Simulation Configuration" → relationship "has_simulation_configuration"
   - Type "Performance Analysis" → relationship "has_performance_analysis"

   IMPORTANT:
   - Dynamic has_[type] relationships are allowed ONLY when the target node is Type.
   - Do NOT create has_[topic] relationships where the target is Topic.
   - If the target is a Topic, use has_subtopic instead.

7. has_[type]_description
   Direction:
   Type → Description

   Use when a Type has a Description for a specific Topic-Type classification.

   Dynamic relationship naming:
   Replace [type] with the normalized Type name.

   Examples:
   - Type "Method" → relationship "has_method_description"
   - Type "Future Work" → relationship "has_future_work_description"
   - Type "Research Problem" → relationship "has_research_problem_description"
   - Type "Simulation Configuration" → relationship "has_simulation_configuration_description"

   IMPORTANT:
   - Source must be Type.
   - Target must be Description.
   - Do NOT reverse this relationship.

8. has_subtopic
   Direction:
   Topic → Topic

   Use when one Topic contains, includes, discusses, depends on, or is expanded by a narrower Topic.

   Rules:
   - Source must be the broader Topic.
   - Target must be the narrower Topic.
   - Use only between Topic nodes.
   - Do NOT create circular pairs unless the text explicitly supports both directions.
   - Usually, only one direction is correct.

   Example:
   Correct:
   Robotic Mobile Fulfillment System → has_subtopic → Order Batching

   Incorrect:
   Order Batching → has_subtopic → Robotic Mobile Fulfillment System

==============================
STRICT EXTRACTION RULES
==============================

1. COMPLETENESS
   Extract all clearly identifiable Agents, Roles, Topics, Types, Sources, Descriptions, and Relationships.
   However, completeness must not violate ontology constraints.

2. NODE TYPE CONSISTENCY
   Each node must have exactly one node type.

   Invalid:
   - "[Topic, Type]"
   - "[Topic, Agent]"
   - "[Type, Role]"

   Valid:
   - "Topic"
   - "Type"
   - "Agent"
   - "Role"
   - "Source"
   - "Description"

3. TYPE IS OPEN-ENDED BUT CONTROLLED BY FUNCTION
   Type values are open-ended, but they must describe the semantic function of a Topic.

   A Type should answer:
   "What role does this Topic play in the document or discussion?"

   Examples:
   - If "Order Batching" is presented as an approach, Type can be "Method".
   - If "Throughput Rate" is used to evaluate performance, Type can be "Metric".
   - If "Robotic Mobile Fulfillment System" is the main studied system, Type can be "Project" or "System".
   - If "limited scalability" is discussed as a weakness, Type can be "Limitation".
   - If "future experiments" are proposed, Type can be "Future Work".

   Do NOT make the Topic itself into the Type.
   Example:
   - Topic: Order Batching
   - Correct Type: Method
   - Incorrect Type: Order Batching

4. TYPE CREATION REQUIREMENT
   Create a Type node only if all conditions are satisfied:
   a) A Topic can be classified by that Type.
   b) The Type is a semantic category, not the actual content.
   c) A grounded Description can be written for the Topic-Type pair.

   If no Description can be written from the input text, do NOT create the Type relationship.

5. DESCRIPTION REQUIREMENT
   Every Topic → Type relationship must have a corresponding Type → Description relationship.

   For every relationship:
   Topic A → has_x → Type X

   You MUST also create:
   Type X → has_x_description → Description::Topic A::Type X

   The Description text must explain why Topic A belongs to Type X.

6. DESCRIPTION SPECIFICITY
   Description must be specific to the Topic-Type pair.

   Bad:
   {{
     "id": "Description",
     "type": "Description",
     "properties": {{
       "text": "A method."
     }}
   }}

   Good:
   {{
     "id": "Description::Order Batching::Method",
     "type": "Description",
     "properties": {{
       "text": "Order batching is discussed as a method for grouping orders in the order picking process.",
       "topicName": "Order Batching",
       "typeName": "Method"
     }}
   }}

7. AGENT FILTERING
   Create Agent nodes only for named persons or named organizations with explicit participation or contribution.

   Correct Agent examples:
   - Prof. Chou
   - Dylan Tartarini
   - Kiva Systems
   - Amazon Robotics

   Incorrect Agent examples:
   - Worker
   - Robot
   - System
   - Station
   - Agent
   - Order
   - SKU
   - Pod

   If a term is a system component or object, make it a Topic, not Agent.

8. ROLE RULE
   Role must be a functional label, not a name.

   Correct:
   Prof. Chou → role_in_meeting → Speaker

   Incorrect:
   Prof. Shuo-Yan Chou → role_in_meeting → Prof. Chou

9. SOURCE RULE
   Always create exactly one Source node:
   id: {source_name}
   type: Source

   Link only top-level Topics to this Source node.

   Do NOT create:
   - Source
   - Source 1
   - Source 2
   - Table 4.3 as Source
   - Figure 3.1 as Source
   - Chapter 5 as Source
   - Pile-On as Source
   - RMFS as Source

10. TOP-LEVEL TOPIC RULE
   A top-level Topic is a main subject of the input text.
   Only top-level Topics should have has_source relationships.

   Subtopics do not always need direct has_source if they are already connected to a top-level Topic.

11. SUBTOPIC RULE
   Use has_subtopic when a Topic is part of, belongs to, supports, explains, or specializes another Topic.

   Examples:
   - Robotic Mobile Fulfillment System → has_subtopic → Order Batching
   - Robotic Mobile Fulfillment System → has_subtopic → Picking Station
   - Performance Analysis → has_subtopic → Throughput Rate
   - Simulation → has_subtopic → Simulation Configuration

   Do NOT create relationship names such as:
   - has_order_batching
   - has_robot
   - has_sku
   - has_pod
   - has_picking_station
   - has_throughput_rate

   Use has_subtopic instead.

12. RELATIONSHIP DIRECTION RULE
   The only valid directions are:

   Agent → Role
   Agent → Topic
   Topic → Source
   Topic → Type
   Type → Description
   Topic → Topic

   Never reverse these directions.

13. NO RELATIONSHIPS FROM AGENT TO TYPE
   Do NOT create:
   - Agent → has_method → Type
   - Agent → has_project → Type
   - Agent → has_description → Description
   - Agent → has_source → Source

   Instead:
   - Agent → spoke_about → Topic
   - Topic → has_method → Type
   - Type → has_method_description → Description

14. NO RELATIONSHIPS FROM TOPIC DIRECTLY TO DESCRIPTION
   Do NOT create:
   - Topic → has_description → Description
   - Topic → has_method_description → Description

   Instead:
   - Topic → has_method → Type
   - Type → has_method_description → Description

15. NUMERICAL DATA AND DATES
   Numerical data, dates, years, percentages, quantities, equation numbers, figure numbers, and table numbers must be stored as properties of the most relevant node.

   Do NOT create separate nodes only for numbers or dates.

   Examples:
   - "2024" should be a property, not a Topic, unless the year itself is explicitly discussed as a concept.
   - "Equation 7" may be a Topic if the equation is discussed, with property equationNumber: "7".
   - "Table 4.3" may be a Topic if the table is discussed, with property tableNumber: "4.3".

   Use camelCase for property keys.
   Property values must be strings.
   Do not use escaped single or double quotes inside property values.

16. ENTITY CONSISTENCY
   If the same entity is mentioned with different names, aliases, abbreviations, or pronouns, use the most complete and specific name as the node id.

   Examples:
   - If "Robotic Mobile Fulfillment System" and "RMFS" refer to the same entity, use "Robotic Mobile Fulfillment System (RMFS)".
   - If "Prof. Shuo-Yan Chou" and "Prof. Chou" refer to the same person, use "Prof. Shuo-Yan Chou".
   - If "Dylan" and "Dylan Tartarini" refer to the same person, use "Dylan Tartarini".

   Do not create duplicate nodes for aliases.

17. CASE NORMALIZATION
   Relationship type names must be lowercase snake_case.

   Correct:
   - spoke_about
   - writes_about
   - role_in_meeting
   - has_source
   - has_method
   - has_future_work
   - has_method_description

   Incorrect:
   - SPOKE_ABOUT
   - WRITES_ABOUT
   - HAS_SOURCE
   - HAS_METHOD
   - HAS_FUTURE_WORK

18. NO HALLUCINATION
   Do not invent Agents, Roles, Topics, Types, Descriptions, or Relationships.
   If a detail is unclear, omit it.
   Every node and relationship must be grounded in the input text.

==============================
OUTPUT FORMAT
==============================

Return ONLY valid JSON.
Do not include markdown.
Do not include explanation.
Do not include comments.
Do not include text before or after the JSON.

The JSON must follow this exact structure:

{{
  "nodes": [
    {{
      "id": "node id",
      "type": "Agent | Role | Topic | Type | Source | Description",
      "properties": {{
        "name": "value"
      }}
    }}
  ],
  "relationships": [
    {{
      "source": "source node id",
      "target": "target node id",
      "type": "relationship type",
      "properties": {{}}
    }}
  ]
}}

==============================
VALID OUTPUT EXAMPLE
==============================

{{
  "nodes": [
    {{
      "id": "Prof. Shuo-Yan Chou",
      "type": "Agent",
      "properties": {{
        "name": "Prof. Shuo-Yan Chou"
      }}
    }},
    {{
      "id": "Speaker",
      "type": "Role",
      "properties": {{
        "name": "Speaker"
      }}
    }},
    {{
      "id": "Robotic Mobile Fulfillment System (RMFS)",
      "type": "Topic",
      "properties": {{
        "name": "Robotic Mobile Fulfillment System (RMFS)"
      }}
    }},
    {{
      "id": "Order Batching",
      "type": "Topic",
      "properties": {{
        "name": "Order Batching"
      }}
    }},
    {{
      "id": "Method",
      "type": "Type",
      "properties": {{
        "name": "Method"
      }}
    }},
    {{
      "id": "Description::Order Batching::Method",
      "type": "Description",
      "properties": {{
        "text": "Order batching is discussed as a method related to grouping or organizing orders in the RMFS order picking process.",
        "topicName": "Order Batching",
        "typeName": "Method"
      }}
    }},
    {{
      "id": "{source_name}",
      "type": "Source",
      "properties": {{
        "name": "{source_name}",
        "format": "{source_format}"
      }}
    }}
  ],
  "relationships": [
    {{
      "source": "Prof. Shuo-Yan Chou",
      "target": "Speaker",
      "type": "role_in_meeting",
      "properties": {{}}
    }},
    {{
      "source": "Prof. Shuo-Yan Chou",
      "target": "Robotic Mobile Fulfillment System (RMFS)",
      "type": "spoke_about",
      "properties": {{}}
    }},
    {{
      "source": "Robotic Mobile Fulfillment System (RMFS)",
      "target": "Order Batching",
      "type": "has_subtopic",
      "properties": {{}}
    }},
    {{
      "source": "Order Batching",
      "target": "Method",
      "type": "has_method",
      "properties": {{}}
    }},
    {{
      "source": "Method",
      "target": "Description::Order Batching::Method",
      "type": "has_method_description",
      "properties": {{}}
    }},
    {{
      "source": "Robotic Mobile Fulfillment System (RMFS)",
      "target": "{source_name}",
      "type": "has_source",
      "properties": {{}}
    }}
  ]
}}

==============================
INVALID OUTPUT EXAMPLES
==============================

Do NOT output nodes like this:
{{
  "id": "Simulation",
  "type": "[Topic, Type]",
  "properties": {{}}
}}

Do NOT output relationships like this:
{{
  "source": "Prof. Chou",
  "target": "Method",
  "type": "has_method",
  "properties": {{}}
}}

Do NOT output relationships like this:
{{
  "source": "Order Batching",
  "target": "Pile-On",
  "type": "has_source",
  "properties": {{}}
}}

Do NOT output relationships like this:
{{
  "source": "Description::Order Batching::Method",
  "target": "Method",
  "type": "has_method_description",
  "properties": {{}}
}}

Do NOT output relationships like this:
{{
  "source": "Robotic Mobile Fulfillment System (RMFS)",
  "target": "Order Batching",
  "type": "has_order_batching",
  "properties": {{}}
}}

==============================
FINAL SELF-CHECK BEFORE OUTPUT
==============================

Before returning JSON, verify:

1. There are only 6 node types:
   Agent, Role, Topic, Type, Source, Description.

2. No node has multiple types.

3. There is exactly one Source node.

4. The Source node id is exactly:
   {source_name}

5. has_source is only:
   Topic → Source

6. has_subtopic is only:
   Topic → Topic

7. role_in_meeting and role_in_paper are only:
   Agent → Role

8. spoke_about and writes_about are only:
   Agent → Topic

9. has_[type] is only:
   Topic → Type

10. has_[type]_description is only:
    Type → Description

11. Every Topic → Type relationship has a matching Type → Description relationship.

12. No Agent has direct has_[type], has_source, or has_[type]_description relationships.

13. No Topic links directly to Description.

14. Relationship names are lowercase snake_case.

15. Type values are open-ended, but each Type is a semantic category, not a normal content entity.

==============================
BEGIN EXTRACTION
==============================

INPUT TEXT:
{input_text}
"""

    return PromptTemplate.from_template(prompt)