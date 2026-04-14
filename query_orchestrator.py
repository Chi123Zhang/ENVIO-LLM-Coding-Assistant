import os
import json
from typing import Dict
from openai import OpenAI


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def _parse_json_safely(text: str) -> Dict:
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return json.loads(text)


def understand_query(
    user_id: str,
    raw_query: str,
    has_uploaded_project_doc: bool = False
) -> Dict:
    """
    Person 2 - Query Understanding
    Produces a Query Understanding Object.
    """

    client = _get_openai_client()

    prompt = f"""
You are the query understanding module of a personalized explanation agent.

Your job is to classify the user's query and decide what type of context is needed.

Return ONLY valid JSON with exactly these keys:

- query_id
- user_id
- raw_query
- query_type
- topic
- subtopics
- intent
- domain
- requires_background_retrieval
- requires_project_context
- requires_external_knowledge
- needs_clarification
- clarification_reason
- suggested_clarification_question
- recommended_background_chunk_types
- recommended_next_step

Allowed values:
- query_type must be one of:
  ["concept_explanation", "project_explanation", "comparison_question", "workflow_explanation", "document_based_question", "clarification_needed"]

- recommended_background_chunk_types must be chosen from:
  ["role_identity", "domain_context", "technical_exposure", "knowledge_boundary", "expression_preference", "current_project"]

- recommended_next_step must be one of:
  ["clarification", "retrieve_background_then_explain", "retrieve_background_and_project_then_explain", "external_knowledge_then_explain"]

Important routing rules:
1. If the user asks a general concept question like:
   - What is RAG?
   - What is an orchestrator?
   - What is a vector database?
   - Explain API gateway
   then this is usually:
   - query_type = "concept_explanation"
   - requires_external_knowledge = true
   - requires_project_context = false

2. Only set requires_project_context = true when the question clearly depends on uploaded project documents, such as:
   - Explain this project
   - Explain this architecture
   - What is the study design in the uploaded document?
   - Summarize the uploaded note
   - What does this project do?

3. If the question is vague and depends on missing context, set:
   - needs_clarification = true

4. Background retrieval is usually useful for personalization, unless the query is purely generic and user modeling would add little value.

Current context:
- user_id = "{user_id}"
- has_uploaded_project_doc = {str(has_uploaded_project_doc)}
- raw_query = "{raw_query}"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    result = _parse_json_safely(content)

    valid_query_types = {
        "concept_explanation",
        "project_explanation",
        "comparison_question",
        "workflow_explanation",
        "document_based_question",
        "clarification_needed",
    }

    valid_next_steps = {
        "clarification",
        "retrieve_background_then_explain",
        "retrieve_background_and_project_then_explain",
        "external_knowledge_then_explain",
    }

    valid_chunk_types = {
        "role_identity",
        "domain_context",
        "technical_exposure",
        "knowledge_boundary",
        "expression_preference",
        "current_project",
    }

    if result.get("query_type") not in valid_query_types:
        result["query_type"] = "concept_explanation"

    if not isinstance(result.get("subtopics"), list):
        result["subtopics"] = []

    for key in [
        "requires_background_retrieval",
        "requires_project_context",
        "requires_external_knowledge",
        "needs_clarification",
    ]:
        if not isinstance(result.get(key), bool):
            result[key] = False

    if not isinstance(result.get("recommended_background_chunk_types"), list):
        result["recommended_background_chunk_types"] = []

    result["recommended_background_chunk_types"] = [
        x for x in result["recommended_background_chunk_types"]
        if x in valid_chunk_types
    ]

    if result.get("recommended_next_step") not in valid_next_steps:
        result["recommended_next_step"] = "retrieve_background_then_explain"

    result["user_id"] = user_id
    result["raw_query"] = raw_query
    if not result.get("query_id"):
        result["query_id"] = "q_auto"

    return result


def route_query(query_understanding_object: Dict) -> Dict:
    """
    Person 2 - Routing decision
    """

    if query_understanding_object.get("needs_clarification", False):
        return {
            "route": "clarification",
            "message": query_understanding_object.get(
                "suggested_clarification_question",
                "Could you clarify what kind of explanation you want?"
            )
        }

    if query_understanding_object.get("requires_external_knowledge", False):
        return {
            "route": "external_knowledge_then_expression",
            "background_request": {
                "user_id": query_understanding_object["user_id"],
                "query": query_understanding_object["raw_query"],
                "recommended_background_chunk_types": query_understanding_object.get(
                    "recommended_background_chunk_types", []
                )
            }
        }

    if query_understanding_object.get("requires_project_context", False):
        return {
            "route": "background_and_project_then_expression",
            "background_request": {
                "user_id": query_understanding_object["user_id"],
                "query": query_understanding_object["raw_query"],
                "recommended_background_chunk_types": query_understanding_object.get(
                    "recommended_background_chunk_types", []
                )
            }
        }

    return {
        "route": "background_retrieval_then_expression",
        "background_request": {
            "user_id": query_understanding_object["user_id"],
            "query": query_understanding_object["raw_query"],
            "recommended_background_chunk_types": query_understanding_object.get(
                "recommended_background_chunk_types", []
            )
        }
    }


def process_query(
    user_id: str,
    raw_query: str,
    has_uploaded_project_doc: bool = False
) -> Dict:
    """
    One-step wrapper:
    returns both query understanding and routing decision.
    """

    q_obj = understand_query(
        user_id=user_id,
        raw_query=raw_query,
        has_uploaded_project_doc=has_uploaded_project_doc
    )

    routing = route_query(q_obj)

    return {
        "query_understanding_object": q_obj,
        "routing_decision": routing
    }
