import os
import json
import sqlite3
from typing import Dict, List
from openai import OpenAI


DB_PATH = "background_memory.db"


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def _parse_json_safely(text: str) -> Dict:
    if not text:
        raise ValueError("Empty response from model.")

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    return json.loads(text)


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            structured_profile_json TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS background_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            chunk_type TEXT,
            chunk_text TEXT
        )
    """)

    conn.commit()
    conn.close()


def _combine_raw_background(raw_background_inputs: List[Dict]) -> str:
    texts = []
    for item in raw_background_inputs:
        raw_text = item.get("raw_text", "")
        if raw_text:
            texts.append(raw_text.strip())
    return "\n\n".join(texts)


def _fallback_profile(raw_text: str) -> Dict:
    text_lower = raw_text.lower()

    role_lens = "general"
    if "product manager" in text_lower or "pm" in text_lower:
        role_lens = "pm"
    elif "engineer" in text_lower or "software" in text_lower or "developer" in text_lower:
        role_lens = "engineer"
    elif "business" in text_lower or "strategy" in text_lower or "marketing" in text_lower:
        role_lens = "business"

    technical_depth = "medium"
    if any(x in text_lower for x in ["machine learning", "backend", "distributed", "python", "software engineer"]):
        technical_depth = "high"
    elif any(x in text_lower for x in ["non-technical", "beginner", "high-level only"]):
        technical_depth = "low"

    preferred_style = ["high_level"]
    if "step-by-step" in text_lower or "step by step" in text_lower:
        preferred_style = ["step_by_step"]
    if "analogy" in text_lower:
        preferred_style.append("analogy_driven")

    return {
        "current_role": role_lens,
        "role_lens": role_lens,
        "industry_domain": [],
        "technical_depth": technical_depth,
        "business_depth": "medium",
        "preferred_explanation_style": preferred_style,
        "jargon_tolerance": "medium",
        "strength_areas": [],
        "weak_areas": [],
        "current_projects": [],
        "short_reason": "Fallback profile generated from uploaded background text."
    }


def _parse_background_with_llm(raw_text: str) -> Dict:
    client = _get_openai_client()

    prompt = f"""
You are extracting a structured user profile from resume/background text.

Return ONLY valid JSON with exactly these keys:
- current_role
- role_lens
- industry_domain
- technical_depth
- business_depth
- preferred_explanation_style
- jargon_tolerance
- strength_areas
- weak_areas
- current_projects
- short_reason

Rules:
- role_lens must be one of: ["general", "pm", "engineer", "business"]
- technical_depth must be one of: ["low", "medium", "high"]
- business_depth must be one of: ["low", "medium", "high"]
- jargon_tolerance must be one of: ["low", "medium", "high"]
- industry_domain must be a list of strings
- preferred_explanation_style must be a list of strings
- strength_areas must be a list of strings
- weak_areas must be a list of strings
- current_projects must be a list of strings
- short_reason should be one sentence

Background text:
{raw_text[:6000]}
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
    return _parse_json_safely(content)


def _normalize_profile(profile: Dict) -> Dict:
    if not isinstance(profile, dict):
        raise ValueError("Parsed profile is not a dictionary.")

    valid_roles = {"general", "pm", "engineer", "business"}
    valid_levels = {"low", "medium", "high"}

    role_lens = profile.get("role_lens", "general")
    if role_lens not in valid_roles:
        role_lens = "general"

    technical_depth = profile.get("technical_depth", "medium")
    if technical_depth not in valid_levels:
        technical_depth = "medium"

    business_depth = profile.get("business_depth", "medium")
    if business_depth not in valid_levels:
        business_depth = "medium"

    jargon_tolerance = profile.get("jargon_tolerance", "medium")
    if jargon_tolerance not in valid_levels:
        jargon_tolerance = "medium"

    def ensure_list(x):
        return x if isinstance(x, list) else []

    return {
        "current_role": profile.get("current_role", role_lens),
        "role_lens": role_lens,
        "industry_domain": ensure_list(profile.get("industry_domain")),
        "technical_depth": technical_depth,
        "business_depth": business_depth,
        "preferred_explanation_style": ensure_list(profile.get("preferred_explanation_style")),
        "jargon_tolerance": jargon_tolerance,
        "strength_areas": ensure_list(profile.get("strength_areas")),
        "weak_areas": ensure_list(profile.get("weak_areas")),
        "current_projects": ensure_list(profile.get("current_projects")),
        "short_reason": profile.get("short_reason", "")
    }


def _build_background_chunks(user_id: str, raw_text: str, structured_profile: Dict) -> List[Dict]:
    chunks = []

    if structured_profile.get("role_lens"):
        chunks.append({
            "chunk_id": f"{user_id}_role_01",
            "chunk_type": "role_identity",
            "text": f"The user role lens is {structured_profile['role_lens']}."
        })

    if structured_profile.get("industry_domain"):
        chunks.append({
            "chunk_id": f"{user_id}_domain_01",
            "chunk_type": "domain_context",
            "text": "The user works in these domains: " + ", ".join(structured_profile["industry_domain"])
        })

    if structured_profile.get("technical_depth"):
        chunks.append({
            "chunk_id": f"{user_id}_tech_01",
            "chunk_type": "technical_exposure",
            "text": f"The user's technical depth is {structured_profile['technical_depth']}."
        })

    if structured_profile.get("weak_areas"):
        chunks.append({
            "chunk_id": f"{user_id}_boundary_01",
            "chunk_type": "knowledge_boundary",
            "text": "The user's weaker areas include: " + ", ".join(structured_profile["weak_areas"])
        })

    if structured_profile.get("preferred_explanation_style"):
        chunks.append({
            "chunk_id": f"{user_id}_pref_01",
            "chunk_type": "expression_preference",
            "text": "The user prefers explanations that are: " + ", ".join(structured_profile["preferred_explanation_style"])
        })

    if structured_profile.get("current_projects"):
        chunks.append({
            "chunk_id": f"{user_id}_project_01",
            "chunk_type": "current_project",
            "text": "The user's current projects include: " + ", ".join(structured_profile["current_projects"])
        })

    if not chunks and raw_text:
        chunks.append({
            "chunk_id": f"{user_id}_fallback_01",
            "chunk_type": "domain_context",
            "text": raw_text[:500]
        })

    return chunks


def onboard_user_background(user_id: str, raw_background_inputs: List[Dict]) -> Dict:
    _init_db()

    raw_text = _combine_raw_background(raw_background_inputs)

    try:
        parsed = _parse_background_with_llm(raw_text)
        structured_profile = _normalize_profile(parsed)
    except Exception:
        structured_profile = _fallback_profile(raw_text)

    chunks = _build_background_chunks(user_id, raw_text, structured_profile)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR REPLACE INTO user_profiles (user_id, structured_profile_json)
        VALUES (?, ?)
        """,
        (user_id, json.dumps(structured_profile, ensure_ascii=False))
    )

    cur.execute("DELETE FROM background_chunks WHERE user_id = ?", (user_id,))
    for chunk in chunks:
        cur.execute(
            """
            INSERT INTO background_chunks (user_id, chunk_type, chunk_text)
            VALUES (?, ?, ?)
            """,
            (user_id, chunk["chunk_type"], chunk["text"])
        )

    conn.commit()
    conn.close()

    return {
        "user_id": user_id,
        "structured_profile": structured_profile,
        "background_chunks": chunks
    }


def retrieve_user_background(user_id: str, query: str, recommended_chunk_types: List[str]) -> Dict:
    _init_db()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "SELECT structured_profile_json FROM user_profiles WHERE user_id = ?",
        (user_id,)
    )
    row = cur.fetchone()

    structured_profile = {}
    if row and row[0]:
        try:
            structured_profile = json.loads(row[0])
        except Exception:
            structured_profile = {}

    if recommended_chunk_types:
        placeholders = ",".join(["?"] * len(recommended_chunk_types))
        sql = f"""
            SELECT chunk_type, chunk_text
            FROM background_chunks
            WHERE user_id = ?
            AND chunk_type IN ({placeholders})
        """
        params = [user_id] + recommended_chunk_types
        cur.execute(sql, params)
    else:
        cur.execute(
            """
            SELECT chunk_type, chunk_text
            FROM background_chunks
            WHERE user_id = ?
            """,
            (user_id,)
        )

    rows = cur.fetchall()
    conn.close()

    retrieved_background_chunks = [
        {"chunk_type": chunk_type, "text": chunk_text}
        for chunk_type, chunk_text in rows
    ]

    return {
        "user_id": user_id,
        "structured_profile": structured_profile,
        "retrieved_background_chunks": retrieved_background_chunks
    }
