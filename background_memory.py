import os
import json
import sqlite3
from typing import Dict, List, Optional
from openai import OpenAI

DB_PATH = "background_memory.db"


# =========================================================
# DB helpers
# =========================================================

def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id TEXT PRIMARY KEY,
        current_role TEXT,
        role_lens TEXT,
        industry_domain TEXT,
        technical_depth TEXT,
        business_depth TEXT,
        preferred_explanation_style TEXT,
        jargon_tolerance TEXT,
        strength_areas TEXT,
        weak_areas TEXT,
        current_projects TEXT,
        short_reason TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS background_chunks (
        chunk_id TEXT PRIMARY KEY,
        user_id TEXT,
        chunk_type TEXT,
        text TEXT,
        FOREIGN KEY(user_id) REFERENCES user_profiles(user_id)
    )
    """)

    conn.commit()
    conn.close()


# =========================================================
# OpenAI helper
# =========================================================

def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


# =========================================================
# Step 1 + 2: Parse raw background into structured profile
# =========================================================

def parse_user_background(user_id: str, raw_background_inputs: List[Dict]) -> Dict:
    """
    Parse raw user background into a structured profile.

    raw_background_inputs example:
    [
        {"source_type": "resume", "raw_text": "..."},
        {"source_type": "self_intro", "raw_text": "..."}
    ]
    """
    combined_text = "\n\n".join(
        f"[{item.get('source_type', 'unknown')}]\n{item.get('raw_text', '')}"
        for item in raw_background_inputs
        if item.get("raw_text")
    ).strip()

    if not combined_text:
        raise ValueError("No background text provided.")

    client = _get_openai_client()

    prompt = f"""
You are extracting a structured user background profile for a personalized explanation agent.

Given the user's background text, return ONLY valid JSON with exactly these keys:

- user_id
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
- industry_domain should be a list of strings
- preferred_explanation_style should be a list of strings
- strength_areas should be a list of strings
- weak_areas should be a list of strings
- current_projects should be a list of strings
- short_reason should be a short explanation of the inferred profile

Set user_id to "{user_id}"

Background text:
{combined_text}
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
    profile = json.loads(content)

    valid_role_lens = {"general", "pm", "engineer", "business"}
    valid_depth = {"low", "medium", "high"}

    if profile.get("role_lens") not in valid_role_lens:
        profile["role_lens"] = "general"
    if profile.get("technical_depth") not in valid_depth:
        profile["technical_depth"] = "medium"
    if profile.get("business_depth") not in valid_depth:
        profile["business_depth"] = "medium"
    if profile.get("jargon_tolerance") not in valid_depth:
        profile["jargon_tolerance"] = "medium"

    for key in [
        "industry_domain",
        "preferred_explanation_style",
        "strength_areas",
        "weak_areas",
        "current_projects",
    ]:
        if not isinstance(profile.get(key), list):
            profile[key] = []

    if "short_reason" not in profile:
        profile["short_reason"] = ""

    return profile


# =========================================================
# Step 3: Chunk background
# =========================================================

def chunk_user_background(raw_background_inputs: List[Dict], structured_profile: Dict) -> List[Dict]:
    """
    Convert background/profile into typed chunks.
    """
    user_id = structured_profile["user_id"]
    chunks = []

    current_role = structured_profile.get("current_role", "")
    role_lens = structured_profile.get("role_lens", "")
    domains = structured_profile.get("industry_domain", [])
    technical_depth = structured_profile.get("technical_depth", "medium")
    business_depth = structured_profile.get("business_depth", "medium")
    preferences = structured_profile.get("preferred_explanation_style", [])
    jargon_tolerance = structured_profile.get("jargon_tolerance", "medium")
    strengths = structured_profile.get("strength_areas", [])
    weak_areas = structured_profile.get("weak_areas", [])
    projects = structured_profile.get("current_projects", [])

    if current_role or role_lens:
        chunks.append({
            "chunk_id": f"{user_id}_role_01",
            "user_id": user_id,
            "chunk_type": "role_identity",
            "text": f"The user currently works as {current_role} and should generally be addressed through a {role_lens} lens."
        })

    if domains:
        chunks.append({
            "chunk_id": f"{user_id}_domain_01",
            "user_id": user_id,
            "chunk_type": "domain_context",
            "text": f"The user's domain background includes: {', '.join(domains)}."
        })

    chunks.append({
        "chunk_id": f"{user_id}_technical_01",
        "user_id": user_id,
        "chunk_type": "technical_exposure",
        "text": f"The user's technical depth is {technical_depth}, business depth is {business_depth}, and jargon tolerance is {jargon_tolerance}."
    })

    if weak_areas:
        chunks.append({
            "chunk_id": f"{user_id}_boundary_01",
            "user_id": user_id,
            "chunk_type": "knowledge_boundary",
            "text": f"The user is less comfortable with: {', '.join(weak_areas)}."
        })

    if preferences:
        chunks.append({
            "chunk_id": f"{user_id}_pref_01",
            "user_id": user_id,
            "chunk_type": "expression_preference",
            "text": f"The user prefers explanations that are: {', '.join(preferences)}."
        })

    if strengths:
        chunks.append({
            "chunk_id": f"{user_id}_strength_01",
            "user_id": user_id,
            "chunk_type": "strength_area",
            "text": f"The user's strength areas include: {', '.join(strengths)}."
        })

    if projects:
        chunks.append({
            "chunk_id": f"{user_id}_project_01",
            "user_id": user_id,
            "chunk_type": "current_project",
            "text": f"The user is currently working on: {', '.join(projects)}."
        })

    raw_text = "\n".join(
        item.get("raw_text", "") for item in raw_background_inputs if item.get("raw_text")
    ).strip()

    if raw_text:
        chunks.append({
            "chunk_id": f"{user_id}_raw_01",
            "user_id": user_id,
            "chunk_type": "raw_background",
            "text": raw_text[:1500]
        })

    return chunks


# =========================================================
# Step 4: Store structured profile
# =========================================================

def store_profile(user_id: str, structured_profile: Dict) -> Dict:
    init_db()
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    INSERT OR REPLACE INTO user_profiles (
        user_id,
        current_role,
        role_lens,
        industry_domain,
        technical_depth,
        business_depth,
        preferred_explanation_style,
        jargon_tolerance,
        strength_areas,
        weak_areas,
        current_projects,
        short_reason
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        structured_profile.get("current_role", ""),
        structured_profile.get("role_lens", "general"),
        json.dumps(structured_profile.get("industry_domain", []), ensure_ascii=False),
        structured_profile.get("technical_depth", "medium"),
        structured_profile.get("business_depth", "medium"),
        json.dumps(structured_profile.get("preferred_explanation_style", []), ensure_ascii=False),
        structured_profile.get("jargon_tolerance", "medium"),
        json.dumps(structured_profile.get("strength_areas", []), ensure_ascii=False),
        json.dumps(structured_profile.get("weak_areas", []), ensure_ascii=False),
        json.dumps(structured_profile.get("current_projects", []), ensure_ascii=False),
        structured_profile.get("short_reason", "")
    ))

    conn.commit()
    conn.close()

    return {
        "user_id": user_id,
        "profile_status": "stored",
        "store_type": "sqlite"
    }


# =========================================================
# Step 5: Store background chunks
# =========================================================

def store_chunks(user_id: str, background_chunks: List[Dict]) -> List[Dict]:
    init_db()
    conn = get_conn()
    cur = conn.cursor()

    # remove old chunks for this user
    cur.execute("DELETE FROM background_chunks WHERE user_id = ?", (user_id,))

    for chunk in background_chunks:
        cur.execute("""
        INSERT OR REPLACE INTO background_chunks (
            chunk_id,
            user_id,
            chunk_type,
            text
        )
        VALUES (?, ?, ?, ?)
        """, (
            chunk["chunk_id"],
            chunk["user_id"],
            chunk["chunk_type"],
            chunk["text"]
        ))

    conn.commit()
    conn.close()

    return [
        {
            "chunk_id": chunk["chunk_id"],
            "storage_status": "stored"
        }
        for chunk in background_chunks
    ]


# =========================================================
# Full onboarding pipeline
# =========================================================

def onboard_user_background(user_id: str, raw_background_inputs: List[Dict]) -> Dict:
    init_db()

    structured_profile = parse_user_background(user_id, raw_background_inputs)
    background_chunks = chunk_user_background(raw_background_inputs, structured_profile)

    profile_store_result = store_profile(user_id, structured_profile)
    chunk_store_result = store_chunks(user_id, background_chunks)

    return {
        "structured_profile": structured_profile,
        "background_chunks": background_chunks,
        "profile_store_result": profile_store_result,
        "chunk_store_result": chunk_store_result
    }


# =========================================================
# Step 6: Retrieval at query time
# =========================================================

def _simple_text_score(query: str, text: str) -> int:
    q_terms = set(query.lower().split())
    t_terms = set(text.lower().split())
    return len(q_terms & t_terms)


def retrieve_user_background(
    user_id: str,
    query: str,
    recommended_chunk_types: Optional[List[str]] = None,
    top_k: int = 4
) -> Dict:
    init_db()
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT
        user_id,
        current_role,
        role_lens,
        industry_domain,
        technical_depth,
        business_depth,
        preferred_explanation_style,
        jargon_tolerance,
        strength_areas,
        weak_areas,
        current_projects,
        short_reason
    FROM user_profiles
    WHERE user_id = ?
    """, (user_id,))
    row = cur.fetchone()

    if row is None:
        conn.close()
        return {
            "user_id": user_id,
            "structured_profile": None,
            "retrieved_background_chunks": []
        }

    structured_profile = {
        "user_id": row[0],
        "current_role": row[1],
        "role_lens": row[2],
        "industry_domain": json.loads(row[3]) if row[3] else [],
        "technical_depth": row[4],
        "business_depth": row[5],
        "preferred_explanation_style": json.loads(row[6]) if row[6] else [],
        "jargon_tolerance": row[7],
        "strength_areas": json.loads(row[8]) if row[8] else [],
        "weak_areas": json.loads(row[9]) if row[9] else [],
        "current_projects": json.loads(row[10]) if row[10] else [],
        "short_reason": row[11],
    }

    if recommended_chunk_types:
        placeholders = ",".join("?" * len(recommended_chunk_types))
        sql = f"""
        SELECT chunk_id, user_id, chunk_type, text
        FROM background_chunks
        WHERE user_id = ?
        AND chunk_type IN ({placeholders})
        """
        params = [user_id] + recommended_chunk_types
        cur.execute(sql, params)
    else:
        cur.execute("""
        SELECT chunk_id, user_id, chunk_type, text
        FROM background_chunks
        WHERE user_id = ?
        """, (user_id,))

    rows = cur.fetchall()
    conn.close()

    chunks = [
        {
            "chunk_id": r[0],
            "user_id": r[1],
            "chunk_type": r[2],
            "text": r[3]
        }
        for r in rows
    ]

    scored = []
    for chunk in chunks:
        score = _simple_text_score(query, chunk["text"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for score, chunk in scored[:top_k]]

    return {
        "user_id": user_id,
        "structured_profile": {
            "role_lens": structured_profile.get("role_lens", "general"),
            "technical_depth": structured_profile.get("technical_depth", "medium"),
            "business_depth": structured_profile.get("business_depth", "medium"),
            "jargon_tolerance": structured_profile.get("jargon_tolerance", "medium"),
            "preferred_explanation_style": structured_profile.get("preferred_explanation_style", []),
            "short_reason": structured_profile.get("short_reason", "")
        },
        "retrieved_background_chunks": top_chunks
    }


# =========================================================
# Optional debug helpers
# =========================================================

def get_full_profile(user_id: str) -> Optional[Dict]:
    result = retrieve_user_background(user_id=user_id, query="", recommended_chunk_types=None, top_k=10)
    return result["structured_profile"]


def get_all_chunks(user_id: str) -> List[Dict]:
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    SELECT chunk_id, user_id, chunk_type, text
    FROM background_chunks
    WHERE user_id = ?
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "chunk_id": r[0],
            "user_id": r[1],
            "chunk_type": r[2],
            "text": r[3]
        }
        for r in rows
    ]
