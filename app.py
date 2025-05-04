import os
import re
import json
import pickle

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# ───────────────────────────────────────────────────────────
# 1. Configuration & Model/Index Loading
# ───────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your environment")

client = genai.Client(api_key=GEMINI_API_KEY)

sbert = SentenceTransformer("all-mpnet-base-v2")

def load_metadata(path="shl_meta.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

df_meta = load_metadata()

meta_lookup = {row["url"]: {"adaptive_support": row["adaptive_support"],
                                "description": row["description"],
                                "duration": int(row["duration"]),
                                "remote_support": row["remote_support"],
                                "test_type": row["test_type"]}
               for _, row in df_meta.iterrows()}

embeddings = np.load("shl_embeddings.npy").astype("float32")

# Few-shot examples
EXAMPLES = [
    {
        "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
        "candidates": [
            {"url": "https://.../java-8-new/",    "duration": 45, "remote": "Yes", "adaptive": "No",  "types": ["Ability & Aptitude"], "desc": "Test core Java knowledge at Java 8 level..."},
            {"url": "https://.../core-java-entry-level-new/", "duration": 40, "remote": "Yes", "adaptive": "Yes", "types": ["Ability & Aptitude"], "desc": "Assess fundamental Java syntax, OOP concepts..."},
            {"url": "https://.../java-frameworks-new/","duration": 50, "remote": "No",  "adaptive": "No",  "types": ["Knowledge & Skills"], "desc": "Evaluate understanding of Spring, Hibernate..."},
            {"url": "https://.../automata-fix-new/",  "duration": 35, "remote": "Yes", "adaptive": "No",  "types": ["Simulations"],        "desc": "Hands-on debugging of Java code snippets..."},
            {"url": "https://.../agile-software-development/", "duration": 60, "remote": "Yes", "adaptive": "No", "types": ["Knowledge & Skills"],  "desc": "Scenario-based questions on Agile methodology..."}
        ],
        "top": ["https://.../core-java-entry-level-new/", "https://.../java-8-new/", "https://.../automata-fix-new/"]
    },
    {
        "query": "I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options",
        "candidates": [
            {"url": "https://.../entry-level-sales-7-1/",       "duration": 55, "remote": "Yes", "adaptive": "No", "types": ["Sales & Service"], "desc": "Assess sales aptitude and customer-service style..."},
            {"url": "https://.../technology-professional-8-0-job-focused-assessment/", "duration": 60, "remote": "Yes", "adaptive": "No", "types": ["Ability & Aptitude"], "desc": "General tech & problem-solving skills..."},
            {"url": "https://.../entry-level-sales-sift-out-7-1/","duration": 50, "remote": "Yes", "adaptive": "Yes","types": ["Sales & Service"], "desc": "Adaptive questions on sales scenarios..."},
            {"url": "https://.../sales-and-service-phone-solution/","duration": 45, "remote": "No",  "adaptive": "No",  "types": ["Sales & Service"], "desc": "Simulated phone-sales exercise..."},
            {"url": "https://.../english-comprehension-new/",   "duration": 30, "remote": "Yes", "adaptive": "No",  "types": ["Ability & Aptitude"], "desc": "Reading and comprehension in business English..."}
        ],
        "top": ["https://.../entry-level-sales-sift-out-7-1/", "https://.../entry-level-sales-7-1/", "https://.../sales-and-service-phone-solution/"]
    }
]

# ───────────────────────────────────────────────────────────
# 2. FastAPI Setup
# ───────────────────────────────────────────────────────────

app = FastAPI(title="SHL Few-Shot Gemini API", version="1.1")

from fastapi.middleware.cors import CORSMiddleware

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    url: str
    adaptive_support: Literal["Yes", "No"]
    description: str
    duration: int
    remote_support: Literal["Yes", "No"]
    test_type: List[str]

class RecommendResponse(BaseModel):
    recommended_assessments: List[Assessment]

# ───────────────────────────────────────────────────────────
# 3. Helpers
# ───────────────────────────────────────────────────────────

def parse_filters(text: str):
    dur = None
    m = re.search(r"(?:under|<)\s*(\d{1,3})\s*minutes?", text, re.I)
    if m:
        dur = int(m.group(1))
    remote_req = "remote" in text.lower()
    adaptive_req = "adaptive" in text.lower() or "irt" in text.lower()
    return dur, remote_req, adaptive_req


def build_few_shot_prompt(q: str, candidates: List[dict]) -> str:
    lines = ["System: Rank candidates and return top 3 URLs only.\n"]
    for ex in EXAMPLES:
        lines.append(f"Query: “{ex['query']}”")
        lines.append("Candidates:")
        for c in ex['candidates']:
            lines.append(
                f"- {c['url']} | Duration: {c['duration']} mins, Remote: {c['remote']}, Adaptive: {c['adaptive']}, Types: {c['types']} | {c['desc']}"
            )
        lines.append("Top 3:")
        for url in ex['top']:
            lines.append(f"- {url}")
        lines.append("")

    # new query
    lines.append(f"Query: “{q}”")
    lines.append("Candidates:")
    for c in candidates[:5]:
        lines.append(
            f"- {c['url']} | Duration: {c['duration']} mins, Remote: {c['remote_support']}, Adaptive: {c['adaptive_support']}, Types: {c['test_type']} | {c['description'][:100]}..."
        )
    lines.append("Top 3:")
    return "\n".join(lines)

# ───────────────────────────────────────────────────────────
# 4. Endpoints
# ───────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(400, "Query text is required")

    dur_cap, need_remote, need_adaptive = parse_filters(q)

    # Phase-1: pre-filter
    filtered = [i for i,row in df_meta.iterrows()
                if meta_lookup[row['url']]['duration'] <= (dur_cap or float('inf'))]
    if not filtered:
        filtered = list(range(len(df_meta)))
    sub_vecs = embeddings[filtered]
    sub_index = faiss.IndexFlatL2(sub_vecs.shape[1])
    sub_index.add(sub_vecs)

    q_emb = sbert.encode([q], convert_to_numpy=True)
    _, I = sub_index.search(q_emb, k=540)
    cand_idxs = [filtered[i] for i in I[0]]
    candidates = [{**meta_lookup[df_meta.loc[i,'url']], 'url': df_meta.loc[i,'url']} for i in cand_idxs]

    # Phase-2: few-shot LLM rerank
    prompt = build_few_shot_prompt(q, candidates)
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=256)
    )
    lines = resp.text.strip().splitlines()
    ranked = [re.search(r"https?://[^\s,]+", line).group(0)
              for line in lines if re.search(r"https?://[^\s,]+", line)]

    # Final filter & collect top 10
    final = []
    for url in ranked:
        m = meta_lookup[url]
        if dur_cap and m['duration']>dur_cap: continue
        if need_remote and m['remote_support']!='Yes': continue
        if need_adaptive and m['adaptive_support']!='Yes': continue
        final.append({'url': url, **m})
        if len(final)==10: break

    if len(final)<10:
        for url in [c['url'] for c in candidates]:
            if any(f['url']==url for f in final): continue
            final.append({'url': url, **meta_lookup[url]})
            if len(final)==10: break

    return {"recommended_assessments": final}