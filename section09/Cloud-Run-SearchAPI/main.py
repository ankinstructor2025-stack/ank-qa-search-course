from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware


# =========================
# 初期設定
# =========================

app = FastAPI(
    title="QA Search API",
    version="1.0",
    description="TF-IDF ベースの QA 検索 API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

VEC_PATH = BASE_DIR / "tfidf_vectorizer.joblib"
MAT_PATH = BASE_DIR / "tfidf_matrix.npz"
META_PATH = BASE_DIR / "meta.csv"


# =========================
# データロード（起動時）
# =========================

vectorizer = joblib.load(VEC_PATH)
X = sp.load_npz(MAT_PATH)
meta = pd.read_csv(META_PATH)


# =========================
# ヘルスチェック
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# 検索 API
# =========================

@app.get("/search")
def search(
    query: str = Query(..., min_length=1, description="検索クエリ"),
    top_k: int = Query(5, ge=1, le=50, description="上位K件"),
):
    # クエリをベクトル化
    q_vec = vectorizer.transform([query])

    # 類似度計算（内積）
    scores = (X @ q_vec.T).toarray().ravel()

    # top-k 抽出
    if top_k >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results: List[Dict[str, Any]] = []

    for rank, idx in enumerate(top_indices, start=1):
        row = meta.iloc[int(idx)]

        # numpy / pandas 型をすべて Python 標準型に変換
        qa_id_val = row.get("qa_id", "")
        qa_id = "" if pd.isna(qa_id_val) else str(qa_id_val)

        results.append({
            "rank": int(rank),
            "score": float(scores[int(idx)]),
            "qa_id": qa_id,
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
        })

    return {
        "query": query,
        "count": int(len(results)),
        "results": results,
    }
