from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, Query

app = FastAPI(title="QA Search API", version="1.0")

# ====== 起動時に一度だけロード（Cloud Run向け） ======
BASE_DIR = Path(__file__).resolve().parent

VEC_PATH = BASE_DIR / "tfidf_vectorizer.joblib"
MAT_PATH = BASE_DIR / "tfidf_matrix.npz"
META_PATH = BASE_DIR / "meta.csv"

# lazy load ではなく、起動時ロードで速度と安定性を優先
vectorizer = joblib.load(VEC_PATH)
X = sp.load_npz(MAT_PATH)
meta = pd.read_csv(META_PATH)

# meta.csv の必須カラム（最低限）
REQUIRED_COLS = {"question", "answer"}
missing = REQUIRED_COLS - set(meta.columns)
if missing:
    raise RuntimeError(f"meta.csv に必須カラムがありません: {sorted(missing)}")


@app.get("/")
def health() -> Dict[str, Any]:
    """稼働確認"""
    return {"status": "ok"}


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, description="検索クエリ"),
    top_k: int = Query(5, ge=1, le=50, description="上位K件"),
) -> Dict[str, Any]:
    """
    search_local.py と同じ:
    - vec.transform([query])
    - scores = (X @ qv.T).toarray().ravel()
    - top-k を argpartition + sort で取得
    """
    qv = vectorizer.transform([query])  # L2-normalized by vectorizer（元コードと同じ前提）
    scores = (X @ qv.T).toarray().ravel()

    k = min(top_k, len(scores))
    if k == 0:
        return {"query": query, "count": 0, "results": []}

    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results: List[Dict[str, Any]] = []
    for rank, i in enumerate(top_idx, start=1):
        row = meta.iloc[int(i)]
        results.append(
            {
                "rank": rank,
                "score": float(scores[int(i)]),
                "qa_id": row.get("qa_id", ""),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }
        )

    return {"query": query, "count": len(results), "results": results}


if __name__ == "__main__":
    # Cloud Run は PORT を環境変数で渡す
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
