import argparse
from pathlib import Path
import pandas as pd
import scipy.sparse as sp
import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index_dir', default=str(Path(__file__).resolve().parent))
    ap.add_argument('--query', required=True)
    ap.add_argument('--top_k', type=int, default=5)
    args = ap.parse_args()

    idx = Path(args.index_dir)
    vec = joblib.load(idx / 'tfidf_vectorizer.joblib')
    X = sp.load_npz(idx / 'tfidf_matrix.npz')
    meta = pd.read_csv(idx / 'meta.csv')

    qv = vec.transform([args.query])  # L2-normalized by vectorizer
    # cosine similarity for L2-normalized vectors == dot product
    scores = (X @ qv.T).toarray().ravel()

    top_k = min(args.top_k, len(scores))
    top_idx = np.argpartition(-scores, top_k-1)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    for rank, i in enumerate(top_idx, start=1):
        row = meta.iloc[i]
        print(f'#{rank} score={scores[i]:.4f} qa_id={row.get("qa_id", "")}')
        print('Q:', row['question'])
        print('A:', row['answer'])
        print('-' * 60)


if __name__ == '__main__':
    main()
