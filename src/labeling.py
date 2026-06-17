import ast
import pandas as pd
from src.config import CATEGORIES


def pseudo_label(results: pd.DataFrame) -> list[dict]:
    total_score = results['score'].sum()
    output = []
    for cat in CATEGORIES:
        cat_score = sum(
            row['score']
            for _, row in results.iterrows()
            if cat in ast.literal_eval(row['labels'])
        )
        output.append({
            'label': cat,
            'confidence': cat_score / total_score if total_score > 0 else 0.0,
        })
    return sorted(output, key=lambda x: x['confidence'], reverse=True)
