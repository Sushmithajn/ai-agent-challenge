import pdfplumber
import pandas as pd
from pathlib import Path
import numpy as np

def parse(pdf_path: str) -> pd.DataFrame:
    """Extract tables from PDF into DataFrame, normalize blanks to NaN, and convert numeric columns."""
    with pdfplumber.open(pdf_path) as pdf:
        all_tables = []
        for page in pdf.pages:
            tables = page.extract_tables()
            for t in tables:
                all_tables.extend(t)

        if not all_tables:
            return pd.DataFrame()

        headers = [h.strip() for h in all_tables[0]]
        rows = []
        for row in all_tables[1:]:
            if not any(cell and cell.strip() for cell in row):
                continue
            if [c.strip() for c in row] == headers:
                continue
            clean = [c.strip() if c and c.strip() else np.nan for c in row]
            rows.append(clean)

        df = pd.DataFrame(rows, columns=headers)

        # Auto-convert numeric columns using modern approach
        # Auto-convert numeric columns safely
        for col in df.columns:
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().any():
                    df[col] = converted
            except Exception:
                continue


        return df
