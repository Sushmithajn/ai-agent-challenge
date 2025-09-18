import pdfplumber
import pandas as pd
from pathlib import Path

def parse(pdf_path: str) -> pd.DataFrame:
    """Extracts first table from the PDF into a DataFrame."""
    with pdfplumber.open(pdf_path) as pdf:
        all_tables = []
        for page in pdf.pages:
            tables = page.extract_tables()
            for t in tables:
                all_tables.extend(t)
        if not all_tables:
            return pd.DataFrame()
        # assume first row is header
        headers = all_tables[0]
        rows = all_tables[1:]
        return pd.DataFrame(rows, columns=headers)

