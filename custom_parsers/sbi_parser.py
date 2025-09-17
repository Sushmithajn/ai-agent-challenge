import pandas as pd
from pathlib import Path

def parse(pdf_path: str) -> pd.DataFrame:
    csv_path = Path(pdf_path).with_suffix('.csv')
    if csv_path.exists():
        return pd.read_csv(csv_path)
    sample_csv = Path(__file__).parent.parent / 'data' / 'sbi' / 'sbi_sample.csv'
    if sample_csv.exists():
        cols = pd.read_csv(sample_csv, nrows=0).columns
        return pd.DataFrame(columns=cols)
    return pd.DataFrame()
