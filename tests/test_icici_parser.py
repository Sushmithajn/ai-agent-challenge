import pandas as pd
from pathlib import Path
from custom_parsers.icici_parser import parse

def test_parse_returns_dataframe():
    sample_pdf = Path(r'C:\Users\sushmitha\ai-agent-challenge\data\icici\icici_sample.pdf')
    df = parse(str(sample_pdf))
    assert isinstance(df, pd.DataFrame)
