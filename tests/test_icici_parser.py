import pandas as pd
from pathlib import Path
from custom_parsers.icici_parser import parse

def test_parse_matches_csv():
    csv_file = Path(r'data\icici\icici_sample.csv')
    pdf_file = Path(r'data\icici\icici_sample.pdf')
    expected = pd.read_csv(csv_file)
    result = parse(str(pdf_file))
    pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                  expected.reset_index(drop=True),
                                  check_dtype=False)
