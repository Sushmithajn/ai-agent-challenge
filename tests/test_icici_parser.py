import pandas as pd
from pathlib import Path
from custom_parsers.icici_parser import parse

def test_parse_matches_csv():
    expected = pd.read_csv(Path(r'data\icici\icici_sample.csv'))
    result = parse(str(Path(r'data\icici\icici_sample.pdf')))
    result = result[expected.columns.tolist()]
    pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                  expected.reset_index(drop=True),
                                  check_dtype=False, check_exact=False)
