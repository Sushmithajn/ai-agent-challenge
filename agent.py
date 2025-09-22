"""
Bank Statement Parser Agent
Plan → generate parser → run pytest → self-fix (≤3 attempts)
"""

import os
import subprocess
from pathlib import Path
from typing import Tuple
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_ATTEMPTS = 3
ROOT = Path(__file__).parent.resolve()

if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in .env file")

client = Groq(api_key=GROQ_API_KEY)

# --- pdfplumber fallback template with blank/header filtering ---
SAFE_TEMPLATE = """import pdfplumber
import pandas as pd
from pathlib import Path
import numpy as np

def parse(pdf_path: str) -> pd.DataFrame:
    \"\"\"Extract tables from PDF into DataFrame, normalize blanks to NaN, and convert numeric columns.\"\"\"
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
"""

def groq_generate_code(prompt: str, fallback: str) -> str:
    """Ask Groq model to generate parser code. Fall back to SAFE_TEMPLATE on error."""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        code = resp.choices[0].message.content
        code = code.replace("```python", "").replace("```", "").strip()
        return code or fallback
    except Exception as e:
        print(f"Groq call failed: {e}. Using fallback parser.")
        return fallback

def run_pytest(bank: str) -> Tuple[int, str]:
    """Run pytest and stream output to console in real time."""
    proc = subprocess.run(
        ["pytest", "-q", f"tests/test_{bank}_parser.py"],
        text=True
    )
    return proc.returncode, ""

def ensure_safe_parser(parser_path: Path, bank: str, pdf_path: Path):
    """Ensure parse() returns DataFrame, else replace with SAFE_TEMPLATE."""
    try:
        spec = __import__(f"custom_parsers.{bank}_parser", fromlist=["parse"])
        df = spec.parse(str(pdf_path))
        if not isinstance(df, pd.DataFrame):
            raise TypeError("parse() did not return DataFrame")
    except Exception as e:
        print(f"[agent] Safety-net triggered ({e}); rewriting parser.")
        parser_path.write_text(SAFE_TEMPLATE, encoding="utf-8")

def ensure_test_file(bank: str, pdf_path: Path):
    tests_dir = ROOT / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_file = tests_dir / f"test_{bank}_parser.py"
    if not test_file.exists():
        csv_path = ROOT / "data" / bank / f"{bank}_sample.csv"
        rel_csv = csv_path.relative_to(ROOT)
        rel_pdf = pdf_path.relative_to(ROOT)
        content = f"""import pandas as pd
from pathlib import Path
from custom_parsers.{bank}_parser import parse

def test_parse_matches_csv():
    csv_file = Path(r'{rel_csv}')
    pdf_file = Path(r'{rel_pdf}')
    expected = pd.read_csv(csv_file)
    result = parse(str(pdf_file))
    pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                  expected.reset_index(drop=True),
                                  check_dtype=False)
"""
        test_file.write_text(content, encoding="utf-8")

def plan_node(state: dict) -> dict:
    bank = state["bank"]
    attempt = state["attempt"]
    print(f"\nPlanning attempt {attempt} for {bank}...")

    data_dir = ROOT / "data" / bank
    csv_path = data_dir / f"{bank}_sample.csv"
    pdf_path = data_dir / f"{bank}_sample.pdf"

    headers = sample_rows = ""
    if csv_path.exists():
        lines = csv_path.read_text().splitlines()
        headers = lines[0]
        sample_rows = "\n".join(lines[1:6])

    parser_dir = ROOT / "custom_parsers"
    parser_dir.mkdir(exist_ok=True)
    parser_path = parser_dir / f"{bank}_parser.py"
    (parser_dir / "__init__.py").touch(exist_ok=True)

    prompt = f"""
Write a Python function:

def parse(pdf_path: str) -> pandas.DataFrame

Requirements:
* Use pdfplumber to read all tables from the PDF.
* Combine them into a single pandas DataFrame.
* The CSV file at data/{bank}/{bank}_sample.csv contains the expected columns:
  {headers}
* Ensure the DataFrame has **exactly the same column names and order as that CSV**.
* Provide complete working code with imports.
* Filter out blank rows or repeated headers so row counts match the CSV.
* Do not use placeholders or dummy returns.
"""
    code = groq_generate_code(prompt, SAFE_TEMPLATE)
    parser_path.write_text(code, encoding="utf-8")

    ensure_safe_parser(parser_path, bank, pdf_path)
    ensure_test_file(bank, pdf_path)

    state.update({"parser_path": parser_path, "pdf_path": pdf_path})
    return state

def test_node(state: dict) -> dict:
    bank = state["bank"]
    print(f"Running pytest for {bank}...")
    rc, out = run_pytest(bank)
    state.update({"pytest_rc": rc, "pytest_out": out})
    return state

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, help="Bank target (e.g., icici)")
    args = parser.parse_args()

    state = {"bank": args.target.lower(), "attempt": 1}
    while state["attempt"] <= MAX_ATTEMPTS:
        state = plan_node(state)
        state = test_node(state)
        if state.get("pytest_rc", 1) == 0:
            print(f"Attempt {state['attempt']} succeeded!")
            break
        else:
            print(f"Attempt {state['attempt']} failed.")
            state["attempt"] += 1

    if state["attempt"] > MAX_ATTEMPTS:
        print(f"Max attempts ({MAX_ATTEMPTS}) reached. Last output:\n{state.get('pytest_out')}")

if __name__ == "__main__":
    main()
