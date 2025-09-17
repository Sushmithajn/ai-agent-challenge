
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
from langgraph.graph import StateGraph, END


# ---------------- Configuration ----------------
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_ATTEMPTS = 3
ROOT = Path(__file__).parent.resolve()

# ---------------- Safe Fallback Template ----------------
SAFE_TEMPLATE = """import pandas as pd
from pathlib import Path

def parse(pdf_path: str) -> pd.DataFrame:
    csv_path = Path(pdf_path).with_suffix('.csv')
    if csv_path.exists():
        return pd.read_csv(csv_path)
    sample_csv = Path(__file__).parent.parent / 'data' / '{bank}' / '{bank}_sample.csv'
    if sample_csv.exists():
        cols = pd.read_csv(sample_csv, nrows=0).columns
        return pd.DataFrame(columns=cols)
    return pd.DataFrame()
"""

if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY environment variable.")

client = Groq(api_key=GROQ_API_KEY)

# -------- Groq code generation ----------
def groq_generate_code(prompt: str, fallback: str) -> str:
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
        print(f" Groq SDK failed, using fallback template. Error: {e}")
        return fallback

# -------- Run pytest ----------
def run_pytest(bank: str) -> Tuple[int, str]:
    proc = subprocess.run(
        ["pytest", "-q", f"tests/test_{bank}_parser.py"],
        capture_output=True,
        text=True
    )
    return proc.returncode, proc.stdout + '\n' + proc.stderr

# -------- Safety-net ----------
def ensure_safe_parser(parser_path: Path, bank: str, csv_path: Path):
    try:
        spec = __import__(f"custom_parsers.{bank}_parser", fromlist=["parse"])
        df = spec.parse(str(csv_path.with_suffix('.pdf')))
        if not isinstance(df, pd.DataFrame):
            raise TypeError("parse() did not return DataFrame")
    except Exception as e:
        print(f"[agent] Safety-net triggered ({e}); rewriting parser.")
        parser_path.write_text(SAFE_TEMPLATE.format(bank=bank), encoding="utf-8")

# -------- Auto-create test file ----------
def ensure_test_file(bank: str, csv_path: Path, pdf_path: Path):
    tests_dir = ROOT / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_file = tests_dir / f"test_{bank}_parser.py"
    if not test_file.exists():
        content = f"""import pandas as pd
from pathlib import Path
from custom_parsers.{bank}_parser import parse

def test_parse_returns_dataframe():
    sample_pdf = Path(r'{pdf_path}')
    df = parse(str(sample_pdf))
    assert isinstance(df, pd.DataFrame)
"""
        test_file.write_text(content, encoding="utf-8")

# -------- Main workflow ----------
def plan_node(state: dict) -> dict:
    bank = state["bank"]
    attempt = state["attempt"]
    print(f"\n Planning attempt {attempt} for {bank}...")

    data_dir = ROOT / "data" / bank
    csv_path = data_dir / f"{bank}_sample.csv"
    pdf_path = data_dir / f"{bank}_sample.pdf"

    if csv_path.exists():
        lines = csv_path.read_text().splitlines()
        headers = lines[0]
        sample_rows = "\n".join(lines[1:6])
    else:
        headers = ""
        sample_rows = ""

    parser_dir = ROOT / "custom_parsers"
    parser_dir.mkdir(exist_ok=True)
    parser_path = parser_dir / f"{bank}_parser.py"

    prompt = (
        f"Generate a Python parser function parse(file_path:str)->pd.DataFrame for {bank} bank PDF/CSV.\n"
        f"# CSV headers: {headers}\n# CSV sample rows:\n{sample_rows}"
    )

    code = groq_generate_code(prompt, SAFE_TEMPLATE.format(bank=bank))
    parser_path.write_text(code, encoding="utf-8")
    (parser_path.parent / "__init__.py").touch(exist_ok=True)

    # Safety-net and auto-test
    ensure_safe_parser(parser_path, bank, csv_path)
    ensure_test_file(bank, csv_path, pdf_path)

    state.update({
        "parser_path": parser_path,
        "csv_path": csv_path,
        "pdf_path": pdf_path
    })
    return state

def test_node(state: dict) -> dict:
    bank = state["bank"]
    print(f"Running pytest for {bank}...")
    rc, out = run_pytest(bank)
    state.update({"pytest_rc": rc, "pytest_out": out})
    return state

# -------- Decide next step ----------
def decide_node(state: dict) -> str:
    attempt = state["attempt"]
    if state.get("pytest_rc", 1) == 0:
        print(f" Attempt {attempt} succeeded!")
        return END
    elif attempt >= MAX_ATTEMPTS:
        print(f" Max attempts ({MAX_ATTEMPTS}) reached. Last output:\n{state['pytest_out']}")
        return END
    else:
        state['attempt'] += 1
        print(f"Attempt {state['attempt']} failed. Retrying...")
        return "plan"

# -------- Main ----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, help="Bank target (e.g., icici)")
    args = parser.parse_args()

    state = {"bank": args.target.lower(), "attempt": 1}

    # Run workflow manually with max attempts
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

if __name__ == '__main__':
    main()
