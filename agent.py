import os, subprocess
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_ATTEMPTS = 3
ROOT = Path(__file__).parent.resolve()
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in .env file")

client = Groq(api_key=GROQ_API_KEY)

# Safe fallback parser (clean and minimal)
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
def groq_generate_code(prompt:str, fallback:str)->str:
    try:
        code = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.1
        ).choices[0].message.content
        return code.replace("```python","").replace("```","").strip() or fallback
    except: 
        return fallback

def ensure_safe_parser(parser_path:Path, bank:str, pdf_path:Path):
    """Check that parse() returns a DataFrame; otherwise, fallback."""
    try:
        spec = __import__(f"custom_parsers.{bank}_parser", fromlist=["parse"])
        df = spec.parse(str(pdf_path))
        if not isinstance(df, pd.DataFrame): raise TypeError
    except Exception as e:
        print(f"[agent] Safety-net triggered ({e}), using fallback.")
        parser_path.write_text(SAFE_TEMPLATE, encoding="utf-8")

def ensure_test_file(bank:str, pdf_path:Path):
    """Create a pytest file if missing."""
    tests_dir = ROOT / "tests"; tests_dir.mkdir(exist_ok=True)
    test_file = tests_dir / f"test_{bank}_parser.py"
    if not test_file.exists():
        csv_path = ROOT / "data" / bank / f"{bank}_sample.csv"
        content = f"""import pandas as pd
from pathlib import Path
from custom_parsers.{bank}_parser import parse

def test_parse_matches_csv():
    expected = pd.read_csv(Path(r'{csv_path.relative_to(ROOT)}'))
    result = parse(str(Path(r'{pdf_path.relative_to(ROOT)}')))
    result = result[expected.columns.tolist()]
    pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                  expected.reset_index(drop=True),
                                  check_dtype=False, check_exact=False)
"""
        test_file.write_text(content, encoding="utf-8")
        
def run_pytest(bank:str):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')

    proc = subprocess.run(
        ["pytest", f"tests/test_{bank}_parser.py", "--full-trace", "-s"],
        capture_output=True, text=True, check=False,
        cwd=ROOT, # Ensure the command is run from the root directory
        env=env
    )
    return proc.returncode, proc.stdout + "\n" + proc.stderr

def plan_node(state:dict)->dict:
    bank = state["bank"]; attempt = state["attempt"]
    print(f"\n=== {bank.upper()} | Attempt {attempt} ===\n")
    data_dir = ROOT / "data" / bank
    csv_path = data_dir / f"{bank}_sample.csv"
    pdf_path = data_dir / f"{bank}_sample.pdf"

    headers = sample_rows = ""
    if csv_path.exists():
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        headers = lines[0]; sample_rows = "\n".join(lines[1:6])

    parser_dir = ROOT / "custom_parsers"; parser_dir.mkdir(exist_ok=True)
    parser_path = parser_dir / f"{bank}_parser.py"; (parser_dir / "__init__.py").touch(exist_ok=True)

    if attempt == 1:
        prompt = f"""
You are an expert Python developer.
Write a COMPLETE Python parser function:

def parse(pdf_path: str) -> pandas.DataFrame

Requirements:
- Use pdfplumber to read all tables from the PDF.
- Combine tables into a single pandas DataFrame.
- Column names must match CSV exactly: {headers}
- Sample expected rows:
{sample_rows}
- Filter blank rows or repeated headers.
- Return cleaned numeric columns where applicable.
- Provide a clean, ready-to-use Python file (no commentary outside the code).
"""
    else:
        prompt = f"""
Previous attempt failed.
Previous code:
{state.get('previous_code','')}
Pytest output:
{state.get('test_failure_output','')}
Provide complete corrected Python code ONLY, ready for execution.
"""
    code = groq_generate_code(prompt, SAFE_TEMPLATE)
    parser_path.write_text(code, encoding="utf-8")
    ensure_safe_parser(parser_path, bank, pdf_path)
    ensure_test_file(bank, pdf_path)

    state.update({"parser_path": parser_path, "pdf_path": pdf_path, "current_code": code})
    return state

def test_node(state:dict)->dict:
    rc, out = run_pytest(state["bank"])
    state.update({"pytest_rc": rc, "pytest_out": out})
    return state

def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--target', required=True)
    args = parser.parse_args()
    state = {"bank": args.target.lower(), "attempt": 1, "previous_code": "", "test_failure_output": ""}
    print(f"\n--- Starting Parser Agent for {state['bank'].upper()} ---\n")

    while state["attempt"] <= MAX_ATTEMPTS:
        state = plan_node(state)
        state = test_node(state)
        if state["pytest_rc"] == 0:
            print(f"\n✅ {state['bank'].upper()} | Attempt {state['attempt']} succeeded!\n")
            break
        else:
            print(f"\n❌ {state['bank'].upper()} | Attempt {state['attempt']} failed.\n")
            print(f"--- Error Summary (last 500 chars) ---\n{state['pytest_out'][-500:]}\n---------------------------")
            state["previous_code"] = state["current_code"]
            state["test_failure_output"] = state["pytest_out"]
            state["attempt"] += 1

    if state["attempt"] > MAX_ATTEMPTS:
        print(f"\n🚨 {state['bank'].upper()} | Max attempts reached. Parser failed.\n")

if __name__=="__main__":
    main()
