#!/usr/bin/env python3
"""
verity.py — Unified CLI for Verity Fake News Detection
=======================================================
Cross-platform (Linux/Windows/macOS) unified command-line interface
for all Verity operations: data, training, evaluation, and pipeline.

Usage:
    python verity.py <command> [options]

Examples:
    python verity.py data --full
    python verity.py train phobert --variant features --epochs 5
    python verity.py eval --real --samples 50
    python verity.py analyze "Tin giả về vaccine..."
    python verity.py ui
    python verity.py test
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ─── Bootstrap ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
# Ensure project modules are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── ANSI Colors (cross-platform) ──────────────────────────────────────────────

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        _ENABLE_VIRTUAL_TERMINAL = 0x0004
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), _ENABLE_VIRTUAL_TERMINAL
        )
        _USE_ANSI = True
    except Exception:
        _USE_ANSI = False
else:
    _USE_ANSI = True

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"

def c(col: str, text: str) -> str:
    """Colorize text if ANSI is available."""
    if _USE_ANSI:
        return f"{col}{text}{RESET}"
    return text

def run_command(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run command with uv run, cross-platform safe."""
    result = subprocess.run(cmd, **kwargs)
    return result

def shell(cmd: str, cwd: Path | None = None, check: bool = True) -> int:
    """Run shell command (uv run prefix added automatically for Python scripts)."""
    env = os.environ.copy()
    if cmd.startswith("uv run "):
        full_cmd = cmd.split()
    else:
        full_cmd = ["uv", "run"] + cmd.split()

    result = subprocess.run(
        full_cmd,
        cwd=cwd or PROJECT_ROOT,
        env=env,
    )
    if check and result.returncode != 0:
        print(c(RED, f"[ERROR] Command failed: {cmd}"))
        sys.exit(result.returncode)
    return result.returncode

# ─── Subparsers ────────────────────────────────────────────────────────────────

def cmd_data(args: argparse.Namespace) -> None:
    """Data pipeline commands."""
    if args.data_cmd == "list":
        _list_data()

    elif args.data_cmd == "download":
        print(c(CYAN, ">> Downloading ViFactCheck dataset..."))
        shell(f"python dataset/manager.py --download")
        print(c(GREEN, "   Done!"))

    elif args.data_cmd == "preprocess":
        print(c(CYAN, ">> Preprocessing Vietnamese text..."))
        shell(f"python dataset/manager.py --preprocess")
        print(c(GREEN, "   Done!"))

    elif args.data_cmd == "full":
        print(c(CYAN, ">> Running full data pipeline..."))
        shell(f"python dataset/manager.py --full")
        print(c(GREEN, "   Done!"))
    else:
        print(c(RED, f"Unknown data command: {args.data_cmd}"))

def cmd_train(args: argparse.Namespace) -> None:
    """Training commands."""
    if args.train_cmd == "tfidf":
        print(c(CYAN, ">> Training TF-IDF + Logistic Regression baseline..."))
        shell("python model/baseline_logreg.py")

    elif args.train_cmd == "phobert":
        cmd_parts = [
            "python", "model/train_phobert.py",
            "--variant", args.variant or "baseline",
        ]
        if args.epochs:
            cmd_parts.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            cmd_parts.extend(["--batch-size", str(args.batch_size)])
        if args.lr:
            cmd_parts.extend(["--lr", str(args.lr)])
        if args.max_seq_len:
            cmd_parts.extend(["--max-seq-len", str(args.max_seq_len)])
        if args.seed:
            cmd_parts.extend(["--seed", str(args.seed)])
        if args.lr_scheduler:
            cmd_parts.extend(["--lr-scheduler-type", args.lr_scheduler])
        if args.weight_decay:
            cmd_parts.extend(["--weight-decay", str(args.weight_decay)])
        if args.warmup_ratio:
            cmd_parts.extend(["--warmup-ratio", str(args.warmup_ratio)])
        if args.dropout:
            cmd_parts.extend(["--dropout", str(args.dropout)])
        if args.label_smoothing:
            cmd_parts.extend(["--label-smoothing-factor", str(args.label_smoothing)])
        if args.model_name:
            cmd_parts.extend(["--model-name", args.model_name])

        print(c(CYAN, f">> Training PhoBERT ({args.variant or 'baseline'})..."))
        cmd_str = " ".join(cmd_parts[1:])  # skip "python"
        shell(f"python {' '.join(cmd_parts[1:])}")

    elif args.train_cmd == "list":
        _list_models()
    else:
        print(c(RED, f"Unknown train command: {args.train_cmd}"))

def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluation commands."""
    if args.eval_cmd == "models":
        cmd_parts = ["python", "evaluation/evaluate_models.py"]
        if args.real:
            cmd_parts.append("--real")
        if args.samples:
            cmd_parts.extend(["--samples", str(args.samples)])
        print(c(CYAN, f">> Evaluating models (real={args.real}, samples={args.samples or 20})..."))
        shell(" ".join(cmd_parts[1:]))

    elif args.eval_cmd == "ablation":
        print(c(CYAN, ">> Running ablation study..."))
        shell("python evaluation/ablation_study.py")

    elif args.eval_cmd == "confusion":
        print(c(CYAN, ">> Generating confusion matrix..."))
        shell("python evaluation/confusion_matrix.py")

    elif args.eval_cmd == "all":
        print(c(CYAN, ">> Running all evaluations..."))
        shell("python evaluation/evaluate_models.py --real" if args.real else
              "python evaluation/evaluate_models.py")
        shell("python evaluation/ablation_study.py")

def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze single text or URL."""
    if args.text:
        text = args.text
        print(c(CYAN, f">> Analyzing text: {text[:80]}{'...' if len(text) > 80 else ''}"))
    elif args.url:
        text = args.url
        print(c(CYAN, f">> Analyzing URL: {text}"))
    else:
        print(c(RED, "[ERROR] Provide --text or --url"))
        sys.exit(1)

    mock = not args.real
    provider = args.provider or os.getenv("LLM_PROVIDER", "nvidia")
    print(c(DIM, f"   Mode: {'REAL' if not mock else 'MOCK'} | Provider: {provider}"))

    try:
        from sequential_adversarial.pipeline import SequentialAdversarialPipeline
        pipeline = SequentialAdversarialPipeline(mock=mock)
        result = pipeline.run(text)

        # Print verdict
        if result.verity_report:
            verdict = result.verity_report.conclusion
            conf = result.verity_report.confidence or 0.5
            label_color = GREEN if verdict in ("True", "Mostly True") else RED
            print()
            print(c(BOLD, "=" * 60))
            print(c(label_color, f"  VERDICT: {verdict.upper()} (confidence: {conf:.1%})"))
            print(c(BOLD, "=" * 60))
            # Safe access to optional fields
            summary = getattr(result.verity_report, "summary", None) or getattr(result.verity_report, "description", "")
            if summary:
                print()
                print(c(CYAN, "Summary:"))
                print(f"  {summary[:300]}...")
            if result.claims:
                print()
                print(c(CYAN, f"Claims detected ({len(result.claims)}):"))
                for i, claim in enumerate(result.claims[:3], 1):
                    claim_text = getattr(claim, "claim_text", None) or getattr(claim, "text", str(claim))
                    print(f"  {i}. {claim_text[:100]}...")
        else:
            print(c(YELLOW, "   [WARN] No verity report in result"))

    except Exception as e:
        print(c(RED, f"[ERROR] {e}"))
        sys.exit(1)

def cmd_ui(args: argparse.Namespace) -> None:
    """Launch Streamlit UI."""
    print(c(CYAN, ">> Launching Streamlit UI..."))
    print(c(DIM, "   Open http://localhost:8501 in your browser"))
    shell("streamlit run ui/app.py", check=False)

def cmd_test(args: argparse.Namespace) -> None:
    """Run tests."""
    if args.test_cmd == "all":
        print(c(CYAN, ">> Running all tests..."))
        shell("pytest tests/ -v --tb=short")

    elif args.test_cmd == "unit":
        print(c(CYAN, ">> Running unit tests..."))
        shell("pytest tests/ -v -m unit --tb=short")

    elif args.test_cmd == "cov":
        print(c(CYAN, ">> Running tests with coverage..."))
        shell("pytest tests/ -v --cov=. --cov-report=html --cov-report=term")

    elif args.test_cmd == "lint":
        print(c(CYAN, ">> Running linter..."))
        shell("ruff check .")

    elif args.test_cmd == "format":
        print(c(CYAN, ">> Formatting code..."))
        shell("ruff format .")

    elif args.test_cmd == "types":
        print(c(CYAN, ">> Running type checker..."))
        shell("mypy . --ignore-missing-imports")

def cmd_db(args: argparse.Namespace) -> None:
    """Database operations."""
    db_path = PROJECT_ROOT / "sequential_adversarial" / "data" / "verity_reports.db"

    if args.db_cmd == "init":
        print(c(CYAN, ">> Initializing database..."))
        import sqlite3
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                conclusion TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                manipulation_score REAL NOT NULL DEFAULT 0.0,
                full_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
            conn.commit()
        print(c(GREEN, "   Done!"))

    elif args.db_cmd == "stats":
        if not db_path.exists():
            print(c(YELLOW, "   [WARN] Database not found. Run 'verity db init' first."))
            return
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM reports")
        total = cur.fetchone()[0]
        cur.execute("SELECT conclusion, COUNT(*) FROM reports GROUP BY conclusion")
        conclusions = dict(cur.fetchall())
        cur.execute("SELECT AVG(confidence), MIN(confidence), MAX(confidence) FROM reports")
        avg_conf, min_conf, max_conf = cur.fetchone()
        conn.close()
        print()
        print(c(BOLD, "  Database Statistics"))
        print(c(BOLD, "  ─" + "─" * 40))
        print(f"  Total reports:  {total}")
        print(f"  Avg confidence: {avg_conf:.2f}" if avg_conf else "  Avg confidence: N/A")
        print(f"  Confidence range: {min_conf:.2f} - {max_conf:.2f}" if min_conf else "")
        if conclusions:
            print()
            print(c(DIM, "  By Conclusion:"))
            for conclusion, count in sorted(conclusions.items(), key=lambda x: -x[1]):
                print(f"    {conclusion:<20} {count:>5} ({count*100//max(total,1)}%)")
        print()

    elif args.db_cmd == "export":
        output = Path(args.output) if args.output else PROJECT_ROOT / "reports_export.json"
        if not db_path.exists():
            print(c(RED, "   [ERROR] Database not found"))
            return
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT * FROM reports")
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        conn.close()
        data = [dict(zip(cols, row)) for row in rows]
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(c(GREEN, f"   Exported {len(data)} reports to {output}"))

def cmd_collect(args: argparse.Namespace) -> None:
    """Collect news from Vietnamese sources."""
    print(c(CYAN, ">> Collecting news articles..."))
    shell("python dataset/collect_news.py")

def cmd_clean(args: argparse.Namespace) -> None:
    """Clean cache and artifacts."""
    targets = [
        PROJECT_ROOT / "__pycache__",
        PROJECT_ROOT / ".pytest_cache",
        PROJECT_ROOT / "htmlcov",
        PROJECT_ROOT / ".ruff_cache",
        PROJECT_ROOT / "saved_models",
        PROJECT_ROOT / "visuals",
    ]
    # Find and clean all __pycache__ dirs
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        targets.append(pycache)
    for pyc in PROJECT_ROOT.rglob("*.pyc"):
        targets.append(pyc)

    removed = 0
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t, ignore_errors=True)
            removed += 1
        elif t.is_file():
            t.unlink(missing_ok=True)
            removed += 1

    print(c(GREEN, f"   Cleaned {removed} items"))

# ─── Helper Commands ───────────────────────────────────────────────────────────

def _list_models() -> None:
    """List available trained models."""
    models_dir = PROJECT_ROOT / "saved_models"
    print()
    print(c(BOLD, "  Available Models"))
    print(c(BOLD, "  ─" + "─" * 35))
    if not models_dir.exists():
        print(c(DIM, "  No models found. Run training first."))
    else:
        for f in sorted(models_dir.glob("*.pt")):
            size = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:<35} {size:.1f} MB")
        for f in sorted(models_dir.glob("*.pkl")):
            size = f.stat().st_size / 1024
            print(f"  {f.name:<35} {size:.1f} KB")
    print()

def _list_data() -> None:
    """List available datasets."""
    from config import DATASET_PROCESSED_DIR, VIFACTCHECK_DIR
    print()
    print(c(BOLD, "  Available Datasets"))
    print(c(BOLD, "  ─" + "─" * 35))

    for name, path in [
        ("Raw (ViFactCheck)", VIFACTCHECK_DIR),
        ("Processed", DATASET_PROCESSED_DIR),
    ]:
        if path.exists():
            files = list(path.glob("*.csv"))
            total = sum(f.stat().st_size for f in files)
            print(f"  {name}:")
            for f in files[:8]:
                rows = sum(1 for _ in open(f, encoding="utf-8", errors="ignore")) - 1
                print(f"    {f.name:<30} {rows:>6} rows")
            if len(files) > 8:
                print(f"    ... and {len(files) - 8} more files")
            print(f"    Total: {total / (1024*1024):.1f} MB")
    print()

# ─── Main CLI ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verity",
        description=c(BOLD, "Verity") + " — Vietnamese Fake News Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  verity data --full                       Run full data pipeline
  verity train phobert --variant features  Train PhoBERT with features
  verity eval models --real --samples 50   Evaluate with real LLM
  verity analyze --text "Tin giả..."       Analyze single text
  verity ui                                Launch Streamlit UI
  verity test all                          Run all tests
  verity db stats                          Show database statistics
  verity info                              Show system information
        """,
    )
    parser.add_argument("--version", action="version", version="verity 0.2.0")

    subparsers = parser.add_subparsers(dest="command", title="commands", metavar="<command>")

    # ── data ─────────────────────────────────────────────────────────────────
    p_data = subparsers.add_parser("data", help="Data pipeline operations")
    p_data.add_argument("data_cmd", choices=["download", "preprocess", "full", "list"],
                        nargs="?", default="list")
    p_data.set_defaults(func=cmd_data)

    # ── train ───────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Model training")
    p_train.add_argument("train_cmd", choices=["tfidf", "phobert", "list"],
                         nargs="?", default="list")
    p_train.add_argument("--variant", choices=["baseline", "features"],
                         help="PhoBERT variant (default: baseline)")
    p_train.add_argument("--epochs", type=int, help="Number of training epochs")
    p_train.add_argument("--batch-size", type=int, help="Training batch size")
    p_train.add_argument("--lr", type=float, help="Learning rate (e.g. 2e-5)")
    p_train.add_argument("--max-seq-len", type=int, help="Max sequence length")
    p_train.add_argument("--seed", type=int, help="Random seed")
    p_train.add_argument("--lr-scheduler", help="LR scheduler type")
    p_train.add_argument("--weight-decay", type=float, help="AdamW weight decay")
    p_train.add_argument("--warmup-ratio", type=float, help="Warmup ratio")
    p_train.add_argument("--dropout", type=float, help="Dropout probability")
    p_train.add_argument("--label-smoothing", type=float, help="Label smoothing")
    p_train.add_argument("--model-name", help="HuggingFace model name")
    p_train.set_defaults(func=cmd_train)

    # ── eval ────────────────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("eval", help="Model evaluation")
    p_eval.add_argument("eval_cmd", choices=["models", "ablation", "confusion", "all"],
                        nargs="?", default="models")
    p_eval.add_argument("--real", action="store_true", help="Use real LLM (not mock)")
    p_eval.add_argument("--samples", type=int, help="Number of samples to evaluate")
    p_eval.set_defaults(func=cmd_eval)

    # ── analyze ─────────────────────────────────────────────────────────────
    p_analyze = subparsers.add_parser("analyze", help="Analyze single text/URL")
    p_analyze.add_argument("--text", help="Text to analyze")
    p_analyze.add_argument("--url", help="URL to analyze")
    p_analyze.add_argument("--real", action="store_true", help="Use real LLM")
    p_analyze.add_argument("--provider", help="LLM provider (nvidia/gemini/qwen)")
    p_analyze.set_defaults(func=cmd_analyze)

    # ── ui ──────────────────────────────────────────────────────────────────
    p_ui = subparsers.add_parser("ui", help="Launch Streamlit UI")
    p_ui.set_defaults(func=cmd_ui)

    # ── test ────────────────────────────────────────────────────────────────
    p_test = subparsers.add_parser("test", help="Run tests and linting")
    p_test.add_argument("test_cmd", choices=["all", "unit", "cov", "lint", "format", "types"],
                        nargs="?", default="all")
    p_test.set_defaults(func=cmd_test)

    # ── db ──────────────────────────────────────────────────────────────────
    p_db = subparsers.add_parser("db", help="Database operations")
    p_db.add_argument("db_cmd", choices=["init", "stats", "export"],
                      nargs="?", default="stats")
    p_db.add_argument("--output", help="Export output file path")
    p_db.set_defaults(func=cmd_db)

    # ── collect ─────────────────────────────────────────────────────────────
    p_collect = subparsers.add_parser("collect", help="Collect news from Vietnamese sources")
    p_collect.set_defaults(func=cmd_collect)

    # ── clean ───────────────────────────────────────────────────────────────
    p_clean = subparsers.add_parser("clean", help="Clean cache and artifacts")
    p_clean.set_defaults(func=cmd_clean)

    # ── info ────────────────────────────────────────────────────────────────
    p_info = subparsers.add_parser("info", help="Show system information")
    p_info.set_defaults(func=lambda _args: _show_info())

    return parser


def _show_info() -> None:
    """Show system and project information."""
    import platform
    import sys

    print()
    print(c(BOLD, "  Verity System Information"))
    print(c(BOLD, "  ─" + "─" * 35))
    print(f"  Python:      {sys.version.split()[0]}")
    print(f"  Platform:    {platform.platform()}")
    print(f"  Project:     {PROJECT_ROOT.name}")
    print(f"  LLM Provider:{ os.getenv('LLM_PROVIDER', 'nvidia')}")

    # Check key files
    from config import VIFACTCHECK_DIR, DATASET_PROCESSED_DIR, MODELS_ARTIFACTS_DIR
    checks = [
        ("ViFactCheck data", VIFACTCHECK_DIR / "train.csv"),
        ("Processed data", DATASET_PROCESSED_DIR / "train.csv"),
        ("TF-IDF model", MODELS_ARTIFACTS_DIR / "tfidf_vectorizer.pkl"),
        (".env file", PROJECT_ROOT / ".env"),
    ]
    print()
    print(c(BOLD, "  File Checks"))
    print(c(BOLD, "  ─" + "─" * 35))
    for name, path in checks:
        status = c(GREEN, "OK") if path.exists() else c(YELLOW, "MISSING")
        print(f"  {status}  {name}")
    print()

    # List available commands
    print(c(BOLD, "  Available Commands"))
    print(c(BOLD, "  ─" + "─" * 35))
    cmds = [
        ("data", "Data pipeline (download/preprocess/full)"),
        ("train", "Model training (tfidf/phobert)"),
        ("eval", "Evaluation (models/ablation/confusion)"),
        ("analyze", "Analyze text or URL"),
        ("ui", "Launch Streamlit UI"),
        ("test", "Run tests (all/unit/coverage/lint)"),
        ("db", "Database operations (init/stats/export)"),
        ("clean", "Clean cache and artifacts"),
        ("info", "Show system information"),
    ]
    for name, desc in cmds:
        print(f"  {c(CYAN, name):<12} {desc}")
    print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print()
        _show_info()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    if IS_WINDOWS:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()