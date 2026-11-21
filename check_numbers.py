"""Rerun every script in analysis/ and compare its printed output with the saved copy in results/.
The saved outputs hold every number in the paper; results/README.md says where each section's numbers are printed.
Usage: python check_numbers.py            (compare)
       python check_numbers.py --update   (overwrite results/ with fresh output)"""
import subprocess, sys, difflib, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent; A = ROOT / "analysis"; R = ROOT / "results"
scripts = sorted(p for p in A.glob("*.py") if p.stem not in ("common", "section5_common"))
failed = 0
for p in scripts:
    t = time.time(); out = subprocess.run([sys.executable, p.name], cwd=A, capture_output=True, text=True)
    text = out.stdout + out.stderr; ref = R / f"{p.stem}.txt"
    if "--update" in sys.argv: ref.write_text(text); print(f"updated {ref.name}"); continue
    if out.returncode != 0: failed += 1; print(f"FAIL  {p.name}: exit {out.returncode}\n{text[-800:]}"); continue
    old = ref.read_text() if ref.exists() else ""
    diff = list(difflib.unified_diff(old.splitlines(), text.splitlines(), "saved", "now", lineterm="", n=0))
    if diff: failed += 1; print(f"DIFF  {p.name} ({time.time() - t:.0f}s)"); print("\n".join(diff[:40]))
    else: print(f"same  {p.name} ({time.time() - t:.0f}s)")
print("all outputs identical" if not failed else f"{failed} script(s) differ or failed"); sys.exit(1 if failed else 0)
