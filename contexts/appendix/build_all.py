"""Rebuild everything of the appendices from the raw data, then insert it into thesis-main.tex (from the backup).
Order: tables of Appendix B, figures and data of D, then the assembly of A to F, lint, insertion, label check."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
steps = [
    ['contexts/appendix/A_verify.py'], ['contexts/appendix/build_A_overrides.py'], ['contexts/appendix/gen_A_sweep_table.py'], ['contexts/appendix/gen_A.py'],
    ['flop_analysis/appendix_tables.py'], ['flop_analysis/appendix_tables_verify.py'], ['contexts/appendix/B_independent_check.py'], ['contexts/appendix/gen_B.py'],
    ['contexts/appendix/C_totals_check.py'], ['contexts/appendix/C_flop_tables.py'], ['contexts/appendix/gen_C_tables.py'],
    ['contexts/appendix/D_learning.py'], ['contexts/appendix/gen_D_figures.py'], ['contexts/appendix/gen_D.py'],
    ['contexts/appendix/E_sessions.py'], ['contexts/appendix/gen_E.py'],
    ['contexts/appendix/F_clock_verify.py'], ['contexts/appendix/gen_F.py'],
    ['contexts/appendix/texlint.py'] + [f'contexts/appendix/appendix_{c}.tex' for c in 'ABCDEF'],
    ['contexts/appendix/insert_appendix.py'],
    ['contexts/appendix/labelcheck.py', '../../Thesis Report/thesis-main.tex', '../../Thesis Report/thesis-main.tex.bak-appendix'],
]
for st in steps:
    r = subprocess.run([sys.executable, '-I'] + st, cwd=R, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [''])[-1][:140]
    print(('OK  ' if r.returncode == 0 else 'FAIL'), st[0], '|', last)
    if r.returncode:
        print(r.stderr[-1500:])
        sys.exit(1)
