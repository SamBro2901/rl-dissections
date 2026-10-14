"""Appendix A: assemble appendix_A.tex from the template and the sweep rows (run gen_A_sweep_table.py first)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(HERE, 'appendix_A.tpl.tex'), encoding='utf-8').read()
rows = open(os.path.join(HERE, 'appendix_data', 'A_sweep_rows.tex'), encoding='utf-8').read().rstrip('\n')
assert '%%SWEEP_ROWS%%' in tpl
open(os.path.join(HERE, 'appendix_A.tex'), 'w', encoding='utf-8').write(tpl.replace('%%SWEEP_ROWS%%', rows))
print('appendix_A.tex written')
