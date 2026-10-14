"""List the regions in which thesis-main.tex differs from the backup (only the insertion region and the deleted todo lines are expected)."""
import difflib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
T = os.path.join(os.path.dirname(os.path.dirname(R)), 'Thesis Report', 'thesis-main.tex')
a = open(T + '.bak-appendix', encoding='utf-8').read().split('\n')
b = open(T, encoding='utf-8').read().split('\n')
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
n_del = n_rep = 0
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'equal':
        continue
    print(f'{tag:8s} old lines {i1 + 1}-{i2}  ->  new lines {j1 + 1}-{j2}   ({i2 - i1} old, {j2 - j1} new)')
    if tag == 'delete':
        n_del += i2 - i1
        for l in a[i1:i2]:
            print('     deleted:', l.strip()[:100])
    if tag in ('replace', 'insert'):
        n_rep += j2 - j1
print('deleted lines:', n_del, ' lines in replaced/inserted regions:', n_rep, ' old lines', len(a), ' new lines', len(b))
