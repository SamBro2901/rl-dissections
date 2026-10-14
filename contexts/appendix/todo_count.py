"""Count and list the \\todo{} commands of thesis-main.tex that wait for the Linux machine or the author (NOT RUN / AUTHOR),
and the research-PC rows of tab:thr-provenance (todos without a label)."""
import os, re, collections
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
T = os.path.join(os.path.dirname(os.path.dirname(R)), 'Thesis Report', 'thesis-main.tex')
lines = open(T, encoding='utf-8').read().split('\n')
BS = chr(92)
cnt = collections.Counter()
rows = []
for i, l in enumerate(lines):
    if l.lstrip().startswith('%'):
        continue
    for m in re.finditer(re.escape(BS) + r'todo\{', l):
        head = l[m.end():m.end() + 70]
        kind = 'NOT RUN' if head.startswith('NOT RUN') else ('AUTHOR' if head.startswith('AUTHOR') else 'other')
        cnt[kind] += 1
        rows.append((i + 1, kind, head[:60]))
print(dict(cnt), 'total todos', sum(cnt.values()))
for r in rows:
    if r[0] > 6000 or r[1] != 'other':
        print(r)
