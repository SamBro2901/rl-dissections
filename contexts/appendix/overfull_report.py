"""Map the overfull boxes of a lualatex log (lines >= FIRST) to the nearest table/figure label of the .tex file.
Usage: overfull_report.py build_dir [first_line] [min_pt]"""
import bisect, re, sys, os

d = sys.argv[1]
first = int(sys.argv[2]) if len(sys.argv) > 2 else 7016
minpt = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
logname = sys.argv[4] if len(sys.argv) > 4 else 'thesis-main.log'
log = open(os.path.join(d, logname), errors='replace').read()
tex = open(os.path.join(d, 'thesis-main.tex'), encoding='utf-8').read().split('\n')
BS = chr(92)
labels = []
for i, l in enumerate(tex):
    m = re.search(re.escape(BS) + r'label\{([^}]*)\}', l)
    if m:
        labels.append((i + 1, m.group(1)))
ls = [a for a, _ in labels]


def near(line):
    k = bisect.bisect_right(ls, line) - 1
    nxt = labels[k + 1] if k + 1 < len(labels) else None
    if nxt and nxt[0] - line <= 12 and nxt[1].startswith(('tab:', 'fig:')):
        return nxt[1]
    return labels[k][1]


pat = re.compile(r'Overfull .hbox \(([\d.]+)pt too wide\) (?:in paragraph|in alignment|detected) at lines (\d+)--(\d+)')
agg = {}
for w, a, b in pat.findall(log):
    a = int(a)
    if a < first:
        continue
    lab = near(a)
    agg[lab] = max(agg.get(lab, 0), float(w))
for k, v in sorted(agg.items(), key=lambda t: -t[1]):
    if v >= minpt:
        print(f'{v:8.1f}  {k}')
print('total labels with overfull boxes >= %.1f pt: %d' % (minpt, sum(1 for v in agg.values() if v >= minpt)))
err = re.findall(r'^\./thesis-main\.tex:(\d+): (.*)$', log, flags=re.M)
print('errors:', len(err), err[:6])
und = re.findall(r"(?:Reference|Citation) `([^']*)' on page", log)
print('undefined references/citations in log:', len(set(und)), sorted(set(und))[:10])
