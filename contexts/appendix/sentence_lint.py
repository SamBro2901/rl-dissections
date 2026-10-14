"""List lines of the appendix chapters that hold more than one sentence (one sentence per line is the convention of the thesis source).
Usage: sentence_lint.py [--fix]   (--fix splits prose lines at '. ' before an upper-case word or a \\Zcref/\\zcref command)"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
BS = chr(92)
fix = '--fix' in sys.argv
# a sentence end: a lowercase letter, digit, closing brace or parenthesis, then '. ' and an upper-case word or \Zcref
pat = re.compile(r'(?<=[a-z0-9\)\}])\. (?=[A-Z' + re.escape(BS) + r'])')
abbr = ('e.g.', 'i.e.', 'Fig.', 'cf.', 'et al.', 'vs.')
total = 0
for letter in 'ABCDEF':
    fn = os.path.join(HERE, f'appendix_{letter}.tpl.tex')
    if letter in 'BE':
        pass
    lines = open(fn, encoding='utf-8').read().split('\n')
    out = []
    for i, l in enumerate(lines):
        s = l.lstrip()
        prose = (not s.startswith('%') and '&' not in l and not s.startswith(BS + 'caption') and not s.startswith(BS + 'todo') and not s.startswith(BS + 'item')
                 and not s.startswith(BS + 'begin') and not s.startswith(BS + 'end') and '%%' not in l)
        if prose and pat.search(l) and not any(a in l for a in abbr):
            parts = pat.split(l)
            total += len(parts) - 1
            if fix:
                out.extend(p if k == len(parts) - 1 else p + '.' for k, p in enumerate(parts))
                continue
            print(f'{letter}:{i + 1}: {l[:110]}')
        out.append(l)
    if fix:
        open(fn, 'w', encoding='utf-8').write('\n'.join(out))
print('multi-sentence prose lines (split points):', total)
