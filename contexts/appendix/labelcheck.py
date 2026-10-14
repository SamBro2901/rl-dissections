"""Static label check: unique labels, defined reference targets (comments stripped), appendix labels used vs defined,
brace and begin/end balance.  Usage: labelcheck.py new.tex [old.tex]"""
import re, sys, collections

BS = chr(92)


def strip_comments(s):
    out = []
    for line in s.splitlines():
        i = 0
        while True:
            j = line.find('%', i)
            if j < 0:
                break
            if j > 0 and line[j - 1] == BS:
                i = j + 1
                continue
            line = line[:j]
            break
        out.append(line)
    return '\n'.join(out)


def analyse(path):
    raw = open(path, encoding='utf-8').read()
    s = strip_comments(raw)
    labels = re.findall(r'\\label\{([^}]*)\}', s)
    cnt = collections.Counter(labels)
    dup = [k for k, v in cnt.items() if v > 1]
    refs = []
    for m in re.finditer(r'\\(?:zcref|Zcref|ref|autoref|eqref|zcpageref)(?:\[[^\]]*\])?\{([^}]*)\}', s):
        for t in m.group(1).split(','):
            refs.append(t.strip())
    undefined = sorted({r for r in refs if r not in cnt})
    # texttt-protected labels in todos are not references; ignore
    env = collections.Counter()
    for m in re.finditer(r'\\(begin|end)\{([^}]*)\}', s):
        env[m.group(2)] += 1 if m.group(1) == 'begin' else -1
    unbalanced = {k: v for k, v in env.items() if v}
    depth = 0
    neg = 0
    for k, ch in enumerate(s):
        if ch == '{' and (k == 0 or s[k - 1] != BS):
            depth += 1
        elif ch == '}' and (k == 0 or s[k - 1] != BS):
            depth -= 1
            if depth < 0:
                neg += 1
                depth = 0
    return dict(labels=cnt, dup=dup, refs=refs, undefined=undefined, unbalanced=unbalanced, brace_depth=depth, brace_neg=neg, text=s)


if __name__ == '__main__':
    new = analyse(sys.argv[1])
    print('labels', len(new['labels']), 'duplicates', new['dup'])
    print('reference targets', len(new['refs']), 'undefined', len(new['undefined']))
    print('unbalanced environments', new['unbalanced'], 'brace depth', new['brace_depth'], 'negative', new['brace_neg'])
    if len(sys.argv) > 2:
        old = analyse(sys.argv[2])
        print('old undefined', len(old['undefined']))
        print('NEW undefined (not in old):', sorted(set(new['undefined']) - set(old['undefined'])))
        print('resolved since old:', sorted(set(old['undefined']) - set(new['undefined'])))
        print('old undefined still undefined:', sorted(set(old['undefined']) & set(new['undefined'])))
    apps = sorted({r for r in new['refs'] if r.startswith('app:') or r.startswith('tab:app-') or r.startswith('sec:app-') or r.startswith('fig:app-')})
    print('app-related references:', len(apps), 'undefined among them:', [a for a in apps if a not in new['labels']])
    print('cited app: labels:', sorted({r for r in new['refs'] if r.startswith('app:')}))
    print('defined app: labels:', sorted(k for k in new['labels'] if k.startswith('app:')))
