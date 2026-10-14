"""Small LaTeX lint for the appendix drafts: unescaped _ in \\texttt, unbalanced braces, \\begin/\\end balance."""
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


def check(path):
    s = strip_comments(open(path, encoding='utf-8').read())
    probs = []
    for m in re.finditer(r'\\texttt\{((?:[^{}]|\{[^{}]*\})*)\}', s):
        body = m.group(1)
        if re.search(r'(?<!\\)_', body):
            line = s.count('\n', 0, m.start()) + 1
            probs.append(f'{path}:{line}: unescaped _ in texttt: {body[:60]}')
    depth = 0
    for k, ch in enumerate(s):
        if ch == '{' and (k == 0 or s[k - 1] != BS):
            depth += 1
        elif ch == '}' and (k == 0 or s[k - 1] != BS):
            depth -= 1
            if depth < 0:
                probs.append(f'{path}: negative brace depth at line {s.count(chr(10), 0, k) + 1}')
                depth = 0
    if depth != 0:
        probs.append(f'{path}: brace imbalance {depth}')
    env = collections.Counter()
    for m in re.finditer(r'\\(begin|end)\{([^}]*)\}', s):
        env[m.group(2)] += 1 if m.group(1) == 'begin' else -1
    for k, v in env.items():
        if v:
            probs.append(f'{path}: environment {k} unbalanced ({v})')
    return probs


if __name__ == '__main__':
    allp = []
    for p in sys.argv[1:]:
        allp += check(p)
    print('\n'.join(allp) if allp else 'lint ok')
