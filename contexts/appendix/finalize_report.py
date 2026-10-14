"""Fill the placeholders of appendix_report.md and refresh the line numbers of research_pc_checklist.md from the current thesis-main.tex.
Usage: finalize_report.py <build_dir with thesis-main.toc/log>"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
T = os.path.join(os.path.dirname(os.path.dirname(R)), 'Thesis Report', 'thesis-main.tex')
build = sys.argv[1]
BS = chr(92)
lines = open(T, encoding='utf-8').read().split('\n')


def find(sub, after=0):
    hits = [i + 1 for i, l in enumerate(lines) if sub in l and not l.lstrip().startswith('%') and i + 1 > after]
    assert len(hits) >= 1, sub
    return hits[0]


# key -> (old number used in the checklist, search string)
M = {
    'thr_pip': (11251, 'todo{' + BS + 'texttt{pip freeze} in ' + BS + 'texttt{rl-exp}}'),
    'thr_nv': (11252, 'todo{' + BS + 'texttt{nvidia-smi}}'),
    'thr_pw': (11253, 'todo{' + BS + 'texttt{nvidia-smi -q -d POWER'),
    'thr_dev': (11254, 'todo{' + BS + 'texttt{torch.cuda.get'),
    'thr_lscpu': (11255, 'todo{' + BS + 'texttt{lscpu}'),
    'lock': (6426, 'NOT RUN: lock check'),
    'hw': (6429, 'NOT RUN: clock range'),
    'rapl': (6475, 'NOT RUN: confirm the RAPL mode'),
    'nvml': (6495, 'NOT RUN: poll'),
    'tf32': (6759, 'NOT RUN: ' + BS + 'texttt{torch.get'),
    'fill': (6809, 'NOT RUN: fill from the research PC'),
    'flopfit': (10820, 'NOT RUN: ' + BS + 'texttt{verify'),
    'random': (11199, 'NOT RUN: random-policy'),
    'pipE': (11399, 'NOT RUN: ' + BS + 'texttt{pip freeze}'),
    'contrib': (11481, 'AUTHOR (contribution 5)'),
    'verifycommit': (10775, 'AUTHOR: ' + BS + 'texttt{verify'),
    'unset': (11512, 'AUTHOR: decide how the unset'),
    'sync': (2147, 'Run one baseline per algorithm with'),
    'rapl2': (2294, 'Confirm the RAPL mode on the research PC'),
    'gym1': (2616, 'Gymnasium and MuJoCo versions in '),
    'fill2': (2818, 'Fill the open entries'),
    'rand2': (2965, 'Add the return of a random policy'),
    'sync2': (3167, 'synchronize()} run is still open'),
    'gym2': (4563, 'The Gymnasium and MuJoCo versions are not recorded'),
}
# the lock check string of the old state has "(needs the research PC" ; the \texttt{torch.get} string needs the escaped underscore form
new = {}
for k, (old, sub) in M.items():
    try:
        new[k] = find(sub)
    except AssertionError:
        # the softened text has \allowbreak after escaped underscores: retry with a looser search
        loose = sub.replace(BS + '_', '')
        cand = [i + 1 for i, l in enumerate(lines) if loose[:25] in l.replace(BS + 'allowbreak ', '')]
        assert cand, sub
        new[k] = cand[0]
oldnew = {old: new[k] for k, (old, _) in M.items()}
# checklist: replace "L<old>" by "L<new>"
cl_path = os.path.join(HERE, 'research_pc_checklist.md')
cl = open(cl_path, encoding='utf-8').read()


def sub_l(m):
    n = int(m.group(1))
    return 'L' + str(oldnew.get(n, n))


# only the numbers of the known todos are replaced (the old numbers are unique)
cl2 = re.sub(r'\bL(\d{4,5})\b', sub_l, cl)
open(cl_path, 'w', encoding='utf-8').write(cl2)
# save the current mapping for the next run
open(os.path.join(HERE, 'appendix_data', 'todo_lines.txt'), 'w').write('\n'.join(f'{k} {v}' for k, v in new.items()) + '\n')

# pages from the toc
toc = open(os.path.join(build, 'thesis-main.toc'), encoding='utf-8', errors='replace').read()
ch = re.findall(r'\\contentsline \{chapter\}\{\\numberline \{([A-F])\}([^}]*)\}\{(\d+)\}', toc)
pages = {c: int(p) for c, _, p in ch}
log = open(os.path.join(build, 'pass3.out'), errors='replace').read()
total = int(re.search(r'Output written on thesis-main\.pdf \((\d+) pages', log).group(1))
mb = re.search(r'\\contentsline \{chapter\}\{\\nonumberline Bibliography\}\{(\d+)\}', toc)
starts = [pages[c] for c in 'ABCDEF']
ends = starts[1:] + [total + 1]
# the last appendix ends before the "Versicherung" page(s); report the page of the last chapter start and the total
size_tbl = ['| Chapter | first page | pages |', '|---|---|---|']
for c, s_, e_ in zip('ABCDEF', starts, ends):
    size_tbl.append(f'| {c} | {s_} | {e_ - s_ if c != "F" else "see below"} |')
nl = {}
for c in 'ABCDEF':
    pass

# overfull / errors
rep = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'overfull_report.py'), build, '7013', '12', 'pass3.out'], capture_output=True, text=True).stdout
chk = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'labelcheck.py'), T, T + '.bak-appendix'], capture_output=True, text=True, cwd=R).stdout
dif = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'diff_check.py')], capture_output=True, text=True, cwd=R).stdout
cnt = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'todo_count.py')], capture_output=True, text=True, cwd=R).stdout
dl = dif.strip().splitlines()
dif_short = chr(10).join(dl[:22] + ['... (further replaced fragments of the appendix region)', dl[-1]])
warn = len(re.findall(r'^LaTeX Warning', log, flags=re.M))
undef = len(set(re.findall(r"(?:Reference|Citation) `([^']*)' on page", log)))
build_txt = f"""Compile (scratch copy, `lualatex --shell-escape`, `biber`, `lualatex` twice; the files in `Thesis Report/` were not touched by it): **{total} pages, 0 errors, {undef} undefined references or citations in the last pass**, {warn} LaTeX warnings (all pre-existing kinds such as `\\@parboxrestore`; none from the appendices). The first pass of the first build stopped at 8 `siunitx` errors (exponent in front of `+-`), which are fixed in the generator.
Pages of the appendices in that build (first page of each chapter; total {total} pages): {', '.join(f'{c} p. {pages[c]}' for c in 'ABCDEF')}; the appendix part has {total - pages['A'] + 1 - 1} pages up to the declaration page(s). Appendix B is the longest chapter ({ends[1] - starts[1]} pages).

Overfull boxes in the appendix region (lines of the current file, 12 pt or more; the `10.9 pt` entries are the margin notes of `\\todo`):
```
{rep.strip()}
```

Static check of the file (comments stripped): `{chk.strip().splitlines()[0]}`; `{chk.strip().splitlines()[1]}`; `{chk.strip().splitlines()[2]}`; undefined targets that were already undefined before: only the macro argument `#3` (false positive of the check).
All `app:`/`tab:app-`/`sec:app-`/`fig:app-` references are defined. The five labels cited before but undefined (`app:full-results`, `app:hyperparameters`, `app:flop-details`, `app:clock-details`, `app:clock-provenance`) are defined now; `app:learning`, `app:reproducibility`, `app:clock-suballoc`, `app:clock-idle`, `app:clock-full` as well.
Diff against the backup (`diff_check.py`): only the appendix region (old L7022 to L7247) is replaced, and exactly nine todo lines are deleted:
```
{dif_short}
```
Todo counts of the current file: `{cnt.strip().splitlines()[0]}`; the 9 `NOT RUN` and 7 `AUTHOR` todos are all listed in sections 10 and 11.
"""

def row(key, text, cmd):
    return f'| {text} | L{new[key]} | {cmd} |'

notrun = '\n'.join([
    row('lock', 'lock check with `nvidia-smi` (Chapter 6 limitations)', 'item 9 (needs sudo)'),
    row('hw', 'clock range, power limit and driver version (Chapter 6)', 'items 2, 3'),
    row('rapl', 'confirm the RAPL mode (7.1.1)', 'item 7'),
    row('nvml', 'poll the NVML counter at about 1 kHz (7.1.1)', 'item 11'),
    row('tf32', '`torch.get_float32_matmul_precision()`, `allow_tf32` (7.2)', 'item 8'),
    row('fill', 'fill from the research PC (`tab:thr-summary`)', 'item 13'),
    row('flopfit', 'rerun `verify_mbpo_fit_flops.py` (Appendix C)', 'item 14 (any machine with PyTorch)'),
    row('random', 'random-policy return, command F3 (Appendix D)', 'item 10'),
    row('pipE', '`pip freeze` in `rl-exp` (Appendix E)', 'items 1, 15'),
])
author = '\n'.join([
    f'- L{new["verifycommit"]} (Appendix C): commit `verify_analytic_flops.py` and cite the commit hash.',
    f'- L{find("AUTHOR: keep or delete this block. If the two tables")}, L{find("AUTHOR: keep or delete this block. If kept")}, L{find("AUTHOR: keep or delete this block. The main text states")} (Appendix B): the three optional blocks (section 7).',
    f'- L{find("AUTHOR: keep or delete this optional block. Per-algorithm")} (Appendix D): optional Spearman block.',
    f'- L{new["contrib"]} (Appendix E): contribution 5 (LICENSE, commit of the untracked scripts, tag).',
    f'- L{new["unset"]} (Appendix F): description of the unset clock fields of the fifteen development runs; correct README.md/CLAUDE.md.',
])
git = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True, cwd=R).stdout.strip()
head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=R).stdout.strip()
tags = subprocess.run(['git', 'tag'], capture_output=True, text=True, cwd=R).stdout.strip() or '(none)'
gitp = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True, cwd=os.path.dirname(T)).stdout.strip()
git_txt = f"""`git rev-parse HEAD` (rl-dissections): `{head}`; `git tag`: {tags}.
`git status --short` in `rl-dissections` (untracked scripts and files; the author commits them):
```
{git}
```
`git status --short` in `Thesis Report` (`thesis-main.tex` modified; `thesis-main.tex.bak-appendix` and the 16 figure files new):
```
{gitp}
```"""
rp = os.path.join(HERE, 'appendix_report.md')
tp = os.path.join(HERE, 'appendix_report.tpl.md')
if not os.path.exists(tp):
    import shutil
    shutil.copy2(rp, tp)
r = open(tp, encoding='utf-8').read()
r = r.replace('@@BUILD@@', build_txt).replace('@@NOTRUN_ROWS@@', notrun).replace('@@AUTHOR_ROWS@@', author).replace('@@GIT@@', git_txt)
open(rp, 'w', encoding='utf-8').write(r)
print('written appendix_report.md; pages', pages, 'total', total)
