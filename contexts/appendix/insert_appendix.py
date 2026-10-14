"""Section 9 of the instructions: insert the six appendix chapters into thesis-main.tex and delete the todos they resolve.

Run once.  Makes the backup thesis-main.tex.bak-appendix (if it does not exist), writes contexts/appendix/edit_log.md.
Positions are located by content, not by line number.
"""
import os, re, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
THESIS_DIR = os.path.join(os.path.dirname(os.path.dirname(R)), 'Thesis Report')
TEX = os.path.join(THESIS_DIR, 'thesis-main.tex')
BAK = TEX + '.bak-appendix'
BS = chr(92)

if not os.path.exists(BAK):
    shutil.copy2(TEX, BAK)
tex = open(BAK, encoding='utf-8').read()   # always edit from the backup (idempotent)
orig_lines = tex.split('\n')


def lineno(s, idx):
    return s.count('\n', 0, idx) + 1


log = []

# ---------------------------------------------------------------- 1. appendix chapters
banner = '%==============================================================================\n% APPENDIX BLOCKS (place in the appendix chapters)'
i0 = tex.index(banner)
end_marker = '\n%%% ===============================================================================\n\n' + BS + 'pagestyle{empty}'
i1 = tex.index(end_marker, i0)
old_region = tex[i0:i1]
chapters = []


def soften(t):
    """Allow a line break after an escaped underscore (long identifiers in texttt overflow the margin otherwise)."""
    t = re.sub(re.escape(BS + '_') + r'(?!' + re.escape(BS + 'allowbreak') + ')', lambda m: BS + '_' + BS + 'allowbreak ', t)
    # and after a slash inside \texttt{...} (file paths)
    def slash(m):
        return BS + 'texttt{' + re.sub(r'/(?!' + re.escape(BS + 'allowbreak') + ')', lambda k: '/' + BS + 'allowbreak ', m.group(1)) + '}'
    return re.sub(re.escape(BS + 'texttt{') + r'([^{}]*)\}', slash, t)


for letter, name in zip('ABCDEF', ['hyperparameters', 'full-results', 'flop-details', 'learning', 'reproducibility', 'clock-details']):
    fn = os.path.join(R, f'contexts/appendix/appendix_{letter}.tex')
    body = soften(open(fn, encoding='utf-8').read()).rstrip('\n')
    open(fn, 'w', encoding='utf-8').write(body + '\n')
    chapters.append(f'%%% ===============================================================================\n%                                  APPENDIX {letter}\n%%% ===============================================================================\n{body}\n')
new_region = '\n'.join(chapters)
open(os.path.join(R, 'contexts/appendix/appendices_inline.tex'), 'w', encoding='utf-8').write(new_region)
tex = tex[:i0] + new_region + tex[i1:]
log.append(('1', 'Appendix region', f'old lines {lineno(open(BAK, encoding="utf-8").read(), i0)}..{lineno(open(BAK, encoding="utf-8").read(), i1)} ("APPENDIX BLOCKS" banner, commented clock block, ' + BS + 'chapter{Learning check details}, ' + BS + 'chapter{Reproducibility details})',
            f'replaced by appendix_A.tex .. appendix_F.tex (inline), {len(new_region.splitlines())} lines'))

# ---------------------------------------------------------------- 2. todos resolved by the appendices (delete whole lines)
dels = [
    ('Add this reconciliation as a script to the repository', 'FLOP details: reconciliation script (Appendix C has it; the commit todo is an AUTHOR todo in Appendix C)'),
    ('Placeholder labels still undefined: ' + BS + 'texttt{sec:compute-accounting}', 'placeholder labels sec:compute-accounting and app:flop-details (both defined now)'),
    ('Appendix~A must state that', 'MBPO refit correction (Appendix A, Section "Refit Interval of the MBPO Dynamics Model")'),
    ('Appendix~B should hold the composition', 'MBPO composition and idle-floor fractions (Appendix B, tab:app-mbpo-comp)'),
    ('Scripts ' + BS + 'texttt{flop' + BS + '_analysis/section' + BS + '_5' + BS + '_2' + BS + '_analysis.py}', 'scripts of Section 5.2 and per-segment statistics and fits (Appendix E scripts table, Appendix B)'),
    ('Per-seed orderings and pairwise seed counts (', 'per-seed orderings and pairwise seed counts (Appendix B, tab:app-seed-orderings, tab:app-pairwise)'),
    ('Add the scripts ' + BS + 'texttt{flop' + BS + '_analysis/section' + BS + '_5' + BS + '_3' + BS + '_analysis.py}', 'scripts of Section 5.3 (Appendix E scripts table)'),
    ('Appendix ' + BS + 'zcref{app:clock-provenance} (Provenance of the Two Runs) is a commented block', 'Chapter 6 todo: move app:clock-provenance (Appendix F is active)'),
    ('Appendix ' + BS + 'zcref{app:clock-details} (Allocated Sub-Segments and Idle Windows) is a commented block', 'Chapter 6 todo: move app:clock-details (Appendix F is active)'),
]
lines = tex.split('\n')
for prefix, why in dels:
    hits = [k for k, l in enumerate(lines) if l.lstrip().startswith(BS + 'todo{' + prefix)]
    assert len(hits) == 1, (prefix, hits)
    k = hits[0]
    old = lines[k]
    # line number in the backup (for the log)
    bk = [n for n, l in enumerate(orig_lines) if l == old]
    assert len(bk) == 1, (prefix, bk)
    log.append(('2', why, f'old line {bk[0] + 1}: ' + old.strip()[:110] + '...', 'deleted'))
    del lines[k]
tex = '\n'.join(lines)

open(TEX, 'w', encoding='utf-8').write(tex)

# ---------------------------------------------------------------- edit log
new_lines = tex.split('\n')
with open(os.path.join(R, 'contexts/appendix/edit_log.md'), 'w', encoding='utf-8') as f:
    f.write('# Edit log: insertion of the appendices into `thesis-main.tex`\n\n')
    f.write(f'Backup: `{os.path.basename(BAK)}` (copy of the file before the first edit). Lines before: {len(orig_lines)}, after: {len(new_lines)}.\n\n')
    f.write('## Applied\n\n')
    for e in log:
        if e[0] == '1':
            f.write(f'1. **Appendix chapters.** {e[2]} -> {e[3]}.\n')
        else:
            f.write(f'- **Todo deleted** ({e[1]}). Old: `{e[2]}`. {e[3].capitalize()}.\n')
    # new positions of the chapters
    f.write('\n## New positions (after the edits)\n\n')
    for k, l in enumerate(new_lines):
        if l.startswith(BS + 'chapter{') and k > 7000:
            f.write(f'- line {k + 1}: `{l.strip()}`\n')
prop = os.path.join(R, 'contexts/appendix/edit_log_proposed.md')
if os.path.exists(prop):
    with open(os.path.join(R, 'contexts/appendix/edit_log.md'), 'a', encoding='utf-8') as f:
        f.write('\n' + open(prop, encoding='utf-8').read())
print('done; lines', len(orig_lines), '->', len(new_lines))
