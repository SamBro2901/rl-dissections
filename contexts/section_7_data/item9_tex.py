import re, sys
import pandas as pd
BS=re.escape(chr(92))
from pathlib import Path
D=Path(r"C:/Users/saman/Desktop/Thesis/Thesis Report"); OUT=Path(__file__).parent
tex=(D/"thesis-main.tex").read_text(encoding="utf8",errors="replace").splitlines()
bib=(D/"bibliography.bib").read_text(encoding="utf8",errors="replace")
def strip(l):
    return re.sub("(?<!"+BS+")%.*$","",l)
chap="(preamble)"; labels=[]; cites=[]; refs=[]; chapters=[]
CITE=re.compile(BS+r"(?:[Pp]arencite|[Tt]extcite|[Aa]utocite|[Cc]ite[a-z]*|[Ff]ootcite|[Cc]ites|[Ss]martcite|[Nn]ocite)\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}")
REF=re.compile(BS+r"(?:[Zz]cref|zcpageref|[Cc]ref|autoref|[Vv]ref|ref|pageref|nameref|eqref)\*?(?:\[[^\]]*\])?\{([^}]*)\}")
LAB=re.compile(BS+r"label\{([^}]*)\}")
for i,l in enumerate(tex,1):
    s=strip(l)
    m=re.search(BS+r"chapter\*?\{([^}]*)\}",s)
    if m: chap=m.group(1); chapters.append((i,chap))
    for k in LAB.findall(s): labels.append(dict(file="thesis-main.tex",chapter=chap,line=i,label=k))
    for g in CITE.findall(s):
        for k in g.split(","): 
            if k.strip(): cites.append(dict(chapter=chap,line=i,key=k.strip()))
    for g in REF.findall(s):
        for k in g.split(","):
            if k.strip(): refs.append(dict(chapter=chap,line=i,key=k.strip()))
L=pd.DataFrame(labels); C=pd.DataFrame(cites); Rf=pd.DataFrame(refs)
bibkeys=re.findall(r"^@\w+\s*\{\s*([^,\s]+)\s*,",bib,flags=re.M)
L.to_csv(OUT/"item9_labels.csv",index=False); pd.Series(bibkeys,name="bibkey").to_csv(OUT/"item9_bibkeys.csv",index=False)
undef_c=sorted(set(C.key)-set(bibkeys)); undef_r=sorted(set(Rf.key)-set(L.label))
dup=L.label[L.label.duplicated()].tolist()
C[C.key.isin(undef_c)].to_csv(OUT/"item9_undefined_cites.csv",index=False); Rf[Rf.key.isin(undef_r)].to_csv(OUT/"item9_undefined_refs.csv",index=False)
print("lines",len(tex),"chapters",chapters); print("labels",len(L),"dups",dup)
print(L.groupby("chapter").size().to_string())
print("bib keys",len(bibkeys),"cited keys",C.key.nunique(),"uncited bib keys",len(set(bibkeys)-set(C.key)))
print("undefined cites:",undef_c); print("undefined refs:",undef_r)
print("duplicate bib keys:",[k for k in set(bibkeys) if bibkeys.count(k)>1])
