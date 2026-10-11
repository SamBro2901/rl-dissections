import pandas as pd, numpy as np
def fmt(v,d=3):
    if isinstance(v,(float,np.floating)):
        if np.isnan(v): return ""
        if abs(v)>=1e5 or (v!=0 and abs(v)<1e-3): return f"{v:.4g}"
        return f"{v:.{d}f}".rstrip("0").rstrip(".") if abs(v)<100 else f"{v:.1f}"
    return str(v)
def md(df,d=3):
    h="| "+" | ".join(map(str,df.columns))+" |\n|"+"|".join("---" for _ in df.columns)+"|\n"
    return h+"\n".join("| "+" | ".join(fmt(v,d) for v in r)+" |" for r in df.itertuples(index=False))+"\n"
