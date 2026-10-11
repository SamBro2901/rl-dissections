import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common7 import *
rows=[]
def add(n,item,found,ok,src): rows.append(dict(claim=n,item=item,found=found,status="VERIFIED" if ok else "MISMATCH",source=src))
# 7
sw=pd.read_csv(REPO/"contexts/section_5.3_data/o2_swaps_vs_baseline.csv")
e=sw[sw.metric=="energy"]; j=sw[(sw.metric=="jpf")&(sw.sweep!="baseline")]
add(7,"energy ordering td3<sac<mbpo at all 8 sweep values, both envs",f"{int(e.same_as_baseline.sum())}/{len(e)} same as baseline; orderings={sorted(e.ordering.unique())}",
    e.same_as_baseline.all() and list(e.ordering.unique())==["td3 < sac < mbpo"],"contexts/section_5.3_data/o2_swaps_vs_baseline.csv")
o3=pd.read_csv(REPO/"contexts/section_5.3_data/o3_seed_level.csv"); oe=o3[o3.metric=="energy"]
sc=[f"seed_{s}" for s in CANON]; n=sum((oe[c]=="td3 < sac < mbpo").sum() for c in sc); tot=len(oe)*5
add(7,"seed-level energy orderings equal td3<sac<mbpo",f"{n}/{tot}",n==80 and tot==80,"contexts/section_5.3_data/o3_seed_level.csv")
for env,g in j.groupby("env"):
    add(7,f"J/FLOP non-baseline sweep values differing from baseline ordering ({env})",f"{int((~g.same_as_baseline).sum())} of {len(g)}",(~g.same_as_baseline).sum()==6 and len(g)==7,"o2_swaps_vs_baseline.csv (metric=jpf)")
# 8 from pooled_summary
ps=pd.read_csv(REPO/"contexts/section_5.4_data/pooled_summary.csv"); a=ps[ps.group.str.startswith("ALL")].iloc[0]
add(8,"total energy Ant/HC ratio range over 27 pairs",f"{a.E_total_min:.3f}–{a.E_total_max:.3f}",round(a.E_total_min,3)==1.027 and round(a.E_total_max,3)==1.248,"contexts/section_5.4_data/pooled_summary.csv (ratio of means per pair)")
add(8,"rollout energy Ant/HC ratio range",f"{a.E_rollout_min:.4f}–{a.E_rollout_max:.4f}",round(a.E_rollout_min,3)==1.106 and round(a.E_rollout_max,3)==2.288,"pooled_summary.csv E_rollout_min/max (1.1055 rounds to 1.106 at 3dp: 1.10552 -> 1.106)")
# 9 clock study direct from run dirs
for clk,rd,E_,D_,P_ in [("R2000","results/mbpo/HalfCheetah-v5/seed_0/20260904_122958",189.5,27.3,115.7),("R200","results/mbpo/HalfCheetah-v5/seed_0/20260904_130915",568.2,95.5,99.1)]:
    import glob
    d=pd.read_csv(glob.glob(str(REPO/rd/"emissions_base_*.csv"))[0])
    d["seg"]=d.task_name.str.replace(r"_\d+$","",regex=True)
    tr=d[~d.seg.isin(["idle_baseline_head","idle_baseline_tail","warmup"])]
    en=tr.energy_consumed.sum()*3.6e6/1e3; du=tr.duration.sum()/60; pw=tr.energy_consumed.sum()*3.6e6/tr.duration.sum()
    ok=abs(en-E_)<0.06 and abs(du-D_)<0.06 and abs(pw-P_)<0.06
    add(9,f"{clk} ({rd}) training tasks (non-idle, non-warmup)",f"{en:.1f} kJ, {du:.2f} min, {pw:.1f} W",ok,"emissions_base_*.csv, sum over tasks excluding idle_baseline_*/warmup")
# 10
dirs=[p.parent for p in (REPO/"results").rglob("metadata.json")]
canon=[p for p in dirs if any(f"seed_{s}"==p.parent.name for s in CANON)]
R=runs(); 
add(10,"300 run directories; 280 canonical; 20 dev",f"total={len(dirs)}, canonical={len(canon)}, dev={len(dirs)-len(canon)}, configs={R.groupby(['algo','env','config']).ngroups}",len(dirs)==300 and len(canon)==280 and R.groupby(['algo','env','config']).ngroups==56,"rglob results/**/metadata.json")
# 11
T=pd.read_pickle(OUT/"_task_table.pkl"); st=[]
for (algo,seg),g in T[T.seg.isin(["rollout","synthetic_rollout_generation"])].groupby(["algo","seg"]):
    st.append(dict(algo=algo,seg=seg,n_tasks=len(g),min_s=g.duration.min(),median_s=g.duration.median(),p99_s=g.duration.quantile(.99),max_s=g.duration.max(),n_ge_1s=int((g.duration>=1).sum())))
st=pd.DataFrame(st); st.to_csv(OUT/"item1_claim11_task_durations.csv",index=False); print(st.to_string())
def med(a,s): return st[(st.algo==a)&(st.seg==s)].iloc[0]
add(11,"SAC rollout median s / n",f"{med('sac','rollout').median_s:.4f} / {med('sac','rollout').n_tasks}",abs(med('sac','rollout').median_s-0.3409)<5e-4 and med('sac','rollout').n_tasks==7000,"item1_claim11_task_durations.csv")
add(11,"TD3 rollout median s / n",f"{med('td3','rollout').median_s:.4f} / {med('td3','rollout').n_tasks}",abs(med('td3','rollout').median_s-0.2175)<5e-4 and med('td3','rollout').n_tasks==8000,"")
add(11,"MBPO rollout median s / n",f"{med('mbpo','rollout').median_s:.4f} / {med('mbpo','rollout').n_tasks}",abs(med('mbpo','rollout').median_s-0.3536)<5e-4 and med('mbpo','rollout').n_tasks==8000,"")
add(11,"MBPO synthetic_rollout_generation median s",f"{med('mbpo','synthetic_rollout_generation').median_s:.6f}",abs(med('mbpo','synthetic_rollout_generation').median_s-0.008523)<5e-6,"")
for a in ["sac","td3","mbpo","tdmpc2"]:
    m=med(a,"rollout"); add(11,f"all {a} rollout tasks < 1 s (max)",f"max={m.max_s:.3f}s, n>=1s={m.n_ge_1s}/{m.n_tasks}",m.n_ge_1s==0,"")
out=pd.DataFrame(rows); out.to_csv(OUT/"item1_claims_part2.csv",index=False); print(out.to_string())
