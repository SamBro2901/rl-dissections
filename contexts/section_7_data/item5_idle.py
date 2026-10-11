import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common7 import *
I=pd.read_csv(REPO/"idle_power.csv"); assert len(I)==280 and I.seed.isin(CANON).all(), len(I)
print("variant col:",I.variant.value_counts().to_dict())
# ---- 5a
def q(s): return dict(median=s.median(),q1=s.quantile(.25),q3=s.quantile(.75),max=s.max())
rows=[]
for scope,g in [("ALL",I)]+[(f"{a}|{e}",x) for (a,e),x in I.groupby(["algo","env"])]:
    for conv,h,t in [("as_recorded","head_total_w","tail_total_w"),("window_corrected","head_total_w_corr","tail_total_w_corr")]:
        rel=(g[t]-g[h]).abs()/g[h]*100; ab=(g[t]-g[h]).abs()
        r=dict(scope=scope,convention=conv,n_runs=len(g),n_tail_gt_head=int((g[t]>g[h]).sum()),
               rel_pct_median=rel.median(),rel_pct_q1=rel.quantile(.25),rel_pct_q3=rel.quantile(.75),rel_pct_max=rel.max(),
               abs_W_median=ab.median(),abs_W_q1=ab.quantile(.25),abs_W_q3=ab.quantile(.75),abs_W_max=ab.max(),
               signed_W_mean=(g[t]-g[h]).mean(),head_W_mean=g[h].mean(),tail_W_mean=g[t].mean())
        rows.append(r)
D=pd.DataFrame(rows); D.to_csv(OUT/"item5a_idle_drift.csv",index=False)
pd.set_option("display.width",250); print(D.round(3).to_string())
# ---- 5b
R=runs(); R=R[R.config=="base"].copy(); assert len(R)==40
T=pd.read_pickle(OUT/"_task_table.pkl"); T=T[T.config=="base"]
MEAS=["rollout","gradient_updates","dynamics_model_update","synthetic_rollout_generation","world_model_pretrain"]
M=T[T.seg.isin(MEAS)].groupby(["run_dir","seg"]).agg(E_kWh=("energy_consumed","sum"),T_s=("duration","sum")).reset_index()
M["E_gross"]=M.E_kWh*3.6e6
Ii=I.assign(run_dir="results/"+I.run_dir).set_index("run_dir")
M["P_head"]=M.run_dir.map(Ii.head_total_w_corr); M["P_tail"]=M.run_dir.map(Ii.tail_total_w_corr)
M["algo"]=M.run_dir.map(R.set_index("run_dir").algo); M["env"]=M.run_dir.map(R.set_index("run_dir").env)
# sanity: gross total vs pipeline total
chk=M.groupby("run_dir").E_gross.sum().rename("tot").to_frame().join(R.set_index("run_dir").total_E); print("max rel err task-sum vs pipeline total_E:",((chk.tot-chk.total_E).abs()/chk.total_E).max())
F=R.set_index("run_dir").total_F
out=[]; ratios=[]
VAR={"mean_head_tail":lambda d:(d.P_head+d.P_tail)/2,"head_only":lambda d:d.P_head,"tail_only":lambda d:d.P_tail}
for v,f in VAR.items():
    M["P_idle"]=f(M); M["E_net"]=M.E_gross-M.P_idle*M.T_s
    for (a,e),g in M.groupby(["algo","env"]):
        runs_=g.run_dir.unique(); tg=g.groupby("run_dir").E_gross.sum(); tn=g.groupby("run_dir").E_net.sum(); Fm=F.loc[runs_].mean()
        for seg,s in g.groupby("seg"):
            out.append(dict(variant=v,algo=a,env=e,segment=seg,n_runs=len(s),E_gross_J=s.E_gross.mean(),E_net_J=s.E_net.mean(),T_s=s.T_s.mean(),P_idle_W_mean=s.P_idle.mean(),
                share_gross_pct=(s.set_index("run_dir").E_gross/tg).mean()*100,share_net_pct=(s.set_index("run_dir").E_net/tn).mean()*100))
        out.append(dict(variant=v,algo=a,env=e,segment="TRAINING_TOTAL",n_runs=len(runs_),E_gross_J=tg.mean(),E_net_J=tn.mean(),T_s=g.groupby("run_dir").T_s.sum().mean(),
            P_idle_W_mean=g.groupby("run_dir").P_idle.mean().mean(),share_gross_pct=100,share_net_pct=100,
            jpf_gross_pJ=tg.mean()/Fm*1e12,jpf_net_pJ=tn.mean()/Fm*1e12,net_over_gross=tn.mean()/tg.mean(),total_flops=Fm))
O=pd.DataFrame(out); O.to_csv(OUT/"gross_vs_net.csv",index=False)
# ordering + ratios
tot=O[O.segment=="TRAINING_TOTAL"]
for (v,e),g in tot.groupby(["variant","env"]):
    for basis in ["gross","net"]:
        s=g.set_index("algo")[f"E_{basis}_J"].sort_values(); order=" < ".join(s.index)
        d=dict(variant=v,env=e,basis=basis,ordering=order)
        for lo,hi in [("td3","sac"),("sac","mbpo"),("mbpo","tdmpc2")]: d[f"{hi}/{lo}"]=s[hi]/s[lo]
        j=g.set_index("algo")[f"jpf_{basis}_pJ"].sort_values(); d["jpf_ordering"]=" < ".join(j.index); d.update({f"jpf_{k}_pJ":j[k] for k in j.index})
        ratios.append(d)
Rr=pd.DataFrame(ratios); Rr.to_csv(OUT/"gross_vs_net_ratios.csv",index=False); print(Rr.round(3).to_string())
print(tot[tot.variant=="mean_head_tail"][["algo","env","E_gross_J","E_net_J","net_over_gross","jpf_gross_pJ","jpf_net_pJ"]].round(3).to_string())
sh=O[(O.variant=="mean_head_tail")&(O.segment!="TRAINING_TOTAL")]; print(sh[["algo","env","segment","share_gross_pct","share_net_pct"]].round(2).to_string())
