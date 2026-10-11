import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common7 import *
import glob
R=runs(); rows=[]
for _,r in R.iterrows():
    m=jload(r.path/"metadata.json"); e=pd.read_csv(glob.glob(str(r.path/"emissions_base_*.csv"))[0],nrows=1).iloc[0]
    rows.append(dict(run_dir=r.run_dir,torch_version=m["torch_version"],codecarbon_version=m["codecarbon_version"],cuda_device_name=m["cuda_device_name"],device=m["device"],
      emissions_codecarbon_version=e.codecarbon_version,python_version=e.python_version,os=e.os,cpu_model=e.cpu_model,cpu_count=e.cpu_count,gpu_model=e.gpu_model,gpu_count=e.gpu_count,ram_total_size_gb=round(e.ram_total_size,3),
      tracking_mode=e.tracking_mode,metadata_keys=",".join(sorted(m.keys()))))
V=pd.DataFrame(rows); V.to_csv(OUT/"item7_per_run_versions.csv",index=False)
for c in V.columns[1:]:
    print(c, V[c].nunique(), V[c].value_counts().to_dict() if c!="metadata_keys" else "")
