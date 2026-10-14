# Context file for Chapter 2 (Background and Related Work)

Written 2026-10-14 before the LaTeX draft (`contexts/section_2_draft.tex`).
Data files: `contexts/section_2_data/bib_audit.csv`, `contexts/section_2_data/claims_check.md`.
Nothing in this file is a result of this thesis; Chapter 2 contains no numbers from Chapters 4 to 6.

---

## 1. File inventory (what was read)

Repo (`Project Repository/rl-dissections`):
- `CLAUDE.md`, `configs/config.py` (all four algorithm configs), `algorithms/tracker_utils.py` (TrackerTask), `algorithms/tdmpc2.py` (grep of `Q()`, `WorldModel`, SimNorm, two-hot), `flop_analysis/verify_analytic_flops.py` (analytic FLOP derivation), `flop_analysis/flops_per_call.json` (SAC HalfCheetah entry, for the worked example), `flop_analysis/measure_flops.py` (grep: `FlopCounterMode(display=False)`).
- CodeCarbon **3.3.0** source in `.venv/Lib/site-packages/codecarbon` (version file says 3.3.0, same as the thesis runs): `emissions_tracker.py` (`start_task`, `stop_task`, `_do_measurements`, `_measure_power_and_energy`), `external/ram.py`, `external/hardware.py`, `core/cpu.py` (IntelRAPL), `core/rapl.py` (RAPLFile), `core/gpu_device.py`, `core/gpu_nvidia.py`, `core/units.py`, `core/resource_tracker.py`.
- PyTorch `torch/utils/flop_counter.py` at tag v2.13.0 (downloaded from GitHub; the thesis ran 2.13.0; the Windows venv has 2.14.0).

Thesis (`Thesis Report/thesis-main.tex`, 7579 lines before insertion):
- Banner of Chapter 3 at line 2084 (insertion point). Chapter 3 read in full (lines 2089 to 3230): 3.1 Context, 3.2 Metric, 3.3 RQs, 3.4 Harness (incl. 3.4.4 algorithm segments, `tab:segment-meaning`, `tab:status`), 3.5 FLOP accounting, 3.6 Design (baselines, deviations), 3.7 Execution.
- Chapter 5 intro (lines 5605 to 5625) and the `\todo` at line 5957 (Section 2.5.4).
- `thesis-layout.md` section 4 (Chapter 2 spec), `bibliography.bib`, `abbreviations.tex`, `commands.tex`.
- `README.md` was **not** re-read in full (CLAUDE.md summarises it; all protocol statements used in Chapter 2 come from Chapter 3 and the code).

Primary sources downloaded and read as text (PDF to text): Patterson 2021, Strubell 2019, Henderson 2020, Desislavov 2023 (arXiv 2109.05472), Getzner 2023, Yang 2023/24, Gardner 2025, Raffin 2024, Bouza 2023 (abstract and tables via publisher page), Jay 2023, Schwartz 2020, Carbontracker, RL-Scope (abstract), TD-MPC2, MBPO, TD3, SAC and SAC-applications. Metadata of every new bib entry comes from Crossref / arXiv API / publisher pages (see `bib_audit.csv`).

---

## 2. Outline per subsection, facts, sources, labels

Chapter label `ch:background`. Order and labels as in the instruction. Notation as in Chapter 3: $E_s$, $F_s$, $N_{\mathrm{env}}$, $U$, $d$, $B$, $\tau$, $\gamma$.

### 2.1 `sec:rl-prelim` RL preliminaries
Only concepts used later (checked against Chapters 3 to 5):
- MDP $(\mathcal S,\mathcal A,p,r,\gamma)$, policy, return $J(\pi)=\mathbb E[\sum_t\gamma^t r_t]$; episode, **termination** vs. **time limit** (Ch. 3.6.1: HalfCheetah never terminates, Ant can). Source: Sutton and Barto (`sutton2018rl`).
- Action-value $Q^\pi$ and the Bellman equation; the critic regression target $y=r+\gamma(1-\eta)\,\mathrm{(bootstrapped\ value)}$ with terminal flag $\eta$ (written without a clash with $d$).
- Off-policy learning, behaviour policy, **replay buffer** (Lin 1992, `lin1992replay`), minibatch of size $B$, **warmup** with random actions.
- Actor-critic; deterministic policy gradient (Silver 2014, Lillicrap 2016 already in bib).
- **Target networks and Polyak averaging**: $\bar\theta\leftarrow(1-\tau)\bar\theta+\tau\theta$ (Mnih 2015 `mnih2015dqn` for target networks; Polyak and Juditsky 1992 `polyak1992averaging` for averaging). Cost: two elementwise operations per parameter (convention of 3.5.1).
- **Update-to-data ratio** $U$ = gradient steps per environment step; also called replay ratio (Fedus 2020 `fedus2020replay`; D'Oro 2023 `doro2023replayratio`; REDQ `chen2021redq`; MBPO $G$). It changes the number of gradient steps (Ch. 3: $c_{\text{critic}}=N_{\mathrm{env}}U$), not the cost of one step.
- Model-based RL: learned dynamics model, Dyna (Sutton 1991 `sutton1991dyna`), branched short rollouts, model buffer (Janner 2019 verified: Algorithm 2 below).
- Latent world model, **MPPI** (Williams 2017 `williams2017mppi`): sample action sequences from a Gaussian, score by predicted return, refit mean and std to the elites with weights; policy-seeded samples; horizon, iterations, elites. TD-MPC (`hansen2022tdmpc`) and TD-MPC2.

### 2.2 `sec:algos` The four algorithms (published structure)
Verified facts (paper/PDF):
- **SAC** (`haarnoja2018sac`, automatic temperature from `haarnoja2018sacapps`, Table of hyperparameters: "entropy target $-\dim(\mathcal A)$"): stochastic squashed-Gaussian policy; maximum-entropy objective with temperature $\alpha$; **twin critics** with min over the two (clipped double-Q idea from van Hasselt 2010 and Fujimoto 2018); target critics by Polyak averaging; policy loss $\mathbb E[\alpha\log\pi-\min_i Q_i]$; temperature adapted by gradient descent so that the entropy approaches the target (Haarnoja 2018b, Section 5; target $-\dim(A)$ is the paper's hyperparameter table value). The original ICML version (2018a) used a separate value network; the twin-Q form without it is 2018b. The repo's SAC follows 2018b (`SACConfig`: `autotune_alpha`, `target_entropy=None` meaning $-|A|$, `init_alpha=0.2`).
- **TD3** (`fujimoto2018td3`): DDPG plus three changes: clipped double-Q, **target policy smoothing** (noise $\mathcal N(0,0.2)$ clipped to $[-0.5,0.5]$, paper Section 6.1/Alg. 1), **delayed policy updates** (actor and target networks every $d=2$ critic updates). Exploration by Gaussian noise added to the deterministic action. Target $y=r+\gamma\min_iQ'_i(s',\pi'(s')+\epsilon)$.
- **MBPO** (`janner2019mbpo`, Algorithm 2 verified from the PDF): for N epochs: train model on $D_{env}$ by maximum likelihood; for E steps: act in the environment, add to $D_{env}$; for M model rollouts: sample $s_t$ from $D_{env}$, k-step rollout with $\pi$, add to $D_{model}$; for G gradient updates: update the SAC policy on $D_{model}$. Paper Table 1 (Appendix C): E=1000, M=400 per step, ensemble size B=7, 4 hidden layers of 200 (400 for Humanoid), G=40 (HalfCheetah) / 20 (Ant), horizon 1 (HalfCheetah), 1 to 25 over epochs 20 to 100 (Ant). The model is a bootstrap ensemble of probabilistic networks outputting diagonal Gaussians (PETS: `chua2018pets`); a prediction picks one member uniformly at random. **"Elites" are not in the paper**; they come from the public reference implementation (github.com/JannerM/mbpo, per `MBPOConfig` docstring); the chapter says so.
- **TD-MPC2** (`hansen2024tdmpc2`, hyperparameter Table 8 read, partly garbled by PDF extraction, consistent with `TDMPC2Config`): decoder-free; components encoder, latent dynamics, reward head, Q-ensemble (5 networks), policy prior; **SimNorm** latents; reward/value as **discrete regression** (two-hot, 101 bins); TD-target uses the **minimum of two randomly sub-sampled** Q-functions (REDQ idea, Chen 2021; paper text confirms); planning = MPPI: 512 samples, 24 from the policy prior, horizon 3, 6 iterations, 64 elites, temperature 0.5, std in $[0.05,2]$, warm start from previous solution; UTD 1; trained with joint-embedding (consistency), reward and value losses; policy prior maximises Q with entropy bonus. Termination classifier is **not** in the paper (3.4.4 says the same); the reference code has an `episodic` option.

Per-iteration table (concept level, `tab:alg-iteration`) and loops figure (`fig:alg-loops`): segment vocabulary = rollout, buffer sample, critic update, actor update, target update; MBPO: model fit, synthetic rollouts; TD-MPC2: planning (inside rollout), world-model pretraining. `tab:segment-meaning` and `tab:status` are **not** duplicated (forward references).

Published vs. implemented (two-column list used to phrase "as published"):

| | published | implemented in this thesis (Ch. 3.4.4, 3.6.3) |
|---|---|---|
| all | updates interleaved with environment steps (SAC, TD3, TD-MPC2: one update per step) | block schedule: 1000 env steps, then $1000U$ updates, per epoch |
| SAC | width 256 (2018b) | width 1024 |
| TD3 | width (400,300), batch 100, warmup 10000 | width 1024, batch 100 (baseline), warmup 10000 |
| MBPO | G=40 (HalfCheetah) / 20 (Ant), M=400 rollouts per step, refit every 250 steps, E=1000 | U=1, refit and refill once per 1000-step epoch, 10000 start states per refill; policy learner = unmodified SAC |
| TD-MPC2 | benchmarks without termination; update per step | termination classifier on Ant (reference code `episodic`), block schedule, pretraining burst of `warmup_steps` updates as in reference code |

### 2.3 `sec:energy-measurement` Energy measurement on commodity hardware
Facts from CodeCarbon 3.3.0 source (verified, file and function):
- **Task API.** `EmissionsTracker.start_task(name)` (emissions_tracker.py:730): stops the periodic scheduler (`self._scheduler.stop()`), starts the power-monitor thread, sets `_start_time = time.perf_counter()`, calls `hardware.start()` for each device (this re-reads the RAPL files and the NVML energy counter, i.e. re-baselines the counters), snapshots the cumulative emissions data (`_active_task_emissions_at_start`). `stop_task(name)` (line 795): stops the monitor thread, calls `_measure_power_and_energy()` (one last read of all devices), computes `_prepare_emissions_data()` and the **difference to the snapshot** (`compute_delta_emission`), sets the duration to `perf_counter() - task.start_time`. So a task's energy is a counter difference; the polling interval `measure_power_secs` (config default 15 s in the library, 1.0 s in this repo's `ExperimentConfig`) does not limit the task resolution. Reusing a task name: if the name exists in `_tasks`, `start_task` appends a UUID (line ~757); the thesis found corrupted accounting when names were reused and avoids it by unique names (3.4.2; not asserted in Chapter 2).
- **CPU.** `IntelRAPL` (core/cpu.py): reads `energy_uj` files under `/sys/class/powercap/intel-rapl*`; **default: package domains**, `rapl_include_dram=False`, `rapl_prefer_psys=False`; `RAPLFile.delta` handles counter wrap-around using `max_energy_range_uj`. If RAPL is unreadable CodeCarbon falls back to a TDP-based estimate (CPU-load model with a TDP table, or a constant). Machine mode in `tracking_mode`.
- **GPU.** `NvidiaGPUDevice._get_total_energy_consumption` calls `pynvml.nvmlDeviceGetTotalEnergyConsumption` (mJ since the driver was loaded); `GPUDevice.delta` takes the difference between two readings; the "power" CodeCarbon reports for a GPU is derived as energy difference over elapsed time (`Power.from_energies_and_delay`), not read from `nvmlDeviceGetPowerUsage`. NVML doc (R470 archive, read): `nvmlDeviceGetTotalEnergyConsumption` returns energy in mJ since the driver was last reloaded, for Volta or newer; `nvmlDeviceGetPowerUsage` is documented as accurate within $\pm5\,\%$ on Fermi and Kepler.
- **RAM (resolves the inconsistency).** `RAM._calculate_ram_power` (external/ram.py): the docstring still quotes the **old v2 heuristic (3 W per 8 GB)**, but 3.3.0 uses a **per-DIMM model**: estimated DIMM count from the capacity (`_estimate_dimm_count`: $\le$32 GB mapped to 2 or 4 DIMMs; 32 to 64 GB returns **4**; 64 to 128 GB returns 8), then power = 5 W $\times$ DIMMs for up to 4 DIMMs (x86), with 0.9, 0.8, 0.7 factors for the DIMMs above 4, 8, 16; minimum 10 W on x86. For the thesis machine (60.8 GB, 4 estimated DIMMs) this gives $4\times5=\SI{20}{\watt}$, which is exactly the constant reported in Chapter 7. The old rule would give $60.8/8\times3=\SI{22.8}{\watt}$, not 20.0. Therefore: **`thesis-layout.md` (line 121, "3 W per 8 GB") is outdated for CodeCarbon 3.x; the thesis text (20.0 W) is consistent with the code.** Chapter 3.4.6 only says "constant heuristic based on capacity", which is correct. RAM energy = power $\times$ elapsed time since the last measurement (`BaseHardware.measure_power_and_energy`), so it carries no information about memory activity. CodeCarbon docs (methodology page, read) state the same per-slot model.
- **Emissions.** emissions = energy $\times$ carbon intensity (docs); the thesis reports kWh and converts to joules ($\SI{1}{\kilo\watt\hour}=\SI{3.6e6}{\joule}$).

Literature facts for the "general" statements:
- RAPL: counters in the model-specific registers, exposed in Linux by the powercap, perf-events and msr interfaces; domains package, core, uncore, dram, psys; 32-bit counters that overflow ("about 60 s under heavy load" per Intel manual, quoted by Raffin 2024); energy counters updated approximately every 1 ms (Jay 2023 states this; Khan 2018 studied update interval and accuracy); validation studies: Hackenberg 2013 (`hackenberg2013power`), Khan 2018, Desrochers 2016 (DRAM) (`desrochers2016dram`), Raffin 2024 (`raffin2024dissecting`: pitfalls of tools incl. overflow handling and polling timing). Jay 2023 (`jay2023softwaremeters`): nine software meters vs. external meters, tool energy overhead below 1 % on average.
- NVML/nvidia-smi: Yang et al. 2023 (arXiv v3, `yang2023parttime`): the **power** reading has about $\pm5\,\%$ error (not $\pm5$ W as documented), update period about 100 ms on A100 (about 20 ms on V100/Pascal), and on A100/H100 only 25 % of the runtime is sampled for power; cumulative-energy-counter behaviour on a Blackwell card (RTX 5090) is **not** covered by that paper, so the thesis infers its own counter step from the data (3.4.2 / Ch. 7). PETSc docs (web page, not citable here) say 20 to 100 ms; not used.
- Burtscher 2014 (`burtscher2014k20`): K20 built-in sensor reports power with a lag/averaging and needs correction.
- Software meter comparisons: Bouza 2023 (`bouza2023carbon`: seven tools, CodeCarbon closest to wattmeters in their tests, RAPL access matters, process-level tools not comparable to wall power), Jay 2023.
- DVFS (`sec:gpu-dvfs`): DVFS lowers voltage and frequency together; GPUs expose graphics (core/SM) and memory clocks, boost and power states; locking the graphics clock (`nvidia-smi -lgc`, Ch. 3.4.5) fixes it. Mei 2017 and Abe 2014: effect of core and memory scaling on energy depends on architecture and application; Zeus (`you2023zeus`): power limit and batch size trade energy against time. No claims beyond the sources; Chapter 6 holds the discussion.

### 2.4 `sec:compute-accounting` Compute accounting
Convention (must equal 3.5.1; each statement checked against `flop_analysis/verify_analytic_flops.py` and `torch/utils/flop_counter.py`):
1. FLOPs = count; FLOP/s = rate. Schwartz 2020: "FPO", analytic, ADD and MUL, hardware-agnostic; limits (memory, implementation dependence).
2. Matmul only: PyTorch's registry counts `aten.mm`, `addmm`, `bmm`, `baddbmm`, convolutions, scaled-dot-product attention variants (flop_counter.py lines 75 to 130, 729 ff.). Formula `m*n*2*k` ("2k-1 technically", comment in the source). Works from tensor shapes on the dispatch level, device independent, counts backward matmuls when the backward is run inside the context.
3. Dense layer on $B$ rows: $2Bn_{\mathrm{in}}n_{\mathrm{out}}$. Worked example: $B=256$, $n_{\mathrm{in}}=n_{\mathrm{out}}=1024$: $2\cdot256\cdot1024\cdot1024=\num{536870912}$ ($\approx\num{5.4e8}$) FLOPs forward.
4. Backward $\approx 2\times$ forward: weight gradient $G_W=\delta^\top x$ and input gradient $G_x=\delta W^\top$ are each a matmul of the same size; the input gradient of the first layer is not computed when the input needs no gradient (`bwd(..., first_input_grad=False)` in `verify_analytic_flops.py`). Critic update decomposition in the script: policy forward on the batch + four critic forwards (target nets and online nets) + backward of both online critics (weights; no input gradient at the first layer).
5. Not counted: activations, normalisation, losses, optimiser arithmetic, sampling, top-$k$, host-side work. Elementwise Polyak average = two operations per parameter, own unit, never added (the repo: `target_update_elementwise_ops = 2 * (2 * n_params)` for two critics).
6. Why a proxy: Schwartz 2020 (hardware independence, tied to work done); forward-reference `\zcref{sec:related-flop}`. Definition of $e_s=E_s/F_s$ (J/FLOP) **without values**.
7. Roofline (`williams2009roofline`): attainable performance = min(peak, bandwidth $\times$ arithmetic intensity); arithmetic intensity = FLOPs per byte of memory traffic; why FLOPs and energy can diverge (memory-bound, small kernels).
Worked example for one critic update (concept, not values from Ch. 4 to 6): SAC critic of shape $(\mathrm{obs}+\mathrm{act})\to1024\to1024\to1$ at batch 256; the forward of the 1024x1024 layer alone costs $\num{5.4e8}$; the thesis' measured per-call constants live in Chapter 3.5/Appendix C and are not repeated.

### 2.5 Related work
Literature map (all sources verified; see `bib_audit.csv`):

- **2.5.1 Green AI and reporting** (`sec:related-green`): Lacoste 2019 (`lacoste2019quantifying`, ML CO$_2$ calculator, GPU-hours-based), Strubell 2019 (measured via nvidia-smi and RAPL sampling, then $\times$ PUE), Schwartz 2020 (Green AI, FPO), Patterson 2021 (measured average power $\times$ hours $\times$ PUE; compares with calculators), Wu 2022 (`wu2022sustainable`, lifecycle view), Luccioni 2023 BLOOM (`luccioni2023bloom`), Luccioni 2024 (`luccioni2024power`, measured inference energy per task), Henderson 2020.
- **2.5.2 Tools** (`sec:related-tools`): experiment-impact-tracker = Henderson 2020 (RAPL and nvidia-smi sampling; reports kWh and CO$_2$e; RL leaderboard on Pong/Breakout for PPO, A2C, A2C+VTrace, DQN: totals only); Carbontracker (`anthony2020carbontracker`: measures per epoch and predicts the total with a linear model, error 4.9 to 19.1 % after one epoch in its tests); CodeCarbon; Green Algorithms (`lannelongue2021green`); Zeus (energy/time optimisation of training); comparisons Bouza 2023 and Jay 2023.
- **2.5.3 Meter accuracy** (`sec:related-meter-accuracy`): RAPL: Hackenberg 2013, Khan 2018, Desrochers 2016, Raffin 2024; GPU: Burtscher 2014, Yang 2023; tool-level: Bouza, Jay.
- **2.5.4 FLOP-based energy estimation** (`sec:related-flop`): constant efficiency: Desislavov 2023 (energy = FLOPs per forward pass divided by GFLOPS/W taken as peak FLOPS over TDP, i.e. full utilisation; inference only; they note their own caveat on measured power); Henderson 2020 (describes the PFLOPs-hr $\times$ TDP / peak-throughput estimate, and shows on image-classification inference that FPOs vs. energy has $R^2=0.083$ across architectures but $R^2\approx0.999$ within VGG variants: FPOs order models within an architecture class, not across); Patterson 2021 **does not** convert FLOPs with an assumed efficiency (see claims check): it multiplies measured average power by hours and PUE, reports measured FLOPs/s (1.6x (TPU) and 3.5x (GPU) below theoretical peak per Watt), and uses an assumed 40 % of peak only for one hypothetical supercomputer example; Getzner 2023 (`getzner2023accuracy`: sum of layer-wise energy predictions for CNN inference on CPU, measured with CodeCarbon; linear and polynomial regression per layer type; workshop paper). Scope limits: inference or large-scale training of supervised models; none for RL loops.
- **2.5.5 Compute, energy and sample efficiency in RL** (`sec:related-rl-compute`): Henderson 2018 (variance, already cited in 3.6.2); Fedus 2020 and D'Oro 2023 (replay ratio); REDQ; MBPO ($G$ up to 40); TD-MPC2 uses UTD 1 and planning; RL-Scope (`gleeson2021rlscope`, MLSys 2021: profiler attributing CPU/GPU time to algorithmic operations; RL less GPU-bound than supervised learning; on-policy at least 3.5x more simulation-bound than off-policy; time not energy); QuaRL (`krishnan2022quarl`, TMLR 2022: int8 quantisation of actor inference reduces carbon of training 1.9 to 3.8x, measured with experiment-impact-tracker, total per experiment); Gardner 2025 (`gardner2025greener`: seven model-free algorithms (DQN, TRPO, A2C, ARS, PPO, RecurrentPPO, QR-DQN), Atari, CodeCarbon 2.4.2, RAPL + nvidia-smi, logged every 10000 steps, totals per algorithm and game, one normalised metric (return per kWh); no phase breakdown; no FLOP normalisation; no model-based or planning method; arXiv preprint, a journal version exists (Springer, Artificial Intelligence Review, found by search; metadata not verified).
- **2.5.6 Positioning** (`sec:related-positioning`): **Gap statement supported, with a more careful wording.** Closest studies measure *totals* per algorithm (Henderson 2020 leaderboard, Gardner 2025, QuaRL), or *time* per operation (RL-Scope), or estimate energy from FLOPs for supervised inference (Desislavov, Getzner). No study found that attributes hardware-measured energy to training-loop segments and normalises by counted compute across model-free, model-based and planning-based algorithms. Limits of the claim: absence in a web-based literature search up to 2026-10, not a systematic review.

---

## 3. FLOP convention, once, with the worked example

See 2.4 above. Consistency points with 3.5: (a) 3.5.1 repeats "multiply-add counts two FLOPs" and "$2Bn_{in}n_{out}$": identical. (b) 3.5.1 lists the not-counted items: identical list. (c) 3.5.7 uses the first-layer input-gradient exception: stated in 2.4. (d) Polyak 2 ops per parameter: stated. (e) "J/FLOP is energy per matmul FLOP": stated as definition.

---

## 4. Obligations from section 3 of the instruction

1. Labels `sec:compute-accounting`, `sec:related-flop` exist after insertion (static check: the only undefined targets that remain are pre-existing `app:*` and `lst:*`).
2. Todo at the end of 3.1: Section 2.5 carries the references (2.5.1, 2.5.2, 2.5.5, 2.5.6). Section 1.2 not written (chapter 1 is template text).
3. 5.2 verification: **see `claims_check.md`**. Result: desislavov confirmed; strubell confirmed; henderson2020 confirmed (and adds a FLOP discussion); **patterson2021carbon NOT confirmed** (needs rewording in Chapter 5, not edited here).
4. Todo in 3.5.1 (matmul-only scope and factor two): satisfied by 2.4.
5. `TODO(2.4)` comment in 3.5.1: satisfied.
6. Terms to define before first use in Ch. 3 to 5: UTD, target net/Polyak, twin critics/clipped double-Q, entropy temperature (auto-tuning, target entropy), target policy smoothing and delayed updates, probabilistic ensemble/elites/branched rollouts/model buffer, latent world model, MPPI (samples, horizon, iterations, elites, policy-seeded), termination classifier (as in reference `episodic` mode), RAPL, NVML, CodeCarbon task, DVFS/clock lock, arithmetic intensity: all defined in Chapter 2.
7. No contradiction with 3.4.4 and 3.6.3: Chapter 2 states "as published" and defers every implemented deviation to those sections.

---

## 5. Open questions for the author

1. Published version of Gardner et al. 2025 (Springer; the page redirected to a login): should the bib entry cite the journal version? Currently arXiv only.
2. The Yang et al. work has a published SC24 version with a different title ("Accurate and Convenient Energy Measurements for GPUs: A Detailed Study of NVIDIA GPU's Built-In Power Sensor"). The facts used were checked in the arXiv v3 text; cite which?
3. `nvml` entry has no verified author/year (vendor docs); urldate set to 2026-10-14, URL kept; `nvmlapi` added for the Device Queries page that was read (R470 archive).
4. Section 1.2 (introduction) is still template text; the references added here can be reused there.
5. Chapter 5 sentence on Patterson (see claims check) needs a decision.
6. Whether Chapter 2 should keep the word "elite" attributed to the reference implementation only (done), or whether you want to cite the code repository as `@misc` (not added).
