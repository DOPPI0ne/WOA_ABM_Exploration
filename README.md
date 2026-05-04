Behavioral Sink ABM
Companion code for Regime Mapping and Parameter Importance in Stochastic ABM of Behavioral Sink — Carucci, Mascagni, Gastaldo, Moroni, Bertolotti (WOA 2026).
Overview
Stochastic individual-based ABM of the behavioral sink, the endogenous demographic collapse mechanism first described by Calhoun. Supports large-scale parametric exploration across eight parameters, regime mapping in the fertility–mortality plane, and random forest parameter importance ranking.
Files

ABM.py — simulation engine with Numba backend and Python fallback
Extensive_analysis.py — parallel exploration pipeline, outputs summary CSV
Results.ipynb — regime maps, parameter importance, trajectory visualisations

Usage
bashpip install numpy pandas numba scikit-learn matplotlib tqdm
python Extensive_analysis.py
Open Results.ipynb to reproduce all figures.
Reference
Carucci F., Mascagni L., Gastaldo L., Moroni L., Bertolotti F. (2026). Regime Mapping and Parameter Importance in Stochastic ABM of Behavioral Sink. WOA 2026, Salerno.
