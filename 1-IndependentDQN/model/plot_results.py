"""
plot_results.py - visualize ONE Independent-DQN run directory's .mat files.

Adapted from Manuscript/data/canonical_ep600_claims1-3/plot_results.py (same style,
same canonical metrics) for the discrete DQN soft baseline. The DQN runs share the
CMDP repo's .mat names, so the figures are directly comparable to the DDPG results:

  viol_rate.png            per-episode training trace P(AoI>tau), per platoon + worst/net
  violation_canonical.png  per-platoon P(AoI>tau) from AoI_evolution (CANONICAL audit metric)
  AoI.png                  per-episode mean AoI, per platoon + network mean, tau line
  power/demand/V2I/V2V.png  per-step metrics reduced to per-platoon time series
  AoI_evolution_sawtooth.png  intra-episode AoI sawtooth (last logged episode)
  reward.png               task1 (V2V+power) / task2 (V2I+AoI) reward traces
  epsilon.png              exploration schedule (DQN-specific; shows when it went greedy)
  lambda.png / Jain.png    only emitted if present (soft DQN has neither; step-5 hard adds lambda)

Canonical violation = P(AoI>tau) = (AoI_evolution > tau).mean(axis=(1,2)) per platoon.
Locked thresholds tau=8, eps=0.10 (kept here only as visual reference lines; the soft
baseline does not enforce them).

Usage:
  python plot_results.py [run_dir]      # default: current directory
Writes PNGs + metrics.txt INTO run_dir.
"""
import argparse
import traceback
from pathlib import Path

import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU = 8.0      # AoI threshold (slots)        -- reference line
EPS = 0.10     # target violation prob.       -- reference line


def load_mat_files(result_dir):
    data = {}
    for p in sorted(result_dir.glob("*.mat")):
        mat = scipy.io.loadmat(p)
        for k in mat:
            if not k.startswith("__"):
                data[k] = np.asarray(mat[k], dtype=np.float64)
    return data


def _save(result_dir, name):
    plt.tight_layout()
    plt.savefig(result_dir / name, dpi=170, bbox_inches="tight")
    plt.close()


def _platoon_lines(ax, mat2d):
    for j in range(mat2d.shape[0]):
        ax.plot(mat2d[j], lw=0.9, alpha=0.75, label="platoon %d" % j)


def plot_viol_rate(d, rd, name, m):
    if "viol_rate" not in d:
        return
    vr = d["viol_rate"]
    E = vr.shape[1]
    last = slice(max(0, E - 100), E)
    nm = float(vr[:, last].mean())
    wm = float(vr[:, last].mean(axis=1).max())
    m["viol_rate.net_mean_last100"] = nm
    m["viol_rate.worst_platoon_last100"] = wm
    fig, ax = plt.subplots(figsize=(8, 5))
    _platoon_lines(ax, vr)
    ax.plot(vr.mean(axis=0), color="black", lw=2.2, label="network mean")
    ax.plot(vr.max(axis=0), color="crimson", lw=2.2, label="worst platoon")
    ax.axhline(EPS, color="green", ls="--", lw=1.5, label="eps=%.2f" % EPS)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Violation rate  P(AoI>tau)")
    ax.set_title("%s\nviol_rate (train).  last-100ep: net-mean=%.3f  worst=%.3f  (eps=%.2f)"
                 % (name, nm, wm, EPS))
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    _save(rd, "viol_rate.png")


def plot_violation_canonical(d, rd, name, m):
    if "AoI_evolution" not in d:
        return
    ev = d["AoI_evolution"]
    viol_pp = (ev > TAU).mean(axis=(1, 2))
    netmean = float(viol_pp.mean())
    worst = float(viol_pp.max())
    m["canonical.net_mean"] = netmean
    m["canonical.worst_platoon"] = worst
    m["canonical.per_platoon"] = [round(float(v), 4) for v in viol_pp]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(viol_pp))
    bars = ax.bar(xs, viol_pp, color="steelblue", alpha=0.9)
    bars[int(np.argmax(viol_pp))].set_color("crimson")
    ax.axhline(EPS, color="green", ls="--", lw=1.6, label="eps=%.2f" % EPS)
    ax.axhline(netmean, color="black", ls=":", lw=1.6, label="net-mean=%.3f" % netmean)
    for x, v in zip(xs, viol_pp):
        ax.text(x, v + 0.005, "%.3f" % v, ha="center", fontsize=8)
    ax.set_xlabel("platoon j")
    ax.set_ylabel("P(AoI>tau)  [canonical, from AoI_evolution]")
    ax.set_title("%s\nper-platoon violation (CANONICAL).  worst=%.3f  net-mean=%.3f  (eps=%.2f)"
                 % (name, worst, netmean, EPS))
    ax.set_xticks(xs)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(rd, "violation_canonical.png")


def plot_aoi(d, rd, name, m):
    if "AoI" not in d:
        return
    aoi = d["AoI"]
    E = aoi.shape[1]
    mean_last = float(aoi[:, max(0, E - 100):E].mean())
    m["mean_AoI_last100"] = mean_last
    fig, ax = plt.subplots(figsize=(8, 5))
    _platoon_lines(ax, aoi)
    ax.plot(aoi.mean(axis=0), color="black", lw=2.2, label="network mean")
    ax.axhline(TAU, color="orange", ls="--", lw=1.2, label="tau=%.0f" % TAU)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean AoI (slots)")
    ax.set_title("%s\nAoI per episode.  last-100ep mean = %.2f slots" % (name, mean_last))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _save(rd, "AoI.png")


def plot_lambda(d, rd, name, m):
    if "lambda" not in d:
        return
    lam = d["lambda"]
    E = lam.shape[1]
    m["lambda_last100_per_platoon"] = [round(float(v), 3) for v in lam[:, max(0, E - 100):E].mean(axis=1)]
    fig, ax = plt.subplots(figsize=(8, 5))
    _platoon_lines(ax, lam)
    ax.set_xlabel("Episode")
    ax.set_ylabel("lambda_j  (per-platoon multiplier)")
    ax.set_title("%s\nper-platoon Lagrange multiplier" % name)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _save(rd, "lambda.png")


def plot_3d_timeseries(d, key, rd, fname, ylabel, name, m, metric_key):
    if key not in d:
        return
    arr = d[key]
    perp = arr.mean(axis=2)
    pooled = float(arr.mean())
    m[metric_key] = pooled
    fig, ax = plt.subplots(figsize=(8, 5))
    _platoon_lines(ax, perp)
    ax.plot(perp.mean(axis=0), color="black", lw=2.2, label="network mean")
    ax.set_xlabel("Eval episode (rolling last 100)")
    ax.set_ylabel(ylabel)
    ax.set_title("%s\n%s.  pooled mean = %.4g" % (name, key, pooled))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _save(rd, fname)


def plot_aoi_sawtooth(d, rd, name):
    if "AoI_evolution" not in d:
        return
    ev = d["AoI_evolution"]
    last_ep = ev[:, -1, :]
    fig, ax = plt.subplots(figsize=(8, 5))
    _platoon_lines(ax, last_ep)
    ax.axhline(TAU, color="orange", ls="--", lw=1.2, label="tau=%.0f" % TAU)
    ax.set_xlabel("Step (within episode)")
    ax.set_ylabel("Instantaneous AoI (slots)")
    ax.set_title("%s\nintra-episode AoI sawtooth (last logged episode)" % name)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _save(rd, "AoI_evolution_sawtooth.png")


def plot_reward(d, rd, name):
    has = [k for k in ("reward_t1", "reward_t2") if k in d]
    if not has:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    if "reward_t1" in d:
        ax.plot(d["reward_t1"].mean(axis=0), color="blue", lw=1.5, label="task1 (V2V+power)")
    if "reward_t2" in d:
        ax.plot(d["reward_t2"].mean(axis=0), color="red", lw=1.5, label="task2 (V2I+AoI)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("%s\nreward traces (mean over platoons)" % name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(rd, "reward.png")


def plot_epsilon(d, rd, name):
    if "epsilon" not in d:
        return
    eps = np.squeeze(d["epsilon"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, color="teal", lw=1.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("epsilon (exploration)")
    ax.set_title("%s\nepsilon-greedy schedule (DQN exploration)" % name)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    _save(rd, "epsilon.png")


def plot_mode(d, rd, name, m):
    if "mode_v2i" not in d:
        return
    mv = d["mode_v2i"]                       # (P, E) per-platoon V2I(inter) mode fraction
    E = mv.shape[1]
    h1 = float(mv[:, :E // 2].mean()); h2 = float(mv[:, E // 2:].mean())
    m["v2i_mode_frac_half1"] = round(h1, 3)
    m["v2i_mode_frac_half2"] = round(h2, 3)
    fig, ax = plt.subplots(figsize=(8, 5))
    _platoon_lines(ax, mv)
    ax.plot(mv.mean(axis=0), color="black", lw=2.2, label="network mean")
    ax.axhline(0.5, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("V2I (inter) mode fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title("%s\nV2I-mode fraction.  1st-half=%.3f  2nd-half=%.3f  (rise => tilt to V2I, starves V2V)"
                 % (name, h1, h2))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _save(rd, "mode_v2i.png")


def plot_jain(d, rd, name):
    if "Jain" not in d:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.squeeze(d["Jain"]), color="purple", lw=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Jain index over per-platoon AoI")
    ax.set_ylim(0, 1.05)
    ax.set_title("%s\nJain fairness index" % name)
    ax.grid(True, alpha=0.3)
    _save(rd, "Jain.png")


def run_one(result_dir):
    rd = Path(result_dir).resolve()
    if not rd.is_dir():
        raise SystemExit("Not a directory: %s" % rd)
    d = load_mat_files(rd)
    if not d:
        raise SystemExit("No .mat in %s" % rd)
    name = rd.name
    m = {"run": name}

    plots = [
        lambda: plot_viol_rate(d, rd, name, m),
        lambda: plot_violation_canonical(d, rd, name, m),
        lambda: plot_aoi(d, rd, name, m),
        lambda: plot_lambda(d, rd, name, m),
        lambda: plot_3d_timeseries(d, "power", rd, "power.png", "Tx power (dBm)", name, m, "mean_power_dBm"),
        lambda: plot_3d_timeseries(d, "demand", rd, "demand.png", "Remaining V2V demand (bits)", name, m, "mean_remaining_V2V_demand"),
        lambda: plot_3d_timeseries(d, "V2I", rd, "V2I.png", "V2I rate", name, m, "mean_V2I_rate"),
        lambda: plot_3d_timeseries(d, "V2V", rd, "V2V.png", "V2V rate", name, m, "mean_V2V_rate"),
        lambda: plot_aoi_sawtooth(d, rd, name),
        lambda: plot_reward(d, rd, name),
        lambda: plot_epsilon(d, rd, name),
        lambda: plot_mode(d, rd, name, m),
        lambda: plot_jain(d, rd, name),
    ]
    for fn in plots:
        try:
            fn()
        except Exception:
            print("  [warn] a plot failed in %s:\n%s" % (name, traceback.format_exc()))

    lines = ["# DQN soft-baseline metrics for %s  (tau=%.0f, eps=%.2f reference)" % (name, TAU, EPS)]
    for k in ("canonical.net_mean", "canonical.worst_platoon", "canonical.per_platoon",
              "viol_rate.net_mean_last100", "viol_rate.worst_platoon_last100",
              "mean_AoI_last100", "mean_power_dBm", "mean_remaining_V2V_demand",
              "mean_V2I_rate", "mean_V2V_rate", "lambda_last100_per_platoon",
              "v2i_mode_frac_half1", "v2i_mode_frac_half2"):
        if k in m:
            lines.append("%-34s = %s" % (k, m[k]))
    (rd / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[ok] %s  -> canonical worst=%.3f net-mean=%.3f  meanAoI=%.2f  power=%.2f dBm"
          % (name, m.get("canonical.worst_platoon", float("nan")),
             m.get("canonical.net_mean", float("nan")),
             m.get("mean_AoI_last100", float("nan")),
             m.get("mean_power_dBm", float("nan"))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", nargs="?", default=".",
                    help="run dir containing .mat (default: current dir)")
    args = ap.parse_args()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.grid": True, "font.size": 10})
    run_one(args.result_dir)


if __name__ == "__main__":
    main()
