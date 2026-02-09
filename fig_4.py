import os,sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D


def ini_plot():
    config = {
        "figure.dpi": 600,
        "mathtext.fontset": "stix",
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 5,
        "axes.linewidth": 0.8,
        "legend.handletextpad": 0.4,
        "legend.framealpha": 1.0,
        "legend.handlelength": 1.2,
        "patch.linewidth": 0.5,
        "axes.edgecolor": "k",
        "axes.labelcolor": "k",
        "xtick.color": "k",
        "ytick.color": "k",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
    config = {
    "figure.dpi": 600,
    "mathtext.fontset": "cm",
    "mathtext.fontset": "stix",
    "font.family": "Times New Roman",
    "legend.fancybox": True,
    "legend.handletextpad": 0.4,
    "legend.framealpha": 1.0,
    "legend.handlelength": 1.2,
    "patch.linewidth": 0.5,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "xtick.color": "k",
    "ytick.color": "k",
    }
    rcParams.update(config)

ini_plot()

RESULT_DIR = "./PTQMC-PRC/data_fig4/"
ROOT = os.path.abspath(os.path.dirname(__file__))
PYFCIQMC = os.path.join(ROOT, "pyFCIQMC")
sys.path.insert(0, PYFCIQMC)

files = sorted(
    [f for f in os.listdir(RESULT_DIR)
     if f.startswith("weights_g") and f.endswith(".pkl")]
)

if len(files) == 0:
    raise RuntimeError(f"No files found in {RESULT_DIR} matching weights_g*.pkl")

with open(os.path.join(RESULT_DIR, files[0]), "rb") as f:
    data0 = pickle.load(f)
weights0 = data0["w"]
MAX_ORDER = len(weights0) - 2
print(f"[info] detected perturbative orders: w1 ... w{MAX_ORDER}")

g_vals = []
entropy = []
w8_vals = []
w9_vals = []

for fname in files:
    g = float(fname.replace("weights_g", "").replace(".pkl", ""))
    g_vals.append(g)

    with open(os.path.join(RESULT_DIR, fname), "rb") as f:
        data = pickle.load(f)

    weights = data["w"]
    K = len(weights) - 2
    K = min(K, MAX_ORDER)

    cum_P = {}

    S_list = []
    
    w8_sum = 0.0
    w9_sum = 0.0

    for k in range(1, K + 1):
        for D, w in weights[k].items():
            cum_P[D] = cum_P.get(D, 0.0) + abs(w)
        
        if k == 8:
            w8_sum = sum(cum_P.values())
        if k == 9:
            w9_sum = sum(cum_P.values())

        total = sum(cum_P.values())
        if total <= 0:
            S_list.append(0.0)
            continue

        p = np.array([v / total for v in cum_P.values() if v > 0.0], dtype=float)
        S = -np.sum(p * np.log(p))
        S_list.append(S)
    if K < MAX_ORDER:
        S_list += [np.nan] * (MAX_ORDER - K)

    entropy.append(S_list)
    w8_vals.append(w8_sum)
    w9_vals.append(w9_sum)

g_vals = np.array(g_vals, dtype=float)
entropy = np.array(entropy, dtype=float)
w8_vals = np.array(w8_vals, dtype=float)
w9_vals = np.array(w9_vals, dtype=float)

idx = np.argsort(g_vals)
g_vals = g_vals[idx]
entropy = entropy[idx, :]
w8_vals = w8_vals[idx]
w9_vals = w9_vals[idx]

eS_8 = np.exp(entropy[:, 7])
eS_9 = np.exp(entropy[:, 8])
convergence = np.abs((eS_8 - eS_9) / (eS_8 + 1e-10))

fig, ax = plt.subplots(figsize=(4, 3.5))

converged = convergence < 0.002

i = 0
while i < len(g_vals):
    is_converged = converged[i]
    x_start = g_vals[i]
    
    j = i
    while j < len(g_vals) and converged[j] == is_converged:
        j += 1
    
    if j < len(g_vals):
        x_end = g_vals[j]
    else:
        x_end = g_vals[-1]
    
    if is_converged:
        bg_color = '#CCCCCC'
    else:
        bg_color = '#999999'
    
    ax.axvspan(x_start, x_end, alpha=0.3, color=bg_color, zorder=0)
    
    i = j

even_colors = cm.Blues(np.linspace(0.4, 0.9, 5))
odd_colors = cm.Reds(np.linspace(0.4, 0.9, 4))

even_idx = 0
odd_idx = 0

for k in range(2, 11):
    if k - 2 >= MAX_ORDER:
        break
    y = entropy[:, k - 2]
    
    if k % 2 == 0:
        color = even_colors[even_idx]
        linestyle = '-'
        even_idx += 1
    else:
        color = odd_colors[odd_idx]
        linestyle = '-'
        odd_idx += 1
    
    ax.plot(
        g_vals, np.exp(np.array(y)),
        lw=1.0,
        color=color,
        alpha=0.85,
        linestyle=linestyle,
        label=rf"${k}$"
    )

ax.set_xlim(np.min(g_vals), np.max(g_vals))

ax.set_xlabel(r"$g$ [a.u.]", fontsize=18)
ax.set_ylabel(r"$e^S$", fontsize=18)

ax.tick_params(direction="in", top=True, right=True)

ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

ax.tick_params(axis="both", which="major", length=4, width=0.8, labelsize=15, direction="in", top=True, right=True)
ax.tick_params(axis="both", which="minor", length=2, width=0.5, direction="in", top=True, right=True)


norm_even = Normalize(vmin=2, vmax=10)
sm_even = ScalarMappable(cmap=cm.Blues, norm=norm_even)
sm_even.set_array([])

axins_even = inset_axes(
    ax, width="2.5%", height="30%", loc='lower right',
    bbox_to_anchor=(-0.12, 0.06, 0.95, 0.95),
    bbox_transform=ax.transAxes
)

cbar_even = fig.colorbar(
    sm_even,
    cax=axins_even,
    orientation='vertical',
    spacing='proportional'
)

cbar_even.set_ticks([2, 4, 6, 8, 10])
cbar_even.ax.tick_params(labelsize=10)
cbar_even.ax.yaxis.set_ticks_position('left')
cbar_even.ax.yaxis.set_label_position('left')

cbar_even.solids.set_edgecolor('none')
cbar_even.solids.set_linewidth(0)

norm_odd = Normalize(vmin=3, vmax=9)
sm_odd = ScalarMappable(cmap=cm.Reds, norm=norm_odd)
sm_odd.set_array([])
axins_odd = inset_axes(ax, width="2.5%", height="30%", loc='lower right',
                        bbox_to_anchor=(-0.08, 0.06, 0.95, 0.95), bbox_transform=ax.transAxes)
cbar_odd = fig.colorbar(sm_odd, cax=axins_odd, orientation='vertical', drawedges=False)
cbar_odd.ax.tick_params(labelsize=10)
cbar_odd.set_ticks([3, 5, 7, 9])
cbar_odd.solids.set_edgecolor("face")
cbar_odd.solids.set_linewidth(0)
cbar_odd.solids.set_edgecolor("face")

ax.text(0.79, 0.40, '    Orders', transform=ax.transAxes, 
        fontsize=12, ha='center', va='bottom')

plt.savefig(f"Figure_4.png", bbox_inches="tight", transparent=False, dpi=600)
plt.savefig(f"Figure_4.pdf", bbox_inches="tight", transparent=False, dpi=600)

