import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.interpolate import pade
import numpy as np

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

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
    "axes.linewidth": 1.0,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "xtick.color": "k",
    "ytick.color": "k",
    }
plt.rcParams.update(config)
g_fci = np.arange(-2.0, 1.3 + 1e-12, 0.1)
e_fci = corr_fci = [
    -0.69778844,
    -0.64299899,
    -0.58953292,
    -0.53746276,
    -0.48686890,
    -0.43784073,
    -0.39047797,
    -0.34489220,
    -0.30120868,
    -0.25956829,
    -0.22012986,
    -0.18307265,
    -0.14859912,
    -0.11693791,
    -0.08834681,
    -0.06311574,
    -0.04156931,
    -0.02406869,
    -0.01101221,
    -0.00283400,
     0.00000000,
    -0.00300043,
    -0.01233813,
    -0.02851239,
    -0.05199873,
    -0.08322572,
    -0.12255107,
    -0.17023984,
    -0.22644773,
    -0.29121205,
    -0.36445153,
    -0.44597484,
    -0.53549598,
    -0.63265404
]

data_dir = "./data_fig3/"
def read_data(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    g_list = data[:, 0]
    e_list = data[:, 1]
    return g_list, e_list

g_adc3_d, e_adc3_d = read_data(f"{data_dir}adc3-d.txt")
g_uccsd, e_uccsd = read_data(f"{data_dir}uccsd.txt")
g_magnus3, e_magnus3 = read_data(f"{data_dir}magnus3.txt")
g_fciqmc, e_fciqmc = read_data(f"{data_dir}fciqmc.txt")
g_fciqmc_error, e_fciqmc_error = read_data(f"{data_dir}dfciqmc.txt")

order = np.array([
     0,  1,  2,  3,  4,  5,  6,  7,  8,
     9, 10, 11, 12, 13, 14, 15, 16
], dtype=int)

e_a = np.array([
     1.100000,
    -0.000000,
    -0.606466,
     0.199816,
    -0.725832,
    -0.097698,
     0.389294,
    -2.197480,
     2.707014,
    -2.162499,
    -4.254796,
    15.754298,
   -29.226819,
    23.849638,
    31.593009,
  -165.926283,
   335.281696
], dtype=float)

err_a = np.array([
    6.344132e-16,
    3.172066e-16,
    3.172066e-16,
    3.464376e-03,
    3.586252e-03,
    5.106647e-03,
    7.716460e-03,
    1.713027e-02,
    3.591661e-02,
    4.086924e-02,
    6.423088e-02,
    2.221701e-01,
    4.671114e-01,
    5.964727e-01,
    8.236963e-01,
    3.045031e+00,
    6.911933e+00
], dtype=float)

de_a = np.array([
     4.200000,
    -1.100000,
    -0.606466,
     0.806282,
    -0.925648,
     0.628134,
     0.486992,
    -2.586774,
     4.904494,
    -4.869513,
    -2.092297,
    20.009094,
   -44.981117,
    53.076456,
     7.743371,
  -197.519292,
   501.207980
], dtype=float)

derr_a = np.array([
    0.0,0.0,0.0,
    2.449684e-02,
    4.194660e-02,
    5.283547e-02,
    5.676643e-02,
    1.642468e-01,
    3.657316e-01,
    5.185832e-01,
    5.626138e-01,
    1.923803e+00,
    4.810119e+00,
    7.303668e+00,
    7.608783e+00,
    2.581524e+01,
    6.951257e+01
], dtype=float)


e_b = np.array([
    -3.000000,
     0.000000,
    -1.342857,
    -2.578197,
    -3.572312,
    -4.143358,
    -4.162256,
    -3.631906,
    -2.746387,
    -1.873306,
    -1.462204,
    -1.839526,
    -3.040418,
    -4.666574,
    -5.945092,
    -5.995195,
    -4.273484
], dtype=float)

err_b = np.array([
    0.000000e+00,
    0.000000e+00,
    6.344132e-17,
    5.271745e-03,
    1.177708e-02,
    1.813503e-02,
    2.248268e-02,
    2.303982e-02,
    2.028051e-02,
    1.799473e-02,
    2.262830e-02,
    2.918027e-02,
    3.317700e-02,
    4.579663e-02,
    7.068125e-02,
    8.941195e-02,
    8.810243e-02
], dtype=float)

de_b = np.array([
    -4.000000,
     3.000000,
    -1.342857,
    -1.235340,
    -0.994114,
    -0.571047,
    -0.018898,
     0.530350,
     0.885520,
     0.873080,
     0.411102,
    -0.377321,
    -1.200892,
    -1.626157,
    -1.278518,
    -0.050102,
     1.721711
], dtype=float)

derr_b = np.array([
    0.0,0.0,0.0,
    3.727687e-02,
    5.259925e-02,
    5.319142e-02,
    5.271702e-02,
    5.494207e-02,
    7.972613e-02,
    1.055452e-01,
    1.220138e-01,
    1.179709e-01,
    1.522294e-01,
    2.325127e-01,
    2.931232e-01,
    2.728974e-01,
    3.281239e-01
], dtype=float)


g_ptqmc = np.array([1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2])
e_ptqmc = np.array([-0.632145, -0.533357, -0.445803, -0.365073, -0.290930, -0.226204, -0.169141, -0.121198, -0.084954, -0.052023, -0.027440, -0.011831, -0.002875, 0.0, -0.002602, -0.010139, -0.023602, -0.040796, -0.063258, -0.086395, -0.112062, -0.144886, -0.171132, -0.211146, -0.239054, -0.296606])
err_ptqmc = np.array([0.001707, 0.002993, 0.006609, 0.005277, 0.006400, 0.004527, 0.007742, 0.006966, 0.008572, 0.009567, 0.009426, 0.003958, 0.000935, 0.0, 0.001201, 0.004598, 0.004075, 0.007638, 0.008581, 0.010454, 0.013503, 0.017179, 0.025523, 0.028398, 0.032141, 0.030169])

def get_pade_estimate(coeffs, errors, L, M, x=1.0, n_samples=3000, seed=12345,
                      den_min=1e-8, rcond=1e-12, w0_mode="median"):
    coeffs = np.asarray(coeffs, float)
    errors = np.asarray(errors, float)
    N = len(coeffs) - 1

    if L + M > N:
        raise ValueError(f"不满足约束条件: L + M ({L+M}) 必须 <= (len(coeffs)-1) ({N})")

    pos = errors > 0
    if np.any(pos):
        if w0_mode == "median":
            err0 = np.median(errors[pos])
        elif w0_mode == "min":
            err0 = np.min(errors[pos])
        elif w0_mode == "mean":
            err0 = np.mean(errors[pos])
        else:
            raise ValueError("w0_mode must be one of {'median','min','mean'}")
    else:
        err0 = 1.0

    def _solve_single_pade(c_sampled):
        n_eq = N + 1
        n_vars = (L + 1) + M
        A = np.zeros((n_eq, n_vars))
        y = np.zeros(n_eq)

        for n in range(n_eq):
            w = 1.0 / (errors[n] if errors[n] > 0 else err0)

            if n <= L:
                A[n, n] = w

            for j in range(1, M + 1):
                if n - j >= 0:
                    A[n, (L + 1) + (j - 1)] = -c_sampled[n - j] * w

            y[n] = c_sampled[n] * w

        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        keep = s > (rcond * s[0]) if s.size else np.array([], dtype=bool)
        if not np.any(keep):
            return None, None

        s_inv = np.zeros_like(s)
        s_inv[keep] = 1.0 / s[keep]
        sol = Vt.T @ (s_inv * (U.T @ y))
        return sol[:L+1], sol[L+1:]

    rng = np.random.default_rng(seed)
    valid_vals = []

    fixed_mask = np.zeros_like(coeffs, dtype=bool)
    fixed_mask[:3] = True
    idx_to_sample = np.where(~fixed_mask)[0]

    for _ in range(n_samples):
        c_test = coeffs.copy()
        if idx_to_sample.size > 0:
            c_test[idx_to_sample] = rng.normal(coeffs[idx_to_sample], errors[idx_to_sample])

        a, b = _solve_single_pade(c_test)
        if a is None:
            continue

        num = np.polyval(a[::-1], x)
        den = 1.0 + x * np.polyval(b[::-1], x)

        if np.isfinite(num) and np.isfinite(den) and abs(den) > den_min:
            valid_vals.append(num / den)

    if not valid_vals:
        return np.nan, np.nan, 0


    
    vals = np.asarray(valid_vals, float)

    mean = np.mean(vals)
    lo, hi = np.quantile(vals, [0.16, 0.84])
    err = 0.5*(hi - lo)

    return mean, err

def plot_pade(
    ax,
    coeffs,
    errors,
    pade_cases,
    shift_mean=0.0,
    x=1.0,
    n_samples=3000,
    ms=3.5,
    lw=1.2,
    capsize=3,
):

    max_M = (len(coeffs) - 1) // 2
    results = {name: ([], []) for name, _, _, _ in pade_cases}

    for M in range(1, max_M + 1):
        for name, shift, _, _ in pade_cases:
            L = M + shift
            if L < 0 or L + M > len(coeffs) - 1:
                continue

            mean, err = get_pade_estimate(
                coeffs,
                errors,
                L=L,
                M=M,
                x=x,
                n_samples=n_samples
            )

            if not np.isfinite(mean):
                continue

            mean -= shift_mean

            results[name][0].append(L + M)
            results[name][1].append((mean, err))

    for name, _, marker, color in pade_cases:
        xs, ys, yerrs = [], [], []

        for xval, (m, e) in zip(*results[name]):
            xs.append(xval)
            ys.append(m)
            yerrs.append(e)

        ax.errorbar(
            xs,
            ys,
            yerr=yerrs,
            fmt=marker + ':',
            ms=ms,
            lw=lw,
            capsize=capsize,
            color=color,
            label=f'Padé {name}'
        )


fig = plt.figure(figsize=(9,5.5))
gs = GridSpec(
    nrows=2, ncols=2,
    width_ratios=[1.0, 1.0],
    height_ratios=[1.0, 1.0],
    hspace=0.0,
    wspace=0.05
)

ax_lt = fig.add_subplot(gs[0, 0])
ax_lb = fig.add_subplot(gs[1, 0], sharex=ax_lt)

gs_right = gs[:, 1].subgridspec(
    2, 1,
    height_ratios=[3, 1.0],
    hspace=0.05
)

ax_rt = fig.add_subplot(gs_right[0, 0])
ax_rb = fig.add_subplot(gs_right[1, 0], sharex=ax_rt)

ax_lt.errorbar(
    order,
    e_a,
    yerr=err_a,
    fmt='o',
    lw=1.5,
    ms=3.5,
    capsize=3,
    linestyle='None',
    label='PTQMC',
    color='#494949FF'
)

ax_lb.errorbar(
    order,
    e_b,
    yerr=err_b,
    fmt='o',
    lw=1.5,
    ms=3.5,
    capsize=3,
    linestyle='None',
    color='#494949FF'
)

pade_cases = [
    (r"[$L$|$L-1$]",  +1, 's', 'C3'),
    (r"[$L$|$L$]",     0, '^', 'C2'),
    (r"[$L$|$L+1$]",  -1, 'D', 'C0'),
]

plot_pade(
    ax_lt,
    de_a,
    derr_a,
    pade_cases,
    shift_mean=3.1
)

plot_pade(
    ax_lb,
    de_b,
    derr_b,
    pade_cases,
    shift_mean=-1.0
)

ax_lt.set_ylabel(r"$E_{\rm corr}$ [a.u.]", fontsize=17)
ax_lt.legend(frameon=False, loc='lower left', fontsize =12, ncols=2)
center_lt = 2.840432 - 2 - 1.1
ax_lt.set_ylim(center_lt - 10.5, center_lt + 10.5)
ax_lt.text(0.98, 0.97, r'$\mathbf{(a)}$ $g = -1.1$', transform=ax_lt.transAxes, 
           fontsize=18, verticalalignment='top', horizontalalignment='right')
ax_lt.axhline(y=center_lt, color='black', linestyle='--', linewidth=1)
ax_lt.tick_params(top=True, right=True, labelbottom=False, direction='in')
ax_lt.minorticks_on()
ax_lt.tick_params(which='minor', top=True, right=True, direction='in')
ax_lt.xaxis.set_major_locator(MultipleLocator(4))
ax_lt.xaxis.set_minor_locator(MultipleLocator(1))
ax_lt.yaxis.set_major_locator(MultipleLocator(3))
ax_lt.yaxis.set_minor_locator(MultipleLocator(0.6))

ax_lb.set_ylabel(r"$E_{\rm corr}$ [a.u.]", fontsize=17)
ax_lb.tick_params(top=True, right=True, direction='in')
ax_lb.minorticks_on()
ax_lb.tick_params(which='minor', top=True, right=True, direction='in')
center_lb = -4.053469 - 2 + 3.0
ax_lb.set_ylim(center_lb - 7.5, center_lb + 7.5)
line_exact = ax_lb.axhline(y=center_lb, color='black', linestyle='--', linewidth=1)

ax_lb.xaxis.set_major_locator(MultipleLocator(4))
ax_lb.xaxis.set_minor_locator(MultipleLocator(1))
ax_lb.yaxis.set_major_locator(MultipleLocator(3))
ax_lb.yaxis.set_minor_locator(MultipleLocator(0.6))

ax_lb.set_xlabel("Approximation order", fontsize=17)
ax_lb.set_xlim(1, 16)
ax_lb.set_ylim(center_lb - 7.5, center_lb + 7.5)
ax_lb.legend([line_exact], ['FCI'], frameon=False, loc='lower right', fontsize=15, ncols=1)

ax_lb.text(0.98, 0.97, r'$\mathbf{(b)}$ $g = +3.0$', transform=ax_lb.transAxes, 
           fontsize=18, verticalalignment='top', horizontalalignment='right')



COL = {
    "fci": "black",
    "fciqmc": "#d95f02",
    "adc": "#4c72b0",
    "imsrg": "#7570b3",
    "ucc": "#1b9e77",
}

ax_rt.plot(
    g_fci,
    e_fci,
    color=COL["fci"],
    lw=2.0,
    label="FCI",
    zorder=-1
)

ax_rt.plot(
    g_adc3_d,
    e_adc3_d,
    color=COL["adc"],
    linestyle=':',
    lw=1.8,
    label="ADC(3)-D",
    zorder=2
)

ax_rt.plot(
    g_magnus3,
    e_magnus3,
    color=COL["imsrg"],
    linestyle='--',
    lw=1.8,
    label="IMSRG(3)",
    zorder=1
)

mask_uccsd = g_uccsd >= -1.9
ax_rt.scatter(
    g_uccsd[mask_uccsd][::4],
    e_uccsd[mask_uccsd][::4],
    marker='^',
    s=25,
    color=COL["ucc"],
    zorder=3,
    label="UCCSD"
)

ax_rt.errorbar(
    g_fciqmc[::4],
    e_fciqmc[::4],
    yerr=e_fciqmc_error[::4],
    fmt='o',
    ms=4,
    mfc='none',
    mec=COL["fciqmc"],
    capsize=2,
    elinewidth=1.0,
    color=COL["fciqmc"],
    label="FCIQMC",
    zorder=5
)

ax_rt.errorbar(g_ptqmc, e_ptqmc, yerr=err_ptqmc, markersize=3.5, linestyle='None', color='#494949FF', marker='d', label='PTQMC(Resummed)')


ax_rt.set_ylabel(r"$E_{\mathrm{corr}}$ [a.u.]", fontsize=17)
ax_rt.yaxis.set_label_position("right")
ax_rt.yaxis.tick_right()

ax_rt.set_xlim(-1.3 + 1e-3, 1.3)
ax_rt.set_ylim(-1.0, 0.2)

ax_rt.tick_params(
    top=True, right=True, left=True, bottom=True,
    labelbottom=False, direction="in"
)
ax_rt.minorticks_on()
ax_rt.tick_params(which="minor", top=True, right=True, left=True, bottom=True, direction="in")

handles, labels = ax_rt.get_legend_handles_labels()
order = [labels.index(k) for k in [ "PTQMC(Resummed)","FCIQMC", "UCCSD", "IMSRG(3)", "ADC(3)-D", "FCI"]]
ax_rt.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    frameon=False,
    fontsize=13,
    loc="lower left",
    ncols=1
)
ax_rt.text(0.98, 0.97, r'$\mathbf{(c)}$', transform=ax_rt.transAxes, 
           fontsize=18, verticalalignment='top', horizontalalignment='right')
ax_rb.axhline(0.0, color="black", lw=1.0, alpha=0.6)

ax_rb.plot(
    g_fciqmc,
    e_fciqmc - np.interp(g_fciqmc, g_fci, e_fci),
    marker='o',
    markersize=4,
    linestyle='None',
    markerfacecolor='none',
    markeredgecolor=COL["fciqmc"],
    color=COL["fciqmc"]
)

ax_rb.plot(
    g_adc3_d,
    e_adc3_d - np.interp(g_adc3_d, g_fci, e_fci),
    linestyle=':',
    lw=1.6,
    color=COL["adc"]
)

ax_rb.plot(
    g_magnus3,
    e_magnus3 - np.interp(g_magnus3, g_fci, e_fci),
    linestyle='--',
    lw=1.6,
    color=COL["imsrg"]
)

ax_rb.plot(
    g_uccsd,
    e_uccsd - np.interp(g_uccsd, g_fci, e_fci),
    linestyle='-.',
    lw=1.6,
    color=COL["ucc"]
)

ax_rb.set_ylabel(r"$E_{\rm corr} - E_{\rm FCI}$ [a.u.]", fontsize=17)
ax_rb.yaxis.set_label_position("right")
ax_rb.yaxis.tick_right()

ax_rb.tick_params(
    top=True, right=True, left=True, bottom=True,
    direction="in"
)
ax_rb.minorticks_on()
ax_rb.tick_params(which="minor", top=True, right=True, left=True, bottom=True, direction="in")
ax_rb.set_ylim(-0.05, 0.06)
ax_rb.set_xlabel(r"$g$ [a.u.]", fontsize=17)

g_ptqmc = g_ptqmc[::-1]
e_ptqmc = e_ptqmc[::-1]

dick = [-0.30120868,
    -0.25956829,
    -0.22012986,
    -0.18307265,
    -0.14859912,
    -0.11693791,
    -0.08834681,
    -0.06311574,
    -0.04156931,
    -0.02406869,
    -0.01101221,
    -0.00283400,
     0.00000000,
    -0.00300043,
    -0.01233813,
    -0.02851239,
    -0.05199873,
    -0.08322572,
    -0.12255107,
    -0.17023984,
    -0.22644773,
    -0.29121205,
    -0.36445153,
    -0.44597484,
    -0.53549598,
    -0.63265404
]

err_ptqmc = err_ptqmc[::-1]

ax_rb.plot(
    g_ptqmc,
    e_ptqmc - dick,
    markersize = 3,
    linestyle = 'None',
    color='#494949FF',
    marker='d',
    lw=1.5
)

for ax in [ax_lt, ax_lb, ax_rt, ax_rb]:
    ax.tick_params(axis='both', which='major', labelsize=15)

plt.savefig("Figure_3.png", dpi=600, bbox_inches="tight")
