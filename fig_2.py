import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, NullFormatter, AutoMinorLocator
from matplotlib.legend_handler import HandlerErrorbar

from matplotlib.lines import Line2D


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


def _style_axes_manual(ax):
    if ax.get_xscale() == 'log':
        ax.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
        )
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    if ax.get_yscale() == 'log':
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.tick_params(which='major', direction='in', 
                   top=True, bottom=True, left=True, right=True,
                   length=6, width=0.8)
    
    ax.tick_params(which='minor', direction='in',
                   top=True, bottom=True, left=True, right=True,
                   length=3, width=0.5)

walkers = np.array([50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000], dtype=float)

E8 = np.array([
    14.218686, 12.390090, 12.741771, 12.219634, 12.395442,
    12.602056, 12.592321, 12.498192, 12.476767
], dtype=float)

err8 = np.array([
    1.943785e+00, 1.488261e+00, 5.904413e-01, 3.963057e-01,
    2.655297e-01, 2.004215e-01, 1.253650e-01, 1.014163e-01,
    5.771314e-02
], dtype=float)

E_ref = 12.49937419
mask = walkers >= 500
A = np.mean((err8[mask] / E_ref) * np.sqrt(walkers[mask]))

x_smooth = np.logspace(np.log10(walkers.min()), np.log10(walkers.max()), 300)
y_smooth = A * x_smooth**(-0.5)

g = np.array([
    -1.2, -1.19, -1.18, -1.17, -1.16, -1.15, -1.14, -1.13, -1.12, -1.11, -1.1, -1.09,
    -1.08, -1.07, -1.06, -1.05, -1.04, -1.03, -1.02, -1.01, -1.0, -0.99, -0.98, -0.97,
    -0.96, -0.95, -0.94, -0.93, -0.92, -0.91, -0.9, -0.89, -0.88, -0.87, -0.86, -0.85,
    -0.84, -0.83, -0.82, -0.81, -0.8, -0.79, -0.78, -0.77, -0.76, -0.75
], dtype=float)

mbpt4 = np.array([
    -1.2586449981778436, -1.1876950868407805, -1.1214521317606079, -1.0595710623656238,
    -1.0017343576836382, -0.9476496247867989, -0.8970474120972196, -0.8496792325679112,
    -0.805315774647557, -0.7637452814692085, -0.7247720809210207, -0.6882152512032422,
    -0.6539074081856828, -0.6216936023846338, -0.5914303147040085, -0.5629845412553789,
    -0.5362329586051242, -0.5110611617112735, -0.48736296762247644, -0.46503977872982816,
    -0.4440000000000000, -0.42415850518511755, -0.4054361475094077, -0.3877593107822257,
    -0.3710594972882002, -0.3552729491632949, -0.34034030028580453, -0.3262062559977772,
    -0.31281929822901056, -0.3001314138259139, -0.28809784409407957, -0.27667685374901874,
    -0.2658295176363952, -0.25551952373335585, -0.24571299107789546, -0.23637830139525606,
    -0.22748594330054894, -0.21900836805629487, -0.21091985595355567, -0.2031963924667779,
    -0.19581555340615964, -0.1887563983582301, -0.18199937176591474, -0.17552621105439759,
    -0.16931986125909582, -0.16336439565747574
])

mbpt8 = np.array([
    12.49937418503955, 10.746715147831774, 9.238528950659074, 7.940042465618278,
    6.821572538388459, 5.857754951333812, 5.026894715234552, 4.310417813737301,
    3.692407917344929, 3.159214373095826, 2.699120073753127, 2.3020597065474995,
    1.959380449294179, 1.6636384802338733, 1.408425745331642, 1.188222322180307,
    0.9982704650549517, 0.8344670371785523, 0.6932715552864326, 0.5716275056776516,
    0.4668949545496104, 0.37679278041562014, 0.2993491126094421, 0.23285877538934363,
    0.17584671867364543, 0.12703656952967446, 0.08532356781556993, 0.04975125867828911,
    0.019491407138883865, -0.006173321585255298, -0.027864305834675474, -0.04611986863379247,
    -0.061406421422956825, -0.07412809471961612, -0.084635065169544, -0.09323075871251207,
    -0.10017808419729635, -0.1057048300824297, -0.11000833829799905, -0.11325955345219896,
    -0.1156065319483699, -0.1171774838989128, -0.11808341069738004, -0.11842039249811753,
    -0.11827157244919961, -0.11770887815459163
])

mbpt12 = np.array([
    -265.20901573476425, -211.99993209306467, -169.58646027998543, -135.7489384012865,
    -108.73142525703534, -87.14303143960171, -69.88067450250624, -56.06850399478917,
    -45.0103215621352, -36.15214980059766, -29.05274085838824, -23.360307088666655,
    -18.794135602222948, -15.130042371418334, -12.188849413109896, -9.827245658235924,
    -7.930529976238889, -6.406842341549499, -5.182573129788102, -4.1987062674532725,
    -3.4079034852283066, -2.7721773765561637, -2.2610327682913822, -1.849980953771512,
    -1.5193510859324753, -1.2533386204334884, -1.0392430265965615, -0.8668567431479,
    -0.7279750909326097, -0.6160029929413793, -0.5256392283707756, -0.452622825625673,
    -0.39352928670925635, -0.3456067959786897, -0.30664452897643946, -0.27486674411916256,
    -0.24884759228617437, -0.2274425808694498, -0.20973343046147574, -0.19498370444520008,
    -0.18260310641929944, -0.17211875320016423, -0.16315206245391867, -0.1554001600696977,
    -0.14862092615592992, -0.14262097038490262
])

mbpt16 = np.array([
    6528.839827157325, 4831.101637489142, 3578.188350309551, 2652.491239154191,
    1967.8216698402869, 1460.9189036090388, 1085.2824592047689, 806.6840266389398,
    599.8954851503191, 446.29926158042025, 332.1401254442142, 247.24416889674833,
    184.0786710507358, 137.06112054614374, 102.05065321933958, 75.97325141782846,
    56.54517607552156, 42.068643190671786, 31.28070437652224, 23.241360144847853,
    17.250639013788284, 12.787087170247654, 9.462101447167475, 6.985998083504897,
    5.142783061729165, 3.771380120264411, 2.7516552077803285, 1.9940062677384915,
    1.4316051288065568, 1.0146134788169987, 0.7058691082425734, 0.47766776969880365,
    0.3093618512809311, 0.18556825617897, 0.09483080491336615, 0.02862184838532622,
    -0.01940290953932422, -0.053960546887889205, -0.07855590836055759, -0.09579119576701034,
    -0.10759686466042995, -0.1154037689314853, -0.12027144865761841, -0.1229836874746959,
    -0.12411965016019177, -0.1241068074579208
])

g_ptqmc = np.array([
    -1.20, -1.15, -1.10,
    -1.05, -1.00, -0.95, -0.90, -0.85, -0.80
])


ptqmc4 = np.array([
    -1.268100,
    -0.953447,
    -0.728281,
    -0.561001,
    -0.445205,
    -0.355505,
    -0.288668,
    -0.237359,
    -0.196548
])

err4 = np.array([
    5.833800e-03,
    4.342570e-03,
    3.296558e-03,
    2.720347e-03,
    2.099937e-03,
    1.428909e-03,
    1.152788e-03,
    1.033087e-03,
    8.150466e-04
])


ptqmc8 = np.array([
    12.638999,
     5.930733,
     2.729639,
     1.177875,
     0.466107,
     0.125731,
    -0.028409,
    -0.093977,
    -0.115966
])

err8p = np.array([
    1.214073e-01,
    6.095963e-02,
    3.072890e-02,
    1.831570e-02,
    9.888675e-03,
    4.851330e-03,
    2.557041e-03,
    1.662980e-03,
    9.905511e-04
])

ptqmc12 = np.array([
    100.0,
    100.0,
    100.0,
    -9.758858,
    -3.420467,
    -1.248972,
    -0.529326,
    -0.277878,
    -0.184432
])

err12 = np.array([
    0.0, 0.0, 0.0,
    1.601506e-01,
    5.524773e-02,
    1.704979e-02,
    6.082725e-03,
    2.144064e-03,
    1.130875e-03
])


ptqmc16 = np.array([
    100.0,
    100.0,
    100.0,
    75.669454,
    17.416879,
     3.782640,
     0.714392,
     0.032240,
    -0.107499
])

err16 = np.array([
    0.0, 0.0, 0.0,
    1.639520e+00,
    3.850294e-01,
    8.363223e-02,
    2.037935e-02,
    5.330891e-03,
    1.377056e-03
])

fig = plt.figure(figsize=(8, 4))
gs = GridSpec(2, 2, width_ratios=[1.0, 1.0], hspace=0.0, wspace=0.08)

axL1 = fig.add_subplot(gs[0, 0])
axL2 = fig.add_subplot(gs[1, 0], sharex=axL1)
axR  = fig.add_subplot(gs[:, 1])
axL1.tick_params(labelbottom=False)

c_dot = "#494949FF"
c_line = 'gray'

h_mbpt = axL1.errorbar(
    walkers, E8, yerr=err8,
    fmt='D:', capsize=3, markersize=4,
    color=c_dot, zorder=0
)

h_exact = axL1.axhline(
    E_ref, ls='-', color=c_line, zorder=0, linewidth=1.2
)

axL1.set_xscale("log")
axL1.set_ylabel(r"$E_{\rm corr}$ [a.u.]", fontsize=15)
axL1.text(0.97, 0.05, "(a)", transform=axL1.transAxes,
          ha='right', va='bottom', fontweight='bold', fontsize=15)

axL1.legend(
    handles=[h_mbpt, h_exact],
    labels=['PTQMC(8)', 'MBPT(8)'],
    handler_map={type(h_mbpt): HandlerErrorbar()},
    fontsize=13.5,
    loc='upper right',
    frameon=False
)

axL1.tick_params(axis='y', labelsize=13.5)
axL2.tick_params(axis='x', labelsize=13.5)
axL2.tick_params(axis='y', labelsize=13.5)


h_err = axL2.errorbar(
    walkers, err8 / E_ref,
    fmt='d', capsize=3, markersize=6,
    color=c_dot, linestyle='None'
)

h_fit, = axL2.plot(
    x_smooth, y_smooth,
    '--', color=c_line
)

axL2.set_xscale("log")
axL2.set_yscale("log")
axL2.set_xlabel("Number of walkers", fontsize=15)
axL2.set_ylabel("Relative error", fontsize=15)
axL2.text(0.97, 0.93, "(b)", transform=axL2.transAxes,
          ha='right', va='top', fontweight='bold', fontsize=15)

axL2.text(0.8, 0.65, r"$g=-1.2$", transform=axL2.transAxes,
          ha='center', va='center', fontsize=15)

axL2.legend(
    handles=[h_err, h_fit],
    labels=['Error', r'$\propto N_w^{-1/2}$'],
    fontsize=13.5,
    loc='lower left',
    frameon=False
)

axR.plot(g, mbpt4,  lw=1.5, color='C0')
axR.plot(g, mbpt8,  lw=1.5, color='C1')
axR.plot(g, mbpt12, lw=1.5, color='C2')
axR.plot(g, mbpt16, lw=1.5, color='C3')

axR.errorbar(g_ptqmc, ptqmc4,  yerr=err4,  fmt='o', capsize=4, color='C0', markersize=5)
axR.errorbar(g_ptqmc, ptqmc8,  yerr=err8p, fmt='o', capsize=4, color='C1', markersize=5)
axR.errorbar(g_ptqmc, ptqmc12, yerr=err12, fmt='o', capsize=4, color='C2', markersize=5)
axR.errorbar(g_ptqmc, ptqmc16, yerr=err16, fmt='o', capsize=4, color='C3', markersize=5)

axR.set_xlabel(r"$g$ [a.u.]", fontsize=15)
axR.set_ylabel(r"$E_{\rm corr}$ [a.u.]", fontsize=15)
axR.set_ylim([-2, 2])

axR.yaxis.tick_right()
axR.yaxis.set_label_position("right")
axR.tick_params(axis='y', labelleft=False, labelright=True,
                left=True, right=True, labelsize=13.5)
axR.tick_params(axis='x', labelsize=13.5)
axR.spines['left'].set_visible(True)

axR.text(0.02, 0.97, "(c)", transform=axR.transAxes,
         ha='left', va='top', fontweight='bold', fontsize=15)

axR.text(0.08, 0.6, r"$N_w=10^4$", transform=axR.transAxes,
         ha='left', va='center', fontsize=15)

legend_top = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='black', markersize=5,
           linestyle='None', label='PTQMC'),
    Line2D([0], [0], color='black', lw=1.5, label='MBPT')
]

legend_bottom = [
    Line2D([0], [0], color='C0', lw=6, label='4th order'),
    Line2D([0], [0], color='C1', lw=6, label='8th order'),
    Line2D([0], [0], color='C2', lw=6, label='12th order'),
    Line2D([0], [0], color='C3', lw=6, label='16th order'),
]

leg1 = axR.legend(handles=legend_top, loc='upper right',
                  fontsize=12, frameon=False)
leg2 = axR.legend(handles=legend_bottom, loc='lower right',
                  fontsize=12, frameon=False)
axR.add_artist(leg1)

_style_axes_manual(axL1)
_style_axes_manual(axL2)
_style_axes_manual(axR)

plt.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.12, hspace=0.0, wspace=0.08)

plt.savefig("Figure_2.png", dpi=600, bbox_inches='tight')