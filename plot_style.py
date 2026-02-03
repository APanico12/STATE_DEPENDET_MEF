import matplotlib as mpl
import matplotlib.pyplot as plt
def set_publication_style( use_latex=False):
        """
        Apply a publication-quality Matplotlib style.
        Optimized for figures intended for journals (Nature, Science, PNAS, etc.).

        Parameters
        ----------
        use_latex : bool, optional
            If True, enables LaTeX rendering for text and math (requires local LaTeX install).
        """

        # 1️⃣ BASE STYLE (clean & minimal)
        plt.style.use('seaborn-v0_8-white')

        # 2️⃣ FONT SETTINGS — Helvetica or Arial preferred by most journals
        font_params = {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'mathtext.fontset': 'stixsans',  # Sans-serif math look
        }

        # 3️⃣ OPTIONAL: LATEX RENDERING (for math-heavy papers)
        if use_latex:
            font_params.update({
                'text.usetex': True,
                'font.family': 'serif',
                'text.latex.preamble': (
                    r'\usepackage{amsmath} \usepackage{helvet} '
                    r'\usepackage{sansmath} \sansmath'
                )
            })

        # 4️⃣ AXES & GRID STYLE
        axes_params = {
            'axes.edgecolor': 'black',
            'axes.linewidth': 0.8,
            'axes.grid': False,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }

        # 5️⃣ TICKS — fine-tuned for print readability
        tick_params = {
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
        }

        # 6️⃣ COLORBLIND-FRIENDLY PALETTE (Okabe–Ito)
        cb_palette = [
            '#E69F00', '#56B4E9', '#009E73', '#F0E442',
            '#0072B2', '#D55E00', '#CC79A7', '#000000'
        ]

        # 7️⃣ APPLY ALL STYLE SETTINGS
        mpl.rcParams.update(font_params)
        mpl.rcParams.update(axes_params)
        mpl.rcParams.update(tick_params)
        mpl.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
            'axes.prop_cycle': plt.cycler('color', cb_palette),
        })