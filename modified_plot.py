import numpy as np
import corner
import itertools

import astropy.units as u
import astropy.constants as consts
from astropy.time import Time
from astropy.visualization import hist

from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import rc
import matplotlib as mpl

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 600
rc("text", usetex=True)
rc("font", size=14)
rc("legend", fontsize=13)
plt.rcParams[
    "text.latex.preamble"
] = r"""\usepackage{bm} \usepackage{amsmath} \usepackage{cmbright}
	\usepackage{siunitx} \usepackage{lmodern} \usepackage{sfmath} \sffamily \boldmath"""
from matplotlib import rcParams
import matplotlib.font_manager as font_manager

from erfa import ErfaWarning

# define modified color map for default use in orbit plots
plasma_modified = cm.get_cmap("plasma", 10_000)
cmp = ListedColormap(plasma_modified(np.linspace(0.0, 0.88, 10_000)))
cmap = cmp

my_colors = [
    "#0d0887",
    "#2c0594",
    "#43039e",
    "#5901a5",
    "#6e00a8",
    "#8305a7",
    "#9511a1",
    "#a72197",
    "#b6308b",
    "#c5407e",
    "#d14e72",
    "#dd5e66",
    "#e76e5b",
    "#f07f4f",
    "#f79044",
    "#fca338",
    "#feb72d",
    "#fccd25",
]


def y_jump_hist(orb_element_y, cutoff=None, xlim=None):
    """
    Make a histogram of the jump in histogram y-values for an orbital element dist.
    """

    orb_element_y_diffs = np.abs(np.diff(orb_element_y))

    fig, ax = plt.subplots()

    orb_element_y_diff_y, orb_element_y_diff_x, _ = hist(
        orb_element_y_diffs,
        bins=200,
        density=True,
        histtype="stepfilled",
        color="#db5c68",
    )
    # middle (not edge) values of i
    orb_element_y_diff_x_mid = moving_average(orb_element_y_diff_x, 2)

    orb_element_y_diff_peak = orb_element_y_diff_y.max()
    orb_element_y_diff_peak_i = int(
        np.where(orb_element_y_diff_y == orb_element_y_diff_peak)[0][0]
    )
    orb_element_y_diff_peak_x = float(
        orb_element_y_diff_x_mid[orb_element_y_diff_peak_i]
    )
    print(f"Peak y-diff in distribution: {orb_element_y_diff_peak_x}")

    ax.axvline(x=orb_element_y_diff_peak_x, color="k", linestyle="--", lw=1)

    ax.axvline(x=cutoff, color="blue", linestyle="-.", lw=1)

    if xlim is not None:
        ax.set_xlim(xlim)

    plt.show()

    return None


def equal_pair_is(orb_element_peak_i, orb_element_y, eps=0.01):
    """
    Find the pairs of indices for equal histogram y-values to 0.5% of max.
    histogram y-value.
    """

    # try this out? It should be relative.
    orb_element_y_eq_eps = eps

    orb_element_y_eq_pair_is = (
        []
    )  # initialize to add indices corresponding to the left side of a pair

    orb_element_i = 0  # initialize here
    while orb_element_i < orb_element_peak_i:  # loop from the left to the peak

        orb_element_y1 = orb_element_y[
            orb_element_i
        ]  # orb_element_i many to the left of the end

        orb_element_i_right = (
            orb_element_y.size - 1
        )  # starts at zero, so size -1 is the last index
        # now, loop from the end to the peak (stopping once we find a match to the left y-value)
        searching = True  # start by not having found a match
        hit_peak = False  # don't search too far!
        while (searching == True) and (hit_peak == False):

            orb_element_y2 = orb_element_y[
                orb_element_i_right
            ]  # orb_element_i_right starts at the far end, in this case

            # if close enough to = (abs(diff) < eps)
            if np.abs(orb_element_y1 - orb_element_y2) < orb_element_y_eq_eps:
                # append to the list of indices to keep for these equal pairs
                orb_element_y_eq_pair_is.append(
                    np.array([orb_element_i, orb_element_i_right])
                )
                searching = False  # stop searching — we found the match!

            # stop if we hit the peak from the right
            elif orb_element_i_right == orb_element_peak_i:
                hit_peak = True

            orb_element_i_right -= (
                1  # move to the left one after checking the above conditions
            )

        orb_element_i += 1  # increment the "while" loop

    orb_element_y_eq_pair_is = np.array(
        orb_element_y_eq_pair_is
    )  # convert list => array

    return orb_element_y_eq_pair_is


def best_orb_element_bounds(orb_element_x, orb_element_y, orb_element_y_eq_pair_is):
    """
    Find the closest pair of equal_pair_is to enclosing 68% probability.
    """

    # size of each histogram bin in the x-direction
    orb_element_x_width = np.diff(orb_element_x)[
        orb_element_x.size // 2
    ]  # choose the middle-most diff

    # calculate the areas between each pair (probabilities)
    area_orb_element_eq_pairs = np.array(
        [
            np.sum(orb_element_x_width * orb_element_y[pair[0] : pair[1]])
            for pair in orb_element_y_eq_pair_is
        ]
    )

    # difference in enclosed probabilities from 68%
    orb_element_area_diff_68 = np.abs(area_orb_element_eq_pairs - 0.68)
    closest_orb_element_area_diff = np.min(
        orb_element_area_diff_68
    )  # smallest difference

    second_closest_orb_element_area_diff = np.partition(orb_element_area_diff_68, 1)[1]
    third_closest_orb_element_area_diff = np.partition(orb_element_area_diff_68, 2)[2]

    # find which pair is at the difference minimum
    orb_element_i_pair_close_to_68 = np.where(
        orb_element_area_diff_68 == closest_orb_element_area_diff
    )[0][0]
    best_orb_element_bounds = orb_element_y_eq_pair_is[orb_element_i_pair_close_to_68]

    second_orb_element_i_pair_close_to_68 = np.where(
        orb_element_area_diff_68 == second_closest_orb_element_area_diff
    )[0][0]
    second_best_orb_element_bounds = orb_element_y_eq_pair_is[
        second_orb_element_i_pair_close_to_68
    ]

    third_orb_element_i_pair_close_to_68 = np.where(
        orb_element_area_diff_68 == third_closest_orb_element_area_diff
    )[0][0]
    third_best_orb_element_bounds = orb_element_y_eq_pair_is[
        third_orb_element_i_pair_close_to_68
    ]
    print()
    print(
        f"Index of best sma index pair in orb_element_y_eq_pair_is: {orb_element_i_pair_close_to_68}"
    )

    print(f"With difference from 68% of: {closest_orb_element_area_diff:.1%}")

    print()
    print(f"These bounds are: {repr(best_orb_element_bounds)}")

    print()
    print(
        f"\nIndex of second best sma index pair in orb_element_y_eq_pair_is: {second_orb_element_i_pair_close_to_68}"
    )

    print(f"With difference from 68% of: {second_closest_orb_element_area_diff:.1%}")

    print()
    print(f"These bounds are: {repr(second_best_orb_element_bounds)}")

    print()
    print(
        f"\nIndex of third best sma index pair in orb_element_y_eq_pair_is: {third_orb_element_i_pair_close_to_68}"
    )

    print(f"With difference from 68% of: {third_closest_orb_element_area_diff:.1%}")

    print()
    print(f"These bounds are: {repr(third_best_orb_element_bounds)}")

    return best_orb_element_bounds[0], best_orb_element_bounds[1]


def moving_average(x, w):
    """
    moving average of array 'x' with box size 'w'
    """
    return np.convolve(x, np.ones(w), "valid") / w


def gaussian(x, A, mu, var, b, a):
    """
    Gaussian distribution with a linear offset ax + b
    """
    G = A / (np.sqrt(var) * np.sqrt(2 * np.pi)) * np.exp((-1 / 2) * (x - mu) ** 2 / var)
    return G + a * x + b


def configure_matplotlib(dpi, size_fac):
    if not hasattr(configure_matplotlib, "configured"):
        plt.style.use("default")  # Reset to defaults once
        rc("text", usetex=True)
        rc("font", size=14 * size_fac)
        rc("legend", fontsize=14 * size_fac)

        # Combine latex preamble settings
        plt.rcParams.update(
            {
                "figure.dpi": dpi,
                "savefig.dpi": dpi,
                "text.latex.preamble": r"\usepackage{bm} \usepackage{amsmath} \usepackage{cmbright}"
                r"\usepackage{siunitx} \usepackage{lmodern} \usepackage{sfmath}"
                r"\sffamily \boldmath",
                "xtick.major.pad": 6 * size_fac,
                "ytick.major.pad": 6 * size_fac,
            }
        )
        configure_matplotlib.configured = True


def element_hist(
    element,
    element_str,
    var,
    unit,
    xlabel,
    annotation_str,
    i_bounds,
    bins="knuth",
    xlim=None,
    plot_jump_hist=True,
    fit_peak=False,
    plot_fit=False,
    fit_cut_fracs=(0.9, 1.1),
    ann_x_frac=0.35,
    param=None,
    color="blue",
    obj="object",
    sampler="sampler",
    annotation_y_frac=0.75,
    jump_cut_frac=0.2,
    alpha=1,
    dpi=300,
    ann_nom_add=0,
    size_fac=2,  # Added size factor parameter
    show_xlabel=True,
    show_ylabel=True,
):
    # Configure matplotlib with consistent styling
    configure_matplotlib(dpi, size_fac)

    fig, ax = plt.subplots(dpi=dpi, figsize=(10, 10))

    # Set base font sizes
    base_fontsize = 14 * size_fac
    tick_fontsize = 14 * size_fac
    label_fontsize = 14 * size_fac
    annotation_fontsize = 14 * size_fac

    # Update font sizes for all text elements
    plt.rcParams.update(
        {
            "font.size": base_fontsize,
            "axes.labelsize": label_fontsize,
            "axes.titlesize": label_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
        }
    )

    # Pre-compute array operations
    if xlim is None:
        xlim = (np.min(element), np.max(element))

    # Calculate margins to accommodate labels while maintaining square plot
    left_margin = 0.18 if show_ylabel else 0.12
    right_margin = 0.05
    bottom_margin = 0.18 if show_xlabel else 0.12
    top_margin = 0.05

    # Set margins to maintain square plotting area
    plt.subplots_adjust(
        left=left_margin,
        right=1 - right_margin,
        bottom=bottom_margin,
        top=1 - top_margin,
    )

    print(f"element: {element}")
    print(f"bins: {bins}")
    # Plot histogram
    y, x, _ = hist(
        element,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=color,
        alpha=alpha,
    )

    bins = y.size

    # Vectorized computation of bin centers
    x_mid = (x[1:] + x[:-1]) / 2

    # Find peak using numpy operations
    peak = y.max()
    peak_i = np.argmax(y)
    peak_x_mid = x_mid[peak_i]

    # Initialize fit parameters
    if param is None:
        param = np.array([np.max(y) * 1000, peak_x_mid, peak_x_mid / 2, -0.1, -0.0005])

    if fit_peak:
        # Vectorized domain cutting for fitting
        cut_mask = np.logical_and(
            x_mid > peak_x_mid * fit_cut_fracs[0], x_mid < peak_x_mid * fit_cut_fracs[1]
        )
        cut_x_mid = x_mid[cut_mask]
        cut_y = y[cut_mask]

        def gaussian(x, A, mu, sigma, b, a):
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + b + a * x

        param, cov = curve_fit(gaussian, cut_x_mid, cut_y, p0=param)

        if plot_fit:
            finer_cut_x_mid = np.linspace(cut_x_mid[0], cut_x_mid[-1], 1000)
            finer_gaussian = gaussian(finer_cut_x_mid, *param)
            ax.plot(
                finer_cut_x_mid,
                finer_gaussian,
                ls="-",
                color="green",
                lw=1.5 * size_fac,
                label=r"\textbf{Gaussian Fit}",
                alpha=0.6,
            )

        peak_fit_x_mid = param[1]
        peak_i = np.argmin(np.abs(peak_fit_x_mid - x_mid))
        peak_x_mid = peak_fit_x_mid

    # Unpack bounds
    low_i, high_i = i_bounds
    low = x_mid[low_i]
    high = x_mid[high_i]

    # Compute areas
    bin_width = np.diff(x)[0]
    area_high = np.sum(y[peak_i:high_i]) * bin_width
    area_low = np.sum(y[low_i:peak_i]) * bin_width

    # Calculate errors
    err_plus = high - peak_x_mid
    err_minus = peak_x_mid - low

    # Line width scaling
    base_lw = 3 * size_fac

    # Add vertical lines with consistent styling
    ax.axvline(x=peak_x_mid, color="k", linestyle="--", lw=base_lw)
    ax.axvline(x=low, color="k", linestyle=":", lw=base_lw * 0.75)
    ax.axvline(x=high, color="k", linestyle=":", lw=base_lw * 0.75)

    # Get ylim and maintain square aspect ratio
    ax_ylim = ax.get_ylim()
    ax.set_aspect(abs(xlim[1] - xlim[0]) / abs(ax_ylim[1] - ax_ylim[0]))

    # Add annotation with consistent font size
    ann_x = xlim[0] + ann_x_frac * (xlim[1] - xlim[0])
    ax.annotate(
        annotation_str.format(peak_x_mid + ann_nom_add, err_plus, err_minus),
        (ann_x, ax_ylim[1] * annotation_y_frac),
        fontweight="bold",
        fontsize=annotation_fontsize,
    )

    # Configure labels with proper padding
    label_pad = 14 * size_fac
    if show_xlabel:
        ax.set_xlabel(xlabel, labelpad=label_pad, fontsize=label_fontsize)
    else:
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(
            r"\textbf{Probability Density}", labelpad=label_pad, fontsize=label_fontsize
        )
    else:
        ax.tick_params(labelleft=False)

    # Update spine and tick parameters
    spine_width = 2 * size_fac * 1.5
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    ax.tick_params(
        axis="both",
        which="major",
        width=2 * size_fac * 1.5,
        length=8 * size_fac * 1.25,
        direction="in",
        labelsize=tick_fontsize,
    )

    # Save figure with consistent parameters
    plt.savefig(
        f"{obj}_{element_str}_histogram_{sampler}.jpg",
        facecolor="w",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.1 * size_fac,
    )

    return param if fit_peak else None
