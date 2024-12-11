import numpy as np  # numpy
from arviz import hdi  # arviz
from functools import lru_cache  # functools
import gc  # garbage collection
from astropy.visualization import hist  # astropy
import cv2  # openCV

# scipy
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy import linalg as la
from scipy.interpolate import RegularGridInterpolator

# KDEpy
from KDEpy import FFTKDE
from KDEpy.bw_selection import improved_sheather_jones, silvermans_rule

# numba
from numba import jit, prange
from numba_progress import ProgressBar

# matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
from matplotlib import rc
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

rcParams.update(mpl.rcParamsDefault)
rc("text", usetex=True)
plt.rcParams[
    "text.latex.preamble"
] = r"""\usepackage{bm} \usepackage{amsmath} \usepackage{cmbright}
    \usepackage{siunitx} \usepackage{lmodern} \usepackage{sfmath} \sffamily \boldmath"""
from matplotlib import rcParams
import matplotlib.font_manager as font_manager

fontleg = font_manager.FontProperties(weight="bold")  # bold legends
import matplotlib.collections as mcoll
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize


@jit(nopython=True)
def point_in_polygon(x, y, poly):
    """
    Checks if a 2-D point at coordinate (x, y) is inside of the polygon "poly"
    poly is a reshaped contour from pypolot.contour
    uses the JIT compiler from numba for speed

    returns 1.0 if inside, -1.0 if outside

    """
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]

    # loop though the polygon boundaries, comparing the point (x, y) to each
    # checks are performed in y (min, max) and then in x (max, min)
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside

        p1x, p1y = p2x, p2y

    return 1.0 if inside else -1.0  # Return 1 if inside, -1 if outside


def determine_hole_sums(mpl_contours):
    """
    Loop through pyplot contours to determine if they are within another,
    larger contour (i.e., a hole, which may have scatter points shown inside)

    mpl_contours is a list of reshaped output from pyplot.contour.allsegs[0] output
    """
    hole_sums = np.zeros(len(mpl_contours))

    # loop though contiours
    for i, contour in enumerate(mpl_contours):
        test_point = tuple(contour[0].flatten())  # test the first point in the contour

        # loop though all of the other contours and use pointPolygonTest to determine containment
        for j, other_contour in enumerate(mpl_contours):
            if i != j:
                hole_sums[i] += cv2.pointPolygonTest(other_contour, test_point, False)

    print(f"\nhole_sums: {hole_sums}\n")
    return hole_sums


@jit(nopython=True, parallel=True)
def check_points_outside(x, y, contours, hole_sums, n_points, progress_proxy):
    """
    Use point_in_polygon and the hole_sums determined for all the contours to
    establish which scatter points should be plotted (i.e., only those below the first contour)

    x and y are arrays of point coordinates, contours are reshaped from mpl_contours (list of pypolot.contour.allsegs[0] output)

    n_points = len(x) = len(y)

    progress_proxy is used to have a tqdm-like progress bar while using numba, updated after checking each
    scatter point with point_in_polygon

    returns an array of "outside" values for each point (x,y) - indicating how many contours it is outside
    or inside of. We will only plot points outside all contours, or within a hole below the first contour.
    """
    n_contours = len(contours)
    print(f"n_contours: {n_contours}")
    outside = np.zeros((n_points,))
    is_hole = hole_sums != -1.0 * (n_contours - 1)

    for k in prange(n_points):
        point_status = np.zeros(n_contours)
        for j in range(n_contours):
            point_status[j] = point_in_polygon(x[k], y[k], contours[j])

        # Point is outside all contours
        if np.all(point_status < 0):
            outside[k] = -1.0 * n_contours
        # Point is inside a hole
        elif np.any((point_status > 0) & is_hole):
            outside[k] = -1.0 * n_contours
        # Point is inside a non-hole contour
        else:
            outside[k] = np.sum(point_status)

        progress_proxy.update(1)

    return outside


def optimized_contour_point_check(element_x, element_y, mpl_contours):
    """
    use check_points_outside for 2-D plot of element_x vs. element_y,
    given mpl_contours from pyplot.contour.allsegs[0]
    returns arrays to scatter (x_outside, y_outside)
    """

    print("Finding scatter points outside first contour...")

    # Determine hole sums
    hole_sums = determine_hole_sums(mpl_contours)

    # Convert contours to the format expected by Numba
    contours = [c.reshape(-1, 2) for c in mpl_contours]

    # Check points against contours
    n_points = element_x.size
    with ProgressBar(total=n_points) as progress:
        outside = check_points_outside(
            element_x, element_y, contours, hole_sums, n_points, progress
        )

    # Apply mask
    print(f"np.max(outside): {np.max(outside)}")
    print(f"np.min(outside): {np.min(outside)}\n")
    mask = outside == -1.0 * len(mpl_contours)
    x_outside = element_x[mask]
    y_outside = element_y[mask]

    print(f"Number of points outside: {np.sum(mask)}")

    return x_outside, y_outside


# from https://github.com/mwaskom/seaborn/blob/master/seaborn/distributions.py
def quantile_to_level(densities, quantile):
    """
    Return densities levels corresponding to quantile cuts of mass.
    i.e., input 'z' for a contour plot as densities, and quantile % of volume within it.
    For a 2-D standard normal, quantiles are 1 - exp(-num_sigma^2 / 2)
    """
    isoprop = np.asarray(quantile)

    values = np.ravel(densities)
    sorted_values = np.sort(values)[::-1]
    normalized_values = np.cumsum(sorted_values) / values.sum()

    idx = np.searchsorted(normalized_values, 1 - isoprop)
    levels = np.take(sorted_values, idx, mode="clip")

    return levels


def fftkde_cred_int(
    data_vec,  # sample
    param_min,  # min value of parameter distributed in data_vec
    param_max,  # max value of parameter distributed in data_vec
    fftkde_res=2**25,  # # of points to evaluate FFTKDE at along each dimension
    hdi_pc=68.0,  # percent of samples to contain in the highest-density credible interval
    min_mirror=False,  # mirror data about the minimum to force d/dx = 0 at boundary
    max_mirror=False,  # mirror data about the maximum to force d/dx = 0 at boundary
    circ_param=False,  # repeat data for circular parameters (e.g., angles on [0, 2pi])
    fit_log=False,  # fit to the base-10 logarithm of the data (e.g., log-normal distribution)
    bw_rule="silvermans",  # plug-in bandwidth selector: "silvermans" or "isj"
    interp_pc=5.0,  # what percent of points near the max-a-posteriori peak to interpolate
):
    """
    computes highest-density credible intervals using FFTKDE to find the max-a-posteriori peak
    in the distribution data_vec. Flexibility to mirror or repeat data depending on
    boundary conditions for the parameter.

    fftkde_res must be a power of 2
    """
    # Convert input to numpy array if not already
    data_vec = np.asarray(data_vec)

    # Early exit for empty or invalid input
    if len(data_vec) == 0:
        raise ValueError("Empty input data")

    # Pre-compute log transform if needed
    if fit_log:
        print("fitting log")
        fit_data_vec = np.log10(data_vec)
        fit_param_min = (
            np.log10(param_min) if param_min > 0 else np.min(fit_data_vec) - 0.3
        )
        fit_param_max = np.log10(param_max)
    else:
        fit_data_vec = data_vec
        fit_param_min = param_min
        fit_param_max = param_max

    # Standardize the data
    data_mean = np.mean(fit_data_vec)
    data_std = np.std(fit_data_vec, ddof=1)
    fit_data_vec_c = (fit_data_vec - data_mean) / data_std

    # Bandwidth selection (cached for repeated calls with similar data)
    @lru_cache(maxsize=128)
    def get_bandwidth(data_tuple, rule):
        data_array = np.array(data_tuple)
        if rule == "isj":
            return improved_sheather_jones(data_array[:, np.newaxis])
        return silvermans_rule(data_array[:, np.newaxis])

    # Convert data to tuple for caching
    data_tuple = tuple(fit_data_vec_c)
    bw = get_bandwidth(data_tuple, bw_rule)

    # Efficient mirroring using pre-allocated arrays
    bc_count = 0
    if min_mirror or max_mirror or circ_param:
        arrays_to_concat = [fit_data_vec]

        if min_mirror:
            arrays_to_concat.append(2 * fit_param_min - fit_data_vec)
            bc_count += 1
        if max_mirror:
            arrays_to_concat.append(2 * fit_param_max - fit_data_vec)
            bc_count += 1
        if circ_param:
            param_diff = param_max - param_min
            arrays_to_concat.extend(
                [fit_data_vec - param_diff, fit_data_vec + param_diff]
            )
            bc_count += 2

        new_fit_data_vec = np.concatenate(arrays_to_concat)
        fftkde_res *= 2 ** (bc_count)

    else:  # no mirroring
        new_fit_data_vec = np.copy(fit_data_vec)

    # standardize the mirrored data
    new_fit_data_vec_c = (new_fit_data_vec - data_mean) / data_std

    # Compute KDE with optimized resolution for the standardized, non-mirrored data
    kdex, kdey = (
        FFTKDE(bw=bw, kernel="gaussian").fit(new_fit_data_vec_c).evaluate(fftkde_res)
    )

    # return to mirrored, non-standardized
    kdex = kdex * data_std + data_mean

    # Use boolean indexing for filtering
    valid_mask = (kdex > fit_param_min) & (kdex < fit_param_max)
    kdex = kdex[valid_mask]
    kdey = kdey[valid_mask]

    # Transform back to linear scale if needed
    if fit_log:
        kdex = 10**kdex
        kdey /= kdex  # Vectorized division

    # Normalize using Simpson's rule
    kdey /= simpson(kdey, x=kdex)

    # Find peak efficiently
    peak_idx = np.argmax(kdey)
    fftkde_peak = kdex[peak_idx]

    # Optimize interpolation
    interp_min = fftkde_peak * (1 - interp_pc / 100)
    interp_max = fftkde_peak * (1 + interp_pc / 100)
    interp_mask = (kdex >= interp_min) & (kdex <= interp_max)
    interp_len = min(
        1000, 100 * np.sum(interp_mask)
    )  # Cap maximum interpolation points

    # Efficient interpolation
    interpx = (
        np.logspace(np.log(interp_min), np.log(interp_max), interp_len, base=np.e)
        if fit_log
        else np.linspace(interp_min, interp_max, interp_len)
    )

    nat_cs = CubicSpline(kdex[interp_mask], kdey[interp_mask], bc_type="natural")
    nat_csy = nat_cs(interpx)

    # Calculate final results
    central_val = interpx[np.argmax(nat_csy)]
    param_hdi = hdi(data_vec, hdi_pc / 100.0)

    # max-a-posteriori value, uncertainty in either direction for cred. interval,
    # arrays of kde points to plot, the interpolated points in x, and their points in y
    return (
        central_val,
        param_hdi[1] - central_val,  # uncert_plus
        central_val - param_hdi[0],  # uncert_minus
        kdex,
        kdey,
        interpx,
        nat_csy,
    )


def fftkde_plot(
    element,
    element_params,
    color="k",
    obj="object",  # another label for the output image filename
    sampler="sampler",  # this just adds an extra label to the output image filename
    annotation_y_frac=0.75,
    alpha=1,
    dpi=150,
    fftkde_res=2**25,
    hdi_pc=68.0,
    bw_rule="silvermans",
    plot_log=False,
    lw=2,
    interp_pc=5.0,
    fill_alpha=0.2,
    save=True,
    size_fac=4,  # scale plot size
    annotate=True,  # hdi annotated on plot
    show_xlabel=True,  # label x-axis
    show_ylabel=True,  # label y-axis
    jpeg=True,  # jpeg? otherwise, png created
    base_margin=0.5,  # these are padding for the plot
    label_margin=1.96875,
    pad_inches=0.0,
    base_size=10,
    ylab_offset=-0.175,
    ann_title=False,  # put the hdi as a title?
    ann_str=None,  # string format for hdi (e.g., with units)
    plot_hist=False,  # histogram under KDE for 1-D plot?
    hist_bins=400,
    hist_color="#fca338",
    no_ticks=True,  # tick marks?
    no_tick_labels=True,  # label ticks?
    calc_kde=True,  # do you want the KDE, or just a histogram?
):
    """
    creates a 1-D FFTKDE plot for parameter "element"
    many options available as kwargs, but default values are provided.
    """

    base_margin *= size_fac / 2.75
    label_margin *= size_fac / 2.75
    ylab_offset *= size_fac / 2.75

    # Calculate base figure size to achieve 1327 pixels
    base_size_inches = base_size

    # Create figure with calculated dimensions
    fig = plt.figure(dpi=dpi)

    width_inches = base_size + label_margin * int(show_ylabel) + 2 * base_margin
    if no_tick_labels:
        width_inches -= 0.5 * label_margin * int(show_ylabel)
    height_inches = (
        base_size + (0.65) * label_margin * int(show_xlabel) + 2 * base_margin
    )

    # add some extra room for the title
    if ann_title:
        height_inches += 0.25 * label_margin
        v = [
            Size.Fixed((0.65) * label_margin * int(show_xlabel) + base_margin),
            Size.Fixed(base_size),
            Size.Fixed(base_margin + 0.25 * label_margin),
        ]
    else:
        v = [
            Size.Fixed((0.65) * label_margin * int(show_xlabel) + base_margin),
            Size.Fixed(base_size),
            Size.Fixed(base_margin),
        ]

    # Set figure dimensions
    fig.set_size_inches(width_inches, height_inches)

    if no_tick_labels:
        h = [
            Size.Fixed(label_margin * 0.5 * int(show_ylabel) + base_margin),
            Size.Fixed(base_size),
            Size.Fixed(base_margin),
        ]
    else:
        h = [
            Size.Fixed(label_margin * int(show_ylabel) + base_margin),
            Size.Fixed(base_size),
            Size.Fixed(base_margin),
        ]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)

    ax = fig.add_axes(
        divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1)
    )

    # Update font sizes based on intended display size
    base_fontsize = 14 * size_fac
    tick_fontsize = 14 * size_fac
    label_fontsize = 14 * size_fac

    # Unpack parameters efficiently
    (
        element_str,
        var,
        unit,
        xlabel,
        annotation_str,
        param_min,
        param_max,
        xlim,
        ann_x_frac,
        min_mirror,
        max_mirror,
        circ_param,
        fit_log,
        interp_pc,
        n_nom,
        n_errp,
        n_errm,
        ann_nom_add,
        peak,
    ) = element_params  # plotting parameters (str labels, boundary conditions, number of significant
    # figures to report values, etc.

    print(f"Creating Plot for {element_str}...")

    if calc_kde:
        # Calculate KDE
        peak_x_mid, err_plus, err_minus, kdex, kdey, interpx, nat_csy = fftkde_cred_int(
            element,
            param_min,
            param_max,
            min_mirror=min_mirror,
            max_mirror=max_mirror,
            circ_param=circ_param,
            fit_log=fit_log,
            bw_rule=bw_rule,
            hdi_pc=hdi_pc,
            fftkde_res=fftkde_res,
            interp_pc=interp_pc,
        )

        # Main KDE plot
        lines = []
        lines.append(
            ax.plot(kdex, kdey, color=color, lw=lw * size_fac, ls="-", zorder=2)[0]
        )

        # Uncertainty range
        hdi_mask = (kdex >= peak_x_mid - err_minus) & (kdex <= peak_x_mid + err_plus)
        ax.fill_between(
            kdex[hdi_mask],
            np.zeros_like(kdey[hdi_mask]),
            kdey[hdi_mask],
            color=color,
            alpha=fill_alpha,
            zorder=1,
            lw=0,
        )

        # Print result
        print(
            f"Total Probability in bounds of: {hdi_pc/100:.1%}\n"
            f"{var} = ({peak_x_mid:.4f} +{err_plus:.4f} or -{err_minus:.4f}) {unit}\n"
        )

        # Add vertical lines
        line_styles = [
            (peak_x_mid, "--", lw * size_fac),
            (peak_x_mid - err_minus, ":", lw * size_fac * 0.75),
            (peak_x_mid + err_plus, ":", lw * size_fac * 0.75),
        ]

        for x, ls, line_width in line_styles:
            lines.append(
                ax.axvline(x=x, color="k", linestyle=ls, lw=line_width, zorder=2)
            )

    # histogram (optional)
    if plot_hist:
        elem_range = np.max(element) - np.min(element)
        xlim_range = xlim[1] - xlim[0]

        # Density is True, but we need more bins if there are values outside of xlim
        try:  # integer bins passes
            if int(hist_bins) == hist_bins:
                plt.hist(
                    element,
                    bins=int(hist_bins * elem_range / xlim_range),
                    density=True,
                    histtype="stepfilled",
                    zorder=0,
                    color=hist_color,
                )
            else:  # e.g., 'knuth', 'freedman'
                hist(
                    element,
                    bins=hist_bins,
                    density=True,
                    histtype="stepfilled",
                    zorder=0,
                    color=hist_color,
                )

        except:  # default here if int(hist_bins) throws an exception (same as else: above)
            hist(
                element,
                bins=hist_bins,
                density=True,
                histtype="stepfilled",
                zorder=0,
                color=hist_color,
            )

    # x- and y-limits
    ax_ylim = ax.get_ylim()
    ylim = (0, ax_ylim[1])

    # Add annotation if requested
    if annotate:
        ann_x = xlim[0] + ann_x_frac * (xlim[1] - xlim[0])
        if ann_str is None:
            formatted_values = (
                float("{:.{p}g}".format(peak_x_mid + ann_nom_add, p=n_nom)),
                float("{:.{p}g}".format(err_plus, p=n_errp)),
                float("{:.{p}g}".format(err_minus, p=n_errm)),
            )
            ax.annotate(
                annotation_str.format(*formatted_values),
                (ann_x, ax_ylim[1] * annotation_y_frac),
                fontweight="bold",
                fontsize=14 * size_fac,
                zorder=2,
            )
        else:
            ax.annotate(
                ann_str,
                (ann_x, ax_ylim[1] * annotation_y_frac),
                fontweight="bold",
                fontsize=14 * size_fac,
                zorder=2,
            )
    elif ann_title:
        if ann_str is None:
            formatted_values = (
                float("{:.{p}g}".format(peak_x_mid + ann_nom_add, p=n_nom)),
                float("{:.{p}g}".format(err_plus, p=n_errp)),
                float("{:.{p}g}".format(err_minus, p=n_errm)),
            )
            ax.set_title(
                annotation_str.format(*formatted_values),
                fontsize=14 * size_fac,
                pad=8 * size_fac,
            )
        else:
            ax.set_title(ann_str, fontsize=14 * size_fac, pad=8 * size_fac)

    # Configure axes with square aspect ratio for data display
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Force square aspect ratio for the data display area
    ax.set_aspect(abs(xlim[1] - xlim[0]) / abs(ylim[1] - ylim[0]))

    # label x-axis (optional)
    if show_xlabel:
        t = ax.text(
            0.5,
            ylab_offset * (0.6),
            xlabel,
            rotation=0,
            verticalalignment="top",
            horizontalalignment="center",
            transform=ax.transAxes,
            fontsize=label_fontsize,
        )

    else:
        ax.tick_params(labelbottom=False)

    # label y-axis (optional)
    if show_ylabel:
        if no_ticks:  # ylabel, but no ticks
            t = ax.text(
                ylab_offset / 3,
                0.5,
                r"\textbf{Probability Density}",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                transform=ax.transAxes,
                fontsize=label_fontsize,
            )

            if no_tick_labels:  # ylabel, no ticks, no tick labels
                ax.tick_params(labelleft=False, left=False)
            else:  # ylabel, no ticks, but add tick labels
                ax.tick_params(labelleft=True, left=False)
        else:  # ylabel and ticks
            t = ax.text(
                ylab_offset,
                0.5,
                r"\textbf{Probability Density}",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                transform=ax.transAxes,
                fontsize=label_fontsize,
            )
            if no_tick_labels:  # y label and ticks, but no tick labels
                ax.tick_params(labelleft=False, left=True)
            else:  # ylabel and ticks and also tick labels
                ax.tick_params(labelleft=True, left=True)
    else:
        if no_ticks:  # no ylabel and no ticks
            if no_tick_labels:  # no ylabel, no ticks, no tick labels
                ax.tick_params(labelleft=False, left=False)
            else:  # no ylabel, no ticks, but add tick labels
                ax.tick_params(labelleft=True, left=False)
        else:  # no ylabel and yes ticks
            if no_tick_labels:  # no ylabel and yes ticks, but no tick labels
                ax.tick_params(labelleft=False, left=True)
            else:  # no ylabel and yes ticks and also tick labels
                ax.tick_params(labelleft=True, left=True)

    # Handle log scale plotting (optiona)
    if plot_log:
        ax.set_xscale("log")
        ax.tick_params(
            axis="both",
            which="minor",
            width=2 * size_fac / 2,
            length=6 * size_fac / 2,
            direction="in",
        )

    # Update spine and tick parameters
    spine_width = 2 * size_fac
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    # tick parameters
    ax.tick_params(
        axis="both",
        which="major",
        width=2 * size_fac,
        length=6 * size_fac,
        direction="in",
        labelsize=tick_fontsize,
        zorder=5,
    )

    # Save the image, if requested
    if save:
        print("saving figure...\n")
        save_kwargs = {"facecolor": "w", "dpi": dpi}

        if jpeg:
            fig.savefig(
                f"{obj}_{element_str}_KDE_{sampler}_{bw_rule}.jpg", **save_kwargs
            )
        else:
            fig.savefig(
                f"{obj}_{element_str}_KDE_{sampler}_{bw_rule}.png", **save_kwargs
            )
        plt.close("all")

        # try to clear up some space, but not all of these are generated depending on options
        try:
            del fig, ax, lines, hdi_mask, element_params, kdex, kdey, nat_csy, interpx
        except:
            try:
                del fig, ax, element_params
            except:
                pass

        gc.collect()  # free up some memory, if possible

        try:
            return peak_x_mid, err_plus, err_minus  # return hdi values, if calculated,
            # and saving the plot
        except:
            return None, None, None  # otherwise, return none
    else:  # if saving, do the same thing, but directly return fig, ax
        try:
            return peak_x_mid, err_plus, err_minus, fig, ax
        except:
            return None, None, None


def off_diagonal_fftkde_plot(
    param_x,
    param_y,
    element_x,
    element_y,
    big_df,
    kernel="gaussian",
    bw_rule="silvermans",
    norm=2,
    grid_points=2**12,  # in each dimension
    N=997,  # number of contours for contourf
    spine_color="k",
    ticklab_color="k",
    tick_color="k",
    # solid contour levels
    std_norm_levels=[np.exp(-((n_sigma) ** 2) / 2) for n_sigma in [0.5, 1, 1.5]],
    clear_1sig=True,  # clear contouf outside of the bottom contour
    min_contour_lw=0.5,  # solid contour lowest line width
    cmap="plasma_r",
    cmap_max=1.0,
    cmap_min=0.0,
    dpi=150,
    scatter_cmp_val=0.55,
    scatter_alpha=10 / 510,
    show_xlabel=True,
    show_ylabel=True,
    jpeg=False,
    size_fac=3.5,
    plot_raw_kde=False,  # for debugging purposes only
    base_margin=0.5,
    label_margin=1.96875,
    pad_inches=0.0,
    base_size=10,
    ylab_offset=-0.175,
    show_peak_x_mid=True,
    show_peak_y_mid=True,
    lw=2,
    scatter_size_fac=4,
):
    """
    creates a 2-D FFTKDE plot for parameters "param_x, param_y"
    many options available as kwargs, but default values are provided.
    """

    # Set square figure size
    base_size_inches = base_size

    ylab_offset *= size_fac / 2.75
    base_margin *= size_fac / 2.75
    label_margin *= size_fac / 2.75

    # Initialize colormap once
    cmap_mod = plt.get_cmap(cmap, 10_000)
    cmp = ListedColormap(cmap_mod(np.linspace(cmap_min, cmap_max, 10_000)))

    # Extract parameters efficiently
    element_params_x = big_df.loc[param_x]
    element_params_y = big_df.loc[param_y]

    # Unpack parameters (using array indexing instead of tuple unpacking)
    params_x = np.array(element_params_x)
    params_y = np.array(element_params_y)

    # Get transformation parameters from arrays
    fit_logx = params_x[12]
    fit_logy = params_y[12]
    min_mirror_x = params_x[9]
    max_mirror_x = params_x[10]
    min_mirror_y = params_y[9]
    max_mirror_y = params_y[10]
    circ_param_x = params_x[11]
    circ_param_y = params_y[11]
    peak_x_mid = params_x[18]
    peak_y_mid = params_y[18]

    # Transform data more efficiently
    fit_element_x = np.log10(element_x) if fit_logx else element_x.copy()
    fit_element_y = np.log10(element_y) if fit_logy else element_y.copy()

    # Handle parameter bounds
    fit_param_min_x = (
        (np.log10(params_x[5]) if params_x[5] > 0 else np.min(fit_element_x) - 0.3)
        if fit_logx
        else params_x[5]
    )
    fit_param_max_x = np.log10(params_x[6]) if fit_logx else params_x[6]
    fit_param_min_y = (
        (np.log10(params_y[5]) if params_y[5] > 0 else np.min(fit_element_y) - 0.3)
        if fit_logy
        else params_y[5]
    )
    fit_param_max_y = np.log10(params_y[6]) if fit_logy else params_y[6]

    # Combine original data for SVD before mirroring
    original_data = np.column_stack((fit_element_x, fit_element_y))

    # Perform SVD with optimized settings on original data
    data_mean = np.mean(original_data, axis=0)
    data_std = np.std(original_data, axis=0, ddof=1)
    original_data_c = (original_data - data_mean) / data_std

    # Use more efficient SVD algorithm
    U, s, Vt = la.svd(original_data_c, full_matrices=False, lapack_driver="gesdd")
    V = Vt.T
    n_orbs = len(element_x)

    # Rotate the standardized data to PCA-coordinates
    rotated_data_c = original_data_c @ V @ np.diag(1 / s) * np.sqrt(n_orbs)

    # Compute bandwidths along PCA axes for the standardized data
    if bw_rule == "silvermans":
        both_bws_i = [silvermans_rule(rotated_data_c[:, [i]]) for i in range(2)]
    elif bw_rule == "isj":  # isj
        both_bws_i = [improved_sheather_jones(rotated_data_c[:, [i]]) for i in range(2)]

    # Now apply boundary conditions to mirror/repeat the original data before standardization
    # / PCA-alignment
    x_bc_count = 0
    new_fit_element_x = np.copy(fit_element_x)
    new_fit_element_y = np.copy(fit_element_y)

    if min_mirror_x and not max_mirror_x:
        print(f"mirroring at min_x value: {fit_param_min_x}")
        new_fit_element_x = np.concatenate(
            (new_fit_element_x, 2 * fit_param_min_x - fit_element_x)
        )
        new_fit_element_y = np.concatenate((new_fit_element_y, fit_element_y))
        x_bc_count = 1
    elif max_mirror_x and not min_mirror_x:
        print(f"mirroring at max_x value: {fit_param_max_x}")
        new_fit_element_x = np.concatenate(
            (new_fit_element_x, 2 * fit_param_max_x - fit_element_x)
        )
        new_fit_element_y = np.concatenate((new_fit_element_y, fit_element_y))
        x_bc_count = 1
    elif max_mirror_x and min_mirror_x:
        print(f"mirroring at min_x value: {fit_param_min_x}")
        print(f"mirroring at max_x value: {fit_param_max_x}")
        new_fit_element_x = np.concatenate(
            (
                new_fit_element_x,
                2 * fit_param_max_x - fit_element_x,
                2 * fit_param_min_x - fit_element_x,
            )
        )
        new_fit_element_y = np.concatenate(
            (new_fit_element_y, fit_element_y, fit_element_y)
        )
        x_bc_count = 2
    elif circ_param_x:
        print(
            f"Repeating at circular min_x and max_x values: {fit_param_min_x}, {fit_param_max_x}"
        )
        period_x = fit_param_max_x - fit_param_min_x
        new_fit_element_x = np.concatenate(
            (fit_element_x - period_x, new_fit_element_x, fit_element_x + period_x)
        )
        new_fit_element_y = np.concatenate(
            (fit_element_y, new_fit_element_y, fit_element_y)
        )
        x_bc_count = 2

    # Handle boundary conditions for y-axis
    y_bc_count = 0
    if min_mirror_y and not max_mirror_y:
        print(f"mirroring at min_y value: {fit_param_min_y}")
        new_fit_element_y = np.concatenate(
            (new_fit_element_y, 2 * fit_param_min_y - fit_element_y)
        )
        new_fit_element_x = np.concatenate((new_fit_element_x, fit_element_x))
        y_bc_count = 1

    elif max_mirror_y and not min_mirror_y:
        print(f"mirroring at max_y value: {fit_param_max_y}")
        new_fit_element_y = np.concatenate(
            (new_fit_element_y, 2 * fit_param_max_y - fit_element_y)
        )
        new_fit_element_x = np.concatenate((new_fit_element_x, fit_element_x))
        y_bc_count = 1

    elif max_mirror_y and min_mirror_y:
        print(f"mirroring at min_y value: {fit_param_min_y}")
        print(f"mirroring at max_y value: {fit_param_max_y}")
        new_fit_element_y = np.concatenate(
            (
                new_fit_element_y,
                2 * fit_param_max_y - fit_element_y,
                2 * fit_param_min_y - fit_element_y,
            )
        )
        new_fit_element_x = np.concatenate(
            (new_fit_element_x, fit_element_x, fit_element_x)
        )
        y_bc_count = 2

    elif circ_param_y:
        print(
            f"Repeating at circular min_y and max_y values: {fit_param_min_y}, {fit_param_max_y}"
        )
        period_y = fit_param_max_y - fit_param_min_y
        new_fit_element_y = np.concatenate(
            (fit_element_y - period_y, new_fit_element_y, fit_element_y + period_y)
        )
        new_fit_element_x = np.concatenate(
            (fit_element_x, new_fit_element_x, fit_element_x)
        )
        y_bc_count = 2

    # Adjust grid points based on boundary conditions
    print(f"x_bc_count: {x_bc_count}")
    print(f"y_bc_count: {y_bc_count}")
    print(f"grid_points before adjustment: {grid_points}")
    grid_points *= 2 ** np.max([x_bc_count, y_bc_count])
    print(f"grid_points after adjustment: {grid_points}")
    # Combine mirrored data for KDE
    off_diag_data = np.column_stack((new_fit_element_x, new_fit_element_y))

    # Center and scale mirrored data using original data statistics
    off_diag_data_c = (off_diag_data - data_mean) / data_std

    # Rotate mirrored data to un-mirrored PCA frame
    rotated_data_c = off_diag_data_c @ V @ np.diag(1 / s) * np.sqrt(n_orbs)

    # KDE computation using bandwidths from original data (effectively)
    print("computing kde...")
    kde = FFTKDE(kernel=kernel, norm=norm, bw=1)
    # scale by the bw first so that bw=1 makes sense
    scaled_rotated_data_c = rotated_data_c / np.array(both_bws_i)
    # evaluate KDE on mirrored data
    grid, points = kde.fit(scaled_rotated_data_c).evaluate((grid_points, grid_points))

    # Scale back to both_bws_i along original PCA axes
    grid = grid * np.array(both_bws_i)
    points = points / np.prod(both_bws_i)

    # Rotate grid back efficiently
    grid_rot = grid / np.sqrt(n_orbs) @ np.linalg.inv(V @ np.diag(1 / s))

    # Resample grid using new bandwidths for the grid
    print("resampling grid...")
    print("finding ISJ bws...")
    both_grid_bws = [improved_sheather_jones(grid_rot[:, [i]]) for i in range(2)]
    grid_rot_scaled = grid_rot / np.array(both_grid_bws)

    print("computing kde for resample...")
    kde = FFTKDE(kernel="gaussian", norm=norm, bw=1)
    grid, points = kde.fit(grid_rot_scaled, weights=points).evaluate(
        (grid_points, grid_points)
    )

    # Scale back
    grid = grid * np.array(both_grid_bws)
    points = points / np.prod(both_grid_bws)

    # Prepare plotting data
    print("preparing x y z to plot...")
    x = np.unique(grid[:, 0]) * data_std[0] + data_mean[0]
    y = np.unique(grid[:, 1]) * data_std[1] + data_mean[1]
    z = points.reshape(grid_points, grid_points).T

    # Check shapes
    if z.shape != (len(y), len(x)):
        raise ValueError(
            f"Shape mismatch: z.shape={z.shape}, expected={(len(y), len(x))}"
        )

    # Truncate KDE to valid parameter ranges
    print(f"np.min(x) before mask: {np.min(x)}")
    print(f"np.max(x) before mask: {np.max(x)}")
    print(f"\nnp.min(y) before mask: {np.min(y)}")
    print(f"np.max(y) before mask: {np.max(y)}")
    print(f"\nfit_param_min_x: {fit_param_min_x}")
    print(f"fit_param_max_x: {fit_param_max_x}")
    print(f"\nfit_param_min_y: {fit_param_min_y}")
    print(f"fit_param_max_y: {fit_param_max_y}")
    valid_kde_is_x = np.where(
        np.logical_and(x >= fit_param_min_x, x <= fit_param_max_x)
    )[0]
    valid_kde_is_y = np.where(
        np.logical_and(y >= fit_param_min_y, y <= fit_param_max_y)
    )[0]

    # Update x, y, and z arrays to only include valid regions
    x = x[valid_kde_is_x]
    y = y[valid_kde_is_y]
    z = z[np.ix_(valid_kde_is_y, valid_kde_is_x)]  # Check the order of indices

    # Handle log scaling
    if fit_logx:
        x = 10**x
        z = np.divide(z, x, out=np.zeros_like(z))
    if fit_logy:
        y = 10**y
        z = np.divide(z, y[:, np.newaxis], out=np.zeros_like(z))

    print(f"np.min(x) after mask and un-log (if appl.): {np.min(x)}")
    print(f"np.max(x) after mask and un-log (if appl.): {np.max(x)}")
    print(f"\nnp.min(y) after mask and un-log (if appl.): {np.min(y)}")
    print(f"np.max(y) after mask and un-log (if appl.): {np.max(y)}")

    # First integration over y
    print("normalizing KDE...")
    integrated_y = simpson(z, axis=0, x=y)

    # Second integration over x
    total_prob = simpson(integrated_y, axis=0, x=x)

    # Normalize z
    if total_prob > 0:
        z /= total_prob
    else:
        print("Warning: Total probability is zero; normalization skipped.")

    # Create figure with calculated dimensions
    fig = plt.figure(dpi=dpi)

    width_inches = base_size + label_margin * int(show_ylabel) + 2 * base_margin
    height_inches = (
        base_size + (0.65) * label_margin * int(show_xlabel) + 2 * base_margin
    )

    # Set figure dimensions
    fig.set_size_inches(width_inches, height_inches)

    h = [
        Size.Fixed(label_margin * int(show_ylabel) + base_margin),
        Size.Fixed(base_size),
        Size.Fixed(base_margin),
    ]
    v = [
        Size.Fixed((0.65) * label_margin * int(show_xlabel) + base_margin),
        Size.Fixed(base_size),
        Size.Fixed(base_margin),
    ]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)

    ax = fig.add_axes(
        divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1)
    )

    # Update font sizes based on intended display size
    base_fontsize = 14 * size_fac
    tick_fontsize = 14 * size_fac
    label_fontsize = 14 * size_fac

    # contour levels and their colors
    KDE_levels = quantile_to_level(z, std_norm_levels)
    color_arr = [cmp(1.0) for i in range(len(KDE_levels))]

    # Plot contours with adjusted linewidths
    print("adding contours")
    contour_lw_base = min_contour_lw * size_fac
    contours = ax.contour(
        x,
        y,
        z,
        levels=KDE_levels,
        linewidths=[
            contour_lw_base * (1 + (i / (len(KDE_levels) - 1)) / 2)
            for i in range(len(KDE_levels))
        ],
        colors=color_arr,
        zorder=1,
    )

    # Get contours for point checking
    contours1 = contours.allsegs[0]
    mpl_contours = [
        np.array(c).reshape((-1, 1, 2)).astype(np.float32) for c in contours1
    ]

    # Use provided functions for point checking
    x_outside, y_outside = optimized_contour_point_check(
        element_x, element_y, mpl_contours
    )

    # Clear KDE outside first contour if requested
    if clear_1sig:
        z[z < contours.levels[0]] = 0

    # Plot filled contours
    print("plotting filled contours...")
    ax.contourf(
        x, y, z, levels=N, cmap=cmp, zorder=0, vmin=contours.levels[0], vmax=np.max(z)
    )

    # Plot scatter points with adjusted size
    print("scattering points...")
    scatter_size = (4 * 72.0 * scatter_size_fac / dpi) ** 2  # 4x4 pixels
    ax.scatter(
        x_outside,
        y_outside,
        color=cmp(scatter_cmp_val),
        zorder=0,
        lw=0,
        s=scatter_size,
        alpha=scatter_alpha,
        marker="s",
    )

    # Style plot with consistent spine width matching fftkde_plot
    spine_width = 2 * size_fac
    for axis in ["left", "bottom", "top", "right"]:
        ax.spines[axis].set_linewidth(spine_width)
        ax.spines[axis].set_color(spine_color)

    # Set tick parameters to match fftkde_plot exactly
    ax.tick_params(
        axis="both",
        which="major",
        width=2 * size_fac,  # Match spine width
        length=6 * size_fac,
        direction="in",
        color=tick_color,
        labelcolor=ticklab_color,
        labelsize=tick_fontsize,
    )

    # Set plot limits and force square aspect ratio for data display
    xlim = params_x[7]
    ylim = params_y[7]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Force square aspect ratio for the data display area
    ax.set_aspect(abs(xlim[1] - xlim[0]) / abs(ylim[1] - ylim[0]))

    # x-axis label
    if show_xlabel:
        t = ax.text(
            0.5,
            ylab_offset * (0.6),
            params_x[3],
            rotation=0,
            verticalalignment="top",
            horizontalalignment="center",
            transform=ax.transAxes,
            fontsize=label_fontsize,
        )
    else:
        ax.tick_params(labelbottom=False)

    # y-axis label (optional)
    if show_ylabel:
        t = ax.text(
            ylab_offset,
            0.5,
            params_y[3],
            rotation=90,
            verticalalignment="center",
            horizontalalignment="right",
            transform=ax.transAxes,
            fontsize=label_fontsize,
        )
    else:
        ax.tick_params(labelleft=False)

    # Raw KDE plot with matching parameters (only for debugging)
    if plot_raw_kde:
        print("creating raw KDE for visualization")
        fig2, ax2 = plt.subplots(dpi=dpi)

        # Use same dimensions and margins for consistency
        fig2.set_size_inches(width_inches, height_inches)

        plt.subplots_adjust(
            left=left_margin,
            right=1 - right_margin,
            bottom=bottom_margin,
            top=1 - top_margin,
        )

        raw_levels = N // 10
        ax2.contourf(x, y, z, levels=raw_levels, cmap=cmp, zorder=0)

        # Apply consistent styling to raw KDE plot
        for axis in ["left", "bottom", "top", "right"]:
            ax2.spines[axis].set_linewidth(spine_width)
            ax2.spines[axis].set_color(spine_color)

        ax2.tick_params(
            axis="both",
            which="major",
            width=2 * size_fac,
            length=6 * size_fac,
            direction="in",
            color=tick_color,
            labelcolor=ticklab_color,
            labelsize=tick_fontsize,
        )

        if show_xlabel:
            t = ax.text(
                0.5,
                ylab_offset * (0.6),
                params_x[3],
                rotation=0,
                verticalalignment="top",
                horizontalalignment="center",
                transform=ax.transAxes,
                fontsize=label_fontsize,
            )
        else:
            ax2.tick_params(labelbottom=False)

        if show_ylabel:
            t = ax2.text(
                ylab_offset,
                0.5,
                params_y[3],
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                transform=ax2.transAxes,
                fontsize=label_fontsize,
            )
        else:
            ax2.tick_params(labelleft=False)

        print("saving raw kde...")
        fig2.savefig(
            f"{params_y[1]}_v_{params_x[1]}_gp_{grid_points}_N_{N}_cmp_{cmap}_{bw_rule}"
            + "_rawkde_"
            + ".jpg",
            dpi=dpi,
        )

    # plot dashed lines at "truth" values of some sort (like max-a-posteriori in marginal dist.)
    if show_peak_x_mid:
        ax.axvline(x=peak_x_mid, color="k", linestyle="--", lw=lw * size_fac, zorder=6)
    if show_peak_y_mid:
        ax.axhline(y=peak_y_mid, color="k", linestyle="--", lw=lw * size_fac, zorder=6)

    # save figure
    print("saving figure...\n")
    if jpeg:
        fig.savefig(
            f"{params_y[1]}_v_{params_x[1]}_gp_{grid_points}_N_{N}_cmp_{cmap}_{bw_rule}.jpg",
            dpi=dpi,
        )
    else:
        fig.savefig(
            f"{params_y[1]}_v_{params_x[1]}_gp_{grid_points}_N_{N}_cmp_{cmap}_{bw_rule}.png",
            dpi=dpi,
        )

    # free up some memory
    plt.close("all")
    gc.collect()

    return None
