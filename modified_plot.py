import numpy as np
import corner
import warnings
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

import orbitize
import orbitize.kepler as kepler


# TODO: deprecatation warning for plots in results

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


def plot_corner(results, param_list=None, **corner_kwargs):
    """
    Make a corner plot of posterior on orbit fit from any sampler

    Args:
            param_list (list of strings): each entry is a name of a parameter to include.
                    Valid strings::

                            sma1: semimajor axis
                            ecc1: eccentricity
                            inc1: inclination
                            aop1: argument of periastron
                            pan1: position angle of nodes
                            tau1: epoch of periastron passage, expressed as fraction of orbital period
                            per1: period
                            K1: stellar radial velocity semi-amplitude
                            [repeat for 2, 3, 4, etc if multiple objects]
                            plx:  parallax
                            pm_ra: RA proper motion
                            pm_dec: Dec proper motion
                            alpha0: primary offset from reported Hipparcos RA @ alphadec0_epoch (generally 1991.25)
                            delta0: primary offset from reported Hipparcos Dec @ alphadec0_epoch (generally 1991.25)
                            gamma: rv offset
                            sigma: rv jitter
                            mi: mass of individual body i, for i = 0, 1, 2, ... (only if fit_secondary_mass == True)
                            mtot: total mass (only if fit_secondary_mass == False)

            **corner_kwargs: any remaining keyword args are sent to ``corner.corner``.
                                                    See `here <https://corner.readthedocs.io/>`_.
                                                    Note: default axis labels used unless overwritten by user input.

    Return:
            ``matplotlib.pyplot.Figure``: corner plot

    .. Note:: **Example**: Use ``param_list = ['sma1,ecc1,inc1,sma2,ecc2,inc2']`` to only
            plot posteriors for semimajor axis, eccentricity and inclination
            of the first two companions

    Written: Henry Ngo, 2018
    """

    # Define array of default axis labels (overwritten if user specifies list)
    default_labels = {
        "sma": "$a_{0}$ [au]",
        "ecc": "$ecc_{0}$",
        "inc": "$inc_{0}$ [$^\\circ$]",
        "aop": "$\\omega_{0}$ [$^\\circ$]",
        "pan": "$\\Omega_{0}$ [$^\\circ$]",
        "tau": "$\\tau_{0}$",
        "plx": "$\\pi$ [mas]",
        "gam": "$\\gamma$ [km/s]",
        "sig": "$\\sigma$ [km/s]",
        "mtot": "$M_T$ [M$_{{\\odot}}$]",
        "m0": "$M_0$ [M$_{{\\odot}}$]",
        "m": "$M_{0}$ [M$_{{\\rm Jup}}$]",
        "pm_ra": "$\\mu_{{\\alpha}}$ [mas/yr]",
        "pm_dec": "$\\mu_{{\\delta}}$ [mas/yr]",
        "alpha0": "$\\alpha^{{*}}_{{0}}$ [mas]",
        "delta0": "$\\delta_0$ [mas]",
        "m": "$M_{0}$ [M$_{{\\rm Jup}}$]",
        "per": "$P_{0}$ [yr]",
        "K": "$K_{0}$ [km/s]",
        "x": "$X_{0}$ [AU]",
        "y": "$Y_{0}$ [AU]",
        "z": "$Z_{0}$ [AU]",
        "xdot": "$xdot_{0}$ [km/s]",
        "ydot": "$ydot_{0}$ [km/s]",
        "zdot": "$zdot_{0}$ [km/s]",
    }

    if param_list is None:
        param_list = results.labels

    param_indices = []
    angle_indices = []
    secondary_mass_indices = []
    for i, param in enumerate(param_list):
        index_num = results.param_idx[param]

        # only plot non-fixed parameters
        if np.std(results.post[:, index_num]) > 0:
            param_indices.append(index_num)
            label_key = param
            if (
                label_key.startswith("aop")
                or label_key.startswith("pan")
                or label_key.startswith("inc")
            ):
                angle_indices.append(i)
            if label_key.startswith("m") and label_key != "m0" and label_key != "mtot":
                secondary_mass_indices.append(i)

    samples = np.copy(
        results.post[:, param_indices]
    )  # keep only chains for selected parameters
    samples[:, angle_indices] = np.degrees(
        samples[:, angle_indices]
    )  # convert angles from rad to deg
    samples[:, secondary_mass_indices] *= u.solMass.to(
        u.jupiterMass
    )  # convert to Jupiter masses for companions

    if (
        "labels" not in corner_kwargs
    ):  # use default labels if user didn't already supply them
        reduced_labels_list = []
        for i in np.arange(len(param_indices)):
            label_key = param_list[i]
            if label_key.startswith("m") and label_key != "m0" and label_key != "mtot":
                body_num = label_key[1]
                label_key = "m"
            elif (
                label_key == "m0" or label_key == "mtot" or label_key.startswith("plx")
            ):
                body_num = ""
                # maintain original label key
            elif label_key in ["pm_ra", "pm_dec", "alpha0", "delta0"]:
                body_num = ""
            elif label_key.startswith("gamma") or label_key.startswith("sigma"):
                body_num = ""
                label_key = label_key[0:3]
            else:
                body_num = label_key[-1]
                label_key = label_key[0:-1]
            reduced_labels_list.append(default_labels[label_key].format(body_num))

        corner_kwargs["labels"] = reduced_labels_list

    figure = corner.corner(samples, **corner_kwargs)
    return figure


def plot_orbits(
    results,
    object_to_plot=1,
    start_mjd=51544.0,
    num_orbits_to_plot=100,
    num_epochs_to_plot=100,
    square_plot=True,
    show_colorbar=True,
    cmap=cmap,
    sep_pa_color="lightgrey",
    sep_pa_end_year=2025.0,
    cbar_param="Epoch [year]",
    mod180=False,
    rv_time_series=False,
    plot_astrometry=True,
    plot_astrometry_insts=False,
    plot_errorbars=True,
    fig=None, ms=5, capsize=2,elinewidth=1,capthick=1
):
    """
    Plots one orbital period for a select number of fitted orbits
    for a given object, with line segments colored according to time

    Args:
        object_to_plot (int): which object to plot (default: 1)
        start_mjd (float): MJD in which to start plotting orbits (default: 51544,
            the year 2000)
        num_orbits_to_plot (int): number of orbits to plot (default: 100)
        num_epochs_to_plot (int): number of points to plot per orbit (default: 100)
        square_plot (Boolean): Aspect ratio is always equal, but if
            square_plot is True (default), then the axes will be square,
            otherwise, white space padding is used
        show_colorbar (Boolean): Displays colorbar to the right of the plot [True]
        cmap (matplotlib.cm.ColorMap): color map to use for making orbit tracks
            (default: modified Purples_r)
        sep_pa_color (string): any valid matplotlib color string, used to set the
            color of the orbit tracks in the Sep/PA panels (default: 'lightgrey').
        sep_pa_end_year (float): decimal year specifying when to stop plotting orbit
            tracks in the Sep/PA panels (default: 2025.0).
        cbar_param (string): options are the following: 'Epoch [year]', 'sma1', 'ecc1', 'inc1', 'aop1',
            'pan1', 'tau1', 'plx. Number can be switched out. Default is Epoch [year].
        mod180 (Bool): if True, PA will be plotted in range [180, 540]. Useful for plotting short
            arcs with PAs that cross 360 deg during observations (default: False)
        rv_time_series (Boolean): if fitting for secondary mass using MCMC for rv fitting and want to
            display time series, set to True.
        plot_astrometry (Boolean): set to True by default. Plots the astrometric data.
        plot_astrometry_insts (Boolean): set to False by default. Plots the astrometric data by instruments.
        plot_errorbars (Boolean): set to True by default. Plots error bars of measurements
        fig (matplotlib.pyplot.Figure): optionally include a predefined Figure object to plot the orbit on.
            Most users will not need this keyword. 

    Return:
        ``matplotlib.pyplot.Figure``: the orbit plot if input is valid, ``None`` otherwise


    (written): Henry Ngo, Sarah Blunt, 2018
    Additions by Malena Rice, 2019

    """

    if Time(start_mjd, format="mjd").decimalyear >= sep_pa_end_year:
        raise ValueError(
            "start_mjd keyword date must be less than sep_pa_end_year keyword date."
        )

    if object_to_plot > results.num_secondary_bodies:
        raise ValueError(
            "Only {0} secondary bodies being fit. Requested to plot body {1} which is out of range".format(
                results.num_secondary_bodies, object_to_plot
            )
        )

    if object_to_plot == 0:
        raise ValueError(
            "Plotting the primary's orbit is currently unsupported. Stay tuned."
        )

    if rv_time_series and "m0" not in results.labels:
        rv_time_series = False

        warnings.warn(
            "It seems that the stellar and companion mass "
            "have not been fitted separately. Setting "
            "rv_time_series=True is therefore not possible "
            "so the argument is set to False instead."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)

        data = results.data[results.data["object"] == object_to_plot]
        possible_cbar_params = ["sma", "ecc", "inc", "aop" "pan", "tau", "plx"]

        if cbar_param == "Epoch [year]":
            pass
        elif cbar_param[0:3] in possible_cbar_params:
            index = results.param_idx[cbar_param]
        else:
            raise Exception(
                "Invalid input; acceptable inputs include 'Epoch [year]', 'plx', 'sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'sma2', 'ecc2', ...)"
            )
        # Select random indices for plotted orbit
        num_orbits = len(results.post[:, 0])
        if num_orbits_to_plot > num_orbits:
            num_orbits_to_plot = num_orbits
        choose = np.random.randint(0, high=num_orbits, size=num_orbits_to_plot)

        # Get posteriors from random indices
        standard_post = []
        if results.sampler_name == "MCMC":
            # Convert the randomly chosen posteriors to standard keplerian set
            for i in np.arange(num_orbits_to_plot):
                orb_ind = choose[i]
                param_set = np.copy(results.post[orb_ind])
                standard_post.append(results.basis.to_standard_basis(param_set))
        else:  # For OFTI, posteriors are already converted
            for i in np.arange(num_orbits_to_plot):
                orb_ind = choose[i]
                standard_post.append(results.post[orb_ind])

        standard_post = np.array(standard_post)

        sma = standard_post[
            :, results.standard_param_idx["sma{}".format(object_to_plot)]
        ]
        ecc = standard_post[
            :, results.standard_param_idx["ecc{}".format(object_to_plot)]
        ]
        inc = standard_post[
            :, results.standard_param_idx["inc{}".format(object_to_plot)]
        ]
        aop = standard_post[
            :, results.standard_param_idx["aop{}".format(object_to_plot)]
        ]
        pan = standard_post[
            :, results.standard_param_idx["pan{}".format(object_to_plot)]
        ]
        tau = standard_post[
            :, results.standard_param_idx["tau{}".format(object_to_plot)]
        ]
        plx = standard_post[:, results.standard_param_idx["plx"]]

        # Then, get the other parameters
        if "mtot" in results.labels:
            mtot = standard_post[:, results.standard_param_idx["mtot"]]
        elif "m0" in results.labels:
            m0 = standard_post[:, results.standard_param_idx["m0"]]
            m1 = standard_post[
                :, results.standard_param_idx["m{}".format(object_to_plot)]
            ]
            mtot = m0 + m1

        raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        vz_star = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
        # Need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
        for i in np.arange(num_orbits_to_plot):
            # Compute period (from Kepler's third law)
            period = np.sqrt(
                4 * np.pi ** 2.0 * (sma * u.AU) ** 3 / (consts.G * (mtot * u.Msun))
            )
            period = period.to(u.day).value

            # Create an epochs array to plot num_epochs_to_plot points over one orbital period
            epochs[i, :] = np.linspace(
                start_mjd, float(start_mjd + period[i]), num_epochs_to_plot
            )

            # Calculate ra/dec offsets for all epochs of this orbit
            raoff0, deoff0, _ = kepler.calc_orbit(
                epochs[i, :],
                sma[i],
                ecc[i],
                inc[i],
                aop[i],
                pan[i],
                tau[i],
                plx[i],
                mtot[i],
                tau_ref_epoch=results.tau_ref_epoch,
            )

            raoff[i, :] = raoff0 / 1000  # Convert to arcsec
            deoff[i, :] = deoff0 / 1000

        # Create a linearly increasing colormap for our range of epochs
        if cbar_param != "Epoch [year]":
            cbar_param_arr = results.post[:, index]
            norm = mpl.colors.Normalize(
                vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr)
            )
            norm_yr = mpl.colors.Normalize(
                vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr)
            )

        elif cbar_param == "Epoch [year]":

            min_cbar_date = np.min(epochs)
            max_cbar_date = np.mean(epochs[:, -1])

            min_cbar_date_bepoch = 1900 + (min_cbar_date - 15019.81352) / 365.242198781
            max_cbar_date_bepoch = 1900 + (max_cbar_date - 15019.81352) / 365.242198781

            # if we're plotting orbital periods greater than 1,000 yrs, limit the colorbar dynamic range
            #             if max_cbar_date - min_cbar_date > 1000 * 365.25:
            #                 max_cbar_date = min_cbar_date + 1000 * 365.25

            norm = mpl.colors.Normalize(vmin=min_cbar_date, vmax=max_cbar_date)

            norm_yr = mpl.colors.Normalize(
                vmin=Time(min_cbar_date, format="mjd").decimalyear,
                vmax=Time(max_cbar_date, format="mjd").decimalyear,
            )

        # Before starting to plot rv data, make sure rv data exists:
        rv_indices = np.where(data["quant_type"] == "rv")
        if rv_time_series and len(rv_indices) == 0:
            warnings.warn("Unable to plot radial velocity data.")
            rv_time_series = False

        # Create figure for orbit plots
        if fig is None:
            fig = plt.figure(figsize=(14, 6))
            if rv_time_series:
                #fig = plt.figure(figsize=(28, 18))
                fig = plt.figure(figsize=(14, 9))
                # spans 11 rows out of 18, and 6 columns out of 14
                ax = plt.subplot2grid((40, 16), (0, 0), rowspan=23, colspan=6)# orbits axis
            else:
                fig = plt.figure(figsize=(14, 6))
                ax = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=6)
        else:
            plt.set_current_figure(fig)
            if rv_time_series:
                ax = plt.subplot2grid((19, 16), (0, 0), rowspan=11, colspan=6)
            else:
                ax = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=6)

        astr_inds = np.where((~np.isnan(data["quant1"])) & (~np.isnan(data["quant2"])))
        astr_epochs = data["epoch"][astr_inds]

        radec_inds = np.where(data["quant_type"] == "radec")
        seppa_inds = np.where(data["quant_type"] == "seppa")

        # transform RA/Dec points to Sep/PA
        sep_data = np.copy(data["quant1"])
        sep_err = np.copy(data["quant1_err"])
        pa_data = np.copy(data["quant2"])
        pa_err = np.copy(data["quant2_err"])

        if len(radec_inds[0] > 0):

            sep_from_ra_data, pa_from_dec_data = orbitize.system.radec2seppa(
                data["quant1"][radec_inds], data["quant2"][radec_inds]
            )

            num_radec_pts = len(radec_inds[0])
            sep_err_from_ra_data = np.empty(num_radec_pts)
            pa_err_from_dec_data = np.empty(num_radec_pts)
            for j in np.arange(num_radec_pts):

                (
                    sep_err_from_ra_data[j],
                    pa_err_from_dec_data[j],
                    _,
                ) = orbitize.system.transform_errors(
                    np.array(data["quant1"][radec_inds][j]),
                    np.array(data["quant2"][radec_inds][j]),
                    np.array(data["quant1_err"][radec_inds][j]),
                    np.array(data["quant2_err"][radec_inds][j]),
                    np.array(data["quant12_corr"][radec_inds][j]),
                    orbitize.system.radec2seppa,
                )

            sep_data[radec_inds] = sep_from_ra_data
            sep_err[radec_inds] = sep_err_from_ra_data

            pa_data[radec_inds] = pa_from_dec_data
            pa_err[radec_inds] = pa_err_from_dec_data

        # Transform Sep/PA points to RA/Dec
        ra_data = np.copy(data["quant1"])
        ra_err = np.copy(data["quant1_err"])
        dec_data = np.copy(data["quant2"])
        dec_err = np.copy(data["quant2_err"])

        if len(seppa_inds[0] > 0):

            ra_from_seppa_data, dec_from_seppa_data = orbitize.system.seppa2radec(
                data["quant1"][seppa_inds], data["quant2"][seppa_inds]
            )

            num_seppa_pts = len(seppa_inds[0])
            ra_err_from_seppa_data = np.empty(num_seppa_pts)
            dec_err_from_seppa_data = np.empty(num_seppa_pts)
            for j in np.arange(num_seppa_pts):

                (
                    ra_err_from_seppa_data[j],
                    dec_err_from_seppa_data[j],
                    _,
                ) = orbitize.system.transform_errors(
                    np.array(data["quant1"][seppa_inds][j]),
                    np.array(data["quant2"][seppa_inds][j]),
                    np.array(data["quant1_err"][seppa_inds][j]),
                    np.array(data["quant2_err"][seppa_inds][j]),
                    np.array(data["quant12_corr"][seppa_inds][j]),
                    orbitize.system.seppa2radec,
                )

            ra_data[seppa_inds] = ra_from_seppa_data
            ra_err[seppa_inds] = ra_err_from_seppa_data

            dec_data[seppa_inds] = dec_from_seppa_data
            dec_err[seppa_inds] = dec_err_from_seppa_data

        # For plotting different astrometry instruments
        if plot_astrometry_insts:
            astr_colors = (
                "purple",
                "#FF7F11",
                "#11FFE3",
                "#14FF11",
                "#7A11FF",
                "#FF1919",
            )
            astr_symbols = ("o", "*", "p", "s")

            ax_colors = itertools.cycle(astr_colors)
            ax_symbols = itertools.cycle(astr_symbols)

            astr_data = data[astr_inds]
            astr_insts = np.unique(data[astr_inds]["instrument"])

            # Indices corresponding to each instrument in datafile
            astr_inst_inds = {}
            for i in range(len(astr_insts)):
                astr_inst_inds[astr_insts[i]] = np.where(
                    astr_data["instrument"] == astr_insts[i].encode()
                )[0]

        # Plot each orbit (each segment between two points coloured using colormap)
        for i in np.arange(num_orbits_to_plot):
            points = np.array([raoff[i, :], deoff[i, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.0)
            if cbar_param != "Epoch [year]":
                lc.set_array(np.ones(len(epochs[0])) * cbar_param_arr[i])
            elif cbar_param == "Epoch [year]":
                lc.set_array(epochs[i, :])
            ax.add_collection(lc)

        if plot_astrometry:

            # Plot astrometry along with instruments
            if plot_astrometry_insts:
                for i in range(len(astr_insts)):
                    ra = ra_data[astr_inst_inds[astr_insts[i]]]
                    dec = dec_data[astr_inst_inds[astr_insts[i]]]
                    if plot_errorbars:
                        xerr = ra_err[astr_inst_inds[astr_insts[i]]]
                        yerr = dec_err[astr_inst_inds[astr_insts[i]]]
                    else:
                        xerr = None
                        yerr = None

                    ax.errorbar(
                        ra/1000,
                        dec/1000,
                        xerr=xerr/1000,
                        yerr=yerr/1000,
                        marker=next(ax_symbols),
                        c=next(ax_colors),
                        zorder=10,
                        label=astr_insts[i],
                        linestyle="",
                        ms=ms,
                        capsize=capsize,elinewidth=elinewidth,capthick=capthick
                    )
            else:
                if plot_errorbars:
                    xerr = ra_err
                    yerr = dec_err
                else:
                    xerr = None
                    yerr = None

                ax.errorbar(
                    ra_data/1000,
                    dec_data/1000,
                    xerr=xerr/1000,
                    yerr=yerr/1000,
                    marker="o",
                    c="k",
                    zorder=10,
                    linestyle="",
                    ms=ms,capsize=capsize,elinewidth=elinewidth,capthick=capthick
                )

        # modify the axes
        if square_plot:
            adjustable_param = "datalim"
        else:
            adjustable_param = "box"
        ax.set_aspect("equal", adjustable=adjustable_param)
        ax.set_xlabel("$\\Delta$RA [arcsec]")
        ax.set_ylabel("$\\Delta$Dec [arcsec]")
        ax.locator_params(axis="x", nbins=6)
        ax.locator_params(axis="y", nbins=6)
        ax.invert_xaxis()  # To go to a left-handed coordinate system

        # plot sep/PA and/or rv zoom-in panels
        if rv_time_series:
            # sep vs. time
            ax1 = plt.subplot2grid((40, 16), (0, 10), colspan=6, rowspan=11)
            ax1.tick_params(labelbottom=False) # no year numbers, they're below
            
            # pa vs. time
            ax2 = plt.subplot2grid((40, 16), (12, 10), colspan=6, rowspan=11)
            
            # RV vs. time
            ax3 = plt.subplot2grid((40, 16), (29, 0), colspan=16, rowspan=12)
            
            ax2.set_ylabel("PA [$^{{\\circ}}$]")
            ax1.set_ylabel("$\\rho$ [mas]")
            ax3.set_ylabel("RV [km/s]")
            ax3.set_xlabel("Epoch")
            ax2.set_xlabel("Epoch")
            plt.subplots_adjust(hspace=0.3)
        else:
            ax1 = plt.subplot2grid((2, 14), (0, 9), colspan=6)
            ax2 = plt.subplot2grid((2, 14), (1, 9), colspan=6)
            ax2.set_ylabel("PA [$^{{\\circ}}$]")
            ax1.set_ylabel("$\\rho$ [mas]")
            ax2.set_xlabel("Epoch")

        if plot_astrometry_insts:
            ax1_colors = itertools.cycle(astr_colors)
            ax1_symbols = itertools.cycle(astr_symbols)

            ax2_colors = itertools.cycle(astr_colors)
            ax2_symbols = itertools.cycle(astr_symbols)

        epochs_seppa = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        for i in np.arange(num_orbits_to_plot):

            epochs_seppa[i, :] = np.linspace(
                start_mjd,
                Time(sep_pa_end_year, format="decimalyear").mjd,
                num_epochs_to_plot,
            )

            # Calculate ra/dec offsets for all epochs of this orbit
            if rv_time_series:
                raoff0, deoff0, _ = kepler.calc_orbit(
                    epochs_seppa[i, :],
                    sma[i],
                    ecc[i],
                    inc[i],
                    aop[i],
                    pan[i],
                    tau[i],
                    plx[i],
                    mtot[i],
                    tau_ref_epoch=results.tau_ref_epoch,
                    mass_for_Kamp=m0[i],
                )

                raoff[i, :] = raoff0
                deoff[i, :] = deoff0
            else:
                raoff0, deoff0, _ = kepler.calc_orbit(
                    epochs_seppa[i, :],
                    sma[i],
                    ecc[i],
                    inc[i],
                    aop[i],
                    pan[i],
                    tau[i],
                    plx[i],
                    mtot[i],
                    tau_ref_epoch=results.tau_ref_epoch,
                )

                raoff[i, :] = raoff0
                deoff[i, :] = deoff0

            yr_epochs = Time(epochs_seppa[i, :], format="mjd").decimalyear

            seps, pas = orbitize.system.radec2seppa(
                raoff[i, :], deoff[i, :], mod180=mod180
            )

            plt.sca(ax1)

            seps /= 1000  # mas to arcsec
            plt.plot(yr_epochs, seps, color=sep_pa_color, zorder=1)

            plt.sca(ax2)
            plt.plot(yr_epochs, pas, color=sep_pa_color, zorder=1)

        # Plot sep/pa instruments
        if plot_astrometry_insts:
            for i in range(len(astr_insts)):
                sep = sep_data[astr_inst_inds[astr_insts[i]]]
                pa = pa_data[astr_inst_inds[astr_insts[i]]]
                epochs = astr_epochs[astr_inst_inds[astr_insts[i]]]
                if plot_errorbars:
                    serr = sep_err[astr_inst_inds[astr_insts[i]]]
                    perr = pa_err[astr_inst_inds[astr_insts[i]]]
                else:
                    yerr = None
                    perr = None

                plt.sca(ax1)
                plt.errorbar(
                    Time(epochs, format="mjd").decimalyear,
                    sep,
                    yerr=serr,
                    ms=ms,
                    linestyle="",
                    marker=next(ax1_symbols),
                    c=next(ax1_colors),
                    zorder=10,
                    label=astr_insts[i],
                    capsize=capsize,elinewidth=elinewidth,capthick=capthick
                )
                plt.sca(ax2)
                plt.errorbar(
                    Time(epochs, format="mjd").decimalyear,
                    pa,
                    yerr=perr,
                    ms=ms,
                    linestyle="",
                    marker=next(ax2_symbols),
                    c=next(ax2_colors),
                    zorder=10,
                    capsize=capsize,elinewidth=elinewidth,capthick=capthick
                )
            plt.sca(ax1)
            plt.legend(title="Instruments", bbox_to_anchor=(1.3, 1), loc="upper right")
        else:
            if plot_errorbars:
                serr = sep_err
                perr = pa_err
            else:
                yerr = None
                perr = None

            sep_data /= 1000  # mas to arcsec
            serr /= 1000  # mas to arcsec

            plt.sca(ax1)  # set current axis
            for j, epoch_time in enumerate(Time(astr_epochs, format="mjd").decimalyear):

                plt.errorbar(
                    epoch_time,
                    sep_data[j],
                    yerr=serr[j],
                    ms=ms,
                    linestyle="",
                    marker="o",
                    c=cmap(
                        (epoch_time - min_cbar_date_bepoch)
                        / (max_cbar_date_bepoch - min_cbar_date_bepoch)
                    ),
                    zorder=2,
                    capsize=capsize,elinewidth=elinewidth,capthick=capthick
                )

            plt.sca(ax2)
            for k, epoch_time in enumerate(Time(astr_epochs, format="mjd").decimalyear):
                plt.errorbar(
                    epoch_time,
                    pa_data[k],
                    yerr=perr[k],
                    ms=ms,
                    linestyle="",
                    marker="o",
                    c=cmap(
                        (epoch_time - min_cbar_date_bepoch)
                        / (max_cbar_date_bepoch - min_cbar_date_bepoch)
                    ),
                    zorder=2,
                    capsize=capsize,elinewidth=elinewidth,capthick=capthick
                )

        if rv_time_series:

            rv_data = results.data[results.data["object"] == 0]
            rv_data = rv_data[rv_data["quant_type"] == "rv"]

            # switch current axis to rv panel
            plt.sca(ax3)

            # get list of rv instruments
            insts = np.unique(rv_data["instrument"])
            if len(insts) == 0:
                insts = ["defrv"]

            # get gamma/sigma labels and corresponding positions in the posterior
            gams = ["gamma_" + inst for inst in insts]
            sigs = ["sigma_" + inst for inst in insts]

            if isinstance(results.labels, list):
                labels = np.array(results.labels)
            else:
                labels = results.labels

            # get the indices corresponding to each gamma within results.labels
            gam_idx = [np.where(labels == inst_gamma)[0] for inst_gamma in gams]

            # indices corresponding to each instrument in the datafile
            inds = {}
            for i in range(len(insts)):
                inds[insts[i]] = np.where(rv_data["instrument"] == insts[i].encode())[0]

            # choose the orbit with the best log probability
            best_like = np.where(results.lnlike == np.amax(results.lnlike))[0][0]

            # Get the posteriors for this index and convert to standard basis
            best_post = results.basis.to_standard_basis(results.post[best_like].copy())

            # Get the masses for the best posteriors:
            best_m0 = best_post[results.standard_param_idx["m0"]]
            best_m1 = best_post[
                results.standard_param_idx["m{}".format(object_to_plot)]
            ]
            best_mtot = best_m0 + best_m1

            # colour/shape scheme scheme for rv data points
            clrs = ("purple", "#0496FF", "#372554", "#FF1053", "#3A7CA5", "#143109")
            symbols = ("o", "^", "v", "s")

            ax3_colors = itertools.cycle(clrs)
            ax3_symbols = itertools.cycle(symbols)

            # get rvs and plot them
            min_epoch_i = 3000# Bessellian big number
            max_epoch_i = 1000# Bessellian small number
            for i, name in enumerate(inds.keys()):
                inst_data = rv_data[inds[name]]
                rvs = inst_data["quant1"]
                epochs = inst_data["epoch"]
                epochs = Time(epochs, format="mjd").decimalyear
                min_epoch_i = np.min([min_epoch_i, np.min(epochs)])
                max_epoch_i = np.max([max_epoch_i, np.max(epochs)])
                rvs -= best_post[results.param_idx[gams[i]]]
                if plot_errorbars:
                    yerr = inst_data["quant1_err"]
                    yerr = np.sqrt(
                        yerr ** 2 + best_post[results.param_idx[sigs[i]]] ** 2
                    )
                for i,epoch in enumerate(epochs):
                    plt.errorbar(
                        epoch,
                        rvs[i],
                        yerr=yerr[i],
                        ms=ms,
                        linestyle="",
                        marker="o",
                        c=cmap(
                            (epoch - min_cbar_date_bepoch)
                            / (max_cbar_date_bepoch - min_cbar_date_bepoch)
                        ),
                        label=name,
                        zorder=5,
                        capsize=capsize,elinewidth=elinewidth,capthick=capthick
                        )
            if len(inds.keys()) == 1 and "defrv" in inds.keys():
                pass
            else:
                plt.legend()

            # calculate the predicted rv trend using the best orbit
            # update: provide manual epochs for RV plot
            # epochs_seppa[i, :] = np.linspace(
            #    start_mjd,
            #    Time(sep_pa_end_year, format="decimalyear").mjd,
            #    num_epochs_to_plot,
            #)
            my_min_epoch = np.min([start_mjd, Time(min_epoch_i, format='decimalyear').mjd])
            my_max_epoch = np.max([np.max(epochs_seppa[0, :]), Time(max_epoch_i, format='decimalyear').mjd])
            
            epochs_rv = np.linspace(my_min_epoch, my_max_epoch, int(num_epochs_to_plot*(my_max_epoch-my_min_epoch)/(np.max(epochs_seppa[0,:])-np.min(epochs_seppa[0,:]))))
            _, _, vz = kepler.calc_orbit(
                epochs_rv,
                best_post[results.standard_param_idx["sma{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["ecc{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["inc{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["aop{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["pan{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["tau{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["plx"]],
                best_mtot,
                tau_ref_epoch=results.tau_ref_epoch,
                mass_for_Kamp=best_m0,
            )

            vz = vz * -(best_m1) / np.median(best_m0)

            # plot rv trend
            plt.plot(
                Time(epochs_rv, format="mjd").decimalyear,
                vz,
                color=sep_pa_color,
                zorder=1,
            )

        # add colorbar
        if show_colorbar:
#            if rv_time_series:
            # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
            # You can change x1.0.05 to adjust the distance between the main image and the colorbar.
            # You can change 0.02 to adjust the width of the colorbar.
            cbar_ax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.005,
                    ax.get_position().y0,
                    0.02,
                    ax.get_position().height,
                ]
            )
            cbar = mpl.colorbar.ColorbarBase(
                cbar_ax,
                cmap=cmap,
                norm=norm_yr,
                orientation="vertical",
                label=cbar_param,
            )
#            else:
#                # xpos, ypos, width, height, in fraction of figure size
#                cbar_ax = fig.add_axes([0.47, 0.15, 0.015, 0.7])
#                cbar = mpl.colorbar.ColorbarBase(
#                    cbar_ax,
#                    cmap=cmap,
#                    norm=norm_yr,
#                    orientation="vertical",
#                    label=cbar_param,
#                )

        ax1.locator_params(axis="x", nbins=6)
        ax1.locator_params(axis="y", nbins=6)
        ax2.locator_params(axis="x", nbins=6)
        ax2.locator_params(axis="y", nbins=6)

        return fig, cbar


#### Mine: ##############


def my_plot_orbits(
    results,
    object_to_plot=1,
    cmap="viridis",
    start_mjd=51544.0,
    num_orbits_to_plot=100,
    num_epochs_to_plot=100,
    square_plot=True,
    show_colorbar=True,
    sep_pa_color="lightgrey",
    sep_pa_end_year=2025.0,
    cbar_param="Epoch [year]",
    mod180=False,
    rv_time_series=False,
    plot_astrometry=True,
    plot_astrometry_insts=False,
    plot_errorbars=True,
    fig=None,
    ms=5,
    capsize=2,
    elinewidth=1,
    capthick=1,
    fs=20,
    spine_w=2,
    tick_l=8,
    joint=False,
):
    """
    Plots one orbital period for a select number of fitted orbits
    for a given object, with line segments colored according to time

    Args:
        object_to_plot (int): which object to plot (default: 1)
        start_mjd (float): MJD in which to start plotting orbits (default: 51544,
            the year 2000)
        num_orbits_to_plot (int): number of orbits to plot (default: 100)
        num_epochs_to_plot (int): number of points to plot per orbit (default: 100)
        square_plot (Boolean): Aspect ratio is always equal, but if
            square_plot is True (default), then the axes will be square,
            otherwise, white space padding is used
        show_colorbar (Boolean): Displays colorbar to the right of the plot [True]
        cmap (matplotlib.cm.ColorMap): color map to use for making orbit tracks
            (default: modified Purples_r)
        sep_pa_color (string): any valid matplotlib color string, used to set the
            color of the orbit tracks in the Sep/PA panels (default: 'lightgrey').
        sep_pa_end_year (float): decimal year specifying when to stop plotting orbit
            tracks in the Sep/PA panels (default: 2025.0).
        cbar_param (string): options are the following: 'Epoch [year]', 'sma1', 'ecc1', 'inc1', 'aop1',
            'pan1', 'tau1', 'plx. Number can be switched out. Default is Epoch [year].
        mod180 (Bool): if True, PA will be plotted in range [180, 540]. Useful for plotting short
            arcs with PAs that cross 360 deg during observations (default: False)
        rv_time_series (Boolean): if fitting for secondary mass using MCMC for rv fitting and want to
            display time series, set to True.
        plot_astrometry (Boolean): set to True by default. Plots the astrometric data.
        plot_astrometry_insts (Boolean): set to False by default. Plots the astrometric data by instruments.
        plot_errorbars (Boolean): set to True by default. Plots error bars of measurements
        fig (matplotlib.pyplot.Figure): optionally include a predefined Figure object to plot the orbit on.
            Most users will not need this keyword. 

    Return:
        ``matplotlib.pyplot.Figure``: the orbit plot if input is valid, ``None`` otherwise


    (written): Henry Ngo, Sarah Blunt, 2018
    Additions by Malena Rice, 2019

    """

    if Time(start_mjd, format="mjd").decimalyear >= sep_pa_end_year:
        raise ValueError(
            "start_mjd keyword date must be less than sep_pa_end_year keyword date."
        )

    if object_to_plot > results.num_secondary_bodies:
        raise ValueError(
            "Only {0} secondary bodies being fit. Requested to plot body {1} which is out of range".format(
                results.num_secondary_bodies, object_to_plot
            )
        )

    if object_to_plot == 0:
        raise ValueError(
            "Plotting the primary's orbit is currently unsupported. Stay tuned."
        )

    if rv_time_series and "m0" not in results.labels:
        rv_time_series = False

        warnings.warn(
            "It seems that the stellar and companion mass "
            "have not been fitted separately. Setting "
            "rv_time_series=True is therefore not possible "
            "so the argument is set to False instead."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)

        data = results.data[results.data["object"] == object_to_plot]
        possible_cbar_params = ["sma", "ecc", "inc", "aop" "pan", "tau", "plx"]

        if cbar_param == "Epoch [year]":
            pass
        elif cbar_param[0:3] in possible_cbar_params:
            index = results.param_idx[cbar_param]
        else:
            raise Exception(
                "Invalid input; acceptable inputs include 'Epoch [year]', 'plx', 'sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'sma2', 'ecc2', ...)"
            )
        # Select random indices for plotted orbit
        num_orbits = len(results.post[:, 0])
        if num_orbits_to_plot > num_orbits:
            num_orbits_to_plot = num_orbits
        choose = np.random.randint(0, high=num_orbits, size=num_orbits_to_plot)

        # Get posteriors from random indices
        standard_post = []
        if results.sampler_name == "MCMC":
            # Convert the randomly chosen posteriors to standard keplerian set
            for i in np.arange(num_orbits_to_plot):
                orb_ind = choose[i]
                param_set = np.copy(results.post[orb_ind])
                standard_post.append(results.basis.to_standard_basis(param_set))
        else:  # For OFTI, posteriors are already converted
            for i in np.arange(num_orbits_to_plot):
                orb_ind = choose[i]
                standard_post.append(results.post[orb_ind])

        standard_post = np.array(standard_post)

        sma = standard_post[
            :, results.standard_param_idx["sma{}".format(object_to_plot)]
        ]
        ecc = standard_post[
            :, results.standard_param_idx["ecc{}".format(object_to_plot)]
        ]
        inc = standard_post[
            :, results.standard_param_idx["inc{}".format(object_to_plot)]
        ]
        aop = standard_post[
            :, results.standard_param_idx["aop{}".format(object_to_plot)]
        ]
        pan = standard_post[
            :, results.standard_param_idx["pan{}".format(object_to_plot)]
        ]
        tau = standard_post[
            :, results.standard_param_idx["tau{}".format(object_to_plot)]
        ]
        plx = standard_post[:, results.standard_param_idx["plx"]]

        # Then, get the other parameters
        if "mtot" in results.labels:
            mtot = standard_post[:, results.standard_param_idx["mtot"]]
        elif "m0" in results.labels:
            m0 = standard_post[:, results.standard_param_idx["m0"]]
            m1 = standard_post[
                :, results.standard_param_idx["m{}".format(object_to_plot)]
            ]
            mtot = m0 + m1

        raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        vz_star = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
        # Need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
        for i in np.arange(num_orbits_to_plot):
            # Compute period (from Kepler's third law)
            period = np.sqrt(
                4 * np.pi ** 2.0 * (sma * u.AU) ** 3 / (consts.G * (mtot * u.Msun))
            )
            period = period.to(u.day).value

            # Create an epochs array to plot num_epochs_to_plot points over one orbital period
            epochs[i, :] = np.linspace(
                start_mjd, float(start_mjd + period[i]), num_epochs_to_plot
            )

            # Calculate ra/dec offsets for all epochs of this orbit
            raoff0, deoff0, _ = kepler.calc_orbit(
                epochs[i, :],
                sma[i],
                ecc[i],
                inc[i],
                aop[i],
                pan[i],
                tau[i],
                plx[i],
                mtot[i],
                tau_ref_epoch=results.tau_ref_epoch,
            )

            raoff[i, :] = raoff0 / 1000  # Convert to arcsec
            deoff[i, :] = deoff0 / 1000

        # Create a linearly increasing colormap for our range of epochs
        if cbar_param != "Epoch [year]":
            cbar_param_arr = results.post[:, index]
            norm = mpl.colors.Normalize(
                vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr)
            )
            norm_yr = mpl.colors.Normalize(
                vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr)
            )

        elif cbar_param == "Epoch [year]":

            min_cbar_date = np.min(epochs)
            max_cbar_date = np.mean(epochs[:, -1])

            min_cbar_date_bepoch = 1900 + (min_cbar_date - 15019.81352) / 365.242198781
            max_cbar_date_bepoch = 1900 + (max_cbar_date - 15019.81352) / 365.242198781

            # if we're plotting orbital periods greater than 1,000 yrs, limit the colorbar dynamic range
            #               if max_cbar_date - min_cbar_date > 1000 * 365.25:
            #                   max_cbar_date = min_cbar_date + 1000 * 365.25

            norm = mpl.colors.Normalize(vmin=min_cbar_date, vmax=max_cbar_date)

            norm_yr = mpl.colors.Normalize(
                vmin=Time(min_cbar_date, format="mjd").decimalyear,
                vmax=Time(max_cbar_date, format="mjd").decimalyear,
            )

        # Before starting to plot rv data, make sure rv data exists:
        rv_indices = np.where(data["quant_type"] == "rv")
        if rv_time_series and len(rv_indices) == 0:
            warnings.warn("Unable to plot radial velocity data.")
            rv_time_series = False

        # Create figure for orbit plots
        if fig is None:
            fig = plt.figure(figsize=(14, 6))
            if rv_time_series:
                # fig = plt.figure(figsize=(28, 18))
                fig = plt.figure(figsize=(14, 9))
                # spans 11 rows out of 18, and 6 columns out of 14
                ax = plt.subplot2grid(
                    (40, 16), (0, 0), rowspan=23, colspan=6
                )  # orbits axis
            else:
                fig = plt.figure(figsize=(14, 6))
                ax = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=6)
        else:
            plt.set_current_figure(fig)
            if rv_time_series:
                ax = plt.subplot2grid((19, 16), (0, 0), rowspan=11, colspan=6)
            else:
                ax = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=6)

        astr_inds = np.where((~np.isnan(data["quant1"])) & (~np.isnan(data["quant2"])))
        astr_epochs = data["epoch"][astr_inds]

        radec_inds = np.where(data["quant_type"] == "radec")
        seppa_inds = np.where(data["quant_type"] == "seppa")

        # transform RA/Dec points to Sep/PA
        sep_data = np.copy(data["quant1"])
        sep_err = np.copy(data["quant1_err"])
        pa_data = np.copy(data["quant2"])
        pa_err = np.copy(data["quant2_err"])

        if len(radec_inds[0] > 0):

            sep_from_ra_data, pa_from_dec_data = orbitize.system.radec2seppa(
                data["quant1"][radec_inds], data["quant2"][radec_inds]
            )

            num_radec_pts = len(radec_inds[0])
            sep_err_from_ra_data = np.empty(num_radec_pts)
            pa_err_from_dec_data = np.empty(num_radec_pts)
            for j in np.arange(num_radec_pts):

                (
                    sep_err_from_ra_data[j],
                    pa_err_from_dec_data[j],
                    _,
                ) = orbitize.system.transform_errors(
                    np.array(data["quant1"][radec_inds][j]),
                    np.array(data["quant2"][radec_inds][j]),
                    np.array(data["quant1_err"][radec_inds][j]),
                    np.array(data["quant2_err"][radec_inds][j]),
                    np.array(data["quant12_corr"][radec_inds][j]),
                    orbitize.system.radec2seppa,
                )

            sep_data[radec_inds] = sep_from_ra_data
            sep_err[radec_inds] = sep_err_from_ra_data

            pa_data[radec_inds] = pa_from_dec_data
            pa_err[radec_inds] = pa_err_from_dec_data

        # Transform Sep/PA points to RA/Dec
        ra_data = np.copy(data["quant1"])
        ra_err = np.copy(data["quant1_err"])
        dec_data = np.copy(data["quant2"])
        dec_err = np.copy(data["quant2_err"])

        if len(seppa_inds[0] > 0):

            ra_from_seppa_data, dec_from_seppa_data = orbitize.system.seppa2radec(
                data["quant1"][seppa_inds], data["quant2"][seppa_inds]
            )

            num_seppa_pts = len(seppa_inds[0])
            ra_err_from_seppa_data = np.empty(num_seppa_pts)
            dec_err_from_seppa_data = np.empty(num_seppa_pts)
            for j in np.arange(num_seppa_pts):

                (
                    ra_err_from_seppa_data[j],
                    dec_err_from_seppa_data[j],
                    _,
                ) = orbitize.system.transform_errors(
                    np.array(data["quant1"][seppa_inds][j]),
                    np.array(data["quant2"][seppa_inds][j]),
                    np.array(data["quant1_err"][seppa_inds][j]),
                    np.array(data["quant2_err"][seppa_inds][j]),
                    np.array(data["quant12_corr"][seppa_inds][j]),
                    orbitize.system.seppa2radec,
                )

            ra_data[seppa_inds] = ra_from_seppa_data
            ra_err[seppa_inds] = ra_err_from_seppa_data

            dec_data[seppa_inds] = dec_from_seppa_data
            dec_err[seppa_inds] = dec_err_from_seppa_data

        # For plotting different astrometry instruments
        if plot_astrometry_insts:
            astr_colors = (
                "purple",
                "#FF7F11",
                "#11FFE3",
                "#14FF11",
                "#7A11FF",
                "#FF1919",
            )
            astr_symbols = ("o", "*", "p", "s")

            ax_colors = itertools.cycle(astr_colors)
            ax_symbols = itertools.cycle(astr_symbols)

            astr_data = data[astr_inds]
            astr_insts = np.unique(data[astr_inds]["instrument"])

            # Indices corresponding to each instrument in datafile
            astr_inst_inds = {}
            for i in range(len(astr_insts)):
                astr_inst_inds[astr_insts[i]] = np.where(
                    astr_data["instrument"] == astr_insts[i].encode()
                )[0]

        # Plot each orbit (each segment between two points coloured using colormap)
        for i in np.arange(num_orbits_to_plot):
            points = np.array([raoff[i, :], deoff[i, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.0)
            if cbar_param != "Epoch [year]":
                lc.set_array(np.ones(len(epochs[0])) * cbar_param_arr[i])
            elif cbar_param == "Epoch [year]":
                lc.set_array(epochs[i, :])
            ax.add_collection(lc)

        if plot_astrometry:

            # Plot astrometry along with instruments
            if plot_astrometry_insts:
                for i in range(len(astr_insts)):
                    ra = ra_data[astr_inst_inds[astr_insts[i]]]
                    dec = dec_data[astr_inst_inds[astr_insts[i]]]
                    if plot_errorbars:
                        xerr = ra_err[astr_inst_inds[astr_insts[i]]]
                        yerr = dec_err[astr_inst_inds[astr_insts[i]]]
                    else:
                        xerr = None
                        yerr = None

                    ax.errorbar(
                        ra / 1000,
                        dec / 1000,
                        xerr=xerr / 1000,
                        yerr=yerr / 1000,
                        marker=next(ax_symbols),
                        c=next(ax_colors),
                        zorder=10,
                        label=astr_insts[i],
                        linestyle="",
                        ms=ms,
                        capsize=capsize,
                        elinewidth=elinewidth,
                        capthick=capthick,
                    )
            else:
                if plot_errorbars:
                    xerr = ra_err
                    yerr = dec_err
                else:
                    xerr = None
                    yerr = None

                ax.errorbar(
                    ra_data / 1000,
                    dec_data / 1000,
                    xerr=xerr / 1000,
                    yerr=yerr / 1000,
                    marker="o",
                    c="k",
                    zorder=10,
                    linestyle="",
                    ms=ms,
                    capsize=capsize,
                    elinewidth=elinewidth,
                    capthick=capthick,
                )

        # modify the axes
        if square_plot:
            adjustable_param = "datalim"
        else:
            adjustable_param = "box"
        ax.set_aspect("equal", adjustable=adjustable_param)
        ax.set_xlabel("$\\Delta$RA [arcsec]")
        ax.set_ylabel("$\\Delta$Dec [arcsec]")
        ax.locator_params(axis="x", nbins=6)
        ax.locator_params(axis="y", nbins=6)
        ax.invert_xaxis()  # To go to a left-handed coordinate system

        # plot sep/PA and/or rv zoom-in panels
        if rv_time_series:
            # sep vs. time
            ax1 = plt.subplot2grid((40, 16), (0, 10), colspan=6, rowspan=11)
            ax1.tick_params(labelbottom=False)  # no year numbers, they're below

            # pa vs. time
            ax2 = plt.subplot2grid((40, 16), (12, 10), colspan=6, rowspan=11)

            # RV vs. time
            ax3 = plt.subplot2grid((40, 16), (29, 0), colspan=16, rowspan=12)

            ax2.set_ylabel("PA [$^{{\\circ}}$]")
            ax1.set_ylabel("$\\rho$ [mas]")
            ax3.set_ylabel("RV [km/s]")
            ax3.set_xlabel("Epoch")
            ax2.set_xlabel("Epoch")
            plt.subplots_adjust(hspace=0.3)
        else:
            ax1 = plt.subplot2grid((2, 14), (0, 9), colspan=6)
            ax2 = plt.subplot2grid((2, 14), (1, 9), colspan=6)
            ax2.set_ylabel("PA [$^{{\\circ}}$]")
            ax1.set_ylabel("$\\rho$ [mas]")
            ax2.set_xlabel("Epoch")

        if plot_astrometry_insts:
            ax1_colors = itertools.cycle(astr_colors)
            ax1_symbols = itertools.cycle(astr_symbols)

            ax2_colors = itertools.cycle(astr_colors)
            ax2_symbols = itertools.cycle(astr_symbols)

        epochs_seppa = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        for i in np.arange(num_orbits_to_plot):

            epochs_seppa[i, :] = np.linspace(
                start_mjd,
                Time(sep_pa_end_year, format="decimalyear").mjd,
                num_epochs_to_plot,
            )

            # Calculate ra/dec offsets for all epochs of this orbit
            if rv_time_series:
                raoff0, deoff0, _ = kepler.calc_orbit(
                    epochs_seppa[i, :],
                    sma[i],
                    ecc[i],
                    inc[i],
                    aop[i],
                    pan[i],
                    tau[i],
                    plx[i],
                    mtot[i],
                    tau_ref_epoch=results.tau_ref_epoch,
                    mass_for_Kamp=m0[i],
                )

                raoff[i, :] = raoff0
                deoff[i, :] = deoff0
            else:
                raoff0, deoff0, _ = kepler.calc_orbit(
                    epochs_seppa[i, :],
                    sma[i],
                    ecc[i],
                    inc[i],
                    aop[i],
                    pan[i],
                    tau[i],
                    plx[i],
                    mtot[i],
                    tau_ref_epoch=results.tau_ref_epoch,
                )

                raoff[i, :] = raoff0
                deoff[i, :] = deoff0

            yr_epochs = Time(epochs_seppa[i, :], format="mjd").decimalyear

            seps, pas = orbitize.system.radec2seppa(
                raoff[i, :], deoff[i, :], mod180=mod180
            )

            plt.sca(ax1)

            seps /= 1000  # mas to arcsec
            plt.plot(yr_epochs, seps, color=sep_pa_color, zorder=1)

            plt.sca(ax2)
            plt.plot(yr_epochs, pas, color=sep_pa_color, zorder=1)

        # Plot sep/pa instruments
        if plot_astrometry_insts:
            for i in range(len(astr_insts)):
                sep = sep_data[astr_inst_inds[astr_insts[i]]]
                pa = pa_data[astr_inst_inds[astr_insts[i]]]
                epochs = astr_epochs[astr_inst_inds[astr_insts[i]]]
                if plot_errorbars:
                    serr = sep_err[astr_inst_inds[astr_insts[i]]]
                    perr = pa_err[astr_inst_inds[astr_insts[i]]]
                else:
                    yerr = None
                    perr = None

                plt.sca(ax1)
                plt.errorbar(
                    Time(epochs, format="mjd").decimalyear,
                    sep,
                    yerr=serr,
                    ms=ms,
                    linestyle="",
                    marker=next(ax1_symbols),
                    c=next(ax1_colors),
                    zorder=10,
                    label=astr_insts[i],
                    capsize=capsize,
                    elinewidth=elinewidth,
                    capthick=capthick,
                )
                plt.sca(ax2)
                plt.errorbar(
                    Time(epochs, format="mjd").decimalyear,
                    pa,
                    yerr=perr,
                    ms=ms,
                    linestyle="",
                    marker=next(ax2_symbols),
                    c=next(ax2_colors),
                    zorder=10,
                    capsize=capsize,
                    elinewidth=elinewidth,
                    capthick=capthick,
                )
            plt.sca(ax1)
            plt.legend(title="Instruments", bbox_to_anchor=(1.3, 1), loc="upper right")
        else:
            if plot_errorbars:
                serr = sep_err
                perr = pa_err
            else:
                yerr = None
                perr = None

            sep_data /= 1000  # mas to arcsec
            serr /= 1000  # mas to arcsec

            plt.sca(ax1)  # set current axis
            for j, epoch_time in enumerate(Time(astr_epochs, format="mjd").decimalyear):

                plt.errorbar(
                    epoch_time,
                    sep_data[j],
                    yerr=serr[j],
                    ms=ms,
                    linestyle="",
                    marker="o",
                    c=cmap(
                        (epoch_time - min_cbar_date_bepoch)
                        / (max_cbar_date_bepoch - min_cbar_date_bepoch)
                    ),
                    zorder=2,
                    capsize=capsize,
                    elinewidth=elinewidth,
                    capthick=capthick,
                )

            plt.sca(ax2)
            for k, epoch_time in enumerate(Time(astr_epochs, format="mjd").decimalyear):
                plt.errorbar(
                    epoch_time,
                    pa_data[k],
                    yerr=perr[k],
                    ms=ms,
                    linestyle="",
                    marker="o",
                    c=cmap(
                        (epoch_time - min_cbar_date_bepoch)
                        / (max_cbar_date_bepoch - min_cbar_date_bepoch)
                    ),
                    zorder=2,
                    capsize=capsize,
                    elinewidth=elinewidth,
                    capthick=capthick,
                )

        if rv_time_series:

            rv_data = results.data[results.data["object"] == 0]
            rv_data = rv_data[rv_data["quant_type"] == "rv"]

            # switch current axis to rv panel
            plt.sca(ax3)

            # get list of rv instruments
            insts = np.unique(rv_data["instrument"])
            if len(insts) == 0:
                insts = ["defrv"]

            # get gamma/sigma labels and corresponding positions in the posterior
            gams = ["gamma_" + inst for inst in insts]
            sigs = ["sigma_" + inst for inst in insts]

            if isinstance(results.labels, list):
                labels = np.array(results.labels)
            else:
                labels = results.labels

            # get the indices corresponding to each gamma within results.labels
            gam_idx = [np.where(labels == inst_gamma)[0] for inst_gamma in gams]

            # indices corresponding to each instrument in the datafile
            inds = {}
            for i in range(len(insts)):
                inds[insts[i]] = np.where(rv_data["instrument"] == insts[i].encode())[0]

            # choose the orbit with the best log probability
            best_like = np.where(results.lnlike == np.amax(results.lnlike))[0][0]

            # Get the posteriors for this index and convert to standard basis
            best_post = results.basis.to_standard_basis(results.post[best_like].copy())
            print(
                f"Posterior Orbital Elements for max. log-likelihood:\n\n{best_post}\n"
            )

            # Get the masses for the best posteriors:
            best_m0 = best_post[results.standard_param_idx["m0"]]
            best_m1 = best_post[
                results.standard_param_idx["m{}".format(object_to_plot)]
            ]
            best_mtot = best_m0 + best_m1

            # colour/shape scheme scheme for rv data points
            clrs = ("purple", "#0496FF", "#372554", "#FF1053", "#3A7CA5", "#143109")
            symbols = ("o", "^", "v", "s")

            ax3_colors = itertools.cycle(clrs)
            ax3_symbols = itertools.cycle(symbols)

            # get rvs and plot them
            min_epoch_i = 3000  # Bessellian big number
            max_epoch_i = 1000  # Bessellian small number
            for i, name in enumerate(inds.keys()):
                inst_data = rv_data[inds[name]]
                rvs = inst_data["quant1"]
                epochs = inst_data["epoch"]
                epochs = Time(epochs, format="mjd").decimalyear
                min_epoch_i = np.min([min_epoch_i, np.min(epochs)])
                max_epoch_i = np.max([max_epoch_i, np.max(epochs)])
                rvs -= best_post[results.param_idx[gams[i]]]
                if plot_errorbars:
                    yerr = inst_data["quant1_err"]
                    yerr = np.sqrt(
                        yerr ** 2 + best_post[results.param_idx[sigs[i]]] ** 2
                    )
                for i, epoch in enumerate(epochs):
                    plt.errorbar(
                        epoch,
                        rvs[i],
                        yerr=yerr[i],
                        ms=ms,
                        linestyle="",
                        marker="o",
                        c=cmap(
                            (epoch - min_cbar_date_bepoch)
                            / (max_cbar_date_bepoch - min_cbar_date_bepoch)
                        ),
                        label=name,
                        zorder=5,
                        capsize=capsize,
                        elinewidth=elinewidth,
                        capthick=capthick,
                    )
            if len(inds.keys()) == 1 and "defrv" in inds.keys():
                pass
            else:
                plt.legend()

            # calculate the predicted rv trend using the best orbit
            # update: provide manual epochs for RV plot
            # epochs_seppa[i, :] = np.linspace(
            #      start_mjd,
            #      Time(sep_pa_end_year, format="decimalyear").mjd,
            #      num_epochs_to_plot,
            # )
            my_min_epoch = np.min(
                [start_mjd, Time(min_epoch_i, format="decimalyear").mjd]
            )
            my_max_epoch = np.max(
                [
                    np.max(epochs_seppa[0, :]),
                    Time(max_epoch_i, format="decimalyear").mjd,
                ]
            )

            epochs_rv = np.linspace(
                my_min_epoch,
                my_max_epoch,
                int(
                    num_epochs_to_plot
                    * (my_max_epoch - my_min_epoch)
                    / (np.max(epochs_seppa[0, :]) - np.min(epochs_seppa[0, :]))
                ),
            )
            _, _, vz = kepler.calc_orbit(
                epochs_rv,
                best_post[results.standard_param_idx["sma{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["ecc{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["inc{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["aop{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["pan{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["tau{}".format(object_to_plot)]],
                best_post[results.standard_param_idx["plx"]],
                best_mtot,
                tau_ref_epoch=results.tau_ref_epoch,
                mass_for_Kamp=best_m0,
            )

            vz = vz * -(best_m1) / np.median(best_m0)

            # plot rv trend
            plt.plot(
                Time(epochs_rv, format="mjd").decimalyear,
                vz,
                color=sep_pa_color,
                zorder=1,
            )

        # add colorbar
        if show_colorbar:
            if rv_time_series:
                # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
                # You can change x1.0.05 to adjust the distance between the main image and the colorbar.
                # You can change 0.02 to adjust the width of the colorbar.
                cbar_ax = fig.add_axes(
                    [
                        ax.get_position().x1 + 0.005,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height,
                    ]
                )
                cbar = mpl.colorbar.ColorbarBase(
                    cbar_ax,
                    cmap=cmap,
                    norm=norm_yr,
                    orientation="vertical",
                    label=cbar_param,
                )
            else:
                # xpos, ypos, width, height, in fraction of figure size
                cbar_ax = fig.add_axes([0.47, 0.15, 0.015, 0.7])
                cbar = mpl.colorbar.ColorbarBase(
                    cbar_ax,
                    cmap=cmap,
                    norm=norm_yr,
                    orientation="vertical",
                    label=cbar_param,
                )

            cbar_ax.set_ylabel(r"\textbf{Year}", fontsize=fs)

            for spine in cbar_ax.spines.values():
                spine.set(visible=True, lw=spine_w, edgecolor="black")

                # bold ticks
            cbar_ax.tick_params(
                axis="both",
                which="major",
                labelsize=fs,
                width=spine_w,
                length=tick_l,
                direction="in",
                color="w",
            )

        ax1.locator_params(axis="x", nbins=6)
        ax1.locator_params(axis="y", nbins=6)
        ax2.locator_params(axis="x", nbins=6)
        ax2.locator_params(axis="y", nbins=6)

        # orbits
        ax.plot(0, 0, marker="*", color="k", markersize=16)
        ax.set_xlabel(r"$\bm{\Delta \alpha}$ \textbf{($\bm{''}$)}", fontsize=fs)
        ax.set_ylabel(r"$\bm{\Delta \delta}$ \textbf{($\bm{''}$)}", fontsize=fs)

        # sep v. time
        ax1.set_ylabel(r"$\bm{\rho}$ \textbf{($\bm{''}$)}", fontsize=fs)

        # pa v. time
        ax2.set_ylabel(r"\textbf{PA (\textdegree)}", fontsize=fs)
        ax2.set_xlabel(r"\textbf{Year}", fontsize=fs)

        # RVs
        if joint:
            # RVs y- and x-labels
            ax3.set_ylabel(
                r"\textbf{$\boldsymbol{\Delta}$RV ($\mathbf{\mathrm{km}/\mathrm{s}}$)}",
                fontsize=fs,
            )
            ax3.set_xlabel(r"\textbf{Year}", fontsize=fs)

            # bold spines, bold ticks
            for axis in ["left", "bottom", "top", "right"]:
                ax3.spines[axis].set_linewidth(spine_w)

            ax3.tick_params(
                axis="both",
                which="major",
                labelsize=fs,
                width=spine_w,
                length=tick_l,
                direction="in",
            )

        # BOLD SPINES
        # set the axis line width in pixels
        for axis in ["left", "bottom", "top", "right"]:
            ax.spines[axis].set_linewidth(spine_w)
            ax1.spines[axis].set_linewidth(spine_w)
            ax2.spines[axis].set_linewidth(spine_w)

        # BOLD TICKS
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=fs,
            width=spine_w,
            length=tick_l,
            direction="in",
        )
        ax1.tick_params(
            axis="both",
            which="major",
            labelsize=fs,
            width=spine_w,
            length=tick_l,
            direction="in",
        )
        ax2.tick_params(
            axis="both",
            which="major",
            labelsize=fs,
            width=spine_w,
            length=tick_l,
            direction="in",
        )

        return fig, min_cbar_date, max_cbar_date


#### Mine: ##############


def plot_orbits_from_post(
    post,
    lnlike,
    lab,
    tau_ref_epoch,
    data,
    sampler,
    labels=None,
    object_to_plot=1,
    start_mjd=51544.0,
    num_orbits_to_plot=100,
    num_epochs_to_plot=100,
    square_plot=True,
    show_colorbar=True,
    cmap='viridis',
    sep_pa_color="lightgrey",
    sep_pa_end_year=2025.0,
    cbar_param="Epoch [year]",
    mod180=False,
    rv_time_series=False,
    plot_astrometry=True,
    plot_astrometry_insts=False,
    plot_errorbars=True,
    fig=None,
    ms=5
):
    """
    Plots one orbital period for a select number of fitted orbits
    for a given object, with line segments colored according to time

    Args:
        object_to_plot (int): which object to plot (default: 1)
        start_mjd (float): MJD in which to start plotting orbits (default: 51544,
            the year 2000)
        num_orbits_to_plot (int): number of orbits to plot (default: 100)
        num_epochs_to_plot (int): number of points to plot per orbit (default: 100)
        square_plot (Boolean): Aspect ratio is always equal, but if
            square_plot is True (default), then the axes will be square,
            otherwise, white space padding is used
        show_colorbar (Boolean): Displays colorbar to the right of the plot [True]
        cmap (matplotlib.cm.ColorMap): color map to use for making orbit tracks
            (default: modified Purples_r)
        sep_pa_color (string): any valid matplotlib color string, used to set the
            color of the orbit tracks in the Sep/PA panels (default: 'lightgrey').
        sep_pa_end_year (float): decimal year specifying when to stop plotting orbit
            tracks in the Sep/PA panels (default: 2025.0).
        cbar_param (string): options are the following: 'Epoch [year]', 'sma1', 'ecc1', 'inc1', 'aop1',
            'pan1', 'tau1', 'plx. Number can be switched out. Default is Epoch [year].
        mod180 (Bool): if True, PA will be plotted in range [180, 540]. Useful for plotting short
            arcs with PAs that cross 360 deg during observations (default: False)
        rv_time_series (Boolean): if fitting for secondary mass using MCMC for rv fitting and want to
            display time series, set to True.
        plot_astrometry (Boolean): set to True by default. Plots the astrometric data.
        plot_astrometry_insts (Boolean): set to False by default. Plots the astrometric data by instruments.
        plot_errorbars (Boolean): set to True by default. Plots error bars of measurements
        fig (matplotlib.pyplot.Figure): optionally include a predefined Figure object to plot the orbit on.
            Most users will not need this keyword. 

    Return:
        ``matplotlib.pyplot.Figure``: the orbit plot if input is valid, ``None`` otherwise


    (written): Henry Ngo, Sarah Blunt, 2018
    Additions by Malena Rice, 2019

    """

    if Time(start_mjd, format="mjd").decimalyear >= sep_pa_end_year:
        raise ValueError(
            "start_mjd keyword date must be less than sep_pa_end_year keyword date."
        )

    # if object_to_plot > results.num_secondary_bodies:
    #       raise ValueError("Only {0} secondary bodies being fit. Requested to plot body {1} which is out of range".format(results.num_secondary_bodies, object_to_plot))

    if object_to_plot == 0:
        raise ValueError(
            "Plotting the primary's orbit is currently unsupported. Stay tuned."
        )

    if rv_time_series and "m0" not in lab:
        rv_time_series = False

        warnings.warn(
            "It seems that the stellar and companion mass "
            "have not been fitted separately. Setting "
            "rv_time_series=True is therefore not possible "
            "so the argument is set to False instead."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)

        data_obj = data[data["object"] == object_to_plot]
        possible_cbar_params = ["sma", "ecc", "inc", "aop" "pan", "tau", "plx"]

        if cbar_param == "Epoch [year]":
            pass
        elif cbar_param[0:3] in possible_cbar_params:
            index = lab[cbar_param]
        else:
            raise Exception(
                "Invalid input; acceptable inputs include 'Epoch [year]', 'plx', 'sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'sma2', 'ecc2', ...)"
            )
        # Select random indices for plotted orbit
        num_orbits = len(post[:, 0])
        if num_orbits_to_plot > num_orbits:
            num_orbits_to_plot = num_orbits
        choose = np.random.randint(0, high=num_orbits, size=num_orbits_to_plot)

        # Get posteriors from random indices
        standard_post = []
        # if sampler == 'MCMC':
        # Convert the randomly chosen posteriors to standard keplerian set
        #      for i in np.arange(num_orbits_to_plot):
        #          orb_ind = choose[i]
        #          param_set = np.copy(results.post[orb_ind])
        #      standard_post.append(my_to_standard_basis(param_set))

        # if sampler == 'OFTI':
        for i in np.arange(num_orbits_to_plot):
            orb_ind = choose[i]
            standard_post.append(post[orb_ind])

        standard_post = np.array(standard_post)

        sma = standard_post[:, lab["sma{}".format(object_to_plot)]]
        ecc = standard_post[:, lab["ecc{}".format(object_to_plot)]]
        inc = standard_post[:, lab["inc{}".format(object_to_plot)]]
        aop = standard_post[:, lab["aop{}".format(object_to_plot)]]
        pan = standard_post[:, lab["pan{}".format(object_to_plot)]]
        tau = standard_post[:, lab["tau{}".format(object_to_plot)]]
        plx = standard_post[:, lab["plx"]]

        # Then, get the other parameters
        if "mtot" in lab:
            mtot = standard_post[:, lab["mtot"]]
        elif "m0" in lab:
            m0 = standard_post[:, lab["m0"]]
            m1 = standard_post[:, lab["m{}".format(object_to_plot)]]
            mtot = m0 + m1

        raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        vz_star = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
        # Need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
        for i in np.arange(num_orbits_to_plot):
            # Compute period (from Kepler's third law)
            period = np.sqrt(
                4 * np.pi ** 2.0 * (sma * u.AU) ** 3 / (consts.G * (mtot * u.Msun))
            )
            period = period.to(u.day).value

            # Create an epochs array to plot num_epochs_to_plot points over one orbital period
            epochs[i, :] = np.linspace(
                start_mjd, float(start_mjd + period[i]), num_epochs_to_plot
            )

            # Calculate ra/dec offsets for all epochs of this orbit
            raoff0, deoff0, vz_star0 = kepler.calc_orbit(
                epochs[i, :],
                sma[i],
                ecc[i],
                inc[i],
                aop[i],
                pan[i],
                tau[i],
                plx[i],
                mtot[i],
                mass_for_Kamp=m1[
                    i
                ],  # adding this so that the mass of Procyon B is creating the stellar RV signal
                tau_ref_epoch=tau_ref_epoch,
            )

            raoff[i, :] = raoff0 / 1000  # Convert to arcsec
            deoff[i, :] = deoff0 / 1000
            vz_star[i, :] = vz_star0  # km/s

        # Create a linearly increasing colormap for our range of epochs
        if cbar_param != "Epoch [year]":
            cbar_param_arr = post[:, index]
            norm = mpl.colors.Normalize(
                vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr)
            )
            norm_yr = mpl.colors.Normalize(
                vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr)
            )

        elif cbar_param == "Epoch [year]":

            min_cbar_date = np.min(epochs)
            max_cbar_date = np.mean(epochs[:, -1])
            min_cbar_date_bepoch = 1900 + (min_cbar_date - 15019.81352) / 365.242198781
            max_cbar_date_bepoch = 1900 + (max_cbar_date - 15019.81352) / 365.242198781

            # if we're plotting orbital periods greater than 1,000 yrs, limit the colorbar dynamic range
            #               if max_cbar_date - min_cbar_date > 1000 * 365.25:
            #                   max_cbar_date = min_cbar_date + 1000 * 365.25

            norm = mpl.colors.Normalize(vmin=min_cbar_date, vmax=max_cbar_date)

            norm_yr = mpl.colors.Normalize(
                vmin=Time(min_cbar_date, format="mjd").decimalyear,
                vmax=Time(max_cbar_date, format="mjd").decimalyear,
            )

        # Before starting to plot rv data, make sure rv data exists:
        rv_indices = np.where(data["quant_type"] == "rv")
        if rv_time_series and len(rv_indices) == 0:
            warnings.warn("Unable to plot radial velocity data.")
            rv_time_series = False

        # Create figure for orbit plots
        if fig is None:
            fig = plt.figure(figsize=(14, 6))
            if rv_time_series:
                fig = plt.figure(figsize=(14, 9))
                ax = plt.subplot2grid((3, 14), (0, 0), rowspan=2, colspan=6)
            else:
                fig = plt.figure(figsize=(14, 6))
                ax = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=6)
        else:
            plt.set_current_figure(fig)
            if rv_time_series:
                ax = plt.subplot2grid((3, 14), (0, 0), rowspan=2, colspan=6)
            else:
                ax = plt.subplot2grid((2, 14), (0, 0), rowspan=2, colspan=6)

        astr_inds = np.where((~np.isnan(data["quant1"])) & (~np.isnan(data["quant2"])))
        astr_epochs = data["epoch"][astr_inds]

        radec_inds = np.where(data["quant_type"] == "radec")
        seppa_inds = np.where(data["quant_type"] == "seppa")

        # transform RA/Dec points to Sep/PA
        sep_data = np.copy(data["quant1"])
        sep_err = np.copy(data["quant1_err"])
        pa_data = np.copy(data["quant2"])
        pa_err = np.copy(data["quant2_err"])

        if len(radec_inds[0] > 0):

            sep_from_ra_data, pa_from_dec_data = orbitize.system.radec2seppa(
                data["quant1"][radec_inds], data["quant2"][radec_inds]
            )

            num_radec_pts = len(radec_inds[0])
            sep_err_from_ra_data = np.empty(num_radec_pts)
            pa_err_from_dec_data = np.empty(num_radec_pts)
            for j in np.arange(num_radec_pts):

                (
                    sep_err_from_ra_data[j],
                    pa_err_from_dec_data[j],
                    _,
                ) = orbitize.system.transform_errors(
                    np.array(data["quant1"][radec_inds][j]),
                    np.array(data["quant2"][radec_inds][j]),
                    np.array(data["quant1_err"][radec_inds][j]),
                    np.array(data["quant2_err"][radec_inds][j]),
                    np.array(data["quant12_corr"][radec_inds][j]),
                    orbitize.system.radec2seppa,
                )

            sep_data[radec_inds] = sep_from_ra_data
            sep_err[radec_inds] = sep_err_from_ra_data

            pa_data[radec_inds] = pa_from_dec_data
            pa_err[radec_inds] = pa_err_from_dec_data

        # Transform Sep/PA points to RA/Dec
        ra_data = np.copy(data["quant1"])
        ra_err = np.copy(data["quant1_err"])
        dec_data = np.copy(data["quant2"])
        dec_err = np.copy(data["quant2_err"])

        if len(seppa_inds[0] > 0):

            ra_from_seppa_data, dec_from_seppa_data = orbitize.system.seppa2radec(
                data["quant1"][seppa_inds], data["quant2"][seppa_inds]
            )

            num_seppa_pts = len(seppa_inds[0])
            ra_err_from_seppa_data = np.empty(num_seppa_pts)
            dec_err_from_seppa_data = np.empty(num_seppa_pts)
            for j in np.arange(num_seppa_pts):

                (
                    ra_err_from_seppa_data[j],
                    dec_err_from_seppa_data[j],
                    _,
                ) = orbitize.system.transform_errors(
                    np.array(data["quant1"][seppa_inds][j]),
                    np.array(data["quant2"][seppa_inds][j]),
                    np.array(data["quant1_err"][seppa_inds][j]),
                    np.array(data["quant2_err"][seppa_inds][j]),
                    np.array(data["quant12_corr"][seppa_inds][j]),
                    orbitize.system.seppa2radec,
                )

            ra_data[seppa_inds] = ra_from_seppa_data
            ra_err[seppa_inds] = ra_err_from_seppa_data

            dec_data[seppa_inds] = dec_from_seppa_data
            dec_err[seppa_inds] = dec_err_from_seppa_data

        # For plotting different astrometry instruments
        if plot_astrometry_insts:
            astr_colors = (
                "purple",
                "#FF7F11",
                "#11FFE3",
                "#14FF11",
                "#7A11FF",
                "#FF1919",
            )
            astr_symbols = ("o", "*", "p", "s")

            ax_colors = itertools.cycle(astr_colors)
            ax_symbols = itertools.cycle(astr_symbols)

            astr_data = data[astr_inds]
            astr_insts = np.unique(data[astr_inds]["instrument"])

            # Indices corresponding to each instrument in datafile
            astr_inst_inds = {}
            for i in range(len(astr_insts)):
                astr_inst_inds[astr_insts[i]] = np.where(
                    astr_data["instrument"] == astr_insts[i].encode()
                )[0]

        # Plot each orbit (each segment between two points coloured using colormap)
        for i in np.arange(num_orbits_to_plot):
            points = np.array([raoff[i, :], deoff[i, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.0)
            if cbar_param != "Epoch [year]":
                lc.set_array(np.ones(len(epochs[0])) * cbar_param_arr[i])
            elif cbar_param == "Epoch [year]":
                lc.set_array(epochs[i, :])
            ax.add_collection(lc)

        if plot_astrometry:

            # Plot astrometry along with instruments
            if plot_astrometry_insts:
                for i in range(len(astr_insts)):
                    ra = ra_data[astr_inst_inds[astr_insts[i]]]
                    dec = dec_data[astr_inst_inds[astr_insts[i]]]
                    if plot_errorbars:
                        xerr = ra_err[astr_inst_inds[astr_insts[i]]]
                        yerr = dec_err[astr_inst_inds[astr_insts[i]]]
                    else:
                        xerr = None
                        yerr = None

                    ax.errorbar(
                        ra / 1000,
                        dec / 1000,
                        xerr=xerr / 1000,
                        yerr=yerr / 1000,
                        marker=next(ax_symbols),
                        c=next(ax_colors),
                        zorder=10,
                        label=astr_insts[i],
                        linestyle="",
                        ms=ms,
                        capsize=2,
                    )
            else:
                if plot_errorbars:
                    xerr = ra_err
                    yerr = dec_err
                else:
                    xerr = None
                    yerr = None

                ax.errorbar(
                    ra_data / 1000,
                    dec_data / 1000,
                    xerr=xerr / 1000,
                    yerr=yerr / 1000,
                    marker="o",
                    c="k",
                    zorder=10,
                    linestyle="",
                    capsize=2,
                    ms=ms,
                )

        # modify the axes
        if square_plot:
            adjustable_param = "datalim"
        else:
            adjustable_param = "box"
        ax.set_aspect("equal", adjustable=adjustable_param)
        ax.set_xlabel("$\\Delta$RA [arcsec]")
        ax.set_ylabel("$\\Delta$Dec [arcsec]")
        ax.locator_params(axis="x", nbins=6)
        ax.locator_params(axis="y", nbins=6)
        ax.invert_xaxis()  # To go to a left-handed coordinate system

        # plot sep/PA and/or rv zoom-in panels
        if rv_time_series:
            ax1 = plt.subplot2grid((3, 14), (0, 8), colspan=6)
            ax2 = plt.subplot2grid((3, 14), (1, 8), colspan=6)
            ax3 = plt.subplot2grid((3, 14), (2, 0), colspan=14, rowspan=1)
            ax2.set_ylabel("PA [$^{{\\circ}}$]")
            ax1.set_ylabel("$\\rho$ [mas]")
            ax3.set_ylabel("RV [km/s]")
            ax3.set_xlabel("Epoch")
            ax2.set_xlabel("Epoch")
            plt.subplots_adjust(hspace=0.3)
        else:
            ax1 = plt.subplot2grid((2, 14), (0, 9), colspan=6)
            ax2 = plt.subplot2grid((2, 14), (1, 9), colspan=6)
            ax2.set_ylabel("PA [$^{{\\circ}}$]")
            ax1.set_ylabel("$\\rho$ [mas]")
            ax2.set_xlabel("Epoch")

        if plot_astrometry_insts:
            ax1_colors = itertools.cycle(astr_colors)
            ax1_symbols = itertools.cycle(astr_symbols)

            ax2_colors = itertools.cycle(astr_colors)
            ax2_symbols = itertools.cycle(astr_symbols)

        epochs_seppa = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        for i in np.arange(num_orbits_to_plot):

            epochs_seppa[i, :] = np.linspace(
                start_mjd,
                Time(sep_pa_end_year, format="decimalyear").mjd,
                num_epochs_to_plot,
            )

            # Calculate ra/dec offsets for all epochs of this orbit
            if rv_time_series:
                raoff0, deoff0, vz_star0 = kepler.calc_orbit(
                    epochs_seppa[i, :],
                    sma[i],
                    ecc[i],
                    inc[i],
                    aop[i],
                    pan[i],
                    tau[i],
                    plx[i],
                    mtot[i],
                    tau_ref_epoch=tau_ref_epoch,
                    mass_for_Kamp=m1[i],  # why was this m0???? We measure stellar RV
                )

                raoff[i, :] = raoff0
                deoff[i, :] = deoff0
                vz_star[i, :] = vz_star0
            else:
                raoff0, deoff0, vz_star0 = kepler.calc_orbit(
                    epochs_seppa[i, :],
                    sma[i],
                    ecc[i],
                    inc[i],
                    aop[i],
                    pan[i],
                    tau[i],
                    plx[i],
                    mtot[i],
                    tau_ref_epoch=tau_ref_epoch,
                    mass_for_Kamp=m1[i],
                )

                raoff[i, :] = raoff0
                deoff[i, :] = deoff0
                vz_star[i, :] = vz_star0

            yr_epochs = Time(epochs_seppa[i, :], format="mjd").decimalyear

            seps, pas = orbitize.system.radec2seppa(
                raoff[i, :], deoff[i, :], mod180=mod180
            )

            plt.sca(ax1)

            seps /= 1000  # mas to arcsec
            plt.plot(
                yr_epochs, seps, color=sep_pa_color, zorder=1
            )  # gray seps v. time (top right)

            plt.sca(ax2)
            plt.plot(
                yr_epochs, pas, color=sep_pa_color, zorder=1
            )  # gray PA v. time (bottom right)

        # Plot sep/pa instruments
        if plot_astrometry_insts:
            for i in range(len(astr_insts)):
                sep = sep_data[astr_inst_inds[astr_insts[i]]]
                pa = pa_data[astr_inst_inds[astr_insts[i]]]
                epochs = astr_epochs[astr_inst_inds[astr_insts[i]]]
                if plot_errorbars:
                    serr = sep_err[astr_inst_inds[astr_insts[i]]]
                    perr = pa_err[astr_inst_inds[astr_insts[i]]]
                else:
                    yerr = None
                    perr = None

                plt.sca(ax1)
                plt.errorbar(
                    Time(epochs, format="mjd").decimalyear,
                    sep,
                    yerr=serr,
                    ms=ms,
                    linestyle="",
                    marker=next(ax1_symbols),
                    c=next(ax1_colors),
                    zorder=10,
                    label=astr_insts[i],
                    capsize=2,
                )
                plt.sca(ax2)
                plt.errorbar(
                    Time(epochs, format="mjd").decimalyear,
                    pa,
                    yerr=perr,
                    ms=ms,
                    linestyle="",
                    marker=next(ax2_symbols),
                    c=next(ax2_colors),
                    zorder=10,
                    capsize=2,
                )
            plt.sca(ax1)
            plt.legend(title="Instruments", bbox_to_anchor=(1.3, 1), loc="upper right")
        else:
            if plot_errorbars:
                serr = sep_err
                perr = pa_err
            else:
                yerr = None
                perr = None

            sep_data /= 1000  # mas to arcsec
            serr /= 1000  # mas to arcsec

            plt.sca(ax1)  # set current axis
            for j, epoch_time in enumerate(Time(astr_epochs, format="mjd").decimalyear):

                plt.errorbar(
                    epoch_time,
                    sep_data[j],
                    yerr=serr[j],
                    ms=ms,
                    linestyle="",
                    marker="o",
                    c=cmap(
                        (epoch_time - min_cbar_date_bepoch)
                        / (max_cbar_date_bepoch - min_cbar_date_bepoch)
                    ),
                    zorder=2,
                    capsize=2,
                )

            plt.sca(ax2)
            for k, epoch_time in enumerate(Time(astr_epochs, format="mjd").decimalyear):
                plt.errorbar(
                    epoch_time,
                    pa_data[k],
                    yerr=perr[k],
                    ms=ms,
                    linestyle="",
                    marker="o",
                    c=cmap(
                        (epoch_time - min_cbar_date_bepoch)
                        / (max_cbar_date_bepoch - min_cbar_date_bepoch)
                    ),
                    zorder=2,
                    capsize=2,
                )

        if rv_time_series:
            # print(f'data: {data}\n')
            rv_data = data[data["object"] == 0]  # stellar data
            # print(f'stellar data: {rv_data}\n')
            rv_data = rv_data[rv_data["quant_type"] == "rv"]  # stellar RV data
            # print(f'stellar rv data: {rv_data}')

            # switch current axis to rv panel
            plt.sca(ax3)

            # get list of rv instruments
            insts = np.unique(rv_data["instrument"])
            if len(insts) == 0:
                insts = ["defrv"]  # default instrument

            # get gamma/sigma labels and corresponding positions in the posterior
            gams = ["gamma_" + inst for inst in insts]  # gamma_defrv as default
            sigs = ["sigma_" + inst for inst in insts]  # gamma_defrv as default

            if isinstance(lab, list):
                lab = np.array(lab)
            else:
                lab = lab

            # get the indices corresponding to each gamma within results.labels
            gam_idx = [
                np.where(np.array([*lab]) == inst_gamma)[0] for inst_gamma in gams
            ]

            # indices corresponding to each instrument in the datafile
            inds = {}
            for i in range(len(insts)):
                inds[insts[i]] = np.where(rv_data["instrument"] == insts[i].encode())[0]

            # choose the orbit with the best log probability
            best_like = np.where(lnlike == np.amax(lnlike))[0][0]

            # Get the posteriors for this index and convert to standard basis
            best_post = post[best_like].copy()
            print(f"best_post: {best_post}\n")

            # Get the masses for the best posteriors:
            best_m0 = best_post[lab["m0"]]
            best_m1 = best_post[lab["m{}".format(object_to_plot)]]
            best_mtot = best_m0 + best_m1
            print(f"best_m0: {best_m0}, best_m1: {best_m1}, best_mtot: {best_mtot}\n")

            # colour/shape scheme scheme for rv data points
            clrs = ("purple", "#0496FF", "#372554", "#FF1053", "#3A7CA5", "#143109")
            symbols = ("o", "^", "v", "s")

            ax3_colors = itertools.cycle(clrs)
            ax3_symbols = itertools.cycle(symbols)

            # get rvs and plot them
            for i, name in enumerate(inds.keys()):
                inst_data = rv_data[inds[name]]
                # print(f'inst_data:{inst_data} from rv_data:{rv_data} with name:{name} and inds:{inds}')
                rvs = inst_data["quant1"]
                epochs = inst_data["epoch"]
                epochs = Time(epochs, format="mjd").decimalyear
                rvs -= best_post[lab[gams[i]]]
                if plot_errorbars:
                    yerr = inst_data["quant1_err"]
                    yerr = np.sqrt(yerr ** 2 + best_post[lab[sigs[i]]] ** 2)

                # print(f'Plotting epochs:{epochs} and rvs:{rvs} with yerr:{yerr}')
                plt.errorbar(
                    epochs,
                    rvs,
                    yerr=yerr,
                    ms=5,
                    linestyle="",
                    marker=next(ax3_symbols),
                    c=next(ax3_colors),
                    label=name,
                    zorder=5,
                    capsize=2,
                )
            if len(inds.keys()) == 1 and "defrv" in inds.keys():
                pass
            else:
                plt.legend()

            # calculate the predicted rv trend using the best orbit
            _, _, vz = kepler.calc_orbit(
                epochs_seppa[0, :],
                best_post[lab["sma{}".format(object_to_plot)]],
                best_post[lab["ecc{}".format(object_to_plot)]],
                best_post[lab["inc{}".format(object_to_plot)]],
                best_post[lab["aop{}".format(object_to_plot)]],
                best_post[lab["pan{}".format(object_to_plot)]],
                best_post[lab["tau{}".format(object_to_plot)]],
                best_post[lab["plx"]],
                best_mtot,
                tau_ref_epoch=tau_ref_epoch,
                mass_for_Kamp=best_m1,  # give me stellar RV!
            )
            # raoff0, deoff0, _ = kepler.calc_orbit(

            vz = vz * -(best_m1) / np.median(best_m0)

            # plot rv trend
            plt.plot(
                Time(epochs_seppa[0, :], format="mjd").decimalyear,
                vz,
                color=sep_pa_color,
                zorder=1,
                label="best_fit",
            )

        # add colorbar
        if show_colorbar:
            if rv_time_series:
                # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
                # You can change x1.0.05 to adjust the distance between the main image and the colorbar.
                # You can change 0.02 to adjust the width of the colorbar.
                cbar_ax = fig.add_axes(
                    [
                        ax.get_position().x1 + 0.005,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height,
                    ]
                )
                cbar = mpl.colorbar.ColorbarBase(
                    cbar_ax,
                    cmap=cmap,
                    norm=norm_yr,
                    orientation="vertical",
                    label=cbar_param,
                )
            else:
                # xpos, ypos, width, height, in fraction of figure size
                cbar_ax = fig.add_axes([0.47, 0.15, 0.015, 0.7])
                cbar = mpl.colorbar.ColorbarBase(
                    cbar_ax,
                    cmap=cmap,
                    norm=norm_yr,
                    orientation="vertical",
                    label=cbar_param,
                )

        ax1.locator_params(axis="x", nbins=6)
        ax1.locator_params(axis="y", nbins=6)
        ax2.locator_params(axis="x", nbins=6)
        ax2.locator_params(axis="y", nbins=6)

        return fig


def my_chop_chains(post, lnlike, burn, num_walkers, trim=0):
    """
        Permanently removes steps from beginning (and/or end) of chains from the
        Results object. Also updates `curr_pos` if steps are removed from the
        end of the chain.

        Args:
            burn (int): The number of steps to remove from the beginning of the chains
            trim (int): The number of steps to remove from the end of the chians (optional)

        .. Warning:: Does not update bookkeeping arrays within `MCMC` sampler object.

        (written): Henry Ngo, 2019
        """

    # Retrieve information from results object
    flatchain = np.copy(post)
    total_samples, n_params = flatchain.shape
    n_steps = int(total_samples / num_walkers)
    flatlnlikes = np.copy(lnlike)

    # Reshape chain to (nwalkers, nsteps, nparams)
    chn = flatchain.reshape((num_walkers, n_steps, n_params))
    # Reshape lnlike to (nwalkers, nsteps)
    lnlikes = flatlnlikes.reshape((num_walkers, n_steps))

    # Find beginning and end indices for steps to keep
    keep_start = int(burn)
    keep_end = int(n_steps - trim)
    n_chopped_steps = int(n_steps - trim - burn)

    # Update arrays
    chopped_chain = chn[:, keep_start:keep_end, :]
    chopped_lnlikes = lnlikes[:, keep_start:keep_end]

    # Flatten likelihoods and samples
    flat_chopped_chain = chopped_chain.reshape(num_walkers * n_chopped_steps, n_params)
    flat_chopped_lnlikes = chopped_lnlikes.reshape(num_walkers * n_chopped_steps)

    # Print a confirmation
    print("Chains successfully chopped. Returned: chopped_post, chopped_lnlike")

    return flat_chopped_chain, flat_chopped_lnlikes


def my_examine_chains(
    post,
    num_walkers,
    thin,
    outfile,
    lab,
    param_list=None,
    walker_list=None,
    n_walkers=None,
    step_range=None,
    transparency=1,
    color="k",
    lw=1,
    fs=12,
    logx=False,
    plot_med=False,
    set_ylim=False,
    plot_mad=False,
    xlim=None,
    fill_color="k",
    fill_edge_color="grey",
    highlight_edges=False,
    fill_alpha=0.3,
    med_color="k",
    joint=False,
    xlabel=r"\textbf{MCMC Step}",
):
    """
    Plots position of walkers at each step from Results object. Returns list of figures, one per parameter
    Args:
        param_list: List of strings of parameters to plot (e.g. "sma1")
            If None (default), all parameters are plotted
        walker_list: List or array of walker numbers to plot
            If None (default), all walkers are plotted
        n_walkers (int): Randomly select `n_walkers` to plot
            Overrides walker_list if this is set
            If None (default), walkers selected as per `walker_list`
        step_range (array or tuple): Start and end values of step numbers to plot
            If None (default), all the steps are plotted
        transparency (int or float): Determines visibility of the plotted function
            If 1 (default) results plot at 100% opacity

    Returns:
        List of ``matplotlib.pyplot.Figure`` objects:
            Walker position plot for each parameter selected

    (written): Henry Ngo, 2019
    """

    # Get the flattened chain from Results object (nwalkers*nsteps, nparams)
    flatchain = np.copy(post)
    total_samples, n_params = flatchain.shape
    n_steps = int(total_samples / num_walkers)
    print(f"n_steps after thinning: {n_steps}\n")
    # Reshape it to (nwalkers, nsteps, nparams)
    chn = flatchain.reshape((num_walkers, n_steps, n_params))

    # Get list of walkers to use
    if n_walkers is not None:  # If n_walkers defined, randomly choose that many walkers
        walkers_to_plot = np.random.choice(num_walkers, size=n_walkers, replace=False)
    elif walker_list is not None:  # if walker_list is given, use that list
        walkers_to_plot = np.array(walker_list)
    else:  # both n_walkers and walker_list are none, so use all walkers
        walkers_to_plot = np.arange(num_walkers)

    # Get list of parameters to use
    if param_list is None:
        params_to_plot = np.arange(n_params)
    else:  # build list from user input strings
        params_plot_list = []
        for i in param_list:
            if i in lab:
                params_plot_list.append(lab[i])
            else:
                raise Exception(
                    "Invalid param name: {}. See system.basis.param_idx.".format(i)
                )
        params_to_plot = np.array(params_plot_list)

    # labels for parameter y-axes
    if joint:
        param_ylabels = {
            "sma1": r"\textbf{$\bm{a}$ (au)}",
            "ecc1": r"$\bm{e}$",
            "inc1": r"\textbf{$\bm{i}$ (\textdegree)}",
            "aop1": r"\textbf{$\boldsymbol{\omega}$ (\textdegree)}",
            "pan1": r"\textbf{$\boldsymbol{\Omega}$ (\textdegree)}",
            "tau1": r"$\boldsymbol{\tau}$",
            "plx": r"\textbf{$\mathbf{\boldsymbol{\pi}}$ (mas)}",
            "gamma_defrv": r"$\gamma \ \mathbf{(\text{\textbf{km}}/\text{\textbf{s}})}$",
            "gamma_Irwin": r"$\gamma_\text{\textbf{Irwin}} \ \mathbf{(\text{\textbf{km}}/\text{\textbf{s}})}$",
            "gamma_Modern": r"$\gamma_\text{\textbf{Modern}} \ \mathbf{(\text{\textbf{km}}/\text{\textbf{s}})}$",
            "sigma_defrv": r"$\sigma \ \mathbf{(\text{\textbf{km}}/\text{\textbf{s}})}$",
            "sigma_Irwin": r"$\sigma_\text{\textbf{Irwin}} \ \mathbf{(\text{\textbf{km}}/\text{\textbf{s}})}$",
            "sigma_Modern": r"$\sigma_\text{\textbf{Modern}} \ \mathbf{(\text{\textbf{km}}/\text{\textbf{s}})}$",
            "m1": r"$\mathbf{M_1} \ (\mathrm{M_\odot})$",
            "m0": r"$\mathbf{M_0} \ (\mathrm{M_\odot})$",
        }

        w, h = figaspect(1 / 2)
        fig, axs = plt.subplots(nrows=4, ncols=3, sharex=True, figsize=(w, h))
        axs[0][2].set_visible(False)  # 12 spots but only 11 params.

    else:
        param_ylabels = {
            "sma1": r"\textbf{$\bm{a}$ (au)}",
            "ecc1": r"$\bm{e}$",
            "inc1": r"\textbf{$\bm{i}$ (\textdegree)}",
            "aop1": r"\textbf{$\boldsymbol{\omega}$ (\textdegree)}",
            "pan1": r"\textbf{$\boldsymbol{\Omega}$ (\textdegree)}",
            "tau1": r"$\boldsymbol{\tau}$",
            "mtot": r"$\mathbf{M_\text{\textbf{tot}}} \ (\mathrm{M_\odot})$",
            "plx": r"\textbf{$\mathbf{\boldsymbol{\pi}}$ (mas)}",
        }

        fig, axs = plt.subplots(nrows=4, ncols=2, sharex=True)

    # loop through each of the parameters
    for i, pp in enumerate(params_to_plot):

        # my new stuff
        angle_list = ["inc1", "aop1", "pan1"]

        # convert angles to degrees from radians
        if param_list[i] in angle_list:
            fac = 180 / np.pi
        else:
            fac = 1

        # get the label for the param being plotted
        param_ylabel = param_ylabels[param_list[i]]

        # fig, ax = plt.subplots()
        # loop through walkers being plotted
        for ww in walkers_to_plot:

            # plot that param for that walker, across all steps being shown
            actual_steps = np.arange(1, len(chn[ww, :, pp]) + 1) * thin

            if logx:
                axs[i % 4 + int(i > 7)][int(i > 3) + int(i > 7)].semilogx(
                    actual_steps,
                    chn[ww, :, pp] * fac,
                    color,
                    alpha=transparency,
                    lw=lw,
                    zorder=0,
                )
            else:
                axs[i % 4 + int(i > 7)][int(i > 3) + int(i > 7)].plot(
                    actual_steps,
                    chn[ww, :, pp] * fac,
                    color,
                    alpha=transparency,
                    lw=lw,
                    zorder=0,
                )
        # y-axis labels
        axs[i % 4 + int(i > 7)][int(i > 3) + int(i > 7)].set_ylabel(
            param_ylabel, fontsize=fs
        )

        # Limit range shown if step_range is set (xlims)
        #########################################
        if step_range is None:
            step_range = (0, len(actual_steps))

        # override for xlim and step_range set
        if xlim is not None:

            axs[i % 4 + int(i > 7)][int(i > 3) + int(i > 7)].set_xlim(xlim)
        ##########################################

        # median, median absolute deviation
        #################################

        # plot the median of the walkers' locations at each step
        if plot_med:

            # calcluate median of parameter across walkers at each step for parameter pp
            # limit to only the step range plotted
            med = np.median(chn[:, :, pp] * fac, axis=0)[step_range[0] : step_range[1]]

            # plot the median
            axs[i % 4 + int(i > 7)][int(i > 3) + int(i > 7)].plot(
                actual_steps[step_range[0] : step_range[1]],
                med,
                color=med_color,
                alpha=1,
                lw=lw * 3.5,
                zorder=2,
            )

        # plot the median absolute deviation of walker positions from the median
        if plot_mad:

            mad = stats.median_abs_deviation(chn[:, :, pp] * fac, axis=0, scale=1)[
                step_range[0] : step_range[1]
            ]

            # fill-in +- 1 MAD
            axs[i % 4 + int(i > 7)][int(i > 3) + int(i > 7)].fill_between(
                actual_steps[step_range[0] : step_range[1]],
                (med - mad),
                (med + mad),  # color=colors[4],
                facecolor=fill_color,
                alpha=fill_alpha,
                zorder=1,
                lw=0,
                label=r"$\bm{\pm} \, 1 \ \text{\textbf{MAD}}$",
            )

            if highlight_edges:
                # highlight the edges
                axs[i % 4 + int(i > 7)][int(i > 3) + int(i > 7)].plot(
                    actual_steps[step_range[0] : step_range[1]],
                    (med - mad),
                    color=fill_edge_color,
                    alpha=np.min([3 * fill_alpha, 1]),
                    zorder=1,
                    lw=2 * lw,
                )

                axs[i % 4 + int(i > 7)][int(i > 3) + int(i > 7)].plot(
                    actual_steps[step_range[0] : step_range[1]],
                    (med + mad),
                    color=fill_edge_color,
                    alpha=np.min([3 * fill_alpha, 1]),
                    zorder=1,
                    lw=2 * lw,
                )
        ##########################################

        # y-lims
        #########################
        if set_ylim and plot_med and not plot_mad:

            # range of median values
            med_range = np.max(med) - np.min(med)

            # 4% of the range above and below the range
            set_ylim_value = (
                np.min(med) - 0.15 * med_range,
                np.max(med) + 0.15 * med_range,
            )

            # set ylim for med
            axs[i % 4 + int(i > 7)][int(i > 3) + int(i > 7)].set_ylim(set_ylim_value)

        # need wider y-lims if the standard deviations are also being plotted
        elif set_ylim and plot_med and plot_mad:

            # range of med +- mad values
            mad_range = np.max(med + mad) - np.min(med - mad)

            # 5% of the range above and below the range
            set_ylim_value = (
                np.min(med - mad) - 0.15 * mad_range,
                np.max(med + mad) + 0.15 * mad_range,
            )

            # set y-lim for med +- mad
            axs[i % 4 + int(i > 7)][int(i > 3) + int(i > 7)].set_ylim(set_ylim_value)
        ############################

        # my bold plots
        ################################################
        # set the axis line width in pixels
        for axis in ["left", "bottom", "top", "right"]:
            if joint:
                for col in range(3):
                    for ax in axs:
                        ax[col].spines[axis].set_linewidth(2 * 0.5)
                        ax[col].tick_params(
                            axis="both",
                            which="major",
                            width=2 * 0.5,
                            length=8 * 0.5,
                            labelsize=fs,
                        )
            else:
                for col in range(2):
                    for ax in axs:
                        ax[col].spines[axis].set_linewidth(2 * 0.5)
                        ax[col].tick_params(
                            axis="both",
                            which="major",
                            width=2 * 0.5,
                            length=8 * 0.5,
                            labelsize=fs,
                        )

        ###############################################

    # overall plot labels
    axs[-1][0].set_xlabel(xlabel, fontsize=1.2 * fs)
    axs[-1][1].set_xlabel(xlabel, fontsize=1.2 * fs)
    if joint:
        axs[-1][2].set_xlabel(r"\textbf{MCMC Step}", fontsize=1.2 * fs)
    fig.suptitle(
        r"\textbf{MCMC Walker Parameter Median $\boldsymbol{\pm}$ MAD}",
        fontsize=1.5 * fs,
    )
    plt.tight_layout()  # cleanup
    # save figure
    plt.savefig(outfile, dpi=600, facecolor="w")

    return fig

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
    
    fig, ax = plt.subplots(dpi=dpi, figsize=(10,10))

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
    
    print(f'element: {element}')
    print(f'bins: {bins}')
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
