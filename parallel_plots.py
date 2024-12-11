import numpy as np
import matplotlib.pyplot as plt
import gc
import warnings
from fftkde_cred_int import *
import gc


def generate_corner_plots(
    param_list,
    adjusted_units_orbits,
    big_df,
    grid_points,
    N,
    std_norm_levels,
    corner_dpi,
    size_fac,
    sampler,
    bw_rule="silvermans",
    plot_raw_kde=False,
    min_contour_lw=0.75,
    fftkde_res=2**25,  # default
    scatter_alpha=10 / 510,
    lw=2,
    scatter_cmp_val=0.55,
    fill_alpha=0.2,
    base_margin=0.4,
    label_margin=1.4495,
    pad_inches=0.0,
    base_size=10,
    ylab_offset=-0.13,
    annotate=False,
    ann_title=True,
    cmap_min=0.1,
    show_peak_x_mid=True,
    show_peak_y_mid=True,
    scatter_size_fac=4,
    hist_bins=400,
    jpeg=False
):
    """Generate corner plots sequentially"""

    for i in range(len(param_list)):  # np.arange(3,len(param_list)):##
        for j in range(
            len(param_list)
        ):  # np.arange(6, len(param_list)):#range(len(param_list)):
            # Set axis label visibility
            show_ylabel = i == 0
            show_xlabel = j == len(param_list) - 1
            if j > i:  # Below diagonal
                try:
                    param_x = param_list[i]
                    param_y = param_list[j]

                    # Get data for current position
                    element_x = adjusted_units_orbits[
                        np.where(param_list == param_x)
                    ].flatten()
                    element_y = adjusted_units_orbits[
                        np.where(param_list == param_y)
                    ].flatten()

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        print(f"Making off-diagonal plot for {param_y} v. {param_x}")
                        off_diagonal_fftkde_plot(
                            param_x,
                            param_y,
                            element_x,
                            element_y,
                            big_df,
                            kernel="gaussian",
                            bw_rule=bw_rule,
                            norm=2,
                            grid_points=grid_points,
                            N=N,
                            spine_color="k",
                            ticklab_color="k",
                            tick_color="k",
                            min_contour_lw=min_contour_lw,
                            cmap="plasma_r",
                            cmap_max=1.0,
                            cmap_min=cmap_min,
                            std_norm_levels=std_norm_levels,
                            dpi=corner_dpi,
                            scatter_cmp_val=scatter_cmp_val,
                            scatter_alpha=scatter_alpha,
                            show_xlabel=show_xlabel,
                            show_ylabel=show_ylabel,
                            jpeg=jpeg,
                            plot_raw_kde=plot_raw_kde,
                            size_fac=size_fac,
                            base_margin=base_margin,
                            label_margin=label_margin,
                            pad_inches=pad_inches,
                            base_size=base_size,
                            ylab_offset=ylab_offset,
                            show_peak_x_mid=show_peak_x_mid,
                            show_peak_y_mid=show_peak_y_mid,
                            scatter_size_fac=scatter_size_fac,
                        )

                    del element_y
                    plt.close("all")
                    gc.collect()

                except Exception as e:
                    print(f"Error in off-diagonal plot ({i}, {j}): {str(e)}")

            elif j == i:  # Diagonal
                try:
                    param_y = param_list[j]
                    element_y = adjusted_units_orbits[
                        np.where(param_list == param_y)
                    ].flatten()
                    element_params = big_df.loc[param_y]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        print(f"Making a diagonal plot for {param_y}")
                        peak, errp, errm = fftkde_plot(
                            element_y,
                            element_params,
                            obj="hii1348b",
                            sampler=sampler,
                            color="k",
                            dpi=corner_dpi,
                            bw_rule=bw_rule,
                            save=True,
                            size_fac=size_fac,
                            annotate=annotate,
                            show_xlabel=show_xlabel,
                            show_ylabel=False,
                            jpeg=jpeg,
                            fftkde_res=fftkde_res,
                            lw=lw,
                            fill_alpha=fill_alpha,
                            base_margin=base_margin,
                            label_margin=label_margin,
                            pad_inches=pad_inches,
                            base_size=base_size,
                            ylab_offset=ylab_offset,
                            ann_title=ann_title,
                            no_ticks=True,
                            no_tick_labels=True,
                            plot_hist=True,
                            hist_bins=hist_bins,
                            hist_color="lightgray",
                        )

                    plt.close("all")
                    del element_y, element_params, peak, errp, errm
                    gc.collect()

                except Exception as e:
                    print(f"Error in diagonal plot ({i}, {j}): {str(e)}")
            plt.close("all")
            gc.collect()
        plt.close("all")
        gc.collect()
