# weible_astr513_501_final_project

This code includes a small-sample version of the results I presented in class on 2-D kernel density estimation for corner plot generation.

```orbit_2d_kde_example_notebook_weible.ipynb``` is a complete, working example of processing 1000 8-dimensional samples of orbital and physical parameters into a corner plot with KDEs. This is a smaller sample than I showed in class so that everything can comply with GitHub's file size restrictions, and decrease computation time. The code should be extensible to at least ~10^7 8-dimensional samples (the most I have tested).

There are some dependencies (imported packages) here that you would likely need to install with, e.g., conda or pip. ```package-list.txt``` contains an environment that works for me to run all of this code, though there are likely some unnecessary packages in my environment that you may not require.

Sample images of the Jupyter Notebook output, as it is currently configured, are included for reference. 2-D KDEs are named as "y-axis parameter"\_v\_"x-axis parameter" ... .jpg

```fftkde_cred_int.py``` contains functions to compute credible intervals, 1-D KDE plots, and 2-D KDE plots. Helper functions for 2-D KDEs establish which points to scatter around the outermost contour of the KDE shown.

```modified_plot.py``` contains functions to plot a standard histogram, configure matplotlib parameters, and determine credible intervals and maximum a posterior estimates from histograms.

```parallel_plots.py``` contains a function, ```generate_corner_plots()```, which calls ```fftkde_plot()``` and ```off_diagonal_fftkde_plot()``` from ```fftkde_cred_int.py``` in a loop to generate the diagonal and off-diagonal subplots of a larger corner plot, which can be manually stacked together in, e.g., Photoshop or GIMP.

```hii1348b_orbitize_data.csv``` contains the relative astrometric measurements to which Keplerian orbits were fit to generate the samples stored in ```orbits_OFTI_1000_walkers0.hdf5```.
