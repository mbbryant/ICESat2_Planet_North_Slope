# Overview
This repository provides the code used to generate the results and figure for the following publication: https://doi.org/10.5194/egusphere-2024-1656. All required data with the
exception of Planet imagery is available at: https://doi.org/10.5281/zenodo.11095271. Users will have to provide their own imagery to run planet_drew_point_bulk_processing.py, 
tc_fig_a1_hitsograms.py, and get_erosion_rates.py.

# Included Files:
- *tc_planet_processing.ipynb* contains the workflow for derving shorelines from Planet imagery
- *tc_fig_a1_histograms.py* generates NDWI the histograms for Figure A1
- *tc_planet_bulk_analyses.py* plots all coastlines derived from Planet for this study, calculates the standard deivations reported in Table A1, and estimates the final Planet
  shoreline position uncertainty
- *tc_ERA5_analyses.py* calculates all environmental metrics reported in Table 2, as well as counts the number of open water days per year based on ERA5 hourly reanalysis data
- *tc_sliderule_drew_point.py* downloads ICESat-2 ATl03 photon data and generated elevation profiles using SlideRule (https://doi.org/10.21105/joss.04982)
- *tc_get_erosion_rates.py* estimates the shoreline change estimates reported in Table 3 and throughout the manuscript. It also generates Figure 2 and the base of Figure 3
- *SDS_centered_transects.py* is modified from the SDS_transects module in CoastSat (https://github.com/kvos/CoastSat) to generate transects that are centered on provided points
- *tc_IS2_paper_plots.py* generates Figures A3, A4, A5, the base for Figure 6 and Figure 7, the shoreline boundaries used for comparison between ICESat-2 and Planet, and the backshore height
  and slope estimates reported in Table 4.
- *tc_is2_correlations.py* generates Figure 4 and Figure 5, calculates the shoreline change estimates reported in Table A2, and estimates the correlation coefficients between planet and
  ICESat-2-derived shoreline change
- *tc_fig_compare_rates.py* Generates Figure A7
- *figures* contains the map overlays used in Figure 6

  # Citation
  This code is assocaited with the following publication:
  Bryant, M. B., Borsa, A. A., Masteller, C. C., Michaelides, R. J., Siegfried, M. R., Young, A. P., and Anderson, E. J.: Multiple modes of shoreline change along the
  Alaskan Beaufort Sea observed using ICESat-2 altimetry and satellite imagery, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-1656, 2024.

  # References
  Shean, D., Swinski, J. P., Smith, B., Sutterley, T., Henderson, S., Ugarte, C., Lidwa, E., and Neumann, T.: SlideRule: Enabling rapid, scalable,680
  open science forthe NASA ICESat-2 mission and beyond, Journal of Open Source Software, 8, 4982, https://doi.org/10.21105/joss.04982,
  2023.

  Vos, K.: SDS_transects.py, https://github.com/kvos/CoastSat/blob/master/coastsat/SDS_transects.py, 2024

  Vos, K., Splinter, K. D., Harley, M. D., Simmons, J. A., and Turner, I. L.: CoastSat: A Google Earth Engine-enabled Python695
  toolkit to extract shorelines from publicly available satellite imagery, Environmental Modelling & Software, 122, 104 528,
  https://doi.org/10.1016/j.envsoft.2019.104528, 2019

  # Contact
  Marnie Bryant (m1bryant@ucsd.edu)
