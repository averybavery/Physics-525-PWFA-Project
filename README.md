# Physics-525-PWFA-Exploration
This repository contains all work related to the computational project of Physics 525 Intro to Plasmas in Spring 2026 taught by Rogerio Jorge. We (Jason and Avery) chose to study plasma wakefield acceleration. Namely, the longitudinal electric field created from the drive beam. We explain the theory and results in the LaTeX report entitled "PWFA Report." More graphs can be found in the presentation.

To run the 1D simulations, you just need to clone the repository and run the individual python files to create the matplotlib graphs. 

For 2D simulations, first install WarpX. We used the example: Examples/Physics_applications/plasma_acceleration/inputs_test_2d_plasma_acceleration_boosted, in the WarpX Repository: https://github.com/BLAST-WarpX/warpx. After running that example, use the plot_pwfa.py file to plot the output, and the scan_sigma_kp.py file to replicate our sigma_z vs E_max plot.
