# Simulation and Real Data Analysis Scripts

Simulation and real data analysis scripts. The code was original written to run on a HPC cluster. 

## How to Run

The following commands run the simulations or real data analysis and produce the figures in the corresponding section.

### Section 6.2 and Section 6.3 (Parameter Recovery and Dimension Selection)

These commands run the simulations and produce Table 1, Figure 1, and Figure 2.

```bash
>>> python simulation_recovery_dim_select.py
>>> cd output_recovery_dim_select/
>>> python parameter_recovery_table.py   
>>> python plot_dim_heatmaps.py        
```

### Section 5.6 of the supplement (Sensitivity to d)

These commands run the simulations and produce Figure S2, Table S7, and Table S8

```bash
>>> python simulation_recovery_dim_select_sensitivity.py
>>> cd output_recovery_dim_select_sensitivity/
>>> python plot_dim_heatmaps.py
>>> python parameter_recovery_table.py 6
>>> python parameter_recovery_table.py 12
```

### Section 7 and BLANK (Banana Trade Network Analysis)

To produce the figures, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook application_POLECAT.ipynb
```

### Section 5.3 of the supplement (Sensitivity to zero-inflation)

The following commands runs the simulations and produce Figure 7(a).

```bash
>>> python simulation_nonedge_sensitivity.py
>>> cd output_nonedge_sensitivty/
>>> python process.py
>>> python plot.py
```

### Section 5.4 of the supplement (Sensitivity to heavy-tailed distributions)

The following commands runs the simulations and produce Figure 7(b).

```bash
>>> python simulation_time_fraction_sensitivity.py
>>> cd output_time_fraction_sensitivty/
>>> python process.py
>>> python plot.py
```


### Section 5.5 of the supplement (Sensitivity to latent space misspecification)


### Section 7.1 of the supplement (Protein-Protein interaction network)


### Section 7.2 of the supplement (Host-Parasite interaction network)
