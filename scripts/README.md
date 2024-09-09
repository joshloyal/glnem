# Simulation and Real Data Analysis Scripts

Simulation and real data analysis scripts. The code was original written to run on a HPC cluster. 

## How to Run

The following commands run the simulations or real data analysis and produce the figures in the corresponding section.

### Section 6.2 and Section 6.3 (Parameter Recovery and Dimension Selection)

These commands run the simulations and produces Table 1, Figure 1, and Figure 2.

```bash
>>> python simulation_recovery_dim_select.py
>>> cd output_recovery_dim_select/
>>> python parameter_recovery_table.py   
>>> python plot_dim_heatmaps.py        
```

### Section 5.6 of the supplement (Sensitivity to d)

These commands run the simulations and produces Figure S2, Table S7, and Table S8.

```bash
>>> python simulation_recovery_dim_select_sensitivity.py
>>> cd output_recovery_dim_select_sensitivity/
>>> python plot_dim_heatmaps.py
>>> python parameter_recovery_table.py 6
>>> python parameter_recovery_table.py 12
```

### Section 7 and Sections 7.3 and 8 of the supplement (Banana Trade Network Analysis)

To produce the results and figures, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook Banana\ Trade.ipynb
```

This notebook produces Figure 3, Figure S7, Figure S8, and Figure S15.

### Section 5.3 of the supplement (Sensitivity to zero-inflation)

The following commands runs the simulations and produces Figure S1, Table S1, and Table S2.

```bash
>>> python simulation_zero_inflation.py
>>> cd output_zero_inflation/
>>> python make_dim_heatmaps.py
>>> python parameter_recovery_table.py
>>> python coef_recovery_table.py
```

### Section 5.4 of the supplement (Sensitivity to heavy-tailed distributions)

The following commands runs the simulations and produces Table S3.

```bash
>>> python simulation_heavy_tails.py
>>> cd output_heavy_tails/
>>> python make_table.py
```

### Section 5.5 of the supplement (Sensitivity to latent space misspecification)

The following commands runs the simulations and produces Table S4, Table S5, and Table S6.

```bash
>>> python simulation_latent_space.py
>>> cd output_latent_space/
>>> python make_table.py distance
>>> python make_table.py distance_sq
>>> python make_table.py bilinear
>>> python coef_recovery_table.py distance
>>> python coef_recovery_table.py distance_sq
>>> python coef_recovery_table.py bilinear
```

### Sections 7.1, 7.3, and 8 of the supplement (Protein-Protein interaction network)

To produce the results and figures, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook Protein\ Interaction\ Network.ipynb
```

This notebook produces Figure S3, Figure S4, Figure S9, and Figure S16.

### Section 7.2, 7.3, and 8 of the supplement (Host-Parasite interaction network)

To produce the results and figures, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook Host\ Parasite.ipynb
```

This notebook produces Figure S5, Figure S6, Figure S10, Figure S11, Figure S12, Figure S13, Figure S17, and Figure S18.
