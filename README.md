# ObsTurbMeasurement
ObsTurbMeasurement (OTM) takes standard astronomy observations (CO(2-1) for now) and converts it into measurements of the molecular gas turbulence

If you use LMTurbCode in your research, please cite the original paper:

Miller, L. J. et al. (2026), The turbulence driving mode in NGC7793 and NGC1313, Monthly Notices of the Royal Astronomical Society, Volume 547, Issue 2 April 2026, stag359, https://doi.org/10.1093/mnras/stag359 

Please also cite the specific software release used in your analysis.

The roving kernel implementation builds on the method described by Gerrard et al. (2023). Please also cite the original methodological work where appropriate.

## Installation

Clone the repository and install the package and its dependencies:

```bash
git clone https://github.com/Lewis-J-Miller/ObsTurbMeasurement.git
cd ObsTurbMeasurement
python -m pip install .
```

To install the package in editable mode for development, use:

```bash
python -m pip install -e .
```
Package dependencies include; `numpy`, `matplotlib`, `astropy`, `cfpack`, `scipy`, `math`, `multiprocessing` and `os`. More details in `pyproject.toml`. 

# Usage Information
The main function in this package is the `RovingKernel()` function (or it's multi-process equivalent, `RovingKernel_MP()`), which measures the turbulence driving parameter, $b$, alongside other turbulent parameters useful in understanding the turbulence in the ISM. More details of the underlying physics are provided in Miller et al. (2026) and Gerrard et al. (2023). 

`RovingKernel()` has slightly different returns depending on the exact process selected. The default method is for when `mommapmethod="Linewidth"` or `mommapmethod="Mom2"`, and uses the linewidth measure to derive the velocity dispersions. When this is run, the function returns the following (and in this order); 
1. A map of $b$, the turbulence driving parameter, measured as $b=\frac{\sigma_{\rho/\rho_0}}{\mathcal{M}}$ (in the magnetic-field-free assumption). 
2. A map of $\mathcal{M}$, the sonic Mach number (assuming a constant soundspeed, default of 0.6km/s)
3. A map of $\sigma_{\rho/\rho_0}$, the volumetric, turbulence-isolated density dispersion measured for each kernel
4. A map of $\sigma_{\Sigma/\Sigma_0}$, the 2-dimensional (column density), turbulence-isolated density dispersion measured for each kernel
5. A map of $\sqrt{\mathcal{R}}$, the (squareroot of the) Brunt R factor from Brunt et al. (2010) to convert the column density dispersion to a volumetric density dispersion
6. A map of $\sigma_{v,LW}$, the linewidth of each spectra. i.e., the square-root of the moment 2. Note that this is the kernel-averaged $\sigma_{v,LW}$, so this is no longer a measure for each spaxel, but rather of the kernel. 
7. A map of $\sigma_{v,3D}$, the same as the 1D velocity dispersion, but converted to a volumetric measure by the process introduced by Stewart and Federrath (2022)

These results are different for `mommapmethod="Mom1"` - i.e., using the moment-1 smoothing and measurement to determine the velocity dispersion. The only difference in output is output number 6, the $\sigma_{v,LW}$, which becomes $\sigma_{v,1D}$, derived from kernel-smoothing and subtraction of the moment 1 map, which is a different measure than the linewidth measure. c.f., Stewart and Federrath (2022). 


