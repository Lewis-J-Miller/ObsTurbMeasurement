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


Inline test: $`b`$

Display test:

```math
b = \frac{1}{2}
```


