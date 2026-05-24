===============================================================================
Structure of Memory in Time Series: A One-dimensional Atomic Chain Model
Xiujiang Li
lee_xiujiang@163.com
===============================================================================
1. data files
-----------
/data
spectral data - Spectral data of fractional Brownian motion (fBm) series with different Hurst exponents, analyzed based on ARCH models, varying alpha1 parameters, and Shanghai Stock Exchange Composite Index (SSEC) data from different years, including full spectrum data, symmetric sub-spectrum data, and anti-symmetric sub-spectrum data.
fBm_series.csv - fbm series the first 1000 pieces of data）
CA_200 - fBmACF（hurst exponent from 0.60 to 0.90）
SSEC_1.xlsx  - SSEC 1 mins close price
2. code files
-----------
/code
fBm - results of fBm (apart from spectral rigidity)
SSEC - results of SSEC (apart from spectral rigidity)
ARCH - results of ARCH (apart from spectral rigidity)
spectral rigidity  -  results of spectral rigidity
===============================================================================
Instruction :
The spectral rigidity was calculated by means of the program accessible via the link in https://github.com/stfxecutables/empyricalRMT.
===============================================================================
