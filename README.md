### Library description
This is an implementation of the geostatistical Monte Carlo Markvo Chain method for inversion. The python script aims to provide necessary functionarity and ensure sufficient flexibility to be adjusted to different purposes or different regions. The package is currently implemented for using the geostatistical MCMC approach to generate subglacial topographies. Please refers to the publication https://doi.org/10.31223/X5SB2R for details. 

### Installation
Fork or download the github repository

import gstatsMCMC.Topography as Topography

import gstatsMCMC.MCMC as MCMC

### Tutorials
Please see the jupyter notebook tutorials (T1_LoadData.ipynb, T2_StatisticalAnalysis.ipynb, T3_LargeScaleChain.ipynb, T4_SmallScaleChain.ipynb)

### Documentation
Please see gstatsMCMC.MCMC.html for documentation for MCMC.py

### Library structure
- Topography: Functionalities for subglacial topography application, including retrieving data, define high-velocity region, and calculating mass conservation
- MCMC: Core geostatistics and Monte Carlo Markov Chain utilities
		
### Future development plan
- use LU decomposition to generate random fields for faster speed
- aggregate generation of random fields in the beginning of each chain segment, or store LU decomposition in matrices

### Contributor
- Niya Shao
- Michael Field
- Emma MacKie
