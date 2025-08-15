### Library description
This is an implementation of the geostatistical Monte Carlo Markvo Chain method, which is also called SGR in papers (cite) and xxx in (cite). The python script aims to provide necessary functionarity and ensure sufficient flexibility to be adjusted to different purposes or different regions. The package also include specific utility code to use the geostatistical MCMC approach to generate subglacial topographies, and the details of this method can be seen in the publication xxx. 

### Installation
TODO

### Recipe for generating topography ensemble
Define location and grids
Load all data needed for the location
- radar (BedMap)
- velocity (InSAR MEaSUREs)
- surface elevation (BedMachine or REMA?)
- surface elevation change (ITS_LIVE)
- surface mass balance (RACMO)
- bedmachine
Obtain inversion domain boundary
Define residual distribution
meso-scale chain
fine-scale chain
plot results

### Library structure
- Topography: Functionalities for subglacial topography application, including retrieving data, define high-velocity region, and calculating mass conservation
- MCMC: Core geostatistics and Monte Carlo Markov Chain utilities
		
### Future development plan
- use LU decomposition to generate random fields for faster speed
- aggregate generation of random fields in the beginning of each chain segment, or store LU decomposition in matrices
- adopt faster geostatistical simulation method
- update tutorials
