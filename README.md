###library description
This is an implementation of the geostatistical Monte Carlo Markvo Chain method, which is also called SGR in papers (cite) and xxx in (cite). The python script aims to provide necessary functionarity and ensure sufficient flexibility to be adjusted to different purposes or different regions. The package also include specific utility code to use the geostatistical MCMC approach to generate subglacial topographies, and the details of this method can be seen in the publication xxx. 

###recipe for generating topography ensemble
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


###library structure
- Topography: Glaciology Example
	- get data
		- load_smb
		- load_heightChange
		- load_velocity
		- load_bedMachine
		- load_bedmap
		- load_radar
	- obtain glacier's boundary
		- get_inversion_domain
	- mass conservation
		- get_mass_conservation_residual
	- preprocessing
		- filter_data_by_std
		- filter_data_neariceshelf (may not need this)
- MCMC: Core geostatistics and Monte Carlo Markov Chain utilities
	- Sequential Gaussian Simulation
		- fit_variogram
	- Random Fields
		- get_random_field
		- get_crf_weight
		- get_blocks
	- MCMC
		- loss_mcres_function
		- loss_data_function
		- mcmc_WRF
		- mcmc_SGS
- Utilities
	- min_dist
	- rescale
	- logistic
	- figure plotting
		- plot_loss
		- plot_diff
		- plot_mcres
		- plot_video
		- plot_autocorr
		
		
### Different styles of coding

endResults = crf_chain(argument A, B, C,...... F)

or 

chain_crf.init(.....)

chain_crf.set_update_type()
chain_crf.set_high_vel_region()
chain_crf.set_update_in_region()

chain_crf.set_loss_type()

chain_crf.run(iter=xx,savefile=True)


crf_chain.A = inputA
crf_chain.B = inputB
crf_chain.C = inputC
crf_chain.run()
crf_chain.plotresult()
