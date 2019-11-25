## Physics-Informed Resolution-Enhancing GANs (PhIRE GANs)
___
### Requirements
- Python v3.7
- TensorFlow v1.13.1
- Recommended:
    - numpy v1.16.3
    - matplotlib v3.0.3
- for data processing:
    - netCDF4
    - pvlib (for GHI decomposition)

### Data

##### WIND Toolkit & NSRDB
LR, MR, and HR wind example data (from WIND Toolkit) can be found in `example_data/`. These datasets are from NREL's WIND Toolkit. The LR and MR data are to be used with the MR and HR models respectively. If you would like to use your own data for the super-resolution it must have the shape: (None, None, None, [ua, va]). Example solar data (from NSRDB) can also be found in `example_data/` and can be treated in the same manner as the WIND Toolkit is treated. If you choose to use your own solar data it should have the shape: (None, None, None, [DNI, DHI]).
The scripts are designed to take in TFRecords. An example of how to convert numpy arrays to compatible TFRecords can be found in `data_processing/TFRecords_gen.py`.

##### CCSM
If you would like to run the CCSM wind data through the pretrained PhIREGANs models, you can download the data from [here](https://esgf-node.llnl.gov/projects/esgf-llnl/) with the following:
- project : CMIP5
- model : CCSM4
- experiment : 1% per year CO<sub>2</sub>
- time_frequency : day
- realm : atmos
- ensemble : r2i1p1
- version : 20130218
- variables: ua, va (for wind), and rsds (for solar)

When super-resolving the ua and va the CCSM data should be formatted the same way as the WIND Toolkit: (None, None, None, [ua, va]). The CCSM rsds is the model's value for GHI and will need to be decomposed into DNI and DHI and shaped as: (None, None, None, [DNI, DHI]). The script (implementation of of TAG and DISC models) to do this can be found in `data_processing/CCSM_GHI_processing.py`.

### Model Weights
Model weights can be found in `models/`. The wind MR and HR models perform a 10x and 5x super-resolution respectively while both solar models perform a 5x super-resolution. Each model is designed to work on the distance scales they were trained on (100 to 10km or 10km to 2km/4km). If you wish to have a different amount of super-resolution you must train the models accordingly.

### Running the Models
An example of how to use the PhIRE GANs model can be found in `main.py`.
It is recommended that data is loaded in as a TFRecord (see Data).
Give more of a description here - load in data, run test.
Outlines of both the pretraining (no adversarial component) and training (with adversarial component) are included in the `PhIREGANs.py` script but their use is not demonstrated in `main.py`.

#### References
[1] Stengel K., Glaws A., Hettinger D., King R. "Physics-informed super-resolution of climatological wind and solar data". 2019  
[2] Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.  
[4] Aguiar, R., and M. T. A. G. Collares-Pereira. "TAG: a time-dependent, autoregressive, Gaussian model for generating synthetic hourly radiation." Solar energy 49.3 (1992): 167-174.  
[5] Maxwell, E.,"DISC Model." Excel Worksheet [link](https://www.nrel.gov/grid/solar-resource/disc.html)  
[6] Holmgren, William F., Clifford W. Hansen, and Mark Mikofski. "pvlib python: a python package for modeling solar energy systems." J. Open Source Software 3.29 (2018): 884.  
[7] Meehl, Gerald A., "CCSM4 model run for CMIP5 with 1\% increasing CO2" (2014). NCAR. doi:10.1594/WDCC/CMIP5.NRS4c1. Served by ESGF (Version 2) [Data set]. World Data Center for Climate (WDCC) at DKRZ.  
[8] Taylor, Karl E., Ronald J. Stouffer, and Gerald A. Meehl. "An overview of CMIP5 and the experiment design." Bulletin of the American Meteorological Society 93.4 (2012): 485-498.  
[9] Cinquini, Luca, et al. "The Earth System Grid Federation: An open infrastructure for access to distributed geospatial data." Future Generation Computer Systems 36 (2014): 400-417.  
[10] Draxl, Caroline, et al. "The wind integration national dataset (wind) toolkit." Applied Energy 151 (2015): 355-366.  
[11] Sengupta, Manajit, et al. "The national solar radiation data base (NSRDB)." Renewable and Sustainable Energy Reviews 89 (2018): 51-60.

#### Acknowledgments
We acknowledge the World Climate Research Program’s Working Group on Coupled Modelling, which is responsible for CMIP, and we thank the climate modeling groups (listed CCSM section) for producing and making available their model output. For CMIP the U.S. Department of Energy’s Program for Climate Model Diagnosis and Intercomparison provides coordinating support and led development of software infrastructure in partnership with the Global Organization for Earth System Science Portals.

This work was authored by the National Renewable Energy Laboratory (NREL), operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by the Laboratory Directed Research and Development (LDRD) Program at NREL. The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
