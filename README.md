## Physics-Informed Resolution-Enhancing GANs (PhIRE GANs)
**Inlcuding solar for now because it is easier to clean everything here. Will remove before PhIREGANs goes public.**   
___
### Requirements
- Python v3.7
- TensorFlow v1.13.1
- Recommended
    - numpy v1.16.3
    - matplotlib v3.0.3

### Data

##### WIND Toolkit
LR, MR, and HR wind example data can be found in `example_data/`. These datasets are from NREL's WIND Toolkit. The LR and MR data are to be used with the MR and HR models respectively. If you would like to use your own data for the super-resolution it must have the shape: (None, None, None, 2).
The scripts are designed to take in TFRecords. An example of how to convert numpy arrays to TFRecords can be found in `data_processing/TFRecords_gen.py`.

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

When super-resolving the ua and va the CCSM data should be formatted as : (None, None, None, [ua, va]). The CCSM rsds is the model's value for GHI and will need to be decomposed into DNI and DHI and shaped as : (None, None, None, [DNI, DHI]). The script (implementation of of TAG and DISC models) to do this can be found in `data_processing/CCSM_GHI_processing.py`.

### Model Weights
Model weights can be found in `models/`. The wind MR and HR models perform a 10x and 5x super-resolution respectively while both solar models perform a 5x super-resolution. Each model is designed to work on the distance scales they were trained on (100 to 10km or 10km to 2km/4km). If you wish to have a different amount of super-resolution you must train the models accordingly.

### Running the Models
An example of how to use the PhIRE GANs model can be found in `example.py`.
It is recommended that data is loaded in as a TFRecord (see Data).
Give more of a description here - load in data, run test.
Both the pretraining (no adversarial component) and training (with adversarial component) are included in the `PhIREGANs.py` script but their use is not demonstrated in `example.py`.

#### References
[1] Our paper here  
[2] SRGANs model  
[3] TensorFlow  
[4] Python  
[5] TAG model  
[6] DISC model
[7] CCSM
[8] ESGF

#### Acknowledgments
Thank NSRDB, WTK, NREL, CCSM.

##### Todo:
- make code flexible for argument inputs
- make code flexible for surface roughness inputs
- add in catching errors
