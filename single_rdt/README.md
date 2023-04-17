# Speech features are weighted by selective attention

This repository contains the code for simulating the computational experiment of modeling selective attention to speech features.

Abstract:

*Listeners typically rely more on one aspect of the speech signal than another when categorizing speech sounds. This is known as feature weighting. We present a rate distortion theory model of feature weighting and use it to ask whether human listeners select feature weights sim- ply by mirroring the feature reliabilities that are present in their input. We show that there is an additional com- ponent (selective attention) listeners appear to use that is not reflected by the input statistics. This suggests that an internal mechanism is at play in governing listeners’ weighting of different aspects of the speech signal, in addition to tracking statistics.*


## Dependencies

All model code was built in a conda environment with Python version 3.10.5 and TensorFlow version 2.8.2.

The dependencies needed to run this code:<br/>
`tensorflow`<br/>
`keras`<br/>
`itertools`<br/>
`scipy`<br/>
`sklearn`<br/>
`numpy`<br/>
`pandas`<br/>
`matplotlib`<br/>

## Run Experiment
To train and test the model:
`python rdt_single.py`

To test the model:
`python full_test_rdt.py`

To analyze the results and compare to human results (code in R):
`analysis.Rmd`

This repository is still in the process of work.

