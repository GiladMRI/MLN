# MLN

Material for: Minimal Linear Networks for MR image reconstructiom

Complex-valued ~~neural~~ linear networks.

The main topology, a *k* and I layers with location-independent (variable-density) kernels with several "time-segments" is able to produce artifact-free images where standard advanced (e.g. compressed sensing, etc.) reconstruction fails.

Example on a Multi-band spiral trajectory with incoherent CAIPI blips:

![Example output](srez_sample_output.png)

Vs. common methods on our benchmark data:

![Example output](srez_sample_output.png)

# Parameters

The system is highly configurable from the human-readable 'params.txt' . Most parameters are rather self-explanatory; some additional information can be found here: https://docs.google.com/document/d/18lZOREQs4aX6HWqjV1Dn5tCwnAgcu9XmiUsXeqp5uRQ/edit?usp=sharing

# Requirements

The code is based on https://github.com/david-gpu/srez, so in case of errors it might be useful to verify srez is working.

You will need Python 3 with Tensorflow, numpy, scipy, h5py and [moviepy](http://zulko.github.io/moviepy/).
See srez -`requirements.txt` for details.

## Dataset
The dataset used for the real data and benchmark test is a collection of randomly chosen slices from the HCP. It can be downloaded from https://figshare.com/s/4e700474da52534efb30 . The data is augmented with random cropping, flipping and 90deg rotation, and a random 2D phase is added. The following parameters deternine the strength of the added phase: 

For the real data, the acquired signal, the trajectory, MIRT-based NUFFT coefficients and time-segments data are included here.
For the benchmark test, the poisson-disc masks and the images used are provided, as well as the resulted recons.

# Training and running the model

Training can be done by calling ''

Running a trained network on a series of .mat files, given in the format Path/Prefix_XX.mat, can be done my setting the following parameters in params.txt:
LoadAndRunOnData=1,LoadAndRunOnData_checkpointP,LoadAndRunOnData_Prefix
LoadAndRunOnData_OutP 
