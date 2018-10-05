# MLN

Material for: Minimal Linear Networks for MR image reconstructiom

The main topology, Complex-valued ~~neural~~ linear networks, a *k* and I layers with location-independent (variable-density) kernels with several "time-segments" is able to produce artifact-free images where standard advanced (e.g. compressed sensing, etc.) reconstruction fails.

Example on a spiral trajectory, compared vs. ESPIRiT with an augmented signal model with B0 inhomogeneities correction:

![Fig5_ReconsOnRealData-01](Fig5_ReconsOnRealData-01.png)

The difference is emphasised in the case of a multi-band spiral trajectory with with incoherent CAIPI blips, vs. BART with augmented signal model:

![BARTMLN_MB](BARTMLN_MB.png)

Vs. common methods on our benchmark data:

![Vs. common methods on our benchmark data:](Benchmark.png)

## TF-NUFFT
NUFFT with B0 field inhomogeneities correction by the time-segments method is implemented in tensorflow to run on the GPU.

# Parameters

The system is highly configurable from the human-readable `params.txt` . Most parameters are rather self-explanatory; some additional information can be found here: https://docs.google.com/document/d/18lZOREQs4aX6HWqjV1Dn5tCwnAgcu9XmiUsXeqp5uRQ/edit?usp=sharing

# Requirements

The code is based on https://github.com/david-gpu/srez, so in case of errors it might be useful to verify srez is working.

You will need Python 3 with Tensorflow, numpy, scipy, h5py and [moviepy](http://zulko.github.io/moviepy/).
See srez -`requirements.txt` for details.

The "CUDA_VISIBLE_DEVICES" is set at the beginning of `srez_main1.py` . The code uses some forced 'with /gpu:0' without soft allocation, to verify the main computation is done entirely on the GPU.

# Dataset
The dataset used for the real data and benchmark test is a collection of randomly chosen slices from the HCP. It can be downloaded from https://figshare.com/s/4e700474da52534efb30 . The data is augmented with random cropping, flipping and 90deg rotation, and a random 2D phase is added. The following parameters deternine the strength of the added phase: RandomPhaseLinearFac, RandomPhaseQuadraticFac, RandomPhaseScaleFac

For the real data, the acquired signal, the trajectory, MIRT-based NUFFT coefficients and time-segments data are included here.
For the benchmark test, the poisson-disc masks and the images used are provided, as well as the reconstructed images using various mathods.

# Training and running the model

Training can be done by calling `python srez_main1.py`

If *ShowRealData*=1, the output every *summary_period* (minutes) will include a run on the data in *RealDataFN*.

Running a trained network on a series of .mat files, given in the format Path/Prefix_XX.mat, can be done my setting the following parameters in params.txt:
LoadAndRunOnData=1,LoadAndRunOnData_checkpointP,LoadAndRunOnData_Prefix
LoadAndRunOnData_OutP - see example in current `params.txt`
