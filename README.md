# MLN

Material for: Minimal Linear Networks for MR image reconstructiom

Complex-valued ~~neural~~ linear networks.

The main topology, a *k* and I layers with location-independent (variable-density) kernels with several "time-segments" is able to produce artifact-free images where standard advanced (e.g. compressed sensing, etc.) reconstruction fails.

Example on a Multi-band spiral trajectory with ncoherent CAIPI blips:
![Example output](srez_sample_output.png)

Vs. common methods on our benchmark data:

![Example output](srez_sample_output.png)

# Parameters

The system is highly configurable from the human-readable params.txt . Most parameters are rather self-explanatory. Otherwise, some additional information can be found here: https://docs.google.com/document/d/18lZOREQs4aX6HWqjV1Dn5tCwnAgcu9XmiUsXeqp5uRQ/edit?usp=sharing

# Requirements

The code is based on https://github.com/david-gpu/srez, so in case of errors it might be useful to verify srez is working.

You will need Python 3 with Tensorflow, numpy, scipy, h5py and [moviepy](http://zulko.github.io/moviepy/).
See srez -`requirements.txt` for details.

## Dataset
The dataset used for the real data and benchmark test is random collection of slices from the HCP. It can be downloaded from https://figshare.com/s/4e700474da52534efb30 .

For the real data, the acquired signal, the trajectory, MIRT-based NUFFT coefficients and time-segments data are included here.
For the benchmark test, the data is included here.

# Training the model

Training with default settings: `python3 srez_main.py --run train`. The script will periodically output an example batch in PNG format onto the `srez/train` folder, and checkpoint data will be stored in the `srez/checkpoint` folder.

After the network has trained you can also produce an animation showing the evolution of the output by running `python3 srez_main.py --run demo`.
