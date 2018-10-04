# MLN

Minimal Linear Networks for MR image reconstructiom

![Example output](srez_sample_output.png)

As you can see, the network is able to produce a very plausible reconstruction of the original face. As the dataset is mainly composed of well-illuminated faces looking straight ahead, the reconstruction is poorer when the face is at an angle, poorly illuminated, or partially occluded by eyeglasses or hands.

This particular example was produced after training the network for 3 hours on a GTX 1080 GPU, equivalent to 130,000 batches or about 10 epochs.

# How it works

In essence the architecture is a DCGAN where the input to the generator network is the 16x16 image rather than a multinomial gaussian distribution.

In addition to that the loss function of the generator has a term that measures the L1 difference between the 16x16 input and downscaled version of the image produced by the generator.

The adversarial term of the loss function ensures the generator produces plausible faces, while the L1 term ensures that those faces resemble the low-res input data. We have found that this L1 term greatly accelerates the convergence of the network during the first batches and also appears to prevent the generator from getting stuck in a poor local solution.

Finally, the generator network relies on ResNet modules as we've found them to train substantially faster than more old-fashioned architectures. The adversarial network is much simpler as the use of ResNet modules did not provide an advantage during our experimentation.

# Requirements

You will need Python 3 with Tensorflow, numpy, scipy and [moviepy](http://zulko.github.io/moviepy/). See `requirements.txt` for details.

## Dataset

After you have the required software above you will also need the `Large-scale CelebFaces Attributes (CelebA) Dataset`. The model expects the `Align&Cropped Images` version. Extract all images to a subfolder named `dataset`. I.e. `srez/dataset/lotsoffiles.jpg`.

# Training the model

Training with default settings: `python3 srez_main.py --run train`. The script will periodically output an example batch in PNG format onto the `srez/train` folder, and checkpoint data will be stored in the `srez/checkpoint` folder.

After the network has trained you can also produce an animation showing the evolution of the output by running `python3 srez_main.py --run demo`.
