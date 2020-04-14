# A Fully Convolutional M2-VAE for Semi-Supervised Semantic Segmentation

To run you will need the following python packages:

*numpy (can be installed with anaconda)
*pandas (can be installed with anaconda)
*keras (can be installed with anaconda)
*scikit-learn (can be installed with anaconda)
*opencv (can be installed with anaconda)
*tensorflow-gpu VERSION 1.15 (can be installed with anaconda) (version is important)
(tensorflow should work instead of tensorflow-gpu so that you can run it on your cpu, although I
have not tested it)
*tensorflow-probability (can be installed with anaconda)
*matplotlib (can be installed with anaconda)
*albumentations (can be installed with pip)


I recommend running the demo first. For the most part, it simply tests the performance of 
the various models. It takes up about 3GBs of gpu memory.

MNIST_Baseline, MNIST_M2_VAE, Pascal_Voc_Baseline, Pascal_Voc_M2_Vae are the scripts
used to create and train the respective models. You are welcome to rerun them, however some take
considerable time and memory to run. The MNIST scripts are the lightest weight to run.

MNIST_Datageneration and Pascal_Voc_Datageneration are the scripts used to adapt the original
datasets to a form useful for semi-supervised learning. Running them could break the other scripts.

Both the raw python scripts and a jupyter notebook are available for each of the scripts. Use whichever you prefer. They do the exact same thing. They were written in a notebook so that
is what I would recommend running them with.



NEW:
As we spoke about, it is probably better if you don't run my programs. Due to its size I have not included the "Pascal VOC" dataset. Because of this, you will not be able to run any of the Pascal_VOC models or the demo. You could still run any of the MNIST models if you would like
