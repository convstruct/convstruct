<br>
<br>

<div align="center">
  <img src="https://i.ibb.co/K0rVXRc/convstruct-github-logo.png" alt="convstruct-github-logo" border="0">
</div>

<br>
<br>

<div align="center">
<strong>What if AI could learn, design and create all by itself, and all it needed to do this was a folder of images?</strong>
<br>

[Convstruct](https://convstruct.org) 
is an open-sourced project to answer that very question. A system designed to create AI all by itself, all it needs is a folder of unlabeled images. The project is divided into three sections: [Datasets](https://convstruct.org/datasets) to help you gather a folder of images, an [Application](https://convstruct.org/app) for an accessible experience that requires no coding knowledge, and this GitHub repository that provides the three functions that make up Convstruct.
For inquiries into the project contact:
hello@convstruct.org
</div>

<br>
<br>

<div align="center">
<img src="https://i.ibb.co/kgG3fR5/cs-results.png" alt="cs_results" border="0">
</div>
<div align="center">
These are the results of using Convstruct and Manga109 dataset, see citations.
</div>

<br>
<br>

## How does it work
3 functions are combined in series to make an AutoML system: 

<br>

### convstruct.<strong>learn()</strong>
  - Based on [Invariant Information Clustering Classification](https://arxiv.org/abs/1807.06653) by Xu Ji, João F. Henriques, Andrea Vedaldi, this function creates 2 AI classifiers from unlabeled data you pass to it. One for common data and one for uncommon.
  <img src="https://i.ibb.co/fGpCDGJ/graphic-a-full.png" alt="graphic-a-full" border="0">

### convstruct.<strong>live()</strong>
  - This function cycles through randomized WGAN-GPIICC models to teach, a Multilayer Perception classifier, what model best fits the unlabeled data you pass it.
  <img src="https://i.ibb.co/gtJDR4V/graphic-b.png" alt="graphic-b" border="0">

### convstruct.<strong>draw()</strong>
  - This function trains a final model using the classifiers from convstruct.learn() and convstruct.live() and can load that final graph to output an image.
  <img src="https://i.ibb.co/X5DfW3m/graphic-c.png" alt="graphic-c" border="0">

## How to use
Convstruct can be used through the prepackaged convstruct.exe, a user interface to make the experience more accessible, or through the convstruct codebase.

<br>

### V1 Specifications and requirements:
- The following Nvidia GPU requirements support creating images with the following sizes:
  - 256px by 256px requires 17GB+ of memory.
  - 128px by 128px requires 7GB+ of memory.
  - 64px by 64px requires at least 4GB of memory.
- V1 takes ~2 days to complete with 4 GPUs, ~4 days with 2 GPUs, and ~6 days with 1 GPU.
> If no Nvidia GPU is detected, the CPU will be used and will create 64px by 64px images.

> Built with Tensorflow GPU 1.14 and Python 3.6.


<br>

### Getting started with the desktop application
- Download the [application](https://github.com/convstruct/convstruct/releases/download/v1.0.0/Convstruct.exe) to your local machine.
- Prepare a dataset, if you don't have one, check out [Datasets](https://convstruct.org/datasets).
- Run the application.
- Follow the application's onboarding to get started.

<br>

### Getting started with the codebase
- Clone this repo to your local machine using `https://github.com/convstruct/convstruct`
- Run ``pip install requirements.text`` in the cloned repo to make sure you have all the dependencies installed on your system.
> Also if you have an nvidia gpu, to make sure it is used, make sure to have Cuda v10.0 installed as well as CuDNN.
#### Example usage
```
import convstruct

cs = Convstruct(compdir=ground_truth_folder_path, indir=optional_starting_point_folder_path)
cs.learn
cs.live()
cs.draw(3)
tf.reset_default_graph()
cs.draw(4)
```
<br>

### Preparing for your first Convstruct session
- You will need a folder containing a minimum of ~5000 images, however the best results come from datasets that are larger then 50000 images. 
- Know that you can stop your session at any time and restart right where you last left off.

<br>

## State of Convstruct
####convstruct.learn():
- IICC has provided a fast solution for clustering unsupervised images, however as this is a recent paper the hyper parameters are still being researched.
- Classifying with IICC on unsupervised images is still a work-in-progress as this tends to require semi-supervision, whereas convstruct is fully unsupervised.
- Using the second layer output of IICC as part of the loss function for the GAN-WP is an experimental concept.

####convstruct.live():
- The time it takes to complete the live() function is a major challenge.
- Progress outputs from the quick training sessions lead to reasonable shapes, however even at a tenth of the epochs used to train the final model still takes a while to complete.
- The image size is reduced as a way to speed up the sessions in the live() function, however it could be possible that reducing the size adds too much noise to the model classifier.

####convstruct.draw():
- There is a bottleneck between epochs as seen in the gpu usage throughout the session on certain hardware.

## Future work
- Removing the challenges facing convstruct.learn() by researching or replacing IICC completely with a more suitable solution.
- Continuing to adjust convstruct.live() and convstruct.draw() functions to reduce the time it takes to complete.
- The main theme of future work will be on data requirements. Currently, the amount and type of data needed for quality results is a limiting factor for generative AI.

<br>

## Citations

    @article{mtap_matsui_2017,
        author={Yusuke Matsui and Kota Ito and Yuji Aramaki and Azuma Fujimoto and Toru Ogawa and Toshihiko Yamasaki and Kiyoharu Aizawa},
        title={Sketch-based Manga Retrieval using Manga109 Dataset},
        journal={Multimedia Tools and Applications},
        volume={76},
        number={20},
        pages={21811--21838},
        doi={10.1007/s11042-016-4020-z},
        year={2017}
    }
    
    @article{multimedia_aizawa_2020,
        author={Kiyoharu Aizawa and Azuma Fujimoto and Atsushi Otsubo and Toru Ogawa and Yusuke Matsui and Koki Tsubota and Hikaru Ikuta},
        title={Building a Manga Dataset ``Manga109'' with Annotations for Multimedia Applications},
        journal={IEEE MultiMedia},
        volume={27},
        number={2},
        pages={8--18},
        doi={10.1109/mmul.2020.2987895},
        year={2020}
    }
    
    @article{ji2019invariant,
      title={Invariant Information Clustering for Unsupervised Image Classification and Segmentation}, 
      author={Xu Ji and João F. Henriques and Andrea Vedaldi},
      year={2019},
      eprint={1807.06653},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

<br>
