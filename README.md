# Distribution Aware image-to-image translation

Image-to-image translation methods faithfully transform a source image to the style of a target domain, enabling a breadth of applications in domain adaptation and arts. 
Existing approaches focus on image quality, little attention is given to the distribution of generated images, which by default follows that of the source domain. 
We propose a method to re-sample a set of generated images to match the distribution of the target domain.
At the core is a NN-module for estimating the relative frequency of image constellations in the source and target domains by matching modes of features.
We demonstrate the versatility of our method by adding it todifferent domain adaptation techniques and evaluating on domains with increasing difficulty.
Our method improves downstream tasks, such as learning animal pose from synthetic examples. It complements existing approaches that focus on the quality of a single image but not the distribution of a set of generated images. Its advantage is that it can be added to any of these without change to the underlying image generation and training process.
