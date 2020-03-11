# Importance sampling in domain transfer tasks

Helge's comments:
> Do you already have an idea of how to split the work? It is a bit difficult in your case as you need to first implement the baseline (or is there code?), then add the importance sampling, and only once both are running you can move on the the experiments. To counter this, can you create toy example, where you can test the sampling approach? Here some suggestions: 

- [ ] 1. Use a dataset of images, each image either entirely black or entirely white. Use a loss that is 1 for white images and 0 for black images. Can you construct an importance sampling function that selects only the black images (because lower loss function is better).

- [ ] 2. Use a dataset with more colors and different constants for each color, e.g. red = 1, blue =0.5, white = 1, black = 0 ... Can you train an importance sampling approach that matches these 'rewards'? 

- [ ] 3. With similarly simple datasets, can you construct a case where you can test the combination of discriminator and importance sampling (still without a generator)? 

- [ ] 4. Construct a toy example where you test importance sampling, generator and discriminator all together (still very simple 
data where you know how the sampling should look like). 

- [ ] 5. Only then move on to proper images like street numbers and horese and zebra.

> With such toy examples you could split the work in that one of you focusses on reimplementing/making the baseline work (using weighted examples). While the other one focusses on implementing the importance sampling. Of course, you should plan and discuss all the steps together, but it would be good to have a split on the implementation side of things.

So parallel to this:

- [ ] Reimplement the work of Binkowski (weighted loss GAN), and make adaptable for Importance Sampling.
- [ ] Make datasets ready (street number and maybe more)

Datasets:
- street numbers-MNIST https://www.kaggle.com/stanfordu/street-view-house-numbers
http://yann.lecun.com/exdb/mnist/
(already split training and test set so we don't need to)

Dealine: April 14th

Our deadline: April 5th
