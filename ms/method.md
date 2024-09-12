# Materials and Methods

## Data Collection and Annotation

The image dataset used in this study was organized into seven distinct subsets, as shown in Figure 1a. These subsets are labeled as follows: “Train” with 954 images (mean: 9.20 ants per image, standard deviation: 7.37), “A01” with 378 images (mean: 4.59, standard deviation: 3.83), “A02” with 289 images (mean: 4.38, standard deviation: 4.30), “A03” with 56 images (mean: 6.11, standard deviation: 6.15), “B01” with 151 images (mean: 13.75, standard deviation: 13.94), “B02” with 206 images (mean: 0.69, standard deviation: 1.26), and “B03” with 60 images (mean: 713.70, standard deviation: 442.76). The values in parentheses indicate the mean number of ants per image and the standard deviation, which reflects variability in the number of ants.

The “Train” subset shares a similar imaging background with the subsets labeled “A” (i.e., “A01”, “A02”, and “A03”). In contrast, the “B” subsets exhibit more complex imaging conditions: “B01” includes images with black stains resembling ants, “B02” contains images with non-uniform backgrounds, and “B03” represents images with dense ant populations containing more than 700 ants per image on average.

To train the CV system to recognize ants, the images were annotated in the YOLO object detection format. In this format, each ant in an image is tracked with a bounding box, which is defined by four coordinates: the x and y positions of the box’s center, the width, and the height. These coordinates are normalized to the range [0, 1] by dividing the x and y values by the image’s width and height, respectively. Each ant is also assigned a class ID. Since the focus of this study is solely on detecting ants without distinguishing between species, all ants were assigned a class ID of 0.



## Study 1: Determining the Amount of Image Resources Required for Generalization

The “Train” subset was used to calibrate the CV system, while the other subsets were employed to evaluate its generalization capabilities. The “A” subsets were designed to mimic the situation where the system is deployed in an ongoing and repetitive experiment, with images having similar imaging backgrounds. In contrast, the “B” subsets aim to test the system robustness in handling images with less controlled imaging conditions. This evaluation 

whereas the “B” subsets tested the system’s ability to generalize to images with different, more complex backgrounds. The “B03” subset specifically evaluated the system robustness on densely populated ant images.




(1) determining the amount of image resources required to generalize the computer system to new images with either the same or different imaging settings, (2) investigating how the system handles dense imaging settings, and (3) exploring how computer vision enables the understanding of spatial and temporal aspects of ant grazing behaviors.