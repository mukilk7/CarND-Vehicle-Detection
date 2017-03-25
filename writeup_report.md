# Vehicle Detection Project

### - Mukil Kesavan

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[i1]: ./examples/car_not_car.png
[hog1]: ./output_images/car-hog1.png
[hog2]: ./output_images/car-hog2.png
[hog3]: ./output_images/car-hog3.png
[scale1]: ./output_images/scale125.png
[tperf1]: ./output_images/test-img-perf-1.png
[tperf2]: ./output_images/test-img-perf-2.png
[sl1]: ./output_images/sl-orig.png
[sl2]: ./output_images/sl-win1.png
[sl3]: ./output_images/sl-win2.png
[sl4]: ./output_images/sl-win3.png
[sl5]: ./output_images/sl-win4.png


[video1]: ./solution.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the IPython notebook `./solution.ipynb` under the code cell titled `Compute Input Features`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][i1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][hog1]
![alt text][hog2]
![alt text][hog3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I settled on the final choice of HOG parameters as follows:

* Plotting HOG images for different parameter settings: I picked parameters that showed that the visual output was able to pick out the distinct outline of a car in an image.
* Classifier performance on test set: I ensured that the parameters were able to maximize the classifier accuracy on the test set and also the test images.
* In addition, I also ran the whole vehicle detection pipeline on the test images to make sure that the HOG parameters picked out the vehicles correctly with little false positives/negatives.

Final HOG Parameters:

| Parameter         | Value         | 
|:-----------------:|:-------------:| 
| orientations      | 9             | 
| pixels per cell   | (8, 8)        |
| cells per block   | (2, 2)        |
| hog channel       | ALL           |
| color space       | YCrCb         |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for extracting the HOG features from an image and training the classifier is contained in the IPython notebook `./solution.ipynb` under the code cell titled `Classifier Training` (please see the function getTrainedClassifier()). I used the dataset that came with the project repository. I noticed that the number of images of car and non-car type was imbalanced. So I ended up using the `min(# cars, # non-cars)` to create a balanced dataset. I then split the images into training and testing set (80-20 split) for training a Linear SVM classifier with the default rbf kernel. I also normalized and mean shifted the features extracted from the dataset images. After a lot of trial and error, I found out that the spatial and color histogram features did not improve the accuracy of the classifier and ended up adding more noise. Therefore, I ended up using only the HOG features which gave me an accuracy of 0.9903 on the test set for the Linear SVM classifier. I also tried the DecisionTreeClassifier but the accuracy wasn't nearly as good that of Linear SVM. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window vehicle detection code is in the IPython notebook `./solution.ipynb` under the code cell titled `Sliding Window Technique` (please see the function slidingWindowCarDetect()). I used hog subsampling to make sure that I only compute the hog features once for a given image/video frame. I also only scanned the bottom half for efficiency. I used the following approach to select the ideal scale from the set of ones I tried (1 to 2.5 with steps of 0.25):

* The scale should not be too small to miss key features of a car
* The scale should not be too large so that a lot of non-car background seeps into the patch fed to the classifier

I plotted the different scales and looked at the output to find the right tradeoff between the above two parameters. I also ran the full vehicle detection pipeline for different scales to see the finished output for the test images. After all of this I settled on just using a `single scale of 1.25`. I also ended up sliding each window by just a single cell to ensure that I don't miss any distant cars that appear small and also produce a richer set of overlapping bounding boxes around a car which came in handy for reducing false positives. Here is how the set of sliding windows of scale 1.25 with a step of 1 looked like when overlayed on a test image:

![alt text][scale1]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The code for the overall vehicle detection pipeline is in the IPython notebook `./solution.ipynb` under the code cell titled `Overall Pipeline`. I tried using multiple scales and it ended up taking a long time for each video frame and it didn't help with improving the false positive or false negative rate. Therefore, I ended up using just a single scale. I instead used multi-frame smoothing to achieve a good result without much overhead (explained in detail below). As explained before, I used a YCrCb 3-channel HOG features and none of the spatial and color of histogram features.  Here is the result of the pipeline on some test images:

![alt text][tperf1] ![alt text][tperf2]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://youtu.be/HCYQO81Kp5U)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is in the IPython notebook `./solution.ipynb` under the code cell titled `Combining Overlaps and False Positives Removal`. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap of car pixels for the given video frame. I also combined heatmaps from multiple video frames (12 in my case) to produce a denser heatmap that would have a lot of heat around actual cars vs. false positives. I then applied thresholding to this multi-frame heatmap. Finally, I used `scipy.ndimage.measurements.label()` to identify individual blobs in the combined heatmap.  I assumed each blob corresponded to a vehicle,  constructed bounding boxes to cover the area of each blob detected and drew them back onto the original image.

Here's an example result showing overall process step by step:

![alt text][sl1] ![alt text][sl2]

![alt text][sl3] ![alt text][sl4]

![alt text][sl5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overall, the false positives removal was a really challenging aspect of this project that required several video renderings with tweaked parameters. I am satisfied with the approach I took although it still produces one or two false positives. Heatmap thresholding and smoothing bounding boxes over multiple frames helped a lot. In the future, I think some amount of image augmentation to account for less than ideal lighting conditions and a better classifier based on a deep neural network might help with further reducing the false positive rate. Also, in some frames of the video where a car was passing another to the right of the camera, the vehicle detection pipeline grouped them as a single vehicle. This may not be ideal and I need to include enough examples to account for this case. Another area that I could improve on is the processing speed per video frame in my pipeline. The video renderings took several minutes even though the original video is barely a minute long. I think techniques that can work with an entire video frame as opposed to sliding windows would help a lot in making the pipeline work in real time. I believe that some deep neural network architectures may also be able to help with this aspect.

