#**Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road

---

### Reflection

###My pipeline consisted of following steps:

First, I converted the images to grayscale, 

then I defined a kernel size and apply Gaussian smoothing,

then I defined thresold parameters (low and high) and applied Canny to the blurred image,

then I defined the hough transform parameters and ran hough on edge detected image,

then iterate over the output lines and drew lines on the image. In order to draw lines on the left and right lanes, I created a region of interest (a triangle) which allowed me to filter out lines which is not in my area of interest.

Then I painted only on lines which were in area of interest.


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the car is moving on the curved path. Since the area of interest is a triangle, it will not cater the need of curved path. 

Another shortcoming could be if the captured area is more than the supplied image size.


###3. Suggest possible improvements to your pipeline

A possible improvement would be to check if any line is intersecting the area of interest and then paint only the section of line coming in the area of interest.

Another potential improvement could be to create area of interest on the fly based on type of path (curved or straight etc.)