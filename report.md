#**Finding Lane Lines on the Road** 

Self-driving car needs percieve the world as humans do when they drive. Humans use their eyes to figure out how fast they go, where the lane lines are and where are the turns. Car does not have eyes but self-driving cars can use cameras and other sensors to keep the similar function. So what does cameras are seeing as we drive down the road is not the same as what we perceive. We have to teach the car how to percieve lane lines and turns. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

Here we are using [OpenCV](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)_A comprehensive Computer Vision library to understand captured images from the car and translate them to mathematical entities so we can show the car where the lane line is to follow. The algorithm I have chosen has multiple steps for maniplating each frame image in order to reduce or better said eliminate noises as much as possible. Since the car does not need to see the trees or clouds in the sky for detecting lane lines. In addition to reducing noises we try to highlight lane lines as much as possible by highlighting them using some image processing algorithms. 
But the question is what features of lane lines do we need to highlight? Well, we can leverage following features to best identify various lane line in the image:
* color
* shape
* orientation
* position of the image

The line detection algorithm has multiple steps and the most important thing is to tweek all the required parameters well enough to not to loose any valuable pixle in the image. 

## Basics

The first step in detecting lane lines is understanding how images are represented. A 2D image can be represented as a rectangular grid, composed of many square cells, called pixels. Just like the black and white squares on the chessboard, pixels are nicely aligned in straight lines, both horizontally and vertically. We will refer to the horizontal ones as rows and to the vertical ones as columns. It is easy to see that a chessboard has 8 rows and 8 columns. But an image can have many rows and columns and we can find that out by using OpenCV **image.shape** . This returns a tuple of number of rows, columns and channels (if image is color).

``` python
  >> image = cv2.imread('kitten.jpg')
  >> print(image.shape)
    (342, 548, 3)
```
If you want to know how many pixles this image has you can use **img.size** which returns the number of pixles in the image.
Each pixle in the image is represented in a 3D space as (R, G, B) values. Each of these values is in range [0-255]. So with this explanation we saw that images are basically **tensors** with different number of rows, columns and elements per each color channel.

Understanding of mathematical features of an image and its representation is a great help solving computer vision challanges and understanding image processing algorithms.


### Lane line detection algorithm
![alt img](./test_images/solidWhiteCurve.jpg)
![alt img](./test_images/solidYellowCurve.jpg)
Now let's desribe the used algorithm to detect lane lines step by step.
Assume we have the above image captured by our car in a highway. 
1. **Grayscale**
  As you see lanes are either white or yellow on the streets. So we need to identify both.First we need to convert our image shape from a tensor (A, B, C) to a Matrix (A, B) to be able to only deal with raw pixles. In this case yellow and white considered both the same. In order to achieve that we can use *OpenCV GrayScale* method.
  ``` python
  def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ```

2. 

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
