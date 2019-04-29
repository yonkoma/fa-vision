Finite State Automata Detection and Recognition
===

Ricardo Aguilera, Michael Carenzo, Kalyan Krause, James Spann


Project Description
---
Finite State Automata (FSA) are a representative way to define a computational model. The diagram associated with FSAs are comprised of different shapes, line configurations and letters that all demonstrate a theoretical machine. Our project seeks to recognize a hand drawn FSA diagram and all of its corresponding components using Computer Vision techniques. We use a unique methodology to recognize the diagram created by a user.

Methodology
---
There were several known problems that had to be solved before we could practically convert images of Finite State Machines to their formal representation. The five core challenges included:

* Glare Reduction
* State Recognition
* Text Classification
* Arrow Detection
* Arrow Label Detection

While we were able to solve the problem of State Recognition rather quickly, the remaining four problems were significantly more challenging.

#### Glare Reduction
An early problem was the presence of glare and inconsistent lighting in images. Multiple approaches were tested to deal with this. A first approach was using Sobel gradients in both directions and merging them. Noise was then removed using an area based threshold on connected components. This gave fairly good results, but was still very unstable. Adaptive thresholding also gave decent results. Both methods suffered from changing the exact borders of the arrows, leading to some later processing for arrow detection to fail. Another attempt was made by running otsu thresholding separately on sections of the image, and while this might have some potential, it didn’t work out at all for us.

#### Text Classification
In order to derive the Finite State Machine’s alphabet and state labels we needed to be able to classify the characters in each subregion containing text. We first attempted to do this by preprocessing each subregion of text and then running K-means against the EMNIST dataset. This proved to be a poor way of attempting this task as the K-means algorithm could make classifications based on pixel values but could not take those values into account spatially. We used a convolutional neural network to better classify the individual letters in our diagram.

#### Arrow Detection
Arrow detection was the next challenge that had to be handled. More specifically, we had to determine the coordinate each arrow started from and pointed to. Unfortunately, there weren’t many out-of-the-box solutions for this problem. Ultimately, we divisied two different approaches for this challenge.


Our first approach primarily involved Skeletonization and a variation of the Hit-Or-Miss algorithm. This method involved removing the regions of a binary image containing states (thus only leaving the arrows) and then performing the following steps:

1. Iterate over all the contours above a specified area and extract their bounding box. Then for each subregion (contained by the bounding box):
   1. Erode the subregion so that the tail of the arrow is removed.
   2. Iterate over all the contours remaining in the subregion and select the largest one. (This in theory is the head of the arrow) Then mark the centroid of this contour as the arrow’s end.
   3. Skeletonize the original (uneroded) subregion.
   4. Apply the Hit-Or-Miss algorithm 4 times to the skeletonized arrow to remove pixels that don’t belong to line-caps. (Each application rotates the line-cap kernel 90-degrees) Then, XOR the 4 resulting images together and dilate to get an image only containing the end caps of the arrow.
   5. Once again iterate over contours of the generated image and select the contour centroid farthest from the arrow’s end. (This should be the arrow’s starting point)

This method, while usually fairly successful, struggled with arrows that looped back on themselves. To solve this problem, a different approach was devised.


Our second approach involved using colored points to mark the start and ends of the arrows. This additional information allowed us to improve our arrow detector significantly. This arrow detector works by performing the following steps:

###### Generate an arrow mask
1. Generate a mask of the arrow bases. Store the centroid of each connected component in the mask.
2. Generate a mask of the head bases. Store the centroid of each connected component in the mask.
3. Remove the arrow and head mask from the thresholded image mask. Fully fill each state mask blob, then flood fill each state in black (removing it). This leaves only the arrow lines and their labels. Call this the arrow-label mask.
4. Reinsert the arrow base blobs to the arrow-label mask, then flood fill from each base centroid pixel with a unique color. This colors the line segments without coloring arrow labels.
5. Throw away the pixels that don’t have that unique color.

That gives us an arrow mask we can use in the next step.

###### Find which start point goes to which end point
1. Generate a new mask by or-ing together the arrow mask, the base mask, and the head mask. This is the arrow-base-head mask.
2. Flood fill the arrow-base-head mask from each base centroid with the array position of the centroid, starting at 1. So one connected base-arrow-head blob will be colored 1, the next one will be colored 2, and so on. This color spreads to the arrow head.
3. Check the color c of each arrow head centroid. Look up the c-th element of the base centroid array.
4. Append the element [arrow head centroid, c-th element of base centroid array] to the list of results.

This gives us a list of [arrow start, arrow end] pairs.
This solution was much more dynamic and provided much better results than the former method.

#### Arrow Label Detection
The arrow labels are found by relating the arrow body centroids with the [arrow start, arrow end] pairs, then finding the non-arrow, non-state blob that is closest to that centroid.

Future Work
---
Our work successfully focuses on the recognition and detection of specific parts of the Finite State Machine. Our techniques as described above describe how we were able to define and recognize an outline of the state machine both efficiently and productively. A next step would be to use this outline to represent the model in memory and have it run on a user’s computer. We hope that this work inspires future works that are able to solve other problems in a similar vein.

References
---
* Abecassis, Félix. “OpenCV - Morphological Skeleton.” Flix Abecassis, 20 Sept. 2011, felix.abecassis.me/2011/09/opencv-morphological-skeleton/.
* Cohen, Gregory, et al. "EMNIST: an extension of MNIST to handwritten letters." arXiv preprint arXiv:1702.05373 (2017).
* “OpenCV Hit-or-Miss.” OpenCV, docs.opencv.org/trunk/db/d06/tutorial_hitOrMiss.html.