# Vehicle-Detection-and-Classification
Vehicle Detection using Saliency, K-Means Clustering, Segmentation algorithms &amp; Vehicle Classification using GoogLeNet Caffemodel

# General Idea
The video is pre-processed with 3rd party software to retrieve 2 frames per second (FPS) and the images are saved into the 'road_images' folder.

The images are pre-processed with Gaussian Smoothing to filter out the noises. The images are further processed using Saliency algorithm (ITTI model) to simulate what human eyes are attracted to.

The saliency map will be further processed using K-Means clustering algorithm and the top 2 clusters with most pixel values are removed.

The other image processing techniques such as erosion, noise removal according to coordinates, dilation, region filling are used to extract the vehicles in binary.

The image segmentation will be used to segment the extracted region of vehicles and sent to the pre-trained model for vehicle classification.

The final result will be put into the original image and displayed. 

# Tech/Framework Used
1. OpenCV
2. GoogLeNet Caffemodel

## Languages
1. C++

# Screenshots
## Original Input
![image](https://user-images.githubusercontent.com/63278063/117318705-408d8300-aebd-11eb-82db-0feef118589b.png)

## Saliency Maps
![image](https://user-images.githubusercontent.com/63278063/117318814-58fd9d80-aebd-11eb-8164-4efaea89270e.png)

![image](https://user-images.githubusercontent.com/63278063/117319035-84808800-aebd-11eb-973d-175875665816.png)

![image](https://user-images.githubusercontent.com/63278063/117319079-8e09f000-aebd-11eb-99dc-9460056da4b9.png)

## K-Means Clustering
![image](https://user-images.githubusercontent.com/63278063/117319160-a11cc000-aebd-11eb-9954-b25823117a35.png)

### Removed Top 2 Clusters
![image](https://user-images.githubusercontent.com/63278063/117319215-abd75500-aebd-11eb-9c47-e0df30291d44.png)

## Eroded Image
![image](https://user-images.githubusercontent.com/63278063/117319386-d3c6b880-aebd-11eb-931d-53cfd89f81ec.png)

## Dilated Image
![image](https://user-images.githubusercontent.com/63278063/117319456-e6d98880-aebd-11eb-9269-4a237221c16b.png)

## Segmented Image
![image](https://user-images.githubusercontent.com/63278063/117319592-083a7480-aebe-11eb-8d8a-248da3601876.png)

![image](https://user-images.githubusercontent.com/63278063/117320234-9c0c4080-aebe-11eb-94c3-d6786051e3ca.png)

## Final Result
![image](https://user-images.githubusercontent.com/63278063/117320309-b0503d80-aebe-11eb-929c-a671b4eaa079.png)

![image](https://user-images.githubusercontent.com/63278063/117320708-1210a780-aebf-11eb-8394-3d96d1021265.png)

![image](https://user-images.githubusercontent.com/63278063/117320789-2359b400-aebf-11eb-926e-391de210a5d9.png)

