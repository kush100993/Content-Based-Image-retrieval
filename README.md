# Content-Based-Image-retrieval
Given a query image retrieve similar images from the database 

This implementation is based on "Aggregating local descriptors into a compact image representation".

The main steps of the paper is: 

* From a database of images extract features from specified number of images and cluster features using k-means to get a specified number of cluster centers.(16 centers prefered)

* Now for each image in the database again extract features and predict the cluster center each feature belongs to for that image, aggregating all features belonging to a cluster center and concatenating them in order to generate a vlad(vector of locally aggregated desriptor) vector.

* Using PCA reduce the dimensions of the vlad vector to (1 x 128) for an image(prefered from the paper).

* Product quantization is used to encode these vectors to generate a codebook(using k-means again) and codes and these are used to retrieve similar images of the given query image.(it is also called approximation of KNN) 
