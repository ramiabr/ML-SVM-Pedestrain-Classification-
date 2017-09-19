# Machine Learning Pedestrain Detection using Support Vector Machine (SVM): 

## Introduction:
This software will detect if humans are present in the given picture or not (output 1 if present or -1 if not present). This project is based on paper Histograms of oriented gradients for human detection. 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05). In this project, supervised learning algorithm will be implemented using Support Vector Machine (SVM) model. The Machine learning algorithm will be provided with labelled data set of images for both positives and negatives. The JPG images will be normalized and Histogram of Gradients (HOG) will extract feature vectors from training dataset. SVM algorithm will come up with classification equation separating positives and negatives. Finally, once the learning is completed the equation will be dumped into a file and used for real time classification. During real time classification, the algorithm will be provided with a real image, it will use the equation and tell us if the picture had human beings or not.


