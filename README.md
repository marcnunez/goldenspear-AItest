# goldenspear-AItest

## 1st problem.
We are given a dataset that contains product reviews and a rating for each product. A product can be rated from one to five. Clearly, the review is highly correlated with the final score that the users give to the product. The dataset provided contains 10K samples. We would like you to analyse the dataset and build a model that is able to predict from a given review its score. For this purpose, the use of open source libraries is encouraged.

1. **Data analysis.** Plot the balance of classes and show the five most predominant words for each class.

   ![alt text](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/Figure_1.png)
   ![alt text](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/Figure_1-1.png)
   ![alt text](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/Figure_1-2.png)
   ![alt text](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/Figure_1-3.png)
   ![alt text](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/Figure_1-4.png)


2. **Data cleaning.** Have you performed data cleaning? If so, what kind of data cleaning and which tools have you used?
   To perfomr data cleaning the following steps has been used: 
   - Remove numbers and special caracters
   - Tokenize the text
   - Remove stop words such as "and", "the", etc, using the python library NLTK

   In addition there could also be a lemmatization process to find the root word and thus facilitate the grouping of words.
   
3. **Learning process.** Answer briefly these questions:
   - What kind of features have you used?
   > It has been proposed to represent the phrases using token counter matrices. Then transform a count matrix to a normalized tf-idf representation 
   - What model or models have you chosen? Why? 
   > Linear SVM's, Logistics Regression, and Naive Bayes Classifier has been tested. Since they are fast, simples and need less effort with good results.
   - What libraries have you used?
   > Pre-Processing and cleaning of lenguage: NLTK. Data structures: Pandas and Numpy. Models and descirptors: Scikit-learn
4. **Models validation.** Evaluate the performance of your estimator using some validation method and answer these questions:                    
   - What validation method have you chosen? 
   > Cross Validation method has been implemented. The data has been split into training and testing sets, with 70% and 30% of the data respectively, keeping the data classes balanced for each set as shown in question 1.
   - What evaluation metric have you chosen?
   > Acuracy, precision, recall and F1-Score
   - Write down your training and testing accuracies.
   ![alt text](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/models.PNG)
5. **Final summary.** Write down what would you have done if we had given you more time and data.
    > Use deep learning networks such as transofmer structures such as BERT that can be fine tuned for a variety of classic NLP tasks. Besides, bert is already pretrained.
    
## 2nd problem. 
We are given a dataset that contains an unlabelled set of images and a smaller dataset that contains labelled images. Those images represent shirts from an e-commerce, so usually the background is plain and the product is the focus of the image. The unlabelled dataset provided contains 5K samples and the labelled one contains just 10 images, all of blue shirts. 
We would like you to analyse the datasets and build a model that is able to retrieve all the blue shirts from the 5K dataset. For this purpose, the use of open source libraries is encouraged.

1. **Problem analysis.** Write down your solution proposal .

> /src/blue_shirts_detector.py
   - Regions of interest have been taken out. 
   - These regions have been switched to HSV colour space and only the Hue channel has been selected.
   - The histogram has been taken from the blue images. 
   - It has been compared the histograms of each one of the images of train with each one of the images of blue shirts. If it has an intersect superior to 2100 it has been considered blue. 
   - The results have been plotted

2. **Models validation.** As we don’t have validation samples on the dataset, find a way to visually demonstrate the capacity of the model.
> A way to visualize the results without having validation, can be with the visual check of grids of small images of those in which it has succeeded. Being the smallest dataset, it could be a simple way to check that there are no False Positives. 

![alt text-1](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/1_100_blue.PNG) ![alt text-2](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/101_200_blue.PNG)
![alt text-1](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/201_300_blue.PNG) ![alt text-2](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/301_400_blue.PNG)
![alt text-1](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/401_500_blue.PNG) ![alt text-2](https://github.com/marcnunez/goldenspear-AItest/blob/master/results/501_600_blue.PNG)


3. **Final summary.** Write down what would you have done if we had given you more time and data.
> The version implemented with the comparison of histograms is simple and efficient, however mechanisms of Machine Learning or Deep Learnign can make it increase its effectiveness. 
On the one hand it is possible to implement with a quantity of similar data semi-supervised clustering methods, where by making use of Scikit-learn libraries a clustering like Mean-Shift can be implemented, in such a way that it learns to differentiate between clusters of histograms automatically from the complete dataset without needing to indicate the number of existing classes, and then with the blue T-shirt dataset, to identify those regions of points that belong to the target cluster. You can use the same input features as the proposed exercise. Histograms of the Regions of Interest of the Hue channel of the HSV color space.
On the other hand, if more blue image datasets were available, it would be possible to train a CNN with the embedings of the input images. 


