# goldenspear-AItest

## 1st problem.
We are given a dataset that contains product reviews and a rating for each product. A product can be rated from one to five. Clearly, the review is highly correlated with the final score that the users give to the product. The dataset provided contains 10K samples. We would like you to analyse the dataset and build a model that is able to predict from a given review its score. For this purpose, the use of open source libraries is encouraged.

You can download the dataset from http://app.goldenspear.com/ratings.csv

1. **Data analysis.** Plot the balance of classes and show the five most predominant words for each class.
2. **Data cleaning.** Have you performed data cleaning? If so, what kind of data cleaning and which tools have you used?
3. **Learning process.** Answer briefly these questions:
   - What kind of features have you used?
   - What model or models have you chosen? Why? 
   - What libraries have you used?
4. **Models validation.** Evaluate the performance of your estimator using some validation method and answer these questions:                    
   - What validation method have you chosen? 
   - What evaluation metric have you chosen? 
   - Write down your training and testing accuracies.
5. **Final summary.** Write down what would you have done if we had given you more time and data.        

## 2nd problem. 
We are given a dataset that contains an unlabelled set of images and a smaller dataset that contains labelled images. Those images represent shirts from an e-commerce, so usually the background is plain and the product is the focus of the image. The unlabelled dataset provided contains 5K samples and the labelled one contains just 10 images, all of blue shirts. 
We would like you to analyse the datasets and build a model that is able to retrieve all the blue shirts from the 5K dataset. For this purpose, the use of open source libraries is encouraged.

You can download the datasets from http://app.goldenspear.com/shirts.tar.gz

1. **Problem analysis.** Write down your solution proposal .
> /src/blue_shirts_detector.py
2. **Models validation.** As we don’t have validation samples on the dataset, find a way to visually demonstrate the capacity of the model.
> Una manera de visualizar los resultados sin tener validacion, puede ser con la comprovacion visual de grids de imagenes pequeñas de esos en los que ha acertado. Siendo el dataset mas reducido, podria ser una manera sencialla de comprovar que no haya Falsos Positivos. 
3. **Final summary.** Write down what would you have done if we had given you more time and data.
> La version implementada con la comparacion de histogramas es sencilla y eficiente, sin embargo mecanismos de Machine Learning o de Deep Learnign pueden hacer que incremente su eficacia. 
Por un lado se puede implementar con una cantidad de datos similares metodos de clustering semi-supervisado, donde haciendo uso de librerias de Sklearn se puede implementar un clustering como el Mean-Shift, de tal modo en que aprendiesea diferenciar entre clusters de histogramas  automaticamente del dataset completo sin necesidad de indicar el numero de clases existentes, y luego con el dataset de camisetas azules, identificar esas regiones de puntos que pertenecen al cluster objetivo. Se pueden utilizar las mismas features de entrada que el ejercicio propuesto. Unos histogramas de las Regions of Interest del canal Hue del espacio de color HSV.
Por otro lado si se disponiera de mas dataset de imagenes azules se podria entrenar una CNN con los embedings de las imagenes de entrada 


