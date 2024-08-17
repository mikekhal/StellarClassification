Using Machine Learning To determine Stellar Classification.

This code uses the following dataset from kaggle: https://www.kaggle.com/datasets/deepu1109/star-dataset?select=6+class+csv.csv

The data consists of the following features of stars: Temperature, Luminosity, Radius, Absolute Magnitude, Star Type and Spectral Class.

Some of the relationships between these variables already known through:

•Stefan-Boltzmann Law, which for a spherical blackbody: $L=4 \pi R^2 \sigma T^4$

•Definition of Stellar Magnitudes, $m_i$: $m_1 - m_2 = -2.5log_{10} (\frac{F_1}{F_2}) (F=\frac{L}{4 \pi d^2}$)

•Luminosity - Mass relation, where for high-mass stars with constant opacity ($k=1$): $L \propto M^3$. As R $\propto M^{0.8} for stars, $L \propto R^{4.375}$

The colour of the star is primarily dependent on the star's photospheric temperature. A sketch illustrating the concept of colour indices for blackbodies of different temperatures, where the vertical bands indicate the location of the filter bands (EM wavelength):
![image](https://github.com/user-attachments/assets/95304eb3-e322-443b-932f-0e37c71de16d)
This shows that higher frequencies are to the left of the graph.

A HR diagram (a plot of the luminosity against effective temperature) can be used to reveal distinct groups like the main sequence, giants, and white dwarfs.

![image](https://github.com/user-attachments/assets/276f8d81-6f91-4357-8350-4ec4116542ca)

ML Algorithms used:

                                        Decision Tree Classifier

A Decision Tree Classifier algorithm is a supervising learning method, which splits the data according to the selected feature, and creates left and right branches of the tree until a stopping criteria is met.

The "sklearn" library was used, with a radius condition used as the root node. The model was trained on 70% of the dataset and tested on the remaining 30%. The performance was evaluated using accuracy, a classification report, and mean squared error (MSE) for both the training and test sets.
The decision tree was visualized using the export_graphviz method, which provides a graphical representation of the decision tree:

![star](https://github.com/user-attachments/assets/c0daeb34-57e8-4b7d-9827-d9e34123168e)

**Results**
Accuracy: The Decision Tree model achieved an accuracy of 100% on the test set.
Errors: The training error was 0 (MSE), and the test error was 0 (MSE).

                                        Random Forest Classifier

A random forest is a method that fits a number of decision tree classifiers on various sub-samples of the dataset.

**Results**
Accuracy: The Random Forest Classifier model achieved an accuracy of 100% on the test set.
Errors: The training error was 0 (MSE), and the test error was 0 (MSE).

**Learning Curves**: The learning curves for both models were plotted to visualize the training and validation errors as a function of the training set size.

Decision Tree:

![learning_curve_decision_tree](https://github.com/user-attachments/assets/8b48ee80-c2bd-4986-8aa5-ab465a375b5c)

Random Forest Classifier:

![learning_curve_random_forest](https://github.com/user-attachments/assets/8420053f-aa6a-42c1-9999-490dae8cbba5)

