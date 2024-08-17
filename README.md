Using Machine Learning To determine Stellar Classification.

This code uses the following dataset from kaggle: https://www.kaggle.com/datasets/deepu1109/star-dataset?select=6+class+csv.csv

The data consists of the following features of stars: Temperature, Luminosity, Radius, Absolute Magnitude, Star Type and Spectral Class.

Some of the relationships between these variables already known through:

•Stefan-Boltzmann Law, which for a spherical blackbody: $L=4 \pi R^2 \sigma T^4$

•Definition of Stellar Magnitudes, $m_i$: $m_1 - m_2 = -2.5log_{10} (\frac{F_1}{F_2}) (F=\frac{L}{4 \pi d^2}$)

•Luminosity - Mass relation, where for high-mass stars with constant opacity ($k=1$): $L \propto M^3$

The colour of the star is primarily dependent on the star's photospheric temperature. A sketch illustrating the concept of colour indices for blackbodies of different temperatures, where the vertical bands indicate the location of the filter bands (EM wavelength):
![image](https://github.com/user-attachments/assets/95304eb3-e322-443b-932f-0e37c71de16d)
This shows that higher frequencies are to the left of the graph.

A HR diagram (a plot of the luminosity against effective temperature) can be used to reveal distinct groups like the main sequence, giants, and white dwarfs.

![image](https://github.com/user-attachments/assets/276f8d81-6f91-4357-8350-4ec4116542ca)

A split of 30/70 was made for the training to testing data. 
