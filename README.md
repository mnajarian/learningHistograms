# learningHistograms
Using deep learning to learn pixel intensity histograms of texture images. 

Final project for COMP 777, Optimal Estimation in Image Analysis.

- `GenerateData.py`: Generates a dataset of pixel values and the corresponding binned histogram values. The data originates from
100 randomly-selected 24x24 pixel crops of each of the original 112 Brodatz texture images. 
- `GenerateHistograms.py`: Generates histogram shape plots for each of the Brodatz texture images. 
- `trainMLP.py`: Trains a multilayer perceptron network on data found in `textureHistograms.pkl`. Optional parameters include number of 
hidden nodes and learning rate.
- `testModel.py`: Tests a designated model on the test data found in `textureHistograms.pkl`.
