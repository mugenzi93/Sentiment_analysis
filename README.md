# Sentiment Analysis

Recently, 1D convolutional neural network (CNN), typically used with dilated kernels, have been used with great success for audio generation and machine translation. In addition to these successes, it has long been known that small 1D convnets can offer a fast alternative to RNNs for simple tasks such as text classification and timeseries forecasting.

In this project, we will see how to classify time series data using 1D convnets and how to best avoid overfitting as well as ways in which hyperparameters can be tuned to improve the model's accuracy. This mini project follows two main steps: first is the data preparation to turn the dataset into an acceptable form we can ingest in the deep neural netwaork. Second is the model building and tuning to improve the model's accuracy.

It is worth noting that classical machine learning methods could be used for time series data, but would require hand crafting features using feature engineering. Deep learning methods such as Recurrent neural nets or convolutional neural nets have shown to provide state-of-the-art results on challenging activity recognition tasks with little or no feature engineering used.