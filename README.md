# Stock Forecasting

This project focuses on building sequence models to forecast Nvidia’s closing stock price using a Recurrent Neural Network (RNN) and a Long Short-Term Memory (LSTM) network. Historical price data was collected from Yahoo Finance and split sequentially into training (60%), validation (20%), and testing (20%) sets to preserve temporal order, which is critical for time series modeling.

The data was log-transformed and standardized to reduce exponential growth and center the values around zero. Sliding windows were then used to convert the time series into sequences suitable for training. Both models were implemented in PyTorch with identical architectures (input size 1, hidden size 8, one recurrent layer) and trained using the Adam optimizer and Mean Squared Error loss for 10 epochs.

Model performance was evaluated on a completely unseen test set using regression metrics including MSE, RMSE, MAE, and R². The LSTM outperformed the RNN across all metrics, which aligns with its ability to better capture longer term temporal dependencies. This project provided hands on experience with sequence modeling and highlighted the importance of proper data splitting and preprocessing for time series forecasting.

Overall, I learned:
- How to build and train RNN and LSTM models in PyTorch
- How to correctly preprocess and split time series data
- How sliding windows affect sequence learning and data leakage
- How to evaluate forecasting models using regression metrics