# Tesla Stock Price Forecasting: LSTM vs RNN Analysis

This notebook implements and compares two popular time series forecasting models (LSTM and RNN) for predicting Tesla stock prices. Below is a detailed explanation of each component.

## 1. Libraries and Dependencies

The code begins by importing necessary libraries:

| Library | Purpose |
|---------|---------|
| **pandas & numpy** | Data manipulation and numerical operations |
| **seaborn & matplotlib** | Data visualization |
| **sklearn** | Data preprocessing, model evaluation, and train-test splitting |
| **tensorflow.keras** | Building and training deep learning models |
| **time** | Measuring model training time |

## 2. Data Preparation Functions

### Data Loading

```python
def load_data(file_path):
```

This function:
- Loads Tesla stock data from CSV
- Removes the first two rows (likely header information)
- Converts data columns to float
- Properly formats the date column and sets it as index

### Feature Engineering

Two key feature engineering functions:

| Function | Description |
|----------|-------------|
| `calculate_pct_change()` | Calculates daily percentage change in closing price |
| `calculate_moving_average()` | Implements a simple moving average over n days |

### Sequence Creation

```python
def create_sequences(data, seq_length):
```

This is crucial for time series modeling:
- Creates sliding window sequences of length `seq_length`
- Each sequence becomes an input (X)
- The next time step becomes the target (y)
- Returns numpy arrays ready for model training

## 3. Model Architecture

The code implements two neural network architectures:

### LSTM Model

```python
def build_lstm_model(input_shape, units=100):
```

Long Short-Term Memory networks are specialized RNNs designed to learn long-term dependencies:
- Input layer accepting the sequence data shape
- LSTM layer with specified units
- Dropout (0.2) to prevent overfitting
- Dense output layer

### RNN Model 

```python
def build_rnn_model(input_shape, units=100):
```

Simple/Vanilla Recurrent Neural Network:
- Mirrors the LSTM architecture for fair comparison
- Uses SimpleRNN layer instead of LSTM
- Same dropout rate and dense output layer

## 4. Forecasting Function

```python
def forecast_future(model, last_sequence, scaler, days_to_predict, seq_length, feature_count, start_date):
```

This implements recursive forecasting:
- Takes the last known sequence of data
- Makes a prediction for the next time step
- Incorporates that prediction into the input window
- Repeats for the specified number of days
- Returns a DataFrame with date index and forecasted values

## 5. Evaluation Metrics

```python
def calculate_metrics(y_true, y_pred):
```

Comprehensive model evaluation using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

## 6. Visualization Functions

The code includes multiple visualization functions:

| Function | Purpose |
|----------|---------|
| `plot_percentage_change()` | Visualizes daily price changes and their distribution |
| `plot_stock_with_ma()` | Shows stock price with moving average and volume |
| `plot_model_comparison()` | Compares LSTM and RNN predictions against actual prices |
| `plot_error_comparison()` | Analyzes prediction errors for both models |
| `plot_close_predictions_detail()` | Detailed view of recent predictions |
| `plot_forecast_comparison()` | Compares future forecasts from both models |
| `plot_training_history()` | Visualizes training and validation loss curves |
| `plot_volatility_analysis()` | Examines price volatility over time |
| `plot_candlestick_chart()` | Traditional financial chart showing OHLC data |
| `compare_model_metrics()` | Side-by-side comparison of performance metrics |

## 7. Main Execution Flow

The `main()` function orchestrates the entire process:

1. **Configuration**:
   - Sets parameters like sequence length, units, batch size, epochs
   - Defines forecast horizon (30 days)
   - Sets train/test/validation split ratios

2. **Data Processing**: 
   - Loads and explores the dataset
   - Visualizes initial data characteristics
   - Scales data using MinMaxScaler
   - Creates input sequences

3. **Model Training**:
   - Builds LSTM and RNN models
   - Trains both models with validation
   - Measures and compares training times
   - Visualizes training history

4. **Evaluation**:
   - Makes predictions on test data
   - Calculates performance metrics
   - Visualizes prediction accuracy

5. **Forecasting**:
   - Generates 30-day forecasts using both models
   - Visualizes and compares forecasts
   - Calculates percentage changes from last known price

6. **Summary**:
   - Comprehensive comparison of model performance
   - Determines which model performed better by different metrics

## 8. Key Insights and Improvements

- The model uses a sequence length of 120 days to predict the next day
- The final layer outputs all features (open, high, low, close, volume)
- The forecasting method recursively predicts multiple steps ahead
- GPU acceleration is used when available
- Both models are compared in terms of:
  - Training time
  - Prediction accuracy
  - Validation loss
  - Forecast divergence

## 9. Potential Enhancements

The code could be improved by:
- Adding feature engineering for technical indicators (RSI, MACD, etc.)
- Implementing attention mechanisms to improve LSTM performance
- Adding transformer-based models for comparison
- Incorporating external factors (market indices, news sentiment)
- Implementing ensemble methods to combine predictions
- Adding hyperparameter tuning
- Implementing early stopping to prevent overfitting

## 10. Usage Instructions

To run this analysis:
1. Ensure you have the required dependencies installed
2. Update the file path to point to your Tesla stock data CSV
3. Modify the configuration parameters if needed
4. Run the main function to execute the complete analysis
