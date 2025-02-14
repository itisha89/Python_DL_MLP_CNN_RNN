# Champagne Sales Prediction using Time Series Forecasting

This repository demonstrates the process of forecasting monthly champagne sales using deep learning models. The dataset used in this project contains monthly champagne sales data and has been processed for time series forecasting.

## Requirements

The following libraries are required for running the project:

- `numpy`
- `keras`
- `matplotlib`
- `pandas`
- `statsmodels`
- `sklearn`

You can install these dependencies via `pip`:

```bash
pip install numpy keras matplotlib pandas statsmodels scikit-learn
```

## Dataset

The dataset used is titled "Perrin Freres monthly champagne sales" and consists of the following columns:

- `Month`: A string representing the year and month of the data point.
- `Perrin Freres monthly champagne sales millions ?64-?72`: The sales figure in millions for that particular month.

The data is loaded from a CSV file:

```python
file_path = "path_to_dataset.csv"
df = pd.read_csv(file_path)
```

## Data Preprocessing

### Handling Missing Values
Any missing values in the dataset are dropped:

```python
df.isnull().sum()
df = df.dropna()
```

### Time Series Plot
A plot of the champagne sales over time is generated to visualize trends and seasonality:

```python
plt.plot(df['Month'], df['Perrin Freres monthly champagne sales millions ?64-?72'], marker='o', linestyle='-')
```

### Seasonal Decomposition
The dataset is decomposed into its seasonal, trend, and residual components using `seasonal_decompose` from `statsmodels`:

```python
result = seasonal_decompose(df[column_name], model='multiplicative')
```

### Transforming to Supervised Learning Problem
The time series data is converted into a supervised learning format, where the last value of each sequence is the target (`y`), and the preceding `n_steps` values are the features (`X`):

```python
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
```

The sequence is then split and reshaped into input-output pairs for training the model.

## Model Architecture

The model architecture consists of various deep learning layers (e.g., `Dense`, `Conv1D`, `MaxPooling1D`, `RNN`) to predict champagne sales.

### Example Code:

```python
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
```

This is an example of using a Convolutional Neural Network (CNN) architecture for time series forecasting. Feel free to explore other architectures like LSTMs or RNNs.

## Model Evaluation

The model's performance is evaluated using common metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) scores:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

## Conclusion

By transforming the time series data into a supervised learning format, this approach enables the use of deep learning models for forecasting monthly champagne sales. The seasonal decomposition helps to better understand the underlying patterns in the data, improving the model’s ability to capture trends and seasonality for accurate predictions.

---

## 🌍 Explore More Projects  
For more exciting machine learning and AI projects, visit **[The iVision](https://theivision.wordpress.com/)** and stay updated with the latest innovations and research! 🚀  

---
