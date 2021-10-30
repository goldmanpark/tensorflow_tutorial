from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

# create dataset
x_data = [degree for degree in range(1000)]
y_data = np.sin(np.array(x_data) * np.pi / 180.)
noise = np.array([np.random.choice([-1, 1]) * np.random.random_sample() / 10. for _ in range(1000)])
y_noised = y_data + noise

SPLIT_INDEX = 800
x_train = x_data[:SPLIT_INDEX]
y_train = y_noised[:SPLIT_INDEX]
x_test = x_data[SPLIT_INDEX:]
y_test = y_data[SPLIT_INDEX:]

def compare(forecast):
    print(keras.metrics.mean_squared_error(y_test, forecast).numpy())
    print(keras.metrics.mean_absolute_error(y_test, forecast).numpy())
    plot_series(x_test, y_test)
    plot_series(x_test, forecast)
    plt.show()

# 1. naive forecast
naive_forecast = y_noised[SPLIT_INDEX - 1:-1]
#compare(naive_forecast)

# 2. moving average forecast
def moving_average_forecast(series, window_size):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
        
    return np.array(forecast)[SPLIT_INDEX - window_size:]

moving_avg = moving_average_forecast(y_noised, 30)
#compare(moving_avg)