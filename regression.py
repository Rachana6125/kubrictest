import requests
import pandas
import scipy
import numpy
import sys
from scipy import stats
import pandas as pd


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    # response = requests.get('TRAIN_DATA_URL')
    # YOUR IMPLEMENTATION HERE
    ...
    data = pd.read_csv('data/linreg_train.csv', header=None);
    x = data.values[0,1:].astype(float)
    y = data.values[1,1:].astype(float)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y);
    print('r_value = ', r_value);
    print('slope = ', slope);
    print('intercept = ', intercept);
    print('p_value = ', p_value);
    print('std_err = ', std_err);



    return slope*area + intercept



if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
