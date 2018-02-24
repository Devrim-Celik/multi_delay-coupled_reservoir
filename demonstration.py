#!/usr/bin/env python3

__author__ = "MistySheep"

import os
import numpy as np
import matplotlib.pyplot as plt

from generate_NARMA10 import generate_NARMA10
from dcr import complete

#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#
#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#
#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#

def NARMA10_TEST(N=800, train_cycles=4000, test_cycles=1000, warmup_cycles=100):

    # generate NARMA10 Data and split into train and test
    X, Y = generate_NARMA10(warmup_cycles+train_cycles+test_cycles)
    X_train, Y_train = X[:train_cycles+warmup_cycles], Y[:train_cycles+warmup_cycles]
    X_test, Y_test = X[warmup_cycles+train_cycles:], Y[warmup_cycles+train_cycles:]

    # get reservoir activity for trainings data and the initialized model
    R_Train, model = complete(X_train, N)
    # remove warmup values
    R_Train = R_Train[warmup_cycles:]

    # calcualte weights, using pseudoinverse
    weights =  np.dot(np.linalg.pinv(R_Train), Y_train[warmup_cycles:])

    # get reservoir activity for test data, and reuse model, since the model
    # mask is generated randomly (and we need consistend models)
    R_Test, _ = complete(X_test, N, model)

    # calculate prediction values
    Yhat = np.dot(R_Test, weights)

    # for calculating the NRMSE, dont use the first 75 values, since the model
    # first needs to get "swinging"
    y_consider = Y_test[50:]
    yhat_consider = Yhat[50:]

    # calculate normalized root mean squared error
    NRMSE = np.sqrt(np.divide(                          \
        np.mean(np.square(y_consider-yhat_consider)),   \
        np.var(y_consider)))

    # plot
    plt.figure("Prediction Plot")
    plt.title("Prediction of NARMA10 Series with NRMSE {0:.4f}".format(NRMSE))
    plt.plot(y_consider, color="blue", linewidth=2, label="NARMA10")
    plt.plot(yhat_consider, color="red", linewidth=0.5, linestyle='-', label="Model Prediction")
    plt.xlabel("Timesteps")
    plt.legend()
    plt.savefig("./images/prediction_plot.png")
    plt.show()

    return NRMSE

#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#
#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#
#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#

if (__name__=="__main__"):

    if not os.path.exists('./images'):
        os.mkdir('./images')

    nrmse = NARMA10_TEST()
    print("[*] NRMSE = {0:.4f}!".format(nrmse))
