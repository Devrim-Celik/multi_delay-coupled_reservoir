#!/usr/bin/env python3

__author__ = "MistySheep"

import numpy as np

#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#
#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#
#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#

def generate_NARMA10(N, delay=0):
    """
    Arguments:
        N               number of values
        delay           delay in NARMA10 (the higher, the harder, more memory
                            needed)
    Returns:
        u               random input values
        y               corresponding NARMA10 values
    """

    while True:
        # random input values
        u = 0.5 * np.random.uniform(low=0.0, high=1.0, size=(N+1000))

        # output arrays
        y_base = np.zeros(shape=(N+1000))
        y = np.zeros(shape=(N, delay+1))

        # calculate the intermediate output
        for i in range(10, N+1000):

            # NARMA10 equation
            y_base[i] = 0.3 * y_base[i-1] + 0.05 * y_base[i-1] * \
                np.sum(y_base[i-10:i]) + 1.5 * u[i-1] * u[i-10] + 0.1

        # delete the first 1000 values for u and y, since we want to take
        # values of the system, after we allowed it to warm up
        u = u[1000:]
        for curr_delay in range(0, delay+1):
            y[:, curr_delay] = y_base[1000 - curr_delay : len(y_base)-curr_delay]

        # if all values of y are finite, return them with the corresponding
        # inputs
        if np.isfinite(y).all():
            return (u, y)

        # otherwise, try again. You random numbers were "unlucky"
        else:
            print('...[*] Divergent time series. Retry...')
