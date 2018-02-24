#!/usr/bin/env python3

__author__ = "MistySheep"

import numpy as np
import time
import sys

#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#
#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#
#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#

class MultiDelayCoupledRC():
    """
    Note: Can only handle 1D Input
    """
    def __init__(self, N, alpha = 1, beta = 0.1, gamma = 0.09, M = None, theta = 0.4, m_amp = 0.1):
        """
        Args:
            N:          number of virtual nodes
            alpha:      scaling param TODO
            beta:       for mackey glass equation
            gamma:      input scaling
            M:          second delay
            theta:      length of subintervals
            m_amp:      mask amplitude
        """

        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.M = M
        self.theta = theta
        self.m_amp = m_amp

        # creates a mask with either value either being m_amp or - m_amp
        self.mask = self.m_amp * (2 * (np.random.uniform(low = 0.0, \
            high =1.0, size = (1,self.N)) < 0.5) - 1)
        self.mask -= np.mean(self.mask)

        if (M is None):
            self.M = 2*self.N - 3

    #=======#=======#=======#=======#=======#=======#=======#=======#=======#

    def transform_mp(self, u):
        """
        Args:
            u:          data to be multiplexed

        Returns:
            J:          mutliplexed data
        """

        # apply mask on input data
        u =  np.reshape(u, (-1,1))
        return np.dot(u, self.mask)

    #=======#=======#=======#=======#=======#=======#=======#=======#=======#

    def mg(self, X, b = 0.4, n = 1):
        """
        Nonlinear Function, the Mackey Glass Equation

        Args:
            x:          input
            b:          scaling parameter
            n:          exponent (the higer, the more nonlinear mg becomes)

        Returns:
            output:     output of mackey glass equation
        """
        return b * (X / (1 + X**n ) )

    #=======#=======#=======#=======#=======#=======#=======#=======#=======#

    def calculate(self, J):
        """
        Calculate Reservoir Activity

        Args:
            J:          multiplexed input

        Returns:
            R:          matrix of reservoir history (rows being timesteps)
        """

        # due to the fact, that we have a delayed differential equation,
        # every calculation needs values from previous timesteps, thus we
        # need some sort of padding
        pad = 3

        # get number of cycles for this data
        cycles = J.shape[0]

        # decay terms with shape N
        b = np.exp(-self.alpha*self.theta*np.arange(1,self.N+1))

        # array for values of virtual nodes
        VN = np.zeros((self.N))
        kx = b[0]
        f0 = 0.1
        phi0 = 0.1

        # reservoir history matrix
        R = 0.1 * np.ones((pad+cycles, self.N))

        # since we introduced this padding to matrix R, we will do the same
        # to input matrix J (with empty padding), so that we can index them
        # the same way
        # furthermore, apply input scaling via gamma
        J = np.vstack((np.zeros((pad, self.N)), J)) * self.gamma

        # iterate, but leave out padding
        for i in range(pad, cycles+pad):

            # for calculating with delay values, we will need values from
            # 3 cycles steps in the past
            delay_val = R[i-3:i].flatten()[-self.M:self.N-self.M]
            # apply nonlinear mackey glass equation
            ff = self.mg(R[i-1] + delay_val + J[i], b=self.beta)
            # value of first virtual node
            VN[0] = ff[0]

            # calculate values for the rest of the virtual nodes
            for t in range(1, self.N):
                VN[t] = kx * (VN[t-1] + ff[t-1]) + ff[t]

            # calculate reservoir
            R[i] = b * (self.theta/2 * f0 + phi0) + self.theta/2 * VN

            # update
            f0 = ff[self.N-1]
            phi0 = R[i, self.N-1]

            # progress bar
            self.update_progress(int(100*(i)/(cycles+pad-1)))

        # remove padding so we are left with "pure" values
        R = R[pad:]
        #np.savetxt("new.npy", R)
        return R



    def update_progress(self, pg):
        if (pg != 100):
            sys.stdout.write('\r[*]Reservoir Progress: [{0}] {1}%'.format('#'*int(pg/2), pg))

        else:
            sys.stdout.write('\r[+]Reservoir Progress: [{0}] {1}%\n'.format('#'*int(pg/2), pg))

        sys.stdout.flush()

#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#
#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#
#=======#=======#=======#=======#=======#=======#=======#=======#=======#======#

def complete(data, N, Model = None):
    """
    First initiate a new instance of the MultiDelayCoupledRC, multiplex the
    input and finally apply reservoir

    Returns the Resrevoir Activity, which can be used for a lineary readout
    """

    if (Model is None):
        # create Multi Delay-Coupled Reservoir Class
        Model = MultiDelayCoupledRC(N)

    # multiplex
    J = Model.transform_mp(data)
    # calculate reservoir activity, measure time
    X = Model.calculate(J)

    # return reservoir activity matrix
    return X, Model
