#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    # raise NotImplementedError

    #### FORWARD  #######
    h1 = np.dot(X, W1) + b1
    h_1_act = sigmoid(h1)
    h2 = np.dot(h_1_act, W2) + b2
    out_layer = softmax(h2)

    cost = - np.sum(labels * np.log(out_layer)) # cross-entropy
    cos2 = - np.sum(np.log(out_layer[np.arange(X.shape[0]), np.argmax(labels, axis=1)]))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    # TODO: Here we must compute backward cross-entropy for this layer
    dcost = out_layer - labels

    gradW2 = np.zeros_like(W2)
    gradb2 = np.zeros_like(b2)
    gradW1 = np.zeros_like(W1)
    gradb1 = np.zeros_like(b1)

    gradW2 = np.dot(h_1_act.T, dcost)
    gradb2 = np.sum(dcost, axis= 0)

    dlayer2 = np.dot(dcost, W2.T)
    dlayer1 = sigmoid_grad(h_1_act)  * dlayer2

    gradW1 = np.dot(X.T, dlayer1)
    gradb1 = np.sum(dlayer1, axis = 0)


    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
