import numpy as np
import matplotlib.pyplot as plt
import copy
import math

def compute_cost(x, y, w, b): 
    m = x.shape[0] 
    total_cost = 0
    for i in range (m):
        total_cost += ((np.dot(x[i],w) + b) - y[i])**2
    total_cost /= 2*m

    return total_cost

def compute_gradient(x, y, w, b): 
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        change = ((np.dot(x[i],w) + b) - y[i])
        dj_dw += change * x[i]
        dj_db += change
    dj_dw /= m
    dj_db /= m
        
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    m = len(x)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
        
    return w, b, J_history, w_history