from functools import lru_cache
import multiprocessing
from multiprocessing import Process, Manager

def sharpe_ratio(rets):
    return rets.mean() / rets.std()

import numpy as np

def positions(x, theta):
    M = len(theta) - 2
    T = len(x)
    Ft = np.zeros(T)
    for t in range(M, T):
        xt = np.concatenate([[1], x[t - M:t], [Ft[t - 1]]])
        Ft[t] = np.tanh(np.dot(theta, xt))
    return Ft

def returns(Ft, x, delta):
    T = len(x)
    rets = Ft[0:T - 1] * x[1:T] - delta * np.abs(Ft[1:T] - Ft[0:T - 1])
    return np.concatenate([[0], rets])

def gradient(x, theta, delta, grads, sharpes):
    Ft = positions(x, theta)
    rets = returns(Ft, x, delta)
    T = len(x)
    M = len(theta) - 2

    A = np.mean(rets)
    B = np.mean(np.square(rets))
    S = A / np.sqrt(B - A ** 2)

    grad = np.zeros(M + 2)  # initialize gradient
    dFpdtheta = np.zeros(M + 2)  # for storing previous dFdtheta

    for t in range(M, T):
        xt = np.concatenate([[1], x[t - M:t], [Ft[t-1]]])
        dRdF = -delta * np.sign(Ft[t] - Ft[t-1])
        dRdFp = x[t] + delta * np.sign(Ft[t] - Ft[t-1])
        dFdtheta = (1 - Ft[t] ** 2) * (xt + theta[-1] * dFpdtheta)
        dSdtheta = (dRdF * dFdtheta + dRdFp * dFpdtheta)
        grad = grad + dSdtheta
        dFpdtheta = dFdtheta

    grads.append(grad)
    sharpes.append(S)
    #return grad, S

def train(x, epochs=1000, M=5, commission=0.0025, learning_rate = 0.1):
    theta = np.ones(M + 2)
    sharpess = np.zeros(epochs) # store sharpes over time
    cpuc = multiprocessing.cpu_count()
    xlist = [x[i*(len(x)//cpuc): (i+1)*(len(x)//cpuc)] for i in range(cpuc)]

    for i in range(epochs):
        manager = Manager()
        grads = manager.list()
        sharpes = manager.list()

        jobs = []
        for xi in xlist:
            jobs.append(Process(target=gradient, args=(x, theta, commission, grads, sharpes)))

        for j in jobs: j.start()
        for j in jobs: j.join()

        #grad, sharpe = gradient(x, theta, commission)
        grad = sum(grads)/len(grads)
        sharpe = sum(sharpes)/len(sharpes)
        theta = theta + grad * learning_rate

        sharpess[i] = sharpe
        print("epoch {} done".format(i))

    print("finished training")
    return theta, sharpess


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (5, 3) # (w, h)
plt.rcParams["figure.dpi"] = 200
import pandas as pd

btc = pd.read_csv("data/ETH.csv")
rets = btc['Close'].diff()[1:]

x = np.array(rets)
x = (x - np.mean(x)) / np.std(x) # normalize
print(len(x))
N = 50000
P = 10000
x_train = x[-(N+P):-P]
x_test = x[-P:]


theta, sharpes = train(x_train, epochs=1000, M=15, commission=0.0025, learning_rate=.001)
'''
plt.plot(sharpes)
plt.xlabel('Epoch Number')
plt.ylabel('Sharpe Ratio');
plt.savefig("history_perf.png")
plt.cla()
plt.clf()
'''
train_returns = returns(positions(x_train, theta), x_train, 0.0025)
plt.plot((train_returns).cumsum(), label="Reinforcement Learning Model")
plt.plot(x_train.cumsum(), label="Buy and Hold")

plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns');
plt.legend()
plt.title("RL Model vs. Buy and Hold - Training Data");
plt.savefig("training_data.png")

plt.cla()
plt.clf()

test_returns = returns(positions(x_test, theta), x_test, 0.0025)
plt.plot((test_returns).cumsum(), label="Reinforcement Learning Model")
plt.plot(x_test.cumsum(), label="Buy and Hold")
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns');
plt.legend()
plt.title("RL Model vs. Buy and Hold - Test Data");
plt.savefig("testing_data.png")

plt.cla()
plt.clf()
