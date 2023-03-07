#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

data_set = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw03_data_set.csv" ,delimiter = ",")

x_train = data_set[1:151, 0]
x_test = data_set[151:, 0]
y_train = data_set[1:151 , 1].astype(int)
y_test = data_set[151:, 1].astype(int)

K = np.max(y_train)
N_train = x_train.shape[0]
N_test = x_test.shape[0]

print("Size of train data:", N_train)
print("Size of test data:", N_test)



minimum_value = 1.5

maximum_value = max(x_train)

data_interval = np.linspace(minimum_value, maximum_value, 361)

bin_width = 0.37

left_borders = np.arange(start = minimum_value,
                         stop = maximum_value,
                         step = bin_width)

right_borders = np.arange(start = minimum_value + bin_width,
                          stop = maximum_value + bin_width,
                          step = bin_width)


g = np.zeros(len(left_borders))


g = np.asarray([np.sum(((left_borders[b] < x_train) & (x_train <= right_borders[b])) * y_train) / np.sum((left_borders[b] < x_train) & (x_train <= right_borders[b])) for b in range(len(left_borders))])




plt.figure(figsize = (15, 5))
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.plot(x_train, y_train,"b.", markersize = 10, label="training")
plt.plot(x_test, y_test,"r.", markersize = 10, label="test")

for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [g[b], g[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [g[b], g[b + 1]], "k-")    
plt.show()


top_sum = 0

for i in range(N_test):
    for b in range(len(left_borders)):
        if (left_borders[b] < x_test[i]) and (x_test[i] <= right_borders[b]):
            top_sum += (y_test[i] - g[b])**2
            
rmse = math.sqrt(top_sum / N_test) 

print("Regressogram => RMSE is", rmse, "when h is 0.37")


g=np.asarray([np.sum((np.abs((x_train - x) / bin_width) <= 0.5) * y_train) / np.sum(np.abs((x_train - x) / bin_width) <= 0.5)for x in data_interval])
                  

plt.figure(figsize = (15, 5))
plt.plot(x_train, y_train, "b.", markersize = 10, label = "training")
plt.plot(x_test, y_test, "r.", markersize = 10, label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.plot(data_interval, g, "k-")
plt.show()






allPointsDiff = [100.0 for i in range(len(data_interval))]

allPointsDiff = np.array(allPointsDiff)
rmse = 0
for i in range(N_test):
    counter = 0
    for x in range(len(data_interval)):
        allPointsDiff[counter] = abs(x_test[i] - data_interval[x])
        counter += 1
    rmse += (y_test[i] - g[np.argmin(allPointsDiff)])**2
rmse = math.sqrt(rmse / N_test)

print("Running Mean Smoother => RMSE is", rmse, "when h is 0.37")


bin_width = 0.37
g = np.asarray([np.sum((1.0 / np.sqrt(2 * math.pi) * \
                           np.exp(-0.5 * (x - x_train)**2 / bin_width**2)) * y_train) / np.sum(1.0 / np.sqrt(2 * math.pi) * \
                           np.exp(-0.5 * (x - x_train)**2 / bin_width**2))
                    for x in data_interval])

plt.figure(figsize = (15, 5))
plt.plot(x_train, y_train, "b.", markersize = 10, label = "training")
plt.plot(x_test, y_test, "r.", markersize = 10, label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.plot(data_interval, g, "k-")
plt.show()

allPointsDiff = [100.0 for i in range(len(data_interval))]

allPointsDiff = np.array(allPointsDiff)
rmse = 0
for i in range(N_test):
    counter = 0
    for x in range(len(data_interval)):
        allPointsDiff[counter] = abs(x_test[i] - data_interval[x])
        counter += 1
    rmse += (y_test[i] - g[np.argmin(allPointsDiff)])**2
rmse = math.sqrt(rmse / N_test)

print("Kernel Smoother => RMSE is", rmse, "when h is 0.37")


