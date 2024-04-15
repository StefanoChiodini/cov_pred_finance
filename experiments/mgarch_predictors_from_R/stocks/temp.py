# read every row from the file with path "experiments\mgarch_predictors_from_R\stocks\mgarch_stocks_adj.csv" and find and print the highest and teh lowest value that you have found in that file

import csv

with open("experiments\mgarch_predictors_from_R\stocks\mgarch_stocks_adj.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)
    highest = float("-inf")
    lowest = float("inf")
    for row in reader:
        for value in row:
            value = float(value)
            if value > highest:
                highest = value
            if value < lowest:
                lowest = value
    
    print(f"The highest value is {highest} and the lowest value is {lowest}")

    # divide the values by 10000 to get the actual values
    highest /= 10000
    lowest /= 10000

    print(f"The highest value is {highest} and the lowest value is {lowest}")

    # The highest value is 0.010795658061972701 and the lowest value is -2.7552137504859803e-05


# now plot a logistic and a linear function in the same plot
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y_logistic = 1 / (1 + np.exp(-x))
y_linear = x

plt.plot(x, y_logistic, label='Logistic')
plt.plot(x, y_linear, label='Linear')

plt.legend()
plt.show()
