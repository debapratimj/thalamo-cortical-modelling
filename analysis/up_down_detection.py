import numpy as np
import matplotlib.pyplot as plt

def detect_state(signal, threshold):
    states = []
    for value in signal:
        if value > threshold:
            states.append("up")
        else:
            states.append("down")
    return states


print("Detecting states of the signal...")

threshold = 0.5
length = 1000


#  create some fake oscillating data
data  =  np.sin(np.linspace(0, 10, length)) + np.random.normal(0, 0.1, length)  


# Plot the data

plt.plot(data)
plt.show()


states = detect_state(data, threshold)
print(states[:10])

