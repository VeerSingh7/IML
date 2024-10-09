import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv(r"C:\Users\Veer\Downloads\perceptron_assignment.csv")  
def perceptron(data, alpha):
    """perceptron algorithm
    Args:
      data: pandas DataFrame with 'x', 'y', and 'result' columns
      alpha: learning rate

    Returns:
      t: Number of iterations until convergence
      theta: final weight vector
    """
    #
    theta = np.array([1, 1]) #Initializing theta  

    t = 0 #Set learning rate
    while True:
        theta_prev = theta.copy()  # Storing previous theta
        for i in range(len(data)):
            x = np.array([data['x'][i], data['y'][i]])
            result = data['result'][i]
            # Perceptron update rule
            if result == 1 and np.dot(theta, x) < 0:
                theta = theta + alpha * x
            elif result == 0 and np.dot(theta, x) >= 0:
                theta = theta - alpha * x

        t += 1
        if np.array_equal(theta, theta_prev):  # Check for convergence
            break


    return t, theta #return final t, theta
#main----
# perceptron algorithm with alpha = 0.5
alpha=0.5
iterations, final_theta = perceptron(data,alpha)
print(f"Iterations to converge (alpha=0.5): {iterations}")
print(f"Final theta (alpha=0.5): {final_theta}")
# Plotting the data 
plt.figure(figsize=(8, 6))
plt.scatter(data['x'][data['result'] == 0], data['y'][data['result'] == 0], color='green', label='Result 0')
plt.scatter(data['x'][data['result'] == 1], data['y'][data['result'] == 1], color='blue', label='Result 1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data Visualization')
plt.legend()
plt.show()
