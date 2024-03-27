from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt 
import numpy as np 
from matplotlib.lines import Line2D

def learning_curve(model, x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    train_errors = []
    val_errors = []

    for m_instance in range(1, len(x_train)):
        
        x_train_data = x_train[:m_instance]
        y_train_data = y_train[:m_instance]

        model.fit(x_train_data, y_train_data)

        y_train_predict = model.predict(x_train_data)

        y_val_predict = model.predict(x_val)

        err_train_set = mean_squared_error(y_train_data, y_train_predict)

        err_validation_set = mean_squared_error(y_val, y_val_predict)

        train_errors.append(err_train_set)
        
        val_errors.append(err_validation_set)
    
    # sqrt MSE
    train_errors = np.sqrt(train_errors)
    val_errors =  np.sqrt(val_errors)

    plt.figure()
    plt.plot(train_errors, c='orange')
    plt.plot(val_errors, c='blue')
    plt.ylabel("RSME")
    plt.ylim(0,5)
    plt.xlabel("Training set size")
    plt.legend( [ Line2D([0], [0],  color='orange', linestyle='-'), Line2D([0], [0], color="blue", linewidth=1, linestyle='-')] , ["train", "val"])

    plt.show()