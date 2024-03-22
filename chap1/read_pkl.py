import pickle
import os 

filename = "random_forest.pkl"
file_output = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename ) 

with open (file_output, 'rb') as f:

    table = pickle.load(f)

    attributes = table["attributes"]
    hyper_param = table["hyper_param"]
    predictions = table["predictions"]
    mse = table["mse"]
    model = table["model"]

    
    print("mse=\n\t{}\n\nbest_param=\n\t{}\n\nfeature_important\n\t{}".format(mse, hyper_param.best_params_, attributes))

