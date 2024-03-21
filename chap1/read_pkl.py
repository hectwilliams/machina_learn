import pickle

with open ("lin_reg.pkl", 'rb') as f:
    data = pickle.load(f)

    training_info = data["training"]
    cross_val_info = data["cross_val"]
    
    print(training_info)
    print(cross_val_info)
