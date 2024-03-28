import threading
import numpy as np

# float_formatter = "{:.5f}".format
# np.set_printoptions(formatter={'float_kind':float_formatter})

class Decision_Boundary:
    
    def __init__(self, x_max, y_max, model, plt_):
        self.x_max = x_max
        self.y_max = y_max
        self.buffer = []
        self.large_search_lock = threading.Lock()
        self.small_search_lock = threading.Lock()    
        self.k_large = 0.1
        self.k_small = 0.01
        self.model = model 
        self.plt = plt_

    def search_large(self):
        p_length = np.linspace(start=0, stop=self.x_max, num=400)
        p_width = np.linspace(start=0, stop=self.y_max, num=400)
        prob_window_50 = [0.48, 0.52]

        for length_i in p_length:

            for width_i in p_width:

                setosa_prob, versicolor_prob, virginica_prob  = self.model.predict_proba(np.c_[[width_i], [length_i]])[0]
                
                count_50 = int(setosa_prob > prob_window_50[0] and setosa_prob < prob_window_50[1]) + int((versicolor_prob >prob_window_50[0] and versicolor_prob < prob_window_50[1] )) + int((virginica_prob > prob_window_50[0] and virginica_prob < prob_window_50[1] )) 
                
                if count_50 > 1: 
                    self.plt.scatter(length_i, width_i, marker=".", s=0.1, color="silver")
                
        
        