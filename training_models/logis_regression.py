import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def prob_vs_petal_width(boundary_x, boundary_y, x, y_proba_is_iris_virginica, y_proba_not_iris_virginica):
    plt.figure()
    plt.ylabel("Probability")
    plt.xlabel("Petal Width (cm)")
    plt.plot(x, y_proba_is_iris_virginica, label="iris virginica")
    plt.plot(x, y_proba_not_iris_virginica, label="not iris virginica")
    plt.vlines( ls="--", x=[boundary_x] , ymin=[0], ymax=[1], label="decision_boundary"  )
    plt.legend(loc="center left")
    plt.xlim((0, 10))
    bbox = dict(boxstyle="round", fc="0.8")

    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=90,rad=50")

    plt.annotate(
        f'({boundary_x:.1f}, {boundary_y:.1f})',
        xy=(boundary_x, boundary_y),
        xytext=(30, 30),
        textcoords="offset points",
        c="red",
        bbox= bbox,
        arrowprops=arrowprops, 
        size = 7,
    )

    plt.show() 


def decision_boundary_width_length_search(start_w, start_l,  model):

    q = []
    k = 0.001
    # x offset position 
    start_w = start_w + 1
    
    # init directions 
    q.append({'dir': 'up', 'pos': np.c_[ [start_w ], [start_l + k ]]})
    q.append({'dir': 'down', 'pos': np.c_[ [start_w ], [start_l - k ]]})

    while q:

        record = q.pop(0)

        curr_pos = record["pos"]
        curr_dir = record['dir']

        width_pos, length_pos = curr_pos[0]

        prediction_width , prediction_length = model.predict_proba(curr_pos)[0]

        if (prediction_width > 0.50 and prediction_width < 0.52) : # and (prediction_length > 0.48 and prediction_length < 0.49):
            return curr_pos[0]
        
        if curr_dir == 'up':
            q.append({'dir': 'up', 'pos': np.c_[ [width_pos ], [length_pos + k ]]})

        if curr_dir == 'down':
            q.append({ 'dir': 'down', 'pos': np.c_[ [width_pos ], [length_pos - k ]]})
        

rng = np.random.default_rng()

iris = datasets.load_iris()

print(iris.keys()) # ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']

n_len = 1000
features = iris["feature_names"]

x = iris["data"] # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x_train_petal_width = iris["data"][:,[3]] # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x_train_petal_length  = iris["data"][:, [2]]

y = iris["target"]
y_train_iris_virginica = (iris['target'] == 2).astype(np.int8)

log_reg = LogisticRegression()
log_reg.fit(x_train_petal_width, y_train_iris_virginica)
x_new = np.linspace(0, 3, n_len).reshape(-1, 1) # reshape large row vector to column vector

# PETAL WIDTH

y_proba_width = log_reg.predict_proba(x_new) # probability_is_iris_virginica  probability_not_iris_virginica
y_proba_is_iris_virginica = y_proba_width[:, [0]]
y_proba_not_iris_virginica = y_proba_width[:, [1]]

# find intersection (lowest delta error between both curves)
y_proba_delta = np.abs(np.diff(y_proba_width) )
decision_boundary_x_index = np.argmin( y_proba_delta)
decision_boundary_x_width = x_new[decision_boundary_x_index][0]
decision_boundary_y_width = y_proba_width[decision_boundary_x_index][0]

# plot 
# prob_vs_petal_width(decision_boundary_x_width, decision_boundary_y_width, x_new, y_proba_is_iris_virginica, y_proba_not_iris_virginica)


# PETAL LENGTH
x_new = np.linspace(0, 10, n_len).reshape(-1, 1) # reshape large row vector to column vector
log_reg.fit(x_train_petal_length, y_train_iris_virginica)
y_proba_length = log_reg.predict_proba(x_new) # probability_is_iris_virginica  probability_not_iris_virginica
y_proba_is_iris_virginica = y_proba_length[:, [0]]
y_proba_not_iris_virginica = y_proba_length[:, [1]]

# find intersection (lowest delta error between both curves)
y_proba_delta = np.abs(np.diff(y_proba_length) )
decision_boundary_x_index = np.argmin( y_proba_delta)
decision_boundary_x_length = x_new[decision_boundary_x_index][0]
decision_boundary_y_length = y_proba_length[decision_boundary_x_index][0]

# plot 
# prob_vs_petal_width(decision_boundary_x_length, decision_boundary_y_length, x_new, y_proba_is_iris_virginica, y_proba_not_iris_virginica)


# PETAL LENGTH + PETAL WIDTH

x2_new = np.c_[x_new, 10 * rng.standard_normal(size=(n_len, 1))]
train_data = np.c_[x_train_petal_width, x_train_petal_length]
log_reg.fit(train_data, y_train_iris_virginica)
y_predict = log_reg.predict(x2_new)
y_predict_prob = log_reg.predict_proba(x2_new)
y_predict_prob_some_item = log_reg.predict_proba([ [decision_boundary_x_width, decision_boundary_x_length] ])

# search for another point on boundary line 
new_decision_boundary_x_length, new_decision_boundary_x_width= decision_boundary_width_length_search(decision_boundary_x_length,decision_boundary_x_width, log_reg )

# use new decision boundary point to calculate slope
widths = np.c_[[decision_boundary_x_width], [new_decision_boundary_x_width]]
lengths = np.c_[[decision_boundary_x_length], [new_decision_boundary_x_length]]
m = np.diff(np.vstack((widths, lengths)))[0][0]
b = new_decision_boundary_x_width - m * new_decision_boundary_x_length  # y = mx + b solve for b 
decision_boundary_line = m * x_new + b

# PLOT

plt.figure()

markers = ["x","*"]
bbox = dict(boxstyle="round", fc="0.8")
arrowprops = dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=50")

plt.ylabel("Petal Width (cm)")
plt.xlabel("Petal Length (cm)")
plt.ylim((0 , 3))
plt.xlim((0 , 10))

plt.scatter(x_new, decision_boundary_line, marker=",", s=0.5, c='red')

for p_width, p_length, id in np.c_[x2_new, y_predict]:
    id = int(id)
    plt.scatter( x=p_length, y=p_width, marker=markers[id])

plt.annotate(
    f'Decision Boundary: \n({decision_boundary_x_length:.1f}, {decision_boundary_x_width:.1f})',
    xy=(decision_boundary_x_length, decision_boundary_x_width),
    xytext=(30, 30),
    textcoords="offset points",
    bbox= bbox,
    arrowprops=arrowprops, 
    size = 6,
)

plt.show()