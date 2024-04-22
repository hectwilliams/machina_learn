import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_sample_image

china = load_sample_image("china.jpg") / 1.0
flower = load_sample_image("flower.jpg")/ 1.0
images = np.array([china, flower])

b, h, w, c = images.shape 

print(f'batch_size-b, height-h, widthw, channel-c')

# create 2 filters 
filters = np.zeros(shape=(7, 7, c, 2), dtype=float)

# vertical
filters[:, 3, :, 0] = 1.0  # h->all w->3 c->all s->0

# horizontal
filters[3, :, :, 1] = 1.0  # h->3 w->all c->all s->1

outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME" )

fig, axes = plt.subplots(nrows= 2, ncols=2 )

# fig.suptitle('test title', fontsize=10)

axes[0,0].set_title("Image One: vertical feature map"  ,   fontsize=6)
axes[0,0].imshow(outputs[0, :, :, 0], cmap="gray")
axes[0,1].set_title("Image One: horizontal feature map" ,   fontsize=6)
axes[0,1].imshow(outputs[0, :, :, 1], cmap="gray")

axes[1,0].set_title("Image Two: vertical feature map" ,   fontsize=6)
axes[1,0].imshow(outputs[1, :, :, 0], cmap="gray")
axes[1,1].set_title("Image Two: horizontal feature map",   fontsize=6)
axes[1,1].imshow(outputs[1, :, :, 1], cmap="gray")

# remove ticks
axes[0,0].set_xticks([])
axes[0,0].set_yticks([])
axes[0,1].set_xticks([])
axes[0,1].set_yticks([])
axes[1,0].set_xticks([])
axes[1,0].set_yticks([])
axes[1,1].set_xticks([])
axes[1,1].set_yticks([])

plt.show()