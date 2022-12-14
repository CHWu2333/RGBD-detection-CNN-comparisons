from numpy import random
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

data = []

##############################################################################
"""
Append to the data, for the table we achieved from our validation program.
Each model has its unique table, don't get wrong!
"""
VGG = np.array([0.9375, 0.99609375, 0.99609375, 0.96875, 0.9921875, 0.99951172,
 0.99975586, 0.99804688, 0.9375, 0.99902344, 0.96875,  0.99902344,
 0.99902344,  0.99609375,  0.99975586,  0.9921875,   0.99951172,  0.99951172,
 0.9375 ,     0.99609375,  0.9375,      0.9921875,   0.984375,    0.99609375,
 0.984375,   0.99804688, 0.99975586, 0.9921875,  0.99975586, 0.99609375,
 0.984375,   0.99987793, 0.984375,   0.99804688, 0.99951172, 0.99951172,
 0.9921875,  0.96875,    0.99902344, 0.99951172])
data.append(VGG)
ResNet = np.array([])
#############################################################################
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

# Creating axes instance
bp = ax.boxplot(data, patch_artist=True,
                notch='True', vert=0)

colors = []

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#8B008B',
                linewidth=1.5,
                linestyle=":")

# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color='#8B008B',
            linewidth=2)

# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color='red',
               linewidth=3)

# # changing style of fliers
# for flier in bp['fliers']:
#     flier.set(marker ='D',
#               color ='#e7298a',
#               alpha = 0.5)

# x-axis labels
y_label = []
################################################################################################
"""
Append to y_label with the name of tables accordingly, 
for example, y_label.append("VGG")
"""



###############################################################################################
ax.set_yticklabels(y_label)

# Adding title
plt.title("Network VS Accuracy")

# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# show plot
plt.savefig('box.png')
plt.show()