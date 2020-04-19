import matplotlib.pyplot as plt
from IPython.display import Markdown, display

# 
def plot_images(img1, img2 = None, title1="", title2=""):
    fig = plt.figure(figsize=[30,30])
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)

    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    ax2.set(xticks=[], yticks=[], title=title2)
    
def printmd(string):
    display(Markdown(string))