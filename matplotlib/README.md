## [matplotlib](https://matplotlib.org)

[github](https://github,com/matplotlib/matplotlib), [matplot gallery](https://matplotlib.org/gallery.html), [tutorial](http://www.loria.fr/~rougier/teaching/matplotlib)


Matplotlib is a Python 2D plotting library which produces publication quality figures 
in a variety of hardcopy formats(png, jpg, etc..) 
and display plots in environments across platforms


      import matplotlib.pyplot as plt
      # to see the plots inside the Jupiter notebook
      # %matplotlib inline
      
      import numpy as np
      x = np.linspace(0,5, 11)
      y = x ** 2
      
      # Functional
      plt.plot(x, y)
      plt.show()
      
      # plt.plot(x, y, 'r-')
      plt.xlabel('X Label')
      plt.ylabel('Y Label')
      plt.title('Title')
      
      
      plt.subplot(1,2,1)
      plt.plot(x, y, 'r')
      plt.subplot(1,2,2)
      plt.plot(y,x,'b')
      
      # Object Oriented Method
      fig = plt.figure()
      axes = fig.add_axes([0.1, 0.1,0.8, 0.8])
      axes.plot(x,y)
      axes.set_xlabel('X Label')
      axes.set_ylabel('Y Label')
      axes.set_title('Set Title')
      
      fig = plt.figure()
      axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
      axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
      axes1.plot(x,y)
      axes1.set_title('LARGER_PLOT')
      axes2.plot(y,x)
      axes2.set_title('SMALLER_PLOT')
      
      fig = plt.figure()
      axes1 = fig.add_axes([0.1,0.1,0.8, 0.8])
      axes1.plot(x,y)
      
      
      fig, axes = plt.subplots(nrows=1, ncols=2) 
      axes[0].plot(x,y)
      axes[0].set_title('First plot')
      axes[1].plot(y,x)
      axes[1].set_title('Second plot')
      plt.tight_layout()
      



Figure size and DPI
      
      fig = plt.figure(figsize=(3,2))
      ax = fig.add_axes([0,0,1,1])
      ax.plot(x,y)
      
      
      fig, axes = plt.subplots(nrows=2, ncols=1,figszie=(8,2))
      axes[0].plot(x,y)
      axes[1].plot(y,x)
      plt.tight_layout()
      
      fig.savefig('my_picture.png', dpi200)
      
      fig =plt.figure()
      ax = fig.add_axes([0,0,1,1])
      ax.plot(x,y)
      ax.set_title('Title')
      ax.set_ylabel('Y')
      ax.set_xlabel('X')
      
      
      fig = plt.figure()
      ax = plt.add_axes([0,0,1,1])
      ax.plot(x, x**2, label='X squared')
      ax,.plot(x,x**3, label='X Cubed')
      
      ax.legend(loc=(0.1,0.1))
      

Plot Appearance

      fig = plt.figure()
      ax = fig.add_axes([0,0,1,1])
      ax.plot(x, y, color='orange', linewidth=3, alpha=0.5)
      # ax.plot(x, y, color='purple', lw=3, linestyle='step') # ls = linstyle
      # ax.plot(x, y, color='#FF8C00', lw=3, ls='-', marker='1', markersize=10)
      # ax.plot(x, y, colort='purple' lw=1, ls='-', marker='o', markersize=20, markerfacecolor='yellow', markeredgewidth=3, markeredgecolor='green')
      
      ax.plot(x, y, color='purple', lw=2, ls='--')
      ax.set_xlim([0,1])
      ax.set_ylim([0,2])
      

Line and marker style

To change the line width, we can use the linewidth or lw keyword argument. The line style can be selected using the linestyle or ls keyword arguments

      fig, ax = plt.subplots(figsize=(12, 6))
      ax.plot(x, x+1, color="red", linewidth=0.25)
      ax.plot(x, x+2, color="red", linewidth=0.50)
      ax.plot(x, x+3, color="red", linewidth=1.00)
      ax.plot(x, x+4, color="red", linewidth=2.00)
      
      #possible linesytpe options '-','-','-.',':','stpes'
      ax.plot(x, x+5, color="green", lw=3, linstyle='-')
      ax.plot(x, x+6, color="green", lw=3, ls='-.')
      ax.plot(x, x+7, color="green", lw=3, ls=':')
      
      # custom dash
      line, = ax.plot(x,x+8, color="black", lw=1.50)
      line.set_dashes([5, 10, 15,10]) # format: line length, space length,...
      
      # piossible marker symbols: marker='+', 'o', '*', 's', ',', '.', '1', '2', '3',
      ax.plot(x, x+9, color="blue", lw=3, ls='-', marker='+')
      ax.plot(x, x+10, color="blue", lw=3, ls='--', marker='o')
      ax.plot(x, x+11, color="blue", lw=3, ls='-', marker='s')
      ax.plot(x, x+12, color="blue", lw=3, ls='--', marker='1')

      # marker size and color
      ax.plot(x, x+13,  color="purple", lw=1, ls='-',marker='o', markersize=2)
      ax.plot(x, x+14,  color="purple", lw=1, ls='-',marker='o', markersize=4)
      ax.plot(x, x+15,  color="purple", lw=1, ls='-',marker='o', markersize=8, markerfacecolor='red')
      ax.plot(x, x+16,  color="purple", lw=1, ls='-',marker='s', markersize=8, 
              markerfacecolor="yellow', markeredgewidth=3, markeredgecolor="green")
      
      
      
Plot range

We can configure the ranges of the axes using the set_ylim and set_xlim methods in the axis object, or axis('tight') 
for automatically getting "tightly fitted" axes ranges:

      fig, axes = plt.subplots(1, 3, figsize=(12, 4))
      
      axes[0].plot(x, x**2, x, x**3)
      axes[0].set_title("default axes ranges")
      
      axes[1].plot(x, x**2, x, x**3)
      axes[1].axis('tight')
      axes[1].set_title("tight axes")
      
      axes[2].plot(x, x**2, x, x**3)
      axes[2].set_ylim([0,60])
      axes[2].set_xlim([2,5])
      axes[2].set_title("custom axes range");
      

Special Plot Types

There are many specialized plots we can create, such as barplots, histograms, scatter plots, and much more. 
Most of these type of plots we will actually create using pandas. But here are a feq axamples of these type of plots

      plt.scatter(x, y)
      
      from random import sample
      data = sample(range(1, 1000), 100)
      plt.hist(data)
      
      data = [np.random.normal(0, std, 100) for std in range(1,4)]
      
      # rectangular box plot
      plt.boxplot(data, vert=True, patch_artist=True)
      
      
