## [matplotlib](https://matplotlib.org)

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
      
      
      
