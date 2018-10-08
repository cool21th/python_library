# Keras Library 

### Keras backends

Keras는 Tensorflow에서 제공하는 High-level API로 low-level연산(tensor 곱, convolution 등)을 제공하지 않습니다.
해당 연산은 Backend Engine을 통해 제공합니다.
    
    Tensorflow : Open-source symbolic tnesor manipulation framwork developed by Google
    Thenao : Open-source symbolic tensor manipulation framwork developed by LISA Lab at University of Montreal
    CNTK: Open-source toolkit for deep learning developed by MS
    

Theano(th)와 Tensorflow(tf)가 서로 호환 하는 하려면 abstract Keras backend API를 사용하면 되는데, 방법은 아래와 같습니다

tf.placeholder(), th.tensor.matrix(), th.tensor.tensor3(), 등등 동일하게 사용할 수 있습니다.

    from keras import backend as K
    
    사용 예 1)
      inputs = K.placeholder(shape=(2,4,5))
      inputs = K.placeholder(shape=(None, 4, 5))
      inputs = K.placeholder(ndim=3)
      
    사용 예 2)
      import numpy as np
      val = np.random.random((3,4,5))
      var = K.variable(value=val)
      
      var = K.zeros(shape=(3,4,5))
      var = K.ones(shape=(3,4,5))
      
    사용 예 3)
      # initializing Tensors with Random Numbers
        ## uniform distribution
        b = K.random_uniform_variable(shape=(3,4), low=0, high=1) 
        ## Gaussian distribution
        c = K.random_normal_variable(shape=(3,4), mean=0, scale=1)
        d = K.random_normal_variable(shape=(3,4), mean=0, scale=1)
        
        # Tensor Arithmetic
        a = b + c* K.abs(d)
        c = K.dot(a,K.transpose(b))
        a = K.sum(b, axis=1)
        a = K.softmax(b)
        a = K.concatenate([b,c], axis=-1)
        
      
