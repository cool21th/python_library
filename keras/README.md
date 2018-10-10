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
        

### Preprocessing

   * Text Preprocessing
   
        *Tokenizer*
        
        tf-idf 기반으로 각각의 텍스트를 정수 시퀀스 또는 토큰 계수를 바이너리가 될수 있는 벡터로 바꿔주는 역할을 합니다.
        
            사용 예)
        
            keras.prepocessing.text.Tokenizer((num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
            
            num_words: 단어빈도에 따라 보관할 최대 단어수
            filters: 필터링되는 문자열(기본값은 모든 구두점과 탭 및 줄 바꿈 제외)
            lower: 소문자로 변환여부
            split: 단어 분리용 구분 기호
            char_level: 토큰 처리 여부     
            oov_token: word_index 추가 및 text_to_sequence 호출 중에서 out-of-vacabulary 대체 
        
