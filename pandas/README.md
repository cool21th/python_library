## Pandas

Panel-Data로 Wes Mckinney에 의해 개발됨

사용가능한 Formats : CSV, Text files, MS Excel, SQL Database, the fast HDF5 format etc.


Series: Numpy arrays 유사(except datetime index, give them named)

    import numpy as np
    import pandas as pd
    
    labels = ['a','b','c']
    my_list = [10, 20, 30]
    arr = np.array([10, 20, 30])
    d= {'a':10, 'b':20, 'c':100}
    
    pd.Series(my_list)
    
    pd.Series(my_list, index=labels)
    
    
    pd.Series(arr)
    
    pd.Series(arr, labels)
    
    
    pd.Series(d)
    
    pd.Series(data=labels)
    
    pd.Series([sum,print,len])
    
    
    ser1 = pd.Series([1,2,3,4], index=['USA','CHINA','FRANCE','GERMANY'])
    
    ser1
    
    ser2 = pd.Series([1,2,3,4], index=['USA','CHINA','ITALY','JAPAN'])
    
    ser2
    
    ser1['USA']
    
    ser2['JAPAN']
    
    ser1 + ser2
    
    
DataFrames
    

    import numpy as np
    import pandas as pd
    
    from numpy.random import randn
    
    np.random.seed(101)
    
    df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
    
    df
    
    
    df['W']
    
    type(df['W'])
    output: pandas.core.series.Series
    
    type(df)
    output: pandas.core.frame.DataFrame
    
    df.W
    
    df[['W','Z']]
    
    df['new'] = df['W'] + df['Y']
    
    df.drop('new', axis=1, inplace=True)
    
    df.drop('E')
    
    df.shape
    
    df.loc['A']
    
    df.loc['C']
    
    df.iloc[2]
    
    df.loc['B','Y']
    
    df.loc[['A','B'],['W','Y']]
    
    booldf = df > 0
    
    df[booldf]
    
    df[df>0]
    
    df['W'] >0
    
    df[df['W']>0]
    
    df[df['Z']<0]
    
    # two step
    resultdf = df[df['W']>0]
    
    resultdf['X']
    
    df[df['W']>0]['X']
    
    df[df['W']>0][['X','Y']]
    
    
    boolser = df['W']>0
    result = df[boolser]
    mycols = ['Y''X']
    result[mycols]
    
    # two conditions
    df[(df['W']>0) & (df['Y']>1)]
    
    df[(df['W']>0) | (df['Y']>1)]
    
    
    
    df.reset_index()
    
    newind = 'CA NY WY OR CO'.split()
    newind
    
    df['States'] = newind
    
    df
    
    df.set_index('States')
    
    
    
    ##hierachy lindex
    
    #index level
    outside = ['G1','G1','G1','G2','G2','G2']
    inside = [1,2,3,1,2,3]
    hier_index = list(zip(outside, inside))
    hier_index = pd.MultiIndex.from_tuples(hier_index)
    
    df = pd.DataFrame(randn(6,2), hier_index,['A','B'])
    
    df
    
    df.loc['G1']
    
    df.index.names = ['Groups','Num']
    
    df.loc['G2'].loc[2]['B']
    
    df.xs('G1')
    
    df.xs(1,level='Num')
    
    
    
    
    
    
    
    
    
