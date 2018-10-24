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
    

Missing Data

    import numpy as np
    import pandas as pd
    
    d = {'A':[1,2,np.nan], 'B':[5,np.nan,np.nan], 'C':[1,2,3]}
    
    df = pd.DataFrame(d)
    
    df 
    
    df.dropna(axis=1)
    
    df.dropna()
    
    df.dropna(thresh=2)
    
    df.fillna(value='FILL VALUE')
    
    df['A'].fillna(value=df['A'].mean())
    

GroupBy with Pandas

    import numpy as np
    import pandas as pd
    
    data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
            'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
            'Sales':[200,120,340,124,243,350]}

    df = pd.DataFrame(data)
    
    df
    
    byComp = df.groupby('Company')
    
    byComp.mean()
    
    byComp.sum()
    
    byComp.std()
    
    byComp.sum().loc['FB']
    
    df.groupby('Company').count()
    
    df.groupby('Company').max()
    
    # 모든정보가 다 나오는 방법
    df.groupby('Company').describe()
    
    df.groupby('Company').describe().transpose()['FB']
    

Merging, Joining & Concatenating

    import pandas as pd
    
    df1 = pd.DataFrame({'A':['A0','A1','A2','A3'],
                        'B':['B0','B1','B2','B3'],
                        'C':['C0','C1','C2','C3'],
                        'D':['D0','D1','D2','D3']},
                        index=[0,1,2,3])

    df2 = pd.DataFrame({'A':['A4','A5','A6','A7'],
                        'B':['B4','B5','B6','B7'],
                        'C':['C4','C5','C6','C7'],
                        'D':['D4','D5','D6','D7']},
                        index=[4,5,6,7])

    df3 = pd.DataFrame({'A':['A8','A9','A10','A11'],
                        'B':['B8','B9','B10','B11'],
                        'C':['C8','C9','C10','C11'],
                        'D':['D8','D9','D10','D11']},
                        index=[8,9,10,11])
    
    df1 
    
    df2
    
    df3
    
    # Concatenation basically flues together DataFrames. Keep in mind that dimensions should match along the axis you are concatenating on. You can use pd.concat and pass in a list of DataFrames to concatenate together:
    
    pd.concat([df1, df2, df3])
    
    pd.concat([df1, df2, df3], axis=1)
    
    left = pd.DataFrame({'key':['K0','K1','K2','K3'],
                         'A':['A0','A1','A2','A3'],
                         'B':['B0','B1','B2','B3']})

    right = pd.DataFrame({'key':['K0','K1','K2','K3'],
                          'C':['C0','C1','C2','C3'],
                          'D':['D0','D1','D2','D3']})

    left
    
    right
    
    
    #Merging: the merge function allws yo to merge DataFrames together using a similar logic as merging SQL Tables together.For example:
    
    pd.merge(left, right, how='inner', on='key')
    
    left = pd.DataFrame({'key1':['K0','K0','K1','K2'],
                         'key2':['K0','K1','K0','K1'],
                         'A':['A0','A1','A2','A3'],
                         'B':['B0','B1','B2','B3']})

    right = pd.DataFrame({'key1':['K0','K1','K1','K2'],
                          'key2':['K0','K0','K0','K0'],
                          'C':['C0','C1','C2','C3'],
                          'D':['D0','D1','D2','D3']})
                          
    # Joining is a convenient method for combining the columns of two potentially differently-indexed DataFrames into a single result DataFrame
    
    left = pd.DataFrame({'A':['A0','A1','A2'],
                         'B':['B0','B1','B2']},
                         index = ['K0','K1','K2'])

    right = pd.DataFrame({'C':['C0','C1','C2'],
                          'D':['D0','D1','D2']},
                          index=['K0','K2','K3'])

    left.join(right)
    
    left.join(right, how='outer')
    
    
    
Pandas Operations
    
    import numpy as np
    import pandas as pd
    
    df = pd.DataFrame({'col1':[1,2,3,4],
                       'col2':[444,555,666,444],
                       'col3':['abc','def','ghi','xyz']})

    df.head()
    
    df['col2'].unique()
    #output: array([444, 555, 666], dtype=int64)
    
    len(df['col2'].unique())
    
    df['col2'].value_counts()
    
    df[df['col1']>2]
    
    df[(df['col1']>2) & (df[df['col2']==444])]
    
    def times2(x):
        return x*2

    df['col1'].apply(times2)
    
    df['col3'].apply(len)
    
    df['col2'].apply(lambda x: x*2)
    
    #df.drop('col1',axis=1, inplace=True)
    
    df.columns
    # output: Index(['col1','col2','col3'], dtype='object')
    
    df.index
    # output: RangeIndex(start=0, stop=4, step=1)
    
    df.sort_values('col2')
    
    df.isnull()
    
    data = {'A':['foo','foo','foo','bar','bar','bar'],
            'B':['one','one','two','two','one','one'],
            'C':['x','y','x','y','x','y],
            'D':[1,3,2,5,4,1]}

    df = pd.DataFrame(data)
    
    df.pivot_table(values='D',index=['A','B'],columns=['C'])
    
Data input and output

    import pandas as pd
    
    pd.read_csv('example.csv')
    
    df = pd.read_csv('example.csv')
    
    df.to_csv('My_output', index=False)
    
    pd.read_csv('My_output')
    
    #conda install xlrd
    
    pd.read_excel('Excel_Sample.xlsx', sheetname='Sheet1')
    
    df.to_excel('Excel_Sample2.xlsx',sheet_name='NewSheet')
    
    data = pd.read_html('https://www.fdic.gov/bank/individual/failed/banklist.html')
    
    type(data)
    # Output: list
    
    data[0].head()
    
    # Sqlite
    from sqlalchemy import create_engine
    
    engine = create_engine('sqlite:///:memory:')
    
    df.to_sql('my_table',engine)
    
    sqldf = pd.read_sql('my_table', con=engine)
    
    sqldf
    
    
### visualization

    import numpy as np
    import seaborn as sns
    import pandas as pd
    
    # matplotlib inline
    
    df1 = pd.read_csv('df1', index_col=0)
    df1.head()
    
    df2 = pd.read_csv('df2')
    df2.head()
    
    df1['A'].hist(bins=30)
    
    df1['A'].plot(kind='hist', bins=30)
    
    df1['A'].plot.hist()
    
    df2.plot.area(alpha=0.4)
    
    df2.plot.bar()
    
    df2.plt.bar(stacked=True)
    
    
    df1['A].plot.hist(bins=50)
    
    
    df1.plot.line(x=df1.index, y='B', figsize=(12,3), lw=1)
    
    
    df1.plot.scatter(x='A',y='B', c='C', cmap='coolwarm')
    
    df1.plot.scatter(x='A', y='B', s=df1['C']*100)
    
    df2.plot.box()
    
    
    df = pd.DataFrame(np.random.randn(1000,2), columns=['a','b'])
    
    df.plot.hexbin(x='a',y='b' , gridsize=25, cmap='coolwarm')
    
    
    
    df2['a'].plot.kde()
    
    df2['a'].plot.densitiy()
    

### Time Series Visualization

    import pandas as pd
    import matplotlib.pyplot as plt
    # %matplotlib inline
    # %matplotlib notebook
    
    mcdon = pd.read_csv('mcdonalds.csv', index_col = 'Data', parse_date=True)
    mcdon.head()
    
    
    mcdon.plot()
    
    mcdon['Adj. Close'].plot()
    
    mcdon['Adj. Volume'].plot(figsize=(12,4))
    
    
    mcdon['Adj. Close'].plot(xlim=['2007-01-01', '2009-01-01'], ylim=(20,50))
    
    mcdon['Adj. Close'].plot(xlim=['2007-01-01', '2009-01-01'], ylim=(20,50), ls='--',c='red')
    
    
    import matplotlib.dates as dates
    mcdon['Adj. Close'].plot(xlim=['2007-01-01', '2009-01-01'], ylim=(0,50))
    
    
    idx = mcdon.loc['2007-01-01':'2007-05-01'].index
    
    stock = mcdon.loc['2007-01-01':'2007-05-01']['Adj. Close']
    
    
    fig, ax = plt.subplots()
    ax.plot_date(idx, stock, '-')
    
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    
    #Locating
    #ax.xaxis.set_major_locator(dates.MonthLocator())
    #Formating
    #ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n%b-%Y'))
    
    #ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=0))
    #ax.xaxis.set_minor_formatter(dates.DateFormatter('%a'))
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    
## Datetime Index

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    # %matplotlib inline
    
    from datetime import datetime
    
    my_year = 2017
    my_month = 1
    my_day = 2
    my_hour = 13
    my_minute =30
    my_second = 15
    
    my_date = datetime(my_year, my_month, my_day)
    
    my_date_time = datetime(my_year, my_month, my_day, my_hour, my_minute, my_second)
    type(my_date_time)
    
    my_date_time.day
    
    my_date_time.month
    
    
    first_two = [datetime(2016,1,1), datetime(2016, 1, 2)]
    type(first_two)
    
    dt_ind = pd.DatetimeIndex(first_two)
    
    
    data = np.random.randn(2,2)
    
    cols = ['a','b']
    
    
    df = pd.DataFrame(data, dt_ind, cols)
    
    df.index.argmaax()
    
    df.index.max()
    
    df.index.argmin()
    
    df.index.min()
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
