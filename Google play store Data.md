# Complete EDA analysis

# importing libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

### Data Loading and Exploration | Cleaning 


```python
df=pd.read_csv('googleplaystore.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Size</th>
      <th>Installs</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Photo Editor &amp; Candy Camera &amp; Grid &amp; ScrapBook</td>
      <td>ART_AND_DESIGN</td>
      <td>4.1</td>
      <td>159</td>
      <td>19M</td>
      <td>10,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>January 7, 2018</td>
      <td>1.0.0</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Coloring book moana</td>
      <td>ART_AND_DESIGN</td>
      <td>3.9</td>
      <td>967</td>
      <td>14M</td>
      <td>500,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design;Pretend Play</td>
      <td>January 15, 2018</td>
      <td>2.0.0</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U Launcher Lite – FREE Live Cool Themes, Hide ...</td>
      <td>ART_AND_DESIGN</td>
      <td>4.7</td>
      <td>87510</td>
      <td>8.7M</td>
      <td>5,000,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>August 1, 2018</td>
      <td>1.2.4</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sketch - Draw &amp; Paint</td>
      <td>ART_AND_DESIGN</td>
      <td>4.5</td>
      <td>215644</td>
      <td>25M</td>
      <td>50,000,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Teen</td>
      <td>Art &amp; Design</td>
      <td>June 8, 2018</td>
      <td>Varies with device</td>
      <td>4.2 and up</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pixel Draw - Number Art Coloring Book</td>
      <td>ART_AND_DESIGN</td>
      <td>4.3</td>
      <td>967</td>
      <td>2.8M</td>
      <td>100,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design;Creativity</td>
      <td>June 20, 2018</td>
      <td>1.1</td>
      <td>4.4 and up</td>
    </tr>
  </tbody>
</table>
</div>




```python
# set options to be maximum for rows and columns
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
```


```python
# Hide all warnings
import warnings
warnings.filterwarnings('ignore')
```


```python
df.columns
```




    Index(['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type',
           'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver',
           'Android Ver'],
          dtype='object')




```python
print("The number of rows are",df.shape[0]," and columns are",df.shape[1])
```

    The number of rows are 10841  and columns are 13
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10841 entries, 0 to 10840
    Data columns (total 13 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   App             10841 non-null  object 
     1   Category        10840 non-null  object 
     2   Rating          9367 non-null   float64
     3   Reviews         10841 non-null  int64  
     4   Size            10841 non-null  object 
     5   Installs        10841 non-null  object 
     6   Type            10840 non-null  object 
     7   Price           10841 non-null  object 
     8   Content Rating  10841 non-null  object 
     9   Genres          10840 non-null  object 
     10  Last Updated    10841 non-null  object 
     11  Current Ver     10833 non-null  object 
     12  Android Ver     10839 non-null  object 
    dtypes: float64(1), int64(1), object(11)
    memory usage: 1.1+ MB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rating</th>
      <th>Reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9367.000000</td>
      <td>1.084100e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.191513</td>
      <td>4.441119e+05</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.515735</td>
      <td>2.927629e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>3.800000e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.300000</td>
      <td>2.094000e+03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.500000</td>
      <td>5.476800e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>7.815831e+07</td>
    </tr>
  </tbody>
</table>
</div>




```python
# how to make size a numeric column?
df['Size'].unique()
```




    array(['19M', '14M', '8.7M', '25M', '2.8M', '5.6M', '29M', '33M', '3.1M',
           '28M', '12M', '20M', '21M', '37M', '2.7M', '5.5M', '17M', '39M',
           '31M', '4.2M', '7.0M', '23M', '6.0M', '6.1M', '4.6M', '9.2M',
           '5.2M', '11M', '24M', 'Varies with device', '9.4M', '15M', '10M',
           '1.2M', '26M', '8.0M', '7.9M', '56M', '57M', '35M', '54M', '201k',
           '3.6M', '5.7M', '8.6M', '2.4M', '27M', '2.5M', '16M', '3.4M',
           '8.9M', '3.9M', '2.9M', '38M', '32M', '5.4M', '18M', '1.1M',
           '2.2M', '4.5M', '9.8M', '52M', '9.0M', '6.7M', '30M', '2.6M',
           '7.1M', '3.7M', '22M', '7.4M', '6.4M', '3.2M', '8.2M', '9.9M',
           '4.9M', '9.5M', '5.0M', '5.9M', '13M', '73M', '6.8M', '3.5M',
           '4.0M', '2.3M', '7.2M', '2.1M', '42M', '7.3M', '9.1M', '55M',
           '23k', '6.5M', '1.5M', '7.5M', '51M', '41M', '48M', '8.5M', '46M',
           '8.3M', '4.3M', '4.7M', '3.3M', '40M', '7.8M', '8.8M', '6.6M',
           '5.1M', '61M', '66M', '79k', '8.4M', '118k', '44M', '695k', '1.6M',
           '6.2M', '18k', '53M', '1.4M', '3.0M', '5.8M', '3.8M', '9.6M',
           '45M', '63M', '49M', '77M', '4.4M', '4.8M', '70M', '6.9M', '9.3M',
           '10.0M', '8.1M', '36M', '84M', '97M', '2.0M', '1.9M', '1.8M',
           '5.3M', '47M', '556k', '526k', '76M', '7.6M', '59M', '9.7M', '78M',
           '72M', '43M', '7.7M', '6.3M', '334k', '34M', '93M', '65M', '79M',
           '100M', '58M', '50M', '68M', '64M', '67M', '60M', '94M', '232k',
           '99M', '624k', '95M', '8.5k', '41k', '292k', '11k', '80M', '1.7M',
           '74M', '62M', '69M', '75M', '98M', '85M', '82M', '96M', '87M',
           '71M', '86M', '91M', '81M', '92M', '83M', '88M', '704k', '862k',
           '899k', '378k', '266k', '375k', '1.3M', '975k', '980k', '4.1M',
           '89M', '696k', '544k', '525k', '920k', '779k', '853k', '720k',
           '713k', '772k', '318k', '58k', '241k', '196k', '857k', '51k',
           '953k', '865k', '251k', '930k', '540k', '313k', '746k', '203k',
           '26k', '314k', '239k', '371k', '220k', '730k', '756k', '91k',
           '293k', '17k', '74k', '14k', '317k', '78k', '924k', '902k', '818k',
           '81k', '939k', '169k', '45k', '475k', '965k', '90M', '545k', '61k',
           '283k', '655k', '714k', '93k', '872k', '121k', '322k', '1.0M',
           '976k', '172k', '238k', '549k', '206k', '954k', '444k', '717k',
           '210k', '609k', '308k', '705k', '306k', '904k', '473k', '175k',
           '350k', '383k', '454k', '421k', '70k', '812k', '442k', '842k',
           '417k', '412k', '459k', '478k', '335k', '782k', '721k', '430k',
           '429k', '192k', '200k', '460k', '728k', '496k', '816k', '414k',
           '506k', '887k', '613k', '243k', '569k', '778k', '683k', '592k',
           '319k', '186k', '840k', '647k', '191k', '373k', '437k', '598k',
           '716k', '585k', '982k', '222k', '219k', '55k', '948k', '323k',
           '691k', '511k', '951k', '963k', '25k', '554k', '351k', '27k',
           '82k', '208k', '913k', '514k', '551k', '29k', '103k', '898k',
           '743k', '116k', '153k', '209k', '353k', '499k', '173k', '597k',
           '809k', '122k', '411k', '400k', '801k', '787k', '237k', '50k',
           '643k', '986k', '97k', '516k', '837k', '780k', '961k', '269k',
           '20k', '498k', '600k', '749k', '642k', '881k', '72k', '656k',
           '601k', '221k', '228k', '108k', '940k', '176k', '33k', '663k',
           '34k', '942k', '259k', '164k', '458k', '245k', '629k', '28k',
           '288k', '775k', '785k', '636k', '916k', '994k', '309k', '485k',
           '914k', '903k', '608k', '500k', '54k', '562k', '847k', '957k',
           '688k', '811k', '270k', '48k', '329k', '523k', '921k', '874k',
           '981k', '784k', '280k', '24k', '518k', '754k', '892k', '154k',
           '860k', '364k', '387k', '626k', '161k', '879k', '39k', '970k',
           '170k', '141k', '160k', '144k', '143k', '190k', '376k', '193k',
           '246k', '73k', '658k', '992k', '253k', '420k', '404k', '470k',
           '226k', '240k', '89k', '234k', '257k', '861k', '467k', '157k',
           '44k', '676k', '67k', '552k', '885k', '1020k', '582k', '619k'],
          dtype=object)



### Observations
- veries with device
- M
- K


```python
df['Size'].isnull().sum()
```




    0



- No missing value in size 


```python
#verify the number of values| obsrtvation
df['Size'].loc[df['Size'].str.contains('M')].value_counts().sum()
```




    8830




```python
df['Size'].loc[df['Size'].str.contains('k')].value_counts().sum()
```




    316




```python
df['Size'].loc[df['Size'].str.contains('Varies with device')].value_counts().sum()
```




    1695




```python
8830+316+1695
```




    10841




```python
len(df)
```




    10841




```python
# convert the whole size column into bytes
# let's define a function
def convert_size(size):
    if isinstance(size, str):
        if 'k' in size:
            return float(size.replace('k', ""))*1024
        elif 'M' in size:
            return float(size.replace('M', ""))*1024*1024
        elif 'Varies with  device' in size:
            return np.nan
    return size
```


```python
df['Size']
```




    0                       19M
    1                       14M
    2                      8.7M
    3                       25M
    4                      2.8M
    5                      5.6M
    6                       19M
    7                       29M
    8                       33M
    9                      3.1M
    10                      28M
    11                      12M
    12                      20M
    13                      21M
    14                      37M
    15                     2.7M
    16                     5.5M
    17                      17M
    18                      39M
    19                      31M
    20                      14M
    21                      12M
    22                     4.2M
    23                     7.0M
    24                      23M
    25                     6.0M
    26                      25M
    27                     6.1M
    28                     4.6M
    29                     4.2M
    30                     9.2M
    31                     5.2M
    32                      11M
    33                      11M
    34                     4.2M
    35                     9.2M
    36                      24M
    37       Varies with device
    38                      11M
    39                     9.4M
    40                      15M
    41                      10M
    42       Varies with device
    43                     1.2M
    44                      12M
    45                      24M
    46                      26M
    47                     8.0M
    48                     7.9M
    49                      25M
    50                      56M
    51                      57M
    52       Varies with device
    53                      35M
    54                      33M
    55                      33M
    56                     5.6M
    57                      54M
    58                     201k
    59                     3.6M
    60                     5.7M
    61                      17M
    62                     8.6M
    63                     2.4M
    64                      27M
    65                     2.7M
    66                     2.5M
    67       Varies with device
    68       Varies with device
    69                     7.0M
    70                      35M
    71                      16M
    72                      16M
    73       Varies with device
    74                      17M
    75                     3.4M
    76                     8.9M
    77                     3.9M
    78                     2.9M
    79                      38M
    80                      32M
    81                      37M
    82                      15M
    83                     5.4M
    84                      18M
    85       Varies with device
    86                      38M
    87                     1.1M
    88       Varies with device
    89       Varies with device
    90                     7.9M
    91                      35M
    92       Varies with device
    93                      17M
    94                      19M
    95                      14M
    96                     2.2M
    97                     4.5M
    98                      17M
    99                      14M
    100                    9.8M
    101                     21M
    102      Varies with device
    103                     52M
    104                     14M
    105                     25M
    106                    9.0M
    107      Varies with device
    108                     12M
    109      Varies with device
    110                     35M
    111                    6.7M
    112                     30M
    113                    5.7M
    114                    2.9M
    115                     17M
    116                    2.9M
    117      Varies with device
    118      Varies with device
    119                    2.6M
    120                    4.2M
    121                    7.1M
    122                     57M
    123                    3.7M
    124                     22M
    125                     24M
    126                    7.4M
    127                     21M
    128                    3.4M
    129                    2.9M
    130                    3.1M
    131                    6.4M
    132                    3.2M
    133                    8.2M
    134                    9.9M
    135                    2.9M
    136                     23M
    137                    4.6M
    138                    3.1M
    139      Varies with device
    140                    4.9M
    141                    9.5M
    142      Varies with device
    143      Varies with device
    144      Varies with device
    145      Varies with device
    146      Varies with device
    147                    4.2M
    148                    5.4M
    149      Varies with device
    150                    2.8M
    151                    5.0M
    152      Varies with device
    153                    5.9M
    154                     13M
    155                    7.1M
    156                    6.7M
    157      Varies with device
    158                     17M
    159                     19M
    160                    6.7M
    161                     21M
    162      Varies with device
    163                    2.7M
    164                     37M
    165                     15M
    166                     23M
    167                     19M
    168                     23M
    169                     73M
    170                    4.9M
    171                    6.8M
    172      Varies with device
    173      Varies with device
    174                    2.9M
    175                    3.5M
    176                    4.0M
    177                     21M
    178                    2.3M
    179      Varies with device
    180                    7.2M
    181                     10M
    182                    6.1M
    183                    2.1M
    184                     42M
    185                    7.3M
    186                     30M
    187      Varies with device
    188      Varies with device
    189      Varies with device
    190                     29M
    191      Varies with device
    192      Varies with device
    193      Varies with device
    194                     35M
    195                    9.1M
    196                     25M
    197                    3.9M
    198                     18M
    199                     12M
    200                     21M
    201      Varies with device
    202      Varies with device
    203      Varies with device
    204      Varies with device
    205                     55M
    206                    3.2M
    207      Varies with device
    208      Varies with device
    209                     23k
    210                     16M
    211      Varies with device
    212                     14M
    213                     37M
    214      Varies with device
    215      Varies with device
    216                     11M
    217                     25M
    218                    7.3M
    219                    6.5M
    220                     25M
    221                    3.1M
    222      Varies with device
    223                    1.5M
    224                    7.5M
    225                    8.6M
    226      Varies with device
    227                    1.2M
    228      Varies with device
    229      Varies with device
    230      Varies with device
    231                     39M
    232                     14M
    233                     19M
    234                    6.8M
    235                     39M
    236      Varies with device
    237                     14M
    238                     20M
    239      Varies with device
    240      Varies with device
    241                     26M
    242                     51M
    243                     41M
    244                     20M
    245      Varies with device
    246                     12M
    247                     48M
    248                     10M
    249                     22M
    250                     11M
    251                    8.5M
    252                    8.6M
    253      Varies with device
    254                     28M
    255                     28M
    256                     37M
    257                    9.0M
    258                     46M
    259      Varies with device
    260                     14M
    261      Varies with device
    262                     26M
    263                     23M
    264      Varies with device
    265      Varies with device
    266                     14M
    267                     20M
    268      Varies with device
    269      Varies with device
    270                     26M
    271                     51M
    272                     41M
    273                     20M
    274      Varies with device
    275                     12M
    276                     48M
    277                     10M
    278                     22M
    279                     11M
    280                    8.5M
    281                    8.6M
    282      Varies with device
    283                     28M
    284      Varies with device
    285      Varies with device
    286      Varies with device
    287                     39M
    288                     14M
    289                     19M
    290                    6.8M
    291                     39M
    292      Varies with device
    293                     35M
    294      Varies with device
    295                     29M
    296                     41M
    297      Varies with device
    298                    4.9M
    299                     27M
    300                     32M
    301                     12M
    302                     15M
    303                     11M
    304                     28M
    305                    2.2M
    306                    3.4M
    307                    8.3M
    308                    4.3M
    309                    4.7M
    310                     10M
    311                     15M
    312                    7.1M
    313                     21M
    314                    6.1M
    315      Varies with device
    316                     15M
    317                     11M
    318                     29M
    319                    3.3M
    320      Varies with device
    321                     22M
    322                     40M
    323                     10M
    324                    9.1M
    325                    4.7M
    326                     38M
    327      Varies with device
    328                    6.7M
    329                     37M
    330      Varies with device
    331                    7.8M
    332                     19M
    333                    5.7M
    334                     35M
    335      Varies with device
    336      Varies with device
    337                     17M
    338      Varies with device
    339      Varies with device
    340      Varies with device
    341      Varies with device
    342      Varies with device
    343                    8.8M
    344      Varies with device
    345                     16M
    346                     11M
    347                     11M
    348      Varies with device
    349                     15M
    350                    6.6M
    351      Varies with device
    352                    6.6M
    353      Varies with device
    354      Varies with device
    355                    5.1M
    356      Varies with device
    357                     18M
    358                    4.0M
    359      Varies with device
    360      Varies with device
    361      Varies with device
    362                     37M
    363      Varies with device
    364                     22M
    365      Varies with device
    366                    3.3M
    367                     32M
    368                     37M
    369      Varies with device
    370      Varies with device
    371      Varies with device
    372      Varies with device
    373      Varies with device
    374                     17M
    375      Varies with device
    376      Varies with device
    377                     35M
    378                     40M
    379                     61M
    380                     66M
    381      Varies with device
    382      Varies with device
    383                     11M
    384                     79k
    385      Varies with device
    386      Varies with device
    387                     25M
    388                     14M
    389      Varies with device
    390      Varies with device
    391      Varies with device
    392      Varies with device
    393      Varies with device
    394      Varies with device
    395                     24M
    396      Varies with device
    397                     17M
    398                    8.3M
    399      Varies with device
    400                    8.2M
    401      Varies with device
    402      Varies with device
    403      Varies with device
    404      Varies with device
    405                    8.4M
    406      Varies with device
    407      Varies with device
    408                    4.0M
    409                     32M
    410      Varies with device
    411      Varies with device
    412      Varies with device
    413                    6.1M
    414      Varies with device
    415      Varies with device
    416                    2.8M
    417      Varies with device
    418      Varies with device
    419                    3.3M
    420                     40M
    421                    2.2M
    422      Varies with device
    423                    4.3M
    424                    4.7M
    425                    2.3M
    426                    2.3M
    427      Varies with device
    428                     11M
    429                    2.7M
    430                     14M
    431      Varies with device
    432                     13M
    433                    3.7M
    434                     10M
    435                     13M
    436                     10M
    437                     13M
    438                    8.8M
    439                    5.5M
    440                     20M
    441                     29M
    442      Varies with device
    443      Varies with device
    444                     11M
    445      Varies with device
    446                     17M
    447                     17M
    448      Varies with device
    449      Varies with device
    450                    118k
    451      Varies with device
    452                     16M
    453                    5.1M
    454      Varies with device
    455                     44M
    456                    7.3M
    457                     30M
    458                    695k
    459      Varies with device
    460                    1.6M
    461                     19M
    462      Varies with device
    463      Varies with device
    464      Varies with device
    465                     11M
    466      Varies with device
    467      Varies with device
    468      Varies with device
    469      Varies with device
    470                     23M
    471                     26M
    472      Varies with device
    473      Varies with device
    474      Varies with device
    475      Varies with device
    476      Varies with device
    477                    6.2M
    478                     20M
    479                     18k
    480                    1.2M
    481                     17M
    482      Varies with device
    483                     15M
    484      Varies with device
    485                     56M
    486                     12M
    487                     11M
    488                     29M
    489      Varies with device
    490                     40M
    491      Varies with device
    492                     53M
    493                    3.1M
    494                     24M
    495                     23M
    496                    5.0M
    497                     44M
    498                     27M
    499                    6.1M
    500      Varies with device
    501                     21M
    502                     11M
    503                     21M
    504                     24M
    505                     31M
    506                     27M
    507                    6.2M
    508      Varies with device
    509                     12M
    510                    8.0M
    511                    5.9M
    512                    7.9M
    513                    5.0M
    514                    1.4M
    515                     13M
    516                     40M
    517                     13M
    518                     19M
    519                    5.0M
    520                     19M
    521                     27M
    522                    3.0M
    523                     13M
    524                    7.2M
    525                     25M
    526      Varies with device
    527                    5.7M
    528                    5.5M
    529      Varies with device
    530                    6.5M
    531                    5.8M
    532      Varies with device
    533                    3.8M
    534                     41M
    535                    7.9M
    536                    2.8M
    537                     20M
    538                     15M
    539                     28M
    540                    9.6M
    541                     13M
    542                     15M
    543                     40M
    544                     56M
    545                     12M
    546                    9.4M
    547                     29M
    548                     11M
    549                     19M
    550                     23M
    551      Varies with device
    552                     27M
    553      Varies with device
    554                     19M
    555                     45M
    556                     21M
    557                     40M
    558                     25M
    559                     15M
    560                     24M
    561      Varies with device
    562                     11M
    563                     38M
    564                     31M
    565                    6.1M
    566                     27M
    567                     32M
    568                     13M
    569                     63M
    570                     44M
    571      Varies with device
    572                     16M
    573                     28M
    574      Varies with device
    575                    5.9M
    576                     13M
    577                    9.0M
    578                     20M
    579                    7.9M
    580                     28M
    581                     49M
    582                     27M
    583                     18M
    584                     14M
    585                     27M
    586                     41M
    587                     38M
    588                    5.5M
    589                     27M
    590                    7.2M
    591                     21M
    592                    8.8M
    593                     38M
    594      Varies with device
    595                     11M
    596                    6.5M
    597                     26M
    598                     19M
    599                    6.1M
    600                     77M
    601                    9.5M
    602                     14M
    603                     38M
    604                     16M
    605                    3.4M
    606                    4.7M
    607                    5.0M
    608                    4.9M
    609                    4.9M
    610                     16M
    611                    3.9M
    612                    4.4M
    613                    4.8M
    614                    4.5M
    615                     70M
    616                    4.9M
    617                    3.7M
    618                    4.9M
    619                     16M
    620                    4.6M
    621                    3.6M
    622                     21M
    623                     13M
    624                    8.7M
    625                    9.2M
    626                    6.9M
    627                     39M
    628                    8.0M
    629                    3.9M
    630                     14M
    631                     11M
    632                     13M
    633                    9.3M
    634                    5.0M
    635                   10.0M
    636                    5.0M
    637                    5.0M
    638                    3.6M
    639                     19M
    640                    5.0M
    641                    9.8M
    642                     14M
    643                    5.0M
    644                    4.9M
    645                    5.5M
    646                     11M
    647                    3.8M
    648                    5.0M
    649                    8.2M
    650                     21M
    651                     12M
    652                     10M
    653                    3.3M
    654                     25M
    655                     15M
    656                     10M
    657                    4.9M
    658                    4.4M
    659                    3.5M
    660                    7.8M
    661                    9.0M
    662                     15M
    663                     19M
    664                     56M
    665                     38M
    666                     16M
    667                    5.0M
    668                     27M
    669                    4.7M
    670      Varies with device
    671                    3.4M
    672                    4.9M
    673                     25M
    674                     29M
    675                    8.1M
    676                     14M
    677                     27M
    678                    3.9M
    679                     13M
    680                    5.0M
    681                    5.5M
    682                     16M
    683                    4.8M
    684                     14M
    685                    4.7M
    686                    3.7M
    687                    4.9M
    688                    3.5M
    689                    4.5M
    690      Varies with device
    691                     36M
    692                    4.5M
    693                     19M
    694                     14M
    695                     77M
    696                     21M
    697                    7.9M
    698                    9.3M
    699      Varies with device
    700                     18M
    701                     18M
    702                     21M
    703                    3.3M
    704                     24M
    705                     39M
    706                    3.2M
    707                    5.1M
    708                     11M
    709                     27M
    710                     37M
    711      Varies with device
    712                     26M
    713                     11M
    714      Varies with device
    715                     41M
    716                     49M
    717                     21M
    718                    8.1M
    719                     51M
    720                     14M
    721                     18M
    722                    3.0M
    723                     19M
    724      Varies with device
    725                     22M
    726                    6.9M
    727                    7.4M
    728                     84M
    729                     25M
    730                     18M
    731      Varies with device
    732                    2.5M
    733                    3.9M
    734      Varies with device
    735                     21M
    736                     27M
    737      Varies with device
    738                     21M
    739                     18M
    740      Varies with device
    741                     18M
    742      Varies with device
    743                     10M
    744      Varies with device
    745      Varies with device
    746      Varies with device
    747      Varies with device
    748                     97M
    749                     17M
    750      Varies with device
    751      Varies with device
    752                    2.0M
    753                    1.9M
    754                    1.8M
    755                    1.8M
    756                     17M
    757                     18M
    758                    5.3M
    759      Varies with device
    760                    5.4M
    761                    5.4M
    762                    5.4M
    763                     15M
    764                    5.3M
    765                     48M
    766                    5.4M
    767                     47M
    768                    556k
    769                     29M
    770                    2.3M
    771      Varies with device
    772                    4.4M
    773                    6.6M
    774      Varies with device
    775                    526k
    776                     12M
    777      Varies with device
    778                     29M
    779      Varies with device
    780      Varies with device
    781      Varies with device
    782                     21M
    783                     18M
    784      Varies with device
    785      Varies with device
    786      Varies with device
    787                     18M
    788                     17M
    789                     10M
    790                     76M
    791      Varies with device
    792                     17M
    793      Varies with device
    794      Varies with device
    795                     18M
    796                     21M
    797      Varies with device
    798                    6.9M
    799      Varies with device
    800                     14M
    801                     76M
    802                     11M
    803                     19M
    804                    6.5M
    805                    7.0M
    806      Varies with device
    807                    3.3M
    808                     21M
    809      Varies with device
    810                    2.6M
    811                     21M
    812                    5.2M
    813                     18M
    814      Varies with device
    815                     16M
    816                     15M
    817                    1.2M
    818      Varies with device
    819      Varies with device
    820                     18M
    821                    1.8M
    822                    7.6M
    823      Varies with device
    824      Varies with device
    825                     59M
    826      Varies with device
    827                    6.9M
    828                     14M
    829                     41M
    830                     19M
    831                     76M
    832      Varies with device
    833                     21M
    834                     21M
    835                     21M
    836      Varies with device
    837                     21M
    838      Varies with device
    839      Varies with device
    840                    5.4M
    841                    7.3M
    842      Varies with device
    843                     59M
    844                     41M
    845      Varies with device
    846                     21M
    847                     76M
    848                    7.6M
    849      Varies with device
    850                     13M
    851                     63M
    852                     44M
    853                     24M
    854                     24M
    855      Varies with device
    856                    4.5M
    857      Varies with device
    858                     11M
    859      Varies with device
    860                    4.6M
    861      Varies with device
    862      Varies with device
    863                    3.3M
    864                    6.5M
    865      Varies with device
    866      Varies with device
    867      Varies with device
    868                     12M
    869      Varies with device
    870                    5.6M
    871                    9.7M
    872                     15M
    873      Varies with device
    874                     52M
    875      Varies with device
    876      Varies with device
    877                    4.5M
    878      Varies with device
    879                     49M
    880                     18M
    881                     13M
    882                    4.0M
    883                     16M
    884                     17M
    885      Varies with device
    886                     24M
    887      Varies with device
    888                     12M
    889      Varies with device
    890                     78M
    891      Varies with device
    892                     25M
    893                     57M
    894      Varies with device
    895                    9.1M
    896                     39M
    897      Varies with device
    898                     16M
    899      Varies with device
    900                    8.5M
    901                     12M
    902                     72M
    903                     12M
    904      Varies with device
    905      Varies with device
    906                     16M
    907                    9.6M
    908      Varies with device
    909                     22M
    910      Varies with device
    911                     13M
    912      Varies with device
    913                     12M
    914      Varies with device
    915                     23M
    916                     20M
    917                     25M
    918                     17M
    919      Varies with device
    920                     19M
    921                     21M
    922                     19M
    923      Varies with device
    924      Varies with device
    925      Varies with device
    926      Varies with device
    927                     19M
    928                    4.4M
    929                     19M
    930                     12M
    931      Varies with device
    932                     11M
    933      Varies with device
    934      Varies with device
    935      Varies with device
    936                     12M
    937                     25M
    938      Varies with device
    939      Varies with device
    940                     21M
    941                     13M
    942                     17M
    943                     20M
    944      Varies with device
    945                     20M
    946                     19M
    947                     17M
    948                     21M
    949                     15M
    950                     19M
    951                     19M
    952                    7.2M
    953                     32M
    954      Varies with device
    955      Varies with device
    956                     44M
    957                     19M
    958      Varies with device
    959                     11M
    960      Varies with device
    961      Varies with device
    962      Varies with device
    963                     25M
    964      Varies with device
    965      Varies with device
    966      Varies with device
    967      Varies with device
    968                     16M
    969      Varies with device
    970                     20M
    971      Varies with device
    972                     21M
    973                     30M
    974      Varies with device
    975      Varies with device
    976                     57M
    977                     19M
    978                     17M
    979      Varies with device
    980                     15M
    981                     19M
    982                     19M
    983                     19M
    984                     44M
    985                     19M
    986                     35M
    987                     44M
    988      Varies with device
    989      Varies with device
    990                     20M
    991                     19M
    992                     25M
    993      Varies with device
    994                     43M
    995      Varies with device
    996      Varies with device
    997                    3.6M
    998                    7.7M
    999                     44M
    1000                    12M
    1001                    53M
    1002                    77M
    1003     Varies with device
    1004                   9.5M
    1005                    36M
    1006                   6.3M
    1007                   5.9M
    1008     Varies with device
    1009                    12M
    1010                   2.8M
    1011                    26M
    1012                   8.7M
    1013                    16M
    1014                    53M
    1015     Varies with device
    1016                    14M
    1017                   8.9M
    1018                    11M
    1019                   334k
    1020                    24M
    1021                    13M
    1022     Varies with device
    1023                   2.8M
    1024                   9.7M
    1025                   1.4M
    1026     Varies with device
    1027                   3.5M
    1028                    61M
    1029                   6.0M
    1030                   2.3M
    1031                   9.0M
    1032                    11M
    1033                    27M
    1034                   3.6M
    1035     Varies with device
    1036                    29M
    1037                   4.4M
    1038                    21M
    1039                    13M
    1040                    27M
    1041                    11M
    1042                    24M
    1043                   6.1M
    1044                   6.3M
    1045                    23M
    1046                    34M
    1047                   3.8M
    1048     Varies with device
    1049     Varies with device
    1050                    42M
    1051     Varies with device
    1052                    19M
    1053                    70M
    1054                    32M
    1055                    93M
    1056     Varies with device
    1057     Varies with device
    1058     Varies with device
    1059                    40M
    1060                    24M
    1061     Varies with device
    1062                    20M
    1063                    15M
    1064                    28M
    1065                    10M
    1066     Varies with device
    1067     Varies with device
    1068     Varies with device
    1069                    14M
    1070     Varies with device
    1071                    42M
    1072                    65M
    1073                    37M
    1074     Varies with device
    1075                    39M
    1076                    47M
    1077                    79M
    1078                    18M
    1079     Varies with device
    1080                   100M
    1081                    32M
    1082                    46M
    1083     Varies with device
    1084                   9.8M
    1085                   8.2M
    1086                    18M
    1087     Varies with device
    1088                    39M
    1089                    44M
    1090     Varies with device
    1091                   5.0M
    1092                    21M
    1093                    22M
    1094                    23M
    1095     Varies with device
    1096     Varies with device
    1097     Varies with device
    1098                    27M
    1099     Varies with device
    1100                   5.0M
    1101                    32M
    1102                   7.4M
    1103                    24M
    1104                    22M
    1105     Varies with device
    1106                    10M
    1107                    17M
    1108                    38M
    1109                    46M
    1110                   3.8M
    1111                   9.3M
    1112                   8.4M
    1113                    28M
    1114     Varies with device
    1115                   9.7M
    1116                    11M
    1117     Varies with device
    1118                   7.4M
    1119                    19M
    1120                    16M
    1121                    14M
    1122                    22M
    1123                    25M
    1124                   3.6M
    1125     Varies with device
    1126     Varies with device
    1127                   4.5M
    1128     Varies with device
    1129                    23M
    1130     Varies with device
    1131     Varies with device
    1132     Varies with device
    1133                   4.2M
    1134     Varies with device
    1135                    58M
    1136     Varies with device
    1137                    22M
    1138     Varies with device
    1139                   1.4M
    1140     Varies with device
    1141                   9.1M
    1142                    22M
    1143     Varies with device
    1144     Varies with device
    1145     Varies with device
    1146     Varies with device
    1147                    21M
    1148     Varies with device
    1149     Varies with device
    1150                    11M
    1151                    38M
    1152                    12M
    1153                    45M
    1154                    50M
    1155                    24M
    1156                    47M
    1157                    23M
    1158                    46M
    1159                    45M
    1160                    50M
    1161                    24M
    1162                    33M
    1163     Varies with device
    1164                    12M
    1165                    53M
    1166                    14M
    1167                    68M
    1168                    37M
    1169                    79M
    1170                    32M
    1171                    46M
    1172                    40M
    1173                    32M
    1174     Varies with device
    1175                    24M
    1176                    42M
    1177                    12M
    1178                    13M
    1179                    13M
    1180                   4.9M
    1181                   8.9M
    1182     Varies with device
    1183                    19M
    1184     Varies with device
    1185                    14M
    1186                    64M
    1187                   8.4M
    1188                    11M
    1189                    66M
    1190     Varies with device
    1191                    41M
    1192                    15M
    1193                    35M
    1194                    36M
    1195                    17M
    1196                    25M
    1197                    35M
    1198                    10M
    1199     Varies with device
    1200                    12M
    1201     Varies with device
    1202                    11M
    1203     Varies with device
    1204     Varies with device
    1205                    17M
    1206                    43M
    1207                    41M
    1208     Varies with device
    1209                    43M
    1210                    16M
    1211                    39M
    1212                    13M
    1213                    30M
    1214                   7.1M
    1215                    23M
    1216                    17M
    1217                    18M
    1218     Varies with device
    1219                    17M
    1220     Varies with device
    1221                    15M
    1222                    22M
    1223     Varies with device
    1224                   9.0M
    1225                   7.2M
    1226     Varies with device
    1227     Varies with device
    1228                   2.3M
    1229                    27M
    1230     Varies with device
    1231                   8.2M
    1232                    76M
    1233                    22M
    1234     Varies with device
    1235                    19M
    1236     Varies with device
    1237     Varies with device
    1238                    28M
    1239                    35M
    1240     Varies with device
    1241                    22M
    1242     Varies with device
    1243     Varies with device
    1244     Varies with device
    1245                    25M
    1246                    30M
    1247                   8.5M
    1248     Varies with device
    1249                    17M
    1250                    34M
    1251                    34M
    1252     Varies with device
    1253                    17M
    1254                    16M
    1255     Varies with device
    1256                    15M
    1257                   2.2M
    1258                    11M
    1259                   6.9M
    1260                    13M
    1261                    11M
    1262                   2.9M
    1263                    25M
    1264                   7.0M
    1265                    27M
    1266                   7.5M
    1267     Varies with device
    1268                    15M
    1269                   9.4M
    1270                   6.4M
    1271                   5.5M
    1272                   3.9M
    1273                    55M
    1274                    28M
    1275     Varies with device
    1276                   8.8M
    1277     Varies with device
    1278                   4.0M
    1279                   6.9M
    1280                    11M
    1281     Varies with device
    1282                   5.5M
    1283     Varies with device
    1284                    15M
    1285                   4.3M
    1286     Varies with device
    1287                   7.1M
    1288                    58M
    1289     Varies with device
    1290     Varies with device
    1291                    18M
    1292     Varies with device
    1293                   1.5M
    1294                   6.5M
    1295                    11M
    1296                    67M
    1297     Varies with device
    1298                    57M
    1299                    59M
    1300                    48M
    1301                    57M
    1302                    13M
    1303     Varies with device
    1304     Varies with device
    1305     Varies with device
    1306                    31M
    1307                    10M
    1308                    57M
    1309     Varies with device
    1310                    21M
    1311                    25M
    1312                    93M
    1313                    60M
    1314                    27M
    1315                    54M
    1316     Varies with device
    1317     Varies with device
    1318                    59M
    1319     Varies with device
    1320                    78M
    1321                    55M
    1322     Varies with device
    1323     Varies with device
    1324                    67M
    1325     Varies with device
    1326                    35M
    1327     Varies with device
    1328                    43M
    1329                    39M
    1330                   4.2M
    1331                    44M
    1332                    45M
    1333     Varies with device
    1334                    28M
    1335                    29M
    1336                    12M
    1337                    15M
    1338                    31M
    1339     Varies with device
    1340     Varies with device
    1341     Varies with device
    1342     Varies with device
    1343                    20M
    1344                    39M
    1345                    94M
    1346                    35M
    1347     Varies with device
    1348     Varies with device
    1349                    41M
    1350                    49M
    1351                    22M
    1352     Varies with device
    1353                    23M
    1354                    28M
    1355                   7.2M
    1356                   6.5M
    1357     Varies with device
    1358                    24M
    1359     Varies with device
    1360     Varies with device
    1361                    20M
    1362     Varies with device
    1363                    15M
    1364     Varies with device
    1365                    55M
    1366     Varies with device
    1367                    38M
    1368                    32M
    1369     Varies with device
    1370     Varies with device
    1371                    40M
    1372                    59M
    1373     Varies with device
    1374     Varies with device
    1375     Varies with device
    1376                    23M
    1377                   6.1M
    1378     Varies with device
    1379                    58M
    1380                    55M
    1381     Varies with device
    1382     Varies with device
    1383                    57M
    1384                    93M
    1385                    60M
    1386     Varies with device
    1387     Varies with device
    1388     Varies with device
    1389     Varies with device
    1390     Varies with device
    1391     Varies with device
    1392                    19M
    1393                   3.8M
    1394                    18M
    1395                    57M
    1396                   5.5M
    1397     Varies with device
    1398                   5.7M
    1399                    51M
    1400                   9.4M
    1401                   3.3M
    1402     Varies with device
    1403                   4.0M
    1404     Varies with device
    1405     Varies with device
    1406                    19M
    1407                   3.8M
    1408                   4.3M
    1409                    18M
    1410                   5.7M
    1411     Varies with device
    1412                   3.6M
    1413                   5.5M
    1414                    57M
    1415                    51M
    1416                    58M
    1417     Varies with device
    1418                   9.4M
    1419                   3.3M
    1420                   4.0M
    1421     Varies with device
    1422     Varies with device
    1423                   9.9M
    1424                    43M
    1425                    37M
    1426     Varies with device
    1427     Varies with device
    1428                    40M
    1429                    73M
    1430                    28M
    1431     Varies with device
    1432                    23M
    1433                    10M
    1434                    15M
    1435                   8.4M
    1436                    31M
    1437     Varies with device
    1438     Varies with device
    1439     Varies with device
    1440                    41M
    1441     Varies with device
    1442     Varies with device
    1443                    20M
    1444                    39M
    1445                    35M
    1446                    34M
    1447                   7.5M
    1448     Varies with device
    1449                    12M
    1450                   7.7M
    1451                    21M
    1452                    18M
    1453                    40M
    1454     Varies with device
    1455                   8.7M
    1456     Varies with device
    1457                   8.6M
    1458     Varies with device
    1459                    18M
    1460                   7.5M
    1461                   9.1M
    1462                    13M
    1463     Varies with device
    1464                    15M
    1465     Varies with device
    1466                    19M
    1467     Varies with device
    1468     Varies with device
    1469                   5.1M
    1470                    26M
    1471                   8.3M
    1472                    27M
    1473                   7.9M
    1474     Varies with device
    1475                    27M
    1476                   5.1M
    1477                    15M
    1478                   3.1M
    1479                    16M
    1480                   9.2M
    1481     Varies with device
    1482     Varies with device
    1483                    10M
    1484                   2.6M
    1485                    13M
    1486                   1.9M
    1487                   7.0M
    1488                    15M
    1489                   4.2M
    1490                   7.8M
    1491                    77M
    1492                   5.7M
    1493                   5.9M
    1494                   7.6M
    1495                   5.8M
    1496     Varies with device
    1497                   5.8M
    1498                   5.3M
    1499                   5.5M
    1500                    25M
    1501                    35M
    1502                   7.9M
    1503                    19M
    1504                   7.5M
    1505                    12M
    1506     Varies with device
    1507                    34M
    1508                    30M
    1509     Varies with device
    1510     Varies with device
    1511                    16M
    1512     Varies with device
    1513                    16M
    1514     Varies with device
    1515                    15M
    1516                    21M
    1517                   1.8M
    1518                   4.3M
    1519                    10M
    1520                   3.7M
    1521                   232k
    1522                    99M
    1523                   624k
    1524                   6.9M
    1525                   7.0M
    1526                   7.0M
    1527                   6.4M
    1528     Varies with device
    1529                    16M
    1530                   5.9M
    1531                    95M
    1532                   8.5k
    1533                    20M
    1534                   2.5M
    1535                    50M
    1536                    10M
    1537                    18M
    1538                    12M
    1539                   2.5M
    1540                   2.1M
    1541                    22M
    1542                    41k
    1543                    22M
    1544                   292k
    1545                   5.5M
    1546     Varies with device
    1547                   2.5M
    1548                   8.1M
    1549                    28M
    1550                    24M
    1551                   3.1M
    1552                   3.4M
    1553                    11k
    1554                    13M
    1555                    19M
    1556                   9.5M
    1557                    21M
    1558                   9.4M
    1559                   2.4M
    1560                    36M
    1561                    55M
    1562                    32M
    1563                    34M
    1564                   5.1M
    1565                   100M
    1566                    80M
    1567                   3.3M
    1568                    28M
    1569                   5.0M
    1570                    11M
    1571                    55M
    1572                   7.1M
    1573                   3.2M
    1574                   4.4M
    1575                    16M
    1576                   6.3M
    1577                    45M
    1578                    23M
    1579                    37M
    1580                    13M
    1581                   8.1M
    1582                    13M
    1583                    14M
    1584                    30M
    1585                    36M
    1586                   3.0M
    1587                   1.4M
    1588                    39M
    1589                   1.8M
    1590                   1.7M
    1591                    19M
    1592                    15M
    1593                  10.0M
    1594                   2.7M
    1595                    16M
    1596                    21M
    1597                    20M
    1598                    14M
    1599                    76M
    1600                    18M
    1601                    12M
    1602                   6.2M
    1603     Varies with device
    1604                   6.8M
    1605     Varies with device
    1606                    24M
    1607                    19M
    1608                    13M
    1609                    15M
    1610                    13M
    1611     Varies with device
    1612     Varies with device
    1613                    44M
    1614     Varies with device
    1615                   9.7M
    1616                    24M
    1617                    13M
    1618                   4.5M
    1619                   4.5M
    1620                    29M
    1621                    12M
    1622                    12M
    1623     Varies with device
    1624                   4.6M
    1625                    10M
    1626                   6.8M
    1627                    33M
    1628                    12M
    1629     Varies with device
    1630                   4.6M
    1631                    10M
    1632                   6.8M
    1633                    33M
    1634                    13M
    1635                    15M
    1636                    23M
    1637                   7.7M
    1638     Varies with device
    1639                    15M
    1640                    25M
    1641     Varies with device
    1642                   3.0M
    1643                    22M
    1644     Varies with device
    1645     Varies with device
    1646                   7.5M
    1647                    24M
    1648     Varies with device
    1649                    44M
    1650     Varies with device
    1651     Varies with device
    1652                    14M
    1653                    67M
    1654                    76M
    1655                    74M
    1656                    23M
    1657                    46M
    1658                    24M
    1659     Varies with device
    1660                    97M
    1661                    62M
    1662                    24M
    1663                    33M
    1664                   7.8M
    1665                    46M
    1666                    69M
    1667                    75M
    1668                    67M
    1669                    50M
    1670                    98M
    1671     Varies with device
    1672                   4.9M
    1673                    37M
    1674                    18M
    1675                    52M
    1676     Varies with device
    1677                    78M
    1678                   3.9M
    1679                    59M
    1680                    97M
    1681                    11M
    1682                    38M
    1683     Varies with device
    1684     Varies with device
    1685     Varies with device
    1686     Varies with device
    1687                    85M
    1688                    78M
    1689                    63M
    1690     Varies with device
    1691                    24M
    1692                    69M
    1693                    75M
    1694     Varies with device
    1695                    70M
    1696                    63M
    1697                    99M
    1698                    49M
    1699     Varies with device
    1700                    76M
    1701                    67M
    1702                    24M
    1703                    52M
    1704                    98M
    1705                    74M
    1706                    69M
    1707                    82M
    1708     Varies with device
    1709                    96M
    1710                    99M
    1711                    46M
    1712     Varies with device
    1713                    99M
    1714                    87M
    1715                   3.9M
    1716                    67M
    1717     Varies with device
    1718                    53M
    1719                   7.8M
    1720     Varies with device
    1721                    97M
    1722     Varies with device
    1723     Varies with device
    1724                    59M
    1725     Varies with device
    1726                    46M
    1727                    75M
    1728                    50M
    1729                    62M
    1730     Varies with device
    1731                    23M
    1732                    72M
    1733                    70M
    1734                    18M
    1735     Varies with device
    1736                    35M
    1737                    78M
    1738                    25M
    1739                    74M
    1740                    97M
    1741                    17M
    1742                    53M
    1743     Varies with device
    1744                    11M
    1745                    60M
    1746                    57M
    1747                    14M
    1748                    67M
    1749                    33M
    1750                    76M
    1751                    74M
    1752     Varies with device
    1753                    40M
    1754                    59M
    1755                    52M
    1756     Varies with device
    1757     Varies with device
    1758                   100M
    1759                    62M
    1760     Varies with device
    1761     Varies with device
    1762     Varies with device
    1763     Varies with device
    1764                    85M
    1765                    87M
    1766                    33M
    1767                    32M
    1768                    40M
    1769                    29M
    1770                    40M
    1771                    24M
    1772                    38M
    1773                    52M
    1774     Varies with device
    1775                    54M
    1776                    71M
    1777                    73M
    1778                    57M
    1779                    38M
    1780                    62M
    1781                    95M
    1782                    16M
    1783                    39M
    1784                    78M
    1785                    36M
    1786     Varies with device
    1787                    31M
    1788     Varies with device
    1789                    98M
    1790                    24M
    1791                    85M
    1792                    33M
    1793                   100M
    1794                    36M
    1795                    86M
    1796                    80M
    1797                    87M
    1798                    79M
    1799                    30M
    1800                    87M
    1801                    91M
    1802                    63M
    1803     Varies with device
    1804     Varies with device
    1805                    95M
    1806                    48M
    1807     Varies with device
    1808     Varies with device
    1809                    93M
    1810                    53M
    1811                    82M
    1812                    99M
    1813                    49M
    1814                    25M
    1815     Varies with device
    1816                    91M
    1817                    52M
    1818                    56M
    1819                    56M
    1820                    85M
    1821                    81M
    1822                    82M
    1823                    98M
    1824                    77M
    1825                    99M
    1826     Varies with device
    1827                    96M
    1828                    96M
    1829                    82M
    1830                    70M
    1831                    63M
    1832                    35M
    1833                    48M
    1834                    84M
    1835                    15M
    1836                    69M
    1837                    46M
    1838                    15M
    1839                    33M
    1840                    48M
    1841                    67M
    1842                    74M
    1843                    57M
    1844                    52M
    1845                    25M
    1846     Varies with device
    1847                    36M
    1848                    31M
    1849                    91M
    1850     Varies with device
    1851                    92M
    1852                    70M
    1853                    93M
    1854                    52M
    1855                    91M
    1856                    37M
    1857                    64M
    1858                    35M
    1859                    97M
    1860                    93M
    1861                    28M
    1862                    83M
    1863                    99M
    1864                    55M
    1865                    82M
    1866                    14M
    1867                    81M
    1868     Varies with device
    1869                    74M
    1870                    67M
    1871                    52M
    1872                    76M
    1873                    67M
    1874                    50M
    1875                    46M
    1876     Varies with device
    1877     Varies with device
    1878                    97M
    1879                    98M
    1880                    70M
    1881                    69M
    1882                   3.9M
    1883                   7.8M
    1884                    18M
    1885                    24M
    1886                    62M
    1887                    11M
    1888     Varies with device
    1889                    87M
    1890                    99M
    1891     Varies with device
    1892                    82M
    1893                    94M
    1894     Varies with device
    1895     Varies with device
    1896     Varies with device
    1897                    96M
    1898                    75M
    1899     Varies with device
    1900                    14M
    1901                    59M
    1902                    74M
    1903                    29M
    1904                    17M
    1905                    97M
    1906                    78M
    1907     Varies with device
    1908     Varies with device
    1909                    63M
    1910     Varies with device
    1911                    25M
    1912                    29M
    1913                    72M
    1914     Varies with device
    1915                    53M
    1916     Varies with device
    1917                    76M
    1918                    33M
    1919                    60M
    1920                    62M
    1921     Varies with device
    1922     Varies with device
    1923                    78M
    1924                    75M
    1925     Varies with device
    1926                    70M
    1927                    99M
    1928                    99M
    1929     Varies with device
    1930                    27M
    1931                    70M
    1932                    88M
    1933                    63M
    1934                    82M
    1935                    48M
    1936     Varies with device
    1937                    31M
    1938                    23M
    1939     Varies with device
    1940                    59M
    1941                    39M
    1942                    54M
    1943                    48M
    1944     Varies with device
    1945                    30M
    1946                    66M
    1947                    32M
    1948                    50M
    1949                    53M
    1950                    61M
    1951                    49M
    1952                    38M
    1953                    20M
    1954     Varies with device
    1955     Varies with device
    1956                    66M
    1957                    57M
    1958                    95M
    1959     Varies with device
    1960     Varies with device
    1961                    32M
    1962                    33M
    1963                    66M
    1964                    75M
    1965                    29M
    1966                    74M
    1967                    69M
    1968                   7.8M
    1969                    33M
    1970                    52M
    1971                    50M
    1972                    46M
    1973                    12M
    1974                    41M
    1975                    63M
    1976                    53M
    1977                    63M
    1978                    99M
    1979     Varies with device
    1980                    20M
    1981                    11M
    1982                    75M
    1983                    13M
    1984                    46M
    1985                    78M
    1986                    49M
    1987                    14M
    1988                   100M
    1989     Varies with device
    1990                    97M
    1991     Varies with device
    1992     Varies with device
    1993                    15M
    1994                    33M
    1995     Varies with device
    1996                    70M
    1997                    10M
    1998                    77M
    1999     Varies with device
    2000     Varies with device
    2001                   4.9M
    2002     Varies with device
    2003                    25M
    2004                    96M
    2005                    23M
    2006                    51M
    2007                    96M
    2008     Varies with device
    2009     Varies with device
    2010     Varies with device
    2011                    70M
    2012                    38M
    2013                    24M
    2014     Varies with device
    2015                    20M
    2016                    67M
    2017                    19M
    2018                    51M
    2019                    22M
    2020                    46M
    2021                    21M
    2022                    39M
    2023                    23M
    2024                    26M
    2025                    39M
    2026                    24M
    2027                    15M
    2028                    20M
    2029                    24M
    2030                    44M
    2031                    38M
    2032                    52M
    2033                    14M
    2034                    49M
    2035                   8.9M
    2036     Varies with device
    2037                    56M
    2038                    99M
    2039                    10M
    2040                   6.9M
    2041                    20M
    2042                    10M
    2043                    19M
    2044                    33M
    2045                    85M
    2046                    48M
    2047                   9.6M
    2048     Varies with device
    2049                    15M
    2050     Varies with device
    2051                    50M
    2052                    51M
    2053                    24M
    2054                    13M
    2055                    16M
    2056     Varies with device
    2057     Varies with device
    2058                    63M
    2059                    15M
    2060                    26M
    2061                    81M
    2062                    79M
    2063     Varies with device
    2064                    94M
    2065                    46M
    2066                    43M
    2067                    45M
    2068                    91M
    2069                    91M
    2070                    53M
    2071                    28M
    2072                    26M
    2073                    56M
    2074                    80M
    2075                    60M
    2076     Varies with device
    2077                    67M
    2078                    78M
    2079                    27M
    2080     Varies with device
    2081                    63M
    2082                    26M
    2083                    46M
    2084                    83M
    2085                    60M
    2086                    44M
    2087                    82M
    2088                    67M
    2089                    99M
    2090     Varies with device
    2091                    24M
    2092                    78M
    2093     Varies with device
    2094     Varies with device
    2095                    51M
    2096                    94M
    2097                    16M
    2098                    93M
    2099                    77M
    2100                    32M
    2101                    58M
    2102                    48M
    2103                    69M
    2104                    73M
    2105                    99M
    2106                    16M
    2107                    54M
    2108                    56M
    2109                    68M
    2110                    25M
    2111                   7.0M
    2112                    54M
    2113                    51M
    2114     Varies with device
    2115                    48M
    2116                    44M
    2117                   8.7M
    2118                    91M
    2119                    97M
    2120                    26M
    2121                    16M
    2122     Varies with device
    2123                    53M
    2124                    95M
    2125                    83M
    2126                    78M
    2127                    56M
    2128                    21M
    2129                    26M
    2130                    60M
    2131                    22M
    2132                    95M
    2133                    48M
    2134                    58M
    2135                    91M
    2136                    22M
    2137                    36M
    2138                    37M
    2139                    70M
    2140                    34M
    2141                    49M
    2142                   9.8M
    2143                    37M
    2144     Varies with device
    2145                    55M
    2146                    84M
    2147                    16M
    2148                    92M
    2149                    57M
    2150     Varies with device
    2151                    24M
    2152                   6.9M
    2153                    44M
    2154                    11M
    2155                    16M
    2156                    17M
    2157                    18M
    2158                    16M
    2159                    19M
    2160                   1.6M
    2161                    47M
    2162                    16M
    2163                    14M
    2164     Varies with device
    2165                    75M
    2166                    25M
    2167                   5.7M
    2168     Varies with device
    2169                    17M
    2170                   3.5M
    2171                    16M
    2172                    42M
    2173                    47M
    2174                    67M
    2175                    26M
    2176                    72M
    2177                    37M
    2178                   1.2M
    2179                    17M
    2180                    19M
    2181                    23M
    2182                    25M
    2183                    83M
    2184                    94M
    2185                    97M
    2186                    37M
    2187                    92M
    2188                    48M
    2189                    12M
    2190                    15M
    2191                    83M
    2192                   9.5M
    2193                    19M
    2194                    23M
    2195                    25M
    2196                    83M
    2197                    94M
    2198                    97M
    2199                    37M
    2200                    92M
    2201                    48M
    2202                    12M
    2203                    15M
    2204                    83M
    2205                   9.5M
    2206                    67M
    2207     Varies with device
    2208     Varies with device
    2209     Varies with device
    2210                    63M
    2211                    49M
    2212                    14M
    2213                    58M
    2214                    22M
    2215                    15M
    2216     Varies with device
    2217                    50M
    2218                    59M
    2219                    34M
    2220     Varies with device
    2221                    54M
    2222                    53M
    2223     Varies with device
    2224     Varies with device
    2225                    37M
    2226                    25M
    2227                    26M
    2228                    61M
    2229                    58M
    2230                    91M
    2231                    17M
    2232                    83M
    2233                    17M
    2234     Varies with device
    2235                    64M
    2236                    41M
    2237                    58M
    2238                    23M
    2239                    71M
    2240     Varies with device
    2241     Varies with device
    2242                    23M
    2243                    12M
    2244                   704k
    2245                   2.9M
    2246                    25M
    2247     Varies with device
    2248                    20M
    2249                    21M
    2250                   5.4M
    2251                    25M
    2252                    42M
    2253                    32M
    2254                   1.8M
    2255                    33M
    2256                   3.8M
    2257                   5.8M
    2258                   9.5M
    2259                    19M
    2260     Varies with device
    2261                   5.0M
    2262                    23M
    2263                   862k
    2264                   9.9M
    2265                    24M
    2266                   2.4M
    2267                   899k
    2268                    38M
    2269                   6.5M
    2270                    14M
    2271                   1.2M
    2272                    48M
    2273                   378k
    2274                    22M
    2275                   4.8M
    2276     Varies with device
    2277                    25M
    2278                   266k
    2279                   375k
    2280                    62M
    2281                   1.2M
    2282     Varies with device
    2283                   2.8M
    2284                    95M
    2285                   2.6M
    2286                   4.6M
    2287     Varies with device
    2288                   5.1M
    2289                   1.8M
    2290     Varies with device
    2291     Varies with device
    2292                    53M
    2293                    39M
    2294                   5.3M
    2295                    14M
    2296                   8.4M
    2297                    18M
    2298                    41M
    2299                   100M
    2300                   3.4M
    2301                    68M
    2302                   2.6M
    2303                    11M
    2304     Varies with device
    2305                    37M
    2306     Varies with device
    2307                   9.8M
    2308                    23M
    2309                    18M
    2310                   7.4M
    2311                    11M
    2312     Varies with device
    2313                    23M
    2314                    34M
    2315                    15M
    2316                    13M
    2317                    24M
    2318                    20M
    2319                    14M
    2320                    69M
    2321                    38M
    2322                    26M
    2323                   2.6M
    2324                    21M
    2325                    38M
    2326                    16M
    2327                    21M
    2328                    10M
    2329                    22M
    2330                    14M
    2331                    36M
    2332                    16M
    2333                    61M
    2334                    11M
    2335                   8.5M
    2336                    37M
    2337     Varies with device
    2338     Varies with device
    2339                    22M
    2340                    17M
    2341                   7.7M
    2342                    39M
    2343                    30M
    2344                   3.7M
    2345                    28M
    2346                    16M
    2347                    20M
    2348                   8.9M
    2349                    97M
    2350                    12M
    2351                    72M
    2352     Varies with device
    2353                    21M
    2354     Varies with device
    2355                    27M
    2356                   5.7M
    2357                   6.6M
    2358                    12M
    2359                   5.9M
    2360                    20M
    2361                    15M
    2362                    26M
    2363                   3.3M
    2364                    44M
    2365                    32M
    2366                    12M
    2367                    36M
    2368                    17M
    2369                   5.8M
    2370                    11M
    2371                    16M
    2372                    25M
    2373                   6.1M
    2374     Varies with device
    2375                    22M
    2376                    28M
    2377                   6.3M
    2378                    25M
    2379                    60M
    2380                    20M
    2381                   8.0M
    2382                    20M
    2383                   6.4M
    2384                    34M
    2385                   3.8M
    2386                    42M
    2387                   2.4M
    2388                    70M
    2389                   5.0M
    2390                    28M
    2391                    21M
    2392                    38M
    2393     Varies with device
    2394                    74M
    2395                   1.8M
    2396                    12M
    2397                    13M
    2398     Varies with device
    2399                   2.9M
    2400                    48M
    2401                    19M
    2402                    68M
    2403                   6.8M
    2404                    12M
    2405                   5.8M
    2406                    41M
    2407                    25M
    2408                    42M
    2409                   6.5M
    2410                    13M
    2411                   1.5M
    2412                   3.2M
    2413                   7.7M
    2414                   1.3M
    2415                    26M
    2416                   5.4M
    2417                    11M
    2418                   9.5M
    2419                    62M
    2420                    18M
    2421     Varies with device
    2422                   3.1M
    2423                   1.8M
    2424                    66M
    2425                   5.0M
    2426                   2.9M
    2427                    22M
    2428                    16M
    2429                    19M
    2430                    40M
    2431                    23M
    2432                    15M
    2433                    29M
    2434                    16M
    2435                    65M
    2436                    29M
    2437                    20M
    2438                   3.8M
    2439                    19M
    2440                    25M
    2441                   5.1M
    2442                    16M
    2443                   1.5M
    2444                  10.0M
    2445                    22M
    2446                   3.6M
    2447                    20M
    2448                    20M
    2449                    13M
    2450                   2.5M
    2451                   3.6M
    2452                    24M
    2453                    10M
    2454                    25M
    2455                    29M
    2456                    17M
    2457                    53M
    2458                    83M
    2459                   4.6M
    2460                   4.6M
    2461                    28M
    2462                    29M
    2463                    46M
    2464                   5.7M
    2465                   1.4M
    2466                    12M
    2467                   2.2M
    2468                    27M
    2469                    18M
    2470                    30M
    2471                   3.5M
    2472                   4.2M
    2473                   6.9M
    2474                   6.6M
    2475                   7.0M
    2476                    11M
    2477                   5.6M
    2478                    29M
    2479                   3.3M
    2480                    20M
    2481                   7.0M
    2482                   3.3M
    2483                    29M
    2484                   7.1M
    2485     Varies with device
    2486                    23M
    2487                    24M
    2488                    14M
    2489                    13M
    2490                    34M
    2491                   4.3M
    2492                    43M
    2493                    69M
    2494                    22M
    2495                    20M
    2496                    25M
    2497                   2.0M
    2498                    40M
    2499                    37M
    2500                    40M
    2501                   4.3M
    2502                   5.7M
    2503                   4.0M
    2504                    17M
    2505                    11M
    2506                   4.4M
    2507                    15M
    2508                   2.3M
    2509                    40M
    2510                   8.4M
    2511                    11M
    2512                   2.9M
    2513                   2.4M
    2514                    15M
    2515                    24M
    2516                    29M
    2517                    24M
    2518                    22M
    2519                    86M
    2520     Varies with device
    2521                    22M
    2522                    11M
    2523                   975k
    2524                   980k
    2525                   6.0M
    2526                    20M
    2527                    26M
    2528                    13M
    2529                    23M
    2530                   2.7M
    2531                   4.2M
    2532                    28M
    2533                   6.1M
    2534                    15M
    2535                   1.2M
    2536                    37M
    2537                   5.9M
    2538                    14M
    2539                   2.3M
    2540                    32M
    2541                   2.7M
    2542                   2.2M
    2543                    26M
    2544     Varies with device
    2545     Varies with device
    2546     Varies with device
    2547                   4.0M
    2548     Varies with device
    2549                   1.5M
    2550     Varies with device
    2551                   3.7M
    2552     Varies with device
    2553     Varies with device
    2554     Varies with device
    2555                   2.8M
    2556                   3.9M
    2557                   3.1M
    2558     Varies with device
    2559                   2.8M
    2560                    20M
    2561                   9.9M
    2562                   5.3M
    2563                    10M
    2564                   2.6M
    2565     Varies with device
    2566                    31M
    2567                   2.3M
    2568                   6.0M
    2569                    17M
    2570                    20M
    2571                    62M
    2572     Varies with device
    2573                    15M
    2574     Varies with device
    2575                    18M
    2576                   5.4M
    2577                   2.8M
    2578                    15M
    2579     Varies with device
    2580                   5.3M
    2581     Varies with device
    2582     Varies with device
    2583                    37M
    2584     Varies with device
    2585                    23M
    2586     Varies with device
    2587     Varies with device
    2588     Varies with device
    2589     Varies with device
    2590                    28M
    2591                    35M
    2592     Varies with device
    2593                    76M
    2594     Varies with device
    2595                    34M
    2596     Varies with device
    2597     Varies with device
    2598     Varies with device
    2599     Varies with device
    2600                   2.7M
    2601     Varies with device
    2602                   4.1M
    2603     Varies with device
    2604     Varies with device
    2605     Varies with device
    2606     Varies with device
    2607                    85M
    2608                    15M
    2609                    89M
    2610     Varies with device
    2611     Varies with device
    2612     Varies with device
    2613                    20M
    2614     Varies with device
    2615                   9.3M
    2616     Varies with device
    2617                    76M
    2618                    34M
    2619                    16M
    2620     Varies with device
    2621     Varies with device
    2622     Varies with device
    2623     Varies with device
    2624     Varies with device
    2625     Varies with device
    2626                   8.4M
    2627     Varies with device
    2628                    20M
    2629     Varies with device
    2630                    76M
    2631                    68M
    2632                    13M
    2633                    50M
    2634                    23M
    2635     Varies with device
    2636     Varies with device
    2637                    28M
    2638     Varies with device
    2639     Varies with device
    2640                   7.0M
    2641                    23M
    2642     Varies with device
    2643                    76M
    2644     Varies with device
    2645                    68M
    2646     Varies with device
    2647     Varies with device
    2648                    13M
    2649                    56M
    2650     Varies with device
    2651     Varies with device
    2652                   8.4M
    2653                    23M
    2654                    22M
    2655                    15M
    2656     Varies with device
    2657                    30M
    2658                    30M
    2659                    27M
    2660     Varies with device
    2661     Varies with device
    2662     Varies with device
    2663                    20M
    2664                    42M
    2665     Varies with device
    2666                    18M
    2667                   4.2M
    2668     Varies with device
    2669                    30M
    2670                    33M
    2671                   7.9M
    2672                    12M
    2673                    15M
    2674     Varies with device
    2675                    10M
    2676                   9.9M
    2677                   9.0M
    2678     Varies with device
    2679                    21M
    2680                    13M
    2681                    11M
    2682                    21M
    2683                   8.1M
    2684     Varies with device
    2685     Varies with device
    2686                    12M
    2687                   7.3M
    2688     Varies with device
    2689                    15M
    2690                    22M
    2691                    11M
    2692                   8.3M
    2693     Varies with device
    2694                    20M
    2695                    38M
    2696                   9.1M
    2697                    24M
    2698     Varies with device
    2699     Varies with device
    2700                    29M
    2701     Varies with device
    2702                    98M
    2703                    25M
    2704     Varies with device
    2705                    18M
    2706     Varies with device
    2707                   6.1M
    2708                    14M
    2709                    21M
    2710     Varies with device
    2711                    43M
    2712                    15M
    2713                    23M
    2714                    52M
    2715     Varies with device
    2716     Varies with device
    2717                    57M
    2718                    20M
    2719                    40M
    2720                    17M
    2721                    13M
    2722                    33M
    2723                    18M
    2724     Varies with device
    2725     Varies with device
    2726     Varies with device
    2727     Varies with device
    2728                    29M
    2729     Varies with device
    2730                    31M
    2731     Varies with device
    2732     Varies with device
    2733     Varies with device
    2734                    20M
    2735                    13M
    2736                   7.3M
    2737     Varies with device
    2738     Varies with device
    2739     Varies with device
    2740                    12M
    2741                   8.8M
    2742                    18M
    2743                    14M
    2744                    21M
    2745                    43M
    2746                    34M
    2747     Varies with device
    2748     Varies with device
    2749                    15M
    2750     Varies with device
    2751                    26M
    2752     Varies with device
    2753                    34M
    2754                   2.7M
    2755                    52M
    2756     Varies with device
    2757                    29M
    2758     Varies with device
    2759     Varies with device
    2760                    20M
    2761                    23M
    2762     Varies with device
    2763                    42M
    2764                    16M
    2765                    12M
    2766                   6.5M
    2767     Varies with device
    2768                    15M
    2769                    17M
    2770                    24M
    2771                    22M
    2772     Varies with device
    2773                    15M
    2774     Varies with device
    2775                    18M
    2776                   8.8M
    2777                    14M
    2778                   2.7M
    2779                    29M
    2780                    20M
    2781                   6.5M
    2782                    15M
    2783     Varies with device
    2784     Varies with device
    2785                   6.2M
    2786                    19M
    2787                    27M
    2788                   1.1M
    2789                   2.7M
    2790     Varies with device
    2791                    16M
    2792                   8.8M
    2793                    12M
    2794                    24M
    2795                    56M
    2796                    15M
    2797                    18M
    2798     Varies with device
    2799                    14M
    2800                    22M
    2801                    28M
    2802                    59M
    2803                    37M
    2804                    36M
    2805                   9.5M
    2806                    82M
    2807                    28M
    2808     Varies with device
    2809     Varies with device
    2810                   9.7M
    2811                    22M
    2812                    17M
    2813     Varies with device
    2814     Varies with device
    2815                   7.7M
    2816     Varies with device
    2817                    11M
    2818                    21M
    2819                    19M
    2820     Varies with device
    2821                   8.7M
    2822                    30M
    2823                    25M
    2824                    17M
    2825     Varies with device
    2826                    13M
    2827                    23M
    2828     Varies with device
    2829     Varies with device
    2830                    22M
    2831                   9.1M
    2832                   8.3M
    2833     Varies with device
    2834                    19M
    2835                    53M
    2836     Varies with device
    2837                    17M
    2838                    10M
    2839     Varies with device
    2840                    29M
    2841                    46M
    2842                   4.5M
    2843                   4.9M
    2844     Varies with device
    2845                    24M
    2846                    23M
    2847                    25M
    2848                    26M
    2849     Varies with device
    2850                   2.0M
    2851                    19M
    2852                   9.6M
    2853     Varies with device
    2854                    13M
    2855                   4.2M
    2856                    16M
    2857                    59M
    2858     Varies with device
    2859     Varies with device
    2860                   9.5M
    2861     Varies with device
    2862                    53M
    2863                   6.6M
    2864     Varies with device
    2865     Varies with device
    2866                    27M
    2867     Varies with device
    2868                    22M
    2869     Varies with device
    2870                    50M
    2871                    30M
    2872     Varies with device
    2873                    17M
    2874                    18M
    2875                  10.0M
    2876                   9.9M
    2877                    23M
    2878                    51M
    2879                    35M
    2880                    16M
    2881                    11M
    2882                    74M
    2883                    48M
    2884     Varies with device
    2885                    13M
    2886                   4.2M
    2887                    16M
    2888                    59M
    2889     Varies with device
    2890                   5.6M
    2891                   1.5M
    2892                   5.7M
    2893                   6.1M
    2894                   1.9M
    2895                   4.0M
    2896                    47M
    2897     Varies with device
    2898                   2.0M
    2899                   9.5M
    2900     Varies with device
    2901                    22M
    2902                    14M
    2903                    10M
    2904     Varies with device
    2905     Varies with device
    2906                   6.9M
    2907                    44M
    2908     Varies with device
    2909     Varies with device
    2910     Varies with device
    2911                    51M
    2912                    48M
    2913     Varies with device
    2914                    43M
    2915                    21M
    2916     Varies with device
    2917                    25M
    2918                   3.9M
    2919                   4.0M
    2920                    51M
    2921     Varies with device
    2922     Varies with device
    2923                   9.6M
    2924     Varies with device
    2925                   9.5M
    2926     Varies with device
    2927                    50M
    2928     Varies with device
    2929     Varies with device
    2930                    21M
    2931     Varies with device
    2932                    31M
    2933                    27M
    2934     Varies with device
    2935     Varies with device
    2936     Varies with device
    2937                    53M
    2938                    34M
    2939     Varies with device
    2940                    12M
    2941                    50M
    2942                    47M
    2943     Varies with device
    2944     Varies with device
    2945     Varies with device
    2946                   9.2M
    2947     Varies with device
    2948     Varies with device
    2949                    51M
    2950                    48M
    2951                    24M
    2952                   1.6M
    2953                    17M
    2954                    31M
    2955                    59M
    2956                    46M
    2957                    53M
    2958                    45M
    2959     Varies with device
    2960                   9.8M
    2961     Varies with device
    2962     Varies with device
    2963     Varies with device
    2964                    34M
    2965                    20M
    2966                    19M
    2967                   6.0M
    2968     Varies with device
    2969                    15M
    2970                   6.5M
    2971                   6.2M
    2972     Varies with device
    2973     Varies with device
    2974     Varies with device
    2975     Varies with device
    2976     Varies with device
    2977     Varies with device
    2978                    14M
    2979     Varies with device
    2980     Varies with device
    2981                   6.1M
    2982                    17M
    2983                    24M
    2984     Varies with device
    2985                    13M
    2986                    10M
    2987                    10M
    2988                   6.9M
    2989                    31M
    2990     Varies with device
    2991     Varies with device
    2992                    28M
    2993                   2.6M
    2994                    30M
    2995                    13M
    2996                    27M
    2997                    29M
    2998                    18M
    2999                    35M
    3000                    17M
    3001     Varies with device
    3002                    18M
    3003     Varies with device
    3004                   3.0M
    3005                    32M
    3006                    24M
    3007     Varies with device
    3008     Varies with device
    3009     Varies with device
    3010     Varies with device
    3011     Varies with device
    3012     Varies with device
    3013                    10M
    3014                    34M
    3015     Varies with device
    3016                   6.6M
    3017                    27M
    3018     Varies with device
    3019     Varies with device
    3020     Varies with device
    3021                    25M
    3022                   6.6M
    3023                    24M
    3024     Varies with device
    3025                    25M
    3026                    33M
    3027                    88M
    3028                   9.3M
    3029     Varies with device
    3030     Varies with device
    3031                    84M
    3032                   4.7M
    3033                    25M
    3034                    12M
    3035     Varies with device
    3036                   9.9M
    3037     Varies with device
    3038                    61M
    3039                    25M
    3040                    52M
    3041                    57M
    3042     Varies with device
    3043                    41M
    3044                   4.6M
    3045                    10M
    3046                    23M
    3047                    27M
    3048     Varies with device
    3049                    33M
    3050     Varies with device
    3051                   4.7M
    3052                   2.2M
    3053     Varies with device
    3054                    95M
    3055                    34M
    3056     Varies with device
    3057                   9.4M
    3058                    19M
    3059                   3.9M
    3060     Varies with device
    3061     Varies with device
    3062     Varies with device
    3063                    34M
    3064     Varies with device
    3065                    21M
    3066                    19M
    3067                   6.6M
    3068                    82M
    3069     Varies with device
    3070     Varies with device
    3071     Varies with device
    3072     Varies with device
    3073                    19M
    3074     Varies with device
    3075                    26M
    3076                   5.2M
    3077                    16M
    3078     Varies with device
    3079                    33M
    3080     Varies with device
    3081     Varies with device
    3082                   6.0M
    3083                    25M
    3084     Varies with device
    3085                    34M
    3086                   8.1M
    3087                    35M
    3088                    25M
    3089     Varies with device
    3090     Varies with device
    3091                    19M
    3092                    48M
    3093                    21M
    3094                   6.6M
    3095                    32M
    3096     Varies with device
    3097                    82M
    3098                    23M
    3099                    52M
    3100     Varies with device
    3101     Varies with device
    3102                    14M
    3103     Varies with device
    3104     Varies with device
    3105                    25M
    3106     Varies with device
    3107                    58M
    3108                    19M
    3109                    29M
    3110                    81M
    3111     Varies with device
    3112     Varies with device
    3113                    28M
    3114     Varies with device
    3115     Varies with device
    3116     Varies with device
    3117     Varies with device
    3118     Varies with device
    3119                   7.6M
    3120                    33M
    3121     Varies with device
    3122                    42M
    3123                   8.3M
    3124                    14M
    3125     Varies with device
    3126     Varies with device
    3127     Varies with device
    3128                    37M
    3129                    14M
    3130     Varies with device
    3131                    21M
    3132                    28M
    3133                    27M
    3134     Varies with device
    3135                    39M
    3136     Varies with device
    3137                    17M
    3138     Varies with device
    3139                    39M
    3140                    51M
    3141                    86M
    3142     Varies with device
    3143     Varies with device
    3144     Varies with device
    3145                    26M
    3146     Varies with device
    3147                    46M
    3148                    29M
    3149                    27M
    3150                    14M
    3151     Varies with device
    3152                    10M
    3153                    71M
    3154                    15M
    3155     Varies with device
    3156     Varies with device
    3157                    11M
    3158     Varies with device
    3159                    62M
    3160                    80M
    3161                   5.4M
    3162                   9.8M
    3163     Varies with device
    3164     Varies with device
    3165     Varies with device
    3166                   7.6M
    3167                    26M
    3168                    13M
    3169     Varies with device
    3170     Varies with device
    3171                    62M
    3172                  10.0M
    3173                    15M
    3174                    80M
    3175                    14M
    3176     Varies with device
    3177                    57M
    3178                    51M
    3179                   8.3M
    3180                    24M
    3181     Varies with device
    3182                    29M
    3183     Varies with device
    3184                   3.1M
    3185                    46M
    3186                    55M
    3187                    19M
    3188     Varies with device
    3189                    19M
    3190                    40M
    3191     Varies with device
    3192                    29M
    3193                    43M
    3194                    50M
    3195     Varies with device
    3196                    13M
    3197     Varies with device
    3198     Varies with device
    3199     Varies with device
    3200                    15M
    3201                   4.0M
    3202     Varies with device
    3203                    14M
    3204     Varies with device
    3205                    19M
    3206     Varies with device
    3207                    29M
    3208                    44M
    3209                    22M
    3210     Varies with device
    3211     Varies with device
    3212     Varies with device
    3213                    28M
    3214     Varies with device
    3215     Varies with device
    3216                    12M
    3217     Varies with device
    3218                    46M
    3219                   7.6M
    3220     Varies with device
    3221                    29M
    3222     Varies with device
    3223     Varies with device
    3224                   4.1M
    3225                   8.5M
    3226                    31M
    3227                   4.1M
    3228     Varies with device
    3229                    22M
    3230     Varies with device
    3231                    28M
    3232     Varies with device
    3233                   5.9M
    3234     Varies with device
    3235     Varies with device
    3236     Varies with device
    3237                   3.9M
    3238     Varies with device
    3239                    15M
    3240                   4.3M
    3241     Varies with device
    3242     Varies with device
    3243     Varies with device
    3244     Varies with device
    3245                   5.3M
    3246     Varies with device
    3247                    16M
    3248                   7.5M
    3249     Varies with device
    3250     Varies with device
    3251     Varies with device
    3252                   3.7M
    3253     Varies with device
    3254                   5.8M
    3255                    17M
    3256     Varies with device
    3257                   8.5M
    3258     Varies with device
    3259                   9.1M
    3260                   7.6M
    3261                   2.5M
    3262                    15M
    3263     Varies with device
    3264     Varies with device
    3265     Varies with device
    3266     Varies with device
    3267                   6.1M
    3268     Varies with device
    3269                    17M
    3270     Varies with device
    3271     Varies with device
    3272     Varies with device
    3273     Varies with device
    3274                    58M
    3275                   4.6M
    3276                    16M
    3277                   1.3M
    3278                   2.3M
    3279                   1.9M
    3280                   2.7M
    3281                    16M
    3282                    11M
    3283                   5.3M
    3284     Varies with device
    3285                   4.2M
    3286     Varies with device
    3287                   9.1M
    3288                   4.3M
    3289     Varies with device
    3290     Varies with device
    3291     Varies with device
    3292     Varies with device
    3293                   1.8M
    3294     Varies with device
    3295                   8.5M
    3296                   4.4M
    3297     Varies with device
    3298                   3.8M
    3299                    11M
    3300     Varies with device
    3301                    11M
    3302     Varies with device
    3303                   7.4M
    3304                   5.3M
    3305                   5.4M
    3306                   3.6M
    3307     Varies with device
    3308                   4.3M
    3309     Varies with device
    3310                    27M
    3311     Varies with device
    3312                   1.8M
    3313                   3.3M
    3314                   2.0M
    3315                   6.1M
    3316                   9.6M
    3317                   9.6M
    3318                    10M
    3319                   8.2M
    3320                    14M
    3321                    25M
    3322                   9.9M
    3323                    22M
    3324     Varies with device
    3325                   8.9M
    3326     Varies with device
    3327                   2.4M
    3328                    25M
    3329     Varies with device
    3330     Varies with device
    3331                   1.8M
    3332                    17M
    3333     Varies with device
    3334                   4.1M
    3335                    18M
    3336                    14M
    3337     Varies with device
    3338     Varies with device
    3339     Varies with device
    3340                   696k
    3341                   4.7M
    3342                   8.1M
    3343     Varies with device
    3344                   7.9M
    3345                   4.6M
    3346                   544k
    3347                   3.5M
    3348     Varies with device
    3349     Varies with device
    3350                   525k
    3351                   2.1M
    3352     Varies with device
    3353                    14M
    3354     Varies with device
    3355                   8.7M
    3356                    21M
    3357     Varies with device
    3358                    12M
    3359                    18M
    3360                    17M
    3361                   3.8M
    3362     Varies with device
    3363                    23M
    3364                    14M
    3365                    15M
    3366                   9.9M
    3367                    12M
    3368                    12M
    3369                    20M
    3370                   7.4M
    3371                   3.0M
    3372     Varies with device
    3373                    14M
    3374                    14M
    3375                   7.2M
    3376                    22M
    3377                   7.2M
    3378     Varies with device
    3379                    26M
    3380                    38M
    3381                   3.9M
    3382     Varies with device
    3383                   5.4M
    3384                   9.8M
    3385                   7.6M
    3386                   1.1M
    3387                   9.1M
    3388                   6.8M
    3389                    12M
    3390                   3.3M
    3391                    12M
    3392                   9.8M
    3393                   920k
    3394                    14M
    3395                   6.8M
    3396                    27M
    3397                   7.4M
    3398     Varies with device
    3399                    29M
    3400     Varies with device
    3401     Varies with device
    3402                   5.9M
    3403                    14M
    3404                    14M
    3405                    26M
    3406                   1.9M
    3407                   3.9M
    3408                   779k
    3409     Varies with device
    3410     Varies with device
    3411                   5.1M
    3412                    10M
    3413                    13M
    3414                   4.0M
    3415                   5.4M
    3416                   3.3M
    3417                   4.3M
    3418                   853k
    3419                    21M
    3420                   6.4M
    3421                   8.6M
    3422                   2.1M
    3423                   7.1M
    3424     Varies with device
    3425     Varies with device
    3426                   3.3M
    3427                   3.5M
    3428                   4.1M
    3429                   9.7M
    3430                   3.3M
    3431                   7.1M
    3432                   8.4M
    3433                   7.0M
    3434                   5.5M
    3435     Varies with device
    3436     Varies with device
    3437     Varies with device
    3438                    13M
    3439     Varies with device
    3440     Varies with device
    3441                   5.9M
    3442     Varies with device
    3443                    24M
    3444                   6.2M
    3445     Varies with device
    3446     Varies with device
    3447                   6.9M
    3448     Varies with device
    3449                   6.1M
    3450     Varies with device
    3451     Varies with device
    3452     Varies with device
    3453                    24M
    3454     Varies with device
    3455                   4.1M
    3456     Varies with device
    3457                    50M
    3458     Varies with device
    3459     Varies with device
    3460                    11M
    3461                   5.1M
    3462                   1.3M
    3463     Varies with device
    3464     Varies with device
    3465     Varies with device
    3466     Varies with device
    3467     Varies with device
    3468                   2.3M
    3469                    16M
    3470     Varies with device
    3471     Varies with device
    3472                    15M
    3473                    61M
    3474                   1.6M
    3475     Varies with device
    3476     Varies with device
    3477     Varies with device
    3478     Varies with device
    3479                    49M
    3480     Varies with device
    3481                   7.2M
    3482     Varies with device
    3483     Varies with device
    3484                   1.5M
    3485     Varies with device
    3486     Varies with device
    3487                    49M
    3488                    14M
    3489     Varies with device
    3490     Varies with device
    3491     Varies with device
    3492     Varies with device
    3493     Varies with device
    3494                   1.2M
    3495     Varies with device
    3496                  10.0M
    3497                   4.3M
    3498                   3.8M
    3499     Varies with device
    3500                    32M
    3501                   4.5M
    3502     Varies with device
    3503                   8.4M
    3504                    12M
    3505                    25M
    3506     Varies with device
    3507     Varies with device
    3508     Varies with device
    3509                    12M
    3510     Varies with device
    3511                   7.1M
    3512     Varies with device
    3513     Varies with device
    3514                    16M
    3515                   6.5M
    3516                   9.2M
    3517                   720k
    3518                   2.2M
    3519     Varies with device
    3520     Varies with device
    3521                   4.1M
    3522     Varies with device
    3523     Varies with device
    3524     Varies with device
    3525     Varies with device
    3526     Varies with device
    3527     Varies with device
    3528     Varies with device
    3529                    60M
    3530     Varies with device
    3531                    37M
    3532     Varies with device
    3533     Varies with device
    3534     Varies with device
    3535     Varies with device
    3536     Varies with device
    3537                   713k
    3538                   4.7M
    3539                   6.6M
    3540                   2.5M
    3541                    19M
    3542     Varies with device
    3543                   4.2M
    3544                    11M
    3545     Varies with device
    3546     Varies with device
    3547     Varies with device
    3548     Varies with device
    3549                    12M
    3550     Varies with device
    3551     Varies with device
    3552                   5.7M
    3553                   9.0M
    3554     Varies with device
    3555                   6.9M
    3556                   8.4M
    3557                   1.9M
    3558     Varies with device
    3559     Varies with device
    3560     Varies with device
    3561                    15M
    3562     Varies with device
    3563     Varies with device
    3564     Varies with device
    3565     Varies with device
    3566     Varies with device
    3567     Varies with device
    3568                    11M
    3569                    61M
    3570                   4.1M
    3571                   3.8M
    3572     Varies with device
    3573                   3.8M
    3574     Varies with device
    3575                   2.8M
    3576     Varies with device
    3577                   8.2M
    3578                   3.3M
    3579                   6.2M
    3580                    85M
    3581                    14M
    3582                    95M
    3583                   5.5M
    3584                    42M
    3585                    15M
    3586                    98M
    3587                   6.4M
    3588                   2.8M
    3589                   6.5M
    3590                   3.6M
    3591                   2.4M
    3592                   3.4M
    3593                   1.5M
    3594                   5.0M
    3595                   6.6M
    3596     Varies with device
    3597     Varies with device
    3598     Varies with device
    3599                    59M
    3600                   6.8M
    3601                    24M
    3602                   6.6M
    3603                   4.9M
    3604                   9.1M
    3605                    11M
    3606                   5.8M
    3607                    24M
    3608                   4.4M
    3609                    16M
    3610                    11M
    3611                   4.9M
    3612                    11M
    3613                    53M
    3614                    20M
    3615                    38M
    3616                   4.9M
    3617                    17M
    3618                    37M
    3619                   5.2M
    3620                    28M
    3621                    71M
    3622                    46M
    3623                    16M
    3624                    18M
    3625                    18M
    3626     Varies with device
    3627                    10M
    3628     Varies with device
    3629                    17M
    3630     Varies with device
    3631                   9.7M
    3632     Varies with device
    3633     Varies with device
    3634                    21M
    3635                   3.2M
    3636                    15M
    3637                    44M
    3638                   4.8M
    3639     Varies with device
    3640                    10M
    3641                    54M
    3642                   9.1M
    3643                    19M
    3644                    12M
    3645     Varies with device
    3646     Varies with device
    3647     Varies with device
    3648                    19M
    3649     Varies with device
    3650                   7.6M
    3651     Varies with device
    3652                    38M
    3653     Varies with device
    3654     Varies with device
    3655                   6.1M
    3656                   9.2M
    3657     Varies with device
    3658                    11M
    3659                   9.2M
    3660                    20M
    3661                    22M
    3662     Varies with device
    3663     Varies with device
    3664                   5.3M
    3665     Varies with device
    3666                   5.6M
    3667                   5.4M
    3668                   2.9M
    3669                    25M
    3670     Varies with device
    3671     Varies with device
    3672                    23M
    3673                   7.2M
    3674                   3.3M
    3675     Varies with device
    3676     Varies with device
    3677                    17M
    3678                   6.0M
    3679                    33M
    3680                   3.1M
    3681                   4.1M
    3682                    27M
    3683                    44M
    3684     Varies with device
    3685                    13M
    3686     Varies with device
    3687     Varies with device
    3688                   3.0M
    3689     Varies with device
    3690     Varies with device
    3691                   6.1M
    3692                    64M
    3693                    47M
    3694     Varies with device
    3695                   5.4M
    3696     Varies with device
    3697     Varies with device
    3698     Varies with device
    3699                    27M
    3700     Varies with device
    3701                    50M
    3702     Varies with device
    3703     Varies with device
    3704                   4.0M
    3705                    13M
    3706     Varies with device
    3707     Varies with device
    3708                    44M
    3709                   2.5M
    3710                    23M
    3711     Varies with device
    3712                    23M
    3713                    14M
    3714                    89M
    3715     Varies with device
    3716     Varies with device
    3717                    16M
    3718                    26M
    3719     Varies with device
    3720                    11M
    3721     Varies with device
    3722     Varies with device
    3723                    56M
    3724                    25M
    3725     Varies with device
    3726                    17M
    3727                   9.1M
    3728                    11M
    3729     Varies with device
    3730                    15M
    3731                   9.7M
    3732                   5.5M
    3733                   6.7M
    3734     Varies with device
    3735                   2.9M
    3736                    13M
    3737                   8.5M
    3738                    25M
    3739     Varies with device
    3740                    12M
    3741                   6.3M
    3742                   6.3M
    3743     Varies with device
    3744                    19M
    3745                    31M
    3746     Varies with device
    3747                   9.0M
    3748                    25M
    3749                   4.6M
    3750                   8.0M
    3751                    23M
    3752                   9.8M
    3753                   8.7M
    3754     Varies with device
    3755     Varies with device
    3756                    12M
    3757     Varies with device
    3758                    10M
    3759     Varies with device
    3760                    19M
    3761                   6.6M
    3762     Varies with device
    3763                    14M
    3764                   8.6M
    3765                    13M
    3766                    13M
    3767     Varies with device
    3768     Varies with device
    3769                    10M
    3770     Varies with device
    3771     Varies with device
    3772     Varies with device
    3773                   7.5M
    3774                    36M
    3775                   8.2M
    3776                    22M
    3777                    23M
    3778     Varies with device
    3779     Varies with device
    3780     Varies with device
    3781                    14M
    3782     Varies with device
    3783                    25M
    3784                    27M
    3785     Varies with device
    3786     Varies with device
    3787                    35M
    3788     Varies with device
    3789                   8.6M
    3790     Varies with device
    3791                   4.6M
    3792                    25M
    3793                    12M
    3794                    18M
    3795     Varies with device
    3796     Varies with device
    3797                   3.1M
    3798     Varies with device
    3799                    14M
    3800     Varies with device
    3801     Varies with device
    3802     Varies with device
    3803                    23M
    3804     Varies with device
    3805                   8.8M
    3806                    35M
    3807     Varies with device
    3808     Varies with device
    3809                    36M
    3810     Varies with device
    3811                    25M
    3812                    12M
    3813     Varies with device
    3814                    23M
    3815     Varies with device
    3816                    13M
    3817     Varies with device
    3818                    13M
    3819     Varies with device
    3820     Varies with device
    3821     Varies with device
    3822     Varies with device
    3823                    22M
    3824     Varies with device
    3825     Varies with device
    3826                    43M
    3827     Varies with device
    3828                    33M
    3829                    32M
    3830     Varies with device
    3831                   6.8M
    3832                   4.0M
    3833     Varies with device
    3834                   3.7M
    3835     Varies with device
    3836                   3.4M
    3837     Varies with device
    3838     Varies with device
    3839     Varies with device
    3840                    48M
    3841                   7.7M
    3842                    60M
    3843                    60M
    3844                   5.6M
    3845     Varies with device
    3846                   5.2M
    3847                    25M
    3848                   3.3M
    3849                    29M
    3850                    49M
    3851                   3.6M
    3852                    48M
    3853     Varies with device
    3854                   5.4M
    3855                    18M
    3856                   3.2M
    3857                    11M
    3858                   8.5M
    3859                    24M
    3860                    26M
    3861                    14M
    3862     Varies with device
    3863     Varies with device
    3864                    78M
    3865     Varies with device
    3866     Varies with device
    3867                   7.0M
    3868                   9.2M
    3869     Varies with device
    3870     Varies with device
    3871                   4.9M
    3872                   6.5M
    3873                    11M
    3874                    96M
    3875                   2.9M
    3876                    24M
    3877                    55M
    3878                    74M
    3879                    24M
    3880                    32M
    3881                    15M
    3882                   6.3M
    3883                    97M
    3884                    74M
    3885     Varies with device
    3886                    14M
    3887                    14M
    3888     Varies with device
    3889     Varies with device
    3890                   5.7M
    3891                    19M
    3892     Varies with device
    3893     Varies with device
    3894                    50M
    3895     Varies with device
    3896                    76M
    3897                    70M
    3898                    29M
    3899                    16M
    3900     Varies with device
    3901                    13M
    3902                    73M
    3903                    17M
    3904     Varies with device
    3905     Varies with device
    3906     Varies with device
    3907     Varies with device
    3908                    85M
    3909     Varies with device
    3910                    99M
    3911     Varies with device
    3912                    92M
    3913                    37M
    3914                    45M
    3915     Varies with device
    3916                    57M
    3917                    26M
    3918                    47M
    3919                    22M
    3920                    56M
    3921                    50M
    3922                    42M
    3923                    34M
    3924     Varies with device
    3925                    70M
    3926                    17M
    3927                   4.1M
    3928     Varies with device
    3929                    11M
    3930                    11M
    3931                    94M
    3932                    24M
    3933                    56M
    3934     Varies with device
    3935                    66M
    3936                    63M
    3937                    91M
    3938                    17M
    3939                    21M
    3940     Varies with device
    3941     Varies with device
    3942                    81M
    3943     Varies with device
    3944                   3.0M
    3945                    59M
    3946     Varies with device
    3947                    83M
    3948                    19M
    3949                    23M
    3950                    32M
    3951                    39M
    3952                    57M
    3953                    52M
    3954                   7.5M
    3955                   7.4M
    3956                   8.9M
    3957                   6.3M
    3958                    18M
    3959                    14M
    3960     Varies with device
    3961                    20M
    3962     Varies with device
    3963                    96M
    3964                    35M
    3965                   8.4M
    3966                   8.6M
    3967                    55M
    3968                    29M
    3969                    13M
    3970                    16M
    3971                    40M
    3972                   6.7M
    3973                   100M
    3974     Varies with device
    3975                    94M
    3976                   4.4M
    3977                    64M
    3978                   4.5M
    3979                   6.0M
    3980     Varies with device
    3981                   7.4M
    3982                   3.7M
    3983                    25M
    3984                    10M
    3985                    26M
    3986                    98M
    3987                    97M
    3988                   1.8M
    3989     Varies with device
    3990                    11M
    3991                    78M
    3992                    50M
    3993     Varies with device
    3994                    74M
    3995                   8.3M
    3996     Varies with device
    3997                   7.7M
    3998                   2.5M
    3999                   5.3M
    4000     Varies with device
    4001                   4.0M
    4002                   4.2M
    4003                    74M
    4004                   6.9M
    4005     Varies with device
    4006                   6.6M
    4007                    16M
    4008                   4.7M
    4009                   5.4M
    4010                    35M
    4011                    26M
    4012                    32M
    4013                   2.7M
    4014                   3.7M
    4015                   7.0M
    4016                   6.9M
    4017                    63M
    4018                   2.2M
    4019                    30M
    4020                   5.8M
    4021                   3.1M
    4022                   1.5M
    4023                   8.6M
    4024                   772k
    4025                   3.3M
    4026                   5.0M
    4027                   1.1M
    4028                   4.4M
    4029                   2.1M
    4030                   1.6M
    4031     Varies with device
    4032                    29M
    4033                    39M
    4034                    29M
    4035                    14M
    4036     Varies with device
    4037                    41M
    4038                   9.7M
    4039     Varies with device
    4040                    60M
    4041                    62M
    4042                    47M
    4043                    89M
    4044                   9.9M
    4045                    64M
    4046                    48M
    4047                    12M
    4048                    61M
    4049                    32M
    4050                    28M
    4051     Varies with device
    4052     Varies with device
    4053                    36M
    4054                    13M
    4055                    69M
    4056                    74M
    4057                    18M
    4058                   3.6M
    4059                    79M
    4060                    66M
    4061                    42M
    4062                   9.8M
    4063                   2.8M
    4064     Varies with device
    4065                    40M
    4066                    20M
    4067                    25M
    4068     Varies with device
    4069     Varies with device
    4070                    44M
    4071                   8.8M
    4072                    16M
    4073                    89M
    4074     Varies with device
    4075                    58M
    4076                    20M
    4077                   4.8M
    4078                   318k
    4079     Varies with device
    4080                    13M
    4081                   4.9M
    4082                    12M
    4083     Varies with device
    4084     Varies with device
    4085     Varies with device
    4086                    66M
    4087                    50M
    4088     Varies with device
    4089                    59M
    4090     Varies with device
    4091                    14M
    4092                   5.9M
    4093                    41M
    4094     Varies with device
    4095                    37M
    4096     Varies with device
    4097     Varies with device
    4098     Varies with device
    4099     Varies with device
    4100                    22M
    4101     Varies with device
    4102                    15M
    4103                   6.1M
    4104     Varies with device
    4105     Varies with device
    4106     Varies with device
    4107                   3.1M
    4108                    58k
    4109                    79M
    4110                    15M
    4111                    55M
    4112                    11M
    4113                    13M
    4114                   4.9M
    4115     Varies with device
    4116                   7.6M
    4117                   3.0M
    4118                    37M
    4119                   9.3M
    4120                    23M
    4121                   241k
    4122     Varies with device
    4123                    11M
    4124                   9.2M
    4125                   4.8M
    4126     Varies with device
    4127                    16M
    4128                    40M
    4129                    13M
    4130                   4.9M
    4131                    31M
    4132     Varies with device
    4133     Varies with device
    4134                    14M
    4135                   3.0M
    4136                   9.9M
    4137     Varies with device
    4138                   3.7M
    4139                   1.7M
    4140                    11M
    4141                    63M
    4142                   2.1M
    4143                   1.4M
    4144     Varies with device
    4145                    36M
    4146                    32M
    4147     Varies with device
    4148                    59M
    4149                    37M
    4150     Varies with device
    4151                   7.9M
    4152                    25M
    4153     Varies with device
    4154                    19M
    4155                    39M
    4156                   196k
    4157                   2.3M
    4158                   8.1M
    4159                    51M
    4160     Varies with device
    4161                   3.4M
    4162                   8.8M
    4163                    28M
    4164     Varies with device
    4165                    11M
    4166                   6.1M
    4167                    14M
    4168                    94M
    4169                    11M
    4170     Varies with device
    4171                    14M
    4172                   1.6M
    4173                    15M
    4174                   2.7M
    4175                   5.7M
    4176                    47M
    4177                   1.6M
    4178                   857k
    4179                    25M
    4180                   1.3M
    4181                   1.5M
    4182                    33M
    4183                   3.9M
    4184                    87M
    4185                    14M
    4186                    10M
    4187                   5.9M
    4188                    58M
    4189                   1.9M
    4190                    48M
    4191                    57M
    4192                    51k
    4193                   4.5M
    4194                   8.9M
    4195                    26M
    4196                    15M
    4197                   1.5M
    4198                   2.2M
    4199                    42M
    4200                   5.6M
    4201                   8.5M
    4202                   8.9M
    4203                    13M
    4204                    13M
    4205     Varies with device
    4206                   5.4M
    4207                    27M
    4208                   3.0M
    4209                    19M
    4210                   8.3M
    4211                    57M
    4212                   1.2M
    4213                   2.8M
    4214                    19M
    4215                   7.2M
    4216     Varies with device
    4217                    52M
    4218     Varies with device
    4219                    65M
    4220                    37M
    4221                   2.9M
    4222                    11M
    4223                    34M
    4224     Varies with device
    4225                   6.5M
    4226                   3.9M
    4227     Varies with device
    4228                   3.9M
    4229                    57M
    4230                   2.8M
    4231                    23M
    4232     Varies with device
    4233                    55M
    4234     Varies with device
    4235                    75M
    4236                    40M
    4237                    50M
    4238     Varies with device
    4239                    25M
    4240                    25M
    4241     Varies with device
    4242                   6.6M
    4243                    36M
    4244                    75M
    4245                   8.8M
    4246                    43M
    4247     Varies with device
    4248                   9.9M
    4249                    96M
    4250                   953k
    4251                   9.0M
    4252                    23M
    4253                   2.6M
    4254                    23M
    4255                    35M
    4256                   7.2M
    4257                    49M
    4258                   4.7M
    4259     Varies with device
    4260                    43M
    4261                    97M
    4262     Varies with device
    4263                   3.1M
    4264                   6.7M
    4265                    13M
    4266                   7.1M
    4267                   9.1M
    4268                    98M
    4269                    30M
    4270                   8.9M
    4271                    13M
    4272                   2.4M
    4273                    44M
    4274                   7.3M
    4275                    33M
    4276                   3.4M
    4277                   3.8M
    4278                   3.6M
    4279     Varies with device
    4280                    40M
    4281                    14M
    4282     Varies with device
    4283     Varies with device
    4284     Varies with device
    4285                    34M
    4286                    40M
    4287     Varies with device
    4288                    11M
    4289                    63M
    4290                    14M
    4291                    61M
    4292                    63M
    4293                   8.9M
    4294                    10M
    4295                    44M
    4296                    10M
    4297                    59M
    4298                   7.4M
    4299                    23M
    4300                    50M
    4301                    44M
    4302     Varies with device
    4303                    15M
    4304                    40M
    4305                    62M
    4306                   2.1M
    4307                    31M
    4308                    19M
    4309                    26M
    4310                   3.6M
    4311                    13M
    4312                   4.5M
    4313                    35M
    4314                    71M
    4315                    15M
    4316                    23M
    4317                   3.5M
    4318                    60M
    4319                    40M
    4320     Varies with device
    4321                    78M
    4322                    20M
    4323     Varies with device
    4324                    49M
    4325                    40M
    4326                    48M
    4327                    87M
    4328                    20M
    4329                    50M
    4330                   865k
    4331                   2.9M
    4332                    67M
    4333                   2.4M
    4334                    61M
    4335                   6.8M
    4336                   2.6M
    4337                    13M
    4338                    29M
    4339                    84M
    4340                   5.4M
    4341                   2.8M
    4342                   2.9M
    4343                   6.9M
    4344                    46M
    4345                    27M
    4346     Varies with device
    4347     Varies with device
    4348                    37M
    4349                    82M
    4350                    44M
    4351                   7.3M
    4352                    37M
    4353                    53M
    4354                    44M
    4355     Varies with device
    4356                    24M
    4357                    34M
    4358     Varies with device
    4359                   9.1M
    4360                    38M
    4361                    15M
    4362                    26M
    4363                    63M
    4364                   251k
    4365     Varies with device
    4366                   4.2M
    4367                   7.3M
    4368                    49M
    4369                   9.7M
    4370                   1.4M
    4371                   4.9M
    4372                   6.2M
    4373                   6.0M
    4374     Varies with device
    4375                   1.2M
    4376                    16M
    4377                    59M
    4378                   7.7M
    4379                    30M
    4380                   6.4M
    4381                    64M
    4382                    46M
    4383                    76M
    4384                    54M
    4385                   8.4M
    4386                    22M
    4387                    23M
    4388                   8.2M
    4389                   3.8M
    4390                    54M
    4391                   9.1M
    4392                    40M
    4393                    28M
    4394                    11M
    4395                    34M
    4396                    58M
    4397                    21M
    4398                    45M
    4399                    22M
    4400                    28M
    4401                    17M
    4402                    31M
    4403                    59M
    4404                    91M
    4405     Varies with device
    4406                    20M
    4407                    69M
    4408                    71M
    4409                    13M
    4410                    33M
    4411                   3.3M
    4412                   6.4M
    4413                    20M
    4414                   7.6M
    4415                    62M
    4416                    32M
    4417                    49M
    4418                   930k
    4419                    34M
    4420                    49M
    4421     Varies with device
    4422     Varies with device
    4423                    24M
    4424                    72M
    4425                   8.7M
    4426     Varies with device
    4427                    40M
    4428                   6.8M
    4429                    47M
    4430                   3.6M
    4431                   2.9M
    4432                    26M
    4433                   1.3M
    4434                    45M
    4435                   540k
    4436                   4.6M
    4437                   313k
    4438                   8.6M
    4439                   746k
    4440                    10M
    4441                    38M
    4442                   5.1M
    4443                   6.1M
    4444                    28M
    4445                   5.2M
    4446                    22M
    4447     Varies with device
    4448                    34M
    4449                    51M
    4450                    39M
    4451                   2.1M
    4452                    12M
    4453                    11M
    4454                   7.5M
    4455                    24M
    4456                    19M
    4457                   1.9M
    4458                   5.2M
    4459                   2.3M
    4460                   4.5M
    4461                   1.7M
    4462                   1.8M
    4463                   6.3M
    4464     Varies with device
    4465                   5.5M
    4466                   6.2M
    4467                    28M
    4468                   4.3M
    4469                   5.7M
    4470                   4.9M
    4471                   3.6M
    4472                   2.0M
    4473                    16M
    4474                   7.4M
    4475                   4.3M
    4476                   7.3M
    4477                    60M
    4478                   3.7M
    4479                    24M
    4480                    10M
    4481                   7.5M
    4482                    13M
    4483                   3.6M
    4484                   2.0M
    4485                    14M
    4486                   2.3M
    4487                    16M
    4488                    20M
    4489                    21M
    4490                   2.1M
    4491                    26M
    4492                    14M
    4493                    18M
    4494                    60M
    4495     Varies with device
    4496                   7.1M
    4497     Varies with device
    4498                    60M
    4499                    40M
    4500                    55M
    4501     Varies with device
    4502                    20M
    4503                    19M
    4504     Varies with device
    4505                    17M
    4506                   3.6M
    4507                   4.5M
    4508                   3.6M
    4509                    33M
    4510                    43M
    4511                    21M
    4512                   2.3M
    4513                    14M
    4514                    19M
    4515                    95M
    4516                    29M
    4517                   2.7M
    4518     Varies with device
    4519     Varies with device
    4520                    16M
    4521                   3.9M
    4522                   3.1M
    4523                    12M
    4524                    39M
    4525                    26M
    4526                    19M
    4527                    67M
    4528                    12M
    4529                   2.8M
    4530                   6.2M
    4531                   2.9M
    4532                   5.9M
    4533                    60M
    4534                   4.9M
    4535                    62M
    4536                    96M
    4537                   9.2M
    4538                   2.8M
    4539                   3.9M
    4540                   2.2M
    4541                   203k
    4542                    37M
    4543                   3.7M
    4544                   7.7M
    4545                   2.9M
    4546                    29M
    4547                    34M
    4548                   3.6M
    4549                   5.7M
    4550                    64M
    4551                   3.4M
    4552                    58k
    4553                   6.7M
    4554                    23M
    4555                   8.9M
    4556                    57M
    4557                    12M
    4558                    13M
    4559                    93M
    4560                    83M
    4561                    41M
    4562                    28M
    4563                   2.7M
    4564                   4.4M
    4565                   3.4M
    4566                    17M
    4567                   2.2M
    4568     Varies with device
    4569                   7.1M
    4570                    20M
    4571                    26k
    4572     Varies with device
    4573                    31M
    4574                    45M
    4575                    33M
    4576                    14M
    4577                    37M
    4578                    24M
    4579                   6.5M
    4580                    40M
    4581                    93M
    4582                   314k
    4583                   3.7M
    4584                    57M
    4585                   2.7M
    4586     Varies with device
    4587                    68M
    4588                    25M
    4589     Varies with device
    4590     Varies with device
    4591                   2.4M
    4592     Varies with device
    4593     Varies with device
    4594                    96M
    4595     Varies with device
    4596                   2.2M
    4597                    33M
    4598     Varies with device
    4599                    19M
    4600     Varies with device
    4601                    21M
    4602     Varies with device
    4603                    26M
    4604                    29M
    4605     Varies with device
    4606                   3.7M
    4607                    25M
    4608                    28M
    4609                   2.1M
    4610     Varies with device
    4611                    15M
    4612                   7.9M
    4613                    46M
    4614                   8.9M
    4615                    13M
    4616                    20M
    4617     Varies with device
    4618                   2.3M
    4619                   4.9M
    4620                    23M
    4621                    35M
    4622                    18M
    4623                    45M
    4624                   5.6M
    4625                   4.5M
    4626                   5.8M
    4627                    24M
    4628                    21M
    4629                    31M
    4630                    74M
    4631                    21M
    4632                   7.3M
    4633                   3.3M
    4634     Varies with device
    4635                    11M
    4636                    53M
    4637                    37M
    4638                    37M
    4639                   2.0M
    4640                    26M
    4641                    42M
    4642                   7.8M
    4643                   3.5M
    4644                    14M
    4645                   5.1M
    4646                   1.9M
    4647                    38M
    4648                   3.0M
    4649                   2.2M
    4650                   3.5M
    4651                   4.0M
    4652                    27M
    4653                    13M
    4654                    21M
    4655                   3.5M
    4656                    22M
    4657                   7.2M
    4658                   4.8M
    4659                    13M
    4660                   7.5M
    4661                   5.4M
    4662                    30M
    4663     Varies with device
    4664                   196k
    4665     Varies with device
    4666                    16M
    4667                   3.8M
    4668                    17M
    4669                    26M
    4670                    28M
    4671     Varies with device
    4672                    38M
    4673                    17M
    4674                    20M
    4675                   6.9M
    4676     Varies with device
    4677                    59M
    4678                    67M
    4679                    61M
    4680     Varies with device
    4681     Varies with device
    4682                   1.6M
    4683     Varies with device
    4684                   3.7M
    4685                   4.9M
    4686     Varies with device
    4687                    30M
    4688                    40M
    4689                    10M
    4690                   100M
    4691                    42M
    4692                    95M
    4693                    28M
    4694                   4.2M
    4695                   3.6M
    4696     Varies with device
    4697                    17M
    4698                   4.6M
    4699                   7.9M
    4700     Varies with device
    4701                   4.3M
    4702                   3.6M
    4703                   5.9M
    4704                    35M
    4705                   1.8M
    4706                   9.5M
    4707                    19M
    4708                    25M
    4709                    37M
    4710                   3.6M
    4711                    17M
    4712     Varies with device
    4713                   2.9M
    4714                   3.4M
    4715     Varies with device
    4716                   2.9M
    4717     Varies with device
    4718     Varies with device
    4719                   3.5M
    4720                    37M
    4721                   1.5M
    4722     Varies with device
    4723                    47M
    4724     Varies with device
    4725                    11M
    4726                   9.1M
    4727                    32M
    4728                    16M
    4729                   2.5M
    4730                   6.7M
    4731                   1.9M
    4732     Varies with device
    4733                    27M
    4734                    79k
    4735                   7.6M
    4736                   8.2M
    4737                    33M
    4738                   3.1M
    4739                    27M
    4740     Varies with device
    4741                   1.5M
    4742     Varies with device
    4743                   4.5M
    4744     Varies with device
    4745                    43M
    4746                    21M
    4747     Varies with device
    4748     Varies with device
    4749                   4.4M
    4750                    16M
    4751                    13M
    4752                    34M
    4753                   8.3M
    4754                   2.5M
    4755                   3.5M
    4756                   6.4M
    4757                   4.3M
    4758                   6.3M
    4759                   8.6M
    4760                   1.9M
    4761                   118k
    4762                    49M
    4763                   3.7M
    4764                   3.5M
    4765                    53M
    4766                   8.0M
    4767     Varies with device
    4768                    27M
    4769                    24M
    4770                    11M
    4771                   3.2M
    4772                   3.7M
    4773                   1.2M
    4774                   8.4M
    4775                    31M
    4776                   9.6M
    4777                    45M
    4778                    56M
    4779                    26M
    4780                    17M
    4781                   6.9M
    4782     Varies with device
    4783                    99M
    4784                    47M
    4785                    36M
    4786                    21M
    4787                   3.0M
    4788                   8.5M
    4789                   2.0M
    4790                    15M
    4791                    50M
    4792                    17M
    4793     Varies with device
    4794                   1.9M
    4795                    70M
    4796     Varies with device
    4797                    41M
    4798                    99M
    4799                    96M
    4800                    16M
    4801                    84M
    4802                   5.0M
    4803                   5.0M
    4804     Varies with device
    4805     Varies with device
    4806                    99M
    4807                    99M
    4808                    26M
    4809     Varies with device
    4810     Varies with device
    4811     Varies with device
    4812     Varies with device
    4813                    37M
    4814     Varies with device
    4815                   9.0M
    4816                    88M
    4817                    87M
    4818                    78M
    4819                    67M
    4820                    59M
    4821     Varies with device
    4822                   1.4M
    4823                    59M
    4824                    47M
    4825                    76M
    4826                    62M
    4827                    80M
    4828                   3.8M
    4829                    56M
    4830                    72M
    4831                    25M
    4832                    79M
    4833                    98M
    4834                    68M
    4835                    91M
    4836                    44M
    4837                    96M
    4838                    61M
    4839     Varies with device
    4840                    19M
    4841                   8.8M
    4842                    96M
    4843                   7.7M
    4844     Varies with device
    4845                    10M
    4846                   2.2M
    4847                    16M
    4848                   5.7M
    4849                    29M
    4850                    24M
    4851                   4.3M
    4852                    48M
    4853                   2.3M
    4854                    13M
    4855                   7.6M
    4856                    99M
    4857                    22M
    4858                    12M
    4859                    88M
    4860                    91M
    4861                    99M
    4862                    10M
    4863     Varies with device
    4864                    57M
    4865                    18M
    4866                    33M
    4867                   8.2M
    4868     Varies with device
    4869                    46M
    4870                    63M
    4871                   239k
    4872     Varies with device
    4873     Varies with device
    4874                    48M
    4875     Varies with device
    4876                    48M
    4877                    18M
    4878                   8.3M
    4879                    54M
    4880     Varies with device
    4881                    30M
    4882                   8.3M
    4883                    23M
    4884                    11M
    4885                   8.4M
    4886                    82M
    4887                   5.0M
    4888                    17M
    4889                    37M
    4890                    45M
    4891                    40M
    4892                   7.0M
    4893                   3.8M
    4894                    22M
    4895                    15M
    4896                   8.5M
    4897                   371k
    4898                   5.2M
    4899     Varies with device
    4900                    13M
    4901                    92M
    4902                   2.0M
    4903                   5.0M
    4904                   1.8M
    4905                   1.7M
    4906                    24M
    4907                    23M
    4908                   5.1M
    4909                   7.9M
    4910                   7.8M
    4911                   4.2M
    4912                    18M
    4913                   3.3M
    4914                   7.7M
    4915                   3.1M
    4916                    27M
    4917                   220k
    4918                    26M
    4919                   3.0M
    4920                    28M
    4921                    11M
    4922                   6.1M
    4923                    26M
    4924                   5.0M
    4925                   2.1M
    4926                   6.1M
    4927                   2.2M
    4928                    14M
    4929                   2.3M
    4930                    28M
    4931                   3.0M
    4932                   9.7M
    4933                   3.4M
    4934                   3.9M
    4935                   9.4M
    4936                    26M
    4937                    27M
    4938                   4.4M
    4939                    26M
    4940                   3.2M
    4941                   7.0M
    4942     Varies with device
    4943                   1.4M
    4944                   1.7M
    4945     Varies with device
    4946     Varies with device
    4947                   1.5M
    4948                    39M
    4949                   6.1M
    4950                   4.3M
    4951     Varies with device
    4952                   8.3M
    4953                   730k
    4954                   2.1M
    4955                    26M
    4956                   756k
    4957                   3.2M
    4958                   4.1M
    4959                    20M
    4960                   3.6M
    4961                    44M
    4962                    26M
    4963                    17M
    4964                   1.1M
    4965                   2.9M
    4966                    24M
    4967                    14M
    4968                   6.9M
    4969                    12M
    4970                    91k
    4971                   9.4M
    4972                   1.7M
    4973                   293k
    4974     Varies with device
    4975                    35M
    4976                   3.8M
    4977                    17k
    4978                    74k
    4979                    20M
    4980                   1.1M
    4981                   4.1M
    4982                    13M
    4983                    14k
    4984                   8.3M
    4985                   9.0M
    4986                   9.0M
    4987     Varies with device
    4988                   3.3M
    4989     Varies with device
    4990                    28M
    4991                    19M
    4992                    59M
    4993                   1.9M
    4994                    12M
    4995                   1.8M
    4996                    30M
    4997     Varies with device
    4998     Varies with device
    4999     Varies with device
    5000                    18M
    5001     Varies with device
    5002                    29M
    5003                    26M
    5004                    12M
    5005                    12M
    5006                    17M
    5007                    18M
    5008                    26M
    5009     Varies with device
    5010                    17M
    5011                    12M
    5012                   317k
    5013                   7.5M
    5014     Varies with device
    5015                    66M
    5016                    30M
    5017                    35M
    5018                    26M
    5019                    15M
    5020                    27M
    5021                    39M
    5022                   9.1M
    5023                    13M
    5024                    35M
    5025                   5.1M
    5026                   6.7M
    5027                    37M
    5028                   3.9M
    5029                   4.0M
    5030                   7.6M
    5031     Varies with device
    5032                   6.6M
    5033                    32M
    5034                   2.9M
    5035                    78k
    5036                   1.2M
    5037                    15M
    5038                    18M
    5039                    15M
    5040                    19M
    5041                   4.1M
    5042                   5.6M
    5043                   3.5M
    5044                   6.4M
    5045                   3.0M
    5046                   924k
    5047                    11M
    5048                   4.9M
    5049     Varies with device
    5050                   6.1M
    5051                   902k
    5052                    53M
    5053                   1.3M
    5054                   4.7M
    5055                   3.1M
    5056                    11M
    5057     Varies with device
    5058     Varies with device
    5059                   3.7M
    5060                   266k
    5061                    19M
    5062                   2.0M
    5063                   4.1M
    5064                   3.4M
    5065                    13M
    5066                    46M
    5067                   3.9M
    5068                    11M
    5069     Varies with device
    5070                   3.5M
    5071     Varies with device
    5072                    11M
    5073     Varies with device
    5074                    17M
    5075                    17M
    5076                    11M
    5077     Varies with device
    5078                    64M
    5079                    10M
    5080                    35M
    5081                    12M
    5082                   2.4M
    5083                   3.7M
    5084                   6.3M
    5085                    22M
    5086                  10.0M
    5087                   3.7M
    5088                   4.6M
    5089                    44M
    5090                    18M
    5091                   9.7M
    5092                   2.7M
    5093                    51M
    5094                    47M
    5095                    56M
    5096     Varies with device
    5097                    24M
    5098     Varies with device
    5099                    52M
    5100                   4.9M
    5101                   1.7M
    5102                    16M
    5103                   6.5M
    5104                    27M
    5105                    36M
    5106                   5.2M
    5107                    43M
    5108                   8.6M
    5109                   2.0M
    5110                    14M
    5111                    15M
    5112                    12M
    5113                    45M
    5114                   3.2M
    5115     Varies with device
    5116                    62M
    5117                    15M
    5118                    13M
    5119                    18M
    5120                   1.7M
    5121                    15M
    5122                   2.9M
    5123                   5.9M
    5124                   4.2M
    5125                    74M
    5126                   1.7M
    5127                    47M
    5128                   6.9M
    5129                   8.7M
    5130                   818k
    5131                    28M
    5132                    29M
    5133                    59M
    5134     Varies with device
    5135                    29M
    5136                    29M
    5137     Varies with device
    5138                    59M
    5139                    27M
    5140                    29M
    5141                    29M
    5142                    14M
    5143                    10M
    5144                    81k
    5145                    28M
    5146                   9.8M
    5147                    29M
    5148                    28M
    5149                   318k
    5150                   5.4M
    5151                    28M
    5152                    29M
    5153                    29M
    5154                    29M
    5155                   3.0M
    5156                    27M
    5157                    29M
    5158                    86M
    5159                    22M
    5160                    29M
    5161                    29M
    5162                    29M
    5163                    16M
    5164                    27M
    5165                    29M
    5166                   5.7M
    5167                   5.7M
    5168                    29M
    5169                   2.0M
    5170                    29M
    5171                    20M
    5172                    28M
    5173                    27M
    5174                    29M
    5175                   9.1M
    5176                    29M
    5177                    29M
    5178                    35M
    5179                   5.9M
    5180                   1.2M
    5181                    39M
    5182                    27M
    5183                   3.2M
    5184                   4.2M
    5185                    31M
    5186                   3.9M
    5187                    25M
    5188     Varies with device
    5189                    69M
    5190                   4.3M
    5191                   9.8M
    5192                    13M
    5193                    16M
    5194                   4.2M
    5195                    99M
    5196                   2.3M
    5197                   3.6M
    5198                    26M
    5199                   3.2M
    5200                   2.8M
    5201                   8.3M
    5202     Varies with device
    5203                   3.8M
    5204                   3.4M
    5205                   3.5M
    5206                   1.9M
    5207                   3.3M
    5208     Varies with device
    5209     Varies with device
    5210                   7.1M
    5211                   7.9M
    5212                   1.4M
    5213                   1.8M
    5214                   7.5M
    5215                   4.3M
    5216                   2.8M
    5217                   7.9M
    5218                   3.7M
    5219                    48M
    5220                    14M
    5221                   5.7M
    5222                   939k
    5223                    58M
    5224                    42M
    5225                    35M
    5226                    43M
    5227                    63M
    5228                    60M
    5229     Varies with device
    5230                   2.8M
    5231                   3.2M
    5232                   4.2M
    5233                    32M
    5234                    22M
    5235                   7.3M
    5236                   5.3M
    5237                    50M
    5238                   7.5M
    5239                    26M
    5240                    17M
    5241                   4.2M
    5242                    31M
    5243                   7.6M
    5244                   1.7M
    5245                   169k
    5246                    63M
    5247                   7.7M
    5248                    45k
    5249                   6.6M
    5250                    14M
    5251                    22M
    5252                   3.6M
    5253                   8.4M
    5254                    25M
    5255                   475k
    5256                   4.3M
    5257                    35M
    5258                   3.6M
    5259                    24M
    5260                    35M
    5261                    91M
    5262                    27M
    5263                    31M
    5264                   2.7M
    5265                    32M
    5266                    27M
    5267                    67M
    5268                   8.2M
    5269                    30M
    5270                   3.9M
    5271                   1.1M
    5272                    48M
    5273                    14M
    5274                    18M
    5275                   2.9M
    5276                   4.4M
    5277                    89M
    5278                    13M
    5279                   3.8M
    5280                    15M
    5281                    15M
    5282                    25M
    5283                    34M
    5284                   6.8M
    5285                    24M
    5286                   3.4M
    5287                    24M
    5288                   5.8M
    5289                   4.9M
    5290                    16M
    5291                    46M
    5292                    33M
    5293                    43M
    5294                    56M
    5295                   2.8M
    5296                    11M
    5297                    26M
    5298                    11M
    5299                   8.8M
    5300                   6.2M
    5301                   7.6M
    5302                   4.9M
    5303                   5.1M
    5304                   4.2M
    5305                   7.0M
    5306                    38M
    5307                   8.7M
    5308                   2.1M
    5309                    12M
    5310                    58M
    5311                    18M
    5312                    20M
    5313                   8.0M
    5314                   3.0M
    5315                    12M
    5316                   8.0M
    5317                    28M
    5318                    29M
    5319                   4.7M
    5320                    40M
    5321                    14M
    5322     Varies with device
    5323                    16M
    5324                   9.7M
    5325                    49M
    5326                   3.7M
    5327     Varies with device
    5328                    27M
    5329                   3.6M
    5330                   5.3M
    5331                   2.4M
    5332                   8.5M
    5333                   6.7M
    5334     Varies with device
    5335                   9.0M
    5336                   266k
    5337     Varies with device
    5338                    27M
    5339                    67M
    5340                    22M
    5341                    11M
    5342                    68M
    5343                    40M
    5344                    15M
    5345     Varies with device
    5346                    23M
    5347                   4.2M
    5348                   2.9M
    5349     Varies with device
    5350                    85M
    5351                   1.8M
    5352                   1.1M
    5353                    22M
    5354                   8.7M
    5355                   2.6M
    5356                   4.7M
    5357                   2.9M
    5358                    22M
    5359                   965k
    5360                   1.8M
    5361                   2.0M
    5362                   2.7M
    5363                    56M
    5364                   2.7M
    5365                   2.9M
    5366                   4.9M
    5367                    20M
    5368                    12M
    5369                   3.8M
    5370                   4.2M
    5371                   1.4M
    5372                    21M
    5373                    41M
    5374     Varies with device
    5375                    57M
    5376                    15M
    5377                   5.1M
    5378                    31M
    5379                    28M
    5380                    39M
    5381     Varies with device
    5382                    34M
    5383     Varies with device
    5384                    31M
    5385                    76M
    5386                    40M
    5387                    46M
    5388                   8.7M
    5389                    80M
    5390                    24M
    5391                    21M
    5392     Varies with device
    5393                   9.7M
    5394                    12M
    5395     Varies with device
    5396     Varies with device
    5397                    57M
    5398                    81M
    5399     Varies with device
    5400                   6.7M
    5401                   3.8M
    5402                   5.9M
    5403                    46M
    5404                   3.3M
    5405                    48M
    5406                    33M
    5407                    42M
    5408                    99M
    5409                   4.5M
    5410                    28M
    5411                   8.1M
    5412                    21M
    5413                    56M
    5414     Varies with device
    5415                    25M
    5416                    26M
    5417                    99M
    5418                    25M
    5419                    99M
    5420                    43M
    5421                    17M
    5422                    80M
    5423                    44M
    5424     Varies with device
    5425     Varies with device
    5426                    12M
    5427                   100M
    5428                    69M
    5429                    90M
    5430                    88M
    5431                    64M
    5432                    39M
    5433                    54M
    5434                    56M
    5435                    14M
    5436     Varies with device
    5437                    75M
    5438                    13M
    5439                   8.1M
    5440     Varies with device
    5441                    20M
    5442                    57M
    5443                    31M
    5444                    39M
    5445                    29M
    5446                   3.6M
    5447                   3.8M
    5448                   4.6M
    5449                   545k
    5450                   2.1M
    5451                    61k
    5452                    13M
    5453                   2.8M
    5454                   9.9M
    5455                   2.5M
    5456                   5.4M
    5457                    14M
    5458                   1.3M
    5459                   2.5M
    5460                   283k
    5461                   2.3M
    5462                   1.6M
    5463                    25M
    5464                   5.4M
    5465                   1.5M
    5466                   5.3M
    5467                   2.0M
    5468                   5.4M
    5469                    41M
    5470                   5.4M
    5471                   2.4M
    5472                   5.4M
    5473                   9.0M
    5474                    25M
    5475                   1.2M
    5476                    45M
    5477                   5.3M
    5478                   3.2M
    5479                    73M
    5480                    11M
    5481                   5.4M
    5482                   655k
    5483                   5.6M
    5484                    17M
    5485                   9.5M
    5486                   7.4M
    5487                   5.4M
    5488                   2.6M
    5489                    96M
    5490                    73M
    5491                   5.4M
    5492                    35M
    5493                    48M
    5494     Varies with device
    5495     Varies with device
    5496                    37M
    5497                    17M
    5498                    35M
    5499                   2.3M
    5500                    16M
    5501                    26M
    5502                   2.9M
    5503                   714k
    5504                    12M
    5505     Varies with device
    5506                    52M
    5507     Varies with device
    5508                   3.2M
    5509                    49M
    5510                    28M
    5511                    12M
    5512     Varies with device
    5513                    17M
    5514                    37M
    5515                   2.0M
    5516     Varies with device
    5517                    82M
    5518     Varies with device
    5519     Varies with device
    5520                    37M
    5521                    48M
    5522                    54M
    5523                    46M
    5524                    51M
    5525                    48M
    5526                    55M
    5527     Varies with device
    5528                    25M
    5529                    33M
    5530                   100M
    5531                    70M
    5532     Varies with device
    5533                   9.2M
    5534                    43M
    5535                    91M
    5536                    29M
    5537     Varies with device
    5538                    44M
    5539                    30M
    5540                    99M
    5541                    90M
    5542                    72M
    5543                    47M
    5544                    23M
    5545     Varies with device
    5546                    26M
    5547                   5.0M
    5548     Varies with device
    5549                    20M
    5550                    30M
    5551                    37M
    5552                    28M
    5553                    85M
    5554     Varies with device
    5555                    22M
    5556                    82M
    5557                    65M
    5558                    43M
    5559                    80M
    5560                    21M
    5561                    93k
    5562                    90M
    5563                    67M
    5564     Varies with device
    5565                    32M
    5566                   2.7M
    5567                    36M
    5568                    35M
    5569                    41M
    5570     Varies with device
    5571                    71M
    5572                    70M
    5573                    22M
    5574                    14M
    5575                   6.0M
    5576     Varies with device
    5577                    53M
    5578                   872k
    5579                   1.3M
    5580                   1.6M
    5581     Varies with device
    5582                    56M
    5583                   3.4M
    5584                   3.1M
    5585                    20M
    5586                   6.2M
    5587                   121k
    5588     Varies with device
    5589                    95M
    5590                   9.4M
    5591     Varies with device
    5592     Varies with device
    5593                   2.5M
    5594     Varies with device
    5595                    93M
    5596                    70M
    5597                   1.7M
    5598                    37M
    5599                    70M
    5600     Varies with device
    5601     Varies with device
    5602     Varies with device
    5603     Varies with device
    5604                   3.3M
    5605                    16M
    5606     Varies with device
    5607                    19M
    5608                    22M
    5609     Varies with device
    5610                    68M
    5611     Varies with device
    5612     Varies with device
    5613     Varies with device
    5614                    34M
    5615                    46M
    5616                    92M
    5617                    45M
    5618                    50M
    5619                   9.5M
    5620     Varies with device
    5621                    36M
    5622                    78M
    5623     Varies with device
    5624                    22M
    5625                    48M
    5626                   1.5M
    5627                    40M
    5628                   6.5M
    5629                    51M
    5630                    12M
    5631                    50M
    5632                    40M
    5633                    54M
    5634                    45M
    5635                    63M
    5636                   4.0M
    5637                    69M
    5638                    39M
    5639                    55M
    5640                    46M
    5641                    99M
    5642                    91M
    5643                    21M
    5644                    46M
    5645                    45M
    5646                    24M
    5647                    52M
    5648                    50M
    5649                    63M
    5650                   5.8M
    5651     Varies with device
    5652                    23M
    5653                    23M
    5654                    26M
    5655                   8.8M
    5656                    27M
    5657                   2.1M
    5658                    12M
    5659                   5.0M
    5660                    23M
    5661                   8.2M
    5662                    14M
    5663                   8.5M
    5664     Varies with device
    5665                    47M
    5666                    12M
    5667                    17M
    5668                   3.9M
    5669                  10.0M
    5670                   4.1M
    5671                    15M
    5672                   3.3M
    5673                   6.7M
    5674                    11M
    5675                   4.1M
    5676                   3.7M
    5677                   4.3M
    5678                    31M
    5679                    25M
    5680                   2.0M
    5681                   5.8M
    5682                   3.7M
    5683                   1.7M
    5684                    12M
    5685                    14M
    5686                    78M
    5687                   6.4M
    5688                    10M
    5689                    44M
    5690                   1.5M
    5691                    85M
    5692                   9.8M
    5693                   5.7M
    5694                   3.7M
    5695     Varies with device
    5696     Varies with device
    5697                    46M
    5698                    30M
    5699                    47M
    5700                    23M
    5701     Varies with device
    5702                    91M
    5703                    44M
    5704     Varies with device
    5705                   4.8M
    5706                   9.5M
    5707                    19M
    5708                   7.3M
    5709                   1.8M
    5710                   5.7M
    5711                   7.3M
    5712                   4.0M
    5713                   3.7M
    5714                    26M
    5715                   5.6M
    5716                   7.3M
    5717                    49M
    5718                   8.7M
    5719                    49M
    5720                    20M
    5721                   7.3M
    5722                   322k
    5723                   6.3M
    5724                   3.4M
    5725                    22M
    5726                    25M
    5727                   6.3M
    5728                    82M
    5729                   8.7M
    5730                    10M
    5731                    24M
    5732     Varies with device
    5733                    43M
    5734     Varies with device
    5735                   3.7M
    5736                    13M
    5737     Varies with device
    5738                   3.9M
    5739                    13M
    5740                   3.0M
    5741                   3.0M
    5742     Varies with device
    5743                    46M
    5744                   3.0M
    5745                    17M
    5746                    63M
    5747                    17M
    5748                   2.9M
    5749                   2.9M
    5750                    26M
    5751                   2.7M
    5752     Varies with device
    5753     Varies with device
    5754                    24M
    5755                   4.7M
    5756                   2.7M
    5757                   1.7M
    5758                   3.0M
    5759                   3.0M
    5760     Varies with device
    5761                    31M
    5762                    36M
    5763                   4.3M
    5764                    63M
    5765                   8.6M
    5766                    57M
    5767     Varies with device
    5768                    13M
    5769                   5.6M
    5770     Varies with device
    5771                   1.0M
    5772     Varies with device
    5773     Varies with device
    5774                    23M
    5775                   1.6M
    5776                    24M
    5777                   5.9M
    5778     Varies with device
    5779     Varies with device
    5780                    44M
    5781                    30M
    5782                    76M
    5783                   976k
    5784                    10M
    5785                    44M
    5786                    19M
    5787                   3.0M
    5788                    15M
    5789                    15M
    5790                    43M
    5791                   2.6M
    5792                   172k
    5793                    24M
    5794                    31M
    5795                    25M
    5796                   2.3M
    5797                    33M
    5798                   9.0M
    5799                   7.4M
    5800                    19M
    5801                   2.3M
    5802                    42M
    5803     Varies with device
    5804                   3.3M
    5805                   9.1M
    5806                   2.8M
    5807                    13M
    5808                    30M
    5809                   3.0M
    5810                   2.2M
    5811                    14M
    5812                   238k
    5813                    37M
    5814                    52M
    5815                   4.4M
    5816     Varies with device
    5817                    28M
    5818                    32M
    5819                   3.4M
    5820                    23M
    5821                    61M
    5822                    15M
    5823                    64M
    5824     Varies with device
    5825                    12M
    5826                   4.4M
    5827                   3.6M
    5828                   2.9M
    5829                    10M
    5830                   7.2M
    5831                   3.8M
    5832                   549k
    5833                   3.9M
    5834                   2.9M
    5835                    42M
    5836                   1.1M
    5837                   9.7M
    5838                   1.7M
    5839                    13M
    5840                    55M
    5841                   6.5M
    5842                   1.9M
    5843                   5.1M
    5844                    36M
    5845                    11M
    5846                    14M
    5847     Varies with device
    5848                    17M
    5849     Varies with device
    5850                   4.3M
    5851                   4.0M
    5852                   4.0M
    5853                   3.0M
    5854                   1.2M
    5855                    29M
    5856     Varies with device
    5857                   3.4M
    5858                   5.1M
    5859                   2.7M
    5860                    97M
    5861                    19M
    5862                   100M
    5863     Varies with device
    5864                    73M
    5865                   100M
    5866                    18M
    5867                    18M
    5868                    44M
    5869     Varies with device
    5870                   5.2M
    5871                   8.9M
    5872                    34M
    5873                    30M
    5874                   1.6M
    5875                   4.4M
    5876                   3.7M
    5877                    11M
    5878                   4.7M
    5879                    33M
    5880                    13M
    5881                    14M
    5882                   3.7M
    5883                   7.3M
    5884     Varies with device
    5885                    37M
    5886                    13M
    5887                    13M
    5888                   8.4M
    5889                    45M
    5890                   7.9M
    5891                   4.2M
    5892                    11M
    5893                   6.2M
    5894                   3.7M
    5895                   6.0M
    5896                   8.8M
    5897                    22M
    5898                    30M
    5899                    11M
    5900                   4.1M
    5901                   3.4M
    5902                    13M
    5903                    39M
    5904                   5.1M
    5905                   4.7M
    5906                   5.5M
    5907                   5.3M
    5908                   2.8M
    5909                   9.0M
    5910                    51M
    5911                   206k
    5912                   954k
    5913                    26M
    5914                   9.4M
    5915                   2.4M
    5916                   5.8M
    5917                    20M
    5918                    17M
    5919                    35M
    5920                    17M
    5921                   2.9M
    5922                    42M
    5923                   5.1M
    5924                    16M
    5925                   4.7M
    5926                    35M
    5927                    12M
    5928                    26M
    5929                    17M
    5930                    60M
    5931                    25M
    5932                    38M
    5933                   2.0M
    5934                   2.5M
    5935                   9.2M
    5936                   444k
    5937                    15M
    5938                    12M
    5939                   4.4M
    5940                    14M
    5941                   6.7M
    5942                   3.7M
    5943                   4.0M
    5944                   2.2M
    5945                    33M
    5946     Varies with device
    5947                   5.7M
    5948                    87M
    5949                    30M
    5950                    68M
    5951                   5.9M
    5952     Varies with device
    5953                    19M
    5954                    40M
    5955                    49M
    5956                    52M
    5957                    63M
    5958                    24M
    5959                   6.0M
    5960                    17M
    5961                   4.2M
    5962                    32M
    5963                   717k
    5964     Varies with device
    5965                   4.3M
    5966                   7.2M
    5967                   5.0M
    5968                   1.2M
    5969                    31M
    5970                    17M
    5971                    46M
    5972                    20M
    5973                   8.3M
    5974                   5.0M
    5975                    22M
    5976                   3.1M
    5977                   2.2M
    5978                   2.0M
    5979                    12M
    5980                   5.5M
    5981                    14M
    5982     Varies with device
    5983                    15M
    5984                    81M
    5985                    62M
    5986                   8.9M
    5987     Varies with device
    5988                   3.0M
    5989                    13M
    5990                    11M
    5991                   3.4M
    5992                    12M
    5993                    29M
    5994     Varies with device
    5995                   2.6M
    5996                   1.7M
    5997                   5.4M
    5998     Varies with device
    5999                    30M
    6000                    31M
    6001                    14M
    6002                    37M
    6003                   4.0M
    6004     Varies with device
    6005                   6.9M
    6006                    14M
    6007                   210k
    6008                    13M
    6009                    18M
    6010                    13M
    6011                    38M
    6012                   3.8M
    6013                   3.2M
    6014                   3.2M
    6015                   609k
    6016                   1.7M
    6017                   4.0M
    6018                   6.5M
    6019                    14M
    6020                   2.8M
    6021                    33M
    6022                   3.3M
    6023                   3.9M
    6024                    15M
    6025                   9.5M
    6026                   7.3M
    6027                   2.4M
    6028                   3.0M
    6029                   3.5M
    6030                   2.4M
    6031                    13M
    6032     Varies with device
    6033                   5.3M
    6034                   6.9M
    6035                   2.3M
    6036                    15M
    6037                   7.2M
    6038                    13M
    6039                   7.3M
    6040                   3.3M
    6041                   7.3M
    6042                   7.3M
    6043                   2.6M
    6044                   308k
    6045                   4.5M
    6046                    21M
    6047                   5.3M
    6048                   2.3M
    6049                    22M
    6050                    40M
    6051     Varies with device
    6052                    21M
    6053                    21M
    6054                    53M
    6055                   1.2M
    6056                    53M
    6057                    11M
    6058                    27M
    6059                   6.8M
    6060                   7.4M
    6061                   3.5M
    6062                   1.7M
    6063                   4.0M
    6064                   8.8M
    6065                    38M
    6066                   1.7M
    6067     Varies with device
    6068     Varies with device
    6069                   6.1M
    6070                    27M
    6071                    38M
    6072                   2.8M
    6073                   6.5M
    6074                   3.6M
    6075                    53M
    6076                   3.9M
    6077     Varies with device
    6078                    12M
    6079                    89M
    6080                    46M
    6081                    54M
    6082                   9.2M
    6083                    24M
    6084                    41M
    6085                   705k
    6086                    25M
    6087                    62M
    6088                    27M
    6089                    37M
    6090                   3.7M
    6091                   4.6M
    6092                   9.9M
    6093                   9.0M
    6094                    29M
    6095                   2.5M
    6096                   6.3M
    6097                    58M
    6098                   4.1M
    6099                   9.6M
    6100                   3.8M
    6101                    13M
    6102                    51M
    6103                   2.7M
    6104                    82M
    6105                   5.0M
    6106                    83M
    6107                   2.5M
    6108                   2.5M
    6109                   6.6M
    6110                   2.5M
    6111                   3.7M
    6112                   4.6M
    6113                    80M
    6114                    11M
    6115                   2.6M
    6116                   2.5M
    6117                    24M
    6118                    12M
    6119                    10M
    6120     Varies with device
    6121                   306k
    6122                    26M
    6123                   2.8M
    6124                   8.1M
    6125                   3.2M
    6126                   3.3M
    6127                   1.6M
    6128                   5.6M
    6129                   4.5M
    6130                    11M
    6131                   2.8M
    6132                   2.8M
    6133                   2.8M
    6134                    17M
    6135                   2.9M
    6136                    19M
    6137                   4.1M
    6138                   8.4M
    6139                   5.0M
    6140                   3.9M
    6141                   2.8M
    6142                   2.6M
    6143                   6.1M
    6144                   2.6M
    6145                   8.9M
    6146                   9.1M
    6147                   2.9M
    6148                   2.0M
    6149                   2.8M
    6150                   2.8M
    6151                   9.4M
    6152                   7.8M
    6153                   8.6M
    6154     Varies with device
    6155                    10M
    6156     Varies with device
    6157                   5.1M
    6158                   7.8M
    6159                   5.9M
    6160                   1.7M
    6161     Varies with device
    6162                   2.3M
    6163                   2.5M
    6164                   2.8M
    6165                   4.1M
    6166                   6.2M
    6167                    20M
    6168                   4.8M
    6169     Varies with device
    6170                    24M
    6171                   2.4M
    6172                    13M
    6173     Varies with device
    6174                   1.8M
    6175                   9.8M
    6176                    29M
    6177                    11M
    6178                   6.4M
    6179                    87M
    6180                   7.8M
    6181                    86M
    6182                   2.1M
    6183                   4.0M
    6184                   1.9M
    6185                   2.4M
    6186                   2.9M
    6187                    21M
    6188                   8.7M
    6189                   1.7M
    6190                   5.1M
    6191                    11M
    6192                   1.3M
    6193                   9.2M
    6194                    35M
    6195                   9.2M
    6196                   4.9M
    6197                    15M
    6198                    15M
    6199                    19M
    6200                   2.2M
    6201                   2.1M
    6202                   8.0M
    6203     Varies with device
    6204                   3.5M
    6205                    16M
    6206                    63M
    6207     Varies with device
    6208                    10M
    6209                   7.5M
    6210                    21M
    6211                   1.8M
    6212                    11M
    6213                    18M
    6214     Varies with device
    6215     Varies with device
    6216                   3.2M
    6217     Varies with device
    6218                    33M
    6219                   8.4M
    6220                    31M
    6221                    22M
    6222                   3.6M
    6223                    15M
    6224                    18M
    6225                    10M
    6226                   2.5M
    6227                    16M
    6228                    15M
    6229                   4.3M
    6230                   4.6M
    6231                    18M
    6232                   8.7M
    6233                    22M
    6234                   1.1M
    6235                   4.8M
    6236                   2.6M
    6237                   5.2M
    6238                   6.0M
    6239                    43M
    6240                   904k
    6241                    20M
    6242                    28M
    6243                    20M
    6244                    16M
    6245     Varies with device
    6246                   8.7M
    6247                   6.0M
    6248                   1.8M
    6249                   3.7M
    6250                   3.2M
    6251                    16M
    6252                   7.4M
    6253                   3.3M
    6254                   3.5M
    6255                    14M
    6256                   8.9M
    6257                    32M
    6258                    14M
    6259                   3.8M
    6260                   4.7M
    6261                   9.9M
    6262                    17M
    6263                   9.1M
    6264                    10M
    6265                    27M
    6266                    25M
    6267                    33M
    6268                    30M
    6269     Varies with device
    6270                    28M
    6271                   7.3M
    6272                    23M
    6273                   201k
    6274     Varies with device
    6275                   2.7M
    6276                   7.7M
    6277     Varies with device
    6278                    21M
    6279                   3.5M
    6280                   3.3M
    6281                    31M
    6282                    12M
    6283                   8.1M
    6284                    14M
    6285                    11M
    6286     Varies with device
    6287                    18M
    6288     Varies with device
    6289     Varies with device
    6290     Varies with device
    6291                   2.9M
    6292                   473k
    6293                    76M
    6294                    18M
    6295                    54M
    6296                    30M
    6297     Varies with device
    6298     Varies with device
    6299                   2.1M
    6300                    48M
    6301                    92M
    6302                    15M
    6303                    30M
    6304     Varies with device
    6305                   2.6M
    6306                    17M
    6307                    35M
    6308                    11M
    6309                    65M
    6310                    25M
    6311                   4.8M
    6312                   4.8M
    6313                    17M
    6314                    24M
    6315                    26M
    6316                   4.8M
    6317                   175k
    6318                    15M
    6319                   4.9M
    6320                   8.8M
    6321                    36M
    6322                   8.7M
    6323                    23M
    6324                   5.1M
    6325                    23M
    6326                    26M
    6327                    53M
    6328                   3.2M
    6329                    23M
    6330                   1.8M
    6331                   5.4M
    6332                   8.0M
    6333                   5.4M
    6334                    59M
    6335                    21M
    6336                   5.1M
    6337     Varies with device
    6338                    55M
    6339                   3.4M
    6340                    33M
    6341                   4.7M
    6342                   1.5M
    6343     Varies with device
    6344                    30M
    6345                    26M
    6346                    54M
    6347                    10M
    6348                   3.6M
    6349                   5.2M
    6350                    25M
    6351                   350k
    6352                    51M
    6353                   5.1M
    6354                   4.6M
    6355                    39M
    6356                    13M
    6357                    31M
    6358                    19M
    6359                   3.1M
    6360                    17M
    6361                   383k
    6362                   2.4M
    6363                   3.2M
    6364                   2.1M
    6365                    30M
    6366                    14M
    6367                    12M
    6368                   3.0M
    6369                   4.8M
    6370                   2.8M
    6371                   4.0M
    6372                    11M
    6373     Varies with device
    6374                   1.8M
    6375                   3.1M
    6376                    20M
    6377                    33M
    6378                    17M
    6379                    28M
    6380                   454k
    6381                   3.1M
    6382                   3.2M
    6383                   6.8M
    6384                    99M
    6385                    17M
    6386                    12M
    6387                    25M
    6388                    13M
    6389                   2.1M
    6390                   5.9M
    6391                   3.7M
    6392                   4.5M
    6393                   4.1M
    6394                    11M
    6395                   3.9M
    6396                   3.0M
    6397                   1.4M
    6398                    11M
    6399                    12M
    6400                    30M
    6401                    25M
    6402                    52M
    6403                   6.3M
    6404                   4.7M
    6405                    25M
    6406                    62M
    6407                   6.4M
    6408                    15M
    6409                    70M
    6410                    12M
    6411                   7.1M
    6412                    47M
    6413                    23M
    6414                    10M
    6415                    40M
    6416                    53M
    6417                   8.2M
    6418                   8.5M
    6419                    33M
    6420                    34M
    6421                    23M
    6422                   2.5M
    6423                   1.8M
    6424     Varies with device
    6425                   1.4M
    6426                    24M
    6427                   2.0M
    6428                   421k
    6429                   1.2M
    6430                    11M
    6431                   7.0M
    6432                   2.1M
    6433                    16M
    6434                    10M
    6435                    28M
    6436                   5.5M
    6437                    41M
    6438                   2.7M
    6439                    93M
    6440                    40M
    6441                   3.4M
    6442                   9.0M
    6443                   4.3M
    6444                   2.9M
    6445                   6.9M
    6446                   1.0M
    6447                   1.1M
    6448                    39M
    6449     Varies with device
    6450     Varies with device
    6451                    31M
    6452                    14M
    6453                    29M
    6454                   4.2M
    6455                   5.3M
    6456                   1.7M
    6457                   6.3M
    6458                    27M
    6459                   3.1M
    6460     Varies with device
    6461                   2.5M
    6462                   5.0M
    6463                   6.3M
    6464                   7.8M
    6465                   4.8M
    6466     Varies with device
    6467                   9.2M
    6468                    34M
    6469                    47M
    6470                   8.1M
    6471                   2.5M
    6472                    36M
    6473                    10M
    6474                   6.2M
    6475                    10M
    6476                   4.4M
    6477                    18M
    6478                    16M
    6479     Varies with device
    6480                   9.9M
    6481                    22M
    6482                   1.9M
    6483                    26M
    6484                   9.3M
    6485     Varies with device
    6486     Varies with device
    6487                    28M
    6488                    10M
    6489                    11M
    6490                   2.3M
    6491                   4.4M
    6492                   8.0M
    6493                    10M
    6494                   3.7M
    6495                    21M
    6496                   4.6M
    6497     Varies with device
    6498                    45M
    6499                   4.2M
    6500                   7.6M
    6501                   8.6M
    6502                    15M
    6503                    70k
    6504                    17M
    6505                   812k
    6506                    13M
    6507                   442k
    6508                   2.0M
    6509     Varies with device
    6510                   842k
    6511                   2.4M
    6512                   417k
    6513                   5.6M
    6514                   412k
    6515                   459k
    6516                    34M
    6517                   478k
    6518                    10M
    6519                   6.2M
    6520                   335k
    6521                   3.8M
    6522                   782k
    6523                   721k
    6524                   430k
    6525                   429k
    6526                   192k
    6527                   200k
    6528                    13M
    6529                   2.5M
    6530                   8.9M
    6531                   417k
    6532     Varies with device
    6533                   460k
    6534                   5.9M
    6535                   728k
    6536                   496k
    6537                   816k
    6538                   6.3M
    6539                   414k
    6540                   334k
    6541                   506k
    6542                   2.0M
    6543                   2.3M
    6544                    34M
    6545                   7.5M
    6546     Varies with device
    6547                    60M
    6548                    60M
    6549                   7.9M
    6550                    24M
    6551                    95M
    6552                    59M
    6553                    23M
    6554                    12M
    6555                    11M
    6556                    14M
    6557                   2.2M
    6558                    30M
    6559                   8.5M
    6560                    19M
    6561                    17M
    6562                    32M
    6563                    33M
    6564                   4.2M
    6565                    33M
    6566                    96M
    6567                   4.7M
    6568                    26M
    6569                    23M
    6570                    30M
    6571                   6.4M
    6572                    28M
    6573                    49M
    6574                    16M
    6575                    51M
    6576                   2.6M
    6577                    25M
    6578     Varies with device
    6579                    27M
    6580                    23M
    6581                    31M
    6582                    25M
    6583                    16M
    6584                   704k
    6585                   7.4M
    6586                   5.0M
    6587                   2.6M
    6588                    45M
    6589                    23M
    6590                   8.4M
    6591                   8.3M
    6592                   7.9M
    6593                    26M
    6594                   6.0M
    6595                   4.8M
    6596                   4.6M
    6597                    16M
    6598                   5.7M
    6599                   5.9M
    6600                   335k
    6601                   2.2M
    6602                   887k
    6603                   613k
    6604                   1.5M
    6605                    12M
    6606                    26M
    6607                    13M
    6608     Varies with device
    6609                   3.3M
    6610                   2.9M
    6611                    10M
    6612                   3.2M
    6613                   2.5M
    6614                   1.8M
    6615                   2.7M
    6616                   7.0M
    6617                    12M
    6618     Varies with device
    6619                   9.6M
    6620                    18M
    6621                   1.9M
    6622                    29M
    6623                   9.4M
    6624                   6.7M
    6625                   1.6M
    6626                   2.0M
    6627     Varies with device
    6628                   2.5M
    6629                   7.2M
    6630                   6.3M
    6631                   4.0M
    6632                   1.2M
    6633     Varies with device
    6634                   5.6M
    6635                   3.5M
    6636                   9.2M
    6637                   1.2M
    6638     Varies with device
    6639                    18M
    6640                    36M
    6641                   1.2M
    6642                   8.8M
    6643                   4.1M
    6644                   2.5M
    6645                   2.7M
    6646                    33M
    6647                    28M
    6648                   6.7M
    6649                    21M
    6650                   3.5M
    6651                   6.0M
    6652     Varies with device
    6653                    22M
    6654     Varies with device
    6655                    27M
    6656                   3.4M
    6657     Varies with device
    6658                    16M
    6659                    12M
    6660                    10M
    6661     Varies with device
    6662                   9.0M
    6663                    36M
    6664                    12M
    6665                   3.6M
    6666                    30M
    6667                    19M
    6668                    13M
    6669                    33M
    6670                    14M
    6671                   243k
    6672                    24M
    6673     Varies with device
    6674                   6.6M
    6675                    44M
    6676                    89M
    6677                   3.5M
    6678                    32M
    6679                    50M
    6680                    15M
    6681                   3.1M
    6682                    30M
    6683                    34M
    6684                    13M
    6685                   2.6M
    6686                   2.5M
    6687     Varies with device
    6688                   7.5M
    6689                   569k
    6690                    26M
    6691                   5.2M
    6692                   5.4M
    6693                   9.1M
    6694                    26M
    6695                    28M
    6696                   3.8M
    6697     Varies with device
    6698                    92M
    6699                   206k
    6700                    19M
    6701                    93M
    6702                    13M
    6703                   1.8M
    6704                    10M
    6705                   3.5M
    6706                   7.5M
    6707                    68M
    6708                   778k
    6709                   2.1M
    6710                   9.2M
    6711                   3.0M
    6712                    72M
    6713                    35M
    6714     Varies with device
    6715                    31M
    6716                   4.1M
    6717     Varies with device
    6718                    13M
    6719                    60M
    6720                    85M
    6721                   1.3M
    6722     Varies with device
    6723                   6.9M
    6724                   5.5M
    6725                    58M
    6726                    94M
    6727                   683k
    6728                   592k
    6729     Varies with device
    6730                   6.4M
    6731                   3.3M
    6732                   319k
    6733                    24M
    6734                    36M
    6735                   186k
    6736                   3.5M
    6737                   840k
    6738                    41M
    6739                   1.5M
    6740                    36M
    6741                   647k
    6742                   5.3M
    6743                    19M
    6744                   2.5M
    6745                    93M
    6746                   2.8M
    6747                   5.6M
    6748                    72M
    6749                    21M
    6750                   191k
    6751                    22M
    6752                    67M
    6753                   842k
    6754                   1.1M
    6755                   6.9M
    6756     Varies with device
    6757                    15M
    6758                   1.1M
    6759                   6.2M
    6760                   1.4M
    6761                    52M
    6762                   8.7M
    6763                   373k
    6764                   3.8M
    6765                   3.8M
    6766                   3.8M
    6767                   8.2M
    6768                   437k
    6769                   5.7M
    6770                   1.6M
    6771                    16M
    6772                   2.6M
    6773                   6.6M
    6774                   598k
    6775                   2.7M
    6776     Varies with device
    6777                   716k
    6778                    14M
    6779                    12M
    6780                    46M
    6781                    13M
    6782                   1.9M
    6783                   1.2M
    6784                   4.3M
    6785                    13M
    6786                   8.4M
    6787                   7.3M
    6788                   585k
    6789                   3.2M
    6790                   6.5M
    6791                   2.7M
    6792                   1.4M
    6793                    17M
    6794                   3.1M
    6795                   2.4M
    6796                   1.6M
    6797     Varies with device
    6798                   982k
    6799                    26M
    6800                    19M
    6801                   2.9M
    6802                   4.5M
    6803                   222k
    6804                    42M
    6805                   7.9M
    6806                   219k
    6807                    55k
    6808                   948k
    6809                   7.3M
    6810                   323k
    6811                    89M
    6812                   5.2M
    6813                    15M
    6814                    19M
    6815                   7.6M
    6816                   5.6M
    6817                    18M
    6818                    24M
    6819                    10M
    6820                    11M
    6821                    29M
    6822                   4.3M
    6823                    10M
    6824                    88M
    6825                   5.3M
    6826                   5.4M
    6827                   3.1M
    6828                   6.0M
    6829                    26M
    6830                   6.5M
    6831                    19M
    6832                    33M
    6833                    40M
    6834                   9.2M
    6835                    28M
    6836                    21M
    6837                    20M
    6838                   4.0M
    6839                    26M
    6840     Varies with device
    6841                    26M
    6842                    33M
    6843                    25M
    6844                    44M
    6845     Varies with device
    6846                    19M
    6847                    27M
    6848                   1.5M
    6849     Varies with device
    6850                    13M
    6851                   4.8M
    6852                   1.6M
    6853                   4.9M
    6854                   3.5M
    6855                   8.7M
    6856                   7.7M
    6857                    19M
    6858                   3.4M
    6859                   2.9M
    6860                   2.4M
    6861                   2.7M
    6862                   5.0M
    6863                   5.9M
    6864                   2.8M
    6865                   3.4M
    6866                    12M
    6867                    20M
    6868                   4.2M
    6869                    10M
    6870                    26M
    6871                   4.0M
    6872                   3.6M
    6873                    98M
    6874                   5.7M
    6875                   3.6M
    6876                    96M
    6877                   2.2M
    6878                    10M
    6879                   4.8M
    6880                   2.0M
    6881                    62M
    6882     Varies with device
    6883                   9.8M
    6884                    34M
    6885                    53M
    6886                   7.3M
    6887     Varies with device
    6888                   2.8M
    6889                    24M
    6890                    73M
    6891                    53M
    6892                    27M
    6893                   4.4M
    6894                    17M
    6895                   1.3M
    6896                    19M
    6897                   8.8M
    6898                   1.3M
    6899                   1.8M
    6900                   4.8M
    6901                   1.8M
    6902                   691k
    6903                    12M
    6904                    20M
    6905                   5.1M
    6906                   7.2M
    6907                    24M
    6908                   7.4M
    6909                    53M
    6910                   1.3M
    6911                    14M
    6912                   8.8M
    6913                   1.7M
    6914                    14M
    6915                    13M
    6916                    17M
    6917                    24M
    6918                    17M
    6919                    26M
    6920                   511k
    6921                   1.7M
    6922                    32M
    6923                    12M
    6924                    23M
    6925                   5.5M
    6926                   9.4M
    6927                    18M
    6928                   1.8M
    6929                   951k
    6930                   1.9M
    6931                   2.6M
    6932                    13M
    6933                    16M
    6934                   963k
    6935                   8.5M
    6936                    23M
    6937                   8.2M
    6938                   3.8M
    6939                   2.4M
    6940                   1.7M
    6941                    10M
    6942                   2.0M
    6943                    25k
    6944                   4.6M
    6945                    31M
    6946                   8.2M
    6947                   554k
    6948                    21M
    6949                   6.9M
    6950                    21M
    6951                    11M
    6952                    34M
    6953                   4.1M
    6954                   1.9M
    6955                   3.5M
    6956                   3.0M
    6957                   4.0M
    6958                   2.2M
    6959                    18M
    6960                   3.7M
    6961                   2.4M
    6962                   9.3M
    6963                    44M
    6964                    31M
    6965                    48M
    6966                    38M
    6967                    49M
    6968                    27M
    6969                   351k
    6970                   8.8M
    6971                    23M
    6972                    48M
    6973                   9.7M
    6974                    27k
    6975                    12M
    6976                    74M
    6977                    44M
    6978                    36M
    6979                   1.4M
    6980     Varies with device
    6981                    25M
    6982     Varies with device
    6983                    44M
    6984                    36M
    6985                    82k
    6986                    57M
    6987     Varies with device
    6988                   9.8M
    6989                    24M
    6990                    47M
    6991                    37M
    6992                   6.9M
    6993                    10M
    6994                    15M
    6995                   9.4M
    6996                    49M
    6997                    24M
    6998                   7.3M
    6999                    24M
    7000                    14M
    7001     Varies with device
    7002     Varies with device
    7003                    21M
    7004                   8.8M
    7005                   4.0M
    7006     Varies with device
    7007                    18M
    7008                    18M
    7009                    13M
    7010                    13M
    7011                    24M
    7012     Varies with device
    7013                    25M
    7014                    43M
    7015                   4.6M
    7016                    47M
    7017                    20M
    7018                    26M
    7019                    69M
    7020                    13M
    7021                   9.2M
    7022                    19M
    7023                    34M
    7024     Varies with device
    7025                    13M
    7026                    34M
    7027                    24M
    7028                   3.8M
    7029                   8.3M
    7030                    25M
    7031                    53M
    7032                    24M
    7033                   5.4M
    7034                   7.7M
    7035                    11M
    7036                   5.6M
    7037                   5.4M
    7038                    17M
    7039                   4.4M
    7040     Varies with device
    7041                   8.2M
    7042                    98M
    7043                    29M
    7044                   2.4M
    7045                   8.5M
    7046                   3.2M
    7047                   3.8M
    7048                    33M
    7049     Varies with device
    7050                    53M
    7051                   6.2M
    7052                    22M
    7053                    17M
    7054                    25M
    7055                    39M
    7056                   4.6M
    7057                    13M
    7058                    36M
    7059                    13M
    7060                   8.5M
    7061                    25M
    7062                    31M
    7063                   3.7M
    7064                   8.5M
    7065                    24M
    7066                    34M
    7067                   6.9M
    7068                    45M
    7069                    12M
    7070                    33M
    7071                    35M
    7072                    25M
    7073                    48M
    7074                   1.0M
    7075                   7.2M
    7076                    14M
    7077     Varies with device
    7078                   8.9M
    7079     Varies with device
    7080                   208k
    7081                   9.4M
    7082                   8.3M
    7083                   6.9M
    7084                    12M
    7085                   5.5M
    7086                    26M
    7087                    98M
    7088                   5.2M
    7089                   7.7M
    7090                   5.3M
    7091                   375k
    7092                   3.7M
    7093                   2.9M
    7094                    18M
    7095                   6.0M
    7096                    34M
    7097                   7.0M
    7098                    38M
    7099                   5.4M
    7100                   1.2M
    7101                   1.2M
    7102                   7.9M
    7103                   2.3M
    7104                   2.3M
    7105                   323k
    7106     Varies with device
    7107                   1.4M
    7108                   2.5M
    7109                    12M
    7110                    15M
    7111                   3.1M
    7112                    12M
    7113                   1.5M
    7114                   9.0M
    7115                   3.7M
    7116                   8.7M
    7117                    19M
    7118                   7.4M
    7119                   3.3M
    7120                   913k
    7121                   4.9M
    7122                   7.8M
    7123                   5.0M
    7124                    12M
    7125                    11M
    7126                    25M
    7127                   2.6M
    7128                   7.2M
    7129                   1.8M
    7130                   2.6M
    7131                    14M
    7132                   2.2M
    7133                   1.9M
    7134                   5.6M
    7135                   514k
    7136                    20M
    7137                    11M
    7138                    12M
    7139                   5.6M
    7140                   3.8M
    7141                   6.4M
    7142                   3.7M
    7143                   3.4M
    7144                   8.4M
    7145                   2.9M
    7146                   2.9M
    7147                   1.8M
    7148                   3.1M
    7149                    80M
    7150                   3.3M
    7151                    15M
    7152                    36M
    7153     Varies with device
    7154                   8.3M
    7155                   3.1M
    7156                   9.5M
    7157                   4.8M
    7158                   6.7M
    7159     Varies with device
    7160                   1.5M
    7161                   9.2M
    7162                   3.4M
    7163                    13M
    7164                   5.8M
    7165                    20M
    7166                    20M
    7167                    63M
    7168                    13M
    7169                    27M
    7170                   3.8M
    7171                    16M
    7172                   2.3M
    7173                    12M
    7174                   5.6M
    7175                    37M
    7176                    22M
    7177     Varies with device
    7178                   1.7M
    7179                   2.6M
    7180                   7.9M
    7181                    20M
    7182                   5.7M
    7183                    16M
    7184                   1.7M
    7185                   4.2M
    7186                   4.6M
    7187                   2.7M
    7188                   7.2M
    7189                    46M
    7190                    14M
    7191                    12M
    7192                   3.7M
    7193                   1.6M
    7194                    12M
    7195                    29M
    7196                    36M
    7197                   4.0M
    7198                    15M
    7199                    42M
    7200                   7.8M
    7201                   4.6M
    7202                   4.3M
    7203                   2.8M
    7204                    27M
    7205                   7.3M
    7206                    23M
    7207                    20M
    7208                   5.4M
    7209                    24M
    7210                    20M
    7211                   1.4M
    7212                   3.6M
    7213                   6.3M
    7214                    21M
    7215                    14M
    7216                    25M
    7217                    16M
    7218                    55M
    7219                    22M
    7220                   3.0M
    7221                   2.6M
    7222                    24M
    7223                   551k
    7224                   4.0M
    7225                   5.1M
    7226                    22M
    7227                    21M
    7228                    18M
    7229                    62M
    7230                    30M
    7231                    23M
    7232     Varies with device
    7233     Varies with device
    7234                    10M
    7235                    29M
    7236                   1.9M
    7237                    29k
    7238                    43M
    7239                   8.3M
    7240                    19M
    7241                   103k
    7242                   6.2M
    7243                   8.3M
    7244                   5.5M
    7245                    11M
    7246                    17M
    7247                   2.4M
    7248                   4.6M
    7249                   3.0M
    7250                    21M
    7251                   2.7M
    7252                    15M
    7253                   4.7M
    7254                   3.0M
    7255                   1.4M
    7256                    13M
    7257                    16M
    7258                   2.0M
    7259                    11M
    7260                   2.9M
    7261                    40M
    7262                    14M
    7263                   9.7M
    7264                   7.8M
    7265                   3.2M
    7266                   8.0M
    7267                    29M
    7268                   3.1M
    7269                   3.5M
    7270                    14M
    7271                   8.7M
    7272                   8.8M
    7273                   9.7M
    7274                   3.1M
    7275                   3.2M
    7276                   8.5M
    7277                   3.5M
    7278                   7.8M
    7279                    11M
    7280                   6.3M
    7281                    19M
    7282                   5.0M
    7283                    16M
    7284                   3.5M
    7285                   2.3M
    7286                   3.3M
    7287                   4.1M
    7288                   4.1M
    7289                   4.1M
    7290                   7.0M
    7291                   2.3M
    7292                   4.1M
    7293                   1.1M
    7294                    18M
    7295                   4.1M
    7296                   1.5M
    7297                   2.3M
    7298                   3.5M
    7299                   3.7M
    7300     Varies with device
    7301                   4.0M
    7302                   1.8M
    7303                   4.0M
    7304                   3.8M
    7305                    10M
    7306                   1.3M
    7307                   6.6M
    7308                   4.1M
    7309                   2.5M
    7310                   4.1M
    7311     Varies with device
    7312                   2.5M
    7313                   2.5M
    7314                   4.3M
    7315                   7.7M
    7316                   6.8M
    7317                   3.4M
    7318                   1.4M
    7319                   3.6M
    7320                   2.3M
    7321                    14M
    7322                    99M
    7323                   4.1M
    7324                   8.7M
    7325                   2.5M
    7326                   2.9M
    7327                   4.3M
    7328                   6.1M
    7329     Varies with device
    7330     Varies with device
    7331     Varies with device
    7332     Varies with device
    7333                    14M
    7334                    15M
    7335                   6.3M
    7336                    66M
    7337                   3.4M
    7338     Varies with device
    7339                    76M
    7340                   7.7M
    7341                   6.5M
    7342                    14M
    7343                    25M
    7344     Varies with device
    7345     Varies with device
    7346                   1.4M
    7347                    50M
    7348                   2.5M
    7349                   2.2M
    7350                   4.6M
    7351                    15M
    7352                    13M
    7353     Varies with device
    7354                   8.1M
    7355     Varies with device
    7356     Varies with device
    7357                    16M
    7358                   9.6M
    7359     Varies with device
    7360     Varies with device
    7361     Varies with device
    7362                    10M
    7363                   3.1M
    7364                    78M
    7365                    27M
    7366     Varies with device
    7367                    10M
    7368                   5.8M
    7369                   3.2M
    7370                   898k
    7371                   3.7M
    7372     Varies with device
    7373                    57M
    7374                    42M
    7375                    36M
    7376                   1.2M
    7377                    46M
    7378                    30M
    7379                   3.2M
    7380                    10M
    7381                   1.8M
    7382                    31M
    7383                   6.6M
    7384                   5.1M
    7385                   3.2M
    7386                   6.0M
    7387                   2.5M
    7388                   9.9M
    7389                    10M
    7390                   172k
    7391                   5.2M
    7392                   5.1M
    7393                   6.4M
    7394                    14M
    7395                    15M
    7396                   3.3M
    7397                   6.9M
    7398                   5.1M
    7399                    20M
    7400                  10.0M
    7401                   6.6M
    7402                   6.1M
    7403                   3.2M
    7404                   100M
    7405                   1.4M
    7406                   2.1M
    7407                   7.2M
    7408                    91M
    7409                   2.8M
    7410                   743k
    7411                    14M
    7412                    14M
    7413                    43M
    7414                   6.9M
    7415                    96M
    7416                    14M
    7417                    26M
    7418                    18M
    7419                    34M
    7420                   2.8M
    7421                    92M
    7422                    11M
    7423                    15M
    7424                   4.0M
    7425                   3.8M
    7426                   1.3M
    7427                    13M
    7428                    67M
    7429                    99M
    7430                   116k
    7431                   2.2M
    7432                    24M
    7433                    43M
    7434                   5.9M
    7435                   2.4M
    7436                    67M
    7437                    67M
    7438                    13M
    7439                   8.2M
    7440                   5.2M
    7441     Varies with device
    7442                   9.2M
    7443                    63M
    7444                   4.2M
    7445                    16M
    7446                    10M
    7447                    18M
    7448                    12M
    7449                    78M
    7450                    38M
    7451                   9.5M
    7452     Varies with device
    7453                    45M
    7454                    28M
    7455     Varies with device
    7456                   7.7M
    7457                   9.5M
    7458                   5.2M
    7459                    38M
    7460                   3.7M
    7461                    25M
    7462                    20M
    7463                   153k
    7464                   5.3M
    7465                   6.3M
    7466                    38M
    7467                    13M
    7468                   6.4M
    7469                   1.9M
    7470                   6.4M
    7471                   2.0M
    7472                   8.6M
    7473                    24M
    7474                    10M
    7475                    39M
    7476                    60M
    7477                    40M
    7478                    15M
    7479                   209k
    7480                   1.5M
    7481                   2.3M
    7482                    13M
    7483                   9.4M
    7484                   3.7M
    7485                   4.2M
    7486                    22M
    7487                    13M
    7488                   8.1M
    7489                    12M
    7490                    19M
    7491     Varies with device
    7492                   4.9M
    7493                   9.7M
    7494                   3.6M
    7495                   2.5M
    7496                   2.5M
    7497                    18M
    7498                    35M
    7499                   6.3M
    7500                   1.3M
    7501                    24M
    7502                   3.4M
    7503                   3.4M
    7504                   3.4M
    7505                   3.4M
    7506                    17M
    7507                   2.2M
    7508     Varies with device
    7509                   5.0M
    7510                   4.6M
    7511                   9.0M
    7512                   1.3M
    7513                    24M
    7514                   3.2M
    7515                    97M
    7516                   353k
    7517                   3.2M
    7518                   2.4M
    7519                   8.4M
    7520                    12M
    7521                   3.4M
    7522                   3.4M
    7523                   7.3M
    7524                   4.2M
    7525                    58M
    7526                   6.8M
    7527                   499k
    7528                   3.2M
    7529                   3.3M
    7530                   3.7M
    7531                    25M
    7532                    12M
    7533                   1.3M
    7534                    40M
    7535                   5.2M
    7536     Varies with device
    7537                    17M
    7538                   173k
    7539     Varies with device
    7540                   5.1M
    7541                   5.8M
    7542                   1.2M
    7543                   5.8M
    7544                   1.8M
    7545                   1.9M
    7546                   3.6M
    7547                   2.5M
    7548                   2.1M
    7549                   1.1M
    7550                    17M
    7551                   2.1M
    7552                   2.8M
    7553     Varies with device
    7554                   5.5M
    7555                    41M
    7556                    14M
    7557                   597k
    7558                   2.7M
    7559                   3.2M
    7560                   809k
    7561                   2.2M
    7562                   8.2M
    7563                   2.8M
    7564                   4.0M
    7565                   1.8M
    7566                    70k
    7567                   2.3M
    7568                   1.3M
    7569                   122k
    7570                   4.1M
    7571                    11M
    7572                   4.2M
    7573                   8.2M
    7574                   411k
    7575                   2.3M
    7576                   5.7M
    7577     Varies with device
    7578                   2.7M
    7579                   1.9M
    7580                    19M
    7581                   2.3M
    7582     Varies with device
    7583                    49M
    7584                    28M
    7585                    41M
    7586                    25M
    7587                    25M
    7588                    38M
    7589                    28M
    7590                    55M
    7591                    58M
    7592                    27M
    7593                    25M
    7594                    31M
    7595                    94M
    7596                    25M
    7597                    21M
    7598                    45M
    7599                    45M
    7600                    24M
    7601                    83M
    7602                    88M
    7603                    51M
    7604                    63M
    7605                    21M
    7606                   7.2M
    7607                    51M
    7608                    98M
    7609                    27M
    7610                   9.3M
    7611                    87M
    7612                    55M
    7613     Varies with device
    7614                    26M
    7615     Varies with device
    7616                    62M
    7617                    22M
    7618                    40M
    7619                   5.5M
    7620                    18M
    7621                    74M
    7622                    16M
    7623                    64M
    7624                    16M
    7625                    25M
    7626                    12M
    7627                    18M
    7628                   4.5M
    7629                    77M
    7630                    85M
    7631                    85M
    7632                    26M
    7633                   2.0M
    7634                   3.9M
    7635                    11M
    7636                    18M
    7637                   8.7M
    7638                   3.9M
    7639                   4.0M
    7640     Varies with device
    7641                    53M
    7642                   4.9M
    7643                    52M
    7644                    23M
    7645                    13M
    7646                    30M
    7647                    18M
    7648                   4.1M
    7649                    13M
    7650                    20M
    7651                    21M
    7652                    12M
    7653                    18M
    7654                    41M
    7655                    12M
    7656                    14M
    7657     Varies with device
    7658     Varies with device
    7659                   6.0M
    7660                    17M
    7661                   400k
    7662                    42M
    7663                    54M
    7664                    11M
    7665                    37M
    7666                    13M
    7667                   6.5M
    7668                    16M
    7669                    19M
    7670                   3.3M
    7671                    35M
    7672                   6.3M
    7673                    12M
    7674                   5.0M
    7675                    43M
    7676                   2.9M
    7677                   4.1M
    7678                   8.5M
    7679                   7.0M
    7680                    19M
    7681                    16M
    7682                    14M
    7683                   801k
    7684                   5.6M
    7685                    36M
    7686                   3.2M
    7687                   7.0M
    7688                   2.9M
    7689                    35M
    7690                    21M
    7691                   5.8M
    7692                   4.7M
    7693                    20M
    7694                   3.9M
    7695                    12M
    7696                   7.8M
    7697                    24M
    7698                    12M
    7699     Varies with device
    7700                   4.9M
    7701                   3.3M
    7702                   5.4M
    7703                   2.4M
    7704                    27M
    7705                    22M
    7706                   1.2M
    7707                   2.5M
    7708                    14M
    7709                   6.8M
    7710                   1.2M
    7711                   787k
    7712                   4.9M
    7713                   5.3M
    7714                   6.5M
    7715                    69M
    7716                    14M
    7717                   3.4M
    7718                    50M
    7719                   8.1M
    7720                   1.7M
    7721                    18M
    7722                    18M
    7723                    36M
    7724                   8.1M
    7725                   5.1M
    7726                   1.6M
    7727                    61M
    7728                   4.1M
    7729                    10M
    7730                   237k
    7731                   3.4M
    7732                    18M
    7733                   3.5M
    7734                   3.4M
    7735                   9.1M
    7736                    15M
    7737                    14M
    7738                   3.8M
    7739                    10M
    7740                    26M
    7741     Varies with device
    7742                   3.3M
    7743                    15M
    7744                    19M
    7745                   3.7M
    7746     Varies with device
    7747                   2.1M
    7748                    11M
    7749                    43M
    7750                   6.9M
    7751                    40M
    7752     Varies with device
    7753                    36M
    7754                   1.5M
    7755                    37M
    7756                    22M
    7757                   3.3M
    7758                   2.6M
    7759                    34M
    7760                   2.9M
    7761                    12M
    7762                   3.7M
    7763                    17M
    7764                    23M
    7765                    17M
    7766                   5.0M
    7767                    18M
    7768                    33M
    7769                    26M
    7770                    32M
    7771                    38M
    7772                    21M
    7773                    41M
    7774                   1.7M
    7775                   4.3M
    7776                   9.9M
    7777                   5.9M
    7778                   2.7M
    7779                    12M
    7780                    16M
    7781                    19M
    7782                   2.8M
    7783                   3.6M
    7784                    37M
    7785                   4.3M
    7786                   1.7M
    7787                   3.1M
    7788                    30M
    7789                   5.2M
    7790                   2.8M
    7791                    18M
    7792                   3.4M
    7793                    41M
    7794                    13M
    7795                   6.7M
    7796                   9.8M
    7797                    14M
    7798                    26M
    7799                    22M
    7800                   8.7M
    7801                    88M
    7802                    18M
    7803                    18M
    7804                   1.3M
    7805                   4.5M
    7806                   7.8M
    7807                    23M
    7808     Varies with device
    7809     Varies with device
    7810                    50k
    7811                    70M
    7812                    47M
    7813                    42M
    7814                   3.7M
    7815                    39M
    7816                    22M
    7817                    22M
    7818                    12M
    7819                    14M
    7820                   4.3M
    7821     Varies with device
    7822                   8.5M
    7823                   6.7M
    7824                   7.5M
    7825                    88M
    7826                   1.1M
    7827                   8.3M
    7828                   5.7M
    7829                   643k
    7830                    44M
    7831                    11M
    7832                    28M
    7833                    40M
    7834                   9.1M
    7835                   2.1M
    7836                    32M
    7837                   7.8M
    7838                   2.4M
    7839                    53M
    7840                   6.0M
    7841                    25M
    7842                   3.3M
    7843                    34M
    7844                   3.6M
    7845                   6.2M
    7846                    68M
    7847                    19M
    7848                   3.6M
    7849                    17M
    7850                    31M
    7851                    18M
    7852                   6.0M
    7853                    29M
    7854                    77M
    7855                    42M
    7856                    14M
    7857                    29M
    7858                   5.7M
    7859                    36M
    7860                   8.9M
    7861                   5.1M
    7862                    19M
    7863                   4.3M
    7864                    62M
    7865                    29M
    7866                   7.8M
    7867                    20M
    7868                   986k
    7869                    18M
    7870                    14M
    7871                   9.3M
    7872                    17M
    7873                   1.4M
    7874                    16M
    7875                   4.0M
    7876                   8.3M
    7877                   9.6M
    7878                    11M
    7879                    20M
    7880                    14M
    7881                    17M
    7882                    15M
    7883                    13M
    7884                    11M
    7885                    46M
    7886                    11M
    7887                   1.6M
    7888                    28M
    7889     Varies with device
    7890                    11M
    7891                   4.9M
    7892                   8.6M
    7893                    23M
    7894                    24M
    7895                   1.6M
    7896                   8.4M
    7897                    97k
    7898                    20M
    7899                    44M
    7900                    88M
    7901                   4.8M
    7902                    32M
    7903                    25M
    7904                    27M
    7905                   7.3M
    7906                    14M
    7907                    15M
    7908     Varies with device
    7909                    16M
    7910                    25M
    7911                    14M
    7912                   2.9M
    7913                    18M
    7914                    35M
    7915                    25M
    7916                    14M
    7917                   6.5M
    7918                   5.5M
    7919                    28M
    7920                    17M
    7921                   7.3M
    7922                    24M
    7923                   2.1M
    7924                    24M
    7925                   2.8M
    7926                   7.2M
    7927                   8.6M
    7928                   2.3M
    7929                    12M
    7930     Varies with device
    7931     Varies with device
    7932                    70M
    7933                    34M
    7934                   2.6M
    7935                    35M
    7936                    24M
    7937                    88M
    7938                   7.0M
    7939                   3.9M
    7940     Varies with device
    7941                   3.8M
    7942                    26M
    7943                   5.0M
    7944                   4.7M
    7945                   5.6M
    7946                   5.3M
    7947                   2.2M
    7948                    26M
    7949                   5.6M
    7950                    26M
    7951                   8.6M
    7952                    21M
    7953                   4.8M
    7954                    15M
    7955                    20M
    7956                    20M
    7957                   5.5M
    7958                   5.7M
    7959                    11M
    7960                   4.7M
    7961                   4.1M
    7962                   6.9M
    7963                    12M
    7964     Varies with device
    7965                   5.9M
    7966                   2.2M
    7967                   7.1M
    7968                   7.9M
    7969                    25M
    7970                    26M
    7971                   4.9M
    7972                    25M
    7973                   8.3M
    7974                   6.0M
    7975                   3.0M
    7976                    19M
    7977                   4.1M
    7978                   6.0M
    7979                   4.6M
    7980                    26M
    7981                   4.4M
    7982                    10M
    7983                   5.0M
    7984                    16M
    7985                    22M
    7986                   8.1M
    7987                    21M
    7988                    13M
    7989                   4.8M
    7990                   6.1M
    7991                   4.3M
    7992                   6.2M
    7993                   3.5M
    7994                   2.4M
    7995                    38M
    7996                   9.5M
    7997                   516k
    7998                   837k
    7999                   2.0M
    8000                    17M
    8001                   6.8M
    8002                   1.1M
    8003                   780k
    8004                   3.7M
    8005                   961k
    8006                   2.6M
    8007                   4.8M
    8008                   7.8M
    8009                    22M
    8010     Varies with device
    8011                   1.5M
    8012                   9.6M
    8013     Varies with device
    8014                   2.4M
    8015     Varies with device
    8016                    55M
    8017                   269k
    8018                   3.7M
    8019                    21M
    8020                    96M
    8021                   1.1M
    8022                    20k
    8023                    12M
    8024     Varies with device
    8025                   498k
    8026                    13M
    8027                   4.2M
    8028                    11M
    8029     Varies with device
    8030                   600k
    8031                    18M
    8032                   3.0M
    8033                    14M
    8034                    72M
    8035     Varies with device
    8036                    54M
    8037                    12M
    8038                    37M
    8039                   6.5M
    8040                   6.6M
    8041                    50M
    8042                   4.3M
    8043                   7.6M
    8044                   5.5M
    8045                    94M
    8046                    35M
    8047                    55M
    8048                    11M
    8049                    12M
    8050                    21M
    8051                    27M
    8052                   5.3M
    8053                    30M
    8054                    10M
    8055                    11M
    8056                   9.8M
    8057                    68M
    8058                   3.8M
    8059                    33M
    8060                    24M
    8061                    35M
    8062                    20M
    8063                   3.4M
    8064                   2.5M
    8065                    32M
    8066                   5.8M
    8067                    18M
    8068                    20M
    8069                    32M
    8070                   6.9M
    8071                    12M
    8072                    17M
    8073                   592k
    8074                    50M
    8075                    19M
    8076                    55M
    8077                   1.0M
    8078                   2.0M
    8079                   7.5M
    8080                    16M
    8081                    10M
    8082     Varies with device
    8083     Varies with device
    8084                   1.1M
    8085                    81M
    8086                    15M
    8087                    13M
    8088                    20M
    8089                   7.1M
    8090                   4.5M
    8091                   1.3M
    8092                    10M
    8093                    29M
    8094                   749k
    8095                    20M
    8096                   8.1M
    8097                   3.8M
    8098                   1.6M
    8099                   6.1M
    8100                   4.9M
    8101     Varies with device
    8102                    20M
    8103                    15M
    8104                   9.3M
    8105                    12M
    8106                   3.9M
    8107                    61M
    8108                   5.8M
    8109                   5.0M
    8110                   4.4M
    8111     Varies with device
    8112                   642k
    8113                   9.4M
    8114                   3.3M
    8115     Varies with device
    8116                   6.4M
    8117                   4.0M
    8118                    46M
    8119                    12M
    8120                    30M
    8121     Varies with device
    8122                   1.4M
    8123                    15M
    8124                   1.8M
    8125                    19M
    8126     Varies with device
    8127                    12M
    8128                    21M
    8129                    25M
    8130                   3.4M
    8131                   8.7M
    8132                    19M
    8133                   9.8M
    8134                    18M
    8135                    18M
    8136                    15M
    8137                    17M
    8138                    43M
    8139                   9.3M
    8140                    11M
    8141                   2.0M
    8142                   2.2M
    8143                   4.4M
    8144                   5.4M
    8145                    12M
    8146                   1.4M
    8147                    17M
    8148                   881k
    8149                   3.3M
    8150                    17M
    8151                    13M
    8152                   7.3M
    8153                    23M
    8154                    12M
    8155                    72k
    8156     Varies with device
    8157                   2.2M
    8158     Varies with device
    8159                   3.5M
    8160                    18M
    8161                   8.6M
    8162                   4.8M
    8163                   7.3M
    8164                    13M
    8165                   5.3M
    8166                   6.2M
    8167                   656k
    8168                   5.0M
    8169                   601k
    8170                   8.3M
    8171     Varies with device
    8172                    73M
    8173                   7.4M
    8174                    22M
    8175                    26M
    8176                    14M
    8177                   7.3M
    8178                   3.1M
    8179                    11M
    8180     Varies with device
    8181                    12M
    8182                    13M
    8183                    33M
    8184     Varies with device
    8185                    49M
    8186                    31M
    8187     Varies with device
    8188                    34M
    8189                    44M
    8190                    11M
    8191                    55M
    8192     Varies with device
    8193                    59M
    8194     Varies with device
    8195                    60M
    8196                    25M
    8197                    56M
    8198                    20M
    8199                   2.3M
    8200                   221k
    8201                   1.5M
    8202                    48M
    8203                    31M
    8204                   7.5M
    8205                   228k
    8206                   2.6M
    8207                   2.3M
    8208                   2.8M
    8209                    16M
    8210                   2.2M
    8211                   4.5M
    8212                    33M
    8213                    72M
    8214                   3.3M
    8215                   6.6M
    8216                   5.2M
    8217                   8.0M
    8218                   108k
    8219                   3.5M
    8220                    11M
    8221                    10M
    8222                   9.4M
    8223                    10M
    8224     Varies with device
    8225                   940k
    8226                   8.7M
    8227                   6.3M
    8228                   4.1M
    8229                    37M
    8230                   3.2M
    8231                    11M
    8232                   2.6M
    8233                   3.0M
    8234                    14M
    8235                    47M
    8236                   2.5M
    8237                   2.3M
    8238                    49M
    8239                    44M
    8240                   3.0M
    8241                   2.3M
    8242                    15M
    8243                    10M
    8244                   7.6M
    8245                    59M
    8246     Varies with device
    8247                    36M
    8248                    32M
    8249                   9.5M
    8250                    95M
    8251                   5.8M
    8252                    35M
    8253                    63M
    8254                   9.7M
    8255                    91M
    8256                    63M
    8257     Varies with device
    8258                   3.1M
    8259                    88M
    8260                    12M
    8261                    92M
    8262                   4.1M
    8263     Varies with device
    8264                    23M
    8265                   7.2M
    8266                   6.4M
    8267                   3.4M
    8268                    91M
    8269                   6.1M
    8270                    22M
    8271                   6.9M
    8272                    91M
    8273                    72M
    8274                    10M
    8275                   2.2M
    8276                   8.5M
    8277                    12M
    8278                    21M
    8279                    11M
    8280                   2.3M
    8281                    16M
    8282                    32M
    8283                    17M
    8284                   1.3M
    8285                    18M
    8286                    10M
    8287                   1.2M
    8288                   8.7M
    8289                    42M
    8290     Varies with device
    8291                    38M
    8292                   5.3M
    8293     Varies with device
    8294     Varies with device
    8295                   5.8M
    8296                    96M
    8297                    21M
    8298     Varies with device
    8299                    10M
    8300                    80M
    8301                   6.1M
    8302                   1.5M
    8303                    31M
    8304                    26M
    8305                    20M
    8306                   2.9M
    8307     Varies with device
    8308     Varies with device
    8309                    38M
    8310                    36M
    8311                    14M
    8312                   176k
    8313                    44M
    8314                    82M
    8315                   3.3M
    8316                    18M
    8317                   8.3M
    8318                    33k
    8319                    14M
    8320     Varies with device
    8321                   8.3M
    8322                   9.9M
    8323                   8.0M
    8324                   1.5M
    8325                   3.6M
    8326                    15M
    8327                   6.7M
    8328                   663k
    8329                   1.4M
    8330                    20M
    8331                   6.0M
    8332                    14M
    8333                   1.5M
    8334                   2.3M
    8335                    27M
    8336                    76M
    8337                    15M
    8338                    32M
    8339                   6.3M
    8340                    24M
    8341                   1.4M
    8342                    36M
    8343                    15M
    8344                   3.0M
    8345                    14M
    8346                    10M
    8347                    15M
    8348                   5.2M
    8349                   2.9M
    8350                   3.3M
    8351                   5.7M
    8352                    36M
    8353                   5.7M
    8354                   2.9M
    8355                   3.3M
    8356                   4.5M
    8357     Varies with device
    8358                   1.6M
    8359                   2.8M
    8360     Varies with device
    8361                   1.1M
    8362                   4.1M
    8363                   1.1M
    8364                   6.8M
    8365                    28M
    8366                    32M
    8367                    12M
    8368                    22M
    8369                   1.4M
    8370                   1.2M
    8371                    20M
    8372                   7.6M
    8373                   2.9M
    8374     Varies with device
    8375                   2.5M
    8376                    11M
    8377                    34M
    8378                    21M
    8379                   4.5M
    8380                    22M
    8381                    34k
    8382                    14M
    8383                    12M
    8384     Varies with device
    8385                   1.3M
    8386                   3.1M
    8387                    11M
    8388                   5.9M
    8389                   3.0M
    8390                    12M
    8391                   1.1M
    8392                    26M
    8393                   4.5M
    8394                   5.9M
    8395                   5.7M
    8396     Varies with device
    8397                    37M
    8398     Varies with device
    8399                    77M
    8400                    43M
    8401                    20M
    8402                   8.0M
    8403                    17M
    8404                    10M
    8405                    34M
    8406                    81M
    8407     Varies with device
    8408                    35M
    8409                   100M
    8410                    48M
    8411                    59M
    8412     Varies with device
    8413                    14M
    8414                    10M
    8415                    16M
    8416                    24M
    8417                   2.3M
    8418                   4.1M
    8419                    28M
    8420                    27M
    8421                    27M
    8422                   6.7M
    8423                    23M
    8424     Varies with device
    8425                   4.9M
    8426                    85M
    8427                    41M
    8428                    14M
    8429                    94M
    8430                    45M
    8431                   942k
    8432                    26M
    8433                    99M
    8434                   5.9M
    8435                    36M
    8436                    52M
    8437                    63M
    8438                    26M
    8439     Varies with device
    8440                    35M
    8441     Varies with device
    8442                   5.4M
    8443                    53M
    8444                   4.0M
    8445                    51M
    8446                    20M
    8447                    39M
    8448                   5.0M
    8449                   5.7M
    8450                    29M
    8451                    87M
    8452                   7.1M
    8453                    61M
    8454     Varies with device
    8455                    30M
    8456                    52M
    8457                    44M
    8458     Varies with device
    8459                   4.6M
    8460                    90M
    8461     Varies with device
    8462                    91M
    8463                    30M
    8464                    20M
    8465                    28M
    8466                    16M
    8467                    16M
    8468                   7.9M
    8469                    57M
    8470                    16M
    8471                    25M
    8472                    34M
    8473                    29M
    8474                    87M
    8475                    38M
    8476                    18M
    8477                    19M
    8478                    70M
    8479                   3.3M
    8480                    14M
    8481                    66M
    8482                   4.3M
    8483                    48M
    8484                    31M
    8485                   3.3M
    8486                    25M
    8487                    11M
    8488                    17M
    8489                    16M
    8490                   3.5M
    8491                    33M
    8492                   6.4M
    8493                   3.9M
    8494                   1.5M
    8495                   6.6M
    8496                    23M
    8497                   2.4M
    8498                    35M
    8499                   1.1M
    8500                    11M
    8501                    13M
    8502                    34M
    8503                   7.1M
    8504                   4.2M
    8505                    17M
    8506                   3.0M
    8507                   5.3M
    8508                    30M
    8509                   3.3M
    8510                   2.5M
    8511                   2.0M
    8512     Varies with device
    8513                   6.3M
    8514     Varies with device
    8515                   2.3M
    8516                   7.2M
    8517     Varies with device
    8518                   2.2M
    8519                   6.2M
    8520                   1.7M
    8521                    10M
    8522                   3.2M
    8523                    30M
    8524                    11M
    8525                   2.4M
    8526                   1.7M
    8527                   3.3M
    8528                    13M
    8529                   3.2M
    8530                   7.9M
    8531                   5.2M
    8532                    22M
    8533                   1.6M
    8534                   9.4M
    8535                   5.4M
    8536                   1.1M
    8537                   3.3M
    8538                   5.5M
    8539                   4.2M
    8540                    10M
    8541                   5.0M
    8542                   2.9M
    8543                    30M
    8544                   5.4M
    8545                   3.7M
    8546                   2.9M
    8547                   3.0M
    8548                   4.2M
    8549                    11M
    8550                   3.0M
    8551                   6.3M
    8552                    20M
    8553                    97M
    8554                   259k
    8555                   4.9M
    8556                    11M
    8557                    36M
    8558                   164k
    8559                   3.0M
    8560                    16M
    8561                   458k
    8562                   245k
    8563                   5.2M
    8564                   5.7M
    8565                   3.3M
    8566                    27M
    8567                    10M
    8568                   7.8M
    8569                   8.5M
    8570                   5.9M
    8571                   629k
    8572                   1.7M
    8573                    92M
    8574                    21M
    8575                   1.2M
    8576                    33M
    8577                    35M
    8578     Varies with device
    8579                   2.4M
    8580                    11M
    8581                    23M
    8582                   4.0M
    8583     Varies with device
    8584                    17M
    8585                   4.1M
    8586                    28k
    8587                   1.8M
    8588                   3.4M
    8589                    32M
    8590                   288k
    8591                   4.2M
    8592                    17M
    8593                    18M
    8594                    37M
    8595                    15M
    8596                   3.8M
    8597                    21M
    8598                   4.2M
    8599                    28M
    8600                   2.3M
    8601                   775k
    8602                    10M
    8603                   5.9M
    8604                    33M
    8605                   7.5M
    8606                    19M
    8607                   3.8M
    8608                   4.0M
    8609     Varies with device
    8610                    62M
    8611     Varies with device
    8612                   7.6M
    8613                    27M
    8614                   2.1M
    8615                    15M
    8616                   3.4M
    8617                    30M
    8618                    18M
    8619                   5.3M
    8620                    31M
    8621                    50M
    8622                    22M
    8623                   2.6M
    8624                    86M
    8625                    71M
    8626                    32M
    8627                   9.2M
    8628     Varies with device
    8629                    16M
    8630     Varies with device
    8631     Varies with device
    8632                   4.2M
    8633                   3.5M
    8634     Varies with device
    8635                    60M
    8636     Varies with device
    8637                   5.9M
    8638                    78M
    8639     Varies with device
    8640                    12M
    8641                   4.9M
    8642     Varies with device
    8643     Varies with device
    8644                   3.0M
    8645                   6.3M
    8646     Varies with device
    8647                   8.4M
    8648                   4.0M
    8649                   1.9M
    8650                   201k
    8651                   5.0M
    8652                   4.0M
    8653                    30M
    8654     Varies with device
    8655                   9.0M
    8656                   8.5M
    8657     Varies with device
    8658     Varies with device
    8659                    13M
    8660                   3.4M
    8661                    20M
    8662                   4.4M
    8663                   8.3M
    8664                    19M
    8665                   6.4M
    8666                   2.4M
    8667     Varies with device
    8668                   5.3M
    8669                   5.2M
    8670                    70M
    8671                    41M
    8672                   1.9M
    8673                    39M
    8674                   5.8M
    8675                    11M
    8676                   6.9M
    8677                    15M
    8678                   4.7M
    8679                   8.2M
    8680                    18M
    8681                   4.4M
    8682                    15M
    8683                   9.4M
    8684                   3.0M
    8685                    12M
    8686                   3.4M
    8687                   5.1M
    8688                   2.0M
    8689                   3.2M
    8690                   5.2M
    8691                    11M
    8692                    14M
    8693                   7.7M
    8694                    12M
    8695                   3.8M
    8696                   8.1M
    8697                   5.4M
    8698                   5.7M
    8699                    10M
    8700                   8.7M
    8701                   3.8M
    8702                   5.6M
    8703                   3.8M
    8704                   4.1M
    8705                    11M
    8706                   5.7M
    8707                   3.8M
    8708                   9.1M
    8709                   2.5M
    8710                   6.7M
    8711                    11M
    8712                   7.7M
    8713                   7.0M
    8714                    43M
    8715                    13M
    8716                    20M
    8717                    59M
    8718                   3.9M
    8719                    27M
    8720                    16M
    8721                    63M
    8722                    12M
    8723                    30M
    8724                    44M
    8725                    43M
    8726                    17M
    8727                    42M
    8728                   6.3M
    8729                   2.6M
    8730                    20M
    8731                    16M
    8732                    13M
    8733                   6.3M
    8734                   2.7M
    8735                    56M
    8736                   1.1M
    8737                    26M
    8738                    56M
    8739                   785k
    8740                    25M
    8741                   4.7M
    8742                   6.0M
    8743                   4.9M
    8744                    18M
    8745     Varies with device
    8746                    66M
    8747                    41M
    8748                    96M
    8749                    76M
    8750                   4.2M
    8751                    36M
    8752                   3.9M
    8753                    24M
    8754                    35M
    8755                    23M
    8756                    13M
    8757                    19M
    8758     Varies with device
    8759                   7.6M
    8760                    96M
    8761                    35M
    8762     Varies with device
    8763                    81M
    8764                    95M
    8765                    10M
    8766                    21M
    8767                   4.7M
    8768                    27M
    8769                    25M
    8770                   3.4M
    8771                    33M
    8772                   5.7M
    8773                   3.1M
    8774     Varies with device
    8775                   4.3M
    8776                   4.5M
    8777                    46M
    8778                    33M
    8779                    58M
    8780                   9.3M
    8781                   6.2M
    8782                   6.7M
    8783                    41M
    8784                   6.0M
    8785                   9.5M
    8786                    60M
    8787                    31M
    8788                    36M
    8789                   4.3M
    8790                   1.5M
    8791                    68M
    8792                   1.2M
    8793                    12M
    8794                    22M
    8795                    45M
    8796                    62M
    8797                    80M
    8798                    34M
    8799                   3.5M
    8800                    67M
    8801                    58M
    8802                   5.1M
    8803                   5.5M
    8804                    12M
    8805                    19M
    8806                    19M
    8807                   8.8M
    8808                    11M
    8809                    12M
    8810                    32M
    8811                    22M
    8812                    38M
    8813     Varies with device
    8814     Varies with device
    8815                    60M
    8816                    35M
    8817                    15M
    8818                    43M
    8819     Varies with device
    8820                   4.4M
    8821                   2.1M
    8822     Varies with device
    8823     Varies with device
    8824                    23M
    8825                   2.3M
    8826                   636k
    8827                   4.2M
    8828                   4.6M
    8829                   7.9M
    8830                   1.4M
    8831                    21M
    8832                   916k
    8833                    28M
    8834                   1.6M
    8835                    84M
    8836                   6.6M
    8837                    12M
    8838                    14M
    8839                    38M
    8840                   5.0M
    8841                   2.9M
    8842                   1.4M
    8843                   8.4M
    8844                   3.0M
    8845                   2.6M
    8846                   3.7M
    8847                   100M
    8848                   5.8M
    8849                    28M
    8850                   4.5M
    8851                   8.9M
    8852                   3.1M
    8853                    22M
    8854                   9.9M
    8855                    49M
    8856                   1.6M
    8857                   8.9M
    8858                    22M
    8859                    23M
    8860                    94M
    8861                   6.3M
    8862                   1.1M
    8863                    95M
    8864                    21M
    8865                   9.5M
    8866                    24M
    8867                   1.5M
    8868                    24M
    8869                   7.9M
    8870                   5.3M
    8871                   1.2M
    8872                    12M
    8873                   5.7M
    8874                    13M
    8875                    24M
    8876                    54M
    8877                   3.9M
    8878     Varies with device
    8879     Varies with device
    8880                    51M
    8881                    49M
    8882                   1.3M
    8883                   994k
    8884                   3.6M
    8885                   4.1M
    8886                    66M
    8887                    32M
    8888                   2.9M
    8889                   8.9M
    8890                    47M
    8891                   2.1M
    8892                    12M
    8893                   7.8M
    8894                    15M
    8895                    25M
    8896                    14M
    8897                    10M
    8898                   8.9M
    8899                   4.8M
    8900                   3.2M
    8901                   4.7M
    8902     Varies with device
    8903                   2.0M
    8904                    14M
    8905                   4.7M
    8906                    22M
    8907                    14M
    8908                   4.3M
    8909     Varies with device
    8910                   5.2M
    8911                   309k
    8912                   8.7M
    8913                   485k
    8914                   2.5M
    8915                   9.2M
    8916                    39M
    8917                    96M
    8918     Varies with device
    8919                   6.9M
    8920                   9.8M
    8921                    16M
    8922                   6.2M
    8923                   914k
    8924                   8.2M
    8925     Varies with device
    8926                    13M
    8927                    48M
    8928                   2.8M
    8929     Varies with device
    8930                    31M
    8931                   1.3M
    8932                    29M
    8933                    17M
    8934                    19M
    8935                   2.6M
    8936                    71M
    8937                   2.6M
    8938                    14M
    8939                    44M
    8940                   5.1M
    8941                    11M
    8942                    13M
    8943                   2.6M
    8944                    65M
    8945                   8.7M
    8946                   903k
    8947                    25M
    8948                   4.0M
    8949                   1.7M
    8950                   2.4M
    8951                   8.5M
    8952                    26M
    8953                   2.0M
    8954                   2.5M
    8955                    18M
    8956                   2.9M
    8957                   1.3M
    8958                   1.7M
    8959                   2.7M
    8960                   608k
    8961                    22M
    8962                   4.2M
    8963                    41M
    8964                    65M
    8965                    16M
    8966                    15M
    8967                   9.3M
    8968                   4.2M
    8969                   5.1M
    8970                   8.9M
    8971                    24M
    8972                   4.0M
    8973     Varies with device
    8974                    31M
    8975                   9.5M
    8976                   8.2M
    8977                    17M
    8978                   1.8M
    8979     Varies with device
    8980                    30M
    8981                    31M
    8982                   8.5M
    8983                   6.3M
    8984                    10M
    8985                   4.8M
    8986                   6.0M
    8987                   5.8M
    8988                   6.3M
    8989                    79M
    8990                   6.3M
    8991                   2.4M
    8992                   5.0M
    8993                   2.0M
    8994                   4.4M
    8995                   8.0M
    8996     Varies with device
    8997                    39M
    8998                   3.5M
    8999                   3.9M
    9000                   9.8M
    9001                   5.9M
    9002                    15M
    9003                   6.3M
    9004                   3.0M
    9005                   2.2M
    9006                   4.2M
    9007                    16M
    9008                   1.5M
    9009     Varies with device
    9010                   500k
    9011                   3.4M
    9012                   3.3M
    9013                    13M
    9014                   2.5M
    9015                    10M
    9016                   3.0M
    9017                    54k
    9018                   4.8M
    9019                   2.9M
    9020                   3.0M
    9021                    40M
    9022                    11M
    9023                   6.6M
    9024                    16M
    9025     Varies with device
    9026                   1.6M
    9027                    48M
    9028                   4.6M
    9029                   2.8M
    9030     Varies with device
    9031                    23M
    9032                    26M
    9033                   2.5M
    9034                    24M
    9035                    56M
    9036                   4.0M
    9037                   2.5M
    9038                    40M
    9039                    73M
    9040                    31M
    9041     Varies with device
    9042                    45M
    9043                    42M
    9044                    39M
    9045                    48M
    9046                   7.2M
    9047                    10M
    9048                    14M
    9049                    11M
    9050                    21M
    9051                    32M
    9052                    47M
    9053     Varies with device
    9054                   2.1M
    9055                    16M
    9056                    33M
    9057                   3.4M
    9058                    17M
    9059                   1.6M
    9060                    42M
    9061                    11M
    9062                    48M
    9063                    40M
    9064     Varies with device
    9065                    20M
    9066                    60M
    9067                   2.6M
    9068                    22M
    9069                   3.2M
    9070     Varies with device
    9071                   5.3M
    9072                    22M
    9073                    10M
    9074                    43M
    9075                   4.8M
    9076                    20M
    9077                   1.9M
    9078                   3.3M
    9079                   1.2M
    9080                    54M
    9081                    26M
    9082     Varies with device
    9083                    67M
    9084                   6.4M
    9085                    50M
    9086                   2.5M
    9087                   1.3M
    9088                    21M
    9089                   7.4M
    9090                    30M
    9091                   3.9M
    9092                   9.8M
    9093                   4.5M
    9094                   9.8M
    9095                   3.2M
    9096                    47M
    9097                   6.0M
    9098                   2.5M
    9099                   2.3M
    9100     Varies with device
    9101                    14M
    9102                   7.0M
    9103                   562k
    9104                    14M
    9105                   4.7M
    9106                   4.5M
    9107                    10M
    9108                   3.0M
    9109                    18M
    9110                    32M
    9111                   847k
    9112                   2.8M
    9113                   9.9M
    9114                    17M
    9115                   2.5M
    9116                   3.4M
    9117                   6.8M
    9118                   957k
    9119                    29M
    9120                   3.6M
    9121     Varies with device
    9122                    28M
    9123                   3.1M
    9124                   2.5M
    9125                   1.6M
    9126                    23M
    9127                    15M
    9128                   6.1M
    9129                    10M
    9130                   2.4M
    9131                   4.6M
    9132                   3.3M
    9133                    10M
    9134                   3.8M
    9135                   3.1M
    9136                    33M
    9137                   6.2M
    9138                   2.1M
    9139                    18M
    9140                    37M
    9141                    58M
    9142                    22M
    9143                    71M
    9144     Varies with device
    9145                    63M
    9146                    49M
    9147                    15M
    9148     Varies with device
    9149                    67M
    9150                    45M
    9151                    29M
    9152                    57M
    9153     Varies with device
    9154                    43M
    9155                    12M
    9156                    96M
    9157                    26M
    9158                    79M
    9159                    98M
    9160                    19M
    9161                    28M
    9162                    50M
    9163                    39M
    9164                    30M
    9165                    99M
    9166                    58M
    9167     Varies with device
    9168                    92M
    9169                    43M
    9170                   100M
    9171                    72M
    9172                    37M
    9173                   3.1M
    9174                    33M
    9175                   3.3M
    9176                   5.6M
    9177                    16M
    9178                   5.4M
    9179                    11M
    9180                    30M
    9181                    31M
    9182                   3.3M
    9183                    30M
    9184                   6.1M
    9185                   5.2M
    9186                    18M
    9187                    16M
    9188                    25M
    9189                   2.3M
    9190                    13M
    9191     Varies with device
    9192                   8.8M
    9193                    15M
    9194                   2.4M
    9195                    70M
    9196                   3.9M
    9197                   3.9M
    9198                   9.0M
    9199                    21M
    9200                   7.3M
    9201                   1.8M
    9202                   1.4M
    9203                   948k
    9204                   5.0M
    9205                   2.2M
    9206                   688k
    9207                   2.2M
    9208                   2.5M
    9209                   7.3M
    9210                    90M
    9211                   3.1M
    9212                   811k
    9213                   4.0M
    9214                   4.1M
    9215                   2.8M
    9216     Varies with device
    9217                    16M
    9218                   4.3M
    9219                   3.8M
    9220                   270k
    9221                   3.7M
    9222     Varies with device
    9223                   3.0M
    9224                   2.3M
    9225                   3.8M
    9226                   2.7M
    9227                   2.2M
    9228                    28M
    9229                   1.5M
    9230                   5.5M
    9231                   4.0M
    9232                    48k
    9233                   7.7M
    9234                    26M
    9235                   8.6M
    9236                   4.3M
    9237                    17M
    9238                   3.8M
    9239                    15M
    9240                   3.4M
    9241                   5.0M
    9242                   2.3M
    9243                   1.6M
    9244                   2.7M
    9245                   8.1M
    9246                   3.8M
    9247                   6.3M
    9248                   7.4M
    9249                    12M
    9250                   7.9M
    9251                   5.4M
    9252                   4.0M
    9253                   3.1M
    9254                   5.5M
    9255                   8.4M
    9256                   3.9M
    9257                   329k
    9258                   3.0M
    9259                   4.2M
    9260                   3.8M
    9261                   5.3M
    9262                   6.0M
    9263                   4.3M
    9264                    30M
    9265                   4.6M
    9266                    17M
    9267                    14M
    9268                    39M
    9269                    13M
    9270                    35M
    9271                   6.9M
    9272                    14M
    9273                    23M
    9274                    15M
    9275                    11M
    9276                    14M
    9277                    14M
    9278                    17M
    9279                    17M
    9280                   9.8M
    9281                    15M
    9282                   4.4M
    9283                    43M
    9284                    19M
    9285     Varies with device
    9286                    36M
    9287                   9.4M
    9288                    29M
    9289                    10M
    9290                    23M
    9291                    10M
    9292                    33M
    9293                    20M
    9294                    33M
    9295                    14M
    9296                   2.3M
    9297                    54M
    9298                   2.5M
    9299                   4.3M
    9300                    11M
    9301                    15M
    9302                   1.6M
    9303                   2.0M
    9304                    31M
    9305     Varies with device
    9306                   3.6M
    9307                   3.7M
    9308                    14M
    9309                   6.6M
    9310                   5.1M
    9311                   4.6M
    9312                   5.3M
    9313                   6.8M
    9314                    53M
    9315                   7.5M
    9316                    23M
    9317                   1.5M
    9318                   2.4M
    9319                    25M
    9320                    30M
    9321     Varies with device
    9322                    13M
    9323                   6.5M
    9324                   4.4M
    9325                    23M
    9326                   5.0M
    9327                    16M
    9328                   7.4M
    9329                   1.1M
    9330                   1.1M
    9331                    38M
    9332                   523k
    9333                   118k
    9334                    18M
    9335                    16M
    9336                    61M
    9337                    56M
    9338                   3.1M
    9339                    12M
    9340                   6.0M
    9341                   4.0M
    9342                    30M
    9343                    17M
    9344                    51M
    9345     Varies with device
    9346                    31M
    9347                    42M
    9348                    99M
    9349                   6.4M
    9350                    22M
    9351                    63M
    9352                    11M
    9353                    99M
    9354                    21M
    9355                    14M
    9356                    46M
    9357                    18M
    9358                    61M
    9359                    99M
    9360                    31M
    9361                    26M
    9362                    26M
    9363                   7.0M
    9364                   3.3M
    9365                   3.9M
    9366                   8.7M
    9367                    62M
    9368                   3.5M
    9369                   6.4M
    9370     Varies with device
    9371                   921k
    9372                   7.1M
    9373                   874k
    9374                   2.3M
    9375                    55M
    9376                    41M
    9377                    52M
    9378                    32M
    9379                    13M
    9380                    44M
    9381                   3.4M
    9382                    94M
    9383                    23M
    9384                    48M
    9385     Varies with device
    9386                    26M
    9387                    19M
    9388                    44M
    9389                    33M
    9390                    35M
    9391                    65M
    9392                   9.2M
    9393                    23M
    9394                    48M
    9395                    37M
    9396                    20M
    9397                   4.8M
    9398                   5.7M
    9399                    29M
    9400                    42M
    9401                    48M
    9402                   3.7M
    9403                    61M
    9404                    66M
    9405                    82M
    9406                   1.6M
    9407                   1.8M
    9408     Varies with device
    9409                    85M
    9410                   981k
    9411                    12M
    9412                   4.5M
    9413                    14M
    9414                    15M
    9415                   3.8M
    9416                   784k
    9417     Varies with device
    9418                   7.7M
    9419                    13M
    9420                   1.5M
    9421                    13M
    9422                   7.1M
    9423                   3.2M
    9424                    21M
    9425                    23M
    9426                    27M
    9427                    19M
    9428                    45M
    9429                    15M
    9430     Varies with device
    9431                    13M
    9432                    33M
    9433                   2.1M
    9434                    16M
    9435                   3.3M
    9436                    20M
    9437                   5.2M
    9438                   8.7M
    9439     Varies with device
    9440                    99M
    9441                    77M
    9442                   3.8M
    9443                   6.0M
    9444                   5.0M
    9445                    55M
    9446                   5.3M
    9447                    37M
    9448                   2.3M
    9449                    12M
    9450                    50M
    9451                   280k
    9452                    31M
    9453                    19M
    9454                    69M
    9455                    25M
    9456                    38M
    9457                    26M
    9458                    13M
    9459                   6.8M
    9460                   4.1M
    9461                    48M
    9462     Varies with device
    9463                    68M
    9464                    45M
    9465     Varies with device
    9466                   9.3M
    9467                    44M
    9468                    41M
    9469                    65M
    9470                   4.4M
    9471                    46M
    9472                    23M
    9473                    30M
    9474                    28M
    9475                    47M
    9476                    59M
    9477                   1.6M
    9478                    14M
    9479                    43M
    9480                   4.1M
    9481     Varies with device
    9482                   3.8M
    9483                    47M
    9484                    24M
    9485                    31M
    9486                   2.7M
    9487                   7.3M
    9488                    59M
    9489                    48M
    9490                   1.3M
    9491                    12M
    9492                   5.0M
    9493                   6.6M
    9494                    55M
    9495                   2.8M
    9496     Varies with device
    9497                    10M
    9498                    30M
    9499                   3.7M
    9500                    18M
    9501                   7.4M
    9502                    17M
    9503                   1.3M
    9504                   3.7M
    9505                    26M
    9506                   5.0M
    9507                   3.3M
    9508                   3.9M
    9509                    29M
    9510                   5.6M
    9511                   3.0M
    9512                   1.8M
    9513                   5.8M
    9514                   5.7M
    9515                   2.6M
    9516                    38M
    9517                   6.2M
    9518                    11M
    9519                   1.5M
    9520                    62M
    9521                   4.0M
    9522                   3.1M
    9523                    56M
    9524                    30M
    9525                   1.9M
    9526                   2.0M
    9527                   4.5M
    9528                   4.3M
    9529                   9.9M
    9530                   2.0M
    9531                   2.3M
    9532                    29M
    9533                    99M
    9534                   5.1M
    9535                   2.7M
    9536                    64M
    9537                    49M
    9538                    79M
    9539                    47M
    9540                    25M
    9541                    49M
    9542                    36M
    9543                    22M
    9544                   7.0M
    9545                    60M
    9546                    13M
    9547     Varies with device
    9548                   3.2M
    9549                    31M
    9550                    50M
    9551                    60M
    9552                    16M
    9553                    25M
    9554                   3.6M
    9555                   3.5M
    9556                    12M
    9557     Varies with device
    9558                    33M
    9559                    24M
    9560                   4.1M
    9561                    49M
    9562     Varies with device
    9563                   3.2M
    9564                    73M
    9565                    22M
    9566                   4.4M
    9567                    32M
    9568     Varies with device
    9569                    63M
    9570     Varies with device
    9571                    15M
    9572                    16M
    9573                   6.4M
    9574                    25M
    9575     Varies with device
    9576                    52M
    9577                    21M
    9578                   2.3M
    9579     Varies with device
    9580                    15M
    9581                   6.8M
    9582                    40M
    9583                    88M
    9584                    59M
    9585     Varies with device
    9586                    24k
    9587                    46M
    9588                    39M
    9589                   6.9M
    9590                    14M
    9591                    52M
    9592                    14M
    9593                    37M
    9594     Varies with device
    9595                    24M
    9596                    28M
    9597     Varies with device
    9598                   4.4M
    9599                    52M
    9600                    51M
    9601                    76M
    9602                   4.4M
    9603     Varies with device
    9604                    26M
    9605                    67M
    9606                    46M
    9607                    78M
    9608                    33M
    9609                    26M
    9610     Varies with device
    9611     Varies with device
    9612                    60M
    9613                   3.0M
    9614                    90M
    9615                    28M
    9616                    60M
    9617                    40M
    9618                    25M
    9619                    41M
    9620     Varies with device
    9621     Varies with device
    9622                    13M
    9623     Varies with device
    9624     Varies with device
    9625     Varies with device
    9626                    42M
    9627                    35M
    9628                    18M
    9629                    53M
    9630                    31M
    9631                   4.3M
    9632     Varies with device
    9633                    11M
    9634                   8.5M
    9635                    21M
    9636     Varies with device
    9637                    43M
    9638                    10M
    9639                   3.4M
    9640     Varies with device
    9641                    15M
    9642                   7.1M
    9643     Varies with device
    9644                    11M
    9645                   1.5M
    9646                    21M
    9647                    34M
    9648                    11M
    9649                   8.9M
    9650                    19M
    9651                   4.2M
    9652                   1.1M
    9653                    11M
    9654                    14M
    9655     Varies with device
    9656                    30M
    9657                    24M
    9658                   518k
    9659     Varies with device
    9660                   6.3M
    9661                    41M
    9662                    18M
    9663                   5.2M
    9664                   6.7M
    9665                   3.1M
    9666                   2.0M
    9667                    36M
    9668                    60M
    9669                    74M
    9670                    11M
    9671                    92M
    9672                    83M
    9673                    54M
    9674                    42M
    9675                    23M
    9676                    31M
    9677                    98M
    9678                    69M
    9679                   754k
    9680                    37M
    9681                    20M
    9682                    77M
    9683                    73M
    9684                    33M
    9685                   2.7M
    9686                    57M
    9687                    83M
    9688                    73M
    9689     Varies with device
    9690                    44M
    9691                   2.8M
    9692                    13M
    9693                   2.1M
    9694                    28M
    9695                   2.3M
    9696                   892k
    9697                    70M
    9698                   2.3M
    9699                    20M
    9700     Varies with device
    9701                   8.8M
    9702                    24M
    9703                    21M
    9704                    15M
    9705                   154k
    9706                    21M
    9707                   4.8M
    9708                    20M
    9709                    31M
    9710                    16M
    9711                    12M
    9712                    27M
    9713                   6.3M
    9714                    11M
    9715                   2.0M
    9716                   1.5M
    9717                    20M
    9718                    39M
    9719                   3.2M
    9720                    27M
    9721                   7.2M
    9722                    23M
    9723                    47M
    9724                    31M
    9725                    21M
    9726                    34M
    9727                   5.3M
    9728                    37M
    9729                   5.7M
    9730                    10M
    9731                    52M
    9732                    69M
    9733                    56M
    9734                    53M
    9735                    62M
    9736                    76M
    9737                    71M
    9738                    72M
    9739                    28M
    9740                   2.8M
    9741                    51M
    9742                    28M
    9743                    40M
    9744                    27M
    9745     Varies with device
    9746                    37M
    9747                    46M
    9748                    50M
    9749                    49M
    9750                    58M
    9751                    41M
    9752                    73M
    9753                    33M
    9754                    46M
    9755                    19M
    9756                    32M
    9757                    30M
    9758                    92M
    9759                    55M
    9760                    45M
    9761                    47M
    9762                    85M
    9763                    48M
    9764                    77M
    9765                    30M
    9766                    28M
    9767                    35M
    9768                   5.8M
    9769                    39M
    9770                    25M
    9771                    32M
    9772                   8.9M
    9773                    84M
    9774                    60M
    9775                   5.3M
    9776                    39M
    9777                    20M
    9778                    20M
    9779                    17M
    9780     Varies with device
    9781                    18k
    9782                    42M
    9783                    29M
    9784     Varies with device
    9785                   6.6M
    9786                   3.2M
    9787                    33k
    9788                   4.4M
    9789                   3.4M
    9790                   860k
    9791                   1.2M
    9792                   1.7M
    9793                   2.8M
    9794                   364k
    9795                   2.2M
    9796                   1.9M
    9797                   1.7M
    9798                   3.0M
    9799                   387k
    9800                   2.7M
    9801                   2.0M
    9802                   5.0M
    9803                   375k
    9804                    11M
    9805                   626k
    9806                    11M
    9807                   2.9M
    9808                   6.5M
    9809                   2.0M
    9810                   4.2M
    9811     Varies with device
    9812                   6.7M
    9813     Varies with device
    9814                   9.2M
    9815                   4.7M
    9816                    15M
    9817                   161k
    9818                    24M
    9819                   1.7M
    9820                   2.7M
    9821                   5.9M
    9822                   9.2M
    9823                    13M
    9824     Varies with device
    9825                   9.2M
    9826     Varies with device
    9827                   6.3M
    9828                   3.8M
    9829                   4.2M
    9830                   5.7M
    9831                    18M
    9832                   9.9M
    9833     Varies with device
    9834                    12M
    9835                   5.3M
    9836                   9.9M
    9837                    63M
    9838     Varies with device
    9839     Varies with device
    9840     Varies with device
    9841     Varies with device
    9842     Varies with device
    9843     Varies with device
    9844                    13M
    9845     Varies with device
    9846                    31M
    9847     Varies with device
    9848     Varies with device
    9849                    35M
    9850                    61M
    9851     Varies with device
    9852     Varies with device
    9853                    62M
    9854                    93M
    9855                    93M
    9856                    79k
    9857                   5.0M
    9858     Varies with device
    9859                    33M
    9860                   8.7M
    9861                    61M
    9862     Varies with device
    9863                   6.1M
    9864                    26M
    9865                    37M
    9866                    19M
    9867                   2.3M
    9868                   4.4M
    9869                    32M
    9870                   879k
    9871                   5.3M
    9872                    17M
    9873                   4.2M
    9874                   5.3M
    9875                   3.4M
    9876                    12M
    9877                   1.8M
    9878                    12M
    9879                   2.5M
    9880                   3.9M
    9881                   2.0M
    9882                   7.2M
    9883                    13M
    9884                    20M
    9885                    16M
    9886                    22M
    9887                   3.5M
    9888     Varies with device
    9889                   2.3M
    9890                    65M
    9891                    21M
    9892                    21M
    9893                    20M
    9894                    12M
    9895                   2.0M
    9896                   2.7M
    9897                    13M
    9898                   7.0M
    9899                   9.4M
    9900                    11M
    9901     Varies with device
    9902                   1.2M
    9903                    18M
    9904                   7.5M
    9905                   2.6M
    9906                    12M
    9907                   3.1M
    9908                    39k
    9909                   4.3M
    9910                    45M
    9911                    15M
    9912                   2.5M
    9913                   5.4M
    9914                   5.2M
    9915                   1.5M
    9916                   970k
    9917                   1.4M
    9918                   2.0M
    9919                    34M
    9920                   2.1M
    9921     Varies with device
    9922                   9.7M
    9923                   2.2M
    9924                   1.6M
    9925                    10M
    9926                    42M
    9927                    13M
    9928                   2.7M
    9929                   8.7M
    9930                    32M
    9931                    10M
    9932                    67M
    9933                    17M
    9934                    40M
    9935                    21M
    9936                    15M
    9937                    12M
    9938                   9.7M
    9939                    31M
    9940                   4.0M
    9941     Varies with device
    9942                   8.6M
    9943                    25M
    9944                    26M
    9945                   6.4M
    9946                    88M
    9947                   5.8M
    9948                    41M
    9949                   2.5M
    9950     Varies with device
    9951                   6.6M
    9952                    24M
    9953                   9.8M
    9954                    32M
    9955                    72M
    9956                    19M
    9957                   8.1M
    9958                   6.2M
    9959                   8.4M
    9960                    19M
    9961                   8.5M
    9962                    43M
    9963                    23M
    9964                    48M
    9965                    20M
    9966                   5.2M
    9967                   3.9M
    9968                    52M
    9969                   1.3M
    9970                    25M
    9971                    11M
    9972                    16M
    9973                   1.0M
    9974                    61M
    9975                    15M
    9976     Varies with device
    9977                    54M
    9978                    19M
    9979                    19M
    9980     Varies with device
    9981                    33M
    9982     Varies with device
    9983                    64M
    9984                    72M
    9985                    45M
    9986     Varies with device
    9987                    27M
    9988                    69M
    9989                   4.9M
    9990                   4.4M
    9991                   8.6M
    9992                    13M
    9993                   8.7M
    9994                    12M
    9995                    29M
    9996     Varies with device
    9997                   5.8M
    9998                    36M
    9999                    17M
    10000                   17M
    10001                  1.9M
    10002    Varies with device
    10003                   12M
    10004                  7.1M
    10005                   96M
    10006                   21M
    10007                  6.5M
    10008    Varies with device
    10009                   62M
    10010    Varies with device
    10011                   82M
    10012    Varies with device
    10013                   69M
    10014                   22M
    10015                   34M
    10016                   63M
    10017    Varies with device
    10018    Varies with device
    10019                   14M
    10020                  1.8M
    10021                   95M
    10022                   30M
    10023    Varies with device
    10024                   16M
    10025    Varies with device
    10026                  4.3M
    10027                   67M
    10028                   12M
    10029                  8.5M
    10030                   48M
    10031                   62M
    10032                   95M
    10033                  5.2M
    10034    Varies with device
    10035                   23M
    10036                  2.8M
    10037    Varies with device
    10038                   13M
    10039                  1.1M
    10040                  4.6M
    10041                  170k
    10042                  1.1M
    10043                  1.0M
    10044    Varies with device
    10045                   11M
    10046                  141k
    10047                  160k
    10048    Varies with device
    10049                   86M
    10050                  144k
    10051                  141k
    10052                  143k
    10053                  6.3M
    10054                   17M
    10055                  7.9M
    10056                  190k
    10057                  6.0M
    10058                  3.5M
    10059                   26M
    10060                   93M
    10061                   21M
    10062                  2.3M
    10063    Varies with device
    10064                   64M
    10065                  8.1M
    10066                  376k
    10067                   13M
    10068                  8.0M
    10069                   29M
    10070                   42M
    10071                   19M
    10072                   17k
    10073                  2.5M
    10074                  3.6M
    10075                  3.9M
    10076                   54M
    10077                  6.8M
    10078                   38M
    10079                   44M
    10080                   30M
    10081                  8.4M
    10082                   27M
    10083                   10M
    10084                   15M
    10085                   27M
    10086                  5.1M
    10087                  3.4M
    10088                   10M
    10089                   19M
    10090                   54M
    10091                  3.1M
    10092                  2.7M
    10093                  1.7M
    10094                  8.4M
    10095                  1.2M
    10096                   19M
    10097                   12M
    10098                   44M
    10099                   26M
    10100                   99M
    10101                   19M
    10102                   18M
    10103                  3.3M
    10104                   30M
    10105                   24M
    10106                  4.5M
    10107                   15M
    10108                   10M
    10109                   18M
    10110                   13M
    10111                  4.5M
    10112                  3.3M
    10113                   21M
    10114                   10M
    10115                   36M
    10116                   33M
    10117                  4.3M
    10118                  9.6M
    10119                  656k
    10120                   61M
    10121                   14M
    10122                  5.4M
    10123                  1.3M
    10124    Varies with device
    10125                  2.8M
    10126                   47M
    10127    Varies with device
    10128    Varies with device
    10129                  7.6M
    10130                  4.6M
    10131                 10.0M
    10132                  4.6M
    10133                   13M
    10134                  1.2M
    10135                  2.9M
    10136                   10M
    10137                  4.8M
    10138                   36M
    10139                  3.5M
    10140    Varies with device
    10141                   50M
    10142                   51k
    10143                   11M
    10144                   29M
    10145                   46M
    10146                   46M
    10147                  1.4M
    10148                  1.2M
    10149                  228k
    10150    Varies with device
    10151                  9.8M
    10152                   39M
    10153                  193k
    10154                   12M
    10155                  192k
    10156                   57M
    10157                  2.1M
    10158                  3.9M
    10159                   11M
    10160                  2.3M
    10161                  8.5M
    10162                  1.4M
    10163                  473k
    10164                  2.4M
    10165                  3.9M
    10166                   68M
    10167                  1.6M
    10168    Varies with device
    10169                  1.8M
    10170                  4.0M
    10171                  1.5M
    10172                   26M
    10173                  3.0M
    10174                   71M
    10175                  246k
    10176                  3.6M
    10177                  1.3M
    10178                   24M
    10179                   68M
    10180                  1.3M
    10181                   44M
    10182                  7.0M
    10183                  5.4M
    10184                  7.6M
    10185                   14M
    10186                   71M
    10187                  2.6M
    10188                   10M
    10189                   48M
    10190                   25M
    10191                  6.3M
    10192                   96M
    10193                   10M
    10194                   47M
    10195                  3.2M
    10196    Varies with device
    10197    Varies with device
    10198                  5.8M
    10199                   23M
    10200    Varies with device
    10201                  1.1M
    10202                  4.6M
    10203    Varies with device
    10204                  6.3M
    10205    Varies with device
    10206                  4.3M
    10207                  4.8M
    10208                  4.2M
    10209                  3.6M
    10210                  4.0M
    10211                  3.9M
    10212                  8.7M
    10213                  9.9M
    10214                  5.1M
    10215                  1.8M
    10216                  1.5M
    10217                  7.0M
    10218    Varies with device
    10219                   18M
    10220                  3.4M
    10221                  3.3M
    10222                  1.5M
    10223                  5.1M
    10224                   10M
    10225                  1.6M
    10226                  3.2M
    10227                  3.0M
    10228                   10M
    10229                  2.8M
    10230                  2.5M
    10231                  1.3M
    10232                   11M
    10233                   15M
    10234                  3.1M
    10235                  2.8M
    10236                  3.7M
    10237                  6.1M
    10238                  4.5M
    10239                   30M
    10240                  6.2M
    10241                   20M
    10242                   38M
    10243                   20M
    10244                  3.1M
    10245                  6.4M
    10246                   71M
    10247                   11M
    10248                   11M
    10249                  2.6M
    10250                   33M
    10251                   21M
    10252                   41M
    10253                   12M
    10254                   21M
    10255                  2.8M
    10256                  3.6M
    10257                   14M
    10258                   29M
    10259                   76M
    10260                  5.7M
    10261                   17M
    10262                   28M
    10263                  4.6M
    10264    Varies with device
    10265                   19M
    10266                  3.1M
    10267                   79M
    10268                   36M
    10269                   17M
    10270                  7.8M
    10271                   89M
    10272                   49M
    10273                   12M
    10274                  3.0M
    10275                   57M
    10276                   16M
    10277                   13M
    10278                  5.9M
    10279                   40M
    10280                   14M
    10281    Varies with device
    10282                   73k
    10283                  3.5M
    10284                   17M
    10285                  7.9M
    10286                   19M
    10287                   30M
    10288                   12M
    10289                   42M
    10290                  7.4M
    10291                  9.1M
    10292                  7.5M
    10293                  4.8M
    10294                  7.4M
    10295                  4.9M
    10296                  9.1M
    10297                  4.2M
    10298                  2.7M
    10299                   75M
    10300                  3.9M
    10301                  2.3M
    10302                   35M
    10303                  3.5M
    10304                  9.1M
    10305                  1.5M
    10306                   26M
    10307                  7.4M
    10308                  2.1M
    10309                  4.7M
    10310                  3.4M
    10311                  4.2M
    10312                  4.6M
    10313                  7.1M
    10314                   21M
    10315                  2.7M
    10316                  4.4M
    10317                   14M
    10318                   17M
    10319    Varies with device
    10320                  9.0M
    10321                  4.5M
    10322                   21M
    10323                   15M
    10324                   21M
    10325                  8.0M
    10326                   11M
    10327                   53M
    10328                   21M
    10329                  2.9M
    10330                   20M
    10331                  2.9M
    10332                   20M
    10333                   47M
    10334                   12M
    10335                   12M
    10336                   17M
    10337                   10M
    10338                   26M
    10339                  9.2M
    10340                  9.8M
    10341                   17M
    10342                  658k
    10343                   30M
    10344                  1.8M
    10345                   97M
    10346                  2.9M
    10347                   15M
    10348                  5.3M
    10349                  8.3M
    10350                   20M
    10351                  3.0M
    10352                  5.6M
    10353                  7.9M
    10354                   22M
    10355                  5.0M
    10356                   11M
    10357                   15M
    10358                  7.5M
    10359                  4.1M
    10360                   13M
    10361                   12M
    10362                   60M
    10363                   25M
    10364                  4.6M
    10365                  1.7M
    10366                  6.5M
    10367                   36M
    10368                   17M
    10369                  6.5M
    10370                   39M
    10371                   46M
    10372                   52M
    10373                  7.7M
    10374                   30M
    10375                   17M
    10376                   39M
    10377                   30M
    10378                   12M
    10379                   45M
    10380                   29M
    10381                   48M
    10382                   50M
    10383    Varies with device
    10384                   66M
    10385                   27M
    10386                   51M
    10387                   69M
    10388                   21M
    10389                   14M
    10390                   19M
    10391                   61M
    10392                   28M
    10393                   49M
    10394                   94M
    10395                   23M
    10396                   94M
    10397                  2.5M
    10398                   13M
    10399                   34M
    10400                   33M
    10401                   16M
    10402                  992k
    10403                   18M
    10404                  8.1M
    10405                   15M
    10406                  8.2M
    10407                  8.1M
    10408                   27M
    10409    Varies with device
    10410                  4.1M
    10411                   20M
    10412                  6.0M
    10413                   17M
    10414                  4.7M
    10415                   14M
    10416                  5.8M
    10417                  9.9M
    10418                  9.9M
    10419                   49M
    10420                   14M
    10421                  7.6M
    10422                  7.8M
    10423                   46M
    10424                   29M
    10425                  3.6M
    10426                   49M
    10427                   29M
    10428                  8.2M
    10429                   54M
    10430                   18M
    10431                   56M
    10432                   26M
    10433                  3.3M
    10434                   40M
    10435                   14M
    10436                   24M
    10437                   37M
    10438    Varies with device
    10439    Varies with device
    10440                   22M
    10441                   16M
    10442                  2.0M
    10443                  3.5M
    10444                  8.5M
    10445                  253k
    10446                  957k
    10447    Varies with device
    10448                  420k
    10449                   13M
    10450                  2.4M
    10451                  3.4M
    10452                  5.1M
    10453    Varies with device
    10454                  9.7M
    10455                   72k
    10456    Varies with device
    10457                  2.4M
    10458                   35M
    10459                  3.9M
    10460                   26k
    10461                   29k
    10462                  5.8M
    10463                  2.8M
    10464                  5.8M
    10465                  2.1M
    10466                  1.4M
    10467                  3.9M
    10468                  7.5M
    10469                   58M
    10470                  4.0M
    10471                  404k
    10472                  3.0M
    10473                  4.1M
    10474                   14M
    10475                  2.6M
    10476                  7.6M
    10477                   11M
    10478                  8.0M
    10479                  2.3M
    10480                   49M
    10481                   43M
    10482                   10M
    10483                   16M
    10484                   24M
    10485                  6.2M
    10486                   27M
    10487                   46M
    10488                  2.4M
    10489                   36M
    10490                   17M
    10491                  8.0M
    10492                   37M
    10493                   33M
    10494                   15M
    10495                   15M
    10496                   13M
    10497                   11M
    10498                   25M
    10499                   16M
    10500                   15M
    10501                   15M
    10502    Varies with device
    10503                   43M
    10504                   43M
    10505                   36M
    10506                   23M
    10507                   99M
    10508                   46M
    10509    Varies with device
    10510                   20M
    10511                   45M
    10512                  2.9M
    10513                   21M
    10514                   35M
    10515                   41M
    10516                  2.2M
    10517                  2.5M
    10518                   35M
    10519                  1.5M
    10520                   54M
    10521                   44M
    10522                   27M
    10523                  8.5M
    10524                   22M
    10525                   26M
    10526                   63M
    10527                   20M
    10528                   26M
    10529                  3.5M
    10530                   26M
    10531                   10M
    10532                  5.5M
    10533                   26M
    10534                   26M
    10535                   26M
    10536                   26M
    10537                   26M
    10538                   26M
    10539                   26M
    10540                  2.4M
    10541                   26M
    10542                   26M
    10543                   26M
    10544                   26M
    10545                   26M
    10546                   26M
    10547                   13M
    10548                   26M
    10549                   25M
    10550                   26M
    10551                   26M
    10552                   15M
    10553                   18M
    10554                   21M
    10555                   26M
    10556                   17M
    10557                   26M
    10558                   12M
    10559                   26M
    10560                   26M
    10561                   26M
    10562                   26M
    10563                   26M
    10564                  2.6M
    10565                   27M
    10566                   15M
    10567                   26M
    10568                   17M
    10569                   45M
    10570                   24M
    10571                   11M
    10572                  3.1M
    10573                   30M
    10574                  3.2M
    10575                  5.6M
    10576                  7.1M
    10577                  7.1M
    10578                   31M
    10579                   34M
    10580                  9.2M
    10581                   36M
    10582                   15M
    10583                  2.0M
    10584                  6.1M
    10585    Varies with device
    10586                   22M
    10587                  4.4M
    10588                   86M
    10589                   13M
    10590                   12M
    10591                   41M
    10592                  8.2M
    10593                  8.0M
    10594                   15M
    10595                  470k
    10596                  5.3M
    10597                  9.4M
    10598                  2.2M
    10599                   43M
    10600                  226k
    10601                   20M
    10602                   21M
    10603                   37M
    10604                   26M
    10605                  3.4M
    10606                   22M
    10607                  7.8M
    10608                  3.4M
    10609                   28M
    10610                  7.8M
    10611                  2.6M
    10612                  3.9M
    10613                  7.6M
    10614                  8.1M
    10615                  9.5M
    10616                  1.8M
    10617                   16M
    10618                  3.2M
    10619                   45M
    10620                   14M
    10621                   12M
    10622                   49M
    10623                   13M
    10624                  9.9M
    10625                  3.9M
    10626                  7.6M
    10627                   28M
    10628                   12M
    10629                   69M
    10630                   34M
    10631                  9.2M
    10632                  5.4M
    10633                  8.7M
    10634                   18M
    10635                  5.7M
    10636                   12M
    10637                   15M
    10638                  1.6M
    10639                   38M
    10640                   11M
    10641                   14M
    10642    Varies with device
    10643                   30M
    10644                   13M
    10645    Varies with device
    10646                   19M
    10647    Varies with device
    10648                   10M
    10649                   45M
    10650                  5.3M
    10651                   19M
    10652                  7.0M
    10653                  5.6M
    10654                   11M
    10655                   12M
    10656                  9.2M
    10657                   11M
    10658                   13M
    10659                  3.3M
    10660                   14M
    10661                  8.8M
    10662                  7.3M
    10663                   14M
    10664                  8.5M
    10665                  3.3M
    10666                  9.0M
    10667                  240k
    10668                  8.2M
    10669                  7.9M
    10670                  1.6M
    10671                   34M
    10672                  2.1M
    10673                   59M
    10674                  5.2M
    10675                   89k
    10676                  3.6M
    10677                  234k
    10678                  257k
    10679    Varies with device
    10680                  9.8M
    10681    Varies with device
    10682                   36M
    10683                   20M
    10684                   28M
    10685                  3.2M
    10686                   50M
    10687                   41M
    10688                  364k
    10689                  3.1M
    10690                  861k
    10691                  1.6M
    10692                   15M
    10693                   17M
    10694                   16M
    10695                   15M
    10696                   11M
    10697                   16M
    10698                   15M
    10699                   19M
    10700                   16M
    10701                  2.8M
    10702                  1.3M
    10703                   21M
    10704                   18M
    10705                  5.2M
    10706                  2.0M
    10707    Varies with device
    10708                  9.7M
    10709                   17M
    10710                  4.3M
    10711                   54M
    10712    Varies with device
    10713    Varies with device
    10714                  7.2M
    10715                  1.4M
    10716                   11M
    10717                   49M
    10718                   15M
    10719                   25M
    10720                  4.0M
    10721                   16M
    10722                   11M
    10723                   40M
    10724                  3.7M
    10725    Varies with device
    10726                   24M
    10727                   16M
    10728                  8.1M
    10729                  6.3M
    10730                   60M
    10731                   46M
    10732                  467k
    10733                  1.4M
    10734                   22M
    10735                  157k
    10736                  2.6M
    10737                   44M
    10738                   11M
    10739                   39M
    10740                  4.4M
    10741                  1.7M
    10742                  7.9M
    10743                  1.2M
    10744                  2.0M
    10745                  5.8M
    10746                   61M
    10747                  3.8M
    10748                  3.3M
    10749                   29M
    10750                  3.3M
    10751                   44k
    10752                   26M
    10753                   12M
    10754                  3.3M
    10755                  676k
    10756                  2.5M
    10757                   72M
    10758                  4.3M
    10759                   67k
    10760                  2.4M
    10761                  8.0M
    10762                   11M
    10763                  552k
    10764                  885k
    10765    Varies with device
    10766                  7.0M
    10767                   16M
    10768                   24M
    10769                   12M
    10770                   41M
    10771                  2.4M
    10772                  3.9M
    10773                  8.9M
    10774                   36M
    10775                  9.0M
    10776                   24M
    10777                  2.2M
    10778                   38M
    10779                   75M
    10780                   50M
    10781                   44M
    10782                   11M
    10783                   72M
    10784                   84M
    10785                  9.5M
    10786                  2.8M
    10787                   48M
    10788                   20M
    10789                   48M
    10790                   20M
    10791                   38M
    10792                   16M
    10793                   78M
    10794                  5.7M
    10795                  4.0M
    10796                  7.8M
    10797                   46M
    10798                 1020k
    10799                  6.8M
    10800                   12M
    10801                   19M
    10802                   28M
    10803                   81M
    10804                   17M
    10805                   15M
    10806                   42M
    10807                  4.2M
    10808                  1.0M
    10809                   24M
    10810                   21M
    10811                  3.9M
    10812                   13M
    10813                  2.7M
    10814                   31M
    10815                  4.9M
    10816                  6.8M
    10817                  8.0M
    10818                  1.5M
    10819                  3.6M
    10820                  8.6M
    10821                  2.5M
    10822                  3.1M
    10823                  2.9M
    10824                   82M
    10825                  7.7M
    10826    Varies with device
    10827                   13M
    10828                   13M
    10829                  7.4M
    10830                  2.3M
    10831                  9.8M
    10832                  582k
    10833                  619k
    10834                  2.6M
    10835                  9.6M
    10836                   53M
    10837                  3.6M
    10838                  9.5M
    10839    Varies with device
    10840                   19M
    Name: Size, dtype: object




```python
# let's apply this funtion
df['Size']=df['Size'].apply(convert_size)
```


```python
df['Size']
```




    0                19922944.0
    1                14680064.0
    2                 9122611.2
    3                26214400.0
    4                 2936012.8
    5                 5872025.6
    6                19922944.0
    7                30408704.0
    8                34603008.0
    9                 3250585.6
    10               29360128.0
    11               12582912.0
    12               20971520.0
    13               22020096.0
    14               38797312.0
    15                2831155.2
    16                5767168.0
    17               17825792.0
    18               40894464.0
    19               32505856.0
    20               14680064.0
    21               12582912.0
    22                4404019.2
    23                7340032.0
    24               24117248.0
    25                6291456.0
    26               26214400.0
    27                6396313.6
    28                4823449.6
    29                4404019.2
    30                9646899.2
    31                5452595.2
    32               11534336.0
    33               11534336.0
    34                4404019.2
    35                9646899.2
    36               25165824.0
    37       Varies with device
    38               11534336.0
    39                9856614.4
    40               15728640.0
    41               10485760.0
    42       Varies with device
    43                1258291.2
    44               12582912.0
    45               25165824.0
    46               27262976.0
    47                8388608.0
    48                8283750.4
    49               26214400.0
    50               58720256.0
    51               59768832.0
    52       Varies with device
    53               36700160.0
    54               34603008.0
    55               34603008.0
    56                5872025.6
    57               56623104.0
    58                 205824.0
    59                3774873.6
    60                5976883.2
    61               17825792.0
    62                9017753.6
    63                2516582.4
    64               28311552.0
    65                2831155.2
    66                2621440.0
    67       Varies with device
    68       Varies with device
    69                7340032.0
    70               36700160.0
    71               16777216.0
    72               16777216.0
    73       Varies with device
    74               17825792.0
    75                3565158.4
    76                9332326.4
    77                4089446.4
    78                3040870.4
    79               39845888.0
    80               33554432.0
    81               38797312.0
    82               15728640.0
    83                5662310.4
    84               18874368.0
    85       Varies with device
    86               39845888.0
    87                1153433.6
    88       Varies with device
    89       Varies with device
    90                8283750.4
    91               36700160.0
    92       Varies with device
    93               17825792.0
    94               19922944.0
    95               14680064.0
    96                2306867.2
    97                4718592.0
    98               17825792.0
    99               14680064.0
    100              10276044.8
    101              22020096.0
    102      Varies with device
    103              54525952.0
    104              14680064.0
    105              26214400.0
    106               9437184.0
    107      Varies with device
    108              12582912.0
    109      Varies with device
    110              36700160.0
    111               7025459.2
    112              31457280.0
    113               5976883.2
    114               3040870.4
    115              17825792.0
    116               3040870.4
    117      Varies with device
    118      Varies with device
    119               2726297.6
    120               4404019.2
    121               7444889.6
    122              59768832.0
    123               3879731.2
    124              23068672.0
    125              25165824.0
    126               7759462.4
    127              22020096.0
    128               3565158.4
    129               3040870.4
    130               3250585.6
    131               6710886.4
    132               3355443.2
    133               8598323.2
    134              10380902.4
    135               3040870.4
    136              24117248.0
    137               4823449.6
    138               3250585.6
    139      Varies with device
    140               5138022.4
    141               9961472.0
    142      Varies with device
    143      Varies with device
    144      Varies with device
    145      Varies with device
    146      Varies with device
    147               4404019.2
    148               5662310.4
    149      Varies with device
    150               2936012.8
    151               5242880.0
    152      Varies with device
    153               6186598.4
    154              13631488.0
    155               7444889.6
    156               7025459.2
    157      Varies with device
    158              17825792.0
    159              19922944.0
    160               7025459.2
    161              22020096.0
    162      Varies with device
    163               2831155.2
    164              38797312.0
    165              15728640.0
    166              24117248.0
    167              19922944.0
    168              24117248.0
    169              76546048.0
    170               5138022.4
    171               7130316.8
    172      Varies with device
    173      Varies with device
    174               3040870.4
    175               3670016.0
    176               4194304.0
    177              22020096.0
    178               2411724.8
    179      Varies with device
    180               7549747.2
    181              10485760.0
    182               6396313.6
    183               2202009.6
    184              44040192.0
    185               7654604.8
    186              31457280.0
    187      Varies with device
    188      Varies with device
    189      Varies with device
    190              30408704.0
    191      Varies with device
    192      Varies with device
    193      Varies with device
    194              36700160.0
    195               9542041.6
    196              26214400.0
    197               4089446.4
    198              18874368.0
    199              12582912.0
    200              22020096.0
    201      Varies with device
    202      Varies with device
    203      Varies with device
    204      Varies with device
    205              57671680.0
    206               3355443.2
    207      Varies with device
    208      Varies with device
    209                 23552.0
    210              16777216.0
    211      Varies with device
    212              14680064.0
    213              38797312.0
    214      Varies with device
    215      Varies with device
    216              11534336.0
    217              26214400.0
    218               7654604.8
    219               6815744.0
    220              26214400.0
    221               3250585.6
    222      Varies with device
    223               1572864.0
    224               7864320.0
    225               9017753.6
    226      Varies with device
    227               1258291.2
    228      Varies with device
    229      Varies with device
    230      Varies with device
    231              40894464.0
    232              14680064.0
    233              19922944.0
    234               7130316.8
    235              40894464.0
    236      Varies with device
    237              14680064.0
    238              20971520.0
    239      Varies with device
    240      Varies with device
    241              27262976.0
    242              53477376.0
    243              42991616.0
    244              20971520.0
    245      Varies with device
    246              12582912.0
    247              50331648.0
    248              10485760.0
    249              23068672.0
    250              11534336.0
    251               8912896.0
    252               9017753.6
    253      Varies with device
    254              29360128.0
    255              29360128.0
    256              38797312.0
    257               9437184.0
    258              48234496.0
    259      Varies with device
    260              14680064.0
    261      Varies with device
    262              27262976.0
    263              24117248.0
    264      Varies with device
    265      Varies with device
    266              14680064.0
    267              20971520.0
    268      Varies with device
    269      Varies with device
    270              27262976.0
    271              53477376.0
    272              42991616.0
    273              20971520.0
    274      Varies with device
    275              12582912.0
    276              50331648.0
    277              10485760.0
    278              23068672.0
    279              11534336.0
    280               8912896.0
    281               9017753.6
    282      Varies with device
    283              29360128.0
    284      Varies with device
    285      Varies with device
    286      Varies with device
    287              40894464.0
    288              14680064.0
    289              19922944.0
    290               7130316.8
    291              40894464.0
    292      Varies with device
    293              36700160.0
    294      Varies with device
    295              30408704.0
    296              42991616.0
    297      Varies with device
    298               5138022.4
    299              28311552.0
    300              33554432.0
    301              12582912.0
    302              15728640.0
    303              11534336.0
    304              29360128.0
    305               2306867.2
    306               3565158.4
    307               8703180.8
    308               4508876.8
    309               4928307.2
    310              10485760.0
    311              15728640.0
    312               7444889.6
    313              22020096.0
    314               6396313.6
    315      Varies with device
    316              15728640.0
    317              11534336.0
    318              30408704.0
    319               3460300.8
    320      Varies with device
    321              23068672.0
    322              41943040.0
    323              10485760.0
    324               9542041.6
    325               4928307.2
    326              39845888.0
    327      Varies with device
    328               7025459.2
    329              38797312.0
    330      Varies with device
    331               8178892.8
    332              19922944.0
    333               5976883.2
    334              36700160.0
    335      Varies with device
    336      Varies with device
    337              17825792.0
    338      Varies with device
    339      Varies with device
    340      Varies with device
    341      Varies with device
    342      Varies with device
    343               9227468.8
    344      Varies with device
    345              16777216.0
    346              11534336.0
    347              11534336.0
    348      Varies with device
    349              15728640.0
    350               6920601.6
    351      Varies with device
    352               6920601.6
    353      Varies with device
    354      Varies with device
    355               5347737.6
    356      Varies with device
    357              18874368.0
    358               4194304.0
    359      Varies with device
    360      Varies with device
    361      Varies with device
    362              38797312.0
    363      Varies with device
    364              23068672.0
    365      Varies with device
    366               3460300.8
    367              33554432.0
    368              38797312.0
    369      Varies with device
    370      Varies with device
    371      Varies with device
    372      Varies with device
    373      Varies with device
    374              17825792.0
    375      Varies with device
    376      Varies with device
    377              36700160.0
    378              41943040.0
    379              63963136.0
    380              69206016.0
    381      Varies with device
    382      Varies with device
    383              11534336.0
    384                 80896.0
    385      Varies with device
    386      Varies with device
    387              26214400.0
    388              14680064.0
    389      Varies with device
    390      Varies with device
    391      Varies with device
    392      Varies with device
    393      Varies with device
    394      Varies with device
    395              25165824.0
    396      Varies with device
    397              17825792.0
    398               8703180.8
    399      Varies with device
    400               8598323.2
    401      Varies with device
    402      Varies with device
    403      Varies with device
    404      Varies with device
    405               8808038.4
    406      Varies with device
    407      Varies with device
    408               4194304.0
    409              33554432.0
    410      Varies with device
    411      Varies with device
    412      Varies with device
    413               6396313.6
    414      Varies with device
    415      Varies with device
    416               2936012.8
    417      Varies with device
    418      Varies with device
    419               3460300.8
    420              41943040.0
    421               2306867.2
    422      Varies with device
    423               4508876.8
    424               4928307.2
    425               2411724.8
    426               2411724.8
    427      Varies with device
    428              11534336.0
    429               2831155.2
    430              14680064.0
    431      Varies with device
    432              13631488.0
    433               3879731.2
    434              10485760.0
    435              13631488.0
    436              10485760.0
    437              13631488.0
    438               9227468.8
    439               5767168.0
    440              20971520.0
    441              30408704.0
    442      Varies with device
    443      Varies with device
    444              11534336.0
    445      Varies with device
    446              17825792.0
    447              17825792.0
    448      Varies with device
    449      Varies with device
    450                120832.0
    451      Varies with device
    452              16777216.0
    453               5347737.6
    454      Varies with device
    455              46137344.0
    456               7654604.8
    457              31457280.0
    458                711680.0
    459      Varies with device
    460               1677721.6
    461              19922944.0
    462      Varies with device
    463      Varies with device
    464      Varies with device
    465              11534336.0
    466      Varies with device
    467      Varies with device
    468      Varies with device
    469      Varies with device
    470              24117248.0
    471              27262976.0
    472      Varies with device
    473      Varies with device
    474      Varies with device
    475      Varies with device
    476      Varies with device
    477               6501171.2
    478              20971520.0
    479                 18432.0
    480               1258291.2
    481              17825792.0
    482      Varies with device
    483              15728640.0
    484      Varies with device
    485              58720256.0
    486              12582912.0
    487              11534336.0
    488              30408704.0
    489      Varies with device
    490              41943040.0
    491      Varies with device
    492              55574528.0
    493               3250585.6
    494              25165824.0
    495              24117248.0
    496               5242880.0
    497              46137344.0
    498              28311552.0
    499               6396313.6
    500      Varies with device
    501              22020096.0
    502              11534336.0
    503              22020096.0
    504              25165824.0
    505              32505856.0
    506              28311552.0
    507               6501171.2
    508      Varies with device
    509              12582912.0
    510               8388608.0
    511               6186598.4
    512               8283750.4
    513               5242880.0
    514               1468006.4
    515              13631488.0
    516              41943040.0
    517              13631488.0
    518              19922944.0
    519               5242880.0
    520              19922944.0
    521              28311552.0
    522               3145728.0
    523              13631488.0
    524               7549747.2
    525              26214400.0
    526      Varies with device
    527               5976883.2
    528               5767168.0
    529      Varies with device
    530               6815744.0
    531               6081740.8
    532      Varies with device
    533               3984588.8
    534              42991616.0
    535               8283750.4
    536               2936012.8
    537              20971520.0
    538              15728640.0
    539              29360128.0
    540              10066329.6
    541              13631488.0
    542              15728640.0
    543              41943040.0
    544              58720256.0
    545              12582912.0
    546               9856614.4
    547              30408704.0
    548              11534336.0
    549              19922944.0
    550              24117248.0
    551      Varies with device
    552              28311552.0
    553      Varies with device
    554              19922944.0
    555              47185920.0
    556              22020096.0
    557              41943040.0
    558              26214400.0
    559              15728640.0
    560              25165824.0
    561      Varies with device
    562              11534336.0
    563              39845888.0
    564              32505856.0
    565               6396313.6
    566              28311552.0
    567              33554432.0
    568              13631488.0
    569              66060288.0
    570              46137344.0
    571      Varies with device
    572              16777216.0
    573              29360128.0
    574      Varies with device
    575               6186598.4
    576              13631488.0
    577               9437184.0
    578              20971520.0
    579               8283750.4
    580              29360128.0
    581              51380224.0
    582              28311552.0
    583              18874368.0
    584              14680064.0
    585              28311552.0
    586              42991616.0
    587              39845888.0
    588               5767168.0
    589              28311552.0
    590               7549747.2
    591              22020096.0
    592               9227468.8
    593              39845888.0
    594      Varies with device
    595              11534336.0
    596               6815744.0
    597              27262976.0
    598              19922944.0
    599               6396313.6
    600              80740352.0
    601               9961472.0
    602              14680064.0
    603              39845888.0
    604              16777216.0
    605               3565158.4
    606               4928307.2
    607               5242880.0
    608               5138022.4
    609               5138022.4
    610              16777216.0
    611               4089446.4
    612               4613734.4
    613               5033164.8
    614               4718592.0
    615              73400320.0
    616               5138022.4
    617               3879731.2
    618               5138022.4
    619              16777216.0
    620               4823449.6
    621               3774873.6
    622              22020096.0
    623              13631488.0
    624               9122611.2
    625               9646899.2
    626               7235174.4
    627              40894464.0
    628               8388608.0
    629               4089446.4
    630              14680064.0
    631              11534336.0
    632              13631488.0
    633               9751756.8
    634               5242880.0
    635              10485760.0
    636               5242880.0
    637               5242880.0
    638               3774873.6
    639              19922944.0
    640               5242880.0
    641              10276044.8
    642              14680064.0
    643               5242880.0
    644               5138022.4
    645               5767168.0
    646              11534336.0
    647               3984588.8
    648               5242880.0
    649               8598323.2
    650              22020096.0
    651              12582912.0
    652              10485760.0
    653               3460300.8
    654              26214400.0
    655              15728640.0
    656              10485760.0
    657               5138022.4
    658               4613734.4
    659               3670016.0
    660               8178892.8
    661               9437184.0
    662              15728640.0
    663              19922944.0
    664              58720256.0
    665              39845888.0
    666              16777216.0
    667               5242880.0
    668              28311552.0
    669               4928307.2
    670      Varies with device
    671               3565158.4
    672               5138022.4
    673              26214400.0
    674              30408704.0
    675               8493465.6
    676              14680064.0
    677              28311552.0
    678               4089446.4
    679              13631488.0
    680               5242880.0
    681               5767168.0
    682              16777216.0
    683               5033164.8
    684              14680064.0
    685               4928307.2
    686               3879731.2
    687               5138022.4
    688               3670016.0
    689               4718592.0
    690      Varies with device
    691              37748736.0
    692               4718592.0
    693              19922944.0
    694              14680064.0
    695              80740352.0
    696              22020096.0
    697               8283750.4
    698               9751756.8
    699      Varies with device
    700              18874368.0
    701              18874368.0
    702              22020096.0
    703               3460300.8
    704              25165824.0
    705              40894464.0
    706               3355443.2
    707               5347737.6
    708              11534336.0
    709              28311552.0
    710              38797312.0
    711      Varies with device
    712              27262976.0
    713              11534336.0
    714      Varies with device
    715              42991616.0
    716              51380224.0
    717              22020096.0
    718               8493465.6
    719              53477376.0
    720              14680064.0
    721              18874368.0
    722               3145728.0
    723              19922944.0
    724      Varies with device
    725              23068672.0
    726               7235174.4
    727               7759462.4
    728              88080384.0
    729              26214400.0
    730              18874368.0
    731      Varies with device
    732               2621440.0
    733               4089446.4
    734      Varies with device
    735              22020096.0
    736              28311552.0
    737      Varies with device
    738              22020096.0
    739              18874368.0
    740      Varies with device
    741              18874368.0
    742      Varies with device
    743              10485760.0
    744      Varies with device
    745      Varies with device
    746      Varies with device
    747      Varies with device
    748             101711872.0
    749              17825792.0
    750      Varies with device
    751      Varies with device
    752               2097152.0
    753               1992294.4
    754               1887436.8
    755               1887436.8
    756              17825792.0
    757              18874368.0
    758               5557452.8
    759      Varies with device
    760               5662310.4
    761               5662310.4
    762               5662310.4
    763              15728640.0
    764               5557452.8
    765              50331648.0
    766               5662310.4
    767              49283072.0
    768                569344.0
    769              30408704.0
    770               2411724.8
    771      Varies with device
    772               4613734.4
    773               6920601.6
    774      Varies with device
    775                538624.0
    776              12582912.0
    777      Varies with device
    778              30408704.0
    779      Varies with device
    780      Varies with device
    781      Varies with device
    782              22020096.0
    783              18874368.0
    784      Varies with device
    785      Varies with device
    786      Varies with device
    787              18874368.0
    788              17825792.0
    789              10485760.0
    790              79691776.0
    791      Varies with device
    792              17825792.0
    793      Varies with device
    794      Varies with device
    795              18874368.0
    796              22020096.0
    797      Varies with device
    798               7235174.4
    799      Varies with device
    800              14680064.0
    801              79691776.0
    802              11534336.0
    803              19922944.0
    804               6815744.0
    805               7340032.0
    806      Varies with device
    807               3460300.8
    808              22020096.0
    809      Varies with device
    810               2726297.6
    811              22020096.0
    812               5452595.2
    813              18874368.0
    814      Varies with device
    815              16777216.0
    816              15728640.0
    817               1258291.2
    818      Varies with device
    819      Varies with device
    820              18874368.0
    821               1887436.8
    822               7969177.6
    823      Varies with device
    824      Varies with device
    825              61865984.0
    826      Varies with device
    827               7235174.4
    828              14680064.0
    829              42991616.0
    830              19922944.0
    831              79691776.0
    832      Varies with device
    833              22020096.0
    834              22020096.0
    835              22020096.0
    836      Varies with device
    837              22020096.0
    838      Varies with device
    839      Varies with device
    840               5662310.4
    841               7654604.8
    842      Varies with device
    843              61865984.0
    844              42991616.0
    845      Varies with device
    846              22020096.0
    847              79691776.0
    848               7969177.6
    849      Varies with device
    850              13631488.0
    851              66060288.0
    852              46137344.0
    853              25165824.0
    854              25165824.0
    855      Varies with device
    856               4718592.0
    857      Varies with device
    858              11534336.0
    859      Varies with device
    860               4823449.6
    861      Varies with device
    862      Varies with device
    863               3460300.8
    864               6815744.0
    865      Varies with device
    866      Varies with device
    867      Varies with device
    868              12582912.0
    869      Varies with device
    870               5872025.6
    871              10171187.2
    872              15728640.0
    873      Varies with device
    874              54525952.0
    875      Varies with device
    876      Varies with device
    877               4718592.0
    878      Varies with device
    879              51380224.0
    880              18874368.0
    881              13631488.0
    882               4194304.0
    883              16777216.0
    884              17825792.0
    885      Varies with device
    886              25165824.0
    887      Varies with device
    888              12582912.0
    889      Varies with device
    890              81788928.0
    891      Varies with device
    892              26214400.0
    893              59768832.0
    894      Varies with device
    895               9542041.6
    896              40894464.0
    897      Varies with device
    898              16777216.0
    899      Varies with device
    900               8912896.0
    901              12582912.0
    902              75497472.0
    903              12582912.0
    904      Varies with device
    905      Varies with device
    906              16777216.0
    907              10066329.6
    908      Varies with device
    909              23068672.0
    910      Varies with device
    911              13631488.0
    912      Varies with device
    913              12582912.0
    914      Varies with device
    915              24117248.0
    916              20971520.0
    917              26214400.0
    918              17825792.0
    919      Varies with device
    920              19922944.0
    921              22020096.0
    922              19922944.0
    923      Varies with device
    924      Varies with device
    925      Varies with device
    926      Varies with device
    927              19922944.0
    928               4613734.4
    929              19922944.0
    930              12582912.0
    931      Varies with device
    932              11534336.0
    933      Varies with device
    934      Varies with device
    935      Varies with device
    936              12582912.0
    937              26214400.0
    938      Varies with device
    939      Varies with device
    940              22020096.0
    941              13631488.0
    942              17825792.0
    943              20971520.0
    944      Varies with device
    945              20971520.0
    946              19922944.0
    947              17825792.0
    948              22020096.0
    949              15728640.0
    950              19922944.0
    951              19922944.0
    952               7549747.2
    953              33554432.0
    954      Varies with device
    955      Varies with device
    956              46137344.0
    957              19922944.0
    958      Varies with device
    959              11534336.0
    960      Varies with device
    961      Varies with device
    962      Varies with device
    963              26214400.0
    964      Varies with device
    965      Varies with device
    966      Varies with device
    967      Varies with device
    968              16777216.0
    969      Varies with device
    970              20971520.0
    971      Varies with device
    972              22020096.0
    973              31457280.0
    974      Varies with device
    975      Varies with device
    976              59768832.0
    977              19922944.0
    978              17825792.0
    979      Varies with device
    980              15728640.0
    981              19922944.0
    982              19922944.0
    983              19922944.0
    984              46137344.0
    985              19922944.0
    986              36700160.0
    987              46137344.0
    988      Varies with device
    989      Varies with device
    990              20971520.0
    991              19922944.0
    992              26214400.0
    993      Varies with device
    994              45088768.0
    995      Varies with device
    996      Varies with device
    997               3774873.6
    998               8074035.2
    999              46137344.0
    1000             12582912.0
    1001             55574528.0
    1002             80740352.0
    1003     Varies with device
    1004              9961472.0
    1005             37748736.0
    1006              6606028.8
    1007              6186598.4
    1008     Varies with device
    1009             12582912.0
    1010              2936012.8
    1011             27262976.0
    1012              9122611.2
    1013             16777216.0
    1014             55574528.0
    1015     Varies with device
    1016             14680064.0
    1017              9332326.4
    1018             11534336.0
    1019               342016.0
    1020             25165824.0
    1021             13631488.0
    1022     Varies with device
    1023              2936012.8
    1024             10171187.2
    1025              1468006.4
    1026     Varies with device
    1027              3670016.0
    1028             63963136.0
    1029              6291456.0
    1030              2411724.8
    1031              9437184.0
    1032             11534336.0
    1033             28311552.0
    1034              3774873.6
    1035     Varies with device
    1036             30408704.0
    1037              4613734.4
    1038             22020096.0
    1039             13631488.0
    1040             28311552.0
    1041             11534336.0
    1042             25165824.0
    1043              6396313.6
    1044              6606028.8
    1045             24117248.0
    1046             35651584.0
    1047              3984588.8
    1048     Varies with device
    1049     Varies with device
    1050             44040192.0
    1051     Varies with device
    1052             19922944.0
    1053             73400320.0
    1054             33554432.0
    1055             97517568.0
    1056     Varies with device
    1057     Varies with device
    1058     Varies with device
    1059             41943040.0
    1060             25165824.0
    1061     Varies with device
    1062             20971520.0
    1063             15728640.0
    1064             29360128.0
    1065             10485760.0
    1066     Varies with device
    1067     Varies with device
    1068     Varies with device
    1069             14680064.0
    1070     Varies with device
    1071             44040192.0
    1072             68157440.0
    1073             38797312.0
    1074     Varies with device
    1075             40894464.0
    1076             49283072.0
    1077             82837504.0
    1078             18874368.0
    1079     Varies with device
    1080            104857600.0
    1081             33554432.0
    1082             48234496.0
    1083     Varies with device
    1084             10276044.8
    1085              8598323.2
    1086             18874368.0
    1087     Varies with device
    1088             40894464.0
    1089             46137344.0
    1090     Varies with device
    1091              5242880.0
    1092             22020096.0
    1093             23068672.0
    1094             24117248.0
    1095     Varies with device
    1096     Varies with device
    1097     Varies with device
    1098             28311552.0
    1099     Varies with device
    1100              5242880.0
    1101             33554432.0
    1102              7759462.4
    1103             25165824.0
    1104             23068672.0
    1105     Varies with device
    1106             10485760.0
    1107             17825792.0
    1108             39845888.0
    1109             48234496.0
    1110              3984588.8
    1111              9751756.8
    1112              8808038.4
    1113             29360128.0
    1114     Varies with device
    1115             10171187.2
    1116             11534336.0
    1117     Varies with device
    1118              7759462.4
    1119             19922944.0
    1120             16777216.0
    1121             14680064.0
    1122             23068672.0
    1123             26214400.0
    1124              3774873.6
    1125     Varies with device
    1126     Varies with device
    1127              4718592.0
    1128     Varies with device
    1129             24117248.0
    1130     Varies with device
    1131     Varies with device
    1132     Varies with device
    1133              4404019.2
    1134     Varies with device
    1135             60817408.0
    1136     Varies with device
    1137             23068672.0
    1138     Varies with device
    1139              1468006.4
    1140     Varies with device
    1141              9542041.6
    1142             23068672.0
    1143     Varies with device
    1144     Varies with device
    1145     Varies with device
    1146     Varies with device
    1147             22020096.0
    1148     Varies with device
    1149     Varies with device
    1150             11534336.0
    1151             39845888.0
    1152             12582912.0
    1153             47185920.0
    1154             52428800.0
    1155             25165824.0
    1156             49283072.0
    1157             24117248.0
    1158             48234496.0
    1159             47185920.0
    1160             52428800.0
    1161             25165824.0
    1162             34603008.0
    1163     Varies with device
    1164             12582912.0
    1165             55574528.0
    1166             14680064.0
    1167             71303168.0
    1168             38797312.0
    1169             82837504.0
    1170             33554432.0
    1171             48234496.0
    1172             41943040.0
    1173             33554432.0
    1174     Varies with device
    1175             25165824.0
    1176             44040192.0
    1177             12582912.0
    1178             13631488.0
    1179             13631488.0
    1180              5138022.4
    1181              9332326.4
    1182     Varies with device
    1183             19922944.0
    1184     Varies with device
    1185             14680064.0
    1186             67108864.0
    1187              8808038.4
    1188             11534336.0
    1189             69206016.0
    1190     Varies with device
    1191             42991616.0
    1192             15728640.0
    1193             36700160.0
    1194             37748736.0
    1195             17825792.0
    1196             26214400.0
    1197             36700160.0
    1198             10485760.0
    1199     Varies with device
    1200             12582912.0
    1201     Varies with device
    1202             11534336.0
    1203     Varies with device
    1204     Varies with device
    1205             17825792.0
    1206             45088768.0
    1207             42991616.0
    1208     Varies with device
    1209             45088768.0
    1210             16777216.0
    1211             40894464.0
    1212             13631488.0
    1213             31457280.0
    1214              7444889.6
    1215             24117248.0
    1216             17825792.0
    1217             18874368.0
    1218     Varies with device
    1219             17825792.0
    1220     Varies with device
    1221             15728640.0
    1222             23068672.0
    1223     Varies with device
    1224              9437184.0
    1225              7549747.2
    1226     Varies with device
    1227     Varies with device
    1228              2411724.8
    1229             28311552.0
    1230     Varies with device
    1231              8598323.2
    1232             79691776.0
    1233             23068672.0
    1234     Varies with device
    1235             19922944.0
    1236     Varies with device
    1237     Varies with device
    1238             29360128.0
    1239             36700160.0
    1240     Varies with device
    1241             23068672.0
    1242     Varies with device
    1243     Varies with device
    1244     Varies with device
    1245             26214400.0
    1246             31457280.0
    1247              8912896.0
    1248     Varies with device
    1249             17825792.0
    1250             35651584.0
    1251             35651584.0
    1252     Varies with device
    1253             17825792.0
    1254             16777216.0
    1255     Varies with device
    1256             15728640.0
    1257              2306867.2
    1258             11534336.0
    1259              7235174.4
    1260             13631488.0
    1261             11534336.0
    1262              3040870.4
    1263             26214400.0
    1264              7340032.0
    1265             28311552.0
    1266              7864320.0
    1267     Varies with device
    1268             15728640.0
    1269              9856614.4
    1270              6710886.4
    1271              5767168.0
    1272              4089446.4
    1273             57671680.0
    1274             29360128.0
    1275     Varies with device
    1276              9227468.8
    1277     Varies with device
    1278              4194304.0
    1279              7235174.4
    1280             11534336.0
    1281     Varies with device
    1282              5767168.0
    1283     Varies with device
    1284             15728640.0
    1285              4508876.8
    1286     Varies with device
    1287              7444889.6
    1288             60817408.0
    1289     Varies with device
    1290     Varies with device
    1291             18874368.0
    1292     Varies with device
    1293              1572864.0
    1294              6815744.0
    1295             11534336.0
    1296             70254592.0
    1297     Varies with device
    1298             59768832.0
    1299             61865984.0
    1300             50331648.0
    1301             59768832.0
    1302             13631488.0
    1303     Varies with device
    1304     Varies with device
    1305     Varies with device
    1306             32505856.0
    1307             10485760.0
    1308             59768832.0
    1309     Varies with device
    1310             22020096.0
    1311             26214400.0
    1312             97517568.0
    1313             62914560.0
    1314             28311552.0
    1315             56623104.0
    1316     Varies with device
    1317     Varies with device
    1318             61865984.0
    1319     Varies with device
    1320             81788928.0
    1321             57671680.0
    1322     Varies with device
    1323     Varies with device
    1324             70254592.0
    1325     Varies with device
    1326             36700160.0
    1327     Varies with device
    1328             45088768.0
    1329             40894464.0
    1330              4404019.2
    1331             46137344.0
    1332             47185920.0
    1333     Varies with device
    1334             29360128.0
    1335             30408704.0
    1336             12582912.0
    1337             15728640.0
    1338             32505856.0
    1339     Varies with device
    1340     Varies with device
    1341     Varies with device
    1342     Varies with device
    1343             20971520.0
    1344             40894464.0
    1345             98566144.0
    1346             36700160.0
    1347     Varies with device
    1348     Varies with device
    1349             42991616.0
    1350             51380224.0
    1351             23068672.0
    1352     Varies with device
    1353             24117248.0
    1354             29360128.0
    1355              7549747.2
    1356              6815744.0
    1357     Varies with device
    1358             25165824.0
    1359     Varies with device
    1360     Varies with device
    1361             20971520.0
    1362     Varies with device
    1363             15728640.0
    1364     Varies with device
    1365             57671680.0
    1366     Varies with device
    1367             39845888.0
    1368             33554432.0
    1369     Varies with device
    1370     Varies with device
    1371             41943040.0
    1372             61865984.0
    1373     Varies with device
    1374     Varies with device
    1375     Varies with device
    1376             24117248.0
    1377              6396313.6
    1378     Varies with device
    1379             60817408.0
    1380             57671680.0
    1381     Varies with device
    1382     Varies with device
    1383             59768832.0
    1384             97517568.0
    1385             62914560.0
    1386     Varies with device
    1387     Varies with device
    1388     Varies with device
    1389     Varies with device
    1390     Varies with device
    1391     Varies with device
    1392             19922944.0
    1393              3984588.8
    1394             18874368.0
    1395             59768832.0
    1396              5767168.0
    1397     Varies with device
    1398              5976883.2
    1399             53477376.0
    1400              9856614.4
    1401              3460300.8
    1402     Varies with device
    1403              4194304.0
    1404     Varies with device
    1405     Varies with device
    1406             19922944.0
    1407              3984588.8
    1408              4508876.8
    1409             18874368.0
    1410              5976883.2
    1411     Varies with device
    1412              3774873.6
    1413              5767168.0
    1414             59768832.0
    1415             53477376.0
    1416             60817408.0
    1417     Varies with device
    1418              9856614.4
    1419              3460300.8
    1420              4194304.0
    1421     Varies with device
    1422     Varies with device
    1423             10380902.4
    1424             45088768.0
    1425             38797312.0
    1426     Varies with device
    1427     Varies with device
    1428             41943040.0
    1429             76546048.0
    1430             29360128.0
    1431     Varies with device
    1432             24117248.0
    1433             10485760.0
    1434             15728640.0
    1435              8808038.4
    1436             32505856.0
    1437     Varies with device
    1438     Varies with device
    1439     Varies with device
    1440             42991616.0
    1441     Varies with device
    1442     Varies with device
    1443             20971520.0
    1444             40894464.0
    1445             36700160.0
    1446             35651584.0
    1447              7864320.0
    1448     Varies with device
    1449             12582912.0
    1450              8074035.2
    1451             22020096.0
    1452             18874368.0
    1453             41943040.0
    1454     Varies with device
    1455              9122611.2
    1456     Varies with device
    1457              9017753.6
    1458     Varies with device
    1459             18874368.0
    1460              7864320.0
    1461              9542041.6
    1462             13631488.0
    1463     Varies with device
    1464             15728640.0
    1465     Varies with device
    1466             19922944.0
    1467     Varies with device
    1468     Varies with device
    1469              5347737.6
    1470             27262976.0
    1471              8703180.8
    1472             28311552.0
    1473              8283750.4
    1474     Varies with device
    1475             28311552.0
    1476              5347737.6
    1477             15728640.0
    1478              3250585.6
    1479             16777216.0
    1480              9646899.2
    1481     Varies with device
    1482     Varies with device
    1483             10485760.0
    1484              2726297.6
    1485             13631488.0
    1486              1992294.4
    1487              7340032.0
    1488             15728640.0
    1489              4404019.2
    1490              8178892.8
    1491             80740352.0
    1492              5976883.2
    1493              6186598.4
    1494              7969177.6
    1495              6081740.8
    1496     Varies with device
    1497              6081740.8
    1498              5557452.8
    1499              5767168.0
    1500             26214400.0
    1501             36700160.0
    1502              8283750.4
    1503             19922944.0
    1504              7864320.0
    1505             12582912.0
    1506     Varies with device
    1507             35651584.0
    1508             31457280.0
    1509     Varies with device
    1510     Varies with device
    1511             16777216.0
    1512     Varies with device
    1513             16777216.0
    1514     Varies with device
    1515             15728640.0
    1516             22020096.0
    1517              1887436.8
    1518              4508876.8
    1519             10485760.0
    1520              3879731.2
    1521               237568.0
    1522            103809024.0
    1523               638976.0
    1524              7235174.4
    1525              7340032.0
    1526              7340032.0
    1527              6710886.4
    1528     Varies with device
    1529             16777216.0
    1530              6186598.4
    1531             99614720.0
    1532                 8704.0
    1533             20971520.0
    1534              2621440.0
    1535             52428800.0
    1536             10485760.0
    1537             18874368.0
    1538             12582912.0
    1539              2621440.0
    1540              2202009.6
    1541             23068672.0
    1542                41984.0
    1543             23068672.0
    1544               299008.0
    1545              5767168.0
    1546     Varies with device
    1547              2621440.0
    1548              8493465.6
    1549             29360128.0
    1550             25165824.0
    1551              3250585.6
    1552              3565158.4
    1553                11264.0
    1554             13631488.0
    1555             19922944.0
    1556              9961472.0
    1557             22020096.0
    1558              9856614.4
    1559              2516582.4
    1560             37748736.0
    1561             57671680.0
    1562             33554432.0
    1563             35651584.0
    1564              5347737.6
    1565            104857600.0
    1566             83886080.0
    1567              3460300.8
    1568             29360128.0
    1569              5242880.0
    1570             11534336.0
    1571             57671680.0
    1572              7444889.6
    1573              3355443.2
    1574              4613734.4
    1575             16777216.0
    1576              6606028.8
    1577             47185920.0
    1578             24117248.0
    1579             38797312.0
    1580             13631488.0
    1581              8493465.6
    1582             13631488.0
    1583             14680064.0
    1584             31457280.0
    1585             37748736.0
    1586              3145728.0
    1587              1468006.4
    1588             40894464.0
    1589              1887436.8
    1590              1782579.2
    1591             19922944.0
    1592             15728640.0
    1593             10485760.0
    1594              2831155.2
    1595             16777216.0
    1596             22020096.0
    1597             20971520.0
    1598             14680064.0
    1599             79691776.0
    1600             18874368.0
    1601             12582912.0
    1602              6501171.2
    1603     Varies with device
    1604              7130316.8
    1605     Varies with device
    1606             25165824.0
    1607             19922944.0
    1608             13631488.0
    1609             15728640.0
    1610             13631488.0
    1611     Varies with device
    1612     Varies with device
    1613             46137344.0
    1614     Varies with device
    1615             10171187.2
    1616             25165824.0
    1617             13631488.0
    1618              4718592.0
    1619              4718592.0
    1620             30408704.0
    1621             12582912.0
    1622             12582912.0
    1623     Varies with device
    1624              4823449.6
    1625             10485760.0
    1626              7130316.8
    1627             34603008.0
    1628             12582912.0
    1629     Varies with device
    1630              4823449.6
    1631             10485760.0
    1632              7130316.8
    1633             34603008.0
    1634             13631488.0
    1635             15728640.0
    1636             24117248.0
    1637              8074035.2
    1638     Varies with device
    1639             15728640.0
    1640             26214400.0
    1641     Varies with device
    1642              3145728.0
    1643             23068672.0
    1644     Varies with device
    1645     Varies with device
    1646              7864320.0
    1647             25165824.0
    1648     Varies with device
    1649             46137344.0
    1650     Varies with device
    1651     Varies with device
    1652             14680064.0
    1653             70254592.0
    1654             79691776.0
    1655             77594624.0
    1656             24117248.0
    1657             48234496.0
    1658             25165824.0
    1659     Varies with device
    1660            101711872.0
    1661             65011712.0
    1662             25165824.0
    1663             34603008.0
    1664              8178892.8
    1665             48234496.0
    1666             72351744.0
    1667             78643200.0
    1668             70254592.0
    1669             52428800.0
    1670            102760448.0
    1671     Varies with device
    1672              5138022.4
    1673             38797312.0
    1674             18874368.0
    1675             54525952.0
    1676     Varies with device
    1677             81788928.0
    1678              4089446.4
    1679             61865984.0
    1680            101711872.0
    1681             11534336.0
    1682             39845888.0
    1683     Varies with device
    1684     Varies with device
    1685     Varies with device
    1686     Varies with device
    1687             89128960.0
    1688             81788928.0
    1689             66060288.0
    1690     Varies with device
    1691             25165824.0
    1692             72351744.0
    1693             78643200.0
    1694     Varies with device
    1695             73400320.0
    1696             66060288.0
    1697            103809024.0
    1698             51380224.0
    1699     Varies with device
    1700             79691776.0
    1701             70254592.0
    1702             25165824.0
    1703             54525952.0
    1704            102760448.0
    1705             77594624.0
    1706             72351744.0
    1707             85983232.0
    1708     Varies with device
    1709            100663296.0
    1710            103809024.0
    1711             48234496.0
    1712     Varies with device
    1713            103809024.0
    1714             91226112.0
    1715              4089446.4
    1716             70254592.0
    1717     Varies with device
    1718             55574528.0
    1719              8178892.8
    1720     Varies with device
    1721            101711872.0
    1722     Varies with device
    1723     Varies with device
    1724             61865984.0
    1725     Varies with device
    1726             48234496.0
    1727             78643200.0
    1728             52428800.0
    1729             65011712.0
    1730     Varies with device
    1731             24117248.0
    1732             75497472.0
    1733             73400320.0
    1734             18874368.0
    1735     Varies with device
    1736             36700160.0
    1737             81788928.0
    1738             26214400.0
    1739             77594624.0
    1740            101711872.0
    1741             17825792.0
    1742             55574528.0
    1743     Varies with device
    1744             11534336.0
    1745             62914560.0
    1746             59768832.0
    1747             14680064.0
    1748             70254592.0
    1749             34603008.0
    1750             79691776.0
    1751             77594624.0
    1752     Varies with device
    1753             41943040.0
    1754             61865984.0
    1755             54525952.0
    1756     Varies with device
    1757     Varies with device
    1758            104857600.0
    1759             65011712.0
    1760     Varies with device
    1761     Varies with device
    1762     Varies with device
    1763     Varies with device
    1764             89128960.0
    1765             91226112.0
    1766             34603008.0
    1767             33554432.0
    1768             41943040.0
    1769             30408704.0
    1770             41943040.0
    1771             25165824.0
    1772             39845888.0
    1773             54525952.0
    1774     Varies with device
    1775             56623104.0
    1776             74448896.0
    1777             76546048.0
    1778             59768832.0
    1779             39845888.0
    1780             65011712.0
    1781             99614720.0
    1782             16777216.0
    1783             40894464.0
    1784             81788928.0
    1785             37748736.0
    1786     Varies with device
    1787             32505856.0
    1788     Varies with device
    1789            102760448.0
    1790             25165824.0
    1791             89128960.0
    1792             34603008.0
    1793            104857600.0
    1794             37748736.0
    1795             90177536.0
    1796             83886080.0
    1797             91226112.0
    1798             82837504.0
    1799             31457280.0
    1800             91226112.0
    1801             95420416.0
    1802             66060288.0
    1803     Varies with device
    1804     Varies with device
    1805             99614720.0
    1806             50331648.0
    1807     Varies with device
    1808     Varies with device
    1809             97517568.0
    1810             55574528.0
    1811             85983232.0
    1812            103809024.0
    1813             51380224.0
    1814             26214400.0
    1815     Varies with device
    1816             95420416.0
    1817             54525952.0
    1818             58720256.0
    1819             58720256.0
    1820             89128960.0
    1821             84934656.0
    1822             85983232.0
    1823            102760448.0
    1824             80740352.0
    1825            103809024.0
    1826     Varies with device
    1827            100663296.0
    1828            100663296.0
    1829             85983232.0
    1830             73400320.0
    1831             66060288.0
    1832             36700160.0
    1833             50331648.0
    1834             88080384.0
    1835             15728640.0
    1836             72351744.0
    1837             48234496.0
    1838             15728640.0
    1839             34603008.0
    1840             50331648.0
    1841             70254592.0
    1842             77594624.0
    1843             59768832.0
    1844             54525952.0
    1845             26214400.0
    1846     Varies with device
    1847             37748736.0
    1848             32505856.0
    1849             95420416.0
    1850     Varies with device
    1851             96468992.0
    1852             73400320.0
    1853             97517568.0
    1854             54525952.0
    1855             95420416.0
    1856             38797312.0
    1857             67108864.0
    1858             36700160.0
    1859            101711872.0
    1860             97517568.0
    1861             29360128.0
    1862             87031808.0
    1863            103809024.0
    1864             57671680.0
    1865             85983232.0
    1866             14680064.0
    1867             84934656.0
    1868     Varies with device
    1869             77594624.0
    1870             70254592.0
    1871             54525952.0
    1872             79691776.0
    1873             70254592.0
    1874             52428800.0
    1875             48234496.0
    1876     Varies with device
    1877     Varies with device
    1878            101711872.0
    1879            102760448.0
    1880             73400320.0
    1881             72351744.0
    1882              4089446.4
    1883              8178892.8
    1884             18874368.0
    1885             25165824.0
    1886             65011712.0
    1887             11534336.0
    1888     Varies with device
    1889             91226112.0
    1890            103809024.0
    1891     Varies with device
    1892             85983232.0
    1893             98566144.0
    1894     Varies with device
    1895     Varies with device
    1896     Varies with device
    1897            100663296.0
    1898             78643200.0
    1899     Varies with device
    1900             14680064.0
    1901             61865984.0
    1902             77594624.0
    1903             30408704.0
    1904             17825792.0
    1905            101711872.0
    1906             81788928.0
    1907     Varies with device
    1908     Varies with device
    1909             66060288.0
    1910     Varies with device
    1911             26214400.0
    1912             30408704.0
    1913             75497472.0
    1914     Varies with device
    1915             55574528.0
    1916     Varies with device
    1917             79691776.0
    1918             34603008.0
    1919             62914560.0
    1920             65011712.0
    1921     Varies with device
    1922     Varies with device
    1923             81788928.0
    1924             78643200.0
    1925     Varies with device
    1926             73400320.0
    1927            103809024.0
    1928            103809024.0
    1929     Varies with device
    1930             28311552.0
    1931             73400320.0
    1932             92274688.0
    1933             66060288.0
    1934             85983232.0
    1935             50331648.0
    1936     Varies with device
    1937             32505856.0
    1938             24117248.0
    1939     Varies with device
    1940             61865984.0
    1941             40894464.0
    1942             56623104.0
    1943             50331648.0
    1944     Varies with device
    1945             31457280.0
    1946             69206016.0
    1947             33554432.0
    1948             52428800.0
    1949             55574528.0
    1950             63963136.0
    1951             51380224.0
    1952             39845888.0
    1953             20971520.0
    1954     Varies with device
    1955     Varies with device
    1956             69206016.0
    1957             59768832.0
    1958             99614720.0
    1959     Varies with device
    1960     Varies with device
    1961             33554432.0
    1962             34603008.0
    1963             69206016.0
    1964             78643200.0
    1965             30408704.0
    1966             77594624.0
    1967             72351744.0
    1968              8178892.8
    1969             34603008.0
    1970             54525952.0
    1971             52428800.0
    1972             48234496.0
    1973             12582912.0
    1974             42991616.0
    1975             66060288.0
    1976             55574528.0
    1977             66060288.0
    1978            103809024.0
    1979     Varies with device
    1980             20971520.0
    1981             11534336.0
    1982             78643200.0
    1983             13631488.0
    1984             48234496.0
    1985             81788928.0
    1986             51380224.0
    1987             14680064.0
    1988            104857600.0
    1989     Varies with device
    1990            101711872.0
    1991     Varies with device
    1992     Varies with device
    1993             15728640.0
    1994             34603008.0
    1995     Varies with device
    1996             73400320.0
    1997             10485760.0
    1998             80740352.0
    1999     Varies with device
    2000     Varies with device
    2001              5138022.4
    2002     Varies with device
    2003             26214400.0
    2004            100663296.0
    2005             24117248.0
    2006             53477376.0
    2007            100663296.0
    2008     Varies with device
    2009     Varies with device
    2010     Varies with device
    2011             73400320.0
    2012             39845888.0
    2013             25165824.0
    2014     Varies with device
    2015             20971520.0
    2016             70254592.0
    2017             19922944.0
    2018             53477376.0
    2019             23068672.0
    2020             48234496.0
    2021             22020096.0
    2022             40894464.0
    2023             24117248.0
    2024             27262976.0
    2025             40894464.0
    2026             25165824.0
    2027             15728640.0
    2028             20971520.0
    2029             25165824.0
    2030             46137344.0
    2031             39845888.0
    2032             54525952.0
    2033             14680064.0
    2034             51380224.0
    2035              9332326.4
    2036     Varies with device
    2037             58720256.0
    2038            103809024.0
    2039             10485760.0
    2040              7235174.4
    2041             20971520.0
    2042             10485760.0
    2043             19922944.0
    2044             34603008.0
    2045             89128960.0
    2046             50331648.0
    2047             10066329.6
    2048     Varies with device
    2049             15728640.0
    2050     Varies with device
    2051             52428800.0
    2052             53477376.0
    2053             25165824.0
    2054             13631488.0
    2055             16777216.0
    2056     Varies with device
    2057     Varies with device
    2058             66060288.0
    2059             15728640.0
    2060             27262976.0
    2061             84934656.0
    2062             82837504.0
    2063     Varies with device
    2064             98566144.0
    2065             48234496.0
    2066             45088768.0
    2067             47185920.0
    2068             95420416.0
    2069             95420416.0
    2070             55574528.0
    2071             29360128.0
    2072             27262976.0
    2073             58720256.0
    2074             83886080.0
    2075             62914560.0
    2076     Varies with device
    2077             70254592.0
    2078             81788928.0
    2079             28311552.0
    2080     Varies with device
    2081             66060288.0
    2082             27262976.0
    2083             48234496.0
    2084             87031808.0
    2085             62914560.0
    2086             46137344.0
    2087             85983232.0
    2088             70254592.0
    2089            103809024.0
    2090     Varies with device
    2091             25165824.0
    2092             81788928.0
    2093     Varies with device
    2094     Varies with device
    2095             53477376.0
    2096             98566144.0
    2097             16777216.0
    2098             97517568.0
    2099             80740352.0
    2100             33554432.0
    2101             60817408.0
    2102             50331648.0
    2103             72351744.0
    2104             76546048.0
    2105            103809024.0
    2106             16777216.0
    2107             56623104.0
    2108             58720256.0
    2109             71303168.0
    2110             26214400.0
    2111              7340032.0
    2112             56623104.0
    2113             53477376.0
    2114     Varies with device
    2115             50331648.0
    2116             46137344.0
    2117              9122611.2
    2118             95420416.0
    2119            101711872.0
    2120             27262976.0
    2121             16777216.0
    2122     Varies with device
    2123             55574528.0
    2124             99614720.0
    2125             87031808.0
    2126             81788928.0
    2127             58720256.0
    2128             22020096.0
    2129             27262976.0
    2130             62914560.0
    2131             23068672.0
    2132             99614720.0
    2133             50331648.0
    2134             60817408.0
    2135             95420416.0
    2136             23068672.0
    2137             37748736.0
    2138             38797312.0
    2139             73400320.0
    2140             35651584.0
    2141             51380224.0
    2142             10276044.8
    2143             38797312.0
    2144     Varies with device
    2145             57671680.0
    2146             88080384.0
    2147             16777216.0
    2148             96468992.0
    2149             59768832.0
    2150     Varies with device
    2151             25165824.0
    2152              7235174.4
    2153             46137344.0
    2154             11534336.0
    2155             16777216.0
    2156             17825792.0
    2157             18874368.0
    2158             16777216.0
    2159             19922944.0
    2160              1677721.6
    2161             49283072.0
    2162             16777216.0
    2163             14680064.0
    2164     Varies with device
    2165             78643200.0
    2166             26214400.0
    2167              5976883.2
    2168     Varies with device
    2169             17825792.0
    2170              3670016.0
    2171             16777216.0
    2172             44040192.0
    2173             49283072.0
    2174             70254592.0
    2175             27262976.0
    2176             75497472.0
    2177             38797312.0
    2178              1258291.2
    2179             17825792.0
    2180             19922944.0
    2181             24117248.0
    2182             26214400.0
    2183             87031808.0
    2184             98566144.0
    2185            101711872.0
    2186             38797312.0
    2187             96468992.0
    2188             50331648.0
    2189             12582912.0
    2190             15728640.0
    2191             87031808.0
    2192              9961472.0
    2193             19922944.0
    2194             24117248.0
    2195             26214400.0
    2196             87031808.0
    2197             98566144.0
    2198            101711872.0
    2199             38797312.0
    2200             96468992.0
    2201             50331648.0
    2202             12582912.0
    2203             15728640.0
    2204             87031808.0
    2205              9961472.0
    2206             70254592.0
    2207     Varies with device
    2208     Varies with device
    2209     Varies with device
    2210             66060288.0
    2211             51380224.0
    2212             14680064.0
    2213             60817408.0
    2214             23068672.0
    2215             15728640.0
    2216     Varies with device
    2217             52428800.0
    2218             61865984.0
    2219             35651584.0
    2220     Varies with device
    2221             56623104.0
    2222             55574528.0
    2223     Varies with device
    2224     Varies with device
    2225             38797312.0
    2226             26214400.0
    2227             27262976.0
    2228             63963136.0
    2229             60817408.0
    2230             95420416.0
    2231             17825792.0
    2232             87031808.0
    2233             17825792.0
    2234     Varies with device
    2235             67108864.0
    2236             42991616.0
    2237             60817408.0
    2238             24117248.0
    2239             74448896.0
    2240     Varies with device
    2241     Varies with device
    2242             24117248.0
    2243             12582912.0
    2244               720896.0
    2245              3040870.4
    2246             26214400.0
    2247     Varies with device
    2248             20971520.0
    2249             22020096.0
    2250              5662310.4
    2251             26214400.0
    2252             44040192.0
    2253             33554432.0
    2254              1887436.8
    2255             34603008.0
    2256              3984588.8
    2257              6081740.8
    2258              9961472.0
    2259             19922944.0
    2260     Varies with device
    2261              5242880.0
    2262             24117248.0
    2263               882688.0
    2264             10380902.4
    2265             25165824.0
    2266              2516582.4
    2267               920576.0
    2268             39845888.0
    2269              6815744.0
    2270             14680064.0
    2271              1258291.2
    2272             50331648.0
    2273               387072.0
    2274             23068672.0
    2275              5033164.8
    2276     Varies with device
    2277             26214400.0
    2278               272384.0
    2279               384000.0
    2280             65011712.0
    2281              1258291.2
    2282     Varies with device
    2283              2936012.8
    2284             99614720.0
    2285              2726297.6
    2286              4823449.6
    2287     Varies with device
    2288              5347737.6
    2289              1887436.8
    2290     Varies with device
    2291     Varies with device
    2292             55574528.0
    2293             40894464.0
    2294              5557452.8
    2295             14680064.0
    2296              8808038.4
    2297             18874368.0
    2298             42991616.0
    2299            104857600.0
    2300              3565158.4
    2301             71303168.0
    2302              2726297.6
    2303             11534336.0
    2304     Varies with device
    2305             38797312.0
    2306     Varies with device
    2307             10276044.8
    2308             24117248.0
    2309             18874368.0
    2310              7759462.4
    2311             11534336.0
    2312     Varies with device
    2313             24117248.0
    2314             35651584.0
    2315             15728640.0
    2316             13631488.0
    2317             25165824.0
    2318             20971520.0
    2319             14680064.0
    2320             72351744.0
    2321             39845888.0
    2322             27262976.0
    2323              2726297.6
    2324             22020096.0
    2325             39845888.0
    2326             16777216.0
    2327             22020096.0
    2328             10485760.0
    2329             23068672.0
    2330             14680064.0
    2331             37748736.0
    2332             16777216.0
    2333             63963136.0
    2334             11534336.0
    2335              8912896.0
    2336             38797312.0
    2337     Varies with device
    2338     Varies with device
    2339             23068672.0
    2340             17825792.0
    2341              8074035.2
    2342             40894464.0
    2343             31457280.0
    2344              3879731.2
    2345             29360128.0
    2346             16777216.0
    2347             20971520.0
    2348              9332326.4
    2349            101711872.0
    2350             12582912.0
    2351             75497472.0
    2352     Varies with device
    2353             22020096.0
    2354     Varies with device
    2355             28311552.0
    2356              5976883.2
    2357              6920601.6
    2358             12582912.0
    2359              6186598.4
    2360             20971520.0
    2361             15728640.0
    2362             27262976.0
    2363              3460300.8
    2364             46137344.0
    2365             33554432.0
    2366             12582912.0
    2367             37748736.0
    2368             17825792.0
    2369              6081740.8
    2370             11534336.0
    2371             16777216.0
    2372             26214400.0
    2373              6396313.6
    2374     Varies with device
    2375             23068672.0
    2376             29360128.0
    2377              6606028.8
    2378             26214400.0
    2379             62914560.0
    2380             20971520.0
    2381              8388608.0
    2382             20971520.0
    2383              6710886.4
    2384             35651584.0
    2385              3984588.8
    2386             44040192.0
    2387              2516582.4
    2388             73400320.0
    2389              5242880.0
    2390             29360128.0
    2391             22020096.0
    2392             39845888.0
    2393     Varies with device
    2394             77594624.0
    2395              1887436.8
    2396             12582912.0
    2397             13631488.0
    2398     Varies with device
    2399              3040870.4
    2400             50331648.0
    2401             19922944.0
    2402             71303168.0
    2403              7130316.8
    2404             12582912.0
    2405              6081740.8
    2406             42991616.0
    2407             26214400.0
    2408             44040192.0
    2409              6815744.0
    2410             13631488.0
    2411              1572864.0
    2412              3355443.2
    2413              8074035.2
    2414              1363148.8
    2415             27262976.0
    2416              5662310.4
    2417             11534336.0
    2418              9961472.0
    2419             65011712.0
    2420             18874368.0
    2421     Varies with device
    2422              3250585.6
    2423              1887436.8
    2424             69206016.0
    2425              5242880.0
    2426              3040870.4
    2427             23068672.0
    2428             16777216.0
    2429             19922944.0
    2430             41943040.0
    2431             24117248.0
    2432             15728640.0
    2433             30408704.0
    2434             16777216.0
    2435             68157440.0
    2436             30408704.0
    2437             20971520.0
    2438              3984588.8
    2439             19922944.0
    2440             26214400.0
    2441              5347737.6
    2442             16777216.0
    2443              1572864.0
    2444             10485760.0
    2445             23068672.0
    2446              3774873.6
    2447             20971520.0
    2448             20971520.0
    2449             13631488.0
    2450              2621440.0
    2451              3774873.6
    2452             25165824.0
    2453             10485760.0
    2454             26214400.0
    2455             30408704.0
    2456             17825792.0
    2457             55574528.0
    2458             87031808.0
    2459              4823449.6
    2460              4823449.6
    2461             29360128.0
    2462             30408704.0
    2463             48234496.0
    2464              5976883.2
    2465              1468006.4
    2466             12582912.0
    2467              2306867.2
    2468             28311552.0
    2469             18874368.0
    2470             31457280.0
    2471              3670016.0
    2472              4404019.2
    2473              7235174.4
    2474              6920601.6
    2475              7340032.0
    2476             11534336.0
    2477              5872025.6
    2478             30408704.0
    2479              3460300.8
    2480             20971520.0
    2481              7340032.0
    2482              3460300.8
    2483             30408704.0
    2484              7444889.6
    2485     Varies with device
    2486             24117248.0
    2487             25165824.0
    2488             14680064.0
    2489             13631488.0
    2490             35651584.0
    2491              4508876.8
    2492             45088768.0
    2493             72351744.0
    2494             23068672.0
    2495             20971520.0
    2496             26214400.0
    2497              2097152.0
    2498             41943040.0
    2499             38797312.0
    2500             41943040.0
    2501              4508876.8
    2502              5976883.2
    2503              4194304.0
    2504             17825792.0
    2505             11534336.0
    2506              4613734.4
    2507             15728640.0
    2508              2411724.8
    2509             41943040.0
    2510              8808038.4
    2511             11534336.0
    2512              3040870.4
    2513              2516582.4
    2514             15728640.0
    2515             25165824.0
    2516             30408704.0
    2517             25165824.0
    2518             23068672.0
    2519             90177536.0
    2520     Varies with device
    2521             23068672.0
    2522             11534336.0
    2523               998400.0
    2524              1003520.0
    2525              6291456.0
    2526             20971520.0
    2527             27262976.0
    2528             13631488.0
    2529             24117248.0
    2530              2831155.2
    2531              4404019.2
    2532             29360128.0
    2533              6396313.6
    2534             15728640.0
    2535              1258291.2
    2536             38797312.0
    2537              6186598.4
    2538             14680064.0
    2539              2411724.8
    2540             33554432.0
    2541              2831155.2
    2542              2306867.2
    2543             27262976.0
    2544     Varies with device
    2545     Varies with device
    2546     Varies with device
    2547              4194304.0
    2548     Varies with device
    2549              1572864.0
    2550     Varies with device
    2551              3879731.2
    2552     Varies with device
    2553     Varies with device
    2554     Varies with device
    2555              2936012.8
    2556              4089446.4
    2557              3250585.6
    2558     Varies with device
    2559              2936012.8
    2560             20971520.0
    2561             10380902.4
    2562              5557452.8
    2563             10485760.0
    2564              2726297.6
    2565     Varies with device
    2566             32505856.0
    2567              2411724.8
    2568              6291456.0
    2569             17825792.0
    2570             20971520.0
    2571             65011712.0
    2572     Varies with device
    2573             15728640.0
    2574     Varies with device
    2575             18874368.0
    2576              5662310.4
    2577              2936012.8
    2578             15728640.0
    2579     Varies with device
    2580              5557452.8
    2581     Varies with device
    2582     Varies with device
    2583             38797312.0
    2584     Varies with device
    2585             24117248.0
    2586     Varies with device
    2587     Varies with device
    2588     Varies with device
    2589     Varies with device
    2590             29360128.0
    2591             36700160.0
    2592     Varies with device
    2593             79691776.0
    2594     Varies with device
    2595             35651584.0
    2596     Varies with device
    2597     Varies with device
    2598     Varies with device
    2599     Varies with device
    2600              2831155.2
    2601     Varies with device
    2602              4299161.6
    2603     Varies with device
    2604     Varies with device
    2605     Varies with device
    2606     Varies with device
    2607             89128960.0
    2608             15728640.0
    2609             93323264.0
    2610     Varies with device
    2611     Varies with device
    2612     Varies with device
    2613             20971520.0
    2614     Varies with device
    2615              9751756.8
    2616     Varies with device
    2617             79691776.0
    2618             35651584.0
    2619             16777216.0
    2620     Varies with device
    2621     Varies with device
    2622     Varies with device
    2623     Varies with device
    2624     Varies with device
    2625     Varies with device
    2626              8808038.4
    2627     Varies with device
    2628             20971520.0
    2629     Varies with device
    2630             79691776.0
    2631             71303168.0
    2632             13631488.0
    2633             52428800.0
    2634             24117248.0
    2635     Varies with device
    2636     Varies with device
    2637             29360128.0
    2638     Varies with device
    2639     Varies with device
    2640              7340032.0
    2641             24117248.0
    2642     Varies with device
    2643             79691776.0
    2644     Varies with device
    2645             71303168.0
    2646     Varies with device
    2647     Varies with device
    2648             13631488.0
    2649             58720256.0
    2650     Varies with device
    2651     Varies with device
    2652              8808038.4
    2653             24117248.0
    2654             23068672.0
    2655             15728640.0
    2656     Varies with device
    2657             31457280.0
    2658             31457280.0
    2659             28311552.0
    2660     Varies with device
    2661     Varies with device
    2662     Varies with device
    2663             20971520.0
    2664             44040192.0
    2665     Varies with device
    2666             18874368.0
    2667              4404019.2
    2668     Varies with device
    2669             31457280.0
    2670             34603008.0
    2671              8283750.4
    2672             12582912.0
    2673             15728640.0
    2674     Varies with device
    2675             10485760.0
    2676             10380902.4
    2677              9437184.0
    2678     Varies with device
    2679             22020096.0
    2680             13631488.0
    2681             11534336.0
    2682             22020096.0
    2683              8493465.6
    2684     Varies with device
    2685     Varies with device
    2686             12582912.0
    2687              7654604.8
    2688     Varies with device
    2689             15728640.0
    2690             23068672.0
    2691             11534336.0
    2692              8703180.8
    2693     Varies with device
    2694             20971520.0
    2695             39845888.0
    2696              9542041.6
    2697             25165824.0
    2698     Varies with device
    2699     Varies with device
    2700             30408704.0
    2701     Varies with device
    2702            102760448.0
    2703             26214400.0
    2704     Varies with device
    2705             18874368.0
    2706     Varies with device
    2707              6396313.6
    2708             14680064.0
    2709             22020096.0
    2710     Varies with device
    2711             45088768.0
    2712             15728640.0
    2713             24117248.0
    2714             54525952.0
    2715     Varies with device
    2716     Varies with device
    2717             59768832.0
    2718             20971520.0
    2719             41943040.0
    2720             17825792.0
    2721             13631488.0
    2722             34603008.0
    2723             18874368.0
    2724     Varies with device
    2725     Varies with device
    2726     Varies with device
    2727     Varies with device
    2728             30408704.0
    2729     Varies with device
    2730             32505856.0
    2731     Varies with device
    2732     Varies with device
    2733     Varies with device
    2734             20971520.0
    2735             13631488.0
    2736              7654604.8
    2737     Varies with device
    2738     Varies with device
    2739     Varies with device
    2740             12582912.0
    2741              9227468.8
    2742             18874368.0
    2743             14680064.0
    2744             22020096.0
    2745             45088768.0
    2746             35651584.0
    2747     Varies with device
    2748     Varies with device
    2749             15728640.0
    2750     Varies with device
    2751             27262976.0
    2752     Varies with device
    2753             35651584.0
    2754              2831155.2
    2755             54525952.0
    2756     Varies with device
    2757             30408704.0
    2758     Varies with device
    2759     Varies with device
    2760             20971520.0
    2761             24117248.0
    2762     Varies with device
    2763             44040192.0
    2764             16777216.0
    2765             12582912.0
    2766              6815744.0
    2767     Varies with device
    2768             15728640.0
    2769             17825792.0
    2770             25165824.0
    2771             23068672.0
    2772     Varies with device
    2773             15728640.0
    2774     Varies with device
    2775             18874368.0
    2776              9227468.8
    2777             14680064.0
    2778              2831155.2
    2779             30408704.0
    2780             20971520.0
    2781              6815744.0
    2782             15728640.0
    2783     Varies with device
    2784     Varies with device
    2785              6501171.2
    2786             19922944.0
    2787             28311552.0
    2788              1153433.6
    2789              2831155.2
    2790     Varies with device
    2791             16777216.0
    2792              9227468.8
    2793             12582912.0
    2794             25165824.0
    2795             58720256.0
    2796             15728640.0
    2797             18874368.0
    2798     Varies with device
    2799             14680064.0
    2800             23068672.0
    2801             29360128.0
    2802             61865984.0
    2803             38797312.0
    2804             37748736.0
    2805              9961472.0
    2806             85983232.0
    2807             29360128.0
    2808     Varies with device
    2809     Varies with device
    2810             10171187.2
    2811             23068672.0
    2812             17825792.0
    2813     Varies with device
    2814     Varies with device
    2815              8074035.2
    2816     Varies with device
    2817             11534336.0
    2818             22020096.0
    2819             19922944.0
    2820     Varies with device
    2821              9122611.2
    2822             31457280.0
    2823             26214400.0
    2824             17825792.0
    2825     Varies with device
    2826             13631488.0
    2827             24117248.0
    2828     Varies with device
    2829     Varies with device
    2830             23068672.0
    2831              9542041.6
    2832              8703180.8
    2833     Varies with device
    2834             19922944.0
    2835             55574528.0
    2836     Varies with device
    2837             17825792.0
    2838             10485760.0
    2839     Varies with device
    2840             30408704.0
    2841             48234496.0
    2842              4718592.0
    2843              5138022.4
    2844     Varies with device
    2845             25165824.0
    2846             24117248.0
    2847             26214400.0
    2848             27262976.0
    2849     Varies with device
    2850              2097152.0
    2851             19922944.0
    2852             10066329.6
    2853     Varies with device
    2854             13631488.0
    2855              4404019.2
    2856             16777216.0
    2857             61865984.0
    2858     Varies with device
    2859     Varies with device
    2860              9961472.0
    2861     Varies with device
    2862             55574528.0
    2863              6920601.6
    2864     Varies with device
    2865     Varies with device
    2866             28311552.0
    2867     Varies with device
    2868             23068672.0
    2869     Varies with device
    2870             52428800.0
    2871             31457280.0
    2872     Varies with device
    2873             17825792.0
    2874             18874368.0
    2875             10485760.0
    2876             10380902.4
    2877             24117248.0
    2878             53477376.0
    2879             36700160.0
    2880             16777216.0
    2881             11534336.0
    2882             77594624.0
    2883             50331648.0
    2884     Varies with device
    2885             13631488.0
    2886              4404019.2
    2887             16777216.0
    2888             61865984.0
    2889     Varies with device
    2890              5872025.6
    2891              1572864.0
    2892              5976883.2
    2893              6396313.6
    2894              1992294.4
    2895              4194304.0
    2896             49283072.0
    2897     Varies with device
    2898              2097152.0
    2899              9961472.0
    2900     Varies with device
    2901             23068672.0
    2902             14680064.0
    2903             10485760.0
    2904     Varies with device
    2905     Varies with device
    2906              7235174.4
    2907             46137344.0
    2908     Varies with device
    2909     Varies with device
    2910     Varies with device
    2911             53477376.0
    2912             50331648.0
    2913     Varies with device
    2914             45088768.0
    2915             22020096.0
    2916     Varies with device
    2917             26214400.0
    2918              4089446.4
    2919              4194304.0
    2920             53477376.0
    2921     Varies with device
    2922     Varies with device
    2923             10066329.6
    2924     Varies with device
    2925              9961472.0
    2926     Varies with device
    2927             52428800.0
    2928     Varies with device
    2929     Varies with device
    2930             22020096.0
    2931     Varies with device
    2932             32505856.0
    2933             28311552.0
    2934     Varies with device
    2935     Varies with device
    2936     Varies with device
    2937             55574528.0
    2938             35651584.0
    2939     Varies with device
    2940             12582912.0
    2941             52428800.0
    2942             49283072.0
    2943     Varies with device
    2944     Varies with device
    2945     Varies with device
    2946              9646899.2
    2947     Varies with device
    2948     Varies with device
    2949             53477376.0
    2950             50331648.0
    2951             25165824.0
    2952              1677721.6
    2953             17825792.0
    2954             32505856.0
    2955             61865984.0
    2956             48234496.0
    2957             55574528.0
    2958             47185920.0
    2959     Varies with device
    2960             10276044.8
    2961     Varies with device
    2962     Varies with device
    2963     Varies with device
    2964             35651584.0
    2965             20971520.0
    2966             19922944.0
    2967              6291456.0
    2968     Varies with device
    2969             15728640.0
    2970              6815744.0
    2971              6501171.2
    2972     Varies with device
    2973     Varies with device
    2974     Varies with device
    2975     Varies with device
    2976     Varies with device
    2977     Varies with device
    2978             14680064.0
    2979     Varies with device
    2980     Varies with device
    2981              6396313.6
    2982             17825792.0
    2983             25165824.0
    2984     Varies with device
    2985             13631488.0
    2986             10485760.0
    2987             10485760.0
    2988              7235174.4
    2989             32505856.0
    2990     Varies with device
    2991     Varies with device
    2992             29360128.0
    2993              2726297.6
    2994             31457280.0
    2995             13631488.0
    2996             28311552.0
    2997             30408704.0
    2998             18874368.0
    2999             36700160.0
    3000             17825792.0
    3001     Varies with device
    3002             18874368.0
    3003     Varies with device
    3004              3145728.0
    3005             33554432.0
    3006             25165824.0
    3007     Varies with device
    3008     Varies with device
    3009     Varies with device
    3010     Varies with device
    3011     Varies with device
    3012     Varies with device
    3013             10485760.0
    3014             35651584.0
    3015     Varies with device
    3016              6920601.6
    3017             28311552.0
    3018     Varies with device
    3019     Varies with device
    3020     Varies with device
    3021             26214400.0
    3022              6920601.6
    3023             25165824.0
    3024     Varies with device
    3025             26214400.0
    3026             34603008.0
    3027             92274688.0
    3028              9751756.8
    3029     Varies with device
    3030     Varies with device
    3031             88080384.0
    3032              4928307.2
    3033             26214400.0
    3034             12582912.0
    3035     Varies with device
    3036             10380902.4
    3037     Varies with device
    3038             63963136.0
    3039             26214400.0
    3040             54525952.0
    3041             59768832.0
    3042     Varies with device
    3043             42991616.0
    3044              4823449.6
    3045             10485760.0
    3046             24117248.0
    3047             28311552.0
    3048     Varies with device
    3049             34603008.0
    3050     Varies with device
    3051              4928307.2
    3052              2306867.2
    3053     Varies with device
    3054             99614720.0
    3055             35651584.0
    3056     Varies with device
    3057              9856614.4
    3058             19922944.0
    3059              4089446.4
    3060     Varies with device
    3061     Varies with device
    3062     Varies with device
    3063             35651584.0
    3064     Varies with device
    3065             22020096.0
    3066             19922944.0
    3067              6920601.6
    3068             85983232.0
    3069     Varies with device
    3070     Varies with device
    3071     Varies with device
    3072     Varies with device
    3073             19922944.0
    3074     Varies with device
    3075             27262976.0
    3076              5452595.2
    3077             16777216.0
    3078     Varies with device
    3079             34603008.0
    3080     Varies with device
    3081     Varies with device
    3082              6291456.0
    3083             26214400.0
    3084     Varies with device
    3085             35651584.0
    3086              8493465.6
    3087             36700160.0
    3088             26214400.0
    3089     Varies with device
    3090     Varies with device
    3091             19922944.0
    3092             50331648.0
    3093             22020096.0
    3094              6920601.6
    3095             33554432.0
    3096     Varies with device
    3097             85983232.0
    3098             24117248.0
    3099             54525952.0
    3100     Varies with device
    3101     Varies with device
    3102             14680064.0
    3103     Varies with device
    3104     Varies with device
    3105             26214400.0
    3106     Varies with device
    3107             60817408.0
    3108             19922944.0
    3109             30408704.0
    3110             84934656.0
    3111     Varies with device
    3112     Varies with device
    3113             29360128.0
    3114     Varies with device
    3115     Varies with device
    3116     Varies with device
    3117     Varies with device
    3118     Varies with device
    3119              7969177.6
    3120             34603008.0
    3121     Varies with device
    3122             44040192.0
    3123              8703180.8
    3124             14680064.0
    3125     Varies with device
    3126     Varies with device
    3127     Varies with device
    3128             38797312.0
    3129             14680064.0
    3130     Varies with device
    3131             22020096.0
    3132             29360128.0
    3133             28311552.0
    3134     Varies with device
    3135             40894464.0
    3136     Varies with device
    3137             17825792.0
    3138     Varies with device
    3139             40894464.0
    3140             53477376.0
    3141             90177536.0
    3142     Varies with device
    3143     Varies with device
    3144     Varies with device
    3145             27262976.0
    3146     Varies with device
    3147             48234496.0
    3148             30408704.0
    3149             28311552.0
    3150             14680064.0
    3151     Varies with device
    3152             10485760.0
    3153             74448896.0
    3154             15728640.0
    3155     Varies with device
    3156     Varies with device
    3157             11534336.0
    3158     Varies with device
    3159             65011712.0
    3160             83886080.0
    3161              5662310.4
    3162             10276044.8
    3163     Varies with device
    3164     Varies with device
    3165     Varies with device
    3166              7969177.6
    3167             27262976.0
    3168             13631488.0
    3169     Varies with device
    3170     Varies with device
    3171             65011712.0
    3172             10485760.0
    3173             15728640.0
    3174             83886080.0
    3175             14680064.0
    3176     Varies with device
    3177             59768832.0
    3178             53477376.0
    3179              8703180.8
    3180             25165824.0
    3181     Varies with device
    3182             30408704.0
    3183     Varies with device
    3184              3250585.6
    3185             48234496.0
    3186             57671680.0
    3187             19922944.0
    3188     Varies with device
    3189             19922944.0
    3190             41943040.0
    3191     Varies with device
    3192             30408704.0
    3193             45088768.0
    3194             52428800.0
    3195     Varies with device
    3196             13631488.0
    3197     Varies with device
    3198     Varies with device
    3199     Varies with device
    3200             15728640.0
    3201              4194304.0
    3202     Varies with device
    3203             14680064.0
    3204     Varies with device
    3205             19922944.0
    3206     Varies with device
    3207             30408704.0
    3208             46137344.0
    3209             23068672.0
    3210     Varies with device
    3211     Varies with device
    3212     Varies with device
    3213             29360128.0
    3214     Varies with device
    3215     Varies with device
    3216             12582912.0
    3217     Varies with device
    3218             48234496.0
    3219              7969177.6
    3220     Varies with device
    3221             30408704.0
    3222     Varies with device
    3223     Varies with device
    3224              4299161.6
    3225              8912896.0
    3226             32505856.0
    3227              4299161.6
    3228     Varies with device
    3229             23068672.0
    3230     Varies with device
    3231             29360128.0
    3232     Varies with device
    3233              6186598.4
    3234     Varies with device
    3235     Varies with device
    3236     Varies with device
    3237              4089446.4
    3238     Varies with device
    3239             15728640.0
    3240              4508876.8
    3241     Varies with device
    3242     Varies with device
    3243     Varies with device
    3244     Varies with device
    3245              5557452.8
    3246     Varies with device
    3247             16777216.0
    3248              7864320.0
    3249     Varies with device
    3250     Varies with device
    3251     Varies with device
    3252              3879731.2
    3253     Varies with device
    3254              6081740.8
    3255             17825792.0
    3256     Varies with device
    3257              8912896.0
    3258     Varies with device
    3259              9542041.6
    3260              7969177.6
    3261              2621440.0
    3262             15728640.0
    3263     Varies with device
    3264     Varies with device
    3265     Varies with device
    3266     Varies with device
    3267              6396313.6
    3268     Varies with device
    3269             17825792.0
    3270     Varies with device
    3271     Varies with device
    3272     Varies with device
    3273     Varies with device
    3274             60817408.0
    3275              4823449.6
    3276             16777216.0
    3277              1363148.8
    3278              2411724.8
    3279              1992294.4
    3280              2831155.2
    3281             16777216.0
    3282             11534336.0
    3283              5557452.8
    3284     Varies with device
    3285              4404019.2
    3286     Varies with device
    3287              9542041.6
    3288              4508876.8
    3289     Varies with device
    3290     Varies with device
    3291     Varies with device
    3292     Varies with device
    3293              1887436.8
    3294     Varies with device
    3295              8912896.0
    3296              4613734.4
    3297     Varies with device
    3298              3984588.8
    3299             11534336.0
    3300     Varies with device
    3301             11534336.0
    3302     Varies with device
    3303              7759462.4
    3304              5557452.8
    3305              5662310.4
    3306              3774873.6
    3307     Varies with device
    3308              4508876.8
    3309     Varies with device
    3310             28311552.0
    3311     Varies with device
    3312              1887436.8
    3313              3460300.8
    3314              2097152.0
    3315              6396313.6
    3316             10066329.6
    3317             10066329.6
    3318             10485760.0
    3319              8598323.2
    3320             14680064.0
    3321             26214400.0
    3322             10380902.4
    3323             23068672.0
    3324     Varies with device
    3325              9332326.4
    3326     Varies with device
    3327              2516582.4
    3328             26214400.0
    3329     Varies with device
    3330     Varies with device
    3331              1887436.8
    3332             17825792.0
    3333     Varies with device
    3334              4299161.6
    3335             18874368.0
    3336             14680064.0
    3337     Varies with device
    3338     Varies with device
    3339     Varies with device
    3340               712704.0
    3341              4928307.2
    3342              8493465.6
    3343     Varies with device
    3344              8283750.4
    3345              4823449.6
    3346               557056.0
    3347              3670016.0
    3348     Varies with device
    3349     Varies with device
    3350               537600.0
    3351              2202009.6
    3352     Varies with device
    3353             14680064.0
    3354     Varies with device
    3355              9122611.2
    3356             22020096.0
    3357     Varies with device
    3358             12582912.0
    3359             18874368.0
    3360             17825792.0
    3361              3984588.8
    3362     Varies with device
    3363             24117248.0
    3364             14680064.0
    3365             15728640.0
    3366             10380902.4
    3367             12582912.0
    3368             12582912.0
    3369             20971520.0
    3370              7759462.4
    3371              3145728.0
    3372     Varies with device
    3373             14680064.0
    3374             14680064.0
    3375              7549747.2
    3376             23068672.0
    3377              7549747.2
    3378     Varies with device
    3379             27262976.0
    3380             39845888.0
    3381              4089446.4
    3382     Varies with device
    3383              5662310.4
    3384             10276044.8
    3385              7969177.6
    3386              1153433.6
    3387              9542041.6
    3388              7130316.8
    3389             12582912.0
    3390              3460300.8
    3391             12582912.0
    3392             10276044.8
    3393               942080.0
    3394             14680064.0
    3395              7130316.8
    3396             28311552.0
    3397              7759462.4
    3398     Varies with device
    3399             30408704.0
    3400     Varies with device
    3401     Varies with device
    3402              6186598.4
    3403             14680064.0
    3404             14680064.0
    3405             27262976.0
    3406              1992294.4
    3407              4089446.4
    3408               797696.0
    3409     Varies with device
    3410     Varies with device
    3411              5347737.6
    3412             10485760.0
    3413             13631488.0
    3414              4194304.0
    3415              5662310.4
    3416              3460300.8
    3417              4508876.8
    3418               873472.0
    3419             22020096.0
    3420              6710886.4
    3421              9017753.6
    3422              2202009.6
    3423              7444889.6
    3424     Varies with device
    3425     Varies with device
    3426              3460300.8
    3427              3670016.0
    3428              4299161.6
    3429             10171187.2
    3430              3460300.8
    3431              7444889.6
    3432              8808038.4
    3433              7340032.0
    3434              5767168.0
    3435     Varies with device
    3436     Varies with device
    3437     Varies with device
    3438             13631488.0
    3439     Varies with device
    3440     Varies with device
    3441              6186598.4
    3442     Varies with device
    3443             25165824.0
    3444              6501171.2
    3445     Varies with device
    3446     Varies with device
    3447              7235174.4
    3448     Varies with device
    3449              6396313.6
    3450     Varies with device
    3451     Varies with device
    3452     Varies with device
    3453             25165824.0
    3454     Varies with device
    3455              4299161.6
    3456     Varies with device
    3457             52428800.0
    3458     Varies with device
    3459     Varies with device
    3460             11534336.0
    3461              5347737.6
    3462              1363148.8
    3463     Varies with device
    3464     Varies with device
    3465     Varies with device
    3466     Varies with device
    3467     Varies with device
    3468              2411724.8
    3469             16777216.0
    3470     Varies with device
    3471     Varies with device
    3472             15728640.0
    3473             63963136.0
    3474              1677721.6
    3475     Varies with device
    3476     Varies with device
    3477     Varies with device
    3478     Varies with device
    3479             51380224.0
    3480     Varies with device
    3481              7549747.2
    3482     Varies with device
    3483     Varies with device
    3484              1572864.0
    3485     Varies with device
    3486     Varies with device
    3487             51380224.0
    3488             14680064.0
    3489     Varies with device
    3490     Varies with device
    3491     Varies with device
    3492     Varies with device
    3493     Varies with device
    3494              1258291.2
    3495     Varies with device
    3496             10485760.0
    3497              4508876.8
    3498              3984588.8
    3499     Varies with device
    3500             33554432.0
    3501              4718592.0
    3502     Varies with device
    3503              8808038.4
    3504             12582912.0
    3505             26214400.0
    3506     Varies with device
    3507     Varies with device
    3508     Varies with device
    3509             12582912.0
    3510     Varies with device
    3511              7444889.6
    3512     Varies with device
    3513     Varies with device
    3514             16777216.0
    3515              6815744.0
    3516              9646899.2
    3517               737280.0
    3518              2306867.2
    3519     Varies with device
    3520     Varies with device
    3521              4299161.6
    3522     Varies with device
    3523     Varies with device
    3524     Varies with device
    3525     Varies with device
    3526     Varies with device
    3527     Varies with device
    3528     Varies with device
    3529             62914560.0
    3530     Varies with device
    3531             38797312.0
    3532     Varies with device
    3533     Varies with device
    3534     Varies with device
    3535     Varies with device
    3536     Varies with device
    3537               730112.0
    3538              4928307.2
    3539              6920601.6
    3540              2621440.0
    3541             19922944.0
    3542     Varies with device
    3543              4404019.2
    3544             11534336.0
    3545     Varies with device
    3546     Varies with device
    3547     Varies with device
    3548     Varies with device
    3549             12582912.0
    3550     Varies with device
    3551     Varies with device
    3552              5976883.2
    3553              9437184.0
    3554     Varies with device
    3555              7235174.4
    3556              8808038.4
    3557              1992294.4
    3558     Varies with device
    3559     Varies with device
    3560     Varies with device
    3561             15728640.0
    3562     Varies with device
    3563     Varies with device
    3564     Varies with device
    3565     Varies with device
    3566     Varies with device
    3567     Varies with device
    3568             11534336.0
    3569             63963136.0
    3570              4299161.6
    3571              3984588.8
    3572     Varies with device
    3573              3984588.8
    3574     Varies with device
    3575              2936012.8
    3576     Varies with device
    3577              8598323.2
    3578              3460300.8
    3579              6501171.2
    3580             89128960.0
    3581             14680064.0
    3582             99614720.0
    3583              5767168.0
    3584             44040192.0
    3585             15728640.0
    3586            102760448.0
    3587              6710886.4
    3588              2936012.8
    3589              6815744.0
    3590              3774873.6
    3591              2516582.4
    3592              3565158.4
    3593              1572864.0
    3594              5242880.0
    3595              6920601.6
    3596     Varies with device
    3597     Varies with device
    3598     Varies with device
    3599             61865984.0
    3600              7130316.8
    3601             25165824.0
    3602              6920601.6
    3603              5138022.4
    3604              9542041.6
    3605             11534336.0
    3606              6081740.8
    3607             25165824.0
    3608              4613734.4
    3609             16777216.0
    3610             11534336.0
    3611              5138022.4
    3612             11534336.0
    3613             55574528.0
    3614             20971520.0
    3615             39845888.0
    3616              5138022.4
    3617             17825792.0
    3618             38797312.0
    3619              5452595.2
    3620             29360128.0
    3621             74448896.0
    3622             48234496.0
    3623             16777216.0
    3624             18874368.0
    3625             18874368.0
    3626     Varies with device
    3627             10485760.0
    3628     Varies with device
    3629             17825792.0
    3630     Varies with device
    3631             10171187.2
    3632     Varies with device
    3633     Varies with device
    3634             22020096.0
    3635              3355443.2
    3636             15728640.0
    3637             46137344.0
    3638              5033164.8
    3639     Varies with device
    3640             10485760.0
    3641             56623104.0
    3642              9542041.6
    3643             19922944.0
    3644             12582912.0
    3645     Varies with device
    3646     Varies with device
    3647     Varies with device
    3648             19922944.0
    3649     Varies with device
    3650              7969177.6
    3651     Varies with device
    3652             39845888.0
    3653     Varies with device
    3654     Varies with device
    3655              6396313.6
    3656              9646899.2
    3657     Varies with device
    3658             11534336.0
    3659              9646899.2
    3660             20971520.0
    3661             23068672.0
    3662     Varies with device
    3663     Varies with device
    3664              5557452.8
    3665     Varies with device
    3666              5872025.6
    3667              5662310.4
    3668              3040870.4
    3669             26214400.0
    3670     Varies with device
    3671     Varies with device
    3672             24117248.0
    3673              7549747.2
    3674              3460300.8
    3675     Varies with device
    3676     Varies with device
    3677             17825792.0
    3678              6291456.0
    3679             34603008.0
    3680              3250585.6
    3681              4299161.6
    3682             28311552.0
    3683             46137344.0
    3684     Varies with device
    3685             13631488.0
    3686     Varies with device
    3687     Varies with device
    3688              3145728.0
    3689     Varies with device
    3690     Varies with device
    3691              6396313.6
    3692             67108864.0
    3693             49283072.0
    3694     Varies with device
    3695              5662310.4
    3696     Varies with device
    3697     Varies with device
    3698     Varies with device
    3699             28311552.0
    3700     Varies with device
    3701             52428800.0
    3702     Varies with device
    3703     Varies with device
    3704              4194304.0
    3705             13631488.0
    3706     Varies with device
    3707     Varies with device
    3708             46137344.0
    3709              2621440.0
    3710             24117248.0
    3711     Varies with device
    3712             24117248.0
    3713             14680064.0
    3714             93323264.0
    3715     Varies with device
    3716     Varies with device
    3717             16777216.0
    3718             27262976.0
    3719     Varies with device
    3720             11534336.0
    3721     Varies with device
    3722     Varies with device
    3723             58720256.0
    3724             26214400.0
    3725     Varies with device
    3726             17825792.0
    3727              9542041.6
    3728             11534336.0
    3729     Varies with device
    3730             15728640.0
    3731             10171187.2
    3732              5767168.0
    3733              7025459.2
    3734     Varies with device
    3735              3040870.4
    3736             13631488.0
    3737              8912896.0
    3738             26214400.0
    3739     Varies with device
    3740             12582912.0
    3741              6606028.8
    3742              6606028.8
    3743     Varies with device
    3744             19922944.0
    3745             32505856.0
    3746     Varies with device
    3747              9437184.0
    3748             26214400.0
    3749              4823449.6
    3750              8388608.0
    3751             24117248.0
    3752             10276044.8
    3753              9122611.2
    3754     Varies with device
    3755     Varies with device
    3756             12582912.0
    3757     Varies with device
    3758             10485760.0
    3759     Varies with device
    3760             19922944.0
    3761              6920601.6
    3762     Varies with device
    3763             14680064.0
    3764              9017753.6
    3765             13631488.0
    3766             13631488.0
    3767     Varies with device
    3768     Varies with device
    3769             10485760.0
    3770     Varies with device
    3771     Varies with device
    3772     Varies with device
    3773              7864320.0
    3774             37748736.0
    3775              8598323.2
    3776             23068672.0
    3777             24117248.0
    3778     Varies with device
    3779     Varies with device
    3780     Varies with device
    3781             14680064.0
    3782     Varies with device
    3783             26214400.0
    3784             28311552.0
    3785     Varies with device
    3786     Varies with device
    3787             36700160.0
    3788     Varies with device
    3789              9017753.6
    3790     Varies with device
    3791              4823449.6
    3792             26214400.0
    3793             12582912.0
    3794             18874368.0
    3795     Varies with device
    3796     Varies with device
    3797              3250585.6
    3798     Varies with device
    3799             14680064.0
    3800     Varies with device
    3801     Varies with device
    3802     Varies with device
    3803             24117248.0
    3804     Varies with device
    3805              9227468.8
    3806             36700160.0
    3807     Varies with device
    3808     Varies with device
    3809             37748736.0
    3810     Varies with device
    3811             26214400.0
    3812             12582912.0
    3813     Varies with device
    3814             24117248.0
    3815     Varies with device
    3816             13631488.0
    3817     Varies with device
    3818             13631488.0
    3819     Varies with device
    3820     Varies with device
    3821     Varies with device
    3822     Varies with device
    3823             23068672.0
    3824     Varies with device
    3825     Varies with device
    3826             45088768.0
    3827     Varies with device
    3828             34603008.0
    3829             33554432.0
    3830     Varies with device
    3831              7130316.8
    3832              4194304.0
    3833     Varies with device
    3834              3879731.2
    3835     Varies with device
    3836              3565158.4
    3837     Varies with device
    3838     Varies with device
    3839     Varies with device
    3840             50331648.0
    3841              8074035.2
    3842             62914560.0
    3843             62914560.0
    3844              5872025.6
    3845     Varies with device
    3846              5452595.2
    3847             26214400.0
    3848              3460300.8
    3849             30408704.0
    3850             51380224.0
    3851              3774873.6
    3852             50331648.0
    3853     Varies with device
    3854              5662310.4
    3855             18874368.0
    3856              3355443.2
    3857             11534336.0
    3858              8912896.0
    3859             25165824.0
    3860             27262976.0
    3861             14680064.0
    3862     Varies with device
    3863     Varies with device
    3864             81788928.0
    3865     Varies with device
    3866     Varies with device
    3867              7340032.0
    3868              9646899.2
    3869     Varies with device
    3870     Varies with device
    3871              5138022.4
    3872              6815744.0
    3873             11534336.0
    3874            100663296.0
    3875              3040870.4
    3876             25165824.0
    3877             57671680.0
    3878             77594624.0
    3879             25165824.0
    3880             33554432.0
    3881             15728640.0
    3882              6606028.8
    3883            101711872.0
    3884             77594624.0
    3885     Varies with device
    3886             14680064.0
    3887             14680064.0
    3888     Varies with device
    3889     Varies with device
    3890              5976883.2
    3891             19922944.0
    3892     Varies with device
    3893     Varies with device
    3894             52428800.0
    3895     Varies with device
    3896             79691776.0
    3897             73400320.0
    3898             30408704.0
    3899             16777216.0
    3900     Varies with device
    3901             13631488.0
    3902             76546048.0
    3903             17825792.0
    3904     Varies with device
    3905     Varies with device
    3906     Varies with device
    3907     Varies with device
    3908             89128960.0
    3909     Varies with device
    3910            103809024.0
    3911     Varies with device
    3912             96468992.0
    3913             38797312.0
    3914             47185920.0
    3915     Varies with device
    3916             59768832.0
    3917             27262976.0
    3918             49283072.0
    3919             23068672.0
    3920             58720256.0
    3921             52428800.0
    3922             44040192.0
    3923             35651584.0
    3924     Varies with device
    3925             73400320.0
    3926             17825792.0
    3927              4299161.6
    3928     Varies with device
    3929             11534336.0
    3930             11534336.0
    3931             98566144.0
    3932             25165824.0
    3933             58720256.0
    3934     Varies with device
    3935             69206016.0
    3936             66060288.0
    3937             95420416.0
    3938             17825792.0
    3939             22020096.0
    3940     Varies with device
    3941     Varies with device
    3942             84934656.0
    3943     Varies with device
    3944              3145728.0
    3945             61865984.0
    3946     Varies with device
    3947             87031808.0
    3948             19922944.0
    3949             24117248.0
    3950             33554432.0
    3951             40894464.0
    3952             59768832.0
    3953             54525952.0
    3954              7864320.0
    3955              7759462.4
    3956              9332326.4
    3957              6606028.8
    3958             18874368.0
    3959             14680064.0
    3960     Varies with device
    3961             20971520.0
    3962     Varies with device
    3963            100663296.0
    3964             36700160.0
    3965              8808038.4
    3966              9017753.6
    3967             57671680.0
    3968             30408704.0
    3969             13631488.0
    3970             16777216.0
    3971             41943040.0
    3972              7025459.2
    3973            104857600.0
    3974     Varies with device
    3975             98566144.0
    3976              4613734.4
    3977             67108864.0
    3978              4718592.0
    3979              6291456.0
    3980     Varies with device
    3981              7759462.4
    3982              3879731.2
    3983             26214400.0
    3984             10485760.0
    3985             27262976.0
    3986            102760448.0
    3987            101711872.0
    3988              1887436.8
    3989     Varies with device
    3990             11534336.0
    3991             81788928.0
    3992             52428800.0
    3993     Varies with device
    3994             77594624.0
    3995              8703180.8
    3996     Varies with device
    3997              8074035.2
    3998              2621440.0
    3999              5557452.8
    4000     Varies with device
    4001              4194304.0
    4002              4404019.2
    4003             77594624.0
    4004              7235174.4
    4005     Varies with device
    4006              6920601.6
    4007             16777216.0
    4008              4928307.2
    4009              5662310.4
    4010             36700160.0
    4011             27262976.0
    4012             33554432.0
    4013              2831155.2
    4014              3879731.2
    4015              7340032.0
    4016              7235174.4
    4017             66060288.0
    4018              2306867.2
    4019             31457280.0
    4020              6081740.8
    4021              3250585.6
    4022              1572864.0
    4023              9017753.6
    4024               790528.0
    4025              3460300.8
    4026              5242880.0
    4027              1153433.6
    4028              4613734.4
    4029              2202009.6
    4030              1677721.6
    4031     Varies with device
    4032             30408704.0
    4033             40894464.0
    4034             30408704.0
    4035             14680064.0
    4036     Varies with device
    4037             42991616.0
    4038             10171187.2
    4039     Varies with device
    4040             62914560.0
    4041             65011712.0
    4042             49283072.0
    4043             93323264.0
    4044             10380902.4
    4045             67108864.0
    4046             50331648.0
    4047             12582912.0
    4048             63963136.0
    4049             33554432.0
    4050             29360128.0
    4051     Varies with device
    4052     Varies with device
    4053             37748736.0
    4054             13631488.0
    4055             72351744.0
    4056             77594624.0
    4057             18874368.0
    4058              3774873.6
    4059             82837504.0
    4060             69206016.0
    4061             44040192.0
    4062             10276044.8
    4063              2936012.8
    4064     Varies with device
    4065             41943040.0
    4066             20971520.0
    4067             26214400.0
    4068     Varies with device
    4069     Varies with device
    4070             46137344.0
    4071              9227468.8
    4072             16777216.0
    4073             93323264.0
    4074     Varies with device
    4075             60817408.0
    4076             20971520.0
    4077              5033164.8
    4078               325632.0
    4079     Varies with device
    4080             13631488.0
    4081              5138022.4
    4082             12582912.0
    4083     Varies with device
    4084     Varies with device
    4085     Varies with device
    4086             69206016.0
    4087             52428800.0
    4088     Varies with device
    4089             61865984.0
    4090     Varies with device
    4091             14680064.0
    4092              6186598.4
    4093             42991616.0
    4094     Varies with device
    4095             38797312.0
    4096     Varies with device
    4097     Varies with device
    4098     Varies with device
    4099     Varies with device
    4100             23068672.0
    4101     Varies with device
    4102             15728640.0
    4103              6396313.6
    4104     Varies with device
    4105     Varies with device
    4106     Varies with device
    4107              3250585.6
    4108                59392.0
    4109             82837504.0
    4110             15728640.0
    4111             57671680.0
    4112             11534336.0
    4113             13631488.0
    4114              5138022.4
    4115     Varies with device
    4116              7969177.6
    4117              3145728.0
    4118             38797312.0
    4119              9751756.8
    4120             24117248.0
    4121               246784.0
    4122     Varies with device
    4123             11534336.0
    4124              9646899.2
    4125              5033164.8
    4126     Varies with device
    4127             16777216.0
    4128             41943040.0
    4129             13631488.0
    4130              5138022.4
    4131             32505856.0
    4132     Varies with device
    4133     Varies with device
    4134             14680064.0
    4135              3145728.0
    4136             10380902.4
    4137     Varies with device
    4138              3879731.2
    4139              1782579.2
    4140             11534336.0
    4141             66060288.0
    4142              2202009.6
    4143              1468006.4
    4144     Varies with device
    4145             37748736.0
    4146             33554432.0
    4147     Varies with device
    4148             61865984.0
    4149             38797312.0
    4150     Varies with device
    4151              8283750.4
    4152             26214400.0
    4153     Varies with device
    4154             19922944.0
    4155             40894464.0
    4156               200704.0
    4157              2411724.8
    4158              8493465.6
    4159             53477376.0
    4160     Varies with device
    4161              3565158.4
    4162              9227468.8
    4163             29360128.0
    4164     Varies with device
    4165             11534336.0
    4166              6396313.6
    4167             14680064.0
    4168             98566144.0
    4169             11534336.0
    4170     Varies with device
    4171             14680064.0
    4172              1677721.6
    4173             15728640.0
    4174              2831155.2
    4175              5976883.2
    4176             49283072.0
    4177              1677721.6
    4178               877568.0
    4179             26214400.0
    4180              1363148.8
    4181              1572864.0
    4182             34603008.0
    4183              4089446.4
    4184             91226112.0
    4185             14680064.0
    4186             10485760.0
    4187              6186598.4
    4188             60817408.0
    4189              1992294.4
    4190             50331648.0
    4191             59768832.0
    4192                52224.0
    4193              4718592.0
    4194              9332326.4
    4195             27262976.0
    4196             15728640.0
    4197              1572864.0
    4198              2306867.2
    4199             44040192.0
    4200              5872025.6
    4201              8912896.0
    4202              9332326.4
    4203             13631488.0
    4204             13631488.0
    4205     Varies with device
    4206              5662310.4
    4207             28311552.0
    4208              3145728.0
    4209             19922944.0
    4210              8703180.8
    4211             59768832.0
    4212              1258291.2
    4213              2936012.8
    4214             19922944.0
    4215              7549747.2
    4216     Varies with device
    4217             54525952.0
    4218     Varies with device
    4219             68157440.0
    4220             38797312.0
    4221              3040870.4
    4222             11534336.0
    4223             35651584.0
    4224     Varies with device
    4225              6815744.0
    4226              4089446.4
    4227     Varies with device
    4228              4089446.4
    4229             59768832.0
    4230              2936012.8
    4231             24117248.0
    4232     Varies with device
    4233             57671680.0
    4234     Varies with device
    4235             78643200.0
    4236             41943040.0
    4237             52428800.0
    4238     Varies with device
    4239             26214400.0
    4240             26214400.0
    4241     Varies with device
    4242              6920601.6
    4243             37748736.0
    4244             78643200.0
    4245              9227468.8
    4246             45088768.0
    4247     Varies with device
    4248             10380902.4
    4249            100663296.0
    4250               975872.0
    4251              9437184.0
    4252             24117248.0
    4253              2726297.6
    4254             24117248.0
    4255             36700160.0
    4256              7549747.2
    4257             51380224.0
    4258              4928307.2
    4259     Varies with device
    4260             45088768.0
    4261            101711872.0
    4262     Varies with device
    4263              3250585.6
    4264              7025459.2
    4265             13631488.0
    4266              7444889.6
    4267              9542041.6
    4268            102760448.0
    4269             31457280.0
    4270              9332326.4
    4271             13631488.0
    4272              2516582.4
    4273             46137344.0
    4274              7654604.8
    4275             34603008.0
    4276              3565158.4
    4277              3984588.8
    4278              3774873.6
    4279     Varies with device
    4280             41943040.0
    4281             14680064.0
    4282     Varies with device
    4283     Varies with device
    4284     Varies with device
    4285             35651584.0
    4286             41943040.0
    4287     Varies with device
    4288             11534336.0
    4289             66060288.0
    4290             14680064.0
    4291             63963136.0
    4292             66060288.0
    4293              9332326.4
    4294             10485760.0
    4295             46137344.0
    4296             10485760.0
    4297             61865984.0
    4298              7759462.4
    4299             24117248.0
    4300             52428800.0
    4301             46137344.0
    4302     Varies with device
    4303             15728640.0
    4304             41943040.0
    4305             65011712.0
    4306              2202009.6
    4307             32505856.0
    4308             19922944.0
    4309             27262976.0
    4310              3774873.6
    4311             13631488.0
    4312              4718592.0
    4313             36700160.0
    4314             74448896.0
    4315             15728640.0
    4316             24117248.0
    4317              3670016.0
    4318             62914560.0
    4319             41943040.0
    4320     Varies with device
    4321             81788928.0
    4322             20971520.0
    4323     Varies with device
    4324             51380224.0
    4325             41943040.0
    4326             50331648.0
    4327             91226112.0
    4328             20971520.0
    4329             52428800.0
    4330               885760.0
    4331              3040870.4
    4332             70254592.0
    4333              2516582.4
    4334             63963136.0
    4335              7130316.8
    4336              2726297.6
    4337             13631488.0
    4338             30408704.0
    4339             88080384.0
    4340              5662310.4
    4341              2936012.8
    4342              3040870.4
    4343              7235174.4
    4344             48234496.0
    4345             28311552.0
    4346     Varies with device
    4347     Varies with device
    4348             38797312.0
    4349             85983232.0
    4350             46137344.0
    4351              7654604.8
    4352             38797312.0
    4353             55574528.0
    4354             46137344.0
    4355     Varies with device
    4356             25165824.0
    4357             35651584.0
    4358     Varies with device
    4359              9542041.6
    4360             39845888.0
    4361             15728640.0
    4362             27262976.0
    4363             66060288.0
    4364               257024.0
    4365     Varies with device
    4366              4404019.2
    4367              7654604.8
    4368             51380224.0
    4369             10171187.2
    4370              1468006.4
    4371              5138022.4
    4372              6501171.2
    4373              6291456.0
    4374     Varies with device
    4375              1258291.2
    4376             16777216.0
    4377             61865984.0
    4378              8074035.2
    4379             31457280.0
    4380              6710886.4
    4381             67108864.0
    4382             48234496.0
    4383             79691776.0
    4384             56623104.0
    4385              8808038.4
    4386             23068672.0
    4387             24117248.0
    4388              8598323.2
    4389              3984588.8
    4390             56623104.0
    4391              9542041.6
    4392             41943040.0
    4393             29360128.0
    4394             11534336.0
    4395             35651584.0
    4396             60817408.0
    4397             22020096.0
    4398             47185920.0
    4399             23068672.0
    4400             29360128.0
    4401             17825792.0
    4402             32505856.0
    4403             61865984.0
    4404             95420416.0
    4405     Varies with device
    4406             20971520.0
    4407             72351744.0
    4408             74448896.0
    4409             13631488.0
    4410             34603008.0
    4411              3460300.8
    4412              6710886.4
    4413             20971520.0
    4414              7969177.6
    4415             65011712.0
    4416             33554432.0
    4417             51380224.0
    4418               952320.0
    4419             35651584.0
    4420             51380224.0
    4421     Varies with device
    4422     Varies with device
    4423             25165824.0
    4424             75497472.0
    4425              9122611.2
    4426     Varies with device
    4427             41943040.0
    4428              7130316.8
    4429             49283072.0
    4430              3774873.6
    4431              3040870.4
    4432             27262976.0
    4433              1363148.8
    4434             47185920.0
    4435               552960.0
    4436              4823449.6
    4437               320512.0
    4438              9017753.6
    4439               763904.0
    4440             10485760.0
    4441             39845888.0
    4442              5347737.6
    4443              6396313.6
    4444             29360128.0
    4445              5452595.2
    4446             23068672.0
    4447     Varies with device
    4448             35651584.0
    4449             53477376.0
    4450             40894464.0
    4451              2202009.6
    4452             12582912.0
    4453             11534336.0
    4454              7864320.0
    4455             25165824.0
    4456             19922944.0
    4457              1992294.4
    4458              5452595.2
    4459              2411724.8
    4460              4718592.0
    4461              1782579.2
    4462              1887436.8
    4463              6606028.8
    4464     Varies with device
    4465              5767168.0
    4466              6501171.2
    4467             29360128.0
    4468              4508876.8
    4469              5976883.2
    4470              5138022.4
    4471              3774873.6
    4472              2097152.0
    4473             16777216.0
    4474              7759462.4
    4475              4508876.8
    4476              7654604.8
    4477             62914560.0
    4478              3879731.2
    4479             25165824.0
    4480             10485760.0
    4481              7864320.0
    4482             13631488.0
    4483              3774873.6
    4484              2097152.0
    4485             14680064.0
    4486              2411724.8
    4487             16777216.0
    4488             20971520.0
    4489             22020096.0
    4490              2202009.6
    4491             27262976.0
    4492             14680064.0
    4493             18874368.0
    4494             62914560.0
    4495     Varies with device
    4496              7444889.6
    4497     Varies with device
    4498             62914560.0
    4499             41943040.0
    4500             57671680.0
    4501     Varies with device
    4502             20971520.0
    4503             19922944.0
    4504     Varies with device
    4505             17825792.0
    4506              3774873.6
    4507              4718592.0
    4508              3774873.6
    4509             34603008.0
    4510             45088768.0
    4511             22020096.0
    4512              2411724.8
    4513             14680064.0
    4514             19922944.0
    4515             99614720.0
    4516             30408704.0
    4517              2831155.2
    4518     Varies with device
    4519     Varies with device
    4520             16777216.0
    4521              4089446.4
    4522              3250585.6
    4523             12582912.0
    4524             40894464.0
    4525             27262976.0
    4526             19922944.0
    4527             70254592.0
    4528             12582912.0
    4529              2936012.8
    4530              6501171.2
    4531              3040870.4
    4532              6186598.4
    4533             62914560.0
    4534              5138022.4
    4535             65011712.0
    4536            100663296.0
    4537              9646899.2
    4538              2936012.8
    4539              4089446.4
    4540              2306867.2
    4541               207872.0
    4542             38797312.0
    4543              3879731.2
    4544              8074035.2
    4545              3040870.4
    4546             30408704.0
    4547             35651584.0
    4548              3774873.6
    4549              5976883.2
    4550             67108864.0
    4551              3565158.4
    4552                59392.0
    4553              7025459.2
    4554             24117248.0
    4555              9332326.4
    4556             59768832.0
    4557             12582912.0
    4558             13631488.0
    4559             97517568.0
    4560             87031808.0
    4561             42991616.0
    4562             29360128.0
    4563              2831155.2
    4564              4613734.4
    4565              3565158.4
    4566             17825792.0
    4567              2306867.2
    4568     Varies with device
    4569              7444889.6
    4570             20971520.0
    4571                26624.0
    4572     Varies with device
    4573             32505856.0
    4574             47185920.0
    4575             34603008.0
    4576             14680064.0
    4577             38797312.0
    4578             25165824.0
    4579              6815744.0
    4580             41943040.0
    4581             97517568.0
    4582               321536.0
    4583              3879731.2
    4584             59768832.0
    4585              2831155.2
    4586     Varies with device
    4587             71303168.0
    4588             26214400.0
    4589     Varies with device
    4590     Varies with device
    4591              2516582.4
    4592     Varies with device
    4593     Varies with device
    4594            100663296.0
    4595     Varies with device
    4596              2306867.2
    4597             34603008.0
    4598     Varies with device
    4599             19922944.0
    4600     Varies with device
    4601             22020096.0
    4602     Varies with device
    4603             27262976.0
    4604             30408704.0
    4605     Varies with device
    4606              3879731.2
    4607             26214400.0
    4608             29360128.0
    4609              2202009.6
    4610     Varies with device
    4611             15728640.0
    4612              8283750.4
    4613             48234496.0
    4614              9332326.4
    4615             13631488.0
    4616             20971520.0
    4617     Varies with device
    4618              2411724.8
    4619              5138022.4
    4620             24117248.0
    4621             36700160.0
    4622             18874368.0
    4623             47185920.0
    4624              5872025.6
    4625              4718592.0
    4626              6081740.8
    4627             25165824.0
    4628             22020096.0
    4629             32505856.0
    4630             77594624.0
    4631             22020096.0
    4632              7654604.8
    4633              3460300.8
    4634     Varies with device
    4635             11534336.0
    4636             55574528.0
    4637             38797312.0
    4638             38797312.0
    4639              2097152.0
    4640             27262976.0
    4641             44040192.0
    4642              8178892.8
    4643              3670016.0
    4644             14680064.0
    4645              5347737.6
    4646              1992294.4
    4647             39845888.0
    4648              3145728.0
    4649              2306867.2
    4650              3670016.0
    4651              4194304.0
    4652             28311552.0
    4653             13631488.0
    4654             22020096.0
    4655              3670016.0
    4656             23068672.0
    4657              7549747.2
    4658              5033164.8
    4659             13631488.0
    4660              7864320.0
    4661              5662310.4
    4662             31457280.0
    4663     Varies with device
    4664               200704.0
    4665     Varies with device
    4666             16777216.0
    4667              3984588.8
    4668             17825792.0
    4669             27262976.0
    4670             29360128.0
    4671     Varies with device
    4672             39845888.0
    4673             17825792.0
    4674             20971520.0
    4675              7235174.4
    4676     Varies with device
    4677             61865984.0
    4678             70254592.0
    4679             63963136.0
    4680     Varies with device
    4681     Varies with device
    4682              1677721.6
    4683     Varies with device
    4684              3879731.2
    4685              5138022.4
    4686     Varies with device
    4687             31457280.0
    4688             41943040.0
    4689             10485760.0
    4690            104857600.0
    4691             44040192.0
    4692             99614720.0
    4693             29360128.0
    4694              4404019.2
    4695              3774873.6
    4696     Varies with device
    4697             17825792.0
    4698              4823449.6
    4699              8283750.4
    4700     Varies with device
    4701              4508876.8
    4702              3774873.6
    4703              6186598.4
    4704             36700160.0
    4705              1887436.8
    4706              9961472.0
    4707             19922944.0
    4708             26214400.0
    4709             38797312.0
    4710              3774873.6
    4711             17825792.0
    4712     Varies with device
    4713              3040870.4
    4714              3565158.4
    4715     Varies with device
    4716              3040870.4
    4717     Varies with device
    4718     Varies with device
    4719              3670016.0
    4720             38797312.0
    4721              1572864.0
    4722     Varies with device
    4723             49283072.0
    4724     Varies with device
    4725             11534336.0
    4726              9542041.6
    4727             33554432.0
    4728             16777216.0
    4729              2621440.0
    4730              7025459.2
    4731              1992294.4
    4732     Varies with device
    4733             28311552.0
    4734                80896.0
    4735              7969177.6
    4736              8598323.2
    4737             34603008.0
    4738              3250585.6
    4739             28311552.0
    4740     Varies with device
    4741              1572864.0
    4742     Varies with device
    4743              4718592.0
    4744     Varies with device
    4745             45088768.0
    4746             22020096.0
    4747     Varies with device
    4748     Varies with device
    4749              4613734.4
    4750             16777216.0
    4751             13631488.0
    4752             35651584.0
    4753              8703180.8
    4754              2621440.0
    4755              3670016.0
    4756              6710886.4
    4757              4508876.8
    4758              6606028.8
    4759              9017753.6
    4760              1992294.4
    4761               120832.0
    4762             51380224.0
    4763              3879731.2
    4764              3670016.0
    4765             55574528.0
    4766              8388608.0
    4767     Varies with device
    4768             28311552.0
    4769             25165824.0
    4770             11534336.0
    4771              3355443.2
    4772              3879731.2
    4773              1258291.2
    4774              8808038.4
    4775             32505856.0
    4776             10066329.6
    4777             47185920.0
    4778             58720256.0
    4779             27262976.0
    4780             17825792.0
    4781              7235174.4
    4782     Varies with device
    4783            103809024.0
    4784             49283072.0
    4785             37748736.0
    4786             22020096.0
    4787              3145728.0
    4788              8912896.0
    4789              2097152.0
    4790             15728640.0
    4791             52428800.0
    4792             17825792.0
    4793     Varies with device
    4794              1992294.4
    4795             73400320.0
    4796     Varies with device
    4797             42991616.0
    4798            103809024.0
    4799            100663296.0
    4800             16777216.0
    4801             88080384.0
    4802              5242880.0
    4803              5242880.0
    4804     Varies with device
    4805     Varies with device
    4806            103809024.0
    4807            103809024.0
    4808             27262976.0
    4809     Varies with device
    4810     Varies with device
    4811     Varies with device
    4812     Varies with device
    4813             38797312.0
    4814     Varies with device
    4815              9437184.0
    4816             92274688.0
    4817             91226112.0
    4818             81788928.0
    4819             70254592.0
    4820             61865984.0
    4821     Varies with device
    4822              1468006.4
    4823             61865984.0
    4824             49283072.0
    4825             79691776.0
    4826             65011712.0
    4827             83886080.0
    4828              3984588.8
    4829             58720256.0
    4830             75497472.0
    4831             26214400.0
    4832             82837504.0
    4833            102760448.0
    4834             71303168.0
    4835             95420416.0
    4836             46137344.0
    4837            100663296.0
    4838             63963136.0
    4839     Varies with device
    4840             19922944.0
    4841              9227468.8
    4842            100663296.0
    4843              8074035.2
    4844     Varies with device
    4845             10485760.0
    4846              2306867.2
    4847             16777216.0
    4848              5976883.2
    4849             30408704.0
    4850             25165824.0
    4851              4508876.8
    4852             50331648.0
    4853              2411724.8
    4854             13631488.0
    4855              7969177.6
    4856            103809024.0
    4857             23068672.0
    4858             12582912.0
    4859             92274688.0
    4860             95420416.0
    4861            103809024.0
    4862             10485760.0
    4863     Varies with device
    4864             59768832.0
    4865             18874368.0
    4866             34603008.0
    4867              8598323.2
    4868     Varies with device
    4869             48234496.0
    4870             66060288.0
    4871               244736.0
    4872     Varies with device
    4873     Varies with device
    4874             50331648.0
    4875     Varies with device
    4876             50331648.0
    4877             18874368.0
    4878              8703180.8
    4879             56623104.0
    4880     Varies with device
    4881             31457280.0
    4882              8703180.8
    4883             24117248.0
    4884             11534336.0
    4885              8808038.4
    4886             85983232.0
    4887              5242880.0
    4888             17825792.0
    4889             38797312.0
    4890             47185920.0
    4891             41943040.0
    4892              7340032.0
    4893              3984588.8
    4894             23068672.0
    4895             15728640.0
    4896              8912896.0
    4897               379904.0
    4898              5452595.2
    4899     Varies with device
    4900             13631488.0
    4901             96468992.0
    4902              2097152.0
    4903              5242880.0
    4904              1887436.8
    4905              1782579.2
    4906             25165824.0
    4907             24117248.0
    4908              5347737.6
    4909              8283750.4
    4910              8178892.8
    4911              4404019.2
    4912             18874368.0
    4913              3460300.8
    4914              8074035.2
    4915              3250585.6
    4916             28311552.0
    4917               225280.0
    4918             27262976.0
    4919              3145728.0
    4920             29360128.0
    4921             11534336.0
    4922              6396313.6
    4923             27262976.0
    4924              5242880.0
    4925              2202009.6
    4926              6396313.6
    4927              2306867.2
    4928             14680064.0
    4929              2411724.8
    4930             29360128.0
    4931              3145728.0
    4932             10171187.2
    4933              3565158.4
    4934              4089446.4
    4935              9856614.4
    4936             27262976.0
    4937             28311552.0
    4938              4613734.4
    4939             27262976.0
    4940              3355443.2
    4941              7340032.0
    4942     Varies with device
    4943              1468006.4
    4944              1782579.2
    4945     Varies with device
    4946     Varies with device
    4947              1572864.0
    4948             40894464.0
    4949              6396313.6
    4950              4508876.8
    4951     Varies with device
    4952              8703180.8
    4953               747520.0
    4954              2202009.6
    4955             27262976.0
    4956               774144.0
    4957              3355443.2
    4958              4299161.6
    4959             20971520.0
    4960              3774873.6
    4961             46137344.0
    4962             27262976.0
    4963             17825792.0
    4964              1153433.6
    4965              3040870.4
    4966             25165824.0
    4967             14680064.0
    4968              7235174.4
    4969             12582912.0
    4970                93184.0
    4971              9856614.4
    4972              1782579.2
    4973               300032.0
    4974     Varies with device
    4975             36700160.0
    4976              3984588.8
    4977                17408.0
    4978                75776.0
    4979             20971520.0
    4980              1153433.6
    4981              4299161.6
    4982             13631488.0
    4983                14336.0
    4984              8703180.8
    4985              9437184.0
    4986              9437184.0
    4987     Varies with device
    4988              3460300.8
    4989     Varies with device
    4990             29360128.0
    4991             19922944.0
    4992             61865984.0
    4993              1992294.4
    4994             12582912.0
    4995              1887436.8
    4996             31457280.0
    4997     Varies with device
    4998     Varies with device
    4999     Varies with device
    5000             18874368.0
    5001     Varies with device
    5002             30408704.0
    5003             27262976.0
    5004             12582912.0
    5005             12582912.0
    5006             17825792.0
    5007             18874368.0
    5008             27262976.0
    5009     Varies with device
    5010             17825792.0
    5011             12582912.0
    5012               324608.0
    5013              7864320.0
    5014     Varies with device
    5015             69206016.0
    5016             31457280.0
    5017             36700160.0
    5018             27262976.0
    5019             15728640.0
    5020             28311552.0
    5021             40894464.0
    5022              9542041.6
    5023             13631488.0
    5024             36700160.0
    5025              5347737.6
    5026              7025459.2
    5027             38797312.0
    5028              4089446.4
    5029              4194304.0
    5030              7969177.6
    5031     Varies with device
    5032              6920601.6
    5033             33554432.0
    5034              3040870.4
    5035                79872.0
    5036              1258291.2
    5037             15728640.0
    5038             18874368.0
    5039             15728640.0
    5040             19922944.0
    5041              4299161.6
    5042              5872025.6
    5043              3670016.0
    5044              6710886.4
    5045              3145728.0
    5046               946176.0
    5047             11534336.0
    5048              5138022.4
    5049     Varies with device
    5050              6396313.6
    5051               923648.0
    5052             55574528.0
    5053              1363148.8
    5054              4928307.2
    5055              3250585.6
    5056             11534336.0
    5057     Varies with device
    5058     Varies with device
    5059              3879731.2
    5060               272384.0
    5061             19922944.0
    5062              2097152.0
    5063              4299161.6
    5064              3565158.4
    5065             13631488.0
    5066             48234496.0
    5067              4089446.4
    5068             11534336.0
    5069     Varies with device
    5070              3670016.0
    5071     Varies with device
    5072             11534336.0
    5073     Varies with device
    5074             17825792.0
    5075             17825792.0
    5076             11534336.0
    5077     Varies with device
    5078             67108864.0
    5079             10485760.0
    5080             36700160.0
    5081             12582912.0
    5082              2516582.4
    5083              3879731.2
    5084              6606028.8
    5085             23068672.0
    5086             10485760.0
    5087              3879731.2
    5088              4823449.6
    5089             46137344.0
    5090             18874368.0
    5091             10171187.2
    5092              2831155.2
    5093             53477376.0
    5094             49283072.0
    5095             58720256.0
    5096     Varies with device
    5097             25165824.0
    5098     Varies with device
    5099             54525952.0
    5100              5138022.4
    5101              1782579.2
    5102             16777216.0
    5103              6815744.0
    5104             28311552.0
    5105             37748736.0
    5106              5452595.2
    5107             45088768.0
    5108              9017753.6
    5109              2097152.0
    5110             14680064.0
    5111             15728640.0
    5112             12582912.0
    5113             47185920.0
    5114              3355443.2
    5115     Varies with device
    5116             65011712.0
    5117             15728640.0
    5118             13631488.0
    5119             18874368.0
    5120              1782579.2
    5121             15728640.0
    5122              3040870.4
    5123              6186598.4
    5124              4404019.2
    5125             77594624.0
    5126              1782579.2
    5127             49283072.0
    5128              7235174.4
    5129              9122611.2
    5130               837632.0
    5131             29360128.0
    5132             30408704.0
    5133             61865984.0
    5134     Varies with device
    5135             30408704.0
    5136             30408704.0
    5137     Varies with device
    5138             61865984.0
    5139             28311552.0
    5140             30408704.0
    5141             30408704.0
    5142             14680064.0
    5143             10485760.0
    5144                82944.0
    5145             29360128.0
    5146             10276044.8
    5147             30408704.0
    5148             29360128.0
    5149               325632.0
    5150              5662310.4
    5151             29360128.0
    5152             30408704.0
    5153             30408704.0
    5154             30408704.0
    5155              3145728.0
    5156             28311552.0
    5157             30408704.0
    5158             90177536.0
    5159             23068672.0
    5160             30408704.0
    5161             30408704.0
    5162             30408704.0
    5163             16777216.0
    5164             28311552.0
    5165             30408704.0
    5166              5976883.2
    5167              5976883.2
    5168             30408704.0
    5169              2097152.0
    5170             30408704.0
    5171             20971520.0
    5172             29360128.0
    5173             28311552.0
    5174             30408704.0
    5175              9542041.6
    5176             30408704.0
    5177             30408704.0
    5178             36700160.0
    5179              6186598.4
    5180              1258291.2
    5181             40894464.0
    5182             28311552.0
    5183              3355443.2
    5184              4404019.2
    5185             32505856.0
    5186              4089446.4
    5187             26214400.0
    5188     Varies with device
    5189             72351744.0
    5190              4508876.8
    5191             10276044.8
    5192             13631488.0
    5193             16777216.0
    5194              4404019.2
    5195            103809024.0
    5196              2411724.8
    5197              3774873.6
    5198             27262976.0
    5199              3355443.2
    5200              2936012.8
    5201              8703180.8
    5202     Varies with device
    5203              3984588.8
    5204              3565158.4
    5205              3670016.0
    5206              1992294.4
    5207              3460300.8
    5208     Varies with device
    5209     Varies with device
    5210              7444889.6
    5211              8283750.4
    5212              1468006.4
    5213              1887436.8
    5214              7864320.0
    5215              4508876.8
    5216              2936012.8
    5217              8283750.4
    5218              3879731.2
    5219             50331648.0
    5220             14680064.0
    5221              5976883.2
    5222               961536.0
    5223             60817408.0
    5224             44040192.0
    5225             36700160.0
    5226             45088768.0
    5227             66060288.0
    5228             62914560.0
    5229     Varies with device
    5230              2936012.8
    5231              3355443.2
    5232              4404019.2
    5233             33554432.0
    5234             23068672.0
    5235              7654604.8
    5236              5557452.8
    5237             52428800.0
    5238              7864320.0
    5239             27262976.0
    5240             17825792.0
    5241              4404019.2
    5242             32505856.0
    5243              7969177.6
    5244              1782579.2
    5245               173056.0
    5246             66060288.0
    5247              8074035.2
    5248                46080.0
    5249              6920601.6
    5250             14680064.0
    5251             23068672.0
    5252              3774873.6
    5253              8808038.4
    5254             26214400.0
    5255               486400.0
    5256              4508876.8
    5257             36700160.0
    5258              3774873.6
    5259             25165824.0
    5260             36700160.0
    5261             95420416.0
    5262             28311552.0
    5263             32505856.0
    5264              2831155.2
    5265             33554432.0
    5266             28311552.0
    5267             70254592.0
    5268              8598323.2
    5269             31457280.0
    5270              4089446.4
    5271              1153433.6
    5272             50331648.0
    5273             14680064.0
    5274             18874368.0
    5275              3040870.4
    5276              4613734.4
    5277             93323264.0
    5278             13631488.0
    5279              3984588.8
    5280             15728640.0
    5281             15728640.0
    5282             26214400.0
    5283             35651584.0
    5284              7130316.8
    5285             25165824.0
    5286              3565158.4
    5287             25165824.0
    5288              6081740.8
    5289              5138022.4
    5290             16777216.0
    5291             48234496.0
    5292             34603008.0
    5293             45088768.0
    5294             58720256.0
    5295              2936012.8
    5296             11534336.0
    5297             27262976.0
    5298             11534336.0
    5299              9227468.8
    5300              6501171.2
    5301              7969177.6
    5302              5138022.4
    5303              5347737.6
    5304              4404019.2
    5305              7340032.0
    5306             39845888.0
    5307              9122611.2
    5308              2202009.6
    5309             12582912.0
    5310             60817408.0
    5311             18874368.0
    5312             20971520.0
    5313              8388608.0
    5314              3145728.0
    5315             12582912.0
    5316              8388608.0
    5317             29360128.0
    5318             30408704.0
    5319              4928307.2
    5320             41943040.0
    5321             14680064.0
    5322     Varies with device
    5323             16777216.0
    5324             10171187.2
    5325             51380224.0
    5326              3879731.2
    5327     Varies with device
    5328             28311552.0
    5329              3774873.6
    5330              5557452.8
    5331              2516582.4
    5332              8912896.0
    5333              7025459.2
    5334     Varies with device
    5335              9437184.0
    5336               272384.0
    5337     Varies with device
    5338             28311552.0
    5339             70254592.0
    5340             23068672.0
    5341             11534336.0
    5342             71303168.0
    5343             41943040.0
    5344             15728640.0
    5345     Varies with device
    5346             24117248.0
    5347              4404019.2
    5348              3040870.4
    5349     Varies with device
    5350             89128960.0
    5351              1887436.8
    5352              1153433.6
    5353             23068672.0
    5354              9122611.2
    5355              2726297.6
    5356              4928307.2
    5357              3040870.4
    5358             23068672.0
    5359               988160.0
    5360              1887436.8
    5361              2097152.0
    5362              2831155.2
    5363             58720256.0
    5364              2831155.2
    5365              3040870.4
    5366              5138022.4
    5367             20971520.0
    5368             12582912.0
    5369              3984588.8
    5370              4404019.2
    5371              1468006.4
    5372             22020096.0
    5373             42991616.0
    5374     Varies with device
    5375             59768832.0
    5376             15728640.0
    5377              5347737.6
    5378             32505856.0
    5379             29360128.0
    5380             40894464.0
    5381     Varies with device
    5382             35651584.0
    5383     Varies with device
    5384             32505856.0
    5385             79691776.0
    5386             41943040.0
    5387             48234496.0
    5388              9122611.2
    5389             83886080.0
    5390             25165824.0
    5391             22020096.0
    5392     Varies with device
    5393             10171187.2
    5394             12582912.0
    5395     Varies with device
    5396     Varies with device
    5397             59768832.0
    5398             84934656.0
    5399     Varies with device
    5400              7025459.2
    5401              3984588.8
    5402              6186598.4
    5403             48234496.0
    5404              3460300.8
    5405             50331648.0
    5406             34603008.0
    5407             44040192.0
    5408            103809024.0
    5409              4718592.0
    5410             29360128.0
    5411              8493465.6
    5412             22020096.0
    5413             58720256.0
    5414     Varies with device
    5415             26214400.0
    5416             27262976.0
    5417            103809024.0
    5418             26214400.0
    5419            103809024.0
    5420             45088768.0
    5421             17825792.0
    5422             83886080.0
    5423             46137344.0
    5424     Varies with device
    5425     Varies with device
    5426             12582912.0
    5427            104857600.0
    5428             72351744.0
    5429             94371840.0
    5430             92274688.0
    5431             67108864.0
    5432             40894464.0
    5433             56623104.0
    5434             58720256.0
    5435             14680064.0
    5436     Varies with device
    5437             78643200.0
    5438             13631488.0
    5439              8493465.6
    5440     Varies with device
    5441             20971520.0
    5442             59768832.0
    5443             32505856.0
    5444             40894464.0
    5445             30408704.0
    5446              3774873.6
    5447              3984588.8
    5448              4823449.6
    5449               558080.0
    5450              2202009.6
    5451                62464.0
    5452             13631488.0
    5453              2936012.8
    5454             10380902.4
    5455              2621440.0
    5456              5662310.4
    5457             14680064.0
    5458              1363148.8
    5459              2621440.0
    5460               289792.0
    5461              2411724.8
    5462              1677721.6
    5463             26214400.0
    5464              5662310.4
    5465              1572864.0
    5466              5557452.8
    5467              2097152.0
    5468              5662310.4
    5469             42991616.0
    5470              5662310.4
    5471              2516582.4
    5472              5662310.4
    5473              9437184.0
    5474             26214400.0
    5475              1258291.2
    5476             47185920.0
    5477              5557452.8
    5478              3355443.2
    5479             76546048.0
    5480             11534336.0
    5481              5662310.4
    5482               670720.0
    5483              5872025.6
    5484             17825792.0
    5485              9961472.0
    5486              7759462.4
    5487              5662310.4
    5488              2726297.6
    5489            100663296.0
    5490             76546048.0
    5491              5662310.4
    5492             36700160.0
    5493             50331648.0
    5494     Varies with device
    5495     Varies with device
    5496             38797312.0
    5497             17825792.0
    5498             36700160.0
    5499              2411724.8
    5500             16777216.0
    5501             27262976.0
    5502              3040870.4
    5503               731136.0
    5504             12582912.0
    5505     Varies with device
    5506             54525952.0
    5507     Varies with device
    5508              3355443.2
    5509             51380224.0
    5510             29360128.0
    5511             12582912.0
    5512     Varies with device
    5513             17825792.0
    5514             38797312.0
    5515              2097152.0
    5516     Varies with device
    5517             85983232.0
    5518     Varies with device
    5519     Varies with device
    5520             38797312.0
    5521             50331648.0
    5522             56623104.0
    5523             48234496.0
    5524             53477376.0
    5525             50331648.0
    5526             57671680.0
    5527     Varies with device
    5528             26214400.0
    5529             34603008.0
    5530            104857600.0
    5531             73400320.0
    5532     Varies with device
    5533              9646899.2
    5534             45088768.0
    5535             95420416.0
    5536             30408704.0
    5537     Varies with device
    5538             46137344.0
    5539             31457280.0
    5540            103809024.0
    5541             94371840.0
    5542             75497472.0
    5543             49283072.0
    5544             24117248.0
    5545     Varies with device
    5546             27262976.0
    5547              5242880.0
    5548     Varies with device
    5549             20971520.0
    5550             31457280.0
    5551             38797312.0
    5552             29360128.0
    5553             89128960.0
    5554     Varies with device
    5555             23068672.0
    5556             85983232.0
    5557             68157440.0
    5558             45088768.0
    5559             83886080.0
    5560             22020096.0
    5561                95232.0
    5562             94371840.0
    5563             70254592.0
    5564     Varies with device
    5565             33554432.0
    5566              2831155.2
    5567             37748736.0
    5568             36700160.0
    5569             42991616.0
    5570     Varies with device
    5571             74448896.0
    5572             73400320.0
    5573             23068672.0
    5574             14680064.0
    5575              6291456.0
    5576     Varies with device
    5577             55574528.0
    5578               892928.0
    5579              1363148.8
    5580              1677721.6
    5581     Varies with device
    5582             58720256.0
    5583              3565158.4
    5584              3250585.6
    5585             20971520.0
    5586              6501171.2
    5587               123904.0
    5588     Varies with device
    5589             99614720.0
    5590              9856614.4
    5591     Varies with device
    5592     Varies with device
    5593              2621440.0
    5594     Varies with device
    5595             97517568.0
    5596             73400320.0
    5597              1782579.2
    5598             38797312.0
    5599             73400320.0
    5600     Varies with device
    5601     Varies with device
    5602     Varies with device
    5603     Varies with device
    5604              3460300.8
    5605             16777216.0
    5606     Varies with device
    5607             19922944.0
    5608             23068672.0
    5609     Varies with device
    5610             71303168.0
    5611     Varies with device
    5612     Varies with device
    5613     Varies with device
    5614             35651584.0
    5615             48234496.0
    5616             96468992.0
    5617             47185920.0
    5618             52428800.0
    5619              9961472.0
    5620     Varies with device
    5621             37748736.0
    5622             81788928.0
    5623     Varies with device
    5624             23068672.0
    5625             50331648.0
    5626              1572864.0
    5627             41943040.0
    5628              6815744.0
    5629             53477376.0
    5630             12582912.0
    5631             52428800.0
    5632             41943040.0
    5633             56623104.0
    5634             47185920.0
    5635             66060288.0
    5636              4194304.0
    5637             72351744.0
    5638             40894464.0
    5639             57671680.0
    5640             48234496.0
    5641            103809024.0
    5642             95420416.0
    5643             22020096.0
    5644             48234496.0
    5645             47185920.0
    5646             25165824.0
    5647             54525952.0
    5648             52428800.0
    5649             66060288.0
    5650              6081740.8
    5651     Varies with device
    5652             24117248.0
    5653             24117248.0
    5654             27262976.0
    5655              9227468.8
    5656             28311552.0
    5657              2202009.6
    5658             12582912.0
    5659              5242880.0
    5660             24117248.0
    5661              8598323.2
    5662             14680064.0
    5663              8912896.0
    5664     Varies with device
    5665             49283072.0
    5666             12582912.0
    5667             17825792.0
    5668              4089446.4
    5669             10485760.0
    5670              4299161.6
    5671             15728640.0
    5672              3460300.8
    5673              7025459.2
    5674             11534336.0
    5675              4299161.6
    5676              3879731.2
    5677              4508876.8
    5678             32505856.0
    5679             26214400.0
    5680              2097152.0
    5681              6081740.8
    5682              3879731.2
    5683              1782579.2
    5684             12582912.0
    5685             14680064.0
    5686             81788928.0
    5687              6710886.4
    5688             10485760.0
    5689             46137344.0
    5690              1572864.0
    5691             89128960.0
    5692             10276044.8
    5693              5976883.2
    5694              3879731.2
    5695     Varies with device
    5696     Varies with device
    5697             48234496.0
    5698             31457280.0
    5699             49283072.0
    5700             24117248.0
    5701     Varies with device
    5702             95420416.0
    5703             46137344.0
    5704     Varies with device
    5705              5033164.8
    5706              9961472.0
    5707             19922944.0
    5708              7654604.8
    5709              1887436.8
    5710              5976883.2
    5711              7654604.8
    5712              4194304.0
    5713              3879731.2
    5714             27262976.0
    5715              5872025.6
    5716              7654604.8
    5717             51380224.0
    5718              9122611.2
    5719             51380224.0
    5720             20971520.0
    5721              7654604.8
    5722               329728.0
    5723              6606028.8
    5724              3565158.4
    5725             23068672.0
    5726             26214400.0
    5727              6606028.8
    5728             85983232.0
    5729              9122611.2
    5730             10485760.0
    5731             25165824.0
    5732     Varies with device
    5733             45088768.0
    5734     Varies with device
    5735              3879731.2
    5736             13631488.0
    5737     Varies with device
    5738              4089446.4
    5739             13631488.0
    5740              3145728.0
    5741              3145728.0
    5742     Varies with device
    5743             48234496.0
    5744              3145728.0
    5745             17825792.0
    5746             66060288.0
    5747             17825792.0
    5748              3040870.4
    5749              3040870.4
    5750             27262976.0
    5751              2831155.2
    5752     Varies with device
    5753     Varies with device
    5754             25165824.0
    5755              4928307.2
    5756              2831155.2
    5757              1782579.2
    5758              3145728.0
    5759              3145728.0
    5760     Varies with device
    5761             32505856.0
    5762             37748736.0
    5763              4508876.8
    5764             66060288.0
    5765              9017753.6
    5766             59768832.0
    5767     Varies with device
    5768             13631488.0
    5769              5872025.6
    5770     Varies with device
    5771              1048576.0
    5772     Varies with device
    5773     Varies with device
    5774             24117248.0
    5775              1677721.6
    5776             25165824.0
    5777              6186598.4
    5778     Varies with device
    5779     Varies with device
    5780             46137344.0
    5781             31457280.0
    5782             79691776.0
    5783               999424.0
    5784             10485760.0
    5785             46137344.0
    5786             19922944.0
    5787              3145728.0
    5788             15728640.0
    5789             15728640.0
    5790             45088768.0
    5791              2726297.6
    5792               176128.0
    5793             25165824.0
    5794             32505856.0
    5795             26214400.0
    5796              2411724.8
    5797             34603008.0
    5798              9437184.0
    5799              7759462.4
    5800             19922944.0
    5801              2411724.8
    5802             44040192.0
    5803     Varies with device
    5804              3460300.8
    5805              9542041.6
    5806              2936012.8
    5807             13631488.0
    5808             31457280.0
    5809              3145728.0
    5810              2306867.2
    5811             14680064.0
    5812               243712.0
    5813             38797312.0
    5814             54525952.0
    5815              4613734.4
    5816     Varies with device
    5817             29360128.0
    5818             33554432.0
    5819              3565158.4
    5820             24117248.0
    5821             63963136.0
    5822             15728640.0
    5823             67108864.0
    5824     Varies with device
    5825             12582912.0
    5826              4613734.4
    5827              3774873.6
    5828              3040870.4
    5829             10485760.0
    5830              7549747.2
    5831              3984588.8
    5832               562176.0
    5833              4089446.4
    5834              3040870.4
    5835             44040192.0
    5836              1153433.6
    5837             10171187.2
    5838              1782579.2
    5839             13631488.0
    5840             57671680.0
    5841              6815744.0
    5842              1992294.4
    5843              5347737.6
    5844             37748736.0
    5845             11534336.0
    5846             14680064.0
    5847     Varies with device
    5848             17825792.0
    5849     Varies with device
    5850              4508876.8
    5851              4194304.0
    5852              4194304.0
    5853              3145728.0
    5854              1258291.2
    5855             30408704.0
    5856     Varies with device
    5857              3565158.4
    5858              5347737.6
    5859              2831155.2
    5860            101711872.0
    5861             19922944.0
    5862            104857600.0
    5863     Varies with device
    5864             76546048.0
    5865            104857600.0
    5866             18874368.0
    5867             18874368.0
    5868             46137344.0
    5869     Varies with device
    5870              5452595.2
    5871              9332326.4
    5872             35651584.0
    5873             31457280.0
    5874              1677721.6
    5875              4613734.4
    5876              3879731.2
    5877             11534336.0
    5878              4928307.2
    5879             34603008.0
    5880             13631488.0
    5881             14680064.0
    5882              3879731.2
    5883              7654604.8
    5884     Varies with device
    5885             38797312.0
    5886             13631488.0
    5887             13631488.0
    5888              8808038.4
    5889             47185920.0
    5890              8283750.4
    5891              4404019.2
    5892             11534336.0
    5893              6501171.2
    5894              3879731.2
    5895              6291456.0
    5896              9227468.8
    5897             23068672.0
    5898             31457280.0
    5899             11534336.0
    5900              4299161.6
    5901              3565158.4
    5902             13631488.0
    5903             40894464.0
    5904              5347737.6
    5905              4928307.2
    5906              5767168.0
    5907              5557452.8
    5908              2936012.8
    5909              9437184.0
    5910             53477376.0
    5911               210944.0
    5912               976896.0
    5913             27262976.0
    5914              9856614.4
    5915              2516582.4
    5916              6081740.8
    5917             20971520.0
    5918             17825792.0
    5919             36700160.0
    5920             17825792.0
    5921              3040870.4
    5922             44040192.0
    5923              5347737.6
    5924             16777216.0
    5925              4928307.2
    5926             36700160.0
    5927             12582912.0
    5928             27262976.0
    5929             17825792.0
    5930             62914560.0
    5931             26214400.0
    5932             39845888.0
    5933              2097152.0
    5934              2621440.0
    5935              9646899.2
    5936               454656.0
    5937             15728640.0
    5938             12582912.0
    5939              4613734.4
    5940             14680064.0
    5941              7025459.2
    5942              3879731.2
    5943              4194304.0
    5944              2306867.2
    5945             34603008.0
    5946     Varies with device
    5947              5976883.2
    5948             91226112.0
    5949             31457280.0
    5950             71303168.0
    5951              6186598.4
    5952     Varies with device
    5953             19922944.0
    5954             41943040.0
    5955             51380224.0
    5956             54525952.0
    5957             66060288.0
    5958             25165824.0
    5959              6291456.0
    5960             17825792.0
    5961              4404019.2
    5962             33554432.0
    5963               734208.0
    5964     Varies with device
    5965              4508876.8
    5966              7549747.2
    5967              5242880.0
    5968              1258291.2
    5969             32505856.0
    5970             17825792.0
    5971             48234496.0
    5972             20971520.0
    5973              8703180.8
    5974              5242880.0
    5975             23068672.0
    5976              3250585.6
    5977              2306867.2
    5978              2097152.0
    5979             12582912.0
    5980              5767168.0
    5981             14680064.0
    5982     Varies with device
    5983             15728640.0
    5984             84934656.0
    5985             65011712.0
    5986              9332326.4
    5987     Varies with device
    5988              3145728.0
    5989             13631488.0
    5990             11534336.0
    5991              3565158.4
    5992             12582912.0
    5993             30408704.0
    5994     Varies with device
    5995              2726297.6
    5996              1782579.2
    5997              5662310.4
    5998     Varies with device
    5999             31457280.0
    6000             32505856.0
    6001             14680064.0
    6002             38797312.0
    6003              4194304.0
    6004     Varies with device
    6005              7235174.4
    6006             14680064.0
    6007               215040.0
    6008             13631488.0
    6009             18874368.0
    6010             13631488.0
    6011             39845888.0
    6012              3984588.8
    6013              3355443.2
    6014              3355443.2
    6015               623616.0
    6016              1782579.2
    6017              4194304.0
    6018              6815744.0
    6019             14680064.0
    6020              2936012.8
    6021             34603008.0
    6022              3460300.8
    6023              4089446.4
    6024             15728640.0
    6025              9961472.0
    6026              7654604.8
    6027              2516582.4
    6028              3145728.0
    6029              3670016.0
    6030              2516582.4
    6031             13631488.0
    6032     Varies with device
    6033              5557452.8
    6034              7235174.4
    6035              2411724.8
    6036             15728640.0
    6037              7549747.2
    6038             13631488.0
    6039              7654604.8
    6040              3460300.8
    6041              7654604.8
    6042              7654604.8
    6043              2726297.6
    6044               315392.0
    6045              4718592.0
    6046             22020096.0
    6047              5557452.8
    6048              2411724.8
    6049             23068672.0
    6050             41943040.0
    6051     Varies with device
    6052             22020096.0
    6053             22020096.0
    6054             55574528.0
    6055              1258291.2
    6056             55574528.0
    6057             11534336.0
    6058             28311552.0
    6059              7130316.8
    6060              7759462.4
    6061              3670016.0
    6062              1782579.2
    6063              4194304.0
    6064              9227468.8
    6065             39845888.0
    6066              1782579.2
    6067     Varies with device
    6068     Varies with device
    6069              6396313.6
    6070             28311552.0
    6071             39845888.0
    6072              2936012.8
    6073              6815744.0
    6074              3774873.6
    6075             55574528.0
    6076              4089446.4
    6077     Varies with device
    6078             12582912.0
    6079             93323264.0
    6080             48234496.0
    6081             56623104.0
    6082              9646899.2
    6083             25165824.0
    6084             42991616.0
    6085               721920.0
    6086             26214400.0
    6087             65011712.0
    6088             28311552.0
    6089             38797312.0
    6090              3879731.2
    6091              4823449.6
    6092             10380902.4
    6093              9437184.0
    6094             30408704.0
    6095              2621440.0
    6096              6606028.8
    6097             60817408.0
    6098              4299161.6
    6099             10066329.6
    6100              3984588.8
    6101             13631488.0
    6102             53477376.0
    6103              2831155.2
    6104             85983232.0
    6105              5242880.0
    6106             87031808.0
    6107              2621440.0
    6108              2621440.0
    6109              6920601.6
    6110              2621440.0
    6111              3879731.2
    6112              4823449.6
    6113             83886080.0
    6114             11534336.0
    6115              2726297.6
    6116              2621440.0
    6117             25165824.0
    6118             12582912.0
    6119             10485760.0
    6120     Varies with device
    6121               313344.0
    6122             27262976.0
    6123              2936012.8
    6124              8493465.6
    6125              3355443.2
    6126              3460300.8
    6127              1677721.6
    6128              5872025.6
    6129              4718592.0
    6130             11534336.0
    6131              2936012.8
    6132              2936012.8
    6133              2936012.8
    6134             17825792.0
    6135              3040870.4
    6136             19922944.0
    6137              4299161.6
    6138              8808038.4
    6139              5242880.0
    6140              4089446.4
    6141              2936012.8
    6142              2726297.6
    6143              6396313.6
    6144              2726297.6
    6145              9332326.4
    6146              9542041.6
    6147              3040870.4
    6148              2097152.0
    6149              2936012.8
    6150              2936012.8
    6151              9856614.4
    6152              8178892.8
    6153              9017753.6
    6154     Varies with device
    6155             10485760.0
    6156     Varies with device
    6157              5347737.6
    6158              8178892.8
    6159              6186598.4
    6160              1782579.2
    6161     Varies with device
    6162              2411724.8
    6163              2621440.0
    6164              2936012.8
    6165              4299161.6
    6166              6501171.2
    6167             20971520.0
    6168              5033164.8
    6169     Varies with device
    6170             25165824.0
    6171              2516582.4
    6172             13631488.0
    6173     Varies with device
    6174              1887436.8
    6175             10276044.8
    6176             30408704.0
    6177             11534336.0
    6178              6710886.4
    6179             91226112.0
    6180              8178892.8
    6181             90177536.0
    6182              2202009.6
    6183              4194304.0
    6184              1992294.4
    6185              2516582.4
    6186              3040870.4
    6187             22020096.0
    6188              9122611.2
    6189              1782579.2
    6190              5347737.6
    6191             11534336.0
    6192              1363148.8
    6193              9646899.2
    6194             36700160.0
    6195              9646899.2
    6196              5138022.4
    6197             15728640.0
    6198             15728640.0
    6199             19922944.0
    6200              2306867.2
    6201              2202009.6
    6202              8388608.0
    6203     Varies with device
    6204              3670016.0
    6205             16777216.0
    6206             66060288.0
    6207     Varies with device
    6208             10485760.0
    6209              7864320.0
    6210             22020096.0
    6211              1887436.8
    6212             11534336.0
    6213             18874368.0
    6214     Varies with device
    6215     Varies with device
    6216              3355443.2
    6217     Varies with device
    6218             34603008.0
    6219              8808038.4
    6220             32505856.0
    6221             23068672.0
    6222              3774873.6
    6223             15728640.0
    6224             18874368.0
    6225             10485760.0
    6226              2621440.0
    6227             16777216.0
    6228             15728640.0
    6229              4508876.8
    6230              4823449.6
    6231             18874368.0
    6232              9122611.2
    6233             23068672.0
    6234              1153433.6
    6235              5033164.8
    6236              2726297.6
    6237              5452595.2
    6238              6291456.0
    6239             45088768.0
    6240               925696.0
    6241             20971520.0
    6242             29360128.0
    6243             20971520.0
    6244             16777216.0
    6245     Varies with device
    6246              9122611.2
    6247              6291456.0
    6248              1887436.8
    6249              3879731.2
    6250              3355443.2
    6251             16777216.0
    6252              7759462.4
    6253              3460300.8
    6254              3670016.0
    6255             14680064.0
    6256              9332326.4
    6257             33554432.0
    6258             14680064.0
    6259              3984588.8
    6260              4928307.2
    6261             10380902.4
    6262             17825792.0
    6263              9542041.6
    6264             10485760.0
    6265             28311552.0
    6266             26214400.0
    6267             34603008.0
    6268             31457280.0
    6269     Varies with device
    6270             29360128.0
    6271              7654604.8
    6272             24117248.0
    6273               205824.0
    6274     Varies with device
    6275              2831155.2
    6276              8074035.2
    6277     Varies with device
    6278             22020096.0
    6279              3670016.0
    6280              3460300.8
    6281             32505856.0
    6282             12582912.0
    6283              8493465.6
    6284             14680064.0
    6285             11534336.0
    6286     Varies with device
    6287             18874368.0
    6288     Varies with device
    6289     Varies with device
    6290     Varies with device
    6291              3040870.4
    6292               484352.0
    6293             79691776.0
    6294             18874368.0
    6295             56623104.0
    6296             31457280.0
    6297     Varies with device
    6298     Varies with device
    6299              2202009.6
    6300             50331648.0
    6301             96468992.0
    6302             15728640.0
    6303             31457280.0
    6304     Varies with device
    6305              2726297.6
    6306             17825792.0
    6307             36700160.0
    6308             11534336.0
    6309             68157440.0
    6310             26214400.0
    6311              5033164.8
    6312              5033164.8
    6313             17825792.0
    6314             25165824.0
    6315             27262976.0
    6316              5033164.8
    6317               179200.0
    6318             15728640.0
    6319              5138022.4
    6320              9227468.8
    6321             37748736.0
    6322              9122611.2
    6323             24117248.0
    6324              5347737.6
    6325             24117248.0
    6326             27262976.0
    6327             55574528.0
    6328              3355443.2
    6329             24117248.0
    6330              1887436.8
    6331              5662310.4
    6332              8388608.0
    6333              5662310.4
    6334             61865984.0
    6335             22020096.0
    6336              5347737.6
    6337     Varies with device
    6338             57671680.0
    6339              3565158.4
    6340             34603008.0
    6341              4928307.2
    6342              1572864.0
    6343     Varies with device
    6344             31457280.0
    6345             27262976.0
    6346             56623104.0
    6347             10485760.0
    6348              3774873.6
    6349              5452595.2
    6350             26214400.0
    6351               358400.0
    6352             53477376.0
    6353              5347737.6
    6354              4823449.6
    6355             40894464.0
    6356             13631488.0
    6357             32505856.0
    6358             19922944.0
    6359              3250585.6
    6360             17825792.0
    6361               392192.0
    6362              2516582.4
    6363              3355443.2
    6364              2202009.6
    6365             31457280.0
    6366             14680064.0
    6367             12582912.0
    6368              3145728.0
    6369              5033164.8
    6370              2936012.8
    6371              4194304.0
    6372             11534336.0
    6373     Varies with device
    6374              1887436.8
    6375              3250585.6
    6376             20971520.0
    6377             34603008.0
    6378             17825792.0
    6379             29360128.0
    6380               464896.0
    6381              3250585.6
    6382              3355443.2
    6383              7130316.8
    6384            103809024.0
    6385             17825792.0
    6386             12582912.0
    6387             26214400.0
    6388             13631488.0
    6389              2202009.6
    6390              6186598.4
    6391              3879731.2
    6392              4718592.0
    6393              4299161.6
    6394             11534336.0
    6395              4089446.4
    6396              3145728.0
    6397              1468006.4
    6398             11534336.0
    6399             12582912.0
    6400             31457280.0
    6401             26214400.0
    6402             54525952.0
    6403              6606028.8
    6404              4928307.2
    6405             26214400.0
    6406             65011712.0
    6407              6710886.4
    6408             15728640.0
    6409             73400320.0
    6410             12582912.0
    6411              7444889.6
    6412             49283072.0
    6413             24117248.0
    6414             10485760.0
    6415             41943040.0
    6416             55574528.0
    6417              8598323.2
    6418              8912896.0
    6419             34603008.0
    6420             35651584.0
    6421             24117248.0
    6422              2621440.0
    6423              1887436.8
    6424     Varies with device
    6425              1468006.4
    6426             25165824.0
    6427              2097152.0
    6428               431104.0
    6429              1258291.2
    6430             11534336.0
    6431              7340032.0
    6432              2202009.6
    6433             16777216.0
    6434             10485760.0
    6435             29360128.0
    6436              5767168.0
    6437             42991616.0
    6438              2831155.2
    6439             97517568.0
    6440             41943040.0
    6441              3565158.4
    6442              9437184.0
    6443              4508876.8
    6444              3040870.4
    6445              7235174.4
    6446              1048576.0
    6447              1153433.6
    6448             40894464.0
    6449     Varies with device
    6450     Varies with device
    6451             32505856.0
    6452             14680064.0
    6453             30408704.0
    6454              4404019.2
    6455              5557452.8
    6456              1782579.2
    6457              6606028.8
    6458             28311552.0
    6459              3250585.6
    6460     Varies with device
    6461              2621440.0
    6462              5242880.0
    6463              6606028.8
    6464              8178892.8
    6465              5033164.8
    6466     Varies with device
    6467              9646899.2
    6468             35651584.0
    6469             49283072.0
    6470              8493465.6
    6471              2621440.0
    6472             37748736.0
    6473             10485760.0
    6474              6501171.2
    6475             10485760.0
    6476              4613734.4
    6477             18874368.0
    6478             16777216.0
    6479     Varies with device
    6480             10380902.4
    6481             23068672.0
    6482              1992294.4
    6483             27262976.0
    6484              9751756.8
    6485     Varies with device
    6486     Varies with device
    6487             29360128.0
    6488             10485760.0
    6489             11534336.0
    6490              2411724.8
    6491              4613734.4
    6492              8388608.0
    6493             10485760.0
    6494              3879731.2
    6495             22020096.0
    6496              4823449.6
    6497     Varies with device
    6498             47185920.0
    6499              4404019.2
    6500              7969177.6
    6501              9017753.6
    6502             15728640.0
    6503                71680.0
    6504             17825792.0
    6505               831488.0
    6506             13631488.0
    6507               452608.0
    6508              2097152.0
    6509     Varies with device
    6510               862208.0
    6511              2516582.4
    6512               427008.0
    6513              5872025.6
    6514               421888.0
    6515               470016.0
    6516             35651584.0
    6517               489472.0
    6518             10485760.0
    6519              6501171.2
    6520               343040.0
    6521              3984588.8
    6522               800768.0
    6523               738304.0
    6524               440320.0
    6525               439296.0
    6526               196608.0
    6527               204800.0
    6528             13631488.0
    6529              2621440.0
    6530              9332326.4
    6531               427008.0
    6532     Varies with device
    6533               471040.0
    6534              6186598.4
    6535               745472.0
    6536               507904.0
    6537               835584.0
    6538              6606028.8
    6539               423936.0
    6540               342016.0
    6541               518144.0
    6542              2097152.0
    6543              2411724.8
    6544             35651584.0
    6545              7864320.0
    6546     Varies with device
    6547             62914560.0
    6548             62914560.0
    6549              8283750.4
    6550             25165824.0
    6551             99614720.0
    6552             61865984.0
    6553             24117248.0
    6554             12582912.0
    6555             11534336.0
    6556             14680064.0
    6557              2306867.2
    6558             31457280.0
    6559              8912896.0
    6560             19922944.0
    6561             17825792.0
    6562             33554432.0
    6563             34603008.0
    6564              4404019.2
    6565             34603008.0
    6566            100663296.0
    6567              4928307.2
    6568             27262976.0
    6569             24117248.0
    6570             31457280.0
    6571              6710886.4
    6572             29360128.0
    6573             51380224.0
    6574             16777216.0
    6575             53477376.0
    6576              2726297.6
    6577             26214400.0
    6578     Varies with device
    6579             28311552.0
    6580             24117248.0
    6581             32505856.0
    6582             26214400.0
    6583             16777216.0
    6584               720896.0
    6585              7759462.4
    6586              5242880.0
    6587              2726297.6
    6588             47185920.0
    6589             24117248.0
    6590              8808038.4
    6591              8703180.8
    6592              8283750.4
    6593             27262976.0
    6594              6291456.0
    6595              5033164.8
    6596              4823449.6
    6597             16777216.0
    6598              5976883.2
    6599              6186598.4
    6600               343040.0
    6601              2306867.2
    6602               908288.0
    6603               627712.0
    6604              1572864.0
    6605             12582912.0
    6606             27262976.0
    6607             13631488.0
    6608     Varies with device
    6609              3460300.8
    6610              3040870.4
    6611             10485760.0
    6612              3355443.2
    6613              2621440.0
    6614              1887436.8
    6615              2831155.2
    6616              7340032.0
    6617             12582912.0
    6618     Varies with device
    6619             10066329.6
    6620             18874368.0
    6621              1992294.4
    6622             30408704.0
    6623              9856614.4
    6624              7025459.2
    6625              1677721.6
    6626              2097152.0
    6627     Varies with device
    6628              2621440.0
    6629              7549747.2
    6630              6606028.8
    6631              4194304.0
    6632              1258291.2
    6633     Varies with device
    6634              5872025.6
    6635              3670016.0
    6636              9646899.2
    6637              1258291.2
    6638     Varies with device
    6639             18874368.0
    6640             37748736.0
    6641              1258291.2
    6642              9227468.8
    6643              4299161.6
    6644              2621440.0
    6645              2831155.2
    6646             34603008.0
    6647             29360128.0
    6648              7025459.2
    6649             22020096.0
    6650              3670016.0
    6651              6291456.0
    6652     Varies with device
    6653             23068672.0
    6654     Varies with device
    6655             28311552.0
    6656              3565158.4
    6657     Varies with device
    6658             16777216.0
    6659             12582912.0
    6660             10485760.0
    6661     Varies with device
    6662              9437184.0
    6663             37748736.0
    6664             12582912.0
    6665              3774873.6
    6666             31457280.0
    6667             19922944.0
    6668             13631488.0
    6669             34603008.0
    6670             14680064.0
    6671               248832.0
    6672             25165824.0
    6673     Varies with device
    6674              6920601.6
    6675             46137344.0
    6676             93323264.0
    6677              3670016.0
    6678             33554432.0
    6679             52428800.0
    6680             15728640.0
    6681              3250585.6
    6682             31457280.0
    6683             35651584.0
    6684             13631488.0
    6685              2726297.6
    6686              2621440.0
    6687     Varies with device
    6688              7864320.0
    6689               582656.0
    6690             27262976.0
    6691              5452595.2
    6692              5662310.4
    6693              9542041.6
    6694             27262976.0
    6695             29360128.0
    6696              3984588.8
    6697     Varies with device
    6698             96468992.0
    6699               210944.0
    6700             19922944.0
    6701             97517568.0
    6702             13631488.0
    6703              1887436.8
    6704             10485760.0
    6705              3670016.0
    6706              7864320.0
    6707             71303168.0
    6708               796672.0
    6709              2202009.6
    6710              9646899.2
    6711              3145728.0
    6712             75497472.0
    6713             36700160.0
    6714     Varies with device
    6715             32505856.0
    6716              4299161.6
    6717     Varies with device
    6718             13631488.0
    6719             62914560.0
    6720             89128960.0
    6721              1363148.8
    6722     Varies with device
    6723              7235174.4
    6724              5767168.0
    6725             60817408.0
    6726             98566144.0
    6727               699392.0
    6728               606208.0
    6729     Varies with device
    6730              6710886.4
    6731              3460300.8
    6732               326656.0
    6733             25165824.0
    6734             37748736.0
    6735               190464.0
    6736              3670016.0
    6737               860160.0
    6738             42991616.0
    6739              1572864.0
    6740             37748736.0
    6741               662528.0
    6742              5557452.8
    6743             19922944.0
    6744              2621440.0
    6745             97517568.0
    6746              2936012.8
    6747              5872025.6
    6748             75497472.0
    6749             22020096.0
    6750               195584.0
    6751             23068672.0
    6752             70254592.0
    6753               862208.0
    6754              1153433.6
    6755              7235174.4
    6756     Varies with device
    6757             15728640.0
    6758              1153433.6
    6759              6501171.2
    6760              1468006.4
    6761             54525952.0
    6762              9122611.2
    6763               381952.0
    6764              3984588.8
    6765              3984588.8
    6766              3984588.8
    6767              8598323.2
    6768               447488.0
    6769              5976883.2
    6770              1677721.6
    6771             16777216.0
    6772              2726297.6
    6773              6920601.6
    6774               612352.0
    6775              2831155.2
    6776     Varies with device
    6777               733184.0
    6778             14680064.0
    6779             12582912.0
    6780             48234496.0
    6781             13631488.0
    6782              1992294.4
    6783              1258291.2
    6784              4508876.8
    6785             13631488.0
    6786              8808038.4
    6787              7654604.8
    6788               599040.0
    6789              3355443.2
    6790              6815744.0
    6791              2831155.2
    6792              1468006.4
    6793             17825792.0
    6794              3250585.6
    6795              2516582.4
    6796              1677721.6
    6797     Varies with device
    6798              1005568.0
    6799             27262976.0
    6800             19922944.0
    6801              3040870.4
    6802              4718592.0
    6803               227328.0
    6804             44040192.0
    6805              8283750.4
    6806               224256.0
    6807                56320.0
    6808               970752.0
    6809              7654604.8
    6810               330752.0
    6811             93323264.0
    6812              5452595.2
    6813             15728640.0
    6814             19922944.0
    6815              7969177.6
    6816              5872025.6
    6817             18874368.0
    6818             25165824.0
    6819             10485760.0
    6820             11534336.0
    6821             30408704.0
    6822              4508876.8
    6823             10485760.0
    6824             92274688.0
    6825              5557452.8
    6826              5662310.4
    6827              3250585.6
    6828              6291456.0
    6829             27262976.0
    6830              6815744.0
    6831             19922944.0
    6832             34603008.0
    6833             41943040.0
    6834              9646899.2
    6835             29360128.0
    6836             22020096.0
    6837             20971520.0
    6838              4194304.0
    6839             27262976.0
    6840     Varies with device
    6841             27262976.0
    6842             34603008.0
    6843             26214400.0
    6844             46137344.0
    6845     Varies with device
    6846             19922944.0
    6847             28311552.0
    6848              1572864.0
    6849     Varies with device
    6850             13631488.0
    6851              5033164.8
    6852              1677721.6
    6853              5138022.4
    6854              3670016.0
    6855              9122611.2
    6856              8074035.2
    6857             19922944.0
    6858              3565158.4
    6859              3040870.4
    6860              2516582.4
    6861              2831155.2
    6862              5242880.0
    6863              6186598.4
    6864              2936012.8
    6865              3565158.4
    6866             12582912.0
    6867             20971520.0
    6868              4404019.2
    6869             10485760.0
    6870             27262976.0
    6871              4194304.0
    6872              3774873.6
    6873            102760448.0
    6874              5976883.2
    6875              3774873.6
    6876            100663296.0
    6877              2306867.2
    6878             10485760.0
    6879              5033164.8
    6880              2097152.0
    6881             65011712.0
    6882     Varies with device
    6883             10276044.8
    6884             35651584.0
    6885             55574528.0
    6886              7654604.8
    6887     Varies with device
    6888              2936012.8
    6889             25165824.0
    6890             76546048.0
    6891             55574528.0
    6892             28311552.0
    6893              4613734.4
    6894             17825792.0
    6895              1363148.8
    6896             19922944.0
    6897              9227468.8
    6898              1363148.8
    6899              1887436.8
    6900              5033164.8
    6901              1887436.8
    6902               707584.0
    6903             12582912.0
    6904             20971520.0
    6905              5347737.6
    6906              7549747.2
    6907             25165824.0
    6908              7759462.4
    6909             55574528.0
    6910              1363148.8
    6911             14680064.0
    6912              9227468.8
    6913              1782579.2
    6914             14680064.0
    6915             13631488.0
    6916             17825792.0
    6917             25165824.0
    6918             17825792.0
    6919             27262976.0
    6920               523264.0
    6921              1782579.2
    6922             33554432.0
    6923             12582912.0
    6924             24117248.0
    6925              5767168.0
    6926              9856614.4
    6927             18874368.0
    6928              1887436.8
    6929               973824.0
    6930              1992294.4
    6931              2726297.6
    6932             13631488.0
    6933             16777216.0
    6934               986112.0
    6935              8912896.0
    6936             24117248.0
    6937              8598323.2
    6938              3984588.8
    6939              2516582.4
    6940              1782579.2
    6941             10485760.0
    6942              2097152.0
    6943                25600.0
    6944              4823449.6
    6945             32505856.0
    6946              8598323.2
    6947               567296.0
    6948             22020096.0
    6949              7235174.4
    6950             22020096.0
    6951             11534336.0
    6952             35651584.0
    6953              4299161.6
    6954              1992294.4
    6955              3670016.0
    6956              3145728.0
    6957              4194304.0
    6958              2306867.2
    6959             18874368.0
    6960              3879731.2
    6961              2516582.4
    6962              9751756.8
    6963             46137344.0
    6964             32505856.0
    6965             50331648.0
    6966             39845888.0
    6967             51380224.0
    6968             28311552.0
    6969               359424.0
    6970              9227468.8
    6971             24117248.0
    6972             50331648.0
    6973             10171187.2
    6974                27648.0
    6975             12582912.0
    6976             77594624.0
    6977             46137344.0
    6978             37748736.0
    6979              1468006.4
    6980     Varies with device
    6981             26214400.0
    6982     Varies with device
    6983             46137344.0
    6984             37748736.0
    6985                83968.0
    6986             59768832.0
    6987     Varies with device
    6988             10276044.8
    6989             25165824.0
    6990             49283072.0
    6991             38797312.0
    6992              7235174.4
    6993             10485760.0
    6994             15728640.0
    6995              9856614.4
    6996             51380224.0
    6997             25165824.0
    6998              7654604.8
    6999             25165824.0
    7000             14680064.0
    7001     Varies with device
    7002     Varies with device
    7003             22020096.0
    7004              9227468.8
    7005              4194304.0
    7006     Varies with device
    7007             18874368.0
    7008             18874368.0
    7009             13631488.0
    7010             13631488.0
    7011             25165824.0
    7012     Varies with device
    7013             26214400.0
    7014             45088768.0
    7015              4823449.6
    7016             49283072.0
    7017             20971520.0
    7018             27262976.0
    7019             72351744.0
    7020             13631488.0
    7021              9646899.2
    7022             19922944.0
    7023             35651584.0
    7024     Varies with device
    7025             13631488.0
    7026             35651584.0
    7027             25165824.0
    7028              3984588.8
    7029              8703180.8
    7030             26214400.0
    7031             55574528.0
    7032             25165824.0
    7033              5662310.4
    7034              8074035.2
    7035             11534336.0
    7036              5872025.6
    7037              5662310.4
    7038             17825792.0
    7039              4613734.4
    7040     Varies with device
    7041              8598323.2
    7042            102760448.0
    7043             30408704.0
    7044              2516582.4
    7045              8912896.0
    7046              3355443.2
    7047              3984588.8
    7048             34603008.0
    7049     Varies with device
    7050             55574528.0
    7051              6501171.2
    7052             23068672.0
    7053             17825792.0
    7054             26214400.0
    7055             40894464.0
    7056              4823449.6
    7057             13631488.0
    7058             37748736.0
    7059             13631488.0
    7060              8912896.0
    7061             26214400.0
    7062             32505856.0
    7063              3879731.2
    7064              8912896.0
    7065             25165824.0
    7066             35651584.0
    7067              7235174.4
    7068             47185920.0
    7069             12582912.0
    7070             34603008.0
    7071             36700160.0
    7072             26214400.0
    7073             50331648.0
    7074              1048576.0
    7075              7549747.2
    7076             14680064.0
    7077     Varies with device
    7078              9332326.4
    7079     Varies with device
    7080               212992.0
    7081              9856614.4
    7082              8703180.8
    7083              7235174.4
    7084             12582912.0
    7085              5767168.0
    7086             27262976.0
    7087            102760448.0
    7088              5452595.2
    7089              8074035.2
    7090              5557452.8
    7091               384000.0
    7092              3879731.2
    7093              3040870.4
    7094             18874368.0
    7095              6291456.0
    7096             35651584.0
    7097              7340032.0
    7098             39845888.0
    7099              5662310.4
    7100              1258291.2
    7101              1258291.2
    7102              8283750.4
    7103              2411724.8
    7104              2411724.8
    7105               330752.0
    7106     Varies with device
    7107              1468006.4
    7108              2621440.0
    7109             12582912.0
    7110             15728640.0
    7111              3250585.6
    7112             12582912.0
    7113              1572864.0
    7114              9437184.0
    7115              3879731.2
    7116              9122611.2
    7117             19922944.0
    7118              7759462.4
    7119              3460300.8
    7120               934912.0
    7121              5138022.4
    7122              8178892.8
    7123              5242880.0
    7124             12582912.0
    7125             11534336.0
    7126             26214400.0
    7127              2726297.6
    7128              7549747.2
    7129              1887436.8
    7130              2726297.6
    7131             14680064.0
    7132              2306867.2
    7133              1992294.4
    7134              5872025.6
    7135               526336.0
    7136             20971520.0
    7137             11534336.0
    7138             12582912.0
    7139              5872025.6
    7140              3984588.8
    7141              6710886.4
    7142              3879731.2
    7143              3565158.4
    7144              8808038.4
    7145              3040870.4
    7146              3040870.4
    7147              1887436.8
    7148              3250585.6
    7149             83886080.0
    7150              3460300.8
    7151             15728640.0
    7152             37748736.0
    7153     Varies with device
    7154              8703180.8
    7155              3250585.6
    7156              9961472.0
    7157              5033164.8
    7158              7025459.2
    7159     Varies with device
    7160              1572864.0
    7161              9646899.2
    7162              3565158.4
    7163             13631488.0
    7164              6081740.8
    7165             20971520.0
    7166             20971520.0
    7167             66060288.0
    7168             13631488.0
    7169             28311552.0
    7170              3984588.8
    7171             16777216.0
    7172              2411724.8
    7173             12582912.0
    7174              5872025.6
    7175             38797312.0
    7176             23068672.0
    7177     Varies with device
    7178              1782579.2
    7179              2726297.6
    7180              8283750.4
    7181             20971520.0
    7182              5976883.2
    7183             16777216.0
    7184              1782579.2
    7185              4404019.2
    7186              4823449.6
    7187              2831155.2
    7188              7549747.2
    7189             48234496.0
    7190             14680064.0
    7191             12582912.0
    7192              3879731.2
    7193              1677721.6
    7194             12582912.0
    7195             30408704.0
    7196             37748736.0
    7197              4194304.0
    7198             15728640.0
    7199             44040192.0
    7200              8178892.8
    7201              4823449.6
    7202              4508876.8
    7203              2936012.8
    7204             28311552.0
    7205              7654604.8
    7206             24117248.0
    7207             20971520.0
    7208              5662310.4
    7209             25165824.0
    7210             20971520.0
    7211              1468006.4
    7212              3774873.6
    7213              6606028.8
    7214             22020096.0
    7215             14680064.0
    7216             26214400.0
    7217             16777216.0
    7218             57671680.0
    7219             23068672.0
    7220              3145728.0
    7221              2726297.6
    7222             25165824.0
    7223               564224.0
    7224              4194304.0
    7225              5347737.6
    7226             23068672.0
    7227             22020096.0
    7228             18874368.0
    7229             65011712.0
    7230             31457280.0
    7231             24117248.0
    7232     Varies with device
    7233     Varies with device
    7234             10485760.0
    7235             30408704.0
    7236              1992294.4
    7237                29696.0
    7238             45088768.0
    7239              8703180.8
    7240             19922944.0
    7241               105472.0
    7242              6501171.2
    7243              8703180.8
    7244              5767168.0
    7245             11534336.0
    7246             17825792.0
    7247              2516582.4
    7248              4823449.6
    7249              3145728.0
    7250             22020096.0
    7251              2831155.2
    7252             15728640.0
    7253              4928307.2
    7254              3145728.0
    7255              1468006.4
    7256             13631488.0
    7257             16777216.0
    7258              2097152.0
    7259             11534336.0
    7260              3040870.4
    7261             41943040.0
    7262             14680064.0
    7263             10171187.2
    7264              8178892.8
    7265              3355443.2
    7266              8388608.0
    7267             30408704.0
    7268              3250585.6
    7269              3670016.0
    7270             14680064.0
    7271              9122611.2
    7272              9227468.8
    7273             10171187.2
    7274              3250585.6
    7275              3355443.2
    7276              8912896.0
    7277              3670016.0
    7278              8178892.8
    7279             11534336.0
    7280              6606028.8
    7281             19922944.0
    7282              5242880.0
    7283             16777216.0
    7284              3670016.0
    7285              2411724.8
    7286              3460300.8
    7287              4299161.6
    7288              4299161.6
    7289              4299161.6
    7290              7340032.0
    7291              2411724.8
    7292              4299161.6
    7293              1153433.6
    7294             18874368.0
    7295              4299161.6
    7296              1572864.0
    7297              2411724.8
    7298              3670016.0
    7299              3879731.2
    7300     Varies with device
    7301              4194304.0
    7302              1887436.8
    7303              4194304.0
    7304              3984588.8
    7305             10485760.0
    7306              1363148.8
    7307              6920601.6
    7308              4299161.6
    7309              2621440.0
    7310              4299161.6
    7311     Varies with device
    7312              2621440.0
    7313              2621440.0
    7314              4508876.8
    7315              8074035.2
    7316              7130316.8
    7317              3565158.4
    7318              1468006.4
    7319              3774873.6
    7320              2411724.8
    7321             14680064.0
    7322            103809024.0
    7323              4299161.6
    7324              9122611.2
    7325              2621440.0
    7326              3040870.4
    7327              4508876.8
    7328              6396313.6
    7329     Varies with device
    7330     Varies with device
    7331     Varies with device
    7332     Varies with device
    7333             14680064.0
    7334             15728640.0
    7335              6606028.8
    7336             69206016.0
    7337              3565158.4
    7338     Varies with device
    7339             79691776.0
    7340              8074035.2
    7341              6815744.0
    7342             14680064.0
    7343             26214400.0
    7344     Varies with device
    7345     Varies with device
    7346              1468006.4
    7347             52428800.0
    7348              2621440.0
    7349              2306867.2
    7350              4823449.6
    7351             15728640.0
    7352             13631488.0
    7353     Varies with device
    7354              8493465.6
    7355     Varies with device
    7356     Varies with device
    7357             16777216.0
    7358             10066329.6
    7359     Varies with device
    7360     Varies with device
    7361     Varies with device
    7362             10485760.0
    7363              3250585.6
    7364             81788928.0
    7365             28311552.0
    7366     Varies with device
    7367             10485760.0
    7368              6081740.8
    7369              3355443.2
    7370               919552.0
    7371              3879731.2
    7372     Varies with device
    7373             59768832.0
    7374             44040192.0
    7375             37748736.0
    7376              1258291.2
    7377             48234496.0
    7378             31457280.0
    7379              3355443.2
    7380             10485760.0
    7381              1887436.8
    7382             32505856.0
    7383              6920601.6
    7384              5347737.6
    7385              3355443.2
    7386              6291456.0
    7387              2621440.0
    7388             10380902.4
    7389             10485760.0
    7390               176128.0
    7391              5452595.2
    7392              5347737.6
    7393              6710886.4
    7394             14680064.0
    7395             15728640.0
    7396              3460300.8
    7397              7235174.4
    7398              5347737.6
    7399             20971520.0
    7400             10485760.0
    7401              6920601.6
    7402              6396313.6
    7403              3355443.2
    7404            104857600.0
    7405              1468006.4
    7406              2202009.6
    7407              7549747.2
    7408             95420416.0
    7409              2936012.8
    7410               760832.0
    7411             14680064.0
    7412             14680064.0
    7413             45088768.0
    7414              7235174.4
    7415            100663296.0
    7416             14680064.0
    7417             27262976.0
    7418             18874368.0
    7419             35651584.0
    7420              2936012.8
    7421             96468992.0
    7422             11534336.0
    7423             15728640.0
    7424              4194304.0
    7425              3984588.8
    7426              1363148.8
    7427             13631488.0
    7428             70254592.0
    7429            103809024.0
    7430               118784.0
    7431              2306867.2
    7432             25165824.0
    7433             45088768.0
    7434              6186598.4
    7435              2516582.4
    7436             70254592.0
    7437             70254592.0
    7438             13631488.0
    7439              8598323.2
    7440              5452595.2
    7441     Varies with device
    7442              9646899.2
    7443             66060288.0
    7444              4404019.2
    7445             16777216.0
    7446             10485760.0
    7447             18874368.0
    7448             12582912.0
    7449             81788928.0
    7450             39845888.0
    7451              9961472.0
    7452     Varies with device
    7453             47185920.0
    7454             29360128.0
    7455     Varies with device
    7456              8074035.2
    7457              9961472.0
    7458              5452595.2
    7459             39845888.0
    7460              3879731.2
    7461             26214400.0
    7462             20971520.0
    7463               156672.0
    7464              5557452.8
    7465              6606028.8
    7466             39845888.0
    7467             13631488.0
    7468              6710886.4
    7469              1992294.4
    7470              6710886.4
    7471              2097152.0
    7472              9017753.6
    7473             25165824.0
    7474             10485760.0
    7475             40894464.0
    7476             62914560.0
    7477             41943040.0
    7478             15728640.0
    7479               214016.0
    7480              1572864.0
    7481              2411724.8
    7482             13631488.0
    7483              9856614.4
    7484              3879731.2
    7485              4404019.2
    7486             23068672.0
    7487             13631488.0
    7488              8493465.6
    7489             12582912.0
    7490             19922944.0
    7491     Varies with device
    7492              5138022.4
    7493             10171187.2
    7494              3774873.6
    7495              2621440.0
    7496              2621440.0
    7497             18874368.0
    7498             36700160.0
    7499              6606028.8
    7500              1363148.8
    7501             25165824.0
    7502              3565158.4
    7503              3565158.4
    7504              3565158.4
    7505              3565158.4
    7506             17825792.0
    7507              2306867.2
    7508     Varies with device
    7509              5242880.0
    7510              4823449.6
    7511              9437184.0
    7512              1363148.8
    7513             25165824.0
    7514              3355443.2
    7515            101711872.0
    7516               361472.0
    7517              3355443.2
    7518              2516582.4
    7519              8808038.4
    7520             12582912.0
    7521              3565158.4
    7522              3565158.4
    7523              7654604.8
    7524              4404019.2
    7525             60817408.0
    7526              7130316.8
    7527               510976.0
    7528              3355443.2
    7529              3460300.8
    7530              3879731.2
    7531             26214400.0
    7532             12582912.0
    7533              1363148.8
    7534             41943040.0
    7535              5452595.2
    7536     Varies with device
    7537             17825792.0
    7538               177152.0
    7539     Varies with device
    7540              5347737.6
    7541              6081740.8
    7542              1258291.2
    7543              6081740.8
    7544              1887436.8
    7545              1992294.4
    7546              3774873.6
    7547              2621440.0
    7548              2202009.6
    7549              1153433.6
    7550             17825792.0
    7551              2202009.6
    7552              2936012.8
    7553     Varies with device
    7554              5767168.0
    7555             42991616.0
    7556             14680064.0
    7557               611328.0
    7558              2831155.2
    7559              3355443.2
    7560               828416.0
    7561              2306867.2
    7562              8598323.2
    7563              2936012.8
    7564              4194304.0
    7565              1887436.8
    7566                71680.0
    7567              2411724.8
    7568              1363148.8
    7569               124928.0
    7570              4299161.6
    7571             11534336.0
    7572              4404019.2
    7573              8598323.2
    7574               420864.0
    7575              2411724.8
    7576              5976883.2
    7577     Varies with device
    7578              2831155.2
    7579              1992294.4
    7580             19922944.0
    7581              2411724.8
    7582     Varies with device
    7583             51380224.0
    7584             29360128.0
    7585             42991616.0
    7586             26214400.0
    7587             26214400.0
    7588             39845888.0
    7589             29360128.0
    7590             57671680.0
    7591             60817408.0
    7592             28311552.0
    7593             26214400.0
    7594             32505856.0
    7595             98566144.0
    7596             26214400.0
    7597             22020096.0
    7598             47185920.0
    7599             47185920.0
    7600             25165824.0
    7601             87031808.0
    7602             92274688.0
    7603             53477376.0
    7604             66060288.0
    7605             22020096.0
    7606              7549747.2
    7607             53477376.0
    7608            102760448.0
    7609             28311552.0
    7610              9751756.8
    7611             91226112.0
    7612             57671680.0
    7613     Varies with device
    7614             27262976.0
    7615     Varies with device
    7616             65011712.0
    7617             23068672.0
    7618             41943040.0
    7619              5767168.0
    7620             18874368.0
    7621             77594624.0
    7622             16777216.0
    7623             67108864.0
    7624             16777216.0
    7625             26214400.0
    7626             12582912.0
    7627             18874368.0
    7628              4718592.0
    7629             80740352.0
    7630             89128960.0
    7631             89128960.0
    7632             27262976.0
    7633              2097152.0
    7634              4089446.4
    7635             11534336.0
    7636             18874368.0
    7637              9122611.2
    7638              4089446.4
    7639              4194304.0
    7640     Varies with device
    7641             55574528.0
    7642              5138022.4
    7643             54525952.0
    7644             24117248.0
    7645             13631488.0
    7646             31457280.0
    7647             18874368.0
    7648              4299161.6
    7649             13631488.0
    7650             20971520.0
    7651             22020096.0
    7652             12582912.0
    7653             18874368.0
    7654             42991616.0
    7655             12582912.0
    7656             14680064.0
    7657     Varies with device
    7658     Varies with device
    7659              6291456.0
    7660             17825792.0
    7661               409600.0
    7662             44040192.0
    7663             56623104.0
    7664             11534336.0
    7665             38797312.0
    7666             13631488.0
    7667              6815744.0
    7668             16777216.0
    7669             19922944.0
    7670              3460300.8
    7671             36700160.0
    7672              6606028.8
    7673             12582912.0
    7674              5242880.0
    7675             45088768.0
    7676              3040870.4
    7677              4299161.6
    7678              8912896.0
    7679              7340032.0
    7680             19922944.0
    7681             16777216.0
    7682             14680064.0
    7683               820224.0
    7684              5872025.6
    7685             37748736.0
    7686              3355443.2
    7687              7340032.0
    7688              3040870.4
    7689             36700160.0
    7690             22020096.0
    7691              6081740.8
    7692              4928307.2
    7693             20971520.0
    7694              4089446.4
    7695             12582912.0
    7696              8178892.8
    7697             25165824.0
    7698             12582912.0
    7699     Varies with device
    7700              5138022.4
    7701              3460300.8
    7702              5662310.4
    7703              2516582.4
    7704             28311552.0
    7705             23068672.0
    7706              1258291.2
    7707              2621440.0
    7708             14680064.0
    7709              7130316.8
    7710              1258291.2
    7711               805888.0
    7712              5138022.4
    7713              5557452.8
    7714              6815744.0
    7715             72351744.0
    7716             14680064.0
    7717              3565158.4
    7718             52428800.0
    7719              8493465.6
    7720              1782579.2
    7721             18874368.0
    7722             18874368.0
    7723             37748736.0
    7724              8493465.6
    7725              5347737.6
    7726              1677721.6
    7727             63963136.0
    7728              4299161.6
    7729             10485760.0
    7730               242688.0
    7731              3565158.4
    7732             18874368.0
    7733              3670016.0
    7734              3565158.4
    7735              9542041.6
    7736             15728640.0
    7737             14680064.0
    7738              3984588.8
    7739             10485760.0
    7740             27262976.0
    7741     Varies with device
    7742              3460300.8
    7743             15728640.0
    7744             19922944.0
    7745              3879731.2
    7746     Varies with device
    7747              2202009.6
    7748             11534336.0
    7749             45088768.0
    7750              7235174.4
    7751             41943040.0
    7752     Varies with device
    7753             37748736.0
    7754              1572864.0
    7755             38797312.0
    7756             23068672.0
    7757              3460300.8
    7758              2726297.6
    7759             35651584.0
    7760              3040870.4
    7761             12582912.0
    7762              3879731.2
    7763             17825792.0
    7764             24117248.0
    7765             17825792.0
    7766              5242880.0
    7767             18874368.0
    7768             34603008.0
    7769             27262976.0
    7770             33554432.0
    7771             39845888.0
    7772             22020096.0
    7773             42991616.0
    7774              1782579.2
    7775              4508876.8
    7776             10380902.4
    7777              6186598.4
    7778              2831155.2
    7779             12582912.0
    7780             16777216.0
    7781             19922944.0
    7782              2936012.8
    7783              3774873.6
    7784             38797312.0
    7785              4508876.8
    7786              1782579.2
    7787              3250585.6
    7788             31457280.0
    7789              5452595.2
    7790              2936012.8
    7791             18874368.0
    7792              3565158.4
    7793             42991616.0
    7794             13631488.0
    7795              7025459.2
    7796             10276044.8
    7797             14680064.0
    7798             27262976.0
    7799             23068672.0
    7800              9122611.2
    7801             92274688.0
    7802             18874368.0
    7803             18874368.0
    7804              1363148.8
    7805              4718592.0
    7806              8178892.8
    7807             24117248.0
    7808     Varies with device
    7809     Varies with device
    7810                51200.0
    7811             73400320.0
    7812             49283072.0
    7813             44040192.0
    7814              3879731.2
    7815             40894464.0
    7816             23068672.0
    7817             23068672.0
    7818             12582912.0
    7819             14680064.0
    7820              4508876.8
    7821     Varies with device
    7822              8912896.0
    7823              7025459.2
    7824              7864320.0
    7825             92274688.0
    7826              1153433.6
    7827              8703180.8
    7828              5976883.2
    7829               658432.0
    7830             46137344.0
    7831             11534336.0
    7832             29360128.0
    7833             41943040.0
    7834              9542041.6
    7835              2202009.6
    7836             33554432.0
    7837              8178892.8
    7838              2516582.4
    7839             55574528.0
    7840              6291456.0
    7841             26214400.0
    7842              3460300.8
    7843             35651584.0
    7844              3774873.6
    7845              6501171.2
    7846             71303168.0
    7847             19922944.0
    7848              3774873.6
    7849             17825792.0
    7850             32505856.0
    7851             18874368.0
    7852              6291456.0
    7853             30408704.0
    7854             80740352.0
    7855             44040192.0
    7856             14680064.0
    7857             30408704.0
    7858              5976883.2
    7859             37748736.0
    7860              9332326.4
    7861              5347737.6
    7862             19922944.0
    7863              4508876.8
    7864             65011712.0
    7865             30408704.0
    7866              8178892.8
    7867             20971520.0
    7868              1009664.0
    7869             18874368.0
    7870             14680064.0
    7871              9751756.8
    7872             17825792.0
    7873              1468006.4
    7874             16777216.0
    7875              4194304.0
    7876              8703180.8
    7877             10066329.6
    7878             11534336.0
    7879             20971520.0
    7880             14680064.0
    7881             17825792.0
    7882             15728640.0
    7883             13631488.0
    7884             11534336.0
    7885             48234496.0
    7886             11534336.0
    7887              1677721.6
    7888             29360128.0
    7889     Varies with device
    7890             11534336.0
    7891              5138022.4
    7892              9017753.6
    7893             24117248.0
    7894             25165824.0
    7895              1677721.6
    7896              8808038.4
    7897                99328.0
    7898             20971520.0
    7899             46137344.0
    7900             92274688.0
    7901              5033164.8
    7902             33554432.0
    7903             26214400.0
    7904             28311552.0
    7905              7654604.8
    7906             14680064.0
    7907             15728640.0
    7908     Varies with device
    7909             16777216.0
    7910             26214400.0
    7911             14680064.0
    7912              3040870.4
    7913             18874368.0
    7914             36700160.0
    7915             26214400.0
    7916             14680064.0
    7917              6815744.0
    7918              5767168.0
    7919             29360128.0
    7920             17825792.0
    7921              7654604.8
    7922             25165824.0
    7923              2202009.6
    7924             25165824.0
    7925              2936012.8
    7926              7549747.2
    7927              9017753.6
    7928              2411724.8
    7929             12582912.0
    7930     Varies with device
    7931     Varies with device
    7932             73400320.0
    7933             35651584.0
    7934              2726297.6
    7935             36700160.0
    7936             25165824.0
    7937             92274688.0
    7938              7340032.0
    7939              4089446.4
    7940     Varies with device
    7941              3984588.8
    7942             27262976.0
    7943              5242880.0
    7944              4928307.2
    7945              5872025.6
    7946              5557452.8
    7947              2306867.2
    7948             27262976.0
    7949              5872025.6
    7950             27262976.0
    7951              9017753.6
    7952             22020096.0
    7953              5033164.8
    7954             15728640.0
    7955             20971520.0
    7956             20971520.0
    7957              5767168.0
    7958              5976883.2
    7959             11534336.0
    7960              4928307.2
    7961              4299161.6
    7962              7235174.4
    7963             12582912.0
    7964     Varies with device
    7965              6186598.4
    7966              2306867.2
    7967              7444889.6
    7968              8283750.4
    7969             26214400.0
    7970             27262976.0
    7971              5138022.4
    7972             26214400.0
    7973              8703180.8
    7974              6291456.0
    7975              3145728.0
    7976             19922944.0
    7977              4299161.6
    7978              6291456.0
    7979              4823449.6
    7980             27262976.0
    7981              4613734.4
    7982             10485760.0
    7983              5242880.0
    7984             16777216.0
    7985             23068672.0
    7986              8493465.6
    7987             22020096.0
    7988             13631488.0
    7989              5033164.8
    7990              6396313.6
    7991              4508876.8
    7992              6501171.2
    7993              3670016.0
    7994              2516582.4
    7995             39845888.0
    7996              9961472.0
    7997               528384.0
    7998               857088.0
    7999              2097152.0
    8000             17825792.0
    8001              7130316.8
    8002              1153433.6
    8003               798720.0
    8004              3879731.2
    8005               984064.0
    8006              2726297.6
    8007              5033164.8
    8008              8178892.8
    8009             23068672.0
    8010     Varies with device
    8011              1572864.0
    8012             10066329.6
    8013     Varies with device
    8014              2516582.4
    8015     Varies with device
    8016             57671680.0
    8017               275456.0
    8018              3879731.2
    8019             22020096.0
    8020            100663296.0
    8021              1153433.6
    8022                20480.0
    8023             12582912.0
    8024     Varies with device
    8025               509952.0
    8026             13631488.0
    8027              4404019.2
    8028             11534336.0
    8029     Varies with device
    8030               614400.0
    8031             18874368.0
    8032              3145728.0
    8033             14680064.0
    8034             75497472.0
    8035     Varies with device
    8036             56623104.0
    8037             12582912.0
    8038             38797312.0
    8039              6815744.0
    8040              6920601.6
    8041             52428800.0
    8042              4508876.8
    8043              7969177.6
    8044              5767168.0
    8045             98566144.0
    8046             36700160.0
    8047             57671680.0
    8048             11534336.0
    8049             12582912.0
    8050             22020096.0
    8051             28311552.0
    8052              5557452.8
    8053             31457280.0
    8054             10485760.0
    8055             11534336.0
    8056             10276044.8
    8057             71303168.0
    8058              3984588.8
    8059             34603008.0
    8060             25165824.0
    8061             36700160.0
    8062             20971520.0
    8063              3565158.4
    8064              2621440.0
    8065             33554432.0
    8066              6081740.8
    8067             18874368.0
    8068             20971520.0
    8069             33554432.0
    8070              7235174.4
    8071             12582912.0
    8072             17825792.0
    8073               606208.0
    8074             52428800.0
    8075             19922944.0
    8076             57671680.0
    8077              1048576.0
    8078              2097152.0
    8079              7864320.0
    8080             16777216.0
    8081             10485760.0
    8082     Varies with device
    8083     Varies with device
    8084              1153433.6
    8085             84934656.0
    8086             15728640.0
    8087             13631488.0
    8088             20971520.0
    8089              7444889.6
    8090              4718592.0
    8091              1363148.8
    8092             10485760.0
    8093             30408704.0
    8094               766976.0
    8095             20971520.0
    8096              8493465.6
    8097              3984588.8
    8098              1677721.6
    8099              6396313.6
    8100              5138022.4
    8101     Varies with device
    8102             20971520.0
    8103             15728640.0
    8104              9751756.8
    8105             12582912.0
    8106              4089446.4
    8107             63963136.0
    8108              6081740.8
    8109              5242880.0
    8110              4613734.4
    8111     Varies with device
    8112               657408.0
    8113              9856614.4
    8114              3460300.8
    8115     Varies with device
    8116              6710886.4
    8117              4194304.0
    8118             48234496.0
    8119             12582912.0
    8120             31457280.0
    8121     Varies with device
    8122              1468006.4
    8123             15728640.0
    8124              1887436.8
    8125             19922944.0
    8126     Varies with device
    8127             12582912.0
    8128             22020096.0
    8129             26214400.0
    8130              3565158.4
    8131              9122611.2
    8132             19922944.0
    8133             10276044.8
    8134             18874368.0
    8135             18874368.0
    8136             15728640.0
    8137             17825792.0
    8138             45088768.0
    8139              9751756.8
    8140             11534336.0
    8141              2097152.0
    8142              2306867.2
    8143              4613734.4
    8144              5662310.4
    8145             12582912.0
    8146              1468006.4
    8147             17825792.0
    8148               902144.0
    8149              3460300.8
    8150             17825792.0
    8151             13631488.0
    8152              7654604.8
    8153             24117248.0
    8154             12582912.0
    8155                73728.0
    8156     Varies with device
    8157              2306867.2
    8158     Varies with device
    8159              3670016.0
    8160             18874368.0
    8161              9017753.6
    8162              5033164.8
    8163              7654604.8
    8164             13631488.0
    8165              5557452.8
    8166              6501171.2
    8167               671744.0
    8168              5242880.0
    8169               615424.0
    8170              8703180.8
    8171     Varies with device
    8172             76546048.0
    8173              7759462.4
    8174             23068672.0
    8175             27262976.0
    8176             14680064.0
    8177              7654604.8
    8178              3250585.6
    8179             11534336.0
    8180     Varies with device
    8181             12582912.0
    8182             13631488.0
    8183             34603008.0
    8184     Varies with device
    8185             51380224.0
    8186             32505856.0
    8187     Varies with device
    8188             35651584.0
    8189             46137344.0
    8190             11534336.0
    8191             57671680.0
    8192     Varies with device
    8193             61865984.0
    8194     Varies with device
    8195             62914560.0
    8196             26214400.0
    8197             58720256.0
    8198             20971520.0
    8199              2411724.8
    8200               226304.0
    8201              1572864.0
    8202             50331648.0
    8203             32505856.0
    8204              7864320.0
    8205               233472.0
    8206              2726297.6
    8207              2411724.8
    8208              2936012.8
    8209             16777216.0
    8210              2306867.2
    8211              4718592.0
    8212             34603008.0
    8213             75497472.0
    8214              3460300.8
    8215              6920601.6
    8216              5452595.2
    8217              8388608.0
    8218               110592.0
    8219              3670016.0
    8220             11534336.0
    8221             10485760.0
    8222              9856614.4
    8223             10485760.0
    8224     Varies with device
    8225               962560.0
    8226              9122611.2
    8227              6606028.8
    8228              4299161.6
    8229             38797312.0
    8230              3355443.2
    8231             11534336.0
    8232              2726297.6
    8233              3145728.0
    8234             14680064.0
    8235             49283072.0
    8236              2621440.0
    8237              2411724.8
    8238             51380224.0
    8239             46137344.0
    8240              3145728.0
    8241              2411724.8
    8242             15728640.0
    8243             10485760.0
    8244              7969177.6
    8245             61865984.0
    8246     Varies with device
    8247             37748736.0
    8248             33554432.0
    8249              9961472.0
    8250             99614720.0
    8251              6081740.8
    8252             36700160.0
    8253             66060288.0
    8254             10171187.2
    8255             95420416.0
    8256             66060288.0
    8257     Varies with device
    8258              3250585.6
    8259             92274688.0
    8260             12582912.0
    8261             96468992.0
    8262              4299161.6
    8263     Varies with device
    8264             24117248.0
    8265              7549747.2
    8266              6710886.4
    8267              3565158.4
    8268             95420416.0
    8269              6396313.6
    8270             23068672.0
    8271              7235174.4
    8272             95420416.0
    8273             75497472.0
    8274             10485760.0
    8275              2306867.2
    8276              8912896.0
    8277             12582912.0
    8278             22020096.0
    8279             11534336.0
    8280              2411724.8
    8281             16777216.0
    8282             33554432.0
    8283             17825792.0
    8284              1363148.8
    8285             18874368.0
    8286             10485760.0
    8287              1258291.2
    8288              9122611.2
    8289             44040192.0
    8290     Varies with device
    8291             39845888.0
    8292              5557452.8
    8293     Varies with device
    8294     Varies with device
    8295              6081740.8
    8296            100663296.0
    8297             22020096.0
    8298     Varies with device
    8299             10485760.0
    8300             83886080.0
    8301              6396313.6
    8302              1572864.0
    8303             32505856.0
    8304             27262976.0
    8305             20971520.0
    8306              3040870.4
    8307     Varies with device
    8308     Varies with device
    8309             39845888.0
    8310             37748736.0
    8311             14680064.0
    8312               180224.0
    8313             46137344.0
    8314             85983232.0
    8315              3460300.8
    8316             18874368.0
    8317              8703180.8
    8318                33792.0
    8319             14680064.0
    8320     Varies with device
    8321              8703180.8
    8322             10380902.4
    8323              8388608.0
    8324              1572864.0
    8325              3774873.6
    8326             15728640.0
    8327              7025459.2
    8328               678912.0
    8329              1468006.4
    8330             20971520.0
    8331              6291456.0
    8332             14680064.0
    8333              1572864.0
    8334              2411724.8
    8335             28311552.0
    8336             79691776.0
    8337             15728640.0
    8338             33554432.0
    8339              6606028.8
    8340             25165824.0
    8341              1468006.4
    8342             37748736.0
    8343             15728640.0
    8344              3145728.0
    8345             14680064.0
    8346             10485760.0
    8347             15728640.0
    8348              5452595.2
    8349              3040870.4
    8350              3460300.8
    8351              5976883.2
    8352             37748736.0
    8353              5976883.2
    8354              3040870.4
    8355              3460300.8
    8356              4718592.0
    8357     Varies with device
    8358              1677721.6
    8359              2936012.8
    8360     Varies with device
    8361              1153433.6
    8362              4299161.6
    8363              1153433.6
    8364              7130316.8
    8365             29360128.0
    8366             33554432.0
    8367             12582912.0
    8368             23068672.0
    8369              1468006.4
    8370              1258291.2
    8371             20971520.0
    8372              7969177.6
    8373              3040870.4
    8374     Varies with device
    8375              2621440.0
    8376             11534336.0
    8377             35651584.0
    8378             22020096.0
    8379              4718592.0
    8380             23068672.0
    8381                34816.0
    8382             14680064.0
    8383             12582912.0
    8384     Varies with device
    8385              1363148.8
    8386              3250585.6
    8387             11534336.0
    8388              6186598.4
    8389              3145728.0
    8390             12582912.0
    8391              1153433.6
    8392             27262976.0
    8393              4718592.0
    8394              6186598.4
    8395              5976883.2
    8396     Varies with device
    8397             38797312.0
    8398     Varies with device
    8399             80740352.0
    8400             45088768.0
    8401             20971520.0
    8402              8388608.0
    8403             17825792.0
    8404             10485760.0
    8405             35651584.0
    8406             84934656.0
    8407     Varies with device
    8408             36700160.0
    8409            104857600.0
    8410             50331648.0
    8411             61865984.0
    8412     Varies with device
    8413             14680064.0
    8414             10485760.0
    8415             16777216.0
    8416             25165824.0
    8417              2411724.8
    8418              4299161.6
    8419             29360128.0
    8420             28311552.0
    8421             28311552.0
    8422              7025459.2
    8423             24117248.0
    8424     Varies with device
    8425              5138022.4
    8426             89128960.0
    8427             42991616.0
    8428             14680064.0
    8429             98566144.0
    8430             47185920.0
    8431               964608.0
    8432             27262976.0
    8433            103809024.0
    8434              6186598.4
    8435             37748736.0
    8436             54525952.0
    8437             66060288.0
    8438             27262976.0
    8439     Varies with device
    8440             36700160.0
    8441     Varies with device
    8442              5662310.4
    8443             55574528.0
    8444              4194304.0
    8445             53477376.0
    8446             20971520.0
    8447             40894464.0
    8448              5242880.0
    8449              5976883.2
    8450             30408704.0
    8451             91226112.0
    8452              7444889.6
    8453             63963136.0
    8454     Varies with device
    8455             31457280.0
    8456             54525952.0
    8457             46137344.0
    8458     Varies with device
    8459              4823449.6
    8460             94371840.0
    8461     Varies with device
    8462             95420416.0
    8463             31457280.0
    8464             20971520.0
    8465             29360128.0
    8466             16777216.0
    8467             16777216.0
    8468              8283750.4
    8469             59768832.0
    8470             16777216.0
    8471             26214400.0
    8472             35651584.0
    8473             30408704.0
    8474             91226112.0
    8475             39845888.0
    8476             18874368.0
    8477             19922944.0
    8478             73400320.0
    8479              3460300.8
    8480             14680064.0
    8481             69206016.0
    8482              4508876.8
    8483             50331648.0
    8484             32505856.0
    8485              3460300.8
    8486             26214400.0
    8487             11534336.0
    8488             17825792.0
    8489             16777216.0
    8490              3670016.0
    8491             34603008.0
    8492              6710886.4
    8493              4089446.4
    8494              1572864.0
    8495              6920601.6
    8496             24117248.0
    8497              2516582.4
    8498             36700160.0
    8499              1153433.6
    8500             11534336.0
    8501             13631488.0
    8502             35651584.0
    8503              7444889.6
    8504              4404019.2
    8505             17825792.0
    8506              3145728.0
    8507              5557452.8
    8508             31457280.0
    8509              3460300.8
    8510              2621440.0
    8511              2097152.0
    8512     Varies with device
    8513              6606028.8
    8514     Varies with device
    8515              2411724.8
    8516              7549747.2
    8517     Varies with device
    8518              2306867.2
    8519              6501171.2
    8520              1782579.2
    8521             10485760.0
    8522              3355443.2
    8523             31457280.0
    8524             11534336.0
    8525              2516582.4
    8526              1782579.2
    8527              3460300.8
    8528             13631488.0
    8529              3355443.2
    8530              8283750.4
    8531              5452595.2
    8532             23068672.0
    8533              1677721.6
    8534              9856614.4
    8535              5662310.4
    8536              1153433.6
    8537              3460300.8
    8538              5767168.0
    8539              4404019.2
    8540             10485760.0
    8541              5242880.0
    8542              3040870.4
    8543             31457280.0
    8544              5662310.4
    8545              3879731.2
    8546              3040870.4
    8547              3145728.0
    8548              4404019.2
    8549             11534336.0
    8550              3145728.0
    8551              6606028.8
    8552             20971520.0
    8553            101711872.0
    8554               265216.0
    8555              5138022.4
    8556             11534336.0
    8557             37748736.0
    8558               167936.0
    8559              3145728.0
    8560             16777216.0
    8561               468992.0
    8562               250880.0
    8563              5452595.2
    8564              5976883.2
    8565              3460300.8
    8566             28311552.0
    8567             10485760.0
    8568              8178892.8
    8569              8912896.0
    8570              6186598.4
    8571               644096.0
    8572              1782579.2
    8573             96468992.0
    8574             22020096.0
    8575              1258291.2
    8576             34603008.0
    8577             36700160.0
    8578     Varies with device
    8579              2516582.4
    8580             11534336.0
    8581             24117248.0
    8582              4194304.0
    8583     Varies with device
    8584             17825792.0
    8585              4299161.6
    8586                28672.0
    8587              1887436.8
    8588              3565158.4
    8589             33554432.0
    8590               294912.0
    8591              4404019.2
    8592             17825792.0
    8593             18874368.0
    8594             38797312.0
    8595             15728640.0
    8596              3984588.8
    8597             22020096.0
    8598              4404019.2
    8599             29360128.0
    8600              2411724.8
    8601               793600.0
    8602             10485760.0
    8603              6186598.4
    8604             34603008.0
    8605              7864320.0
    8606             19922944.0
    8607              3984588.8
    8608              4194304.0
    8609     Varies with device
    8610             65011712.0
    8611     Varies with device
    8612              7969177.6
    8613             28311552.0
    8614              2202009.6
    8615             15728640.0
    8616              3565158.4
    8617             31457280.0
    8618             18874368.0
    8619              5557452.8
    8620             32505856.0
    8621             52428800.0
    8622             23068672.0
    8623              2726297.6
    8624             90177536.0
    8625             74448896.0
    8626             33554432.0
    8627              9646899.2
    8628     Varies with device
    8629             16777216.0
    8630     Varies with device
    8631     Varies with device
    8632              4404019.2
    8633              3670016.0
    8634     Varies with device
    8635             62914560.0
    8636     Varies with device
    8637              6186598.4
    8638             81788928.0
    8639     Varies with device
    8640             12582912.0
    8641              5138022.4
    8642     Varies with device
    8643     Varies with device
    8644              3145728.0
    8645              6606028.8
    8646     Varies with device
    8647              8808038.4
    8648              4194304.0
    8649              1992294.4
    8650               205824.0
    8651              5242880.0
    8652              4194304.0
    8653             31457280.0
    8654     Varies with device
    8655              9437184.0
    8656              8912896.0
    8657     Varies with device
    8658     Varies with device
    8659             13631488.0
    8660              3565158.4
    8661             20971520.0
    8662              4613734.4
    8663              8703180.8
    8664             19922944.0
    8665              6710886.4
    8666              2516582.4
    8667     Varies with device
    8668              5557452.8
    8669              5452595.2
    8670             73400320.0
    8671             42991616.0
    8672              1992294.4
    8673             40894464.0
    8674              6081740.8
    8675             11534336.0
    8676              7235174.4
    8677             15728640.0
    8678              4928307.2
    8679              8598323.2
    8680             18874368.0
    8681              4613734.4
    8682             15728640.0
    8683              9856614.4
    8684              3145728.0
    8685             12582912.0
    8686              3565158.4
    8687              5347737.6
    8688              2097152.0
    8689              3355443.2
    8690              5452595.2
    8691             11534336.0
    8692             14680064.0
    8693              8074035.2
    8694             12582912.0
    8695              3984588.8
    8696              8493465.6
    8697              5662310.4
    8698              5976883.2
    8699             10485760.0
    8700              9122611.2
    8701              3984588.8
    8702              5872025.6
    8703              3984588.8
    8704              4299161.6
    8705             11534336.0
    8706              5976883.2
    8707              3984588.8
    8708              9542041.6
    8709              2621440.0
    8710              7025459.2
    8711             11534336.0
    8712              8074035.2
    8713              7340032.0
    8714             45088768.0
    8715             13631488.0
    8716             20971520.0
    8717             61865984.0
    8718              4089446.4
    8719             28311552.0
    8720             16777216.0
    8721             66060288.0
    8722             12582912.0
    8723             31457280.0
    8724             46137344.0
    8725             45088768.0
    8726             17825792.0
    8727             44040192.0
    8728              6606028.8
    8729              2726297.6
    8730             20971520.0
    8731             16777216.0
    8732             13631488.0
    8733              6606028.8
    8734              2831155.2
    8735             58720256.0
    8736              1153433.6
    8737             27262976.0
    8738             58720256.0
    8739               803840.0
    8740             26214400.0
    8741              4928307.2
    8742              6291456.0
    8743              5138022.4
    8744             18874368.0
    8745     Varies with device
    8746             69206016.0
    8747             42991616.0
    8748            100663296.0
    8749             79691776.0
    8750              4404019.2
    8751             37748736.0
    8752              4089446.4
    8753             25165824.0
    8754             36700160.0
    8755             24117248.0
    8756             13631488.0
    8757             19922944.0
    8758     Varies with device
    8759              7969177.6
    8760            100663296.0
    8761             36700160.0
    8762     Varies with device
    8763             84934656.0
    8764             99614720.0
    8765             10485760.0
    8766             22020096.0
    8767              4928307.2
    8768             28311552.0
    8769             26214400.0
    8770              3565158.4
    8771             34603008.0
    8772              5976883.2
    8773              3250585.6
    8774     Varies with device
    8775              4508876.8
    8776              4718592.0
    8777             48234496.0
    8778             34603008.0
    8779             60817408.0
    8780              9751756.8
    8781              6501171.2
    8782              7025459.2
    8783             42991616.0
    8784              6291456.0
    8785              9961472.0
    8786             62914560.0
    8787             32505856.0
    8788             37748736.0
    8789              4508876.8
    8790              1572864.0
    8791             71303168.0
    8792              1258291.2
    8793             12582912.0
    8794             23068672.0
    8795             47185920.0
    8796             65011712.0
    8797             83886080.0
    8798             35651584.0
    8799              3670016.0
    8800             70254592.0
    8801             60817408.0
    8802              5347737.6
    8803              5767168.0
    8804             12582912.0
    8805             19922944.0
    8806             19922944.0
    8807              9227468.8
    8808             11534336.0
    8809             12582912.0
    8810             33554432.0
    8811             23068672.0
    8812             39845888.0
    8813     Varies with device
    8814     Varies with device
    8815             62914560.0
    8816             36700160.0
    8817             15728640.0
    8818             45088768.0
    8819     Varies with device
    8820              4613734.4
    8821              2202009.6
    8822     Varies with device
    8823     Varies with device
    8824             24117248.0
    8825              2411724.8
    8826               651264.0
    8827              4404019.2
    8828              4823449.6
    8829              8283750.4
    8830              1468006.4
    8831             22020096.0
    8832               937984.0
    8833             29360128.0
    8834              1677721.6
    8835             88080384.0
    8836              6920601.6
    8837             12582912.0
    8838             14680064.0
    8839             39845888.0
    8840              5242880.0
    8841              3040870.4
    8842              1468006.4
    8843              8808038.4
    8844              3145728.0
    8845              2726297.6
    8846              3879731.2
    8847            104857600.0
    8848              6081740.8
    8849             29360128.0
    8850              4718592.0
    8851              9332326.4
    8852              3250585.6
    8853             23068672.0
    8854             10380902.4
    8855             51380224.0
    8856              1677721.6
    8857              9332326.4
    8858             23068672.0
    8859             24117248.0
    8860             98566144.0
    8861              6606028.8
    8862              1153433.6
    8863             99614720.0
    8864             22020096.0
    8865              9961472.0
    8866             25165824.0
    8867              1572864.0
    8868             25165824.0
    8869              8283750.4
    8870              5557452.8
    8871              1258291.2
    8872             12582912.0
    8873              5976883.2
    8874             13631488.0
    8875             25165824.0
    8876             56623104.0
    8877              4089446.4
    8878     Varies with device
    8879     Varies with device
    8880             53477376.0
    8881             51380224.0
    8882              1363148.8
    8883              1017856.0
    8884              3774873.6
    8885              4299161.6
    8886             69206016.0
    8887             33554432.0
    8888              3040870.4
    8889              9332326.4
    8890             49283072.0
    8891              2202009.6
    8892             12582912.0
    8893              8178892.8
    8894             15728640.0
    8895             26214400.0
    8896             14680064.0
    8897             10485760.0
    8898              9332326.4
    8899              5033164.8
    8900              3355443.2
    8901              4928307.2
    8902     Varies with device
    8903              2097152.0
    8904             14680064.0
    8905              4928307.2
    8906             23068672.0
    8907             14680064.0
    8908              4508876.8
    8909     Varies with device
    8910              5452595.2
    8911               316416.0
    8912              9122611.2
    8913               496640.0
    8914              2621440.0
    8915              9646899.2
    8916             40894464.0
    8917            100663296.0
    8918     Varies with device
    8919              7235174.4
    8920             10276044.8
    8921             16777216.0
    8922              6501171.2
    8923               935936.0
    8924              8598323.2
    8925     Varies with device
    8926             13631488.0
    8927             50331648.0
    8928              2936012.8
    8929     Varies with device
    8930             32505856.0
    8931              1363148.8
    8932             30408704.0
    8933             17825792.0
    8934             19922944.0
    8935              2726297.6
    8936             74448896.0
    8937              2726297.6
    8938             14680064.0
    8939             46137344.0
    8940              5347737.6
    8941             11534336.0
    8942             13631488.0
    8943              2726297.6
    8944             68157440.0
    8945              9122611.2
    8946               924672.0
    8947             26214400.0
    8948              4194304.0
    8949              1782579.2
    8950              2516582.4
    8951              8912896.0
    8952             27262976.0
    8953              2097152.0
    8954              2621440.0
    8955             18874368.0
    8956              3040870.4
    8957              1363148.8
    8958              1782579.2
    8959              2831155.2
    8960               622592.0
    8961             23068672.0
    8962              4404019.2
    8963             42991616.0
    8964             68157440.0
    8965             16777216.0
    8966             15728640.0
    8967              9751756.8
    8968              4404019.2
    8969              5347737.6
    8970              9332326.4
    8971             25165824.0
    8972              4194304.0
    8973     Varies with device
    8974             32505856.0
    8975              9961472.0
    8976              8598323.2
    8977             17825792.0
    8978              1887436.8
    8979     Varies with device
    8980             31457280.0
    8981             32505856.0
    8982              8912896.0
    8983              6606028.8
    8984             10485760.0
    8985              5033164.8
    8986              6291456.0
    8987              6081740.8
    8988              6606028.8
    8989             82837504.0
    8990              6606028.8
    8991              2516582.4
    8992              5242880.0
    8993              2097152.0
    8994              4613734.4
    8995              8388608.0
    8996     Varies with device
    8997             40894464.0
    8998              3670016.0
    8999              4089446.4
    9000             10276044.8
    9001              6186598.4
    9002             15728640.0
    9003              6606028.8
    9004              3145728.0
    9005              2306867.2
    9006              4404019.2
    9007             16777216.0
    9008              1572864.0
    9009     Varies with device
    9010               512000.0
    9011              3565158.4
    9012              3460300.8
    9013             13631488.0
    9014              2621440.0
    9015             10485760.0
    9016              3145728.0
    9017                55296.0
    9018              5033164.8
    9019              3040870.4
    9020              3145728.0
    9021             41943040.0
    9022             11534336.0
    9023              6920601.6
    9024             16777216.0
    9025     Varies with device
    9026              1677721.6
    9027             50331648.0
    9028              4823449.6
    9029              2936012.8
    9030     Varies with device
    9031             24117248.0
    9032             27262976.0
    9033              2621440.0
    9034             25165824.0
    9035             58720256.0
    9036              4194304.0
    9037              2621440.0
    9038             41943040.0
    9039             76546048.0
    9040             32505856.0
    9041     Varies with device
    9042             47185920.0
    9043             44040192.0
    9044             40894464.0
    9045             50331648.0
    9046              7549747.2
    9047             10485760.0
    9048             14680064.0
    9049             11534336.0
    9050             22020096.0
    9051             33554432.0
    9052             49283072.0
    9053     Varies with device
    9054              2202009.6
    9055             16777216.0
    9056             34603008.0
    9057              3565158.4
    9058             17825792.0
    9059              1677721.6
    9060             44040192.0
    9061             11534336.0
    9062             50331648.0
    9063             41943040.0
    9064     Varies with device
    9065             20971520.0
    9066             62914560.0
    9067              2726297.6
    9068             23068672.0
    9069              3355443.2
    9070     Varies with device
    9071              5557452.8
    9072             23068672.0
    9073             10485760.0
    9074             45088768.0
    9075              5033164.8
    9076             20971520.0
    9077              1992294.4
    9078              3460300.8
    9079              1258291.2
    9080             56623104.0
    9081             27262976.0
    9082     Varies with device
    9083             70254592.0
    9084              6710886.4
    9085             52428800.0
    9086              2621440.0
    9087              1363148.8
    9088             22020096.0
    9089              7759462.4
    9090             31457280.0
    9091              4089446.4
    9092             10276044.8
    9093              4718592.0
    9094             10276044.8
    9095              3355443.2
    9096             49283072.0
    9097              6291456.0
    9098              2621440.0
    9099              2411724.8
    9100     Varies with device
    9101             14680064.0
    9102              7340032.0
    9103               575488.0
    9104             14680064.0
    9105              4928307.2
    9106              4718592.0
    9107             10485760.0
    9108              3145728.0
    9109             18874368.0
    9110             33554432.0
    9111               867328.0
    9112              2936012.8
    9113             10380902.4
    9114             17825792.0
    9115              2621440.0
    9116              3565158.4
    9117              7130316.8
    9118               979968.0
    9119             30408704.0
    9120              3774873.6
    9121     Varies with device
    9122             29360128.0
    9123              3250585.6
    9124              2621440.0
    9125              1677721.6
    9126             24117248.0
    9127             15728640.0
    9128              6396313.6
    9129             10485760.0
    9130              2516582.4
    9131              4823449.6
    9132              3460300.8
    9133             10485760.0
    9134              3984588.8
    9135              3250585.6
    9136             34603008.0
    9137              6501171.2
    9138              2202009.6
    9139             18874368.0
    9140             38797312.0
    9141             60817408.0
    9142             23068672.0
    9143             74448896.0
    9144     Varies with device
    9145             66060288.0
    9146             51380224.0
    9147             15728640.0
    9148     Varies with device
    9149             70254592.0
    9150             47185920.0
    9151             30408704.0
    9152             59768832.0
    9153     Varies with device
    9154             45088768.0
    9155             12582912.0
    9156            100663296.0
    9157             27262976.0
    9158             82837504.0
    9159            102760448.0
    9160             19922944.0
    9161             29360128.0
    9162             52428800.0
    9163             40894464.0
    9164             31457280.0
    9165            103809024.0
    9166             60817408.0
    9167     Varies with device
    9168             96468992.0
    9169             45088768.0
    9170            104857600.0
    9171             75497472.0
    9172             38797312.0
    9173              3250585.6
    9174             34603008.0
    9175              3460300.8
    9176              5872025.6
    9177             16777216.0
    9178              5662310.4
    9179             11534336.0
    9180             31457280.0
    9181             32505856.0
    9182              3460300.8
    9183             31457280.0
    9184              6396313.6
    9185              5452595.2
    9186             18874368.0
    9187             16777216.0
    9188             26214400.0
    9189              2411724.8
    9190             13631488.0
    9191     Varies with device
    9192              9227468.8
    9193             15728640.0
    9194              2516582.4
    9195             73400320.0
    9196              4089446.4
    9197              4089446.4
    9198              9437184.0
    9199             22020096.0
    9200              7654604.8
    9201              1887436.8
    9202              1468006.4
    9203               970752.0
    9204              5242880.0
    9205              2306867.2
    9206               704512.0
    9207              2306867.2
    9208              2621440.0
    9209              7654604.8
    9210             94371840.0
    9211              3250585.6
    9212               830464.0
    9213              4194304.0
    9214              4299161.6
    9215              2936012.8
    9216     Varies with device
    9217             16777216.0
    9218              4508876.8
    9219              3984588.8
    9220               276480.0
    9221              3879731.2
    9222     Varies with device
    9223              3145728.0
    9224              2411724.8
    9225              3984588.8
    9226              2831155.2
    9227              2306867.2
    9228             29360128.0
    9229              1572864.0
    9230              5767168.0
    9231              4194304.0
    9232                49152.0
    9233              8074035.2
    9234             27262976.0
    9235              9017753.6
    9236              4508876.8
    9237             17825792.0
    9238              3984588.8
    9239             15728640.0
    9240              3565158.4
    9241              5242880.0
    9242              2411724.8
    9243              1677721.6
    9244              2831155.2
    9245              8493465.6
    9246              3984588.8
    9247              6606028.8
    9248              7759462.4
    9249             12582912.0
    9250              8283750.4
    9251              5662310.4
    9252              4194304.0
    9253              3250585.6
    9254              5767168.0
    9255              8808038.4
    9256              4089446.4
    9257               336896.0
    9258              3145728.0
    9259              4404019.2
    9260              3984588.8
    9261              5557452.8
    9262              6291456.0
    9263              4508876.8
    9264             31457280.0
    9265              4823449.6
    9266             17825792.0
    9267             14680064.0
    9268             40894464.0
    9269             13631488.0
    9270             36700160.0
    9271              7235174.4
    9272             14680064.0
    9273             24117248.0
    9274             15728640.0
    9275             11534336.0
    9276             14680064.0
    9277             14680064.0
    9278             17825792.0
    9279             17825792.0
    9280             10276044.8
    9281             15728640.0
    9282              4613734.4
    9283             45088768.0
    9284             19922944.0
    9285     Varies with device
    9286             37748736.0
    9287              9856614.4
    9288             30408704.0
    9289             10485760.0
    9290             24117248.0
    9291             10485760.0
    9292             34603008.0
    9293             20971520.0
    9294             34603008.0
    9295             14680064.0
    9296              2411724.8
    9297             56623104.0
    9298              2621440.0
    9299              4508876.8
    9300             11534336.0
    9301             15728640.0
    9302              1677721.6
    9303              2097152.0
    9304             32505856.0
    9305     Varies with device
    9306              3774873.6
    9307              3879731.2
    9308             14680064.0
    9309              6920601.6
    9310              5347737.6
    9311              4823449.6
    9312              5557452.8
    9313              7130316.8
    9314             55574528.0
    9315              7864320.0
    9316             24117248.0
    9317              1572864.0
    9318              2516582.4
    9319             26214400.0
    9320             31457280.0
    9321     Varies with device
    9322             13631488.0
    9323              6815744.0
    9324              4613734.4
    9325             24117248.0
    9326              5242880.0
    9327             16777216.0
    9328              7759462.4
    9329              1153433.6
    9330              1153433.6
    9331             39845888.0
    9332               535552.0
    9333               120832.0
    9334             18874368.0
    9335             16777216.0
    9336             63963136.0
    9337             58720256.0
    9338              3250585.6
    9339             12582912.0
    9340              6291456.0
    9341              4194304.0
    9342             31457280.0
    9343             17825792.0
    9344             53477376.0
    9345     Varies with device
    9346             32505856.0
    9347             44040192.0
    9348            103809024.0
    9349              6710886.4
    9350             23068672.0
    9351             66060288.0
    9352             11534336.0
    9353            103809024.0
    9354             22020096.0
    9355             14680064.0
    9356             48234496.0
    9357             18874368.0
    9358             63963136.0
    9359            103809024.0
    9360             32505856.0
    9361             27262976.0
    9362             27262976.0
    9363              7340032.0
    9364              3460300.8
    9365              4089446.4
    9366              9122611.2
    9367             65011712.0
    9368              3670016.0
    9369              6710886.4
    9370     Varies with device
    9371               943104.0
    9372              7444889.6
    9373               894976.0
    9374              2411724.8
    9375             57671680.0
    9376             42991616.0
    9377             54525952.0
    9378             33554432.0
    9379             13631488.0
    9380             46137344.0
    9381              3565158.4
    9382             98566144.0
    9383             24117248.0
    9384             50331648.0
    9385     Varies with device
    9386             27262976.0
    9387             19922944.0
    9388             46137344.0
    9389             34603008.0
    9390             36700160.0
    9391             68157440.0
    9392              9646899.2
    9393             24117248.0
    9394             50331648.0
    9395             38797312.0
    9396             20971520.0
    9397              5033164.8
    9398              5976883.2
    9399             30408704.0
    9400             44040192.0
    9401             50331648.0
    9402              3879731.2
    9403             63963136.0
    9404             69206016.0
    9405             85983232.0
    9406              1677721.6
    9407              1887436.8
    9408     Varies with device
    9409             89128960.0
    9410              1004544.0
    9411             12582912.0
    9412              4718592.0
    9413             14680064.0
    9414             15728640.0
    9415              3984588.8
    9416               802816.0
    9417     Varies with device
    9418              8074035.2
    9419             13631488.0
    9420              1572864.0
    9421             13631488.0
    9422              7444889.6
    9423              3355443.2
    9424             22020096.0
    9425             24117248.0
    9426             28311552.0
    9427             19922944.0
    9428             47185920.0
    9429             15728640.0
    9430     Varies with device
    9431             13631488.0
    9432             34603008.0
    9433              2202009.6
    9434             16777216.0
    9435              3460300.8
    9436             20971520.0
    9437              5452595.2
    9438              9122611.2
    9439     Varies with device
    9440            103809024.0
    9441             80740352.0
    9442              3984588.8
    9443              6291456.0
    9444              5242880.0
    9445             57671680.0
    9446              5557452.8
    9447             38797312.0
    9448              2411724.8
    9449             12582912.0
    9450             52428800.0
    9451               286720.0
    9452             32505856.0
    9453             19922944.0
    9454             72351744.0
    9455             26214400.0
    9456             39845888.0
    9457             27262976.0
    9458             13631488.0
    9459              7130316.8
    9460              4299161.6
    9461             50331648.0
    9462     Varies with device
    9463             71303168.0
    9464             47185920.0
    9465     Varies with device
    9466              9751756.8
    9467             46137344.0
    9468             42991616.0
    9469             68157440.0
    9470              4613734.4
    9471             48234496.0
    9472             24117248.0
    9473             31457280.0
    9474             29360128.0
    9475             49283072.0
    9476             61865984.0
    9477              1677721.6
    9478             14680064.0
    9479             45088768.0
    9480              4299161.6
    9481     Varies with device
    9482              3984588.8
    9483             49283072.0
    9484             25165824.0
    9485             32505856.0
    9486              2831155.2
    9487              7654604.8
    9488             61865984.0
    9489             50331648.0
    9490              1363148.8
    9491             12582912.0
    9492              5242880.0
    9493              6920601.6
    9494             57671680.0
    9495              2936012.8
    9496     Varies with device
    9497             10485760.0
    9498             31457280.0
    9499              3879731.2
    9500             18874368.0
    9501              7759462.4
    9502             17825792.0
    9503              1363148.8
    9504              3879731.2
    9505             27262976.0
    9506              5242880.0
    9507              3460300.8
    9508              4089446.4
    9509             30408704.0
    9510              5872025.6
    9511              3145728.0
    9512              1887436.8
    9513              6081740.8
    9514              5976883.2
    9515              2726297.6
    9516             39845888.0
    9517              6501171.2
    9518             11534336.0
    9519              1572864.0
    9520             65011712.0
    9521              4194304.0
    9522              3250585.6
    9523             58720256.0
    9524             31457280.0
    9525              1992294.4
    9526              2097152.0
    9527              4718592.0
    9528              4508876.8
    9529             10380902.4
    9530              2097152.0
    9531              2411724.8
    9532             30408704.0
    9533            103809024.0
    9534              5347737.6
    9535              2831155.2
    9536             67108864.0
    9537             51380224.0
    9538             82837504.0
    9539             49283072.0
    9540             26214400.0
    9541             51380224.0
    9542             37748736.0
    9543             23068672.0
    9544              7340032.0
    9545             62914560.0
    9546             13631488.0
    9547     Varies with device
    9548              3355443.2
    9549             32505856.0
    9550             52428800.0
    9551             62914560.0
    9552             16777216.0
    9553             26214400.0
    9554              3774873.6
    9555              3670016.0
    9556             12582912.0
    9557     Varies with device
    9558             34603008.0
    9559             25165824.0
    9560              4299161.6
    9561             51380224.0
    9562     Varies with device
    9563              3355443.2
    9564             76546048.0
    9565             23068672.0
    9566              4613734.4
    9567             33554432.0
    9568     Varies with device
    9569             66060288.0
    9570     Varies with device
    9571             15728640.0
    9572             16777216.0
    9573              6710886.4
    9574             26214400.0
    9575     Varies with device
    9576             54525952.0
    9577             22020096.0
    9578              2411724.8
    9579     Varies with device
    9580             15728640.0
    9581              7130316.8
    9582             41943040.0
    9583             92274688.0
    9584             61865984.0
    9585     Varies with device
    9586                24576.0
    9587             48234496.0
    9588             40894464.0
    9589              7235174.4
    9590             14680064.0
    9591             54525952.0
    9592             14680064.0
    9593             38797312.0
    9594     Varies with device
    9595             25165824.0
    9596             29360128.0
    9597     Varies with device
    9598              4613734.4
    9599             54525952.0
    9600             53477376.0
    9601             79691776.0
    9602              4613734.4
    9603     Varies with device
    9604             27262976.0
    9605             70254592.0
    9606             48234496.0
    9607             81788928.0
    9608             34603008.0
    9609             27262976.0
    9610     Varies with device
    9611     Varies with device
    9612             62914560.0
    9613              3145728.0
    9614             94371840.0
    9615             29360128.0
    9616             62914560.0
    9617             41943040.0
    9618             26214400.0
    9619             42991616.0
    9620     Varies with device
    9621     Varies with device
    9622             13631488.0
    9623     Varies with device
    9624     Varies with device
    9625     Varies with device
    9626             44040192.0
    9627             36700160.0
    9628             18874368.0
    9629             55574528.0
    9630             32505856.0
    9631              4508876.8
    9632     Varies with device
    9633             11534336.0
    9634              8912896.0
    9635             22020096.0
    9636     Varies with device
    9637             45088768.0
    9638             10485760.0
    9639              3565158.4
    9640     Varies with device
    9641             15728640.0
    9642              7444889.6
    9643     Varies with device
    9644             11534336.0
    9645              1572864.0
    9646             22020096.0
    9647             35651584.0
    9648             11534336.0
    9649              9332326.4
    9650             19922944.0
    9651              4404019.2
    9652              1153433.6
    9653             11534336.0
    9654             14680064.0
    9655     Varies with device
    9656             31457280.0
    9657             25165824.0
    9658               530432.0
    9659     Varies with device
    9660              6606028.8
    9661             42991616.0
    9662             18874368.0
    9663              5452595.2
    9664              7025459.2
    9665              3250585.6
    9666              2097152.0
    9667             37748736.0
    9668             62914560.0
    9669             77594624.0
    9670             11534336.0
    9671             96468992.0
    9672             87031808.0
    9673             56623104.0
    9674             44040192.0
    9675             24117248.0
    9676             32505856.0
    9677            102760448.0
    9678             72351744.0
    9679               772096.0
    9680             38797312.0
    9681             20971520.0
    9682             80740352.0
    9683             76546048.0
    9684             34603008.0
    9685              2831155.2
    9686             59768832.0
    9687             87031808.0
    9688             76546048.0
    9689     Varies with device
    9690             46137344.0
    9691              2936012.8
    9692             13631488.0
    9693              2202009.6
    9694             29360128.0
    9695              2411724.8
    9696               913408.0
    9697             73400320.0
    9698              2411724.8
    9699             20971520.0
    9700     Varies with device
    9701              9227468.8
    9702             25165824.0
    9703             22020096.0
    9704             15728640.0
    9705               157696.0
    9706             22020096.0
    9707              5033164.8
    9708             20971520.0
    9709             32505856.0
    9710             16777216.0
    9711             12582912.0
    9712             28311552.0
    9713              6606028.8
    9714             11534336.0
    9715              2097152.0
    9716              1572864.0
    9717             20971520.0
    9718             40894464.0
    9719              3355443.2
    9720             28311552.0
    9721              7549747.2
    9722             24117248.0
    9723             49283072.0
    9724             32505856.0
    9725             22020096.0
    9726             35651584.0
    9727              5557452.8
    9728             38797312.0
    9729              5976883.2
    9730             10485760.0
    9731             54525952.0
    9732             72351744.0
    9733             58720256.0
    9734             55574528.0
    9735             65011712.0
    9736             79691776.0
    9737             74448896.0
    9738             75497472.0
    9739             29360128.0
    9740              2936012.8
    9741             53477376.0
    9742             29360128.0
    9743             41943040.0
    9744             28311552.0
    9745     Varies with device
    9746             38797312.0
    9747             48234496.0
    9748             52428800.0
    9749             51380224.0
    9750             60817408.0
    9751             42991616.0
    9752             76546048.0
    9753             34603008.0
    9754             48234496.0
    9755             19922944.0
    9756             33554432.0
    9757             31457280.0
    9758             96468992.0
    9759             57671680.0
    9760             47185920.0
    9761             49283072.0
    9762             89128960.0
    9763             50331648.0
    9764             80740352.0
    9765             31457280.0
    9766             29360128.0
    9767             36700160.0
    9768              6081740.8
    9769             40894464.0
    9770             26214400.0
    9771             33554432.0
    9772              9332326.4
    9773             88080384.0
    9774             62914560.0
    9775              5557452.8
    9776             40894464.0
    9777             20971520.0
    9778             20971520.0
    9779             17825792.0
    9780     Varies with device
    9781                18432.0
    9782             44040192.0
    9783             30408704.0
    9784     Varies with device
    9785              6920601.6
    9786              3355443.2
    9787                33792.0
    9788              4613734.4
    9789              3565158.4
    9790               880640.0
    9791              1258291.2
    9792              1782579.2
    9793              2936012.8
    9794               372736.0
    9795              2306867.2
    9796              1992294.4
    9797              1782579.2
    9798              3145728.0
    9799               396288.0
    9800              2831155.2
    9801              2097152.0
    9802              5242880.0
    9803               384000.0
    9804             11534336.0
    9805               641024.0
    9806             11534336.0
    9807              3040870.4
    9808              6815744.0
    9809              2097152.0
    9810              4404019.2
    9811     Varies with device
    9812              7025459.2
    9813     Varies with device
    9814              9646899.2
    9815              4928307.2
    9816             15728640.0
    9817               164864.0
    9818             25165824.0
    9819              1782579.2
    9820              2831155.2
    9821              6186598.4
    9822              9646899.2
    9823             13631488.0
    9824     Varies with device
    9825              9646899.2
    9826     Varies with device
    9827              6606028.8
    9828              3984588.8
    9829              4404019.2
    9830              5976883.2
    9831             18874368.0
    9832             10380902.4
    9833     Varies with device
    9834             12582912.0
    9835              5557452.8
    9836             10380902.4
    9837             66060288.0
    9838     Varies with device
    9839     Varies with device
    9840     Varies with device
    9841     Varies with device
    9842     Varies with device
    9843     Varies with device
    9844             13631488.0
    9845     Varies with device
    9846             32505856.0
    9847     Varies with device
    9848     Varies with device
    9849             36700160.0
    9850             63963136.0
    9851     Varies with device
    9852     Varies with device
    9853             65011712.0
    9854             97517568.0
    9855             97517568.0
    9856                80896.0
    9857              5242880.0
    9858     Varies with device
    9859             34603008.0
    9860              9122611.2
    9861             63963136.0
    9862     Varies with device
    9863              6396313.6
    9864             27262976.0
    9865             38797312.0
    9866             19922944.0
    9867              2411724.8
    9868              4613734.4
    9869             33554432.0
    9870               900096.0
    9871              5557452.8
    9872             17825792.0
    9873              4404019.2
    9874              5557452.8
    9875              3565158.4
    9876             12582912.0
    9877              1887436.8
    9878             12582912.0
    9879              2621440.0
    9880              4089446.4
    9881              2097152.0
    9882              7549747.2
    9883             13631488.0
    9884             20971520.0
    9885             16777216.0
    9886             23068672.0
    9887              3670016.0
    9888     Varies with device
    9889              2411724.8
    9890             68157440.0
    9891             22020096.0
    9892             22020096.0
    9893             20971520.0
    9894             12582912.0
    9895              2097152.0
    9896              2831155.2
    9897             13631488.0
    9898              7340032.0
    9899              9856614.4
    9900             11534336.0
    9901     Varies with device
    9902              1258291.2
    9903             18874368.0
    9904              7864320.0
    9905              2726297.6
    9906             12582912.0
    9907              3250585.6
    9908                39936.0
    9909              4508876.8
    9910             47185920.0
    9911             15728640.0
    9912              2621440.0
    9913              5662310.4
    9914              5452595.2
    9915              1572864.0
    9916               993280.0
    9917              1468006.4
    9918              2097152.0
    9919             35651584.0
    9920              2202009.6
    9921     Varies with device
    9922             10171187.2
    9923              2306867.2
    9924              1677721.6
    9925             10485760.0
    9926             44040192.0
    9927             13631488.0
    9928              2831155.2
    9929              9122611.2
    9930             33554432.0
    9931             10485760.0
    9932             70254592.0
    9933             17825792.0
    9934             41943040.0
    9935             22020096.0
    9936             15728640.0
    9937             12582912.0
    9938             10171187.2
    9939             32505856.0
    9940              4194304.0
    9941     Varies with device
    9942              9017753.6
    9943             26214400.0
    9944             27262976.0
    9945              6710886.4
    9946             92274688.0
    9947              6081740.8
    9948             42991616.0
    9949              2621440.0
    9950     Varies with device
    9951              6920601.6
    9952             25165824.0
    9953             10276044.8
    9954             33554432.0
    9955             75497472.0
    9956             19922944.0
    9957              8493465.6
    9958              6501171.2
    9959              8808038.4
    9960             19922944.0
    9961              8912896.0
    9962             45088768.0
    9963             24117248.0
    9964             50331648.0
    9965             20971520.0
    9966              5452595.2
    9967              4089446.4
    9968             54525952.0
    9969              1363148.8
    9970             26214400.0
    9971             11534336.0
    9972             16777216.0
    9973              1048576.0
    9974             63963136.0
    9975             15728640.0
    9976     Varies with device
    9977             56623104.0
    9978             19922944.0
    9979             19922944.0
    9980     Varies with device
    9981             34603008.0
    9982     Varies with device
    9983             67108864.0
    9984             75497472.0
    9985             47185920.0
    9986     Varies with device
    9987             28311552.0
    9988             72351744.0
    9989              5138022.4
    9990              4613734.4
    9991              9017753.6
    9992             13631488.0
    9993              9122611.2
    9994             12582912.0
    9995             30408704.0
    9996     Varies with device
    9997              6081740.8
    9998             37748736.0
    9999             17825792.0
    10000            17825792.0
    10001             1992294.4
    10002    Varies with device
    10003            12582912.0
    10004             7444889.6
    10005           100663296.0
    10006            22020096.0
    10007             6815744.0
    10008    Varies with device
    10009            65011712.0
    10010    Varies with device
    10011            85983232.0
    10012    Varies with device
    10013            72351744.0
    10014            23068672.0
    10015            35651584.0
    10016            66060288.0
    10017    Varies with device
    10018    Varies with device
    10019            14680064.0
    10020             1887436.8
    10021            99614720.0
    10022            31457280.0
    10023    Varies with device
    10024            16777216.0
    10025    Varies with device
    10026             4508876.8
    10027            70254592.0
    10028            12582912.0
    10029             8912896.0
    10030            50331648.0
    10031            65011712.0
    10032            99614720.0
    10033             5452595.2
    10034    Varies with device
    10035            24117248.0
    10036             2936012.8
    10037    Varies with device
    10038            13631488.0
    10039             1153433.6
    10040             4823449.6
    10041              174080.0
    10042             1153433.6
    10043             1048576.0
    10044    Varies with device
    10045            11534336.0
    10046              144384.0
    10047              163840.0
    10048    Varies with device
    10049            90177536.0
    10050              147456.0
    10051              144384.0
    10052              146432.0
    10053             6606028.8
    10054            17825792.0
    10055             8283750.4
    10056              194560.0
    10057             6291456.0
    10058             3670016.0
    10059            27262976.0
    10060            97517568.0
    10061            22020096.0
    10062             2411724.8
    10063    Varies with device
    10064            67108864.0
    10065             8493465.6
    10066              385024.0
    10067            13631488.0
    10068             8388608.0
    10069            30408704.0
    10070            44040192.0
    10071            19922944.0
    10072               17408.0
    10073             2621440.0
    10074             3774873.6
    10075             4089446.4
    10076            56623104.0
    10077             7130316.8
    10078            39845888.0
    10079            46137344.0
    10080            31457280.0
    10081             8808038.4
    10082            28311552.0
    10083            10485760.0
    10084            15728640.0
    10085            28311552.0
    10086             5347737.6
    10087             3565158.4
    10088            10485760.0
    10089            19922944.0
    10090            56623104.0
    10091             3250585.6
    10092             2831155.2
    10093             1782579.2
    10094             8808038.4
    10095             1258291.2
    10096            19922944.0
    10097            12582912.0
    10098            46137344.0
    10099            27262976.0
    10100           103809024.0
    10101            19922944.0
    10102            18874368.0
    10103             3460300.8
    10104            31457280.0
    10105            25165824.0
    10106             4718592.0
    10107            15728640.0
    10108            10485760.0
    10109            18874368.0
    10110            13631488.0
    10111             4718592.0
    10112             3460300.8
    10113            22020096.0
    10114            10485760.0
    10115            37748736.0
    10116            34603008.0
    10117             4508876.8
    10118            10066329.6
    10119              671744.0
    10120            63963136.0
    10121            14680064.0
    10122             5662310.4
    10123             1363148.8
    10124    Varies with device
    10125             2936012.8
    10126            49283072.0
    10127    Varies with device
    10128    Varies with device
    10129             7969177.6
    10130             4823449.6
    10131            10485760.0
    10132             4823449.6
    10133            13631488.0
    10134             1258291.2
    10135             3040870.4
    10136            10485760.0
    10137             5033164.8
    10138            37748736.0
    10139             3670016.0
    10140    Varies with device
    10141            52428800.0
    10142               52224.0
    10143            11534336.0
    10144            30408704.0
    10145            48234496.0
    10146            48234496.0
    10147             1468006.4
    10148             1258291.2
    10149              233472.0
    10150    Varies with device
    10151            10276044.8
    10152            40894464.0
    10153              197632.0
    10154            12582912.0
    10155              196608.0
    10156            59768832.0
    10157             2202009.6
    10158             4089446.4
    10159            11534336.0
    10160             2411724.8
    10161             8912896.0
    10162             1468006.4
    10163              484352.0
    10164             2516582.4
    10165             4089446.4
    10166            71303168.0
    10167             1677721.6
    10168    Varies with device
    10169             1887436.8
    10170             4194304.0
    10171             1572864.0
    10172            27262976.0
    10173             3145728.0
    10174            74448896.0
    10175              251904.0
    10176             3774873.6
    10177             1363148.8
    10178            25165824.0
    10179            71303168.0
    10180             1363148.8
    10181            46137344.0
    10182             7340032.0
    10183             5662310.4
    10184             7969177.6
    10185            14680064.0
    10186            74448896.0
    10187             2726297.6
    10188            10485760.0
    10189            50331648.0
    10190            26214400.0
    10191             6606028.8
    10192           100663296.0
    10193            10485760.0
    10194            49283072.0
    10195             3355443.2
    10196    Varies with device
    10197    Varies with device
    10198             6081740.8
    10199            24117248.0
    10200    Varies with device
    10201             1153433.6
    10202             4823449.6
    10203    Varies with device
    10204             6606028.8
    10205    Varies with device
    10206             4508876.8
    10207             5033164.8
    10208             4404019.2
    10209             3774873.6
    10210             4194304.0
    10211             4089446.4
    10212             9122611.2
    10213            10380902.4
    10214             5347737.6
    10215             1887436.8
    10216             1572864.0
    10217             7340032.0
    10218    Varies with device
    10219            18874368.0
    10220             3565158.4
    10221             3460300.8
    10222             1572864.0
    10223             5347737.6
    10224            10485760.0
    10225             1677721.6
    10226             3355443.2
    10227             3145728.0
    10228            10485760.0
    10229             2936012.8
    10230             2621440.0
    10231             1363148.8
    10232            11534336.0
    10233            15728640.0
    10234             3250585.6
    10235             2936012.8
    10236             3879731.2
    10237             6396313.6
    10238             4718592.0
    10239            31457280.0
    10240             6501171.2
    10241            20971520.0
    10242            39845888.0
    10243            20971520.0
    10244             3250585.6
    10245             6710886.4
    10246            74448896.0
    10247            11534336.0
    10248            11534336.0
    10249             2726297.6
    10250            34603008.0
    10251            22020096.0
    10252            42991616.0
    10253            12582912.0
    10254            22020096.0
    10255             2936012.8
    10256             3774873.6
    10257            14680064.0
    10258            30408704.0
    10259            79691776.0
    10260             5976883.2
    10261            17825792.0
    10262            29360128.0
    10263             4823449.6
    10264    Varies with device
    10265            19922944.0
    10266             3250585.6
    10267            82837504.0
    10268            37748736.0
    10269            17825792.0
    10270             8178892.8
    10271            93323264.0
    10272            51380224.0
    10273            12582912.0
    10274             3145728.0
    10275            59768832.0
    10276            16777216.0
    10277            13631488.0
    10278             6186598.4
    10279            41943040.0
    10280            14680064.0
    10281    Varies with device
    10282               74752.0
    10283             3670016.0
    10284            17825792.0
    10285             8283750.4
    10286            19922944.0
    10287            31457280.0
    10288            12582912.0
    10289            44040192.0
    10290             7759462.4
    10291             9542041.6
    10292             7864320.0
    10293             5033164.8
    10294             7759462.4
    10295             5138022.4
    10296             9542041.6
    10297             4404019.2
    10298             2831155.2
    10299            78643200.0
    10300             4089446.4
    10301             2411724.8
    10302            36700160.0
    10303             3670016.0
    10304             9542041.6
    10305             1572864.0
    10306            27262976.0
    10307             7759462.4
    10308             2202009.6
    10309             4928307.2
    10310             3565158.4
    10311             4404019.2
    10312             4823449.6
    10313             7444889.6
    10314            22020096.0
    10315             2831155.2
    10316             4613734.4
    10317            14680064.0
    10318            17825792.0
    10319    Varies with device
    10320             9437184.0
    10321             4718592.0
    10322            22020096.0
    10323            15728640.0
    10324            22020096.0
    10325             8388608.0
    10326            11534336.0
    10327            55574528.0
    10328            22020096.0
    10329             3040870.4
    10330            20971520.0
    10331             3040870.4
    10332            20971520.0
    10333            49283072.0
    10334            12582912.0
    10335            12582912.0
    10336            17825792.0
    10337            10485760.0
    10338            27262976.0
    10339             9646899.2
    10340            10276044.8
    10341            17825792.0
    10342              673792.0
    10343            31457280.0
    10344             1887436.8
    10345           101711872.0
    10346             3040870.4
    10347            15728640.0
    10348             5557452.8
    10349             8703180.8
    10350            20971520.0
    10351             3145728.0
    10352             5872025.6
    10353             8283750.4
    10354            23068672.0
    10355             5242880.0
    10356            11534336.0
    10357            15728640.0
    10358             7864320.0
    10359             4299161.6
    10360            13631488.0
    10361            12582912.0
    10362            62914560.0
    10363            26214400.0
    10364             4823449.6
    10365             1782579.2
    10366             6815744.0
    10367            37748736.0
    10368            17825792.0
    10369             6815744.0
    10370            40894464.0
    10371            48234496.0
    10372            54525952.0
    10373             8074035.2
    10374            31457280.0
    10375            17825792.0
    10376            40894464.0
    10377            31457280.0
    10378            12582912.0
    10379            47185920.0
    10380            30408704.0
    10381            50331648.0
    10382            52428800.0
    10383    Varies with device
    10384            69206016.0
    10385            28311552.0
    10386            53477376.0
    10387            72351744.0
    10388            22020096.0
    10389            14680064.0
    10390            19922944.0
    10391            63963136.0
    10392            29360128.0
    10393            51380224.0
    10394            98566144.0
    10395            24117248.0
    10396            98566144.0
    10397             2621440.0
    10398            13631488.0
    10399            35651584.0
    10400            34603008.0
    10401            16777216.0
    10402             1015808.0
    10403            18874368.0
    10404             8493465.6
    10405            15728640.0
    10406             8598323.2
    10407             8493465.6
    10408            28311552.0
    10409    Varies with device
    10410             4299161.6
    10411            20971520.0
    10412             6291456.0
    10413            17825792.0
    10414             4928307.2
    10415            14680064.0
    10416             6081740.8
    10417            10380902.4
    10418            10380902.4
    10419            51380224.0
    10420            14680064.0
    10421             7969177.6
    10422             8178892.8
    10423            48234496.0
    10424            30408704.0
    10425             3774873.6
    10426            51380224.0
    10427            30408704.0
    10428             8598323.2
    10429            56623104.0
    10430            18874368.0
    10431            58720256.0
    10432            27262976.0
    10433             3460300.8
    10434            41943040.0
    10435            14680064.0
    10436            25165824.0
    10437            38797312.0
    10438    Varies with device
    10439    Varies with device
    10440            23068672.0
    10441            16777216.0
    10442             2097152.0
    10443             3670016.0
    10444             8912896.0
    10445              259072.0
    10446              979968.0
    10447    Varies with device
    10448              430080.0
    10449            13631488.0
    10450             2516582.4
    10451             3565158.4
    10452             5347737.6
    10453    Varies with device
    10454            10171187.2
    10455               73728.0
    10456    Varies with device
    10457             2516582.4
    10458            36700160.0
    10459             4089446.4
    10460               26624.0
    10461               29696.0
    10462             6081740.8
    10463             2936012.8
    10464             6081740.8
    10465             2202009.6
    10466             1468006.4
    10467             4089446.4
    10468             7864320.0
    10469            60817408.0
    10470             4194304.0
    10471              413696.0
    10472             3145728.0
    10473             4299161.6
    10474            14680064.0
    10475             2726297.6
    10476             7969177.6
    10477            11534336.0
    10478             8388608.0
    10479             2411724.8
    10480            51380224.0
    10481            45088768.0
    10482            10485760.0
    10483            16777216.0
    10484            25165824.0
    10485             6501171.2
    10486            28311552.0
    10487            48234496.0
    10488             2516582.4
    10489            37748736.0
    10490            17825792.0
    10491             8388608.0
    10492            38797312.0
    10493            34603008.0
    10494            15728640.0
    10495            15728640.0
    10496            13631488.0
    10497            11534336.0
    10498            26214400.0
    10499            16777216.0
    10500            15728640.0
    10501            15728640.0
    10502    Varies with device
    10503            45088768.0
    10504            45088768.0
    10505            37748736.0
    10506            24117248.0
    10507           103809024.0
    10508            48234496.0
    10509    Varies with device
    10510            20971520.0
    10511            47185920.0
    10512             3040870.4
    10513            22020096.0
    10514            36700160.0
    10515            42991616.0
    10516             2306867.2
    10517             2621440.0
    10518            36700160.0
    10519             1572864.0
    10520            56623104.0
    10521            46137344.0
    10522            28311552.0
    10523             8912896.0
    10524            23068672.0
    10525            27262976.0
    10526            66060288.0
    10527            20971520.0
    10528            27262976.0
    10529             3670016.0
    10530            27262976.0
    10531            10485760.0
    10532             5767168.0
    10533            27262976.0
    10534            27262976.0
    10535            27262976.0
    10536            27262976.0
    10537            27262976.0
    10538            27262976.0
    10539            27262976.0
    10540             2516582.4
    10541            27262976.0
    10542            27262976.0
    10543            27262976.0
    10544            27262976.0
    10545            27262976.0
    10546            27262976.0
    10547            13631488.0
    10548            27262976.0
    10549            26214400.0
    10550            27262976.0
    10551            27262976.0
    10552            15728640.0
    10553            18874368.0
    10554            22020096.0
    10555            27262976.0
    10556            17825792.0
    10557            27262976.0
    10558            12582912.0
    10559            27262976.0
    10560            27262976.0
    10561            27262976.0
    10562            27262976.0
    10563            27262976.0
    10564             2726297.6
    10565            28311552.0
    10566            15728640.0
    10567            27262976.0
    10568            17825792.0
    10569            47185920.0
    10570            25165824.0
    10571            11534336.0
    10572             3250585.6
    10573            31457280.0
    10574             3355443.2
    10575             5872025.6
    10576             7444889.6
    10577             7444889.6
    10578            32505856.0
    10579            35651584.0
    10580             9646899.2
    10581            37748736.0
    10582            15728640.0
    10583             2097152.0
    10584             6396313.6
    10585    Varies with device
    10586            23068672.0
    10587             4613734.4
    10588            90177536.0
    10589            13631488.0
    10590            12582912.0
    10591            42991616.0
    10592             8598323.2
    10593             8388608.0
    10594            15728640.0
    10595              481280.0
    10596             5557452.8
    10597             9856614.4
    10598             2306867.2
    10599            45088768.0
    10600              231424.0
    10601            20971520.0
    10602            22020096.0
    10603            38797312.0
    10604            27262976.0
    10605             3565158.4
    10606            23068672.0
    10607             8178892.8
    10608             3565158.4
    10609            29360128.0
    10610             8178892.8
    10611             2726297.6
    10612             4089446.4
    10613             7969177.6
    10614             8493465.6
    10615             9961472.0
    10616             1887436.8
    10617            16777216.0
    10618             3355443.2
    10619            47185920.0
    10620            14680064.0
    10621            12582912.0
    10622            51380224.0
    10623            13631488.0
    10624            10380902.4
    10625             4089446.4
    10626             7969177.6
    10627            29360128.0
    10628            12582912.0
    10629            72351744.0
    10630            35651584.0
    10631             9646899.2
    10632             5662310.4
    10633             9122611.2
    10634            18874368.0
    10635             5976883.2
    10636            12582912.0
    10637            15728640.0
    10638             1677721.6
    10639            39845888.0
    10640            11534336.0
    10641            14680064.0
    10642    Varies with device
    10643            31457280.0
    10644            13631488.0
    10645    Varies with device
    10646            19922944.0
    10647    Varies with device
    10648            10485760.0
    10649            47185920.0
    10650             5557452.8
    10651            19922944.0
    10652             7340032.0
    10653             5872025.6
    10654            11534336.0
    10655            12582912.0
    10656             9646899.2
    10657            11534336.0
    10658            13631488.0
    10659             3460300.8
    10660            14680064.0
    10661             9227468.8
    10662             7654604.8
    10663            14680064.0
    10664             8912896.0
    10665             3460300.8
    10666             9437184.0
    10667              245760.0
    10668             8598323.2
    10669             8283750.4
    10670             1677721.6
    10671            35651584.0
    10672             2202009.6
    10673            61865984.0
    10674             5452595.2
    10675               91136.0
    10676             3774873.6
    10677              239616.0
    10678              263168.0
    10679    Varies with device
    10680            10276044.8
    10681    Varies with device
    10682            37748736.0
    10683            20971520.0
    10684            29360128.0
    10685             3355443.2
    10686            52428800.0
    10687            42991616.0
    10688              372736.0
    10689             3250585.6
    10690              881664.0
    10691             1677721.6
    10692            15728640.0
    10693            17825792.0
    10694            16777216.0
    10695            15728640.0
    10696            11534336.0
    10697            16777216.0
    10698            15728640.0
    10699            19922944.0
    10700            16777216.0
    10701             2936012.8
    10702             1363148.8
    10703            22020096.0
    10704            18874368.0
    10705             5452595.2
    10706             2097152.0
    10707    Varies with device
    10708            10171187.2
    10709            17825792.0
    10710             4508876.8
    10711            56623104.0
    10712    Varies with device
    10713    Varies with device
    10714             7549747.2
    10715             1468006.4
    10716            11534336.0
    10717            51380224.0
    10718            15728640.0
    10719            26214400.0
    10720             4194304.0
    10721            16777216.0
    10722            11534336.0
    10723            41943040.0
    10724             3879731.2
    10725    Varies with device
    10726            25165824.0
    10727            16777216.0
    10728             8493465.6
    10729             6606028.8
    10730            62914560.0
    10731            48234496.0
    10732              478208.0
    10733             1468006.4
    10734            23068672.0
    10735              160768.0
    10736             2726297.6
    10737            46137344.0
    10738            11534336.0
    10739            40894464.0
    10740             4613734.4
    10741             1782579.2
    10742             8283750.4
    10743             1258291.2
    10744             2097152.0
    10745             6081740.8
    10746            63963136.0
    10747             3984588.8
    10748             3460300.8
    10749            30408704.0
    10750             3460300.8
    10751               45056.0
    10752            27262976.0
    10753            12582912.0
    10754             3460300.8
    10755              692224.0
    10756             2621440.0
    10757            75497472.0
    10758             4508876.8
    10759               68608.0
    10760             2516582.4
    10761             8388608.0
    10762            11534336.0
    10763              565248.0
    10764              906240.0
    10765    Varies with device
    10766             7340032.0
    10767            16777216.0
    10768            25165824.0
    10769            12582912.0
    10770            42991616.0
    10771             2516582.4
    10772             4089446.4
    10773             9332326.4
    10774            37748736.0
    10775             9437184.0
    10776            25165824.0
    10777             2306867.2
    10778            39845888.0
    10779            78643200.0
    10780            52428800.0
    10781            46137344.0
    10782            11534336.0
    10783            75497472.0
    10784            88080384.0
    10785             9961472.0
    10786             2936012.8
    10787            50331648.0
    10788            20971520.0
    10789            50331648.0
    10790            20971520.0
    10791            39845888.0
    10792            16777216.0
    10793            81788928.0
    10794             5976883.2
    10795             4194304.0
    10796             8178892.8
    10797            48234496.0
    10798             1044480.0
    10799             7130316.8
    10800            12582912.0
    10801            19922944.0
    10802            29360128.0
    10803            84934656.0
    10804            17825792.0
    10805            15728640.0
    10806            44040192.0
    10807             4404019.2
    10808             1048576.0
    10809            25165824.0
    10810            22020096.0
    10811             4089446.4
    10812            13631488.0
    10813             2831155.2
    10814            32505856.0
    10815             5138022.4
    10816             7130316.8
    10817             8388608.0
    10818             1572864.0
    10819             3774873.6
    10820             9017753.6
    10821             2621440.0
    10822             3250585.6
    10823             3040870.4
    10824            85983232.0
    10825             8074035.2
    10826    Varies with device
    10827            13631488.0
    10828            13631488.0
    10829             7759462.4
    10830             2411724.8
    10831            10276044.8
    10832              595968.0
    10833              633856.0
    10834             2726297.6
    10835            10066329.6
    10836            55574528.0
    10837             3774873.6
    10838             9961472.0
    10839    Varies with device
    10840            19922944.0
    Name: Size, dtype: object




```python
#rename
df.rename(columns={'Size':"Size_in_bytes"},inplace=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Size_in_bytes</th>
      <th>Installs</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Photo Editor &amp; Candy Camera &amp; Grid &amp; ScrapBook</td>
      <td>ART_AND_DESIGN</td>
      <td>4.1</td>
      <td>159</td>
      <td>19922944.0</td>
      <td>10,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>January 7, 2018</td>
      <td>1.0.0</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Coloring book moana</td>
      <td>ART_AND_DESIGN</td>
      <td>3.9</td>
      <td>967</td>
      <td>14680064.0</td>
      <td>500,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design;Pretend Play</td>
      <td>January 15, 2018</td>
      <td>2.0.0</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U Launcher Lite – FREE Live Cool Themes, Hide ...</td>
      <td>ART_AND_DESIGN</td>
      <td>4.7</td>
      <td>87510</td>
      <td>9122611.2</td>
      <td>5,000,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>August 1, 2018</td>
      <td>1.2.4</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sketch - Draw &amp; Paint</td>
      <td>ART_AND_DESIGN</td>
      <td>4.5</td>
      <td>215644</td>
      <td>26214400.0</td>
      <td>50,000,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Teen</td>
      <td>Art &amp; Design</td>
      <td>June 8, 2018</td>
      <td>Varies with device</td>
      <td>4.2 and up</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pixel Draw - Number Art Coloring Book</td>
      <td>ART_AND_DESIGN</td>
      <td>4.3</td>
      <td>967</td>
      <td>2936012.8</td>
      <td>100,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design;Creativity</td>
      <td>June 20, 2018</td>
      <td>1.1</td>
      <td>4.4 and up</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Size_in_bytes'] = pd.to_numeric(df['Size_in_bytes'], errors='coerce')
df['Size_in_Mb'] = df['Size_in_bytes'] / (1024 * 1024)

```


```python
#let's take care of installs
df['Installs'].unique()
```




    array(['10,000+', '500,000+', '5,000,000+', '50,000,000+', '100,000+',
           '50,000+', '1,000,000+', '10,000,000+', '5,000+', '100,000,000+',
           '1,000,000,000+', '1,000+', '500,000,000+', '50+', '100+', '500+',
           '10+', '1+', '5+', '0+', '0'], dtype=object)




```python
df['Installs'].value_counts()
```




    Installs
    1,000,000+        1579
    10,000,000+       1252
    100,000+          1169
    10,000+           1054
    1,000+             908
    5,000,000+         752
    100+               719
    500,000+           539
    50,000+            479
    5,000+             477
    100,000,000+       409
    10+                386
    500+               330
    50,000,000+        289
    50+                205
    5+                  82
    500,000,000+        72
    1+                  67
    1,000,000,000+      58
    0+                  14
    0                    1
    Name: count, dtype: int64




```python
df['Installs'].isnull().sum()
```




    0



- no missing values

### Observations
- remove + sign
- remove ,
- Convert the column into an integer


```python
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
```


```python
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
```


```python
df['Installs']= df['Installs'].apply(lambda x: int(x))
```


```python
df["Installs"].value_counts()
```




    Installs
    1000000       1579
    10000000      1252
    100000        1169
    10000         1054
    1000           908
    5000000        752
    100            719
    500000         539
    50000          479
    5000           477
    100000000      409
    10             386
    500            330
    50000000       289
    50             205
    5               82
    500000000       72
    1               67
    1000000000      58
    0               15
    Name: count, dtype: int64




```python
df["Installs"].value_counts()
```




    Installs
    1000000       1579
    10000000      1252
    100000        1169
    10000         1054
    1000           908
    5000000        752
    100            719
    500000         539
    50000          479
    5000           477
    100000000      409
    10             386
    500            330
    50000000       289
    50             205
    5               82
    500000000       72
    1               67
    1000000000      58
    0               15
    Name: count, dtype: int64




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Size_in_bytes</th>
      <th>Installs</th>
      <th>Size_in_Mb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9367.000000</td>
      <td>1.084100e+04</td>
      <td>9.146000e+03</td>
      <td>1.084100e+04</td>
      <td>9146.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.191513</td>
      <td>4.441119e+05</td>
      <td>2.255921e+07</td>
      <td>1.546291e+07</td>
      <td>21.514141</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.515735</td>
      <td>2.927629e+06</td>
      <td>2.368595e+07</td>
      <td>8.502557e+07</td>
      <td>22.588679</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000e+00</td>
      <td>8.704000e+03</td>
      <td>0.000000e+00</td>
      <td>0.008301</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>3.800000e+01</td>
      <td>5.138022e+06</td>
      <td>1.000000e+03</td>
      <td>4.900000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.300000</td>
      <td>2.094000e+03</td>
      <td>1.363149e+07</td>
      <td>1.000000e+05</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.500000</td>
      <td>5.476800e+04</td>
      <td>3.145728e+07</td>
      <td>5.000000e+06</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>7.815831e+07</td>
      <td>1.048576e+08</td>
      <td>1.000000e+09</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Price column 


```python
df['Price'].value_counts()
```




    Price
    0          10041
    $0.99        148
    $2.99        129
    $1.99         73
    $4.99         72
    $3.99         63
    $1.49         46
    $5.99         30
    $2.49         26
    $9.99         21
    $6.99         13
    $399.99       12
    $14.99        11
    $4.49          9
    $29.99         7
    $24.99         7
    $3.49          7
    $7.99          7
    $5.49          6
    $19.99         6
    $11.99         5
    $6.49          5
    $12.99         5
    $8.99          5
    $10.00         3
    $16.99         3
    $1.00          3
    $2.00          3
    $13.99         2
    $8.49          2
    $17.99         2
    $1.70          2
    $3.95          2
    $79.99         2
    $7.49          2
    $9.00          2
    $10.99         2
    $39.99         2
    $33.99         2
    $1.96          1
    $19.40         1
    $4.80          1
    $3.28          1
    $4.59          1
    $15.46         1
    $3.04          1
    $4.29          1
    $2.60          1
    $2.59          1
    $3.90          1
    $154.99        1
    $4.60          1
    $28.99         1
    $2.95          1
    $2.90          1
    $1.97          1
    $200.00        1
    $89.99         1
    $2.56          1
    $1.20          1
    $1.26          1
    $30.99         1
    $3.61          1
    $394.99        1
    $3.08          1
    $1.61          1
    $109.99        1
    $46.99         1
    $1.50          1
    $15.99         1
    $74.99         1
    $3.88          1
    $25.99         1
    $400.00        1
    $3.02          1
    $1.76          1
    $4.84          1
    $4.77          1
    $2.50          1
    $1.59          1
    $1.29          1
    $5.00          1
    $299.99        1
    $379.99        1
    $37.99         1
    $18.99         1
    $389.99        1
    $19.90         1
    $1.75          1
    $14.00         1
    $4.85          1
    $1.04          1
    Name: count, dtype: int64



### Observations 
- $ sign


```python
df['Price'].loc[df['Price'].str.contains('\$')].value_counts().sum()
```




    800




```python
 df['Price'].loc[(df['Price'].str.contains('0')) & (~df['Price'].str.contains('\$'))].value_counts().sum()
```




    10041




```python
df['Price'] = df['Price'].apply(lambda x: x.replace('$', '') if '$' in str(x) else x)
```


```python
df['Price'].value_counts()
```




    Price
    0         10041
    0.99        148
    2.99        129
    1.99         73
    4.99         72
    3.99         63
    1.49         46
    5.99         30
    2.49         26
    9.99         21
    6.99         13
    399.99       12
    14.99        11
    4.49          9
    29.99         7
    24.99         7
    3.49          7
    7.99          7
    5.49          6
    19.99         6
    11.99         5
    6.49          5
    12.99         5
    8.99          5
    10.00         3
    16.99         3
    1.00          3
    2.00          3
    13.99         2
    8.49          2
    17.99         2
    1.70          2
    3.95          2
    79.99         2
    7.49          2
    9.00          2
    10.99         2
    39.99         2
    33.99         2
    1.96          1
    19.40         1
    4.80          1
    3.28          1
    4.59          1
    15.46         1
    3.04          1
    4.29          1
    2.60          1
    2.59          1
    3.90          1
    154.99        1
    4.60          1
    28.99         1
    2.95          1
    2.90          1
    1.97          1
    200.00        1
    89.99         1
    2.56          1
    1.20          1
    1.26          1
    30.99         1
    3.61          1
    394.99        1
    3.08          1
    1.61          1
    109.99        1
    46.99         1
    1.50          1
    15.99         1
    74.99         1
    3.88          1
    25.99         1
    400.00        1
    3.02          1
    1.76          1
    4.84          1
    4.77          1
    2.50          1
    1.59          1
    1.29          1
    5.00          1
    299.99        1
    379.99        1
    37.99         1
    18.99         1
    389.99        1
    19.90         1
    1.75          1
    14.00         1
    4.85          1
    1.04          1
    Name: count, dtype: int64




```python
# Now we can conver it into numerics
```


```python
df['Price'] = df['Price'].apply(lambda x: float(x))
```


```python
df['Price']
```




    0          0.00
    1          0.00
    2          0.00
    3          0.00
    4          0.00
    5          0.00
    6          0.00
    7          0.00
    8          0.00
    9          0.00
    10         0.00
    11         0.00
    12         0.00
    13         0.00
    14         0.00
    15         0.00
    16         0.00
    17         0.00
    18         0.00
    19         0.00
    20         0.00
    21         0.00
    22         0.00
    23         0.00
    24         0.00
    25         0.00
    26         0.00
    27         0.00
    28         0.00
    29         0.00
    30         0.00
    31         0.00
    32         0.00
    33         0.00
    34         0.00
    35         0.00
    36         0.00
    37         0.00
    38         0.00
    39         0.00
    40         0.00
    41         0.00
    42         0.00
    43         0.00
    44         0.00
    45         0.00
    46         0.00
    47         0.00
    48         0.00
    49         0.00
    50         0.00
    51         0.00
    52         0.00
    53         0.00
    54         0.00
    55         0.00
    56         0.00
    57         0.00
    58         0.00
    59         0.00
    60         0.00
    61         0.00
    62         0.00
    63         0.00
    64         0.00
    65         0.00
    66         0.00
    67         0.00
    68         0.00
    69         0.00
    70         0.00
    71         0.00
    72         0.00
    73         0.00
    74         0.00
    75         0.00
    76         0.00
    77         0.00
    78         0.00
    79         0.00
    80         0.00
    81         0.00
    82         0.00
    83         0.00
    84         0.00
    85         0.00
    86         0.00
    87         0.00
    88         0.00
    89         0.00
    90         0.00
    91         0.00
    92         0.00
    93         0.00
    94         0.00
    95         0.00
    96         0.00
    97         0.00
    98         0.00
    99         0.00
    100        0.00
    101        0.00
    102        0.00
    103        0.00
    104        0.00
    105        0.00
    106        0.00
    107        0.00
    108        0.00
    109        0.00
    110        0.00
    111        0.00
    112        0.00
    113        0.00
    114        0.00
    115        0.00
    116        0.00
    117        0.00
    118        0.00
    119        0.00
    120        0.00
    121        0.00
    122        0.00
    123        0.00
    124        0.00
    125        0.00
    126        0.00
    127        0.00
    128        0.00
    129        0.00
    130        0.00
    131        0.00
    132        0.00
    133        0.00
    134        0.00
    135        0.00
    136        0.00
    137        0.00
    138        0.00
    139        0.00
    140        0.00
    141        0.00
    142        0.00
    143        0.00
    144        0.00
    145        0.00
    146        0.00
    147        0.00
    148        0.00
    149        0.00
    150        0.00
    151        0.00
    152        0.00
    153        0.00
    154        0.00
    155        0.00
    156        0.00
    157        0.00
    158        0.00
    159        0.00
    160        0.00
    161        0.00
    162        0.00
    163        0.00
    164        0.00
    165        0.00
    166        0.00
    167        0.00
    168        0.00
    169        0.00
    170        0.00
    171        0.00
    172        0.00
    173        0.00
    174        0.00
    175        0.00
    176        0.00
    177        0.00
    178        0.00
    179        0.00
    180        0.00
    181        0.00
    182        0.00
    183        0.00
    184        0.00
    185        0.00
    186        0.00
    187        0.00
    188        0.00
    189        0.00
    190        0.00
    191        0.00
    192        0.00
    193        0.00
    194        0.00
    195        0.00
    196        0.00
    197        0.00
    198        0.00
    199        0.00
    200        0.00
    201        0.00
    202        0.00
    203        0.00
    204        0.00
    205        0.00
    206        0.00
    207        0.00
    208        0.00
    209        0.00
    210        0.00
    211        0.00
    212        0.00
    213        0.00
    214        0.00
    215        0.00
    216        0.00
    217        0.00
    218        0.00
    219        0.00
    220        0.00
    221        0.00
    222        0.00
    223        0.00
    224        0.00
    225        0.00
    226        0.00
    227        0.00
    228        0.00
    229        0.00
    230        0.00
    231        0.00
    232        0.00
    233        0.00
    234        4.99
    235        4.99
    236        0.00
    237        0.00
    238        0.00
    239        0.00
    240        0.00
    241        0.00
    242        0.00
    243        0.00
    244        0.00
    245        0.00
    246        0.00
    247        0.00
    248        0.00
    249        0.00
    250        0.00
    251        0.00
    252        0.00
    253        0.00
    254        0.00
    255        0.00
    256        0.00
    257        0.00
    258        0.00
    259        0.00
    260        0.00
    261        0.00
    262        0.00
    263        0.00
    264        0.00
    265        0.00
    266        0.00
    267        0.00
    268        0.00
    269        0.00
    270        0.00
    271        0.00
    272        0.00
    273        0.00
    274        0.00
    275        0.00
    276        0.00
    277        0.00
    278        0.00
    279        0.00
    280        0.00
    281        0.00
    282        0.00
    283        0.00
    284        0.00
    285        0.00
    286        0.00
    287        0.00
    288        0.00
    289        0.00
    290        4.99
    291        4.99
    292        0.00
    293        0.00
    294        0.00
    295        0.00
    296        0.00
    297        0.00
    298        0.00
    299        0.00
    300        0.00
    301        0.00
    302        0.00
    303        0.00
    304        0.00
    305        0.00
    306        0.00
    307        0.00
    308        0.00
    309        0.00
    310        0.00
    311        0.00
    312        0.00
    313        0.00
    314        0.00
    315        0.00
    316        0.00
    317        0.00
    318        0.00
    319        0.00
    320        0.00
    321        0.00
    322        0.00
    323        0.00
    324        0.00
    325        0.00
    326        0.00
    327        0.00
    328        0.00
    329        0.00
    330        0.00
    331        0.00
    332        0.00
    333        0.00
    334        0.00
    335        0.00
    336        0.00
    337        0.00
    338        0.00
    339        0.00
    340        0.00
    341        0.00
    342        0.00
    343        0.00
    344        0.00
    345        0.00
    346        0.00
    347        0.00
    348        0.00
    349        0.00
    350        0.00
    351        0.00
    352        0.00
    353        0.00
    354        0.00
    355        0.00
    356        0.00
    357        0.00
    358        0.00
    359        0.00
    360        0.00
    361        0.00
    362        0.00
    363        0.00
    364        0.00
    365        0.00
    366        0.00
    367        0.00
    368        0.00
    369        0.00
    370        0.00
    371        0.00
    372        0.00
    373        0.00
    374        0.00
    375        0.00
    376        0.00
    377        0.00
    378        0.00
    379        0.00
    380        0.00
    381        0.00
    382        0.00
    383        0.00
    384        0.00
    385        0.00
    386        0.00
    387        0.00
    388        0.00
    389        0.00
    390        0.00
    391        0.00
    392        0.00
    393        0.00
    394        0.00
    395        0.00
    396        0.00
    397        0.00
    398        0.00
    399        0.00
    400        0.00
    401        0.00
    402        0.00
    403        0.00
    404        0.00
    405        0.00
    406        0.00
    407        0.00
    408        0.00
    409        0.00
    410        0.00
    411        0.00
    412        0.00
    413        0.00
    414        0.00
    415        0.00
    416        0.00
    417        0.00
    418        0.00
    419        0.00
    420        0.00
    421        0.00
    422        0.00
    423        0.00
    424        0.00
    425        0.00
    426        0.00
    427        3.99
    428        0.00
    429        0.00
    430        0.00
    431        0.00
    432        0.00
    433        0.00
    434        0.00
    435        0.00
    436        0.00
    437        0.00
    438        0.00
    439        0.00
    440        0.00
    441        0.00
    442        0.00
    443        0.00
    444        0.00
    445        0.00
    446        0.00
    447        0.00
    448        0.00
    449        0.00
    450        0.00
    451        0.00
    452        0.00
    453        0.00
    454        0.00
    455        0.00
    456        0.00
    457        0.00
    458        0.00
    459        0.00
    460        0.00
    461        0.00
    462        0.00
    463        0.00
    464        0.00
    465        0.00
    466        0.00
    467        0.00
    468        0.00
    469        0.00
    470        0.00
    471        0.00
    472        0.00
    473        0.00
    474        0.00
    475        0.00
    476        3.99
    477        6.99
    478        1.49
    479        2.99
    480        3.99
    481        7.99
    482        0.00
    483        0.00
    484        0.00
    485        0.00
    486        0.00
    487        0.00
    488        0.00
    489        0.00
    490        0.00
    491        0.00
    492        0.00
    493        0.00
    494        0.00
    495        0.00
    496        0.00
    497        0.00
    498        0.00
    499        0.00
    500        0.00
    501        0.00
    502        0.00
    503        0.00
    504        0.00
    505        0.00
    506        0.00
    507        0.00
    508        0.00
    509        0.00
    510        0.00
    511        0.00
    512        0.00
    513        0.00
    514        0.00
    515        0.00
    516        0.00
    517        0.00
    518        0.00
    519        0.00
    520        0.00
    521        0.00
    522        0.00
    523        0.00
    524        0.00
    525        0.00
    526        0.00
    527        0.00
    528        0.00
    529        0.00
    530        0.00
    531        0.00
    532        0.00
    533        0.00
    534        0.00
    535        0.00
    536        0.00
    537        0.00
    538        0.00
    539        0.00
    540        0.00
    541        0.00
    542        0.00
    543        0.00
    544        0.00
    545        0.00
    546        0.00
    547        0.00
    548        0.00
    549        0.00
    550        0.00
    551        0.00
    552        0.00
    553        0.00
    554        0.00
    555        0.00
    556        0.00
    557        0.00
    558        0.00
    559        0.00
    560        0.00
    561        0.00
    562        0.00
    563        0.00
    564        0.00
    565        0.00
    566        0.00
    567        0.00
    568        0.00
    569        0.00
    570        0.00
    571        3.99
    572        0.00
    573        0.00
    574        0.00
    575        0.00
    576        0.00
    577        0.00
    578        0.00
    579        0.00
    580        0.00
    581        0.00
    582        0.00
    583        0.00
    584        0.00
    585        0.00
    586        0.00
    587        0.00
    588        0.00
    589        0.00
    590        0.00
    591        0.00
    592        0.00
    593        0.00
    594        0.00
    595        0.00
    596        0.00
    597        0.00
    598        0.00
    599        0.00
    600        0.00
    601        0.00
    602        0.00
    603        0.00
    604        0.00
    605        0.00
    606        0.00
    607        0.00
    608        0.00
    609        0.00
    610        0.00
    611        0.00
    612        0.00
    613        0.00
    614        0.00
    615        0.00
    616        0.00
    617        0.00
    618        0.00
    619        0.00
    620        0.00
    621        0.00
    622        0.00
    623        0.00
    624        0.00
    625        0.00
    626        0.00
    627        0.00
    628        0.00
    629        0.00
    630        0.00
    631        0.00
    632        0.00
    633        0.00
    634        0.00
    635        0.00
    636        0.00
    637        0.00
    638        0.00
    639        0.00
    640        0.00
    641        0.00
    642        0.00
    643        0.00
    644        0.00
    645        0.00
    646        0.00
    647        0.00
    648        0.00
    649        0.00
    650        0.00
    651        0.00
    652        0.00
    653        0.00
    654        0.00
    655        0.00
    656        0.00
    657        0.00
    658        0.00
    659        0.00
    660        0.00
    661        0.00
    662        0.00
    663        0.00
    664        0.00
    665        0.00
    666        0.00
    667        0.00
    668        0.00
    669        0.00
    670        0.00
    671        0.00
    672        0.00
    673        0.00
    674        0.00
    675        0.00
    676        0.00
    677        0.00
    678        0.00
    679        0.00
    680        0.00
    681        0.00
    682        0.00
    683        0.00
    684        0.00
    685        0.00
    686        0.00
    687        0.00
    688        0.00
    689        0.00
    690        0.00
    691        0.00
    692        0.00
    693        0.00
    694        0.00
    695        0.00
    696        0.00
    697        0.00
    698        0.00
    699        0.00
    700        0.00
    701        0.00
    702        0.00
    703        0.00
    704        0.00
    705        0.00
    706        0.00
    707        0.00
    708        0.00
    709        0.00
    710        0.00
    711        0.00
    712        0.00
    713        0.00
    714        0.00
    715        0.00
    716        0.00
    717        0.00
    718        0.00
    719        0.00
    720        0.00
    721        0.00
    722        0.00
    723        0.00
    724        0.00
    725        0.00
    726        0.00
    727        0.00
    728        0.00
    729        0.00
    730        0.00
    731        0.00
    732        0.00
    733        0.00
    734        0.00
    735        0.00
    736        0.00
    737        0.00
    738        0.00
    739        0.00
    740        0.00
    741        0.00
    742        0.00
    743        0.00
    744        0.00
    745        0.00
    746        0.00
    747        0.00
    748        0.00
    749        0.00
    750        0.00
    751        0.00
    752        0.00
    753        0.00
    754        0.00
    755        0.00
    756        0.00
    757        0.00
    758        0.00
    759        0.00
    760        0.00
    761        0.00
    762        0.00
    763        0.00
    764        0.00
    765        0.00
    766        0.00
    767        0.00
    768        0.00
    769        0.00
    770        0.00
    771        0.00
    772        0.00
    773        0.00
    774        0.00
    775        0.00
    776        0.00
    777        0.00
    778        0.00
    779        0.00
    780        0.00
    781        0.00
    782        0.00
    783        0.00
    784        0.00
    785        0.00
    786        0.00
    787        0.00
    788        0.00
    789        0.00
    790        0.00
    791        0.00
    792        0.00
    793        0.00
    794        0.00
    795        0.00
    796        0.00
    797        0.00
    798        0.00
    799        0.00
    800        0.00
    801        0.00
    802        0.00
    803        0.00
    804        0.00
    805        0.00
    806        0.00
    807        0.00
    808        0.00
    809        0.00
    810        0.00
    811        0.00
    812        0.00
    813        0.00
    814        0.00
    815        0.00
    816        0.00
    817        0.00
    818        0.00
    819        0.00
    820        0.00
    821        0.00
    822        0.00
    823        0.00
    824        0.00
    825        0.00
    826        0.00
    827        0.00
    828        0.00
    829        0.00
    830        0.00
    831        0.00
    832        0.00
    833        0.00
    834        0.00
    835        0.00
    836        0.00
    837        0.00
    838        0.00
    839        0.00
    840        0.00
    841        0.00
    842        0.00
    843        0.00
    844        0.00
    845        0.00
    846        0.00
    847        0.00
    848        0.00
    849        0.00
    850        0.00
    851        3.99
    852        5.99
    853        3.99
    854        3.99
    855        0.00
    856        0.00
    857        0.00
    858        0.00
    859        0.00
    860        0.00
    861        0.00
    862        0.00
    863        0.00
    864        0.00
    865        0.00
    866        0.00
    867        0.00
    868        0.00
    869        0.00
    870        0.00
    871        0.00
    872        0.00
    873        0.00
    874        0.00
    875        0.00
    876        0.00
    877        0.00
    878        0.00
    879        0.00
    880        0.00
    881        0.00
    882        0.00
    883        0.00
    884        0.00
    885        0.00
    886        0.00
    887        0.00
    888        0.00
    889        0.00
    890        0.00
    891        0.00
    892        0.00
    893        0.00
    894        0.00
    895        0.00
    896        0.00
    897        0.00
    898        0.00
    899        0.00
    900        0.00
    901        0.00
    902        0.00
    903        0.00
    904        0.00
    905        0.00
    906        0.00
    907        0.00
    908        0.00
    909        0.00
    910        0.00
    911        0.00
    912        0.00
    913        0.00
    914        0.00
    915        0.00
    916        0.00
    917        0.00
    918        0.00
    919        0.00
    920        0.00
    921        0.00
    922        0.00
    923        0.00
    924        0.00
    925        0.00
    926        0.00
    927        0.00
    928        0.00
    929        0.00
    930        0.00
    931        0.00
    932        0.00
    933        0.00
    934        0.00
    935        0.00
    936        0.00
    937        0.00
    938        0.00
    939        0.00
    940        0.00
    941        0.00
    942        0.00
    943        0.00
    944        0.00
    945        0.00
    946        0.00
    947        0.00
    948        0.00
    949        0.00
    950        0.00
    951        0.00
    952        0.00
    953        0.00
    954        0.00
    955        0.00
    956        0.00
    957        0.00
    958        0.00
    959        0.00
    960        0.00
    961        0.00
    962        0.00
    963        0.00
    964        0.00
    965        0.00
    966        0.00
    967        0.00
    968        0.00
    969        0.00
    970        0.00
    971        0.00
    972        0.00
    973        0.00
    974        0.00
    975        0.00
    976        0.00
    977        0.00
    978        0.00
    979        0.00
    980        0.00
    981        0.00
    982        0.00
    983        0.00
    984        0.00
    985        0.00
    986        0.00
    987        0.00
    988        0.00
    989        0.00
    990        0.00
    991        0.00
    992        0.00
    993        0.00
    994        0.00
    995        4.99
    996        0.00
    997        0.00
    998        0.00
    999        0.00
    1000       0.00
    1001       2.99
    1002       0.00
    1003       0.00
    1004       0.00
    1005       0.00
    1006       0.00
    1007       0.00
    1008       0.00
    1009       0.00
    1010       0.00
    1011       0.00
    1012       0.00
    1013       0.00
    1014       0.00
    1015       0.00
    1016       0.00
    1017       0.00
    1018       0.00
    1019       0.00
    1020       0.00
    1021       0.00
    1022       0.00
    1023       0.00
    1024       0.00
    1025       0.00
    1026       0.00
    1027       0.00
    1028       0.00
    1029       0.00
    1030       0.00
    1031       0.00
    1032       0.00
    1033       0.00
    1034       0.00
    1035       0.00
    1036       0.00
    1037       0.00
    1038       0.00
    1039       0.00
    1040       0.00
    1041       0.00
    1042       0.00
    1043       0.00
    1044       0.00
    1045       0.00
    1046       0.00
    1047       0.00
    1048       0.00
    1049       0.00
    1050       0.00
    1051       0.00
    1052       0.00
    1053       0.00
    1054       0.00
    1055       0.00
    1056       0.00
    1057       0.00
    1058       0.00
    1059       0.00
    1060       0.00
    1061       0.00
    1062       0.00
    1063       0.00
    1064       0.00
    1065       0.00
    1066       0.00
    1067       0.00
    1068       0.00
    1069       0.00
    1070       0.00
    1071       0.00
    1072       0.00
    1073       0.00
    1074       0.00
    1075       0.00
    1076       0.00
    1077       0.00
    1078       0.00
    1079       0.00
    1080       0.00
    1081       0.00
    1082       0.00
    1083       0.00
    1084       0.00
    1085       0.00
    1086       0.00
    1087       0.00
    1088       0.00
    1089       0.00
    1090       0.00
    1091       0.00
    1092       0.00
    1093       0.00
    1094       0.00
    1095       0.00
    1096       0.00
    1097       0.00
    1098       0.00
    1099       0.00
    1100       0.00
    1101       0.00
    1102       0.00
    1103       0.00
    1104       0.00
    1105       0.00
    1106       0.00
    1107       0.00
    1108       0.00
    1109       0.00
    1110       0.00
    1111       0.00
    1112       0.00
    1113       0.00
    1114       0.00
    1115       0.00
    1116       0.00
    1117       0.00
    1118       0.00
    1119       0.00
    1120       0.00
    1121       0.00
    1122       0.00
    1123       0.00
    1124       0.00
    1125       0.00
    1126       0.00
    1127       0.00
    1128       0.00
    1129       0.00
    1130       0.00
    1131       0.00
    1132       0.00
    1133       0.00
    1134       0.00
    1135       0.00
    1136       0.00
    1137       0.00
    1138       0.00
    1139       0.00
    1140       0.00
    1141       0.00
    1142       0.00
    1143       0.00
    1144       0.00
    1145       0.00
    1146       0.00
    1147       0.00
    1148       0.00
    1149       0.00
    1150       0.00
    1151       0.00
    1152       0.00
    1153       0.00
    1154       0.00
    1155       0.00
    1156       0.00
    1157       0.00
    1158       0.00
    1159       0.00
    1160       0.00
    1161       0.00
    1162       0.00
    1163       0.00
    1164       0.00
    1165       0.00
    1166       0.00
    1167       0.00
    1168       0.00
    1169       0.00
    1170       0.00
    1171       0.00
    1172       0.00
    1173       0.00
    1174       0.00
    1175       0.00
    1176       0.00
    1177       0.00
    1178       0.00
    1179       0.00
    1180       0.00
    1181       0.00
    1182       0.00
    1183       0.00
    1184       0.00
    1185       0.00
    1186       0.00
    1187       0.00
    1188       0.00
    1189       0.00
    1190       0.00
    1191       0.00
    1192       0.00
    1193       0.00
    1194       0.00
    1195       0.00
    1196       0.00
    1197       0.00
    1198       0.00
    1199       0.00
    1200       0.00
    1201       0.00
    1202       0.00
    1203       0.00
    1204       0.00
    1205       0.00
    1206       0.00
    1207       0.00
    1208       0.00
    1209       0.00
    1210       0.00
    1211       0.00
    1212       0.00
    1213       0.00
    1214       0.00
    1215       0.00
    1216       0.00
    1217       0.00
    1218       0.00
    1219       0.00
    1220       0.00
    1221       0.00
    1222       0.00
    1223       0.00
    1224       0.00
    1225       0.00
    1226       0.00
    1227       3.49
    1228       4.99
    1229       0.00
    1230       0.00
    1231       0.00
    1232       0.00
    1233       0.00
    1234       0.00
    1235       0.00
    1236       0.00
    1237       0.00
    1238       0.00
    1239       0.00
    1240       0.00
    1241       0.00
    1242       0.00
    1243       0.00
    1244       0.00
    1245       0.00
    1246       0.00
    1247       0.00
    1248       0.00
    1249       0.00
    1250       0.00
    1251       0.00
    1252       0.00
    1253       0.00
    1254       0.00
    1255       0.00
    1256       0.00
    1257       0.00
    1258       0.00
    1259       0.00
    1260       0.00
    1261       0.00
    1262       0.00
    1263       0.00
    1264       0.00
    1265       0.00
    1266       0.00
    1267       0.00
    1268       0.00
    1269       0.00
    1270       0.00
    1271       0.00
    1272       0.00
    1273       0.00
    1274       0.00
    1275       0.00
    1276       0.00
    1277       0.00
    1278       0.00
    1279       0.00
    1280       0.00
    1281       0.00
    1282       0.00
    1283       0.00
    1284       0.00
    1285       0.00
    1286       0.00
    1287       0.00
    1288       0.00
    1289       0.00
    1290       0.00
    1291       0.00
    1292       0.00
    1293       0.00
    1294       0.00
    1295       0.00
    1296       0.00
    1297       0.00
    1298       0.00
    1299       0.00
    1300       0.00
    1301       0.00
    1302       0.00
    1303       0.00
    1304       0.00
    1305       0.00
    1306       0.00
    1307       0.00
    1308       0.00
    1309       0.00
    1310       0.00
    1311       0.00
    1312       0.00
    1313       0.00
    1314       0.00
    1315       0.00
    1316       0.00
    1317       0.00
    1318       0.00
    1319       0.00
    1320       0.00
    1321       0.00
    1322       0.00
    1323       0.00
    1324       0.00
    1325       0.00
    1326       0.00
    1327       2.99
    1328       0.00
    1329       0.00
    1330       0.00
    1331       0.00
    1332       0.00
    1333       0.00
    1334       0.00
    1335       3.99
    1336       0.00
    1337       0.00
    1338       0.00
    1339       0.00
    1340       0.00
    1341       2.99
    1342       0.00
    1343       0.00
    1344       0.00
    1345       0.00
    1346       0.00
    1347       2.99
    1348       0.00
    1349       0.00
    1350       0.00
    1351       0.00
    1352       0.00
    1353       0.00
    1354       0.00
    1355       0.00
    1356       0.00
    1357       0.00
    1358       0.00
    1359       0.00
    1360       0.00
    1361       0.00
    1362       0.00
    1363       0.00
    1364       0.00
    1365       0.00
    1366       0.00
    1367       0.00
    1368       0.00
    1369       0.00
    1370       0.00
    1371       0.00
    1372       0.00
    1373       0.00
    1374       0.00
    1375       0.00
    1376       0.00
    1377       0.00
    1378       0.00
    1379       0.00
    1380       0.00
    1381       0.00
    1382       0.00
    1383       0.00
    1384       0.00
    1385       0.00
    1386       0.00
    1387       0.00
    1388       0.00
    1389       0.00
    1390       0.00
    1391       0.00
    1392       0.00
    1393       0.00
    1394       0.00
    1395       0.00
    1396       0.00
    1397       0.00
    1398       0.00
    1399       0.00
    1400       0.00
    1401       0.00
    1402       0.00
    1403       0.00
    1404       0.00
    1405       0.00
    1406       0.00
    1407       0.00
    1408       0.00
    1409       0.00
    1410       0.00
    1411       0.00
    1412       0.00
    1413       0.00
    1414       0.00
    1415       0.00
    1416       0.00
    1417       0.00
    1418       0.00
    1419       0.00
    1420       0.00
    1421       0.00
    1422       0.00
    1423       0.00
    1424       0.00
    1425       0.00
    1426       0.00
    1427       0.00
    1428       0.00
    1429       0.00
    1430       0.00
    1431       0.00
    1432       0.00
    1433       0.00
    1434       0.00
    1435       0.00
    1436       0.00
    1437       0.00
    1438       0.00
    1439       0.00
    1440       0.00
    1441       0.00
    1442       0.00
    1443       0.00
    1444       0.00
    1445       0.00
    1446       0.00
    1447       0.00
    1448       0.00
    1449       0.00
    1450       0.00
    1451       0.00
    1452       0.00
    1453       0.00
    1454       0.00
    1455       0.00
    1456       0.00
    1457       0.00
    1458       0.00
    1459       0.00
    1460       0.00
    1461       0.00
    1462       0.00
    1463       0.00
    1464       0.00
    1465       0.00
    1466       0.00
    1467       0.00
    1468       0.00
    1469       0.00
    1470       0.00
    1471       0.00
    1472       0.00
    1473       0.00
    1474       0.00
    1475       0.00
    1476       0.00
    1477       0.00
    1478       0.00
    1479       0.00
    1480       0.00
    1481       0.00
    1482       0.00
    1483       0.00
    1484       0.00
    1485       0.00
    1486       0.00
    1487       0.00
    1488       0.00
    1489       0.00
    1490       0.00
    1491       0.00
    1492       0.00
    1493       0.00
    1494       0.00
    1495       0.00
    1496       0.00
    1497       0.00
    1498       0.00
    1499       0.00
    1500       0.00
    1501       0.00
    1502       0.00
    1503       0.00
    1504       0.00
    1505       0.00
    1506       0.00
    1507       0.00
    1508       0.00
    1509       0.00
    1510       0.00
    1511       0.00
    1512       0.00
    1513       0.00
    1514       0.00
    1515       0.00
    1516       0.00
    1517       0.00
    1518       0.00
    1519       0.00
    1520       0.00
    1521       0.00
    1522       0.00
    1523       0.00
    1524       0.00
    1525       0.00
    1526       0.00
    1527       0.00
    1528       0.00
    1529       0.00
    1530       0.00
    1531       0.00
    1532       0.00
    1533       0.00
    1534       0.00
    1535       0.00
    1536       0.00
    1537       0.00
    1538       0.00
    1539       0.00
    1540       0.00
    1541       0.00
    1542       0.00
    1543       0.00
    1544       0.00
    1545       0.00
    1546       0.00
    1547       0.00
    1548       0.00
    1549       0.00
    1550       0.00
    1551       0.00
    1552       0.00
    1553       0.00
    1554       0.00
    1555       0.00
    1556       0.00
    1557       0.00
    1558       0.00
    1559       0.00
    1560       0.00
    1561       0.00
    1562       0.00
    1563       0.00
    1564       0.00
    1565       0.00
    1566       0.00
    1567       0.00
    1568       0.00
    1569       0.00
    1570       0.00
    1571       0.00
    1572       0.00
    1573       0.00
    1574       0.00
    1575       0.00
    1576       0.00
    1577       0.00
    1578       0.00
    1579       0.00
    1580       0.00
    1581       0.00
    1582       0.00
    1583       0.00
    1584       0.00
    1585       0.00
    1586       0.00
    1587       0.00
    1588       0.00
    1589       0.00
    1590       0.00
    1591       0.00
    1592       0.00
    1593       0.00
    1594       0.00
    1595       0.00
    1596       0.00
    1597       0.00
    1598       0.00
    1599       0.00
    1600       0.00
    1601       0.00
    1602       0.00
    1603       0.00
    1604       0.00
    1605       0.00
    1606       0.00
    1607       0.00
    1608       0.00
    1609       0.00
    1610       0.00
    1611       0.00
    1612       0.00
    1613       0.00
    1614       0.00
    1615       0.00
    1616       0.00
    1617       0.00
    1618       0.00
    1619       0.00
    1620       0.00
    1621       0.00
    1622       0.00
    1623       0.00
    1624       0.00
    1625       0.00
    1626       0.00
    1627       0.00
    1628       0.00
    1629       0.00
    1630       0.00
    1631       0.00
    1632       0.00
    1633       0.00
    1634       0.00
    1635       0.00
    1636       0.00
    1637       0.00
    1638       0.00
    1639       0.00
    1640       0.00
    1641       0.00
    1642       0.00
    1643       0.00
    1644       0.00
    1645       0.00
    1646       0.00
    1647       0.00
    1648       0.00
    1649       0.00
    1650       0.00
    1651       0.00
    1652       0.00
    1653       0.00
    1654       0.00
    1655       0.00
    1656       0.00
    1657       0.00
    1658       0.00
    1659       0.00
    1660       0.00
    1661       0.00
    1662       0.00
    1663       0.00
    1664       0.00
    1665       0.00
    1666       0.00
    1667       0.00
    1668       0.00
    1669       0.00
    1670       0.00
    1671       0.00
    1672       0.00
    1673       0.00
    1674       0.00
    1675       0.00
    1676       0.00
    1677       0.00
    1678       0.00
    1679       0.00
    1680       0.00
    1681       0.00
    1682       0.00
    1683       0.00
    1684       0.00
    1685       0.00
    1686       0.00
    1687       0.00
    1688       0.00
    1689       0.00
    1690       0.00
    1691       0.00
    1692       0.00
    1693       0.00
    1694       0.00
    1695       0.00
    1696       0.00
    1697       0.00
    1698       0.00
    1699       0.00
    1700       0.00
    1701       0.00
    1702       0.00
    1703       0.00
    1704       0.00
    1705       0.00
    1706       0.00
    1707       0.00
    1708       0.00
    1709       0.00
    1710       0.00
    1711       0.00
    1712       0.00
    1713       0.00
    1714       0.00
    1715       0.00
    1716       0.00
    1717       0.00
    1718       0.00
    1719       0.00
    1720       0.00
    1721       0.00
    1722       0.00
    1723       0.00
    1724       0.00
    1725       0.00
    1726       0.00
    1727       0.00
    1728       0.00
    1729       0.00
    1730       0.00
    1731       0.00
    1732       0.00
    1733       0.00
    1734       0.00
    1735       0.00
    1736       0.00
    1737       0.00
    1738       0.00
    1739       0.00
    1740       0.00
    1741       0.00
    1742       0.00
    1743       0.00
    1744       0.00
    1745       0.00
    1746       0.00
    1747       0.00
    1748       0.00
    1749       0.00
    1750       0.00
    1751       0.00
    1752       0.00
    1753       0.00
    1754       0.00
    1755       0.00
    1756       0.00
    1757       0.00
    1758       0.00
    1759       0.00
    1760       0.00
    1761       0.00
    1762       0.00
    1763       0.00
    1764       0.00
    1765       0.00
    1766       0.00
    1767       0.00
    1768       0.00
    1769       0.00
    1770       0.00
    1771       0.00
    1772       0.00
    1773       0.00
    1774       0.00
    1775       0.00
    1776       0.00
    1777       0.00
    1778       0.00
    1779       0.00
    1780       0.00
    1781       0.00
    1782       0.00
    1783       0.00
    1784       0.00
    1785       0.00
    1786       0.00
    1787       0.00
    1788       0.00
    1789       0.00
    1790       0.00
    1791       0.00
    1792       0.00
    1793       0.00
    1794       0.00
    1795       0.00
    1796       0.00
    1797       0.00
    1798       0.00
    1799       0.00
    1800       0.00
    1801       0.00
    1802       0.00
    1803       0.00
    1804       0.00
    1805       0.00
    1806       0.00
    1807       0.00
    1808       0.00
    1809       0.00
    1810       0.00
    1811       0.00
    1812       0.00
    1813       0.00
    1814       0.00
    1815       0.00
    1816       0.00
    1817       0.00
    1818       0.00
    1819       0.00
    1820       0.00
    1821       0.00
    1822       0.00
    1823       0.00
    1824       0.00
    1825       0.00
    1826       0.00
    1827       0.00
    1828       0.00
    1829       0.00
    1830       0.00
    1831       2.99
    1832       1.99
    1833       4.99
    1834       4.99
    1835       4.99
    1836       5.99
    1837       6.99
    1838       9.99
    1839       4.99
    1840       0.00
    1841       0.00
    1842       0.00
    1843       0.00
    1844       0.00
    1845       0.00
    1846       0.00
    1847       0.00
    1848       0.00
    1849       0.00
    1850       0.00
    1851       0.00
    1852       0.00
    1853       0.00
    1854       0.00
    1855       0.00
    1856       0.00
    1857       0.00
    1858       0.00
    1859       0.00
    1860       0.00
    1861       0.00
    1862       0.00
    1863       0.00
    1864       0.00
    1865       0.00
    1866       0.00
    1867       0.00
    1868       0.00
    1869       0.00
    1870       0.00
    1871       0.00
    1872       0.00
    1873       0.00
    1874       0.00
    1875       0.00
    1876       0.00
    1877       0.00
    1878       0.00
    1879       0.00
    1880       0.00
    1881       0.00
    1882       0.00
    1883       0.00
    1884       0.00
    1885       0.00
    1886       0.00
    1887       0.00
    1888       0.00
    1889       0.00
    1890       0.00
    1891       0.00
    1892       0.00
    1893       0.00
    1894       0.00
    1895       0.00
    1896       0.00
    1897       0.00
    1898       0.00
    1899       0.00
    1900       0.00
    1901       0.00
    1902       0.00
    1903       0.00
    1904       0.00
    1905       0.00
    1906       0.00
    1907       0.00
    1908       0.00
    1909       0.00
    1910       0.00
    1911       0.00
    1912       0.00
    1913       0.00
    1914       0.00
    1915       0.00
    1916       0.00
    1917       0.00
    1918       0.00
    1919       0.00
    1920       0.00
    1921       0.00
    1922       0.00
    1923       0.00
    1924       0.00
    1925       0.00
    1926       0.00
    1927       0.00
    1928       0.00
    1929       0.00
    1930       0.00
    1931       0.00
    1932       0.00
    1933       0.00
    1934       0.00
    1935       0.00
    1936       0.00
    1937       0.00
    1938       0.00
    1939       0.00
    1940       0.00
    1941       0.00
    1942       0.00
    1943       0.00
    1944       0.00
    1945       0.00
    1946       0.00
    1947       0.00
    1948       0.00
    1949       0.00
    1950       0.00
    1951       0.00
    1952       0.00
    1953       0.00
    1954       0.00
    1955       0.00
    1956       0.00
    1957       0.00
    1958       0.00
    1959       0.00
    1960       0.00
    1961       0.00
    1962       0.00
    1963       0.00
    1964       0.00
    1965       0.00
    1966       0.00
    1967       0.00
    1968       0.00
    1969       0.00
    1970       0.00
    1971       0.00
    1972       0.00
    1973       0.00
    1974       0.00
    1975       0.00
    1976       0.00
    1977       0.00
    1978       0.00
    1979       0.00
    1980       0.00
    1981       0.00
    1982       0.00
    1983       0.00
    1984       0.00
    1985       0.00
    1986       0.00
    1987       0.00
    1988       0.00
    1989       0.00
    1990       0.00
    1991       0.00
    1992       0.00
    1993       0.00
    1994       0.00
    1995       0.00
    1996       0.00
    1997       0.00
    1998       0.00
    1999       0.00
    2000       0.00
    2001       0.00
    2002       0.00
    2003       0.00
    2004       0.00
    2005       0.00
    2006       0.00
    2007       0.00
    2008       0.00
    2009       0.00
    2010       0.00
    2011       0.00
    2012       0.00
    2013       0.00
    2014       0.00
    2015       0.00
    2016       0.00
    2017       0.00
    2018       0.00
    2019       0.00
    2020       0.00
    2021       0.00
    2022       0.00
    2023       0.00
    2024       0.00
    2025       0.00
    2026       0.00
    2027       0.00
    2028       0.00
    2029       0.00
    2030       0.00
    2031       0.00
    2032       0.00
    2033       0.00
    2034       0.00
    2035       0.00
    2036       0.00
    2037       0.00
    2038       0.00
    2039       0.00
    2040       0.00
    2041       0.00
    2042       0.00
    2043       0.00
    2044       0.00
    2045       0.00
    2046       0.00
    2047       0.00
    2048       0.00
    2049       0.00
    2050       0.00
    2051       0.00
    2052       0.00
    2053       0.00
    2054       0.00
    2055       0.00
    2056       0.00
    2057       0.00
    2058       0.00
    2059       0.00
    2060       0.00
    2061       0.00
    2062       3.99
    2063       0.00
    2064       0.00
    2065       0.00
    2066       0.00
    2067       0.00
    2068       0.00
    2069       0.00
    2070       0.00
    2071       0.00
    2072       0.00
    2073       0.00
    2074       0.00
    2075       0.00
    2076       0.00
    2077       0.00
    2078       0.00
    2079       0.00
    2080       0.00
    2081       0.00
    2082       0.00
    2083       0.00
    2084       0.00
    2085       2.99
    2086       3.99
    2087       2.99
    2088       0.00
    2089       0.00
    2090       0.00
    2091       0.00
    2092       0.00
    2093       0.00
    2094       0.00
    2095       0.00
    2096       0.00
    2097       0.00
    2098       0.00
    2099       0.00
    2100       0.00
    2101       0.00
    2102       0.00
    2103       0.00
    2104       0.00
    2105       0.00
    2106       0.00
    2107       0.00
    2108       0.00
    2109       0.00
    2110       0.00
    2111       0.00
    2112       0.00
    2113       0.00
    2114       0.00
    2115       0.00
    2116       0.00
    2117       0.00
    2118       0.00
    2119       0.00
    2120       0.00
    2121       0.00
    2122       0.00
    2123       0.00
    2124       0.00
    2125       0.00
    2126       0.00
    2127       0.00
    2128       0.00
    2129       0.00
    2130       0.00
    2131       0.00
    2132       0.00
    2133       0.00
    2134       0.00
    2135       0.00
    2136       0.00
    2137       0.00
    2138       0.00
    2139       0.00
    2140       0.00
    2141       0.00
    2142       0.00
    2143       0.00
    2144       0.00
    2145       0.00
    2146       0.00
    2147       0.00
    2148       0.00
    2149       0.00
    2150       3.99
    2151       3.99
    2152       4.99
    2153       3.99
    2154       2.99
    2155       0.00
    2156       0.00
    2157       0.00
    2158       0.00
    2159       0.00
    2160       0.00
    2161       0.00
    2162       0.00
    2163       0.00
    2164       0.00
    2165       0.00
    2166       0.00
    2167       0.00
    2168       7.49
    2169       0.00
    2170       2.99
    2171       0.99
    2172       0.99
    2173       0.99
    2174       4.99
    2175       2.99
    2176       4.99
    2177       2.99
    2178       4.99
    2179       4.99
    2180       0.00
    2181       0.00
    2182       0.00
    2183       0.00
    2184       0.00
    2185       0.00
    2186       0.00
    2187       0.00
    2188       0.00
    2189       2.99
    2190       2.99
    2191       3.99
    2192       3.99
    2193       0.00
    2194       0.00
    2195       0.00
    2196       0.00
    2197       0.00
    2198       0.00
    2199       0.00
    2200       0.00
    2201       0.00
    2202       2.99
    2203       2.99
    2204       3.99
    2205       3.99
    2206       0.00
    2207       0.00
    2208       0.00
    2209       0.00
    2210       0.00
    2211       0.00
    2212       0.00
    2213       0.00
    2214       0.00
    2215       0.00
    2216       0.00
    2217       0.00
    2218       0.00
    2219       0.00
    2220       0.00
    2221       0.00
    2222       0.00
    2223       0.00
    2224       0.00
    2225       0.00
    2226       0.00
    2227       0.00
    2228       0.00
    2229       0.00
    2230       0.00
    2231       0.00
    2232       0.00
    2233       0.00
    2234       0.00
    2235       0.00
    2236       0.00
    2237       0.00
    2238       0.00
    2239       0.00
    2240       0.00
    2241       6.99
    2242       2.99
    2243       9.00
    2244       0.99
    2245       5.49
    2246       9.99
    2247       6.99
    2248      10.00
    2249       3.99
    2250       5.99
    2251      24.99
    2252      11.99
    2253      79.99
    2254      11.99
    2255       2.99
    2256      16.99
    2257       3.99
    2258       2.99
    2259       9.99
    2260       3.99
    2261      14.99
    2262       2.99
    2263       3.99
    2264       2.99
    2265       1.00
    2266      29.99
    2267       2.99
    2268       2.99
    2269      12.99
    2270       4.99
    2271       2.99
    2272      14.99
    2273       5.99
    2274       3.49
    2275       0.99
    2276       2.49
    2277      24.99
    2278      10.99
    2279       1.99
    2280      24.99
    2281       4.99
    2282       3.99
    2283       2.99
    2284       7.49
    2285       1.50
    2286       2.99
    2287       3.99
    2288       1.99
    2289       9.99
    2290       3.99
    2291       3.99
    2292       3.99
    2293       7.99
    2294      14.99
    2295       9.99
    2296       3.99
    2297      19.99
    2298      29.99
    2299      15.99
    2300       0.99
    2301      33.99
    2302       0.99
    2303       0.00
    2304       0.00
    2305       0.00
    2306       0.00
    2307       0.00
    2308       0.00
    2309       0.00
    2310       0.00
    2311       0.00
    2312       0.00
    2313       0.00
    2314       0.00
    2315       0.00
    2316       0.00
    2317       0.00
    2318       0.00
    2319       0.00
    2320       0.00
    2321       0.00
    2322       0.00
    2323       0.00
    2324       0.00
    2325       0.00
    2326       0.00
    2327       0.00
    2328       0.00
    2329       0.00
    2330       0.00
    2331       0.00
    2332       0.00
    2333       0.00
    2334       0.00
    2335       0.00
    2336       0.00
    2337       0.00
    2338       0.00
    2339       0.00
    2340       0.00
    2341       0.00
    2342       0.00
    2343       0.00
    2344       0.00
    2345       0.00
    2346       0.00
    2347       0.00
    2348       0.00
    2349       0.00
    2350       0.00
    2351       0.00
    2352       0.00
    2353       0.00
    2354       0.00
    2355       0.00
    2356       0.00
    2357       0.00
    2358       0.00
    2359       0.00
    2360       0.00
    2361       0.00
    2362       0.00
    2363       0.00
    2364       0.00
    2365      79.99
    2366       9.00
    2367       0.00
    2368       0.00
    2369       0.00
    2370       0.00
    2371       0.00
    2372      24.99
    2373       0.00
    2374       0.00
    2375       0.00
    2376       0.00
    2377       0.00
    2378       9.99
    2379       0.00
    2380      10.00
    2381       0.00
    2382       0.00
    2383       0.00
    2384       0.00
    2385      16.99
    2386      11.99
    2387      29.99
    2388       0.00
    2389      14.99
    2390      74.99
    2391       0.00
    2392       0.00
    2393       0.00
    2394       0.00
    2395      11.99
    2396       0.00
    2397       0.00
    2398       6.99
    2399       5.49
    2400      14.99
    2401       9.99
    2402      33.99
    2403       0.00
    2404       0.00
    2405       0.00
    2406      29.99
    2407      24.99
    2408       0.00
    2409      12.99
    2410       0.00
    2411       0.00
    2412       0.00
    2413       0.00
    2414      39.99
    2415       0.00
    2416       5.99
    2417       0.00
    2418       2.99
    2419      24.99
    2420      19.99
    2421       0.00
    2422       2.99
    2423       0.99
    2424       5.99
    2425       0.99
    2426       0.00
    2427       0.00
    2428       0.00
    2429       0.00
    2430       0.00
    2431       0.00
    2432       0.00
    2433       0.00
    2434       0.00
    2435       0.00
    2436       0.00
    2437       0.00
    2438       0.00
    2439       0.00
    2440       0.00
    2441       0.00
    2442       0.00
    2443       0.00
    2444       0.00
    2445       0.00
    2446       0.00
    2447       0.00
    2448       0.00
    2449       0.00
    2450       0.00
    2451       0.00
    2452       0.00
    2453       0.00
    2454       0.00
    2455       0.00
    2456       0.00
    2457       0.00
    2458       0.00
    2459       0.00
    2460       0.00
    2461       0.00
    2462       0.00
    2463       0.00
    2464       0.00
    2465       0.00
    2466       0.00
    2467       0.00
    2468       0.00
    2469       0.00
    2470       0.00
    2471       0.00
    2472       0.00
    2473       0.00
    2474       0.00
    2475       0.00
    2476       0.00
    2477       0.00
    2478       0.00
    2479       0.00
    2480       0.00
    2481       0.00
    2482       0.00
    2483       0.00
    2484       0.00
    2485       0.00
    2486       0.00
    2487       0.00
    2488       0.00
    2489       0.00
    2490       0.00
    2491       0.00
    2492       0.00
    2493       0.00
    2494       0.00
    2495       0.00
    2496       0.00
    2497       0.00
    2498       0.00
    2499       0.00
    2500       0.00
    2501       0.00
    2502       0.00
    2503       0.00
    2504       0.00
    2505       0.00
    2506       0.00
    2507       0.00
    2508       0.00
    2509       0.00
    2510       0.00
    2511       0.00
    2512       0.00
    2513       0.00
    2514       0.00
    2515       0.00
    2516       0.00
    2517       0.00
    2518       0.00
    2519       0.00
    2520       0.00
    2521       0.00
    2522       0.00
    2523       0.00
    2524       0.00
    2525       0.00
    2526       0.00
    2527       0.00
    2528       0.00
    2529       0.00
    2530       0.00
    2531       0.00
    2532       0.00
    2533       0.00
    2534       0.00
    2535       0.00
    2536       0.00
    2537       0.00
    2538       0.00
    2539       0.00
    2540       0.00
    2541       0.00
    2542       0.00
    2543       0.00
    2544       0.00
    2545       0.00
    2546       0.00
    2547       0.00
    2548       0.00
    2549       0.00
    2550       0.00
    2551       0.00
    2552       0.00
    2553       0.00
    2554       0.00
    2555       0.00
    2556       0.00
    2557       0.00
    2558       0.00
    2559       0.00
    2560       0.00
    2561       0.00
    2562       0.00
    2563       0.00
    2564       0.00
    2565       0.00
    2566       0.00
    2567       0.00
    2568       0.00
    2569       0.00
    2570       0.00
    2571       0.00
    2572       0.00
    2573       0.00
    2574       0.00
    2575       0.00
    2576       0.00
    2577       0.00
    2578       0.00
    2579       0.00
    2580       0.00
    2581       0.00
    2582       0.00
    2583       0.00
    2584       0.00
    2585       0.00
    2586       0.00
    2587       0.00
    2588       0.00
    2589       0.00
    2590       0.00
    2591       0.00
    2592       0.00
    2593       0.00
    2594       0.00
    2595       0.00
    2596       0.00
    2597       0.00
    2598       0.00
    2599       0.00
    2600       0.00
    2601       0.00
    2602       0.00
    2603       0.00
    2604       0.00
    2605       0.00
    2606       0.00
    2607       0.00
    2608       0.00
    2609       0.00
    2610       0.00
    2611       0.00
    2612       0.00
    2613       0.00
    2614       0.00
    2615       0.00
    2616       0.00
    2617       0.00
    2618       0.00
    2619       0.00
    2620       0.00
    2621       0.00
    2622       0.00
    2623       0.00
    2624       0.00
    2625       0.00
    2626       0.00
    2627       0.00
    2628       0.00
    2629       0.00
    2630       0.00
    2631       0.00
    2632       0.00
    2633       0.00
    2634       0.00
    2635       0.00
    2636       0.00
    2637       0.00
    2638       0.00
    2639       0.00
    2640       0.00
    2641       0.00
    2642       0.00
    2643       0.00
    2644       0.00
    2645       0.00
    2646       0.00
    2647       0.00
    2648       0.00
    2649       0.00
    2650       0.00
    2651       0.00
    2652       0.00
    2653       0.00
    2654       0.00
    2655       0.00
    2656       0.00
    2657       0.00
    2658       0.00
    2659       0.00
    2660       0.00
    2661       0.00
    2662       0.00
    2663       0.00
    2664       0.00
    2665       0.00
    2666       0.00
    2667       0.00
    2668       0.00
    2669       0.00
    2670       0.00
    2671       0.00
    2672       0.00
    2673       0.00
    2674       0.00
    2675       0.00
    2676       0.00
    2677       0.00
    2678       0.00
    2679       0.00
    2680       0.00
    2681       0.00
    2682       0.00
    2683       0.00
    2684       0.00
    2685       0.00
    2686       0.00
    2687       0.00
    2688       0.00
    2689       0.00
    2690       0.00
    2691       0.00
    2692       0.00
    2693       0.00
    2694       0.00
    2695       0.00
    2696       0.00
    2697       0.00
    2698       0.00
    2699       0.00
    2700       0.00
    2701       0.00
    2702       0.00
    2703       0.00
    2704       0.00
    2705       0.00
    2706       0.00
    2707       0.00
    2708       0.00
    2709       0.00
    2710       0.00
    2711       0.00
    2712       0.00
    2713       0.00
    2714       0.00
    2715       0.00
    2716       0.00
    2717       0.00
    2718       0.00
    2719       0.00
    2720       0.00
    2721       0.00
    2722       0.00
    2723       0.00
    2724       0.00
    2725       0.00
    2726       0.00
    2727       0.00
    2728       0.00
    2729       0.00
    2730       0.00
    2731       0.00
    2732       0.00
    2733       0.00
    2734       0.00
    2735       0.00
    2736       0.00
    2737       0.00
    2738       0.00
    2739       0.00
    2740       0.00
    2741       0.00
    2742       0.00
    2743       0.00
    2744       0.00
    2745       0.00
    2746       0.00
    2747       0.00
    2748       0.00
    2749       0.00
    2750       0.00
    2751       0.00
    2752       0.00
    2753       0.00
    2754       0.00
    2755       0.00
    2756       0.00
    2757       0.00
    2758       0.00
    2759       0.00
    2760       0.00
    2761       0.00
    2762       0.00
    2763       0.00
    2764       0.00
    2765       0.00
    2766       0.00
    2767       0.00
    2768       0.00
    2769       0.00
    2770       0.00
    2771       0.00
    2772       0.00
    2773       0.00
    2774       0.00
    2775       0.00
    2776       0.00
    2777       0.00
    2778       0.00
    2779       0.00
    2780       0.00
    2781       0.00
    2782       0.00
    2783       0.00
    2784       0.00
    2785       0.00
    2786       0.00
    2787       0.00
    2788       0.00
    2789       0.00
    2790       0.00
    2791       0.00
    2792       0.00
    2793       0.00
    2794       0.00
    2795       0.00
    2796       0.00
    2797       0.00
    2798       0.00
    2799       0.00
    2800       0.00
    2801       0.00
    2802       0.00
    2803       0.00
    2804       0.00
    2805       0.00
    2806       0.00
    2807       0.00
    2808       0.00
    2809       0.00
    2810       0.00
    2811       0.00
    2812       0.00
    2813       0.00
    2814       0.00
    2815       0.00
    2816       0.00
    2817       0.00
    2818       0.00
    2819       0.00
    2820       0.00
    2821       0.00
    2822       0.00
    2823       0.00
    2824       0.00
    2825       0.00
    2826       0.00
    2827       0.00
    2828       0.00
    2829       0.00
    2830       0.00
    2831       0.00
    2832       0.00
    2833       0.00
    2834       0.00
    2835       0.00
    2836       0.00
    2837       0.00
    2838       0.00
    2839       0.00
    2840       0.00
    2841       0.00
    2842       0.00
    2843       0.00
    2844       0.00
    2845       0.00
    2846       0.00
    2847       0.00
    2848       0.00
    2849       0.00
    2850       0.00
    2851       0.00
    2852       0.00
    2853       0.00
    2854       0.00
    2855       0.00
    2856       0.00
    2857       0.00
    2858       0.00
    2859       0.00
    2860       0.00
    2861       0.00
    2862       0.00
    2863       0.00
    2864       0.00
    2865       0.00
    2866       0.00
    2867       0.00
    2868       0.00
    2869       0.00
    2870       0.00
    2871       0.00
    2872       0.00
    2873       0.00
    2874       0.00
    2875       0.00
    2876       0.00
    2877       0.00
    2878       0.00
    2879       0.00
    2880       0.00
    2881       0.00
    2882       0.00
    2883       5.99
    2884       0.00
    2885       0.00
    2886       0.00
    2887       0.00
    2888       0.00
    2889       0.00
    2890       0.00
    2891       0.00
    2892       0.00
    2893       0.00
    2894       0.00
    2895       0.00
    2896       0.00
    2897       0.00
    2898       0.00
    2899       0.00
    2900       0.00
    2901       0.00
    2902       0.00
    2903       0.00
    2904       0.00
    2905       0.00
    2906       2.99
    2907       0.00
    2908       0.00
    2909       0.00
    2910       0.00
    2911       0.00
    2912       5.99
    2913       3.95
    2914       0.00
    2915       0.00
    2916       0.00
    2917       0.00
    2918       0.00
    2919       0.00
    2920       0.00
    2921       0.00
    2922       0.00
    2923       0.00
    2924       0.00
    2925       0.00
    2926       0.00
    2927       0.00
    2928       0.00
    2929       0.00
    2930       0.00
    2931       0.00
    2932       0.00
    2933       0.00
    2934       0.00
    2935       0.00
    2936       0.00
    2937       0.00
    2938       0.00
    2939       0.00
    2940       0.00
    2941       0.00
    2942       0.00
    2943       0.00
    2944       0.00
    2945       0.00
    2946       0.00
    2947       0.00
    2948       0.00
    2949       0.00
    2950       5.99
    2951       0.00
    2952       0.00
    2953       0.00
    2954       0.00
    2955       0.00
    2956       0.00
    2957       0.00
    2958       0.00
    2959       0.00
    2960       0.00
    2961       0.00
    2962       0.00
    2963       0.00
    2964       0.00
    2965       0.00
    2966       0.00
    2967       0.00
    2968       0.00
    2969       0.00
    2970       0.00
    2971       0.00
    2972       0.00
    2973       0.00
    2974       0.00
    2975       0.00
    2976       0.00
    2977       0.00
    2978       0.00
    2979       0.00
    2980       0.00
    2981       0.00
    2982       0.00
    2983       0.00
    2984       0.00
    2985       0.00
    2986       0.00
    2987       0.00
    2988       0.00
    2989       0.00
    2990       0.00
    2991       0.00
    2992       0.00
    2993       0.00
    2994       0.00
    2995       0.00
    2996       0.00
    2997       0.00
    2998       0.00
    2999       0.00
    3000       0.00
    3001       0.00
    3002       0.00
    3003       0.00
    3004       0.00
    3005       0.00
    3006       0.00
    3007       0.00
    3008       0.00
    3009       0.00
    3010       0.00
    3011       0.00
    3012       0.00
    3013       0.00
    3014       0.00
    3015       0.00
    3016       0.00
    3017       0.00
    3018       0.00
    3019       0.00
    3020       0.00
    3021       0.00
    3022       0.00
    3023       0.00
    3024       0.00
    3025       0.00
    3026       0.00
    3027       0.00
    3028       0.00
    3029       0.00
    3030       0.00
    3031       0.00
    3032       0.00
    3033       0.00
    3034       0.00
    3035       0.00
    3036       0.00
    3037       0.00
    3038       0.00
    3039      29.99
    3040       0.00
    3041       0.00
    3042       0.00
    3043       0.00
    3044       0.00
    3045       0.00
    3046       0.00
    3047       0.00
    3048       0.00
    3049       0.00
    3050       0.00
    3051       0.00
    3052       0.00
    3053       0.00
    3054       0.00
    3055       0.00
    3056       0.00
    3057       0.00
    3058       0.00
    3059       0.00
    3060       0.00
    3061       0.00
    3062       0.00
    3063       0.00
    3064       0.00
    3065       0.00
    3066       0.00
    3067       0.00
    3068       0.00
    3069       0.00
    3070       0.00
    3071       0.00
    3072       0.00
    3073       0.00
    3074       0.00
    3075       0.00
    3076       0.00
    3077       0.00
    3078       0.00
    3079       0.00
    3080       0.00
    3081       0.00
    3082       0.00
    3083       0.00
    3084       0.00
    3085       0.00
    3086       0.00
    3087       0.00
    3088       0.00
    3089       0.00
    3090       0.00
    3091       0.00
    3092       0.00
    3093       0.00
    3094       0.00
    3095       0.00
    3096       0.00
    3097       0.00
    3098       0.00
    3099       0.00
    3100       0.00
    3101       0.00
    3102       0.00
    3103       0.00
    3104       0.00
    3105       0.00
    3106       0.00
    3107       0.00
    3108       0.00
    3109       0.00
    3110       0.00
    3111       0.00
    3112       0.00
    3113       0.00
    3114       0.00
    3115       0.00
    3116       0.00
    3117       0.00
    3118       0.00
    3119       0.00
    3120       0.00
    3121       0.00
    3122       0.00
    3123       0.00
    3124       0.00
    3125       0.00
    3126       0.00
    3127       0.00
    3128       0.00
    3129       0.00
    3130       0.00
    3131       0.00
    3132       0.00
    3133       0.00
    3134       0.00
    3135       0.00
    3136       0.00
    3137       0.00
    3138       0.00
    3139       0.00
    3140       0.00
    3141       0.00
    3142       0.00
    3143       0.00
    3144       0.00
    3145       0.00
    3146       0.00
    3147       0.00
    3148       0.00
    3149       0.00
    3150       0.00
    3151       0.00
    3152       0.00
    3153       0.00
    3154       0.00
    3155       0.00
    3156       0.00
    3157       0.00
    3158       0.00
    3159       0.00
    3160       0.00
    3161       0.00
    3162       0.00
    3163       0.00
    3164       0.00
    3165       0.00
    3166       0.00
    3167       0.00
    3168       0.00
    3169       0.00
    3170       0.00
    3171       0.00
    3172       0.00
    3173       0.00
    3174       0.00
    3175       0.00
    3176       0.00
    3177       0.00
    3178       0.00
    3179       0.00
    3180       0.00
    3181       0.00
    3182       0.00
    3183       0.00
    3184       0.00
    3185       0.00
    3186       0.00
    3187       0.00
    3188       0.00
    3189       0.00
    3190       0.00
    3191       0.00
    3192       0.00
    3193       0.00
    3194       0.00
    3195       0.00
    3196       0.00
    3197       0.00
    3198       0.00
    3199       0.00
    3200       0.00
    3201       0.00
    3202       0.00
    3203       0.00
    3204       0.00
    3205       0.00
    3206       0.00
    3207       0.00
    3208       0.00
    3209       0.00
    3210       0.00
    3211       0.00
    3212       0.00
    3213       0.00
    3214       0.00
    3215       0.00
    3216       0.00
    3217       0.00
    3218       0.00
    3219       0.00
    3220       0.00
    3221       0.00
    3222       0.00
    3223       0.00
    3224       0.00
    3225       0.00
    3226       0.00
    3227       0.00
    3228       0.00
    3229       0.00
    3230       0.00
    3231       0.00
    3232       0.00
    3233       0.00
    3234       0.00
    3235       0.00
    3236       0.00
    3237       0.00
    3238       0.00
    3239       0.00
    3240       0.00
    3241       0.00
    3242       0.00
    3243       0.00
    3244       0.00
    3245       0.00
    3246       0.00
    3247       0.00
    3248       0.00
    3249       0.00
    3250       0.00
    3251       0.00
    3252       0.00
    3253       0.00
    3254       0.00
    3255       0.00
    3256       0.00
    3257       0.00
    3258       0.00
    3259       0.00
    3260       0.00
    3261       0.00
    3262       0.00
    3263       0.00
    3264       0.00
    3265       0.00
    3266       0.00
    3267       0.00
    3268       0.00
    3269       0.00
    3270       0.00
    3271       0.00
    3272       0.00
    3273       0.00
    3274       0.00
    3275       0.00
    3276       0.00
    3277       0.00
    3278       0.00
    3279       0.00
    3280       0.00
    3281       0.00
    3282       0.00
    3283       0.00
    3284       0.00
    3285       0.00
    3286       0.00
    3287       0.00
    3288       0.00
    3289       0.00
    3290       0.00
    3291       0.00
    3292       0.00
    3293       0.00
    3294       0.00
    3295       0.00
    3296       0.00
    3297       0.00
    3298       0.00
    3299       0.00
    3300       0.00
    3301       0.00
    3302       0.00
    3303       0.00
    3304       0.00
    3305       0.00
    3306       0.00
    3307       0.00
    3308       0.00
    3309       0.00
    3310       0.00
    3311       0.00
    3312       0.00
    3313       0.00
    3314       0.00
    3315       0.00
    3316       0.00
    3317       0.00
    3318       0.00
    3319       0.00
    3320       0.00
    3321       0.00
    3322       0.00
    3323       0.00
    3324       0.00
    3325       0.00
    3326       0.00
    3327       0.00
    3328       0.00
    3329       0.00
    3330       0.00
    3331       0.00
    3332       0.00
    3333       0.00
    3334       0.00
    3335       0.00
    3336       0.00
    3337       0.00
    3338       0.00
    3339       0.00
    3340       0.00
    3341       0.00
    3342       0.00
    3343       0.00
    3344       0.00
    3345       0.00
    3346       0.00
    3347       0.00
    3348       0.00
    3349       0.00
    3350       0.00
    3351       0.00
    3352       0.00
    3353       0.00
    3354       0.00
    3355       0.00
    3356       0.00
    3357       0.00
    3358       0.00
    3359       0.00
    3360       0.00
    3361       0.00
    3362       0.00
    3363       0.00
    3364       0.00
    3365       0.00
    3366       0.00
    3367       0.00
    3368       0.00
    3369       0.00
    3370       0.00
    3371       0.00
    3372       0.00
    3373       0.00
    3374       0.00
    3375       0.00
    3376       0.00
    3377       0.00
    3378       0.00
    3379       0.00
    3380       0.00
    3381       0.00
    3382       0.00
    3383       0.00
    3384       0.00
    3385       0.00
    3386       0.00
    3387       0.00
    3388       0.00
    3389       0.00
    3390       0.00
    3391       0.00
    3392       0.00
    3393       0.00
    3394       0.00
    3395       0.00
    3396       0.00
    3397       0.00
    3398       0.00
    3399       0.00
    3400       0.00
    3401       0.00
    3402       0.00
    3403       2.49
    3404       0.00
    3405       0.99
    3406       0.00
    3407       0.00
    3408       0.99
    3409       0.00
    3410       0.00
    3411       0.00
    3412       0.00
    3413       0.00
    3414       0.00
    3415       0.00
    3416       0.00
    3417       0.00
    3418       0.00
    3419       0.00
    3420       0.00
    3421       0.00
    3422       0.00
    3423       0.00
    3424       0.00
    3425       0.00
    3426       0.00
    3427       0.00
    3428       0.00
    3429       0.00
    3430       0.00
    3431       0.00
    3432       0.00
    3433       0.00
    3434       0.00
    3435       0.00
    3436       0.00
    3437       0.00
    3438       0.00
    3439       0.00
    3440       0.00
    3441       9.99
    3442       4.49
    3443       0.00
    3444       0.00
    3445       0.00
    3446       0.00
    3447       0.00
    3448       0.00
    3449       0.00
    3450       0.00
    3451       0.00
    3452       0.00
    3453       0.00
    3454       0.00
    3455       0.00
    3456       0.00
    3457       0.00
    3458       0.00
    3459       0.00
    3460       0.00
    3461       0.00
    3462       0.00
    3463       0.00
    3464       0.00
    3465       0.00
    3466       0.00
    3467       0.00
    3468       0.00
    3469       0.00
    3470       0.00
    3471       0.00
    3472       0.00
    3473       0.00
    3474       0.00
    3475       0.00
    3476       0.00
    3477       0.00
    3478       0.00
    3479       0.00
    3480       0.00
    3481       0.00
    3482       0.00
    3483       0.00
    3484       0.00
    3485       0.00
    3486       0.00
    3487       0.00
    3488       0.00
    3489       0.00
    3490       0.00
    3491       0.00
    3492       0.00
    3493       0.00
    3494       0.00
    3495       0.00
    3496       0.00
    3497       0.00
    3498       0.00
    3499       0.00
    3500       0.00
    3501       0.00
    3502       0.00
    3503       0.00
    3504       0.00
    3505       0.00
    3506       0.00
    3507       0.00
    3508       0.00
    3509       0.00
    3510       0.00
    3511       0.00
    3512       0.00
    3513       0.00
    3514       0.00
    3515       0.00
    3516       0.00
    3517       0.00
    3518       0.00
    3519       0.00
    3520       0.00
    3521       0.00
    3522       0.00
    3523       0.00
    3524       0.00
    3525       0.00
    3526       0.00
    3527       0.00
    3528       0.00
    3529       0.00
    3530       0.00
    3531       0.00
    3532       0.00
    3533       0.00
    3534       0.00
    3535       0.00
    3536       0.00
    3537       0.00
    3538       0.00
    3539       0.00
    3540       0.00
    3541       0.00
    3542       0.00
    3543       0.00
    3544       0.00
    3545       0.00
    3546       0.00
    3547       0.00
    3548       0.00
    3549       0.00
    3550       0.00
    3551       0.00
    3552       0.00
    3553       0.00
    3554       0.00
    3555       0.00
    3556       0.00
    3557       0.00
    3558       0.00
    3559       0.00
    3560       0.00
    3561       0.00
    3562       0.00
    3563       0.00
    3564       5.99
    3565       0.00
    3566       0.00
    3567       0.00
    3568       0.00
    3569       0.00
    3570       0.00
    3571       0.00
    3572       0.00
    3573       0.00
    3574       0.00
    3575       0.00
    3576       0.00
    3577       0.00
    3578       0.00
    3579       0.00
    3580       0.00
    3581       0.00
    3582       0.00
    3583       0.00
    3584       0.00
    3585       0.00
    3586       0.00
    3587       0.00
    3588       0.00
    3589       0.00
    3590       0.00
    3591       0.00
    3592       0.00
    3593       0.00
    3594       0.00
    3595       0.00
    3596       0.00
    3597       0.00
    3598       0.00
    3599       0.00
    3600       0.00
    3601       0.00
    3602       0.00
    3603       0.00
    3604       0.00
    3605       0.00
    3606       0.00
    3607       0.00
    3608       0.00
    3609       0.00
    3610       0.00
    3611       0.00
    3612       0.00
    3613       0.00
    3614       0.00
    3615       0.00
    3616       0.00
    3617       0.00
    3618       0.00
    3619       0.00
    3620       0.00
    3621       0.00
    3622       0.00
    3623       0.00
    3624       0.00
    3625       0.00
    3626       0.00
    3627       0.00
    3628       0.00
    3629       0.00
    3630       0.00
    3631       0.00
    3632       0.00
    3633       0.00
    3634       0.00
    3635       0.00
    3636       0.00
    3637       0.00
    3638       0.00
    3639       0.00
    3640       0.00
    3641       0.00
    3642       0.00
    3643       0.00
    3644       0.00
    3645       0.00
    3646       0.00
    3647       0.00
    3648       0.00
    3649       0.00
    3650       0.00
    3651       0.00
    3652       0.00
    3653       0.00
    3654       0.00
    3655       0.00
    3656       0.00
    3657       0.00
    3658       4.49
    3659       0.00
    3660       0.00
    3661       0.00
    3662       0.00
    3663       0.00
    3664       0.00
    3665       0.00
    3666       0.00
    3667       0.00
    3668       0.00
    3669       0.00
    3670       0.00
    3671       0.00
    3672       0.00
    3673       0.00
    3674       0.00
    3675       0.00
    3676       0.00
    3677       0.00
    3678       0.00
    3679       0.00
    3680       0.00
    3681       0.00
    3682       0.00
    3683       0.00
    3684       0.00
    3685       0.00
    3686       0.00
    3687       0.00
    3688       0.00
    3689       0.00
    3690       0.00
    3691       0.00
    3692       0.00
    3693       0.00
    3694       0.00
    3695       0.00
    3696       0.00
    3697       0.00
    3698       0.00
    3699       0.00
    3700       0.00
    3701       0.00
    3702       0.00
    3703       0.00
    3704       0.00
    3705       0.00
    3706       0.00
    3707       0.00
    3708       0.00
    3709       0.00
    3710       0.00
    3711       0.00
    3712       0.00
    3713       0.00
    3714       0.00
    3715       0.00
    3716       0.00
    3717       0.00
    3718       0.00
    3719       0.00
    3720       0.00
    3721       0.00
    3722       0.00
    3723       0.00
    3724       0.00
    3725       0.00
    3726       0.00
    3727       0.00
    3728       0.00
    3729       0.00
    3730       0.00
    3731       0.00
    3732       0.00
    3733       0.00
    3734       0.00
    3735       0.00
    3736       0.00
    3737       0.00
    3738       0.00
    3739       0.00
    3740       0.00
    3741       0.00
    3742       0.00
    3743       0.00
    3744       0.00
    3745       0.00
    3746       0.00
    3747       0.00
    3748       0.00
    3749       0.00
    3750       0.00
    3751       0.00
    3752       0.00
    3753       0.00
    3754       0.00
    3755       0.00
    3756       0.00
    3757       0.00
    3758       0.00
    3759       0.00
    3760       0.00
    3761       0.00
    3762       0.00
    3763       0.00
    3764       0.00
    3765       0.00
    3766       0.00
    3767       0.00
    3768       0.00
    3769       0.00
    3770       0.00
    3771       0.00
    3772       0.00
    3773       0.00
    3774       0.00
    3775       0.00
    3776       0.00
    3777       0.00
    3778       0.00
    3779       0.00
    3780       0.00
    3781       0.00
    3782       0.00
    3783       0.00
    3784       0.00
    3785       0.00
    3786       0.00
    3787       0.00
    3788       0.00
    3789       0.00
    3790       0.00
    3791       0.00
    3792       0.00
    3793       0.00
    3794       0.00
    3795       0.00
    3796       0.00
    3797       0.00
    3798       0.00
    3799       0.00
    3800       0.00
    3801       0.00
    3802       0.00
    3803       0.00
    3804       0.00
    3805       0.00
    3806       0.00
    3807       0.00
    3808       0.00
    3809       0.00
    3810       0.00
    3811       0.00
    3812       0.00
    3813       0.00
    3814       0.00
    3815       0.00
    3816       0.00
    3817       0.00
    3818       0.00
    3819       0.00
    3820       0.00
    3821       0.00
    3822       0.00
    3823       0.00
    3824       0.00
    3825       0.00
    3826       0.00
    3827       0.00
    3828       0.00
    3829       0.00
    3830       0.00
    3831       0.00
    3832       0.00
    3833       0.00
    3834       0.00
    3835       0.00
    3836       0.00
    3837       0.00
    3838       0.00
    3839       0.00
    3840       0.00
    3841       0.00
    3842       0.00
    3843       0.00
    3844       0.00
    3845       0.00
    3846       0.00
    3847       0.00
    3848       0.00
    3849       0.00
    3850       0.00
    3851       0.00
    3852       0.00
    3853       0.00
    3854       0.00
    3855       0.00
    3856       0.00
    3857       0.00
    3858       0.00
    3859       0.00
    3860       0.00
    3861       0.00
    3862       0.00
    3863       0.00
    3864       0.00
    3865       0.00
    3866       0.00
    3867       0.00
    3868       0.00
    3869       0.00
    3870       0.00
    3871       0.00
    3872       0.00
    3873       0.00
    3874       0.00
    3875       0.00
    3876       0.00
    3877       0.00
    3878       0.00
    3879       0.00
    3880       0.00
    3881       0.00
    3882       0.00
    3883       0.00
    3884       0.00
    3885       0.00
    3886       0.00
    3887       0.00
    3888       0.00
    3889       0.00
    3890       0.00
    3891       0.00
    3892       0.00
    3893       0.00
    3894       0.00
    3895       0.00
    3896       0.00
    3897       0.00
    3898       0.00
    3899       0.00
    3900       0.00
    3901       0.00
    3902       0.00
    3903       0.00
    3904       0.00
    3905       0.00
    3906       0.00
    3907       0.00
    3908       0.00
    3909       0.00
    3910       0.00
    3911       0.00
    3912       0.00
    3913       0.00
    3914       0.00
    3915       0.00
    3916       0.00
    3917       0.00
    3918       0.00
    3919       0.00
    3920       0.00
    3921       0.00
    3922       0.00
    3923       0.00
    3924       0.00
    3925       0.00
    3926       0.00
    3927       0.00
    3928       0.00
    3929       0.00
    3930       0.00
    3931       0.00
    3932       0.00
    3933       0.00
    3934       0.00
    3935       0.00
    3936       0.00
    3937       0.00
    3938       0.00
    3939       0.00
    3940       0.00
    3941       0.00
    3942       0.00
    3943       0.00
    3944       0.00
    3945       0.00
    3946       0.00
    3947       0.00
    3948       0.00
    3949       0.00
    3950       0.00
    3951       0.00
    3952       0.00
    3953       0.00
    3954       0.00
    3955       0.00
    3956       0.00
    3957       1.99
    3958       0.00
    3959       1.49
    3960       0.00
    3961       0.00
    3962       0.00
    3963       0.00
    3964       1.99
    3965       3.99
    3966       1.70
    3967       0.00
    3968       0.99
    3969       0.00
    3970       0.00
    3971       0.00
    3972       0.00
    3973       0.00
    3974       0.00
    3975       0.00
    3976       0.00
    3977       1.99
    3978       0.00
    3979       0.00
    3980       0.00
    3981       0.00
    3982       0.00
    3983       0.00
    3984       0.00
    3985       0.99
    3986       0.00
    3987       0.00
    3988       0.00
    3989       0.00
    3990       0.00
    3991       0.00
    3992       0.00
    3993       2.99
    3994       0.00
    3995       0.00
    3996       0.00
    3997       2.99
    3998       0.00
    3999       0.00
    4000       0.00
    4001       0.00
    4002       0.00
    4003       0.99
    4004       0.00
    4005       0.00
    4006       0.00
    4007       0.00
    4008       0.00
    4009       0.00
    4010       0.00
    4011       0.00
    4012       0.00
    4013       0.00
    4014       0.00
    4015       0.00
    4016       0.00
    4017       0.00
    4018       0.00
    4019       0.00
    4020       0.00
    4021       0.00
    4022       0.00
    4023       0.00
    4024       0.00
    4025       0.00
    4026       0.00
    4027       0.00
    4028       0.00
    4029       0.00
    4030       0.00
    4031       0.00
    4032       0.00
    4033       0.00
    4034       0.99
    4035       0.00
    4036       0.00
    4037       0.00
    4038       0.00
    4039       0.00
    4040       0.00
    4041       0.00
    4042       0.00
    4043       0.00
    4044       0.00
    4045       0.00
    4046       0.00
    4047       0.99
    4048       0.00
    4049       0.00
    4050       0.00
    4051       0.00
    4052       0.00
    4053       0.00
    4054       0.00
    4055       0.00
    4056       0.00
    4057       2.99
    4058       0.00
    4059       0.00
    4060       0.00
    4061       0.00
    4062       0.00
    4063       0.00
    4064       0.00
    4065       0.00
    4066       0.00
    4067       0.00
    4068       0.00
    4069       0.00
    4070       0.00
    4071       0.00
    4072       0.00
    4073       0.00
    4074       0.00
    4075       0.00
    4076       0.00
    4077       0.00
    4078       0.00
    4079       0.00
    4080       0.00
    4081       0.00
    4082       0.00
    4083       0.00
    4084       0.00
    4085       8.99
    4086       0.00
    4087       0.00
    4088       0.00
    4089       0.00
    4090       0.00
    4091       0.00
    4092       0.00
    4093       0.00
    4094       0.00
    4095       0.00
    4096       0.00
    4097       0.00
    4098       0.00
    4099       0.00
    4100       0.00
    4101       0.00
    4102       0.00
    4103       0.00
    4104       0.00
    4105       0.00
    4106       0.00
    4107       0.00
    4108       4.99
    4109       0.00
    4110       0.00
    4111       0.00
    4112       0.00
    4113       0.00
    4114       0.00
    4115       0.00
    4116       0.00
    4117       0.00
    4118       0.00
    4119       0.00
    4120       0.00
    4121       0.00
    4122       0.00
    4123       0.00
    4124       0.00
    4125       0.00
    4126       0.00
    4127       2.99
    4128       0.00
    4129       0.00
    4130       0.00
    4131       0.00
    4132       4.99
    4133       5.99
    4134      39.99
    4135       0.00
    4136       0.00
    4137       2.49
    4138       0.00
    4139       2.00
    4140       1.49
    4141       0.00
    4142       0.00
    4143       0.00
    4144       0.00
    4145       0.00
    4146       0.00
    4147       0.00
    4148       0.00
    4149       0.00
    4150       0.00
    4151       0.00
    4152       0.00
    4153       0.00
    4154       0.00
    4155       0.00
    4156       4.99
    4157       0.00
    4158       1.70
    4159       0.00
    4160       0.00
    4161       1.49
    4162       0.00
    4163       0.00
    4164       0.00
    4165       2.99
    4166       3.88
    4167       0.00
    4168       0.00
    4169       0.99
    4170       0.00
    4171       0.00
    4172      14.99
    4173       1.49
    4174       0.00
    4175       0.99
    4176       0.00
    4177      25.99
    4178       1.49
    4179       0.00
    4180       0.00
    4181       5.99
    4182       0.00
    4183       0.00
    4184       0.00
    4185       0.00
    4186       0.00
    4187       0.00
    4188       0.00
    4189       0.00
    4190       1.99
    4191       0.00
    4192       3.99
    4193       0.00
    4194       0.00
    4195       0.00
    4196       0.00
    4197     399.99
    4198       0.00
    4199       0.00
    4200       0.00
    4201       0.00
    4202       0.00
    4203      17.99
    4204       0.00
    4205       0.00
    4206       0.00
    4207       0.00
    4208       0.00
    4209       0.00
    4210       0.00
    4211       0.00
    4212       0.00
    4213       0.00
    4214       0.00
    4215       0.00
    4216       1.99
    4217       0.00
    4218       0.99
    4219       0.00
    4220       0.00
    4221       0.00
    4222       0.00
    4223       0.00
    4224       0.00
    4225       0.00
    4226       0.00
    4227       0.00
    4228       0.00
    4229       0.00
    4230       0.00
    4231       0.00
    4232       0.00
    4233       0.00
    4234       0.00
    4235       0.00
    4236       0.00
    4237       0.00
    4238       0.00
    4239       0.00
    4240       0.00
    4241       0.00
    4242       0.00
    4243       0.00
    4244       0.00
    4245       0.00
    4246       0.00
    4247       0.00
    4248       0.00
    4249       0.00
    4250       0.00
    4251       0.00
    4252       0.00
    4253       0.00
    4254       0.00
    4255       0.00
    4256       0.00
    4257       0.00
    4258       0.00
    4259       0.00
    4260       0.99
    4261       0.00
    4262       0.00
    4263       0.00
    4264       0.00
    4265       1.99
    4266       0.00
    4267       0.00
    4268       0.00
    4269       0.00
    4270       0.00
    4271       0.00
    4272       0.00
    4273       0.00
    4274       0.00
    4275       0.00
    4276       0.00
    4277       0.00
    4278       0.00
    4279       0.00
    4280       0.00
    4281       0.00
    4282       4.99
    4283       0.00
    4284       0.00
    4285       0.00
    4286       0.00
    4287       0.00
    4288       0.00
    4289       0.00
    4290       0.00
    4291       0.00
    4292       0.00
    4293       1.99
    4294       0.00
    4295       0.00
    4296       0.00
    4297       0.00
    4298       0.00
    4299       1.99
    4300       0.00
    4301       5.99
    4302       0.00
    4303       0.00
    4304       0.00
    4305       0.00
    4306       0.00
    4307       0.00
    4308       0.00
    4309       0.00
    4310       0.00
    4311       0.00
    4312       0.00
    4313       0.00
    4314       0.00
    4315       0.00
    4316       3.99
    4317       0.00
    4318       0.00
    4319       0.00
    4320       0.00
    4321       0.00
    4322       0.00
    4323       0.00
    4324       0.00
    4325       0.00
    4326       0.00
    4327       0.00
    4328       0.00
    4329       0.00
    4330       0.00
    4331       0.00
    4332       0.00
    4333       0.00
    4334       0.00
    4335       0.00
    4336       0.00
    4337       0.00
    4338       0.00
    4339       0.00
    4340       0.00
    4341       0.00
    4342       0.00
    4343       0.00
    4344       0.00
    4345       0.00
    4346       0.00
    4347       6.99
    4348       0.00
    4349       0.00
    4350       0.00
    4351       0.00
    4352       0.00
    4353       0.00
    4354       0.00
    4355       0.00
    4356       0.00
    4357       0.00
    4358       2.99
    4359       0.00
    4360       0.00
    4361       0.00
    4362     399.99
    4363       0.00
    4364       0.00
    4365       0.00
    4366       0.00
    4367     400.00
    4368       0.00
    4369       0.00
    4370       0.00
    4371       0.00
    4372       0.00
    4373       0.00
    4374       0.00
    4375       0.00
    4376       0.00
    4377       0.00
    4378       0.00
    4379       0.00
    4380       2.99
    4381       0.00
    4382       0.00
    4383       0.00
    4384       0.00
    4385       0.00
    4386       0.00
    4387       0.00
    4388       0.00
    4389       0.00
    4390       0.00
    4391       4.99
    4392       0.00
    4393       0.00
    4394       0.00
    4395       2.99
    4396       0.00
    4397       2.99
    4398       0.00
    4399       0.00
    4400       0.99
    4401       0.00
    4402       0.00
    4403       2.49
    4404       0.00
    4405       2.99
    4406       0.00
    4407       0.00
    4408       0.00
    4409       0.99
    4410       0.00
    4411       3.99
    4412       0.00
    4413       2.99
    4414       0.00
    4415       0.00
    4416       2.49
    4417       1.99
    4418       0.00
    4419       4.99
    4420       0.00
    4421       0.00
    4422       0.00
    4423       0.99
    4424       0.00
    4425       0.00
    4426       0.00
    4427       1.99
    4428       0.00
    4429       2.49
    4430       0.00
    4431       0.99
    4432       0.00
    4433       3.02
    4434       0.00
    4435       0.00
    4436       0.00
    4437       6.99
    4438       0.00
    4439       0.00
    4440       0.00
    4441       0.00
    4442       0.00
    4443       0.00
    4444       0.00
    4445       0.00
    4446       0.00
    4447       0.00
    4448       0.00
    4449       0.00
    4450       1.99
    4451       1.49
    4452       0.99
    4453       1.49
    4454       0.00
    4455       0.00
    4456       0.99
    4457       1.49
    4458       1.49
    4459       1.49
    4460       1.49
    4461       0.99
    4462       0.99
    4463       0.00
    4464       0.00
    4465       1.49
    4466       1.49
    4467       0.00
    4468       1.49
    4469       0.00
    4470       0.00
    4471       1.49
    4472       0.00
    4473       0.00
    4474       0.00
    4475       0.00
    4476       0.00
    4477       0.99
    4478       0.00
    4479       0.00
    4480       0.00
    4481       0.00
    4482       0.00
    4483       0.00
    4484       0.00
    4485       0.00
    4486       0.00
    4487       1.49
    4488       0.99
    4489       0.99
    4490       0.00
    4491       0.99
    4492       0.00
    4493       0.00
    4494       0.00
    4495       0.00
    4496       0.00
    4497       0.00
    4498       0.00
    4499       0.00
    4500       0.00
    4501       0.00
    4502       0.00
    4503       0.00
    4504       0.00
    4505       0.00
    4506       0.00
    4507       0.00
    4508       1.49
    4509       0.00
    4510       0.00
    4511       0.00
    4512       0.00
    4513       0.00
    4514       0.00
    4515       0.00
    4516       0.00
    4517       0.00
    4518       0.00
    4519       0.00
    4520       0.00
    4521       1.76
    4522       0.00
    4523       0.00
    4524       0.00
    4525       0.00
    4526       0.00
    4527       0.00
    4528       0.00
    4529       0.00
    4530       0.00
    4531       0.00
    4532       0.00
    4533       0.00
    4534       4.84
    4535       0.00
    4536       0.00
    4537       0.00
    4538       0.00
    4539       0.00
    4540       0.00
    4541       0.00
    4542       1.99
    4543       0.00
    4544       0.00
    4545       0.00
    4546       0.00
    4547       0.00
    4548       0.00
    4549       0.00
    4550       0.00
    4551       0.00
    4552       0.00
    4553       0.00
    4554       1.99
    4555       0.00
    4556       4.99
    4557       0.00
    4558       0.00
    4559       4.77
    4560       0.00
    4561       0.99
    4562       0.00
    4563       0.00
    4564       0.00
    4565       0.00
    4566       0.00
    4567       0.00
    4568       0.00
    4569       0.00
    4570       0.00
    4571       4.99
    4572       0.00
    4573       0.00
    4574       0.00
    4575       0.00
    4576       0.00
    4577       1.99
    4578       0.00
    4579       0.00
    4580       0.00
    4581       0.00
    4582       0.00
    4583       0.00
    4584       0.00
    4585       2.49
    4586       0.00
    4587       0.00
    4588       0.00
    4589       0.00
    4590       0.00
    4591       0.00
    4592       0.00
    4593       0.00
    4594       0.00
    4595       0.00
    4596       0.00
    4597       0.00
    4598       0.00
    4599       0.00
    4600       0.00
    4601       0.00
    4602       0.00
    4603       0.00
    4604       0.00
    4605       0.00
    4606       2.99
    4607       0.00
    4608       0.00
    4609       0.00
    4610       0.00
    4611       0.00
    4612       4.99
    4613       4.99
    4614       0.00
    4615       0.00
    4616       5.99
    4617       2.99
    4618       0.00
    4619       4.99
    4620       1.61
    4621       0.99
    4622       0.00
    4623       0.00
    4624       0.00
    4625       0.00
    4626       0.00
    4627       0.00
    4628       0.00
    4629       0.00
    4630       0.00
    4631       0.00
    4632       0.00
    4633       0.00
    4634       0.00
    4635       0.00
    4636       0.00
    4637       3.99
    4638       0.00
    4639       0.00
    4640       0.00
    4641       0.00
    4642       0.00
    4643       0.00
    4644       0.00
    4645       0.00
    4646       0.00
    4647       0.00
    4648       0.00
    4649       0.00
    4650       0.00
    4651       0.00
    4652       0.00
    4653       0.00
    4654       0.00
    4655       0.00
    4656       0.00
    4657       0.00
    4658       0.00
    4659       0.00
    4660       0.00
    4661       0.00
    4662       0.00
    4663       2.49
    4664       0.00
    4665       0.00
    4666       0.00
    4667       0.00
    4668       0.00
    4669       0.00
    4670       0.00
    4671       0.00
    4672       0.00
    4673       0.00
    4674       0.00
    4675       0.00
    4676       0.00
    4677       0.00
    4678       0.00
    4679       0.00
    4680       0.00
    4681       0.00
    4682       0.00
    4683       0.00
    4684       0.00
    4685       0.00
    4686       0.00
    4687       0.00
    4688       0.00
    4689       0.00
    4690       0.00
    4691       0.00
    4692       0.00
    4693       0.00
    4694       7.99
    4695       0.00
    4696       0.00
    4697       9.99
    4698       0.00
    4699       0.00
    4700       3.99
    4701       0.00
    4702       0.00
    4703       0.00
    4704       0.00
    4705       0.00
    4706       0.00
    4707       0.00
    4708       0.99
    4709       0.00
    4710       0.00
    4711      14.99
    4712       0.00
    4713       0.00
    4714       0.00
    4715       0.00
    4716       1.99
    4717       0.00
    4718       0.00
    4719       0.00
    4720       0.00
    4721       0.99
    4722       0.00
    4723       0.00
    4724       0.00
    4725       0.00
    4726       0.00
    4727       0.00
    4728       0.00
    4729       0.00
    4730       0.00
    4731       0.00
    4732       0.00
    4733       0.00
    4734       0.00
    4735       0.00
    4736       0.00
    4737       0.00
    4738       0.00
    4739       0.00
    4740       0.00
    4741       0.00
    4742       0.00
    4743       1.99
    4744       0.00
    4745       0.00
    4746       0.00
    4747       0.00
    4748       0.00
    4749       0.00
    4750       0.00
    4751       0.00
    4752       0.00
    4753       0.00
    4754       0.00
    4755       1.99
    4756       0.00
    4757       0.00
    4758       0.00
    4759       1.99
    4760       0.00
    4761       0.00
    4762       0.00
    4763       0.00
    4764       1.99
    4765       0.00
    4766       0.00
    4767       0.00
    4768       0.00
    4769       0.99
    4770       0.00
    4771       0.00
    4772       0.00
    4773       0.99
    4774       0.00
    4775       0.00
    4776       2.99
    4777       0.00
    4778       0.00
    4779       0.99
    4780       0.00
    4781       3.99
    4782       0.00
    4783       0.00
    4784       0.00
    4785       0.00
    4786       0.00
    4787       0.00
    4788       1.49
    4789       1.99
    4790       0.00
    4791       0.00
    4792       0.00
    4793       0.00
    4794       0.00
    4795       0.00
    4796       0.00
    4797       0.00
    4798       0.00
    4799       0.00
    4800       0.00
    4801       0.00
    4802       0.00
    4803       0.00
    4804       0.00
    4805       0.00
    4806       0.00
    4807       0.00
    4808       0.00
    4809       0.00
    4810       0.00
    4811       0.00
    4812       0.00
    4813       2.99
    4814       0.00
    4815       0.00
    4816       0.00
    4817       0.00
    4818       0.00
    4819       0.00
    4820       0.00
    4821       0.00
    4822       0.00
    4823       0.00
    4824       0.00
    4825       0.00
    4826       0.00
    4827       0.00
    4828       0.00
    4829       0.00
    4830       0.00
    4831       0.00
    4832       0.00
    4833       0.00
    4834       1.49
    4835       0.00
    4836       0.00
    4837       0.00
    4838       0.00
    4839       0.00
    4840       0.00
    4841       0.00
    4842       0.99
    4843       0.00
    4844       2.50
    4845       5.99
    4846       0.00
    4847       0.00
    4848      19.99
    4849       0.00
    4850       0.00
    4851       0.00
    4852       0.00
    4853       0.00
    4854       0.00
    4855       0.00
    4856       0.00
    4857       0.00
    4858       0.00
    4859       0.00
    4860       0.00
    4861       0.00
    4862       0.00
    4863       0.00
    4864       0.00
    4865       0.00
    4866       0.00
    4867       0.00
    4868       1.99
    4869       0.00
    4870       0.00
    4871       0.00
    4872       0.00
    4873       0.00
    4874       0.00
    4875       0.00
    4876       0.00
    4877       0.00
    4878       0.00
    4879       0.00
    4880       0.00
    4881       0.00
    4882       1.59
    4883       0.00
    4884       0.00
    4885       0.00
    4886       0.00
    4887       0.00
    4888       0.00
    4889       0.00
    4890       0.00
    4891       0.00
    4892       0.00
    4893       0.00
    4894       0.00
    4895       0.00
    4896       0.00
    4897       0.00
    4898       0.00
    4899       0.00
    4900       0.00
    4901       1.99
    4902       0.00
    4903       0.00
    4904       0.00
    4905       0.00
    4906       0.00
    4907       0.00
    4908       0.00
    4909       0.00
    4910       0.00
    4911       0.00
    4912       0.00
    4913       0.00
    4914       0.00
    4915       0.00
    4916       0.00
    4917       0.00
    4918       0.00
    4919       0.00
    4920       0.00
    4921       0.00
    4922       0.00
    4923       0.00
    4924       0.00
    4925       0.00
    4926       0.00
    4927       0.00
    4928       0.00
    4929       0.00
    4930       0.00
    4931       0.00
    4932       0.00
    4933       0.00
    4934       0.00
    4935       0.00
    4936       0.00
    4937       0.00
    4938       9.99
    4939       0.00
    4940       0.00
    4941       0.00
    4942       0.00
    4943       0.00
    4944       0.00
    4945       0.00
    4946       1.49
    4947       0.00
    4948       0.00
    4949       0.00
    4950       0.00
    4951       1.99
    4952       2.99
    4953       0.00
    4954       0.00
    4955       2.99
    4956       0.00
    4957       0.00
    4958       1.99
    4959       0.00
    4960       0.00
    4961       0.00
    4962       2.99
    4963       3.99
    4964       2.99
    4965       0.00
    4966       0.00
    4967       0.00
    4968       0.00
    4969       0.00
    4970       0.00
    4971       0.99
    4972       0.00
    4973       6.49
    4974       0.00
    4975       0.00
    4976       0.00
    4977       1.29
    4978       0.00
    4979       0.00
    4980       2.99
    4981       0.00
    4982       0.00
    4983       0.99
    4984       0.00
    4985       0.00
    4986       0.00
    4987       0.00
    4988       0.99
    4989       0.00
    4990       0.00
    4991       0.00
    4992       0.00
    4993       0.00
    4994       0.00
    4995       0.00
    4996       0.00
    4997       0.00
    4998       0.00
    4999       0.00
    5000       0.00
    5001       0.00
    5002       0.00
    5003       0.00
    5004       0.00
    5005       0.99
    5006       0.00
    5007       0.00
    5008       0.00
    5009       0.00
    5010       0.00
    5011       0.00
    5012       0.00
    5013       0.00
    5014       0.00
    5015       0.00
    5016       0.00
    5017       0.00
    5018       0.00
    5019       0.00
    5020       0.00
    5021       0.00
    5022       0.00
    5023       0.00
    5024       0.00
    5025       0.00
    5026       0.00
    5027       0.00
    5028       0.00
    5029       0.00
    5030       0.00
    5031       0.00
    5032       0.00
    5033       0.00
    5034       0.00
    5035       0.00
    5036      19.99
    5037       0.00
    5038       0.00
    5039       0.00
    5040       4.99
    5041       0.00
    5042       0.00
    5043       0.00
    5044       0.00
    5045       0.00
    5046       0.00
    5047       0.00
    5048       0.00
    5049       0.00
    5050       0.00
    5051       5.00
    5052       0.00
    5053       0.00
    5054       0.00
    5055       0.00
    5056       0.00
    5057       0.00
    5058       0.00
    5059       0.00
    5060       0.00
    5061       0.00
    5062       0.00
    5063       0.00
    5064       0.00
    5065       0.00
    5066       0.00
    5067       0.00
    5068       0.00
    5069       0.00
    5070       0.00
    5071       0.00
    5072       0.00
    5073       0.00
    5074       0.00
    5075       0.00
    5076       0.00
    5077       0.00
    5078       0.00
    5079       0.00
    5080       0.00
    5081       0.00
    5082       0.00
    5083       4.99
    5084       0.00
    5085       0.00
    5086       0.00
    5087       0.00
    5088       0.00
    5089       0.00
    5090       0.00
    5091       0.00
    5092       0.00
    5093       0.00
    5094       0.99
    5095       0.00
    5096       0.00
    5097       0.00
    5098       0.00
    5099       0.00
    5100       0.00
    5101       0.00
    5102       0.00
    5103       0.00
    5104       0.00
    5105       0.00
    5106       0.00
    5107       0.00
    5108       0.00
    5109       0.00
    5110       0.00
    5111       0.00
    5112       0.00
    5113       0.00
    5114       0.00
    5115       0.00
    5116       0.00
    5117       0.00
    5118       0.00
    5119       0.00
    5120       0.00
    5121       0.00
    5122       0.00
    5123       0.00
    5124       0.00
    5125       0.00
    5126       0.99
    5127       0.00
    5128       0.00
    5129       0.00
    5130       0.00
    5131       0.00
    5132       0.00
    5133       0.00
    5134       0.00
    5135       0.00
    5136       0.00
    5137       0.00
    5138       0.00
    5139       0.00
    5140       0.00
    5141       0.00
    5142       0.00
    5143       0.00
    5144       4.99
    5145       0.00
    5146       0.00
    5147       0.00
    5148       0.00
    5149       0.00
    5150       0.00
    5151       0.00
    5152       0.00
    5153       0.00
    5154       0.00
    5155       0.00
    5156       0.00
    5157       0.00
    5158       0.00
    5159       0.00
    5160       0.00
    5161       0.00
    5162       0.00
    5163       0.00
    5164       0.00
    5165       0.00
    5166       0.00
    5167       0.00
    5168       0.00
    5169       0.00
    5170       0.00
    5171       0.00
    5172       0.00
    5173       0.00
    5174       0.00
    5175       0.00
    5176       0.00
    5177       0.00
    5178       0.00
    5179       0.00
    5180       0.00
    5181       1.99
    5182       0.00
    5183       0.00
    5184       0.00
    5185       0.00
    5186       0.00
    5187       0.00
    5188       0.00
    5189       0.00
    5190       0.00
    5191       0.00
    5192       0.00
    5193       0.00
    5194       0.00
    5195       0.00
    5196       0.00
    5197       0.00
    5198       0.00
    5199       0.00
    5200       0.00
    5201       0.00
    5202       0.00
    5203       0.00
    5204       0.00
    5205       0.00
    5206       0.00
    5207       0.00
    5208       0.00
    5209       0.00
    5210       0.00
    5211       0.00
    5212       0.00
    5213       0.00
    5214       0.00
    5215       2.99
    5216       0.00
    5217       2.99
    5218       0.00
    5219       0.00
    5220       0.00
    5221       2.49
    5222       0.00
    5223       0.00
    5224       1.99
    5225       0.00
    5226       0.00
    5227       0.00
    5228       0.00
    5229       0.00
    5230       0.00
    5231       0.00
    5232       0.00
    5233       0.00
    5234       0.00
    5235       0.00
    5236       0.00
    5237       3.99
    5238       0.00
    5239       0.00
    5240       0.00
    5241       0.00
    5242       0.00
    5243       0.00
    5244       0.00
    5245       0.00
    5246       4.99
    5247       0.00
    5248       0.00
    5249       0.00
    5250       0.00
    5251       0.00
    5252       0.99
    5253       0.00
    5254       0.00
    5255       0.00
    5256       0.00
    5257       0.99
    5258       0.00
    5259       0.00
    5260       0.99
    5261       0.00
    5262       0.00
    5263       0.99
    5264       0.00
    5265       0.99
    5266       0.00
    5267       0.00
    5268       0.00
    5269       0.99
    5270       0.00
    5271       0.00
    5272       0.00
    5273       0.00
    5274       0.00
    5275       0.00
    5276       0.00
    5277       0.00
    5278       0.00
    5279       0.00
    5280       0.00
    5281       0.00
    5282       0.00
    5283       0.00
    5284       0.00
    5285       0.00
    5286       0.00
    5287       0.00
    5288       0.00
    5289       0.00
    5290       0.00
    5291       0.00
    5292       0.00
    5293       0.00
    5294       0.00
    5295       0.00
    5296       0.00
    5297       0.00
    5298       0.00
    5299       0.00
    5300       0.00
    5301       0.00
    5302       0.00
    5303       0.00
    5304       0.00
    5305       0.00
    5306       0.00
    5307      13.99
    5308       0.00
    5309       0.00
    5310       0.00
    5311       0.00
    5312       0.00
    5313       0.00
    5314       0.00
    5315       0.00
    5316       0.00
    5317       0.00
    5318       0.00
    5319       0.00
    5320       0.00
    5321       0.00
    5322       0.00
    5323       0.00
    5324       0.00
    5325       0.00
    5326       0.00
    5327       0.00
    5328       0.00
    5329       0.00
    5330       0.00
    5331       0.00
    5332       0.00
    5333       0.00
    5334       0.00
    5335       0.00
    5336       0.00
    5337       0.00
    5338       0.00
    5339       0.00
    5340       0.00
    5341       4.49
    5342       0.00
    5343       0.00
    5344       0.00
    5345       0.00
    5346       0.00
    5347       0.00
    5348       0.00
    5349       0.00
    5350       0.00
    5351     399.99
    5352       0.00
    5353       0.00
    5354     399.99
    5355     299.99
    5356     399.99
    5357     379.99
    5358     399.99
    5359     399.99
    5360      37.99
    5361      18.99
    5362     399.99
    5363       0.00
    5364     399.99
    5365       0.00
    5366     389.99
    5367       4.99
    5368       0.00
    5369     399.99
    5370       0.00
    5371       2.49
    5372       0.00
    5373     399.99
    5374       0.00
    5375       0.00
    5376       0.00
    5377       0.00
    5378       0.00
    5379       0.00
    5380       0.00
    5381       0.00
    5382       0.00
    5383       0.00
    5384       0.00
    5385       0.00
    5386       0.00
    5387       0.00
    5388       0.00
    5389       0.00
    5390       0.00
    5391       0.00
    5392       0.00
    5393       0.00
    5394       0.00
    5395       0.00
    5396       0.00
    5397       0.00
    5398       0.00
    5399       0.00
    5400       0.00
    5401       0.00
    5402       0.00
    5403       0.00
    5404       0.00
    5405       0.00
    5406       0.00
    5407       0.00
    5408       0.00
    5409       0.00
    5410       0.00
    5411       2.99
    5412       2.99
    5413       0.00
    5414       2.99
    5415       0.00
    5416       0.00
    5417       0.00
    5418       0.00
    5419       0.00
    5420       0.00
    5421       0.00
    5422       0.00
    5423       0.00
    5424       0.00
    5425       0.00
    5426       0.00
    5427       0.00
    5428       0.00
    5429       0.00
    5430       0.00
    5431       0.00
    5432       0.00
    5433       0.00
    5434       0.00
    5435       0.00
    5436       0.00
    5437       0.00
    5438       0.00
    5439       0.00
    5440       0.00
    5441       0.00
    5442       0.00
    5443       0.00
    5444       0.00
    5445       0.00
    5446       0.00
    5447       0.00
    5448       0.00
    5449       0.00
    5450       0.00
    5451       0.00
    5452       0.00
    5453       0.00
    5454       0.00
    5455       0.00
    5456       0.00
    5457       0.00
    5458       0.00
    5459       0.00
    5460       0.00
    5461       0.00
    5462       0.00
    5463       0.00
    5464       0.00
    5465       2.99
    5466       2.99
    5467       0.00
    5468       0.00
    5469       0.00
    5470       0.00
    5471       0.00
    5472       0.00
    5473       0.00
    5474       0.00
    5475       9.99
    5476       1.49
    5477       0.00
    5478       0.00
    5479       4.99
    5480       0.99
    5481       0.00
    5482       4.99
    5483       0.00
    5484       0.00
    5485       0.00
    5486       1.99
    5487       0.00
    5488       0.00
    5489      29.99
    5490       1.99
    5491       0.00
    5492       0.00
    5493       0.00
    5494       0.00
    5495       0.00
    5496       0.00
    5497       0.00
    5498       0.00
    5499       0.00
    5500       2.99
    5501       0.00
    5502       0.00
    5503       0.00
    5504       0.00
    5505       0.00
    5506       0.00
    5507       0.00
    5508       0.00
    5509       0.00
    5510       0.00
    5511       0.00
    5512       0.00
    5513       0.00
    5514       0.00
    5515       0.00
    5516       0.00
    5517       0.00
    5518       0.00
    5519       0.00
    5520       0.00
    5521       0.00
    5522       0.00
    5523       0.00
    5524       0.00
    5525       0.00
    5526       0.00
    5527       0.00
    5528       0.00
    5529       0.00
    5530       0.00
    5531       0.00
    5532       0.00
    5533       0.00
    5534       0.00
    5535       0.00
    5536       0.00
    5537       0.00
    5538       0.00
    5539       0.00
    5540       0.00
    5541       0.00
    5542       0.00
    5543       0.00
    5544       0.00
    5545       0.00
    5546       0.00
    5547       0.00
    5548       0.00
    5549       0.00
    5550       0.00
    5551       0.00
    5552       0.00
    5553       0.00
    5554       0.00
    5555       0.00
    5556       0.00
    5557       0.99
    5558       0.00
    5559       0.00
    5560       0.00
    5561       0.00
    5562       0.00
    5563       0.00
    5564       0.00
    5565       0.00
    5566       0.00
    5567       0.00
    5568       0.00
    5569       0.00
    5570       0.00
    5571       0.00
    5572       0.00
    5573       0.00
    5574       0.00
    5575       0.00
    5576       0.00
    5577       0.00
    5578       5.99
    5579       0.00
    5580       0.00
    5581       0.00
    5582       0.00
    5583       0.00
    5584       0.00
    5585       0.00
    5586       1.99
    5587       0.00
    5588       0.00
    5589       0.00
    5590       0.00
    5591       0.00
    5592       0.00
    5593       0.00
    5594       0.00
    5595       0.00
    5596       0.00
    5597       0.00
    5598       0.00
    5599       0.00
    5600       0.00
    5601       0.00
    5602       0.00
    5603       0.00
    5604       0.00
    5605       0.00
    5606       0.00
    5607       0.00
    5608       0.00
    5609       0.00
    5610       0.00
    5611       0.00
    5612       0.00
    5613       0.00
    5614       0.00
    5615       0.00
    5616       0.00
    5617       0.00
    5618       0.00
    5619       0.00
    5620       0.00
    5621       0.00
    5622       0.00
    5623       0.00
    5624       0.00
    5625       0.00
    5626       0.00
    5627       2.99
    5628       0.00
    5629       0.00
    5630       0.00
    5631       2.99
    5632       0.00
    5633       0.00
    5634       0.00
    5635       0.00
    5636       0.00
    5637       0.00
    5638       0.00
    5639       0.00
    5640       0.00
    5641       2.99
    5642       0.00
    5643       0.00
    5644       0.00
    5645       2.99
    5646       0.00
    5647       0.00
    5648       2.99
    5649       0.00
    5650       0.00
    5651       0.00
    5652       0.00
    5653       0.00
    5654       0.00
    5655       0.00
    5656       0.00
    5657       0.00
    5658       0.00
    5659       0.00
    5660       0.99
    5661       0.00
    5662       0.00
    5663       0.00
    5664       0.00
    5665       0.00
    5666       0.00
    5667       0.00
    5668       0.00
    5669       0.00
    5670       0.00
    5671       0.00
    5672       0.00
    5673       0.00
    5674       0.00
    5675       0.00
    5676       0.00
    5677       0.00
    5678       0.00
    5679       0.00
    5680       0.00
    5681       0.00
    5682       0.00
    5683       0.00
    5684       0.00
    5685       0.00
    5686       0.00
    5687       0.00
    5688       0.00
    5689       0.00
    5690       0.00
    5691       0.00
    5692       0.00
    5693       0.00
    5694       0.00
    5695       0.00
    5696       0.00
    5697       0.00
    5698       0.00
    5699       0.00
    5700       0.00
    5701       2.49
    5702       0.00
    5703       0.00
    5704       0.00
    5705       0.00
    5706       0.00
    5707       0.00
    5708       0.00
    5709       0.00
    5710       0.00
    5711       0.00
    5712       2.99
    5713       0.00
    5714       0.00
    5715       0.00
    5716       0.00
    5717       0.00
    5718       0.00
    5719       0.00
    5720       0.00
    5721       0.00
    5722       0.00
    5723       0.00
    5724       0.00
    5725       0.00
    5726       0.00
    5727       0.00
    5728       0.00
    5729       0.00
    5730       0.00
    5731       0.00
    5732       0.00
    5733       0.00
    5734       0.00
    5735       0.00
    5736       0.00
    5737       0.00
    5738       0.00
    5739      19.90
    5740       0.00
    5741       0.00
    5742       0.00
    5743       0.00
    5744       0.00
    5745       0.00
    5746       0.00
    5747       0.00
    5748       0.00
    5749       0.00
    5750       0.00
    5751       1.99
    5752       0.00
    5753       0.00
    5754       0.00
    5755       0.00
    5756       1.99
    5757       0.00
    5758       0.00
    5759       0.00
    5760       0.00
    5761       0.00
    5762       0.00
    5763       0.00
    5764       0.00
    5765       0.00
    5766       0.00
    5767       0.99
    5768       0.00
    5769       0.00
    5770       0.00
    5771       0.00
    5772       1.99
    5773       0.00
    5774       0.00
    5775       0.00
    5776       0.00
    5777       0.99
    5778       0.00
    5779       0.00
    5780       0.00
    5781       0.00
    5782       0.00
    5783       0.00
    5784       0.00
    5785       0.00
    5786       0.00
    5787       0.00
    5788       0.00
    5789       0.00
    5790       0.00
    5791       0.00
    5792       0.00
    5793       0.00
    5794       0.00
    5795       0.00
    5796       0.00
    5797       0.00
    5798       0.00
    5799       0.00
    5800       0.00
    5801       0.00
    5802       0.00
    5803       0.00
    5804       0.99
    5805       0.00
    5806       0.00
    5807       0.00
    5808       0.00
    5809       0.00
    5810       0.00
    5811       0.00
    5812       0.99
    5813       0.00
    5814       0.00
    5815       0.00
    5816       0.00
    5817       0.00
    5818       0.00
    5819       0.00
    5820       0.00
    5821       0.00
    5822       0.00
    5823       0.00
    5824       0.00
    5825       0.00
    5826       0.00
    5827       0.00
    5828       0.00
    5829       0.00
    5830       0.00
    5831       0.00
    5832       1.99
    5833       0.00
    5834       0.99
    5835       0.00
    5836       0.00
    5837       0.00
    5838       0.00
    5839       0.00
    5840       0.00
    5841       0.00
    5842       0.00
    5843       0.00
    5844       0.00
    5845       0.00
    5846       0.99
    5847       4.99
    5848       0.00
    5849       0.00
    5850       0.00
    5851       0.00
    5852       0.00
    5853       0.00
    5854       0.00
    5855       0.00
    5856       0.00
    5857       0.00
    5858       0.00
    5859       0.00
    5860       0.00
    5861       0.00
    5862       0.00
    5863       0.00
    5864       0.00
    5865       0.00
    5866       0.00
    5867       0.00
    5868       0.00
    5869       0.00
    5870       0.00
    5871       0.00
    5872       0.00
    5873       0.00
    5874       0.00
    5875       0.00
    5876       0.00
    5877       0.00
    5878       0.00
    5879       0.00
    5880       0.00
    5881       0.00
    5882       0.00
    5883       0.00
    5884       0.00
    5885       0.00
    5886       0.00
    5887       0.00
    5888       0.00
    5889       0.00
    5890       0.00
    5891       0.00
    5892       0.00
    5893       0.00
    5894       0.00
    5895       0.00
    5896       0.00
    5897       0.00
    5898       0.00
    5899       0.00
    5900       0.00
    5901       0.00
    5902       0.00
    5903       0.00
    5904       0.00
    5905       0.00
    5906       0.00
    5907       0.00
    5908       0.00
    5909       8.49
    5910       0.00
    5911       3.99
    5912       3.99
    5913       0.00
    5914       1.99
    5915       0.00
    5916       0.00
    5917       1.49
    5918       0.00
    5919       0.00
    5920       0.00
    5921       0.00
    5922       0.00
    5923       0.00
    5924       1.99
    5925       0.00
    5926       0.00
    5927       1.49
    5928       0.00
    5929       0.00
    5930       0.00
    5931       0.00
    5932       0.00
    5933       0.00
    5934       0.00
    5935       0.00
    5936       0.00
    5937       0.00
    5938       0.00
    5939       0.00
    5940       0.00
    5941       0.00
    5942       2.99
    5943       0.00
    5944       0.00
    5945       2.99
    5946       0.00
    5947       0.00
    5948       0.00
    5949       0.00
    5950       0.00
    5951       0.00
    5952       0.00
    5953       0.00
    5954       0.00
    5955       2.99
    5956       0.00
    5957       4.49
    5958       0.00
    5959       3.99
    5960       0.00
    5961       0.99
    5962       0.00
    5963       0.00
    5964       0.00
    5965       0.00
    5966       1.75
    5967       0.00
    5968       0.00
    5969       0.00
    5970       0.00
    5971       0.00
    5972       0.00
    5973       4.99
    5974       0.00
    5975       0.00
    5976       0.00
    5977       1.49
    5978       1.49
    5979       0.00
    5980       0.00
    5981       0.00
    5982       4.99
    5983       0.00
    5984       0.00
    5985       0.00
    5986       0.00
    5987       0.00
    5988       0.00
    5989       0.00
    5990       0.00
    5991       0.00
    5992       0.00
    5993       0.00
    5994      11.99
    5995       0.00
    5996       0.00
    5997       0.00
    5998       0.00
    5999       0.00
    6000       0.00
    6001       0.00
    6002       0.00
    6003       0.00
    6004       0.00
    6005       0.00
    6006       0.00
    6007       0.00
    6008       0.00
    6009       0.00
    6010       0.00
    6011       0.00
    6012       0.00
    6013       0.00
    6014       0.00
    6015       0.00
    6016       0.00
    6017       0.00
    6018       0.00
    6019       0.00
    6020       0.00
    6021       0.00
    6022       0.00
    6023       0.00
    6024       0.00
    6025       0.00
    6026       0.00
    6027       0.00
    6028       0.00
    6029       0.00
    6030       0.00
    6031       0.00
    6032       0.00
    6033       0.00
    6034       0.00
    6035       0.00
    6036       0.00
    6037       0.00
    6038       0.00
    6039       0.00
    6040       0.00
    6041       0.00
    6042       0.00
    6043       0.00
    6044       0.00
    6045       0.00
    6046       0.00
    6047       0.00
    6048       0.00
    6049       0.00
    6050       0.00
    6051       0.00
    6052       0.00
    6053       0.00
    6054       0.00
    6055       0.00
    6056       1.99
    6057       0.00
    6058       0.00
    6059       0.00
    6060       0.00
    6061       0.00
    6062       0.00
    6063       0.00
    6064       0.00
    6065       0.00
    6066       0.00
    6067       0.00
    6068       0.00
    6069       0.00
    6070       0.00
    6071       0.00
    6072       0.00
    6073       0.00
    6074       0.00
    6075       0.00
    6076       0.00
    6077       0.00
    6078       0.00
    6079       0.00
    6080       0.00
    6081       0.00
    6082       0.00
    6083       0.00
    6084       0.00
    6085       0.00
    6086       0.00
    6087       0.00
    6088       0.99
    6089       0.00
    6090       0.00
    6091       0.00
    6092       0.00
    6093       0.00
    6094       0.00
    6095       0.00
    6096       0.00
    6097       0.00
    6098       0.00
    6099       0.00
    6100       0.00
    6101       0.00
    6102       0.99
    6103       0.00
    6104       0.00
    6105       0.00
    6106       0.00
    6107       0.00
    6108       0.00
    6109       0.00
    6110       0.00
    6111       0.00
    6112       0.00
    6113       0.00
    6114       0.00
    6115       0.99
    6116       0.00
    6117       0.00
    6118       0.00
    6119       0.00
    6120       1.99
    6121       0.00
    6122       0.00
    6123       0.00
    6124       0.00
    6125       0.00
    6126       0.00
    6127       0.00
    6128       0.00
    6129       0.00
    6130       0.00
    6131       0.00
    6132       5.99
    6133       0.00
    6134       0.00
    6135       0.00
    6136       0.00
    6137       0.00
    6138       0.00
    6139       0.00
    6140       2.99
    6141       0.00
    6142       0.00
    6143       0.00
    6144       0.00
    6145       0.00
    6146       0.00
    6147       0.00
    6148       0.00
    6149       0.00
    6150       0.00
    6151       0.00
    6152       0.00
    6153       0.00
    6154       0.00
    6155       0.00
    6156       0.00
    6157       0.00
    6158       0.00
    6159       0.00
    6160       0.00
    6161       0.00
    6162       0.00
    6163       0.00
    6164       0.00
    6165       0.00
    6166       0.00
    6167       0.00
    6168       0.00
    6169       0.00
    6170       0.00
    6171       0.00
    6172       0.00
    6173       0.00
    6174       0.00
    6175       0.00
    6176       0.00
    6177       0.00
    6178       0.00
    6179       1.49
    6180       9.99
    6181       4.99
    6182       0.00
    6183       0.00
    6184       0.00
    6185       0.00
    6186       0.00
    6187       0.00
    6188       0.00
    6189       0.00
    6190       0.00
    6191       0.00
    6192       0.00
    6193       0.00
    6194       0.00
    6195       0.00
    6196       0.00
    6197       0.00
    6198       7.99
    6199       0.00
    6200       0.00
    6201       3.99
    6202       4.99
    6203       0.00
    6204       0.00
    6205       9.99
    6206       0.00
    6207       0.00
    6208       0.00
    6209       0.00
    6210       0.00
    6211       0.00
    6212       0.00
    6213       0.00
    6214       0.00
    6215       0.00
    6216       0.00
    6217       2.49
    6218       0.00
    6219       0.00
    6220       0.00
    6221       0.00
    6222       0.00
    6223       0.00
    6224       0.00
    6225       0.00
    6226       0.00
    6227       0.00
    6228       0.00
    6229       0.00
    6230       0.00
    6231       0.00
    6232       0.00
    6233       0.00
    6234       0.00
    6235       0.00
    6236       0.00
    6237       0.00
    6238       0.00
    6239       0.00
    6240       0.00
    6241       0.00
    6242       0.00
    6243       0.00
    6244       0.00
    6245       0.00
    6246       0.00
    6247       0.00
    6248       0.00
    6249       0.00
    6250       0.00
    6251       0.00
    6252       4.99
    6253       0.00
    6254       0.00
    6255       0.00
    6256       0.00
    6257       0.00
    6258       0.00
    6259       0.00
    6260       0.00
    6261       0.00
    6262       0.00
    6263       0.00
    6264       0.00
    6265       0.00
    6266       0.00
    6267       0.00
    6268       0.00
    6269       0.00
    6270       0.00
    6271       0.00
    6272       0.00
    6273       0.00
    6274       0.00
    6275       0.00
    6276       0.00
    6277       0.99
    6278       0.00
    6279       0.00
    6280       0.00
    6281       0.00
    6282       0.00
    6283       0.00
    6284       0.00
    6285       0.00
    6286       0.00
    6287       0.00
    6288       0.00
    6289       0.00
    6290       0.00
    6291       0.00
    6292       0.99
    6293       0.00
    6294       0.00
    6295       0.00
    6296       0.00
    6297       0.00
    6298       0.00
    6299       0.00
    6300       0.00
    6301       0.00
    6302       0.00
    6303       0.00
    6304       0.00
    6305       0.00
    6306       0.00
    6307       0.00
    6308       0.00
    6309       0.00
    6310       0.00
    6311       4.49
    6312       0.00
    6313       0.00
    6314       0.00
    6315       0.00
    6316       0.00
    6317       0.00
    6318       0.00
    6319       0.00
    6320       0.00
    6321       0.00
    6322       0.00
    6323       0.00
    6324       0.00
    6325       0.00
    6326       0.00
    6327       0.00
    6328       0.00
    6329       0.00
    6330       0.00
    6331       0.00
    6332       0.00
    6333       0.00
    6334       0.00
    6335       0.00
    6336       0.00
    6337       0.00
    6338       0.00
    6339       0.00
    6340       0.00
    6341      14.00
    6342       0.00
    6343       0.00
    6344       0.00
    6345       0.00
    6346       2.99
    6347       0.00
    6348       0.00
    6349       0.00
    6350       0.00
    6351       0.00
    6352       0.00
    6353       0.00
    6354       0.00
    6355       0.00
    6356       0.00
    6357       0.00
    6358       0.00
    6359       0.00
    6360       0.00
    6361       1.99
    6362       0.00
    6363       0.00
    6364       0.00
    6365       0.00
    6366       4.85
    6367       0.00
    6368       0.00
    6369       0.00
    6370       0.00
    6371       0.00
    6372       0.00
    6373       0.00
    6374       0.00
    6375       0.00
    6376       0.00
    6377       0.00
    6378       0.00
    6379       0.00
    6380       0.00
    6381       0.00
    6382       0.00
    6383       0.00
    6384       0.00
    6385       0.00
    6386       0.00
    6387       0.00
    6388       0.00
    6389       0.99
    6390       0.00
    6391       0.00
    6392       0.00
    6393       0.00
    6394       0.00
    6395       0.00
    6396       0.00
    6397       0.00
    6398       0.00
    6399       0.00
    6400       0.00
    6401       0.00
    6402       0.00
    6403       0.00
    6404       0.00
    6405       0.00
    6406       0.00
    6407       0.00
    6408       0.00
    6409       0.00
    6410       0.00
    6411       0.00
    6412       0.00
    6413      14.99
    6414       0.00
    6415       0.00
    6416       0.00
    6417       0.00
    6418       0.00
    6419       0.00
    6420       0.00
    6421       0.00
    6422       0.00
    6423       0.00
    6424       0.00
    6425       3.99
    6426       0.00
    6427       0.00
    6428       1.99
    6429       3.99
    6430       0.00
    6431       0.00
    6432       2.00
    6433       0.00
    6434       0.00
    6435       0.00
    6436       0.00
    6437       0.00
    6438       0.00
    6439       0.00
    6440       1.99
    6441       0.00
    6442       0.00
    6443       0.00
    6444       0.00
    6445       0.00
    6446       3.99
    6447       0.00
    6448       0.00
    6449       0.00
    6450       0.00
    6451       0.00
    6452       0.00
    6453       0.00
    6454       0.00
    6455       0.00
    6456       0.00
    6457       2.99
    6458       0.00
    6459       3.99
    6460       0.00
    6461       0.00
    6462       0.00
    6463       0.00
    6464       0.00
    6465       0.00
    6466       0.00
    6467       0.00
    6468       0.00
    6469       0.00
    6470       0.00
    6471       0.00
    6472       0.00
    6473       0.00
    6474       0.00
    6475       0.00
    6476       0.00
    6477       0.00
    6478       0.00
    6479       0.00
    6480       0.00
    6481       0.00
    6482       0.00
    6483       0.00
    6484       0.00
    6485       0.00
    6486       0.00
    6487       0.00
    6488       0.00
    6489       0.00
    6490       0.00
    6491       0.99
    6492       0.00
    6493       0.00
    6494       0.00
    6495       0.00
    6496       0.99
    6497       0.00
    6498       0.00
    6499       0.00
    6500       0.00
    6501       0.00
    6502       0.00
    6503       0.00
    6504       0.00
    6505       0.00
    6506       0.00
    6507       0.00
    6508       0.99
    6509       0.00
    6510       0.00
    6511       0.00
    6512       0.00
    6513       0.00
    6514       0.00
    6515       0.00
    6516       0.00
    6517       0.00
    6518       0.00
    6519       0.00
    6520       0.00
    6521       0.00
    6522       0.00
    6523       0.00
    6524       0.00
    6525       0.00
    6526       0.00
    6527       0.00
    6528       0.00
    6529       0.00
    6530       0.00
    6531       0.00
    6532       0.00
    6533       0.00
    6534       6.99
    6535       0.00
    6536       0.00
    6537       0.00
    6538       6.99
    6539       0.00
    6540       0.00
    6541       0.00
    6542       0.00
    6543       0.00
    6544       0.00
    6545       0.00
    6546       0.00
    6547       2.99
    6548       0.00
    6549       2.99
    6550       0.00
    6551       0.00
    6552       0.00
    6553       2.49
    6554       0.00
    6555       1.99
    6556       0.00
    6557       0.00
    6558       0.00
    6559      46.99
    6560       0.99
    6561       0.00
    6562       0.00
    6563       0.00
    6564       0.00
    6565       2.49
    6566       0.00
    6567       0.99
    6568       0.00
    6569       0.00
    6570       2.99
    6571       0.00
    6572       0.00
    6573       0.00
    6574       0.00
    6575       0.00
    6576       0.00
    6577       0.00
    6578       0.00
    6579       0.00
    6580       0.00
    6581       0.00
    6582       0.00
    6583       0.00
    6584       0.99
    6585       0.00
    6586       0.00
    6587       0.00
    6588       0.00
    6589       0.00
    6590       0.00
    6591       0.00
    6592       0.00
    6593       0.00
    6594       0.00
    6595       0.99
    6596       5.99
    6597       0.00
    6598       0.00
    6599       0.00
    6600       0.00
    6601       0.00
    6602       0.00
    6603       0.00
    6604       0.00
    6605       0.00
    6606       0.00
    6607       0.00
    6608       0.00
    6609       0.00
    6610       0.00
    6611       0.00
    6612       0.00
    6613       0.00
    6614       0.00
    6615       0.00
    6616       0.00
    6617       0.00
    6618       5.49
    6619       0.00
    6620       0.00
    6621       0.00
    6622       0.00
    6623       0.00
    6624     109.99
    6625       0.00
    6626       0.00
    6627       0.00
    6628       0.00
    6629       0.00
    6630       0.00
    6631       0.00
    6632       0.00
    6633       0.00
    6634       0.00
    6635       0.00
    6636       0.00
    6637       0.00
    6638       0.00
    6639       0.00
    6640       0.00
    6641       0.00
    6642       0.00
    6643       0.00
    6644       0.00
    6645       0.00
    6646       0.00
    6647       0.00
    6648       0.00
    6649       0.00
    6650       0.00
    6651       0.00
    6652       3.95
    6653       0.00
    6654       0.00
    6655       0.00
    6656       0.00
    6657       0.00
    6658       0.00
    6659       0.00
    6660       0.00
    6661       0.00
    6662       0.00
    6663       0.00
    6664       0.00
    6665       0.00
    6666       0.00
    6667       0.00
    6668       0.00
    6669       0.00
    6670       0.00
    6671       0.00
    6672       0.00
    6673       0.00
    6674       0.00
    6675       0.99
    6676       0.00
    6677       0.00
    6678       0.00
    6679       0.00
    6680       4.49
    6681       0.00
    6682       0.00
    6683       0.00
    6684       0.00
    6685       0.00
    6686       0.00
    6687       1.49
    6688       0.00
    6689       0.00
    6690       0.00
    6691       0.00
    6692     154.99
    6693       0.00
    6694       0.00
    6695       0.00
    6696       0.00
    6697       0.00
    6698       0.00
    6699       0.00
    6700       0.00
    6701       0.00
    6702       0.00
    6703       0.00
    6704       0.00
    6705       0.00
    6706       0.00
    6707       0.00
    6708       0.00
    6709       0.00
    6710       0.00
    6711       0.00
    6712       0.00
    6713       0.00
    6714       0.00
    6715       0.00
    6716       0.00
    6717       0.00
    6718       0.00
    6719       0.00
    6720       0.00
    6721       0.00
    6722       5.99
    6723       0.00
    6724       0.00
    6725       0.00
    6726       0.00
    6727       0.00
    6728       0.00
    6729       0.00
    6730       0.00
    6731       0.00
    6732       0.99
    6733       0.00
    6734       0.00
    6735       0.00
    6736       0.00
    6737       0.00
    6738       0.00
    6739       0.00
    6740       0.00
    6741       0.00
    6742       0.00
    6743       0.00
    6744       0.00
    6745       0.00
    6746       0.00
    6747       0.00
    6748       0.00
    6749       0.00
    6750       0.00
    6751       0.00
    6752       0.00
    6753       1.49
    6754       0.00
    6755       0.00
    6756       0.00
    6757       0.00
    6758       0.00
    6759       0.00
    6760       3.08
    6761       0.00
    6762       0.00
    6763       0.00
    6764       0.00
    6765       0.00
    6766       0.99
    6767       0.00
    6768       0.00
    6769       0.00
    6770       0.00
    6771       0.00
    6772       0.00
    6773       0.00
    6774       0.00
    6775       0.00
    6776       0.00
    6777       0.00
    6778       0.00
    6779       0.00
    6780       0.00
    6781       0.00
    6782       0.00
    6783       0.00
    6784       0.00
    6785       0.00
    6786       0.00
    6787       0.00
    6788       0.00
    6789       0.00
    6790       0.00
    6791       0.00
    6792       0.00
    6793       0.00
    6794       0.00
    6795       0.00
    6796       2.59
    6797       0.00
    6798       0.00
    6799       0.00
    6800       0.00
    6801       0.00
    6802       0.00
    6803       0.00
    6804       0.00
    6805       0.00
    6806       0.00
    6807       0.00
    6808       4.80
    6809       8.99
    6810       0.00
    6811       0.00
    6812       0.00
    6813       0.00
    6814       0.00
    6815       0.00
    6816       0.00
    6817       0.00
    6818       0.00
    6819       0.00
    6820       0.00
    6821       0.00
    6822       0.00
    6823       0.00
    6824       0.00
    6825       0.00
    6826       0.00
    6827       0.00
    6828       0.00
    6829       0.00
    6830       0.00
    6831       0.00
    6832       0.00
    6833       0.00
    6834       0.00
    6835       0.00
    6836       0.00
    6837      17.99
    6838       0.00
    6839       0.00
    6840       0.00
    6841       0.00
    6842       0.00
    6843       0.00
    6844       0.00
    6845       0.00
    6846       0.00
    6847       0.00
    6848       0.00
    6849       0.00
    6850       0.00
    6851       0.00
    6852       0.00
    6853       0.00
    6854       0.00
    6855       0.00
    6856       0.00
    6857       0.00
    6858       0.00
    6859       0.00
    6860       0.00
    6861       0.00
    6862       0.00
    6863       0.00
    6864       0.00
    6865       0.00
    6866       0.00
    6867       0.00
    6868       0.00
    6869       0.00
    6870       0.00
    6871       0.00
    6872       0.00
    6873       0.00
    6874       0.00
    6875       0.00
    6876       0.00
    6877       0.00
    6878       0.00
    6879       0.00
    6880       0.00
    6881       0.00
    6882       0.00
    6883       0.00
    6884       0.00
    6885       0.00
    6886       2.99
    6887       0.00
    6888       0.00
    6889       0.00
    6890       0.00
    6891       0.00
    6892       0.00
    6893       0.00
    6894       0.00
    6895       3.49
    6896       0.00
    6897       0.99
    6898       0.00
    6899       0.00
    6900       0.00
    6901       0.00
    6902       0.00
    6903       0.00
    6904       0.00
    6905       0.00
    6906       0.00
    6907       0.00
    6908       0.00
    6909       0.00
    6910       0.00
    6911       2.49
    6912       0.00
    6913       0.00
    6914       0.00
    6915       0.00
    6916       0.00
    6917       0.00
    6918       0.00
    6919       0.99
    6920       0.00
    6921       0.00
    6922       0.00
    6923       0.00
    6924       0.00
    6925       1.49
    6926       0.00
    6927       0.00
    6928       0.00
    6929       0.99
    6930       0.00
    6931       0.00
    6932       0.00
    6933       0.00
    6934       0.00
    6935       0.00
    6936       0.99
    6937       0.00
    6938       0.00
    6939       0.00
    6940       0.00
    6941       0.00
    6942       0.00
    6943       2.99
    6944       0.00
    6945       0.00
    6946       0.00
    6947       0.00
    6948       4.99
    6949       0.00
    6950       0.00
    6951       0.00
    6952       0.00
    6953       0.00
    6954       0.00
    6955       0.00
    6956       0.00
    6957       0.00
    6958       0.00
    6959       0.00
    6960       0.00
    6961       0.00
    6962       0.00
    6963       0.00
    6964       0.00
    6965       0.00
    6966       0.00
    6967       0.00
    6968       0.00
    6969       0.00
    6970       0.00
    6971       0.00
    6972       0.00
    6973       0.00
    6974       4.99
    6975       0.00
    6976       0.00
    6977       0.00
    6978       0.00
    6979       0.00
    6980       0.00
    6981       0.00
    6982       0.00
    6983       0.00
    6984       0.00
    6985       0.99
    6986       0.00
    6987       0.00
    6988       0.00
    6989       0.00
    6990       0.00
    6991       0.00
    6992       0.00
    6993       0.00
    6994       0.00
    6995       0.00
    6996       0.00
    6997       0.00
    6998       0.00
    6999       0.00
    7000       0.00
    7001       0.00
    7002       0.00
    7003       0.00
    7004       0.00
    7005       0.00
    7006       0.00
    7007       0.00
    7008       0.00
    7009       0.00
    7010       0.00
    7011       0.00
    7012       0.00
    7013       0.00
    7014       0.00
    7015       0.00
    7016       0.00
    7017       0.00
    7018       0.00
    7019       0.00
    7020       0.00
    7021       0.00
    7022       0.00
    7023       0.00
    7024       0.00
    7025       0.00
    7026       0.00
    7027       0.00
    7028       0.00
    7029       0.00
    7030       0.00
    7031       0.00
    7032       0.00
    7033       0.00
    7034       0.00
    7035       0.00
    7036       0.00
    7037       3.99
    7038       0.00
    7039       0.00
    7040       0.00
    7041       0.00
    7042       0.00
    7043       0.00
    7044       0.00
    7045       0.00
    7046       0.00
    7047       0.00
    7048       0.00
    7049       0.00
    7050       0.00
    7051       0.00
    7052       0.00
    7053       0.00
    7054       0.00
    7055       0.00
    7056       0.00
    7057       0.00
    7058       0.00
    7059       0.00
    7060       0.00
    7061       0.00
    7062       0.00
    7063       0.00
    7064       0.00
    7065       0.00
    7066       0.00
    7067       0.00
    7068       0.00
    7069       0.00
    7070       0.00
    7071       0.00
    7072       0.00
    7073       0.00
    7074       0.00
    7075       0.00
    7076       0.00
    7077       0.00
    7078       0.00
    7079       0.00
    7080       0.00
    7081       0.00
    7082       0.00
    7083       0.00
    7084       0.00
    7085       0.00
    7086       0.00
    7087       0.00
    7088       0.00
    7089       0.00
    7090       0.00
    7091       0.00
    7092       0.00
    7093       0.00
    7094       0.00
    7095       0.00
    7096       0.00
    7097       4.99
    7098       0.00
    7099       0.00
    7100       0.99
    7101       0.00
    7102       0.00
    7103       0.00
    7104       0.00
    7105       0.00
    7106       0.00
    7107       0.99
    7108       0.00
    7109       0.00
    7110       0.00
    7111       0.00
    7112       0.00
    7113       0.00
    7114       0.00
    7115       0.00
    7116       0.00
    7117       0.00
    7118       0.00
    7119       0.00
    7120       0.00
    7121       0.00
    7122       0.00
    7123       0.00
    7124       0.00
    7125       0.00
    7126       0.00
    7127       0.00
    7128       0.00
    7129       0.00
    7130       0.00
    7131       0.00
    7132       0.00
    7133       0.00
    7134       0.00
    7135       0.00
    7136       0.00
    7137       0.00
    7138       0.00
    7139       0.00
    7140       0.00
    7141       0.00
    7142       0.00
    7143       0.00
    7144       0.00
    7145       0.99
    7146       0.00
    7147       0.00
    7148       0.99
    7149       0.00
    7150       0.00
    7151       0.00
    7152       0.00
    7153       0.00
    7154       0.00
    7155       0.00
    7156       0.00
    7157       0.00
    7158       0.00
    7159       0.00
    7160       0.00
    7161       0.00
    7162       0.00
    7163       0.00
    7164       0.00
    7165       7.99
    7166       0.00
    7167       0.00
    7168       0.00
    7169       0.00
    7170       0.00
    7171       0.00
    7172       0.00
    7173       0.00
    7174       0.00
    7175       0.00
    7176       0.00
    7177       0.00
    7178       0.00
    7179       0.00
    7180       0.00
    7181       0.00
    7182       0.00
    7183       1.49
    7184       0.00
    7185       0.00
    7186       0.00
    7187       0.00
    7188       0.00
    7189       0.00
    7190       0.00
    7191       0.00
    7192       0.00
    7193       0.99
    7194       0.00
    7195       0.00
    7196       4.99
    7197       0.00
    7198       2.99
    7199       0.00
    7200       0.00
    7201       0.00
    7202       0.00
    7203       0.00
    7204       4.99
    7205       0.00
    7206       0.00
    7207       0.00
    7208       0.00
    7209       0.00
    7210       0.00
    7211       2.00
    7212       0.00
    7213       0.00
    7214       0.00
    7215       2.99
    7216       0.00
    7217       0.00
    7218       0.00
    7219       0.00
    7220       0.00
    7221       0.00
    7222       0.00
    7223       0.00
    7224       0.00
    7225       3.99
    7226       0.00
    7227       0.00
    7228       0.00
    7229       0.00
    7230       0.00
    7231       0.00
    7232       0.00
    7233       2.49
    7234       0.00
    7235       0.00
    7236       0.00
    7237       1.49
    7238       0.00
    7239       0.00
    7240       0.00
    7241       0.00
    7242       0.00
    7243       0.00
    7244       0.99
    7245       0.00
    7246       0.00
    7247       0.00
    7248       0.00
    7249       0.00
    7250       0.00
    7251       0.00
    7252       0.00
    7253       0.00
    7254       0.00
    7255       2.99
    7256       0.00
    7257       0.00
    7258       0.00
    7259       0.00
    7260       0.00
    7261       0.00
    7262       0.00
    7263       0.00
    7264       0.00
    7265       0.00
    7266       0.00
    7267       0.00
    7268       0.00
    7269       0.00
    7270       0.00
    7271       0.00
    7272       0.00
    7273       0.00
    7274       0.00
    7275       0.00
    7276       0.00
    7277       0.00
    7278       0.00
    7279       0.00
    7280       0.00
    7281       0.00
    7282       0.00
    7283       0.00
    7284       0.00
    7285       0.00
    7286       0.00
    7287       0.00
    7288       0.00
    7289       0.00
    7290       0.00
    7291       0.00
    7292       0.00
    7293       0.00
    7294       0.00
    7295       0.00
    7296       0.00
    7297       0.00
    7298       0.00
    7299       0.00
    7300       0.00
    7301       0.00
    7302       0.00
    7303       0.00
    7304       0.00
    7305       0.00
    7306       0.00
    7307       0.00
    7308       0.00
    7309       0.00
    7310       0.00
    7311       0.00
    7312       0.00
    7313       0.00
    7314       0.00
    7315       0.00
    7316       0.00
    7317       0.00
    7318       0.00
    7319       0.00
    7320       0.00
    7321       0.00
    7322       0.00
    7323       0.00
    7324       0.00
    7325       0.00
    7326       0.00
    7327       0.00
    7328       3.99
    7329       0.00
    7330       0.00
    7331       0.00
    7332       2.99
    7333       0.99
    7334       0.00
    7335       1.99
    7336       0.99
    7337       0.00
    7338       0.00
    7339       3.99
    7340       0.00
    7341       0.00
    7342       1.99
    7343       0.00
    7344       0.00
    7345       0.00
    7346       1.96
    7347      19.40
    7348       0.00
    7349       0.00
    7350       0.00
    7351       0.00
    7352       0.00
    7353       0.00
    7354       0.00
    7355       2.99
    7356       3.99
    7357      14.99
    7358       0.00
    7359       3.90
    7360       2.99
    7361       0.00
    7362       0.00
    7363       0.00
    7364       0.99
    7365       0.00
    7366       0.00
    7367       0.00
    7368       0.00
    7369       0.00
    7370       0.00
    7371       0.00
    7372       0.00
    7373       0.00
    7374       0.00
    7375       2.99
    7376       0.00
    7377       0.99
    7378       0.00
    7379       0.00
    7380       0.00
    7381       0.00
    7382       0.00
    7383       0.00
    7384       0.99
    7385       0.00
    7386       0.00
    7387       0.00
    7388       0.00
    7389       0.00
    7390       0.99
    7391       0.00
    7392       0.00
    7393       0.00
    7394       0.00
    7395       0.00
    7396       0.00
    7397       0.00
    7398       0.00
    7399       0.00
    7400       0.00
    7401       0.00
    7402       0.00
    7403       0.00
    7404       0.00
    7405       0.00
    7406       0.00
    7407       0.00
    7408       0.00
    7409       0.00
    7410       0.00
    7411       0.00
    7412       0.00
    7413       0.00
    7414       0.00
    7415       0.00
    7416       0.00
    7417       6.99
    7418       0.00
    7419       0.00
    7420       0.00
    7421       0.00
    7422       0.00
    7423       0.00
    7424       0.00
    7425       0.00
    7426       0.00
    7427       0.00
    7428       0.00
    7429       0.00
    7430       0.00
    7431       0.00
    7432       0.00
    7433       0.00
    7434       0.00
    7435       0.00
    7436       0.00
    7437       0.00
    7438       0.00
    7439       0.00
    7440       0.00
    7441       0.00
    7442       0.00
    7443       0.00
    7444       0.00
    7445       0.00
    7446       0.00
    7447       0.00
    7448       0.00
    7449       0.00
    7450       0.00
    7451       0.00
    7452       0.00
    7453       0.00
    7454       0.00
    7455       0.00
    7456       0.00
    7457       0.00
    7458       0.00
    7459       0.00
    7460       0.00
    7461       0.00
    7462       0.00
    7463       0.00
    7464       0.00
    7465       0.99
    7466       0.99
    7467       0.00
    7468       0.00
    7469       0.00
    7470       0.00
    7471       0.00
    7472       0.00
    7473       0.00
    7474       0.00
    7475       0.00
    7476       0.00
    7477      19.99
    7478       0.00
    7479       0.00
    7480       0.00
    7481       0.00
    7482       0.00
    7483       0.00
    7484       0.00
    7485       0.00
    7486       0.00
    7487       0.00
    7488       0.00
    7489       0.00
    7490       0.00
    7491       0.00
    7492       0.00
    7493       0.00
    7494       0.00
    7495       0.00
    7496       0.00
    7497       0.00
    7498       0.00
    7499       0.00
    7500       0.00
    7501       0.00
    7502       0.00
    7503       0.00
    7504       0.99
    7505       0.00
    7506       0.00
    7507       0.00
    7508       0.00
    7509       0.00
    7510       0.00
    7511       0.00
    7512       0.00
    7513       0.00
    7514       0.00
    7515       0.00
    7516       0.00
    7517       0.00
    7518       0.00
    7519       0.00
    7520       0.00
    7521       0.00
    7522       0.00
    7523       0.00
    7524       0.00
    7525       0.00
    7526       0.00
    7527       0.00
    7528       0.00
    7529       0.00
    7530       0.00
    7531       0.00
    7532       0.00
    7533       0.00
    7534       0.00
    7535       0.00
    7536       0.00
    7537       0.00
    7538       4.99
    7539       0.00
    7540       0.00
    7541       0.00
    7542       0.00
    7543       0.00
    7544       0.00
    7545       0.00
    7546       0.00
    7547       0.00
    7548       0.00
    7549       0.00
    7550       0.00
    7551       0.00
    7552       0.00
    7553       0.00
    7554       0.00
    7555       1.99
    7556       0.00
    7557       0.00
    7558       0.00
    7559       0.00
    7560       0.00
    7561       0.00
    7562       0.00
    7563       0.00
    7564       0.00
    7565       0.00
    7566       0.00
    7567       0.00
    7568       0.00
    7569       0.00
    7570       0.00
    7571       0.00
    7572       0.00
    7573       0.00
    7574       0.00
    7575       0.00
    7576       0.00
    7577       0.00
    7578       2.99
    7579       0.00
    7580       0.99
    7581       0.00
    7582       0.00
    7583       0.00
    7584       3.99
    7585       2.99
    7586       3.99
    7587       0.00
    7588       0.00
    7589       0.00
    7590       0.00
    7591       0.00
    7592       0.00
    7593       0.00
    7594       0.00
    7595       0.00
    7596       0.00
    7597       0.00
    7598       0.00
    7599       0.00
    7600       0.00
    7601       0.00
    7602       0.00
    7603       0.00
    7604       0.00
    7605       0.00
    7606       0.00
    7607       0.00
    7608       0.00
    7609       0.00
    7610       0.99
    7611       0.00
    7612       0.00
    7613       0.00
    7614       1.99
    7615       0.00
    7616       0.00
    7617       2.99
    7618       4.99
    7619       2.99
    7620       2.99
    7621       0.00
    7622       0.00
    7623       0.00
    7624       0.00
    7625       0.00
    7626       0.00
    7627       2.99
    7628       0.00
    7629       2.99
    7630       0.00
    7631       0.00
    7632       0.00
    7633       0.00
    7634       0.00
    7635       0.00
    7636       0.00
    7637       0.00
    7638       0.00
    7639       0.00
    7640       0.00
    7641       0.00
    7642       0.00
    7643       0.00
    7644       0.00
    7645       0.00
    7646       0.00
    7647       0.00
    7648       0.00
    7649       0.00
    7650       0.00
    7651       0.00
    7652       0.00
    7653       0.00
    7654       0.00
    7655       0.00
    7656       0.00
    7657       0.00
    7658       4.59
    7659       0.00
    7660       0.00
    7661       0.00
    7662       0.00
    7663       0.00
    7664       0.00
    7665       0.99
    7666       0.00
    7667       0.00
    7668       0.00
    7669       0.00
    7670       0.00
    7671       0.00
    7672       0.00
    7673       0.00
    7674       0.00
    7675       0.00
    7676       0.00
    7677       0.00
    7678       0.00
    7679       0.00
    7680       0.00
    7681       0.00
    7682       0.00
    7683       0.00
    7684       0.00
    7685       0.00
    7686       0.00
    7687       0.00
    7688       0.00
    7689       0.00
    7690       0.00
    7691       0.00
    7692       0.00
    7693       0.00
    7694       0.00
    7695       0.00
    7696       0.00
    7697       0.00
    7698       0.00
    7699       0.00
    7700       0.00
    7701       0.00
    7702       0.00
    7703       0.00
    7704       0.00
    7705       0.00
    7706       0.00
    7707       0.00
    7708       0.00
    7709       0.00
    7710       0.00
    7711       0.00
    7712       0.00
    7713       0.00
    7714       0.00
    7715       0.00
    7716       0.00
    7717       0.00
    7718       0.00
    7719       0.00
    7720       0.00
    7721       0.00
    7722       0.00
    7723       0.00
    7724       0.00
    7725       0.00
    7726       0.00
    7727       0.00
    7728       0.00
    7729       0.00
    7730       0.99
    7731       0.00
    7732       0.00
    7733       0.00
    7734       0.00
    7735       0.00
    7736       0.00
    7737       0.00
    7738       1.49
    7739       0.00
    7740       0.00
    7741       0.00
    7742       0.00
    7743       0.00
    7744       0.00
    7745       1.49
    7746       0.00
    7747       0.00
    7748       0.00
    7749       0.00
    7750       0.00
    7751       0.00
    7752       0.00
    7753       0.00
    7754       0.00
    7755       0.00
    7756       0.00
    7757       0.00
    7758       0.00
    7759       0.00
    7760       0.00
    7761       0.00
    7762       0.00
    7763       0.00
    7764       0.00
    7765       0.00
    7766       0.00
    7767       0.00
    7768       0.00
    7769       0.00
    7770       0.00
    7771       0.00
    7772       0.00
    7773       0.00
    7774       0.99
    7775       1.99
    7776       0.00
    7777       0.00
    7778       0.00
    7779       0.00
    7780       0.00
    7781       0.00
    7782       0.00
    7783       0.00
    7784       0.00
    7785       0.00
    7786       0.00
    7787       0.00
    7788       0.00
    7789       0.00
    7790       0.00
    7791       0.00
    7792       3.49
    7793       0.00
    7794       0.00
    7795       0.00
    7796       0.00
    7797       0.00
    7798       9.99
    7799       0.00
    7800       0.00
    7801       0.00
    7802       0.00
    7803       0.00
    7804       0.00
    7805       0.00
    7806       0.00
    7807       0.00
    7808       0.00
    7809       0.00
    7810       1.99
    7811       0.00
    7812       0.00
    7813       0.00
    7814       0.00
    7815       0.00
    7816       0.00
    7817       0.00
    7818       0.00
    7819       0.00
    7820       0.00
    7821       0.00
    7822       2.49
    7823       0.00
    7824       0.00
    7825       0.00
    7826       0.00
    7827       0.00
    7828       0.00
    7829       0.00
    7830       0.00
    7831       0.00
    7832       0.99
    7833       0.00
    7834       0.00
    7835       0.00
    7836       0.00
    7837       0.00
    7838       0.00
    7839       0.00
    7840       0.00
    7841       0.00
    7842       0.00
    7843       0.00
    7844       0.00
    7845       0.00
    7846       0.00
    7847       0.00
    7848       0.00
    7849       0.00
    7850       0.00
    7851       0.00
    7852       0.00
    7853       0.00
    7854       0.00
    7855       0.00
    7856       0.00
    7857       0.00
    7858       0.00
    7859       0.00
    7860       0.00
    7861       0.00
    7862       0.99
    7863       0.00
    7864       0.00
    7865       0.00
    7866       0.00
    7867       0.00
    7868       0.00
    7869       0.00
    7870       0.00
    7871       0.00
    7872       0.00
    7873       0.00
    7874       0.00
    7875       0.00
    7876       0.00
    7877       0.00
    7878       0.00
    7879       0.00
    7880       0.00
    7881       0.00
    7882       0.00
    7883       0.00
    7884       0.00
    7885       0.00
    7886       0.00
    7887       4.49
    7888       0.00
    7889       0.00
    7890       0.00
    7891       0.00
    7892       0.00
    7893       9.99
    7894      15.46
    7895       0.00
    7896       0.00
    7897       0.99
    7898       9.99
    7899       6.99
    7900       0.00
    7901       0.00
    7902       0.00
    7903       0.00
    7904       0.00
    7905       0.00
    7906       0.00
    7907       0.00
    7908       0.00
    7909       0.00
    7910       0.00
    7911       0.00
    7912       0.00
    7913       0.00
    7914       0.00
    7915       0.00
    7916       0.00
    7917       0.00
    7918       0.00
    7919       0.00
    7920       0.00
    7921       0.00
    7922       0.00
    7923       0.00
    7924       0.00
    7925       0.00
    7926       0.00
    7927       0.00
    7928       0.00
    7929       0.00
    7930       0.00
    7931       0.00
    7932       2.49
    7933       0.00
    7934       0.00
    7935       0.00
    7936       0.00
    7937       0.00
    7938       0.00
    7939       0.00
    7940       0.00
    7941       0.00
    7942       0.00
    7943       0.00
    7944       0.00
    7945       0.00
    7946       0.00
    7947       0.00
    7948       0.00
    7949       0.00
    7950       0.00
    7951       0.00
    7952       0.00
    7953       0.00
    7954       0.00
    7955       0.00
    7956       0.00
    7957       0.00
    7958       0.00
    7959       0.00
    7960       0.00
    7961       0.00
    7962       4.99
    7963       0.00
    7964       0.00
    7965       0.00
    7966       0.00
    7967       0.00
    7968       0.00
    7969       0.00
    7970       0.00
    7971       0.00
    7972       0.00
    7973       0.00
    7974       0.00
    7975       0.00
    7976       0.00
    7977       0.00
    7978       0.99
    7979       0.00
    7980       0.00
    7981       0.00
    7982       0.00
    7983       0.00
    7984       0.00
    7985       0.00
    7986       0.00
    7987       0.00
    7988       0.00
    7989       0.00
    7990       0.00
    7991       0.00
    7992       0.00
    7993       1.49
    7994       0.00
    7995       0.00
    7996       0.00
    7997       1.49
    7998       0.00
    7999       4.99
    8000       0.00
    8001       0.00
    8002       0.00
    8003       0.99
    8004       1.49
    8005       0.00
    8006       2.99
    8007       0.00
    8008       0.00
    8009       0.00
    8010       0.00
    8011       0.00
    8012       0.99
    8013       3.99
    8014       1.99
    8015       0.00
    8016       0.00
    8017       2.49
    8018       1.49
    8019       2.49
    8020       0.00
    8021       2.49
    8022       0.00
    8023       0.00
    8024       0.00
    8025       0.00
    8026       0.00
    8027       0.00
    8028       0.00
    8029       0.00
    8030       0.00
    8031       0.00
    8032       0.00
    8033       0.00
    8034       0.00
    8035       0.00
    8036       0.00
    8037       0.00
    8038       0.00
    8039       0.00
    8040       0.00
    8041       0.00
    8042       0.00
    8043       0.00
    8044       0.00
    8045       0.00
    8046       0.00
    8047       0.00
    8048       0.00
    8049       0.00
    8050       0.00
    8051       4.99
    8052       0.00
    8053       0.00
    8054       0.00
    8055       0.00
    8056       0.00
    8057       0.00
    8058       0.00
    8059       0.00
    8060       0.00
    8061       0.00
    8062       0.00
    8063       0.00
    8064       0.00
    8065       0.00
    8066       0.00
    8067       0.00
    8068       0.00
    8069       0.00
    8070       0.00
    8071       0.00
    8072       0.00
    8073       0.00
    8074       4.99
    8075       0.00
    8076       0.00
    8077      10.00
    8078       0.00
    8079       0.00
    8080       0.00
    8081       0.00
    8082       0.00
    8083       0.00
    8084       0.00
    8085       1.99
    8086       0.00
    8087       0.00
    8088       0.00
    8089       0.00
    8090       0.00
    8091       0.00
    8092       0.00
    8093       0.00
    8094       0.00
    8095       0.00
    8096       0.00
    8097       0.00
    8098       0.00
    8099       0.00
    8100       0.00
    8101       0.00
    8102       0.00
    8103       0.00
    8104       0.00
    8105       0.00
    8106       0.00
    8107       0.00
    8108       0.00
    8109       0.00
    8110       0.00
    8111       0.00
    8112       0.00
    8113       0.00
    8114       0.00
    8115       0.00
    8116       0.00
    8117       0.00
    8118       0.00
    8119       0.00
    8120       0.00
    8121       0.00
    8122       0.00
    8123       0.00
    8124       0.00
    8125       0.00
    8126       0.00
    8127       0.00
    8128       0.00
    8129       0.00
    8130       0.00
    8131       0.00
    8132       0.00
    8133       0.00
    8134       0.00
    8135       0.00
    8136       0.00
    8137       5.99
    8138       0.00
    8139       5.99
    8140       5.99
    8141       0.99
    8142       0.00
    8143       0.00
    8144       0.00
    8145       5.49
    8146       0.00
    8147       0.00
    8148       0.00
    8149       5.99
    8150       5.99
    8151       5.99
    8152       0.00
    8153       0.00
    8154       0.00
    8155       0.00
    8156       0.00
    8157       0.00
    8158       0.00
    8159       0.00
    8160       0.00
    8161       0.99
    8162       0.00
    8163       0.00
    8164       3.49
    8165       0.00
    8166       0.00
    8167       0.00
    8168       0.00
    8169       0.00
    8170       0.00
    8171       4.99
    8172       0.00
    8173       0.00
    8174       0.00
    8175       0.00
    8176       2.99
    8177       0.00
    8178       0.00
    8179       0.00
    8180       0.00
    8181       0.00
    8182       0.00
    8183       0.00
    8184       0.00
    8185       0.00
    8186       0.00
    8187       0.00
    8188       0.00
    8189       0.00
    8190       0.00
    8191       2.99
    8192       0.00
    8193       0.00
    8194       0.00
    8195       0.00
    8196       0.00
    8197       0.00
    8198       0.00
    8199       0.00
    8200       0.00
    8201       0.00
    8202       0.00
    8203       0.00
    8204       0.00
    8205       0.00
    8206       0.00
    8207       0.00
    8208       0.00
    8209       0.00
    8210       0.00
    8211       8.99
    8212       2.99
    8213       0.00
    8214       0.00
    8215       0.00
    8216       0.00
    8217       0.00
    8218       0.00
    8219       0.99
    8220       0.00
    8221       0.00
    8222       0.00
    8223       0.00
    8224       0.00
    8225       0.99
    8226       0.00
    8227       0.00
    8228       0.00
    8229       0.00
    8230       0.00
    8231       0.00
    8232       0.00
    8233       0.00
    8234       0.00
    8235       0.00
    8236       3.49
    8237       0.00
    8238       0.00
    8239       0.00
    8240       0.00
    8241       0.00
    8242       1.49
    8243       0.00
    8244       0.00
    8245       0.00
    8246       0.00
    8247       0.00
    8248       0.00
    8249       0.00
    8250       0.00
    8251       4.99
    8252       0.00
    8253       0.00
    8254       0.00
    8255       0.00
    8256       0.00
    8257       0.00
    8258       0.00
    8259       0.00
    8260       0.00
    8261       0.00
    8262       0.00
    8263       0.00
    8264       0.00
    8265       0.00
    8266       0.00
    8267       0.00
    8268       0.00
    8269       0.00
    8270       0.00
    8271       0.00
    8272       0.00
    8273       0.00
    8274       0.00
    8275       0.00
    8276       0.00
    8277       0.00
    8278       0.00
    8279       0.00
    8280       0.00
    8281       0.00
    8282       4.99
    8283       0.00
    8284       0.00
    8285       0.00
    8286       2.99
    8287       3.04
    8288       0.00
    8289       0.00
    8290       0.00
    8291       0.00
    8292       0.00
    8293       0.00
    8294       0.00
    8295       0.00
    8296       0.00
    8297       0.00
    8298       0.00
    8299       0.00
    8300       0.00
    8301       0.00
    8302       0.00
    8303       0.00
    8304       0.00
    8305       0.00
    8306       0.00
    8307       0.00
    8308       0.00
    8309       0.00
    8310       0.00
    8311       0.00
    8312       0.00
    8313       0.00
    8314       0.00
    8315       0.00
    8316       0.00
    8317       0.00
    8318       0.00
    8319       0.00
    8320       0.00
    8321       3.99
    8322       0.00
    8323       0.00
    8324       0.00
    8325       0.00
    8326       0.00
    8327       0.00
    8328      29.99
    8329       0.00
    8330       0.00
    8331       0.00
    8332       0.00
    8333       0.00
    8334       0.00
    8335       0.00
    8336       0.00
    8337       0.00
    8338       0.00
    8339       0.00
    8340       0.00
    8341       0.00
    8342       0.00
    8343       0.00
    8344       0.00
    8345       0.00
    8346       2.99
    8347       0.00
    8348       2.99
    8349       0.00
    8350       3.99
    8351       0.00
    8352       0.00
    8353       0.00
    8354       0.00
    8355       0.00
    8356       0.00
    8357       0.00
    8358       0.00
    8359       0.99
    8360       0.00
    8361       0.00
    8362       2.99
    8363       0.00
    8364       8.99
    8365       0.00
    8366       0.00
    8367       0.00
    8368       4.99
    8369       0.00
    8370       0.00
    8371      12.99
    8372       0.00
    8373       0.00
    8374       0.00
    8375       0.00
    8376       2.49
    8377       0.00
    8378       0.00
    8379       0.00
    8380       0.00
    8381       0.00
    8382       0.00
    8383       0.00
    8384       0.00
    8385       0.00
    8386       0.00
    8387       0.00
    8388       0.00
    8389       0.00
    8390       0.00
    8391       0.00
    8392       0.00
    8393       0.00
    8394       0.00
    8395       0.00
    8396       0.00
    8397       0.00
    8398       0.00
    8399       0.00
    8400       0.00
    8401       0.00
    8402       0.00
    8403       0.00
    8404       0.00
    8405       0.00
    8406       0.00
    8407       0.00
    8408       0.00
    8409       0.00
    8410       0.00
    8411       0.00
    8412       0.00
    8413       0.00
    8414       0.00
    8415       0.00
    8416       0.00
    8417       0.00
    8418       0.00
    8419       0.00
    8420       0.00
    8421       0.00
    8422       0.00
    8423       0.00
    8424       0.00
    8425       0.00
    8426       0.00
    8427       0.00
    8428       0.00
    8429       0.00
    8430       0.00
    8431       0.00
    8432       0.00
    8433       0.00
    8434       0.00
    8435       0.00
    8436       0.00
    8437       0.00
    8438       0.00
    8439       0.00
    8440       0.00
    8441       0.00
    8442       0.00
    8443       0.00
    8444       0.00
    8445       0.00
    8446       0.00
    8447       0.00
    8448       0.00
    8449       2.99
    8450       0.00
    8451       0.00
    8452       0.00
    8453       0.00
    8454       0.00
    8455       0.00
    8456       0.00
    8457       0.00
    8458       0.00
    8459       0.00
    8460       0.00
    8461       0.00
    8462       0.00
    8463       0.00
    8464       0.00
    8465       0.00
    8466       0.00
    8467       0.00
    8468       0.00
    8469       0.00
    8470       0.00
    8471       0.00
    8472       0.00
    8473       0.00
    8474       0.00
    8475       0.00
    8476       0.00
    8477       0.00
    8478       0.00
    8479       0.00
    8480       0.00
    8481       0.00
    8482       2.99
    8483       0.00
    8484       0.00
    8485       0.00
    8486       0.00
    8487       0.00
    8488       0.00
    8489       0.99
    8490       0.00
    8491       0.00
    8492       0.00
    8493       0.00
    8494       0.00
    8495       0.00
    8496       0.00
    8497       0.00
    8498       0.00
    8499       0.00
    8500       0.00
    8501       0.00
    8502       0.00
    8503       0.00
    8504       0.00
    8505       0.00
    8506       0.00
    8507       0.00
    8508       0.00
    8509       0.00
    8510       0.00
    8511       2.99
    8512       0.00
    8513       0.00
    8514       0.00
    8515       0.00
    8516       0.00
    8517       0.00
    8518       0.00
    8519       0.00
    8520       0.00
    8521       0.00
    8522       0.00
    8523       0.00
    8524       0.00
    8525       0.00
    8526       0.99
    8527       0.00
    8528       0.00
    8529       0.00
    8530       0.00
    8531       0.00
    8532       0.00
    8533       0.00
    8534       0.00
    8535       0.00
    8536       0.00
    8537       0.00
    8538       0.00
    8539       0.00
    8540       0.00
    8541       0.00
    8542       0.00
    8543       0.00
    8544       0.00
    8545       0.00
    8546       0.00
    8547       0.00
    8548       0.00
    8549       0.00
    8550       0.00
    8551       0.00
    8552       0.00
    8553       0.00
    8554       0.00
    8555       1.99
    8556       0.00
    8557       0.00
    8558       0.00
    8559       0.00
    8560       0.00
    8561       0.00
    8562       0.00
    8563       0.00
    8564       0.00
    8565       0.99
    8566       0.00
    8567       0.00
    8568       0.00
    8569       0.00
    8570       0.00
    8571       0.99
    8572       0.00
    8573       0.00
    8574       1.99
    8575       0.00
    8576       0.00
    8577       0.00
    8578       0.00
    8579       0.00
    8580       0.00
    8581       0.00
    8582       0.00
    8583       0.00
    8584       0.00
    8585       0.00
    8586       0.00
    8587       0.00
    8588       0.00
    8589       0.00
    8590       1.99
    8591       0.00
    8592       0.00
    8593       0.00
    8594       0.00
    8595       0.00
    8596       0.00
    8597       0.00
    8598       0.00
    8599       0.00
    8600       2.99
    8601       0.00
    8602       0.00
    8603       0.00
    8604       4.49
    8605       0.00
    8606       0.00
    8607       0.00
    8608       0.00
    8609       0.00
    8610       0.00
    8611       0.00
    8612       0.00
    8613       0.00
    8614       0.00
    8615       0.00
    8616       8.49
    8617       0.00
    8618       2.49
    8619       0.00
    8620       0.00
    8621       0.00
    8622       0.00
    8623       0.00
    8624       0.00
    8625       0.00
    8626       0.00
    8627       0.00
    8628       0.00
    8629       0.00
    8630       0.00
    8631       0.00
    8632       0.00
    8633       0.00
    8634       0.00
    8635       0.00
    8636       0.00
    8637       0.00
    8638       0.00
    8639       0.00
    8640       0.00
    8641       0.00
    8642       0.00
    8643       0.00
    8644       0.00
    8645       0.00
    8646       0.00
    8647       0.00
    8648       0.00
    8649       0.00
    8650       0.00
    8651       0.00
    8652       0.00
    8653       0.00
    8654       0.00
    8655       0.00
    8656       0.00
    8657       0.00
    8658       0.00
    8659       0.00
    8660       0.00
    8661       0.00
    8662       0.00
    8663       0.00
    8664       0.00
    8665       0.00
    8666       0.00
    8667       0.00
    8668       0.00
    8669       0.00
    8670       0.00
    8671       0.00
    8672       0.00
    8673       0.00
    8674       0.00
    8675       0.00
    8676       0.00
    8677       0.00
    8678       0.00
    8679       0.00
    8680       0.00
    8681       0.00
    8682       0.00
    8683       0.00
    8684       0.00
    8685       0.00
    8686       0.00
    8687       0.00
    8688       0.00
    8689       0.00
    8690       0.00
    8691       0.00
    8692       0.00
    8693       0.00
    8694       0.00
    8695       0.00
    8696       0.00
    8697       0.00
    8698       0.00
    8699       0.00
    8700       0.00
    8701       0.00
    8702       0.00
    8703       0.00
    8704       0.00
    8705       0.00
    8706       0.00
    8707       0.00
    8708       0.00
    8709       0.00
    8710       0.00
    8711       0.00
    8712       0.00
    8713       0.00
    8714       0.00
    8715       0.00
    8716       0.00
    8717       0.00
    8718       0.00
    8719      19.99
    8720       0.00
    8721       9.99
    8722       1.99
    8723       2.99
    8724       4.99
    8725       0.00
    8726      14.99
    8727       0.00
    8728       0.00
    8729       0.00
    8730       0.00
    8731       0.00
    8732       0.00
    8733       9.99
    8734       0.00
    8735       0.00
    8736      13.99
    8737       0.00
    8738       0.00
    8739       0.00
    8740       0.00
    8741       0.00
    8742       0.00
    8743       0.00
    8744       0.00
    8745       0.00
    8746       0.00
    8747       0.00
    8748       0.00
    8749       0.00
    8750       0.00
    8751       0.00
    8752       0.00
    8753       0.00
    8754       0.00
    8755       4.29
    8756       0.00
    8757       0.00
    8758       0.00
    8759       0.00
    8760       0.00
    8761       0.00
    8762       0.00
    8763       0.00
    8764       0.00
    8765       0.00
    8766       0.00
    8767       0.00
    8768       0.00
    8769       0.00
    8770       0.00
    8771       0.00
    8772       0.00
    8773       0.00
    8774       0.00
    8775       0.00
    8776       0.00
    8777       2.99
    8778       2.99
    8779       2.99
    8780       1.99
    8781       0.00
    8782       0.00
    8783       0.00
    8784       0.00
    8785       3.99
    8786       0.00
    8787       4.99
    8788       0.00
    8789       0.00
    8790       0.00
    8791       0.00
    8792       4.99
    8793       3.99
    8794       0.00
    8795       2.99
    8796       0.00
    8797       0.00
    8798       0.00
    8799       0.00
    8800       2.99
    8801       2.99
    8802       0.00
    8803       0.00
    8804       4.99
    8805       0.00
    8806       0.00
    8807       0.00
    8808       0.00
    8809       0.00
    8810       0.00
    8811       0.00
    8812       0.00
    8813       0.00
    8814       0.00
    8815       0.00
    8816       0.00
    8817       0.00
    8818       0.00
    8819       0.00
    8820       0.00
    8821       0.00
    8822       0.00
    8823      12.99
    8824       0.00
    8825       2.60
    8826       0.00
    8827       0.00
    8828       0.00
    8829       0.00
    8830       0.00
    8831       0.00
    8832       0.00
    8833       0.00
    8834       0.00
    8835       0.00
    8836       0.00
    8837       0.00
    8838       0.00
    8839       0.00
    8840       0.00
    8841       3.28
    8842       0.00
    8843       0.00
    8844       0.00
    8845       4.99
    8846       0.00
    8847       0.00
    8848       0.00
    8849       0.00
    8850       0.00
    8851       0.00
    8852       0.00
    8853       2.99
    8854       0.00
    8855       0.00
    8856       0.00
    8857       0.00
    8858       0.00
    8859       0.00
    8860       2.99
    8861       0.00
    8862       0.00
    8863       0.00
    8864       0.00
    8865       0.00
    8866       0.00
    8867       0.00
    8868       0.00
    8869       0.00
    8870       0.00
    8871       0.00
    8872       0.00
    8873       0.00
    8874       0.00
    8875       0.00
    8876       0.00
    8877       0.00
    8878       0.00
    8879       2.99
    8880       0.00
    8881       0.00
    8882       0.00
    8883       0.00
    8884       0.00
    8885       0.00
    8886       0.00
    8887       4.99
    8888       0.00
    8889       0.00
    8890       3.99
    8891       0.00
    8892       0.00
    8893       0.00
    8894       0.00
    8895       0.00
    8896       0.00
    8897       0.00
    8898       0.00
    8899       0.00
    8900       0.00
    8901       0.00
    8902       0.00
    8903       0.00
    8904       0.00
    8905       0.00
    8906       0.00
    8907       0.00
    8908       0.00
    8909       0.00
    8910       0.00
    8911       0.00
    8912       3.99
    8913       0.00
    8914       0.00
    8915       0.00
    8916       0.00
    8917       0.00
    8918       0.00
    8919       0.00
    8920       0.00
    8921       0.00
    8922       0.00
    8923       0.00
    8924       0.00
    8925       0.00
    8926       0.00
    8927       4.60
    8928       0.00
    8929       0.00
    8930       0.00
    8931       0.00
    8932       0.00
    8933       0.00
    8934       0.00
    8935       0.00
    8936       0.00
    8937       0.00
    8938       0.00
    8939       0.00
    8940       0.00
    8941       0.00
    8942       0.00
    8943       0.00
    8944       0.00
    8945       0.00
    8946       0.00
    8947       0.00
    8948       0.00
    8949       0.00
    8950       4.99
    8951       0.00
    8952       0.00
    8953       0.00
    8954       0.00
    8955       0.00
    8956       0.00
    8957       0.00
    8958       0.00
    8959       0.00
    8960       0.00
    8961       0.00
    8962       0.00
    8963       0.00
    8964       0.00
    8965       0.00
    8966       0.00
    8967       0.00
    8968       0.00
    8969       0.00
    8970       0.00
    8971       0.00
    8972       0.00
    8973       0.00
    8974       0.00
    8975       0.00
    8976       0.00
    8977       0.00
    8978       0.00
    8979       0.00
    8980       0.00
    8981       0.00
    8982       0.00
    8983       0.00
    8984       0.00
    8985       0.00
    8986       0.00
    8987       0.00
    8988       0.00
    8989       0.00
    8990       0.00
    8991       0.00
    8992       0.00
    8993       0.99
    8994       0.00
    8995       0.00
    8996       0.00
    8997       0.00
    8998       0.00
    8999       0.00
    9000       0.00
    9001       0.00
    9002       0.00
    9003       0.00
    9004       0.00
    9005       0.00
    9006       0.00
    9007       0.00
    9008       0.00
    9009       0.00
    9010       1.49
    9011       0.00
    9012       0.00
    9013       0.00
    9014       0.00
    9015       0.00
    9016       0.00
    9017       0.00
    9018       0.00
    9019       0.00
    9020       0.00
    9021       0.00
    9022       0.00
    9023       0.00
    9024       2.99
    9025       0.00
    9026       1.00
    9027       0.00
    9028       0.99
    9029       0.00
    9030       5.99
    9031       0.00
    9032       0.00
    9033       0.00
    9034       0.00
    9035       0.00
    9036       0.00
    9037       0.00
    9038       0.00
    9039       0.99
    9040       0.00
    9041       0.00
    9042       0.00
    9043       0.00
    9044       0.00
    9045       2.99
    9046       0.00
    9047       0.00
    9048       0.00
    9049       0.00
    9050       0.00
    9051       0.00
    9052       0.00
    9053       0.00
    9054       2.99
    9055       0.00
    9056       1.99
    9057       0.99
    9058       0.99
    9059       0.00
    9060       0.99
    9061       0.00
    9062       0.00
    9063       1.49
    9064       3.99
    9065       2.99
    9066       0.00
    9067       0.00
    9068       0.00
    9069       0.00
    9070       0.00
    9071       0.00
    9072       0.00
    9073       0.00
    9074       0.00
    9075       0.00
    9076       0.00
    9077       0.00
    9078       0.00
    9079       0.00
    9080       0.00
    9081       0.00
    9082       1.49
    9083       0.00
    9084       0.00
    9085       0.00
    9086       0.00
    9087       0.00
    9088       0.00
    9089       0.00
    9090       0.00
    9091       0.00
    9092       0.00
    9093       0.00
    9094       0.00
    9095       0.00
    9096       0.99
    9097       0.00
    9098       0.00
    9099       0.00
    9100       0.00
    9101       5.99
    9102       0.00
    9103       0.00
    9104      28.99
    9105       0.00
    9106       0.00
    9107       0.00
    9108       0.00
    9109       0.00
    9110       0.00
    9111       0.00
    9112       0.00
    9113       0.00
    9114       0.00
    9115       0.00
    9116       0.00
    9117       0.00
    9118       0.00
    9119       0.00
    9120       0.00
    9121       0.00
    9122       0.00
    9123       0.00
    9124       0.00
    9125       0.00
    9126       0.00
    9127       0.00
    9128       0.00
    9129       0.00
    9130       0.00
    9131       0.00
    9132       0.00
    9133       0.00
    9134       0.00
    9135       0.00
    9136       0.00
    9137       0.00
    9138       0.00
    9139       0.00
    9140       0.00
    9141       0.00
    9142       0.00
    9143       0.00
    9144       0.00
    9145       0.00
    9146       0.00
    9147       0.00
    9148       0.00
    9149       0.00
    9150       0.00
    9151       0.99
    9152       0.00
    9153       0.00
    9154       4.99
    9155       0.00
    9156       0.00
    9157       0.00
    9158       0.00
    9159       0.00
    9160       0.00
    9161       0.00
    9162       0.00
    9163       0.00
    9164       0.00
    9165       0.99
    9166       0.00
    9167       0.00
    9168       0.00
    9169       0.00
    9170       0.99
    9171       0.00
    9172       0.00
    9173       0.00
    9174       0.00
    9175       0.00
    9176       0.00
    9177       0.00
    9178       0.00
    9179       0.00
    9180       0.00
    9181       0.00
    9182       0.00
    9183       0.00
    9184       0.00
    9185       0.00
    9186       0.00
    9187       0.00
    9188       0.00
    9189       0.00
    9190       0.00
    9191       0.00
    9192       0.00
    9193       0.00
    9194       0.00
    9195       0.00
    9196       0.00
    9197       0.00
    9198       0.00
    9199       0.00
    9200       0.00
    9201       0.00
    9202       0.00
    9203       0.00
    9204       0.00
    9205       0.00
    9206       0.99
    9207       0.00
    9208       0.00
    9209       0.00
    9210       0.00
    9211      10.99
    9212       0.00
    9213       0.00
    9214       0.00
    9215       0.00
    9216       0.00
    9217       0.00
    9218       0.00
    9219       0.00
    9220      14.99
    9221       0.00
    9222       0.00
    9223       0.00
    9224       0.00
    9225       0.00
    9226       0.00
    9227       1.99
    9228       0.00
    9229       0.00
    9230       0.00
    9231       0.00
    9232       0.00
    9233       0.00
    9234       0.00
    9235       0.00
    9236       0.00
    9237       0.00
    9238       0.00
    9239       0.00
    9240       0.00
    9241       0.00
    9242       0.00
    9243       0.00
    9244       0.00
    9245       0.00
    9246       0.00
    9247       0.00
    9248       0.00
    9249       0.00
    9250       0.00
    9251       0.00
    9252       0.00
    9253       0.00
    9254       0.00
    9255       0.00
    9256       0.00
    9257       0.00
    9258       0.00
    9259       0.00
    9260       0.00
    9261       0.00
    9262       0.00
    9263       0.00
    9264       0.00
    9265       0.00
    9266       0.00
    9267       0.00
    9268       0.00
    9269       0.00
    9270       0.00
    9271       0.00
    9272       0.00
    9273       0.00
    9274       0.00
    9275       0.00
    9276       0.00
    9277       0.00
    9278       0.00
    9279       0.00
    9280       0.00
    9281       0.00
    9282       0.00
    9283       0.00
    9284       0.00
    9285       0.00
    9286       0.00
    9287       0.00
    9288       0.00
    9289       0.00
    9290       0.00
    9291       0.00
    9292       0.00
    9293       0.00
    9294       9.99
    9295       0.00
    9296       0.00
    9297       0.00
    9298       0.00
    9299       0.00
    9300       0.00
    9301       0.00
    9302       0.00
    9303       0.00
    9304      12.99
    9305       0.00
    9306       0.00
    9307       0.00
    9308       0.00
    9309       0.00
    9310       0.00
    9311       0.00
    9312       0.00
    9313       0.00
    9314       0.00
    9315       0.00
    9316       0.00
    9317       0.00
    9318       0.00
    9319       0.00
    9320       0.00
    9321       0.00
    9322       0.00
    9323       0.00
    9324       0.00
    9325       0.99
    9326       0.00
    9327       0.00
    9328       0.00
    9329       0.00
    9330       0.00
    9331       0.00
    9332       0.00
    9333       0.00
    9334       2.99
    9335       0.00
    9336       0.00
    9337       3.99
    9338       0.00
    9339       0.00
    9340       0.00
    9341       0.00
    9342       0.00
    9343       0.00
    9344       0.00
    9345       0.00
    9346       0.00
    9347       0.00
    9348       0.00
    9349       0.00
    9350       0.00
    9351       0.00
    9352       0.00
    9353       0.00
    9354       0.00
    9355       0.00
    9356       0.00
    9357       1.49
    9358       0.00
    9359       0.00
    9360       0.00
    9361       0.00
    9362       0.00
    9363       0.00
    9364       0.00
    9365       0.00
    9366       0.00
    9367       0.00
    9368       0.00
    9369       0.00
    9370       0.00
    9371       0.00
    9372       0.00
    9373       0.00
    9374       0.00
    9375       0.00
    9376       0.00
    9377       0.00
    9378       0.00
    9379       0.00
    9380       0.00
    9381       0.00
    9382       0.00
    9383       0.00
    9384       0.00
    9385       0.00
    9386       0.00
    9387       0.00
    9388       0.00
    9389       0.00
    9390       0.00
    9391       0.00
    9392       0.00
    9393       0.00
    9394       0.00
    9395       0.00
    9396       0.00
    9397       0.00
    9398       0.00
    9399       0.00
    9400       0.00
    9401       0.00
    9402       0.00
    9403       0.00
    9404       0.00
    9405       0.00
    9406       0.00
    9407       0.00
    9408       0.00
    9409       9.99
    9410       0.00
    9411       0.00
    9412       0.00
    9413       0.00
    9414       0.00
    9415       0.00
    9416       0.00
    9417       0.00
    9418       0.00
    9419       0.00
    9420       0.00
    9421       0.00
    9422       0.00
    9423       0.00
    9424       0.00
    9425       0.00
    9426       0.00
    9427       0.00
    9428       0.00
    9429       0.00
    9430       0.00
    9431       0.00
    9432       0.00
    9433       0.00
    9434       0.00
    9435       0.00
    9436       0.00
    9437       0.00
    9438       0.00
    9439       0.00
    9440       0.00
    9441       0.00
    9442       0.00
    9443       0.00
    9444       0.00
    9445       0.00
    9446       0.00
    9447       0.00
    9448       0.00
    9449       0.00
    9450       0.00
    9451       0.00
    9452       0.00
    9453       0.00
    9454       0.00
    9455       0.00
    9456       0.00
    9457       0.00
    9458       0.00
    9459       0.00
    9460       0.00
    9461       0.00
    9462       0.00
    9463       0.00
    9464       0.00
    9465       2.95
    9466       0.00
    9467       0.00
    9468       0.00
    9469       0.00
    9470       1.99
    9471       0.00
    9472       0.00
    9473       0.00
    9474       0.00
    9475       0.00
    9476       0.00
    9477       0.00
    9478       2.99
    9479       0.00
    9480       1.99
    9481       0.00
    9482       0.00
    9483       0.00
    9484       0.00
    9485       0.00
    9486       0.00
    9487       0.00
    9488       0.00
    9489       0.00
    9490       2.90
    9491       0.00
    9492       0.00
    9493       0.00
    9494       0.00
    9495       0.00
    9496       0.00
    9497       0.00
    9498       0.00
    9499       0.00
    9500       0.00
    9501       0.00
    9502       0.00
    9503       0.00
    9504       0.00
    9505       0.00
    9506       0.00
    9507       0.00
    9508       0.00
    9509       0.00
    9510       0.00
    9511       0.00
    9512       0.00
    9513       0.00
    9514       0.00
    9515       0.00
    9516       0.00
    9517       0.00
    9518       0.00
    9519       0.00
    9520       0.00
    9521       0.00
    9522       0.00
    9523       0.00
    9524       0.00
    9525       0.00
    9526       0.00
    9527       0.00
    9528       0.00
    9529       0.00
    9530       0.00
    9531       0.00
    9532       0.00
    9533       0.00
    9534       0.00
    9535       0.00
    9536       0.00
    9537       0.00
    9538       0.00
    9539       0.00
    9540       0.00
    9541       0.99
    9542       0.00
    9543       0.00
    9544       0.00
    9545       0.00
    9546       0.00
    9547       0.00
    9548       0.00
    9549       0.00
    9550       1.99
    9551       0.00
    9552       0.00
    9553       0.00
    9554       0.00
    9555       0.00
    9556       0.00
    9557       0.00
    9558       0.00
    9559       0.00
    9560       0.00
    9561       0.00
    9562       0.00
    9563       0.00
    9564       0.00
    9565       0.00
    9566       1.97
    9567       0.00
    9568       0.00
    9569       2.99
    9570       0.00
    9571       0.00
    9572       0.00
    9573       0.00
    9574      24.99
    9575       0.00
    9576       0.00
    9577       0.00
    9578       0.00
    9579       0.00
    9580       0.00
    9581       0.00
    9582       0.00
    9583       0.00
    9584       0.00
    9585       0.00
    9586       3.99
    9587       0.00
    9588       0.00
    9589       0.00
    9590       0.00
    9591       0.00
    9592       0.99
    9593       0.00
    9594       0.00
    9595       0.00
    9596       0.00
    9597       0.00
    9598       0.00
    9599       0.00
    9600       0.00
    9601       0.00
    9602       0.00
    9603       0.00
    9604       0.00
    9605       0.00
    9606       0.00
    9607       0.00
    9608       0.00
    9609       0.00
    9610       0.00
    9611       0.00
    9612       0.00
    9613       2.49
    9614       0.00
    9615       0.00
    9616       0.00
    9617       0.00
    9618       0.00
    9619       0.00
    9620       0.00
    9621       0.00
    9622       0.00
    9623       0.00
    9624       0.00
    9625       0.00
    9626       0.00
    9627       4.49
    9628       0.00
    9629       0.00
    9630       0.00
    9631       0.00
    9632       0.00
    9633       0.00
    9634       0.00
    9635       0.00
    9636       0.00
    9637       0.00
    9638       0.00
    9639       0.00
    9640       0.00
    9641       0.00
    9642       0.00
    9643       0.00
    9644       0.00
    9645       0.00
    9646       0.00
    9647       6.99
    9648       0.00
    9649       0.00
    9650       0.00
    9651       0.00
    9652       0.00
    9653       0.00
    9654       0.00
    9655       0.00
    9656       0.00
    9657       4.99
    9658       0.00
    9659       0.00
    9660       0.00
    9661       0.00
    9662       0.00
    9663       0.00
    9664       0.00
    9665       0.00
    9666       0.00
    9667       0.00
    9668       0.00
    9669       1.99
    9670       0.00
    9671       0.00
    9672       2.49
    9673       0.00
    9674       0.00
    9675       0.00
    9676       0.00
    9677       0.00
    9678       1.99
    9679       0.99
    9680       0.00
    9681       0.00
    9682       0.00
    9683       0.00
    9684       0.00
    9685       0.00
    9686       0.00
    9687       0.00
    9688       0.00
    9689       0.00
    9690       2.99
    9691       0.00
    9692       0.00
    9693       0.00
    9694       0.00
    9695       0.00
    9696       0.00
    9697       2.99
    9698       0.00
    9699       0.00
    9700       1.99
    9701       0.00
    9702       0.00
    9703       0.00
    9704       0.99
    9705       0.00
    9706       0.00
    9707       0.00
    9708       0.00
    9709       0.00
    9710       3.99
    9711       0.00
    9712       0.00
    9713       0.00
    9714       0.00
    9715       0.99
    9716       0.00
    9717       1.99
    9718       0.00
    9719     200.00
    9720       0.00
    9721       0.00
    9722       0.00
    9723       0.00
    9724       0.00
    9725       0.00
    9726       0.00
    9727       0.00
    9728       2.99
    9729       0.00
    9730      89.99
    9731       0.00
    9732       0.00
    9733       0.00
    9734       0.00
    9735       0.00
    9736       0.00
    9737       0.00
    9738       0.00
    9739       0.00
    9740       0.00
    9741       0.00
    9742       0.00
    9743       0.00
    9744       0.00
    9745       0.00
    9746       0.00
    9747       0.00
    9748       0.00
    9749       0.00
    9750       0.00
    9751       0.00
    9752       0.00
    9753       0.00
    9754       0.00
    9755       0.00
    9756       0.00
    9757       0.00
    9758       0.00
    9759       0.00
    9760       0.00
    9761       0.00
    9762       0.00
    9763       0.00
    9764       0.00
    9765       0.00
    9766       0.00
    9767       0.00
    9768       0.00
    9769       0.00
    9770       0.00
    9771       0.00
    9772       0.00
    9773       0.00
    9774       0.00
    9775       0.00
    9776       0.00
    9777       0.00
    9778       0.00
    9779       0.00
    9780       0.00
    9781       0.00
    9782       0.00
    9783       0.00
    9784       0.00
    9785       2.99
    9786       0.00
    9787       0.00
    9788       0.00
    9789       0.00
    9790       0.00
    9791       0.00
    9792       0.00
    9793       0.00
    9794       0.00
    9795       0.00
    9796       0.00
    9797       0.00
    9798       0.00
    9799       0.00
    9800       0.00
    9801       0.00
    9802       0.00
    9803       0.00
    9804       0.00
    9805       0.00
    9806       0.00
    9807       0.00
    9808       0.00
    9809       0.00
    9810       0.00
    9811       0.00
    9812       0.00
    9813       0.00
    9814       0.00
    9815       0.00
    9816       0.00
    9817       0.00
    9818       0.00
    9819       0.00
    9820       0.00
    9821       0.00
    9822       0.00
    9823       0.00
    9824       0.00
    9825       0.00
    9826       0.00
    9827       0.00
    9828       2.99
    9829       0.00
    9830       0.00
    9831       0.00
    9832       0.00
    9833       0.00
    9834       0.00
    9835       0.00
    9836       0.00
    9837       0.00
    9838       0.00
    9839       0.00
    9840       0.00
    9841       0.00
    9842       0.00
    9843       0.00
    9844       0.00
    9845       0.00
    9846       0.00
    9847       0.00
    9848       4.99
    9849       0.00
    9850       0.00
    9851       0.00
    9852       0.00
    9853       0.00
    9854       0.00
    9855       0.00
    9856       0.00
    9857       0.00
    9858       0.00
    9859       0.00
    9860       0.00
    9861       0.00
    9862       0.00
    9863       0.00
    9864       0.00
    9865       0.00
    9866       0.00
    9867       0.00
    9868       0.00
    9869       2.56
    9870       0.00
    9871       0.00
    9872       0.00
    9873       0.00
    9874       0.00
    9875       0.00
    9876       0.99
    9877       0.00
    9878       0.00
    9879       0.00
    9880       0.00
    9881       0.00
    9882       0.00
    9883       0.00
    9884       0.00
    9885       0.00
    9886       0.00
    9887       0.00
    9888       0.00
    9889       0.00
    9890       0.00
    9891       0.00
    9892       0.00
    9893       0.00
    9894       0.00
    9895       0.00
    9896       0.00
    9897       0.00
    9898       0.00
    9899       0.00
    9900       0.00
    9901       0.00
    9902       0.00
    9903       0.00
    9904       0.00
    9905      30.99
    9906       0.00
    9907       0.00
    9908       0.00
    9909       0.00
    9910       3.61
    9911       0.00
    9912       0.00
    9913       0.00
    9914       0.00
    9915       0.00
    9916       0.00
    9917     394.99
    9918       0.00
    9919       0.00
    9920       1.26
    9921       0.00
    9922       0.00
    9923       0.00
    9924       0.00
    9925       0.00
    9926       0.00
    9927       0.00
    9928       0.00
    9929       0.00
    9930       0.00
    9931       0.00
    9932       5.99
    9933       0.00
    9934     399.99
    9935       0.00
    9936       0.00
    9937       0.00
    9938       0.00
    9939       0.00
    9940       0.00
    9941       2.99
    9942       0.00
    9943       0.00
    9944       0.00
    9945       0.00
    9946       0.00
    9947       0.00
    9948       0.00
    9949       0.00
    9950       0.00
    9951       0.00
    9952       0.00
    9953       0.00
    9954       0.00
    9955       0.00
    9956       0.00
    9957       0.00
    9958       0.00
    9959       0.00
    9960       0.00
    9961       0.00
    9962       0.00
    9963       0.00
    9964       0.00
    9965       0.00
    9966       0.00
    9967       0.00
    9968       0.00
    9969       0.00
    9970       0.00
    9971       0.00
    9972       0.00
    9973       0.00
    9974       0.00
    9975       0.00
    9976       0.00
    9977       0.00
    9978       0.00
    9979       0.00
    9980       0.00
    9981       0.00
    9982       0.00
    9983       0.00
    9984       0.00
    9985       0.00
    9986       0.00
    9987       0.00
    9988       0.00
    9989       0.00
    9990       0.00
    9991       0.00
    9992       0.00
    9993       0.00
    9994       0.00
    9995       0.00
    9996       0.00
    9997       0.00
    9998       0.00
    9999       0.00
    10000      0.00
    10001      0.00
    10002      0.00
    10003      0.00
    10004      0.00
    10005      0.99
    10006      9.99
    10007      0.00
    10008      0.00
    10009      0.00
    10010      0.00
    10011      0.00
    10012      0.00
    10013      0.00
    10014      0.00
    10015      0.00
    10016      0.00
    10017      0.00
    10018      0.00
    10019      0.00
    10020      0.00
    10021      0.00
    10022      0.00
    10023      0.00
    10024      0.00
    10025      0.00
    10026      0.00
    10027      0.00
    10028      0.00
    10029      0.00
    10030      0.00
    10031      0.00
    10032      0.00
    10033      0.00
    10034      0.00
    10035      0.99
    10036      2.49
    10037      1.99
    10038      0.00
    10039      0.99
    10040      0.00
    10041      4.99
    10042      0.00
    10043      1.49
    10044      0.00
    10045      0.00
    10046      4.99
    10047      4.99
    10048      0.00
    10049      0.00
    10050      4.99
    10051      4.99
    10052      4.99
    10053      1.99
    10054      0.00
    10055      0.00
    10056      0.00
    10057      0.00
    10058      0.00
    10059      0.00
    10060      0.99
    10061      0.00
    10062      0.00
    10063      0.00
    10064      0.00
    10065      0.00
    10066      0.99
    10067      0.00
    10068      0.00
    10069      0.00
    10070      0.00
    10071      0.00
    10072      3.99
    10073      1.49
    10074      0.00
    10075      0.00
    10076      0.00
    10077      0.00
    10078      0.00
    10079      0.00
    10080      0.00
    10081      0.00
    10082      0.00
    10083      0.00
    10084      0.00
    10085      0.00
    10086      0.00
    10087      0.00
    10088      0.00
    10089      0.00
    10090      0.00
    10091      0.00
    10092      0.00
    10093      0.00
    10094      0.00
    10095      0.00
    10096      0.00
    10097      0.00
    10098      0.00
    10099      0.00
    10100      0.00
    10101      0.00
    10102      0.00
    10103      0.00
    10104      0.00
    10105      0.00
    10106      0.00
    10107      0.00
    10108      0.00
    10109      0.00
    10110      0.00
    10111      0.00
    10112      0.00
    10113      0.00
    10114      0.00
    10115      0.00
    10116      0.00
    10117      0.00
    10118      0.00
    10119      0.00
    10120      0.00
    10121      0.00
    10122      2.99
    10123      0.00
    10124      0.00
    10125      0.00
    10126      0.00
    10127      0.00
    10128      0.00
    10129      0.00
    10130      0.00
    10131      0.00
    10132      0.00
    10133      0.00
    10134      0.00
    10135      0.00
    10136      0.00
    10137      0.00
    10138      0.00
    10139      0.00
    10140      0.00
    10141      0.00
    10142      0.00
    10143      0.00
    10144      0.00
    10145      0.00
    10146      0.00
    10147      0.00
    10148      0.00
    10149      0.00
    10150      0.00
    10151      0.00
    10152      0.00
    10153      0.00
    10154      0.00
    10155      0.00
    10156      0.00
    10157      0.00
    10158      0.00
    10159      0.00
    10160      0.00
    10161      0.00
    10162      0.00
    10163      0.00
    10164      0.00
    10165      0.00
    10166      0.00
    10167      0.00
    10168      0.00
    10169      0.00
    10170      0.00
    10171      0.00
    10172      0.00
    10173      0.00
    10174      0.00
    10175      0.00
    10176      0.00
    10177      0.00
    10178      0.00
    10179      0.00
    10180      0.00
    10181      0.00
    10182      0.00
    10183      0.00
    10184      0.00
    10185      0.00
    10186      0.00
    10187      0.00
    10188      0.00
    10189      0.00
    10190      0.00
    10191      0.00
    10192      0.00
    10193      0.00
    10194      0.00
    10195      0.00
    10196      0.00
    10197      0.00
    10198      0.00
    10199      0.00
    10200      0.00
    10201      0.00
    10202      0.00
    10203      0.00
    10204      0.00
    10205      0.00
    10206      0.00
    10207      0.00
    10208      0.00
    10209      0.00
    10210      0.00
    10211      0.00
    10212      0.00
    10213      0.00
    10214      0.00
    10215      0.00
    10216      0.00
    10217      0.00
    10218      0.00
    10219      0.00
    10220      0.00
    10221      0.00
    10222      0.00
    10223      0.00
    10224      0.00
    10225      0.00
    10226      0.00
    10227      0.00
    10228      0.00
    10229      0.00
    10230      0.00
    10231      0.00
    10232      0.00
    10233      0.00
    10234      0.00
    10235      0.00
    10236      0.00
    10237      0.00
    10238      0.00
    10239      0.00
    10240      0.00
    10241      0.00
    10242      0.00
    10243      0.00
    10244      0.00
    10245      0.00
    10246      0.00
    10247      0.00
    10248      0.00
    10249      0.00
    10250      0.00
    10251      0.00
    10252      0.00
    10253      0.00
    10254      0.00
    10255      0.00
    10256      0.00
    10257      0.00
    10258      0.00
    10259      0.00
    10260      0.00
    10261      0.00
    10262      0.00
    10263      0.00
    10264      0.00
    10265      0.00
    10266      0.00
    10267      0.00
    10268      0.00
    10269      0.00
    10270      1.99
    10271      0.00
    10272      0.00
    10273      0.00
    10274      0.00
    10275      0.00
    10276      0.00
    10277      0.00
    10278      0.00
    10279      0.00
    10280      0.00
    10281      0.00
    10282      0.00
    10283      0.00
    10284      0.00
    10285      0.00
    10286      0.00
    10287      0.00
    10288      0.00
    10289      0.00
    10290      0.00
    10291      0.00
    10292      0.00
    10293      0.00
    10294      0.00
    10295      0.00
    10296      0.00
    10297      0.00
    10298      0.00
    10299      0.00
    10300      0.00
    10301      0.00
    10302      0.00
    10303      0.00
    10304      0.00
    10305      0.00
    10306      0.00
    10307      0.00
    10308      0.00
    10309      0.00
    10310      0.00
    10311      0.00
    10312      0.00
    10313      0.00
    10314      0.00
    10315      0.00
    10316      0.00
    10317      0.00
    10318      0.00
    10319      0.00
    10320      0.00
    10321      0.00
    10322      0.00
    10323      0.00
    10324      0.00
    10325      0.00
    10326      0.00
    10327      0.00
    10328      0.00
    10329      0.00
    10330      0.00
    10331      0.00
    10332      0.00
    10333      0.00
    10334      0.00
    10335      0.00
    10336      0.00
    10337      0.00
    10338      0.00
    10339      0.00
    10340      0.00
    10341      0.00
    10342      0.00
    10343      0.00
    10344      0.00
    10345      0.00
    10346      0.00
    10347      0.00
    10348      0.00
    10349      0.00
    10350      0.00
    10351      0.00
    10352      0.00
    10353      0.00
    10354      0.00
    10355      0.00
    10356      0.00
    10357      0.00
    10358      0.00
    10359      0.00
    10360      0.00
    10361      0.00
    10362      0.00
    10363      0.00
    10364      0.00
    10365      0.00
    10366      0.00
    10367      0.00
    10368      0.00
    10369      0.00
    10370      0.00
    10371      0.00
    10372      0.00
    10373      0.00
    10374      0.00
    10375      0.00
    10376      0.00
    10377      0.00
    10378      0.00
    10379      0.00
    10380      0.00
    10381      0.00
    10382      0.00
    10383      0.00
    10384      0.00
    10385      0.00
    10386      0.00
    10387      0.00
    10388      0.00
    10389      0.00
    10390      0.00
    10391      0.00
    10392      0.00
    10393      0.00
    10394      0.00
    10395      0.00
    10396      0.00
    10397      0.00
    10398      0.00
    10399      0.00
    10400      0.00
    10401      0.00
    10402      0.00
    10403      0.00
    10404      0.00
    10405      0.00
    10406      0.00
    10407      0.00
    10408      0.00
    10409      0.00
    10410      0.00
    10411      0.00
    10412      0.00
    10413      0.00
    10414      0.00
    10415      0.00
    10416      0.00
    10417      0.00
    10418      0.00
    10419      0.00
    10420      0.00
    10421      0.00
    10422      0.00
    10423      0.00
    10424      0.00
    10425      0.00
    10426      0.00
    10427      0.00
    10428      0.00
    10429      0.00
    10430      0.00
    10431      0.00
    10432      0.00
    10433      0.00
    10434      0.00
    10435      0.00
    10436      0.00
    10437      0.00
    10438      0.00
    10439      0.00
    10440      0.00
    10441      0.00
    10442      1.99
    10443      0.00
    10444      0.00
    10445      0.00
    10446      2.99
    10447      0.00
    10448      0.99
    10449      0.00
    10450      7.99
    10451      0.00
    10452      0.00
    10453      2.99
    10454      0.00
    10455      0.00
    10456      0.00
    10457      2.99
    10458      0.00
    10459      1.99
    10460      1.00
    10461      0.00
    10462      0.00
    10463      0.00
    10464      0.00
    10465      0.00
    10466      0.00
    10467      0.00
    10468      0.00
    10469      0.00
    10470      0.00
    10471      0.00
    10472      0.00
    10473      0.00
    10474      0.00
    10475      0.00
    10476      0.00
    10477      0.00
    10478      0.00
    10479      0.00
    10480      0.00
    10481      0.00
    10482      0.00
    10483      0.00
    10484      0.00
    10485      0.00
    10486      0.00
    10487      0.00
    10488      0.00
    10489      0.00
    10490      0.00
    10491      0.00
    10492      0.00
    10493      0.00
    10494      0.00
    10495      0.00
    10496      0.00
    10497      0.00
    10498      0.00
    10499      0.00
    10500      0.00
    10501      0.00
    10502      0.00
    10503      0.00
    10504      0.00
    10505      0.00
    10506      0.00
    10507      0.00
    10508      0.00
    10509      0.00
    10510      0.00
    10511      0.00
    10512      0.00
    10513      0.00
    10514      0.00
    10515      0.00
    10516      0.00
    10517      1.49
    10518      0.00
    10519      0.00
    10520      0.00
    10521      0.00
    10522      0.00
    10523      0.00
    10524      0.00
    10525      0.00
    10526      0.00
    10527      0.00
    10528      0.00
    10529      0.00
    10530      0.00
    10531      3.49
    10532      0.00
    10533      0.00
    10534      0.00
    10535      0.00
    10536      0.00
    10537      0.00
    10538      0.00
    10539      0.00
    10540      2.99
    10541      0.00
    10542      0.00
    10543      0.00
    10544      0.00
    10545      0.00
    10546      0.00
    10547      0.00
    10548      0.00
    10549      0.00
    10550      0.00
    10551      0.00
    10552      0.00
    10553      0.00
    10554      0.00
    10555      0.00
    10556      0.00
    10557      0.00
    10558      0.00
    10559      0.00
    10560      0.00
    10561      0.00
    10562      0.00
    10563      0.00
    10564      0.00
    10565      0.00
    10566      0.00
    10567      0.00
    10568      0.00
    10569      0.00
    10570      1.99
    10571      0.00
    10572      0.00
    10573      0.00
    10574      0.00
    10575      0.00
    10576      0.00
    10577      0.00
    10578      0.00
    10579      0.00
    10580      0.00
    10581      0.00
    10582      0.00
    10583      6.99
    10584      0.00
    10585      0.00
    10586      0.99
    10587      0.00
    10588      0.00
    10589      0.00
    10590      0.00
    10591      0.00
    10592      0.00
    10593      0.00
    10594      1.99
    10595      0.00
    10596      0.00
    10597      0.00
    10598      0.00
    10599      0.00
    10600      0.00
    10601      0.00
    10602      0.00
    10603      0.00
    10604      0.00
    10605      0.00
    10606      0.00
    10607      0.00
    10608      0.00
    10609      0.00
    10610      0.00
    10611      0.00
    10612      0.00
    10613      0.00
    10614      0.00
    10615      0.00
    10616      0.00
    10617      0.00
    10618      0.00
    10619      0.00
    10620      0.00
    10621      0.00
    10622      0.00
    10623      0.00
    10624      0.00
    10625      0.00
    10626      0.00
    10627      0.00
    10628      0.00
    10629      0.00
    10630      0.00
    10631      0.00
    10632      0.00
    10633      0.00
    10634      0.00
    10635      0.00
    10636      0.00
    10637      0.00
    10638      0.00
    10639      0.00
    10640      0.00
    10641      0.00
    10642      0.00
    10643      0.00
    10644      0.00
    10645      8.99
    10646      0.00
    10647      0.00
    10648      0.00
    10649      0.00
    10650      5.49
    10651      6.49
    10652      0.00
    10653      0.00
    10654      0.00
    10655      0.00
    10656      0.00
    10657      0.00
    10658      0.00
    10659      0.00
    10660      0.00
    10661      5.99
    10662      6.49
    10663      0.00
    10664      6.49
    10665      0.00
    10666      0.00
    10667      0.00
    10668      6.49
    10669      5.99
    10670      0.00
    10671      0.00
    10672      0.00
    10673      0.00
    10674      5.49
    10675      0.99
    10676      0.00
    10677      0.00
    10678      0.00
    10679      2.99
    10680      0.00
    10681      0.00
    10682      0.99
    10683      0.00
    10684      0.00
    10685      0.00
    10686      0.00
    10687      0.00
    10688      0.00
    10689      0.00
    10690      0.99
    10691      0.00
    10692      0.00
    10693      0.00
    10694      0.00
    10695      0.00
    10696      0.00
    10697      0.99
    10698      0.00
    10699      0.00
    10700      0.00
    10701      0.00
    10702      0.00
    10703      0.00
    10704      0.00
    10705      0.00
    10706      0.00
    10707      0.00
    10708      0.00
    10709      0.00
    10710      0.00
    10711      0.00
    10712      0.00
    10713      0.00
    10714      0.00
    10715      0.00
    10716      0.00
    10717      0.00
    10718      0.00
    10719      0.00
    10720      0.00
    10721      0.00
    10722      0.00
    10723      0.00
    10724      0.00
    10725      0.00
    10726      0.00
    10727      0.00
    10728      0.00
    10729      0.00
    10730      0.00
    10731      0.00
    10732      0.00
    10733      0.00
    10734      0.00
    10735      0.99
    10736      0.00
    10737      0.00
    10738      0.00
    10739      0.00
    10740      0.00
    10741      0.00
    10742      0.00
    10743      0.00
    10744      0.00
    10745      0.00
    10746      0.00
    10747      0.00
    10748      0.00
    10749      0.00
    10750      0.00
    10751      0.00
    10752      0.00
    10753      0.00
    10754      0.00
    10755      0.00
    10756      0.00
    10757      0.00
    10758      0.00
    10759      0.00
    10760      7.99
    10761      0.00
    10762      0.00
    10763      0.00
    10764      0.00
    10765      0.00
    10766      0.00
    10767      0.00
    10768      0.00
    10769      0.00
    10770      0.00
    10771      0.00
    10772      0.00
    10773      0.00
    10774      0.00
    10775      0.00
    10776      0.00
    10777      0.00
    10778      0.00
    10779      0.00
    10780      0.00
    10781      0.00
    10782     16.99
    10783      0.00
    10784      0.00
    10785      1.20
    10786      0.00
    10787      0.00
    10788      0.00
    10789      0.00
    10790      0.00
    10791      0.00
    10792      0.00
    10793      0.00
    10794      0.00
    10795      0.00
    10796      0.00
    10797      0.00
    10798      1.04
    10799      0.00
    10800      0.00
    10801      0.00
    10802      0.00
    10803      0.00
    10804      0.00
    10805      0.00
    10806      0.00
    10807      0.00
    10808      0.00
    10809      0.00
    10810      0.00
    10811      0.00
    10812      0.00
    10813      0.00
    10814      0.00
    10815      0.00
    10816      0.00
    10817      0.00
    10818      0.00
    10819      0.00
    10820      0.00
    10821      0.00
    10822      0.00
    10823      0.00
    10824      0.00
    10825      0.00
    10826      0.00
    10827      0.00
    10828      0.00
    10829      0.00
    10830      0.00
    10831      0.00
    10832      0.00
    10833      0.00
    10834      0.00
    10835      0.00
    10836      0.00
    10837      0.00
    10838      0.00
    10839      0.00
    10840      0.00
    Name: Price, dtype: float64




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Size_in_bytes</th>
      <th>Installs</th>
      <th>Price</th>
      <th>Size_in_Mb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9367.000000</td>
      <td>1.084100e+04</td>
      <td>9.146000e+03</td>
      <td>1.084100e+04</td>
      <td>10841.000000</td>
      <td>9146.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.191513</td>
      <td>4.441119e+05</td>
      <td>2.255921e+07</td>
      <td>1.546291e+07</td>
      <td>1.027273</td>
      <td>21.514141</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.515735</td>
      <td>2.927629e+06</td>
      <td>2.368595e+07</td>
      <td>8.502557e+07</td>
      <td>15.948971</td>
      <td>22.588679</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000e+00</td>
      <td>8.704000e+03</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.008301</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>3.800000e+01</td>
      <td>5.138022e+06</td>
      <td>1.000000e+03</td>
      <td>0.000000</td>
      <td>4.900000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.300000</td>
      <td>2.094000e+03</td>
      <td>1.363149e+07</td>
      <td>1.000000e+05</td>
      <td>0.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.500000</td>
      <td>5.476800e+04</td>
      <td>3.145728e+07</td>
      <td>5.000000e+06</td>
      <td>0.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>7.815831e+07</td>
      <td>1.048576e+08</td>
      <td>1.000000e+09</td>
      <td>400.000000</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
