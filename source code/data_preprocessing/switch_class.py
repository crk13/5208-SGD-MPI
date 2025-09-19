#实现类别型变量的处理

import pandas as pd
from pandas.api.types import CategoricalDtype

def switch_class(df):
    # RatecodeID转为独热编码
    ratecode_dummies = pd.get_dummies(df['RatecodeID'], prefix='Ratecode')
    df = pd.concat([df, ratecode_dummies], axis=1)
    df = df.drop(columns=['RatecodeID'])

    #  Payment_type转为类别型
    payment_type_categories = [1, 2, 3, 4]
    payment_type_type = CategoricalDtype(categories=payment_type_categories, ordered=False)
    df['payment_type'] = df['payment_type'].astype(payment_type_type)
    return df