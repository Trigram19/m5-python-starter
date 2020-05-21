import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy;from scipy import stats
sales = pd.read_csv('sales_train_validation.csv')
stonks = pd.read_csv('sell_prices.csv')
cal = pd.read_csv('calendar[1].csv')
COLUMN_1 = [14, 30, 60, 120, 180, 360]
COLUMN_2 = [7, 14, 28, 56]

def rolling_mean_gen(df):
    for i in COLUMN_1:
        df[f'rolling_s{i}'] = df.groupby(['id'])['d_1'].transform(lambda x: x.shift(28).rolling(i).mean())        
        return df
    
def lag_feats(df):
    for i in range(1, 1913):
        df['lag_t14'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: x.shift(14))
        df['lag_t7'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: x.shift(7))
        df['lag_t28'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: x.shift(28))
        df['lag_t56'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: x.shift(56))
        df['stddev'] = df.groupby(['id'])[f'd_{i}'].transform('std')
        df['lag_t14'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.std(x.shift(14)))
        df['lag_t7'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.std(x.shift(7)))
        df['lag_t28'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.std(x.shift(28)))
        df['lag_t56'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.std(x.shift(56)))
        return df
    
def fourier_et_al(df):
    for i in range(1, 1913):
        df['fourier_lag_t14'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.fft.fft(x.shift(14)))
        df['fourier_lag_t7'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.fft.fft(x.shift(7)))
        df['fourier_lag_t28'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.fft.fft(x.shift(28)))
        df['fourier_lag_t56'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.fft.fft(x.shift(56)))

        df['ifourier_lag_t14'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.fft.ifft(x.shift(14)))
        df['ifourier_lag_t7'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.fft.ifft(x.shift(7)))
        df['ifourier_lag_t28'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.fft.ifft(x.shift(28)))
        df['ifourier_lag_t56'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: np.fft.ifft(x.shift(56)))

        df['dfourier_lag_t14'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.dct(x.shift(14), type=2, norm='ortho'))
        df['dfourier_lag_t7'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.dct(x.shift(7), type=2, norm='ortho'))
        df['dfourier_lag_t28'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.dct(x.shift(28), type=2, norm='ortho'))
        df['dfourier_lag_t56'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.dct(x.shift(56), type=2, norm='ortho'))

        df['idfourier_lag_t14'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.idct(x.shift(14), type=2, norm='ortho'))
        df['idfourier_lag_t7'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.idct(x.shift(7), type=2, norm='ortho'))
        df['idfourier_lag_t28'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.idct(x.shift(28), type=2, norm='ortho'))
        df['idfourier_lag_t56'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.idct(x.shift(56), type=2, norm='ortho'))

        df['dsfourier_lag_t14'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.dst(x.shift(14), type=2, norm='ortho'))
        df['dsfourier_lag_t7'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.dst(x.shift(7), type=2, norm='ortho'))
        df['dsfourier_lag_t28'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.dst(x.shift(28), type=2, norm='ortho'))
        df['dsfourier_lag_t56'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.dst(x.shift(56), type=2, norm='ortho'))

        df['idfourier_lag_t14'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.idst(x.shift(14), type=2, norm='ortho'))
        df['idfourier_lag_t28'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.idst(x.shift(28), type=2, norm='ortho'))
        df['idfourier_lag_t56'] = df.groupby(['id'])[f'd_{i}'].transform(lambda x: scipy.fft.idst(x.shift(56), type=2, norm='ortho'))
    

def other_feats(df):
    df['price_max'] = df.groupby(['store_id','item_id'])['sell_price'].transform('max')
    df['price_min'] = df.groupby(['store_id','item_id'])['sell_price'].transform('min')
    df['price_std'] = df.groupby(['store_id','item_id'])['sell_price'].transform('std')
    df['price_mean'] = df.groupby(['store_id','item_id'])['sell_price'].transform('mean')
    return df

import gc
lag_feats(sales)
rolling_mean_gen(sales)
fourier_et_al(sales)
sales.to_csv('training.csv.zip', index=False)
gc.collect()
