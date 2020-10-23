import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

pixel_len = 1.135  # micrometers
np.random.seed(42)

# takes one series, time(s) sorted by distance(pixels) and returns a series of speed. Sorted by distance
# Last value is undefined, so we just make it the same as the previous velocity.
# This keeps all the columns the same length
def calc_vel(time, pixel_len=pixel_len):
    r = [pixel_len/(time[i+1]-time[i]) for i in range(len(time)-1)]
    r.append(r[-1])
    return r



def get_velocities(df, time_cols, vel_cols):
    for i, j in zip(time_cols, vel_cols):
        df[j] = calc_vel(df[i])


# takes a one wave dataframe, appends to it columns for first 50 vel, max vel, 2nd 50 vel
#second param is a list of column names we're going to be using to calc vel. Defaults should be good
def get_velocities_one_wave(df, time_cols=('wave one max time', 'wave one first fifty time',
                                'wave one second fifty time'),
                      vel_cols=('wave one max velocity', 'wave one first fifty velocity',
                                'wave one second fifty velocity')):
        get_velocities(df, time_cols, vel_cols)

def get_velocities_two_wave(df,
                      time_cols=('wave one max time', 'wave one first fifty time',
                                 'wave one second fifty time', 'wave two max time',
                                 'wave two first fifty time', 'wave two second fifty time'),
                      vel_cols=('wave one max velocity', 'wave one first fifty velocity',
                                'wave one second fifty velocity', 'wave two max velocity',
                                'wave two first fifty velocity', 'wave two second fifty velocity')):
    get_velocities(df, time_cols, vel_cols)




two_wave_data_paths = ['013020S001_velocity.csv', '013020S006_velocity.csv', '013020S015_velocity.csv', '013120S013_velocity.csv',
                '013020S002_velocity.csv', '013020S012_velocity.csv', '013120S011_velocity.csv',
                '013020S004_velocity.csv', '013020S013_velocity.csv', '013120S012_velocity.csv']

one_wave_data_paths = ['013020S009_velocity.csv', '022620S009_velocity.csv', '022820S003_velocity.csv', '022820S008_velocity.csv',
                        '013020S016_velocity.csv', '022820S001_velocity.csv', '022820S004_velocity.csv',
                        '022620S004_velocity.csv', '022820S002_velocity.csv', '022820S005_velocity.csv']
one_wave_data = [pd.read_csv('/home/patrick/Documents/College/2020-Spring/geco/processed_data/regression/1_wave_velocity/' + x,
                             index_col=0) for x in one_wave_data_paths]

two_wave_data = [pd.read_csv('/home/patrick/Documents/College/2020-Spring/geco/processed_data/regression/2_wave_velocity/' + x,
                             index_col=0) for x in two_wave_data_paths]

#one_wave_data = [get_velocities_one_wave(x) for x in one_wave_data]
#two_wave_data = [get_velocities_two_wave(x) for x in two_wave_data]
for x in one_wave_data:
    get_velocities_one_wave(x)

for x in two_wave_data:
    get_velocities_two_wave(x)

for d, n in zip(one_wave_data, one_wave_data_paths):
    d.to_csv('/home/patrick/geco/processed_data/regression/velocity_calculations/1_wave/'+n)

for d, n in zip(two_wave_data, two_wave_data_paths):
    d.to_csv('/home/patrick/geco/processed_data/regression/velocity_calculations/2_wave/'+n)
