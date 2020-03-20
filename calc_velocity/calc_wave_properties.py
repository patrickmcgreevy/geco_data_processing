import pandas as pd
from scipy import signal
from scipy.interpolate import UnivariateSpline

threshold = 0.5
wave_prop_cols = ['wave one max value', 'wave one max time', 'wave one first fifty value', 'wave one first fifty time',
                     'wave one second fifty value', 'wave one second fifty time', 'wave one reached',
                     'wave two max value', 'wave two max time', 'wave two first fifty value', 'wave two first fifty time',
                     'wave two second fifty value', 'wave two second fifty time', 'wave two reached']
cp_labels = ['minima', 'maxima', 'inconclusive', 'endpoint']
n_baseline_frames = 25

# Pass one list of data, each col will be sample
# look at this more closely. I want to really understand this shit
def butterworth_filter_list(list, order=3, wn=0.075, type='low'):
    b, a = signal.butter(order, wn)
    #w, h = signal.freqz(b, a)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, list, zi=zi*list[0])
    z2, _ = signal.lfilter(b,a, z, zi=zi*z[0])

    y = signal.filtfilt(b, a, list)

    return pd.Series(y, list.keys())


def butterworth_filter_data_row_only(df):
    return df.apply(butterworth_filter_list, axis=1, result_type='broadcast')


def second_derivative_test(second_derivative, x):
    if(second_derivative(x) == 0):
        return cp_labels[2]

    return cp_labels[second_derivative(x) < 0]

# Pass row of data as series
def get_critical_points_dist(series):
    spline_4 = UnivariateSpline(series.keys(), series, k=4)
    spline_5 = UnivariateSpline(series.keys(), series, k=5)
    first_derivative = spline_4.derivative()
    second_derivative = spline_5.derivative(2)
    roots = first_derivative.roots()

    #return [(series.keys()[0], series.iloc[0], cp_labels[3])] + [(r, series[r], second_derivative_test(second_derivative, r))
     #                                                                        for r in roots] + [(series.keys()[-1], series.iloc[-1], cp_labels[3])]

    return [(series.keys()[0], spline_4(series.keys()[0]).item(), cp_labels[3])] + [(r, spline_4(r).item(), second_derivative_test(second_derivative, r))
                                                                             for r in roots] + [(series.keys()[-1], spline_4(series.keys()[-1]).item(), cp_labels[3])]

# Pass the whole dataframe
def get_critical_points_data(df):
    return [get_critical_points_dist(df.iloc[i]) for i in range(df.shape[0])]


# Pass list of critical points for a distance; Return time and value of two greatest peaks
def get_peak_pair_time_val(critical_points):
    peaks = [(None, None), (None, None)]
    for t, v, l in critical_points:
        if l is cp_labels[1]:
            if peaks[0] == (None, None):
                peaks[0] = (t, v)
            elif peaks[1] == (None, None):
                peaks[1] = (t, v)
            elif v > peaks[0][1]:
                if peaks[0][1] >= peaks[1][1]:
                    peaks[1] = (t, v)
                else:
                    peaks[0] = peaks[1]
                    peaks[1] = (t, v)
            elif v > peaks[1][1]:
                peaks[1] = (t, v)

    return peaks


def get_adjusted_wave_fifty_val(series, n_baseline_frames, max_val):
    b = series[0:n_baseline_frames].mean()
    return (max_val - b) * 0.5 + b


# Pass list of critical points and the times of the peaks as a tuple
def get_valley_min_time_val(critical_points, peak_times):
    llm = (None, None)
    for t, v, l in critical_points:
        if l is cp_labels[0] and peak_times[0][0] < t < peak_times[1][0]:
           if llm == (None, None) or llm[1] > v:
               llm = (t, v)

    return llm

# Pass row as series & f is function to determine if we're getting first or second fifty
def get_fifty_val_t(series, max_val, f):
    val, time = 0, 0
    adjusted_threshold = get_adjusted_wave_fifty_val(series, n_baseline_frames, max_val)
    for key in series.keys():
        if f(series[key], adjusted_threshold):
            return series[key], key

    return None, None


def get_first_fifty_val_t(series, max_val):
    v, t = get_fifty_val_t(series, max_val, lambda a, b: a >= b)

    if v is None and t is None:
        return series[series.keys()[0]], series.keys()[0]
    else:
        return v, t


def get_second_fifty_val_t(series, max_val):
    v, t = get_fifty_val_t(series, max_val, lambda a, b: a <= b)

    if v is None and t is None:
        return series.iloc[-1], series.keys()[-1]
    else :
        return v, t


def get_max_val_time(series):
    val, time = 0, 0
    for key in series.keys():
        val, time = ((val, time), (series[key], key))[series[key] >= val]
        #val, time = series[key], key if series[key] >= val else val, time = (val, time)

    return val, time

# df of wave data passed by ref; list or float 0<=q<=1
def first_wave_percentile_approx(data, q):
    p = data['wave one max value'].quantile(q)
    for r in range(data.shape[0]):
        if data['wave one max value'][r] <= p:
            data['wave one reached'][r] = 0
        else:
            data['wave one reached'][r] = 1

    return data


def second_wave_percentile_approx(data, q):
    p = data['wave two max value'].quantile(q)
    for r in range(data.shape[0]):
        if data['wave two max value'][r] <= p:
            data['wave two reached'][r] = 0
        else:
            data['wave two reached'][r] = 1

    return data

def get_wave_prop_one_wave_dist(dist):
    max_val, max_time = get_max_val_time(dist)
    first_fifty_val, first_fifty_t = get_first_fifty_val_t(dist[:max_time], max_val)
    second_fifty_val, second_fifty_t = get_second_fifty_val_t(dist[max_time:], max_val)

    return [max_val, max_time, first_fifty_val, first_fifty_t, second_fifty_val, second_fifty_t, None, None, None, None, None, None, None, None]


def get_wave_prop_one_wave_frame(data):
    d = pd.DataFrame.from_records(data.apply(get_wave_prop_one_wave_dist, axis=1), columns=wave_prop_cols)
    return first_wave_percentile_approx(d, .25)


def get_wave_prop_dist(dist):
    max_val, max_time = get_max_val_time(dist)
    first_fifty_val, first_fifty_t = get_first_fifty_val_t(dist[:max_time], max_val)
    second_fifty_val, second_fifty_t = get_second_fifty_val_t(dist[max_time:], max_val)

    return[max_val, max_time, first_fifty_val, first_fifty_t, second_fifty_val, second_fifty_t]


def get_wave_prop_two_wave_dist(dist):
    cp = get_critical_points_dist(butterworth_filter_list(dist))
    approx_peaks = get_peak_pair_time_val(cp)
    llm = get_valley_min_time_val(cp, approx_peaks)
    #true_peaks = [(get_max_val_time(dist[dist.keys()[0]:llm[0]])), (get_max_val_time(dist[llm[0]:dist.keys()[-1]]))]

    return get_wave_prop_dist(dist[:llm[0]]) + [None] + get_wave_prop_dist(dist[llm[0]:]) + [None]

def get_wave_prop_two_wave_frame(data):
    d = pd.DataFrame.from_records(data.apply(get_wave_prop_two_wave_dist, axis=1), columns=wave_prop_cols)
    d = first_wave_percentile_approx(d, .25)
    d = second_wave_percentile_approx(d, .25)
    return d

