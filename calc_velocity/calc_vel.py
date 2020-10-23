import sys
import pandas as pd
from scipy import signal
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from calc_wave_properties import butterworth_filter_data_row_only, get_wave_prop_one_wave_frame, get_wave_prop_two_wave_frame, get_critical_points_dist

time_scalar = 2.58
micro_scale = 10**-6
w_max = .5*(1/(time_scalar*micro_scale))
#w_cutoff = 2*np.pi*.5*w_max
w_cutoff_row = 0.075
w_cutoff_col = 0.075
pixel_len = 1.135

d_scale  = 1.135
time_scale = 2.58
n_x_ticks = 4
n_y_ticks = 4

MINIMA = 'minima'
MAXIMA = 'maxima'
NEITHER = 'neither'

'''
def get_max_list(data):
    return [max(row) for row in data.iloc[:-1]]

def get_50_times(data):
    # Get list of max values for each distance
    max_list = get_max_list(data)



# Pass one row of data, each col will be sample
def butterworth_filter_row(row, order=3, wn=w_cutoff_row, type='low'):
    b, a = signal.butter(order, wn)
    #w, h = signal.freqz(b, a)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, row, zi=zi*row[0])
    z2, _ = signal.lfilter(b,a, z, zi=zi*z[0])

    y = signal.filtfilt(b, a, row)

    return y

def butterworth_filter_col(col, order=3, wn=w_cutoff_col):
    b, a = signal.butter(order, wn)
    #w, h = signal.freqz(b, a)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, col, zi=zi*col[0])
    z2, _ = signal.lfilter(b,a, z, zi=zi*z[0])

    y = signal.filtfilt(b, a, col)
    return y


def butterworth_filter_data(df, order, wn):
    return pd.DataFrame([butterworth_filter_row(df.iloc[i], wn=wn) for i in range(df.shape[0])],
                        columns=list(map(float, df.columns))).apply(butterworth_filter_col, axis=1,
                                                                                           result_type='broadcast')


def univariate_spline_row(y, x, s):
    print(x)
    print(y)
    spl = interpolate.UnivariateSpline(x, y, s=s)
    #spl.set_smoothing_factor(smoothing_factor)
    return spl(y)

def univariate_spline_data(s, data):
    return data.apply(univariate_spline_row, axis=1, raw=False, result_type='broadcast', args=(data.columns, s))


def second_derivative_test(val, second_derivative_spline):
    if second_derivative_spline(val) > 0:       # Positive 2nd derivative means minima
        return MINIMA
    elif second_derivative_spline(val) < 0:     # Negative 2nd derivative means maxima
        return MAXIMA
    else:
        return NEITHER

def get_local_min_max(data, s=.1):
    lsplines_4 = [interpolate.UnivariateSpline(data.columns, data.iloc[y], s=s, k=4) for y in range(data.shape[0])]
    lsplines_5 = [interpolate.UnivariateSpline(data.columns, data.iloc[y], s=s, k=5) for y in range(data.shape[0])]

    lsplines_prime = [i.derivative() for i in lsplines_4]
    lsplines_prime_prime = [i.derivative(2) for i in lsplines_5]

    roots = [i.roots() for i in lsplines_prime]
    root_min_max = [[(r, lsplines_4[x](r), second_derivative_test(r, lsplines_prime_prime[x]))
                     for r in roots[x]] for x in range(len(roots))]

    return root_min_max


# We pass the end point times and the minima maxima list
def compute_peak_ranges(min_time, max_time, mml):
    int1l, int1r, int2l, int2r = 0, 0, 0, 0
    max1_t, max1_m, max2_t, max2_m = 0, 0, 0, 0
    for i in range(len(mml)):
        t, m, l = mml[i]
        m = m.item()
        if m >= max1_m:
            # interval two = interval one
            int2l = int1l
            int2r = int1r
            # max two = max one
            max2_t = max1_t
            max2_m = max1_m
            # max one = new max
            max1_m = m
            max1_t = t
            # interval one = new interval
            if i == 0:
                int1l = min_time
                int1r = mml[i+1][0]
            elif i == len(mml)-1:
                int1l = mml[i-1][0]
                int1r = max_time
            else:
                int1l = mml[i-1][0]
                int1r = mml[i+1][0]
        elif m >= max2_m:
            # interval two = new interval
            if i == 0:
                int2l = min_time
                int2r = mml[i+1][0]
            elif i == len(mml)-1:
                int2l = mml[i-1][0]
                int2r = max_time
            else:
                int2l = mml[i-1][0]
                int2r = mml[i+1][0]
            # max two = new max
            max2_t = t
            max2_m = m

    if int1l < int2l:
        return int1l, int1r, int2l, int2r
    else:
        return int2l, int2r, int1l, int1r


def get_nearest_times(times, int1l, int1r, int2l, int2r):
    t1l, t1r, t2l, t2r = times[0], times[0], times[0], times[0]
    for i in range(len(times)):
        if abs(int1l - times[i]) < abs(int1l - times[t1l]):
            t1l = i
        if abs(int1r - times[i]) < abs(int1r - times[t1r]):
            t1r = i
        if abs(int2l - times[i]) < abs(int2l - times[t2l]):
            t2l = i
        if abs(int2r - times[i]) < abs(t2r - times[t2r]):
            t2r = i

    return t1l, t1r, t2l, t2r
    t1l, t1r, t2l, t2r = times[0], times[0], times[0], times[0]
    for t in times:
        if abs(int1l - t) <= abs(int1l - t1l):
            t1l = t
        if abs(int1r - t) <= abs(int1r - t1r):
            t1r = t
        if abs(int2l - t) <= abs(int2l - t2l):
            t2l = t
        if abs(int2r - t) <= abs(int2r - t2r):
            t2r = t

    return t1l, t1r, t2l, t2r


# Takes times as floats and the approximated minima, maxima list. Returns endpoints of the two intervals as indices
def compute_interval_endpoints(times, mml):
    int1l, int1r, int2l, int2r = compute_peak_ranges(times[0], times[-1], mml)
    return get_nearest_times(times, int1l, int1r, int2l, int2r)


def compute_half_max_wave_points(data, mml):
    vel_data = pd.DataFrame(columns=['distance', 'wave1_fifty_0', 'wave1_fifty_1', 'wave2_fifty_0', 'wave2_fifty_1'])
    for ri in range(data.shape[0]):  # For each distance
        # get interval endpoints
        w1l, w1r, w2l, w2r = compute_interval_endpoints(data.columns, mml[ri])
        w1f1, w1f2, w2f1, w2f2 = 0, 0, 0, 0
        series_1 = data.iloc[ri].loc[w1l:w1r]
        series_2 = data.iloc[ri].loc[w2l:w2r]
        # for both intervals
            # get max in interval
            # get first and second fifty value
        cmax = max(series_1)
        f = .5 * cmax
        for val, label in zip(series_1, series_1.index):
            if w1f1 is 0 and val >= f:
                w1f1 = label
            elif w1f1 is not 0 and w1f2 is 0 and val < f:
                w1f2 = label
        cmax = max(series_2)
        f = .5*cmax
        for val, label in zip(series_2, series_2.index):
            if w2f1 is 0 and val >= f:
                w2f1 = label
            elif w2f1 is not 0 and w2f2 is 0 and val < f:
                w2f2 = label
        vel_data.loc[-1] = [ri*pixel_len, w1f1, w1f2, w2f1, w2f2]
        vel_data.index = vel_data.index + 1
        # append to df

    # sort df by index, return df
    return vel_data.sort_index()
'''

def surface_plot(data):
    X = [[i for x in range(data.shape[1])] for i in range(data.shape[0])]
    Y = [[i for i in range(data.shape[1])] for x in range(data.shape[0])]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Time')
    ax.set_zlabel('Average Pixel Value')
    surf = ax.plot_surface(X, Y, data, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf)

    plt.show()


def wireframe_plot(data):
    X = [[i for x in range(data.shape[1])] for i in range(data.shape[0])]
    Y = [[i for i in range(data.shape[1])] for x in range(data.shape[0])]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, data, rstride=10, cstride=10)
    plt.show()

def wave_properties_plot(start, end, step, vel_data, raw_data):
    filtered_data = butterworth_filter_data_row_only(raw_data)
    for i in range(start, end, step):
        c = vel_data.iloc[i]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(filtered_data.columns, raw_data.iloc[i], 'yo', filtered_data.iloc[i], 'b--')
        if nwaves is 'two':
            ax.plot(c['wave one first fifty time'], c['wave one first fifty value'], 'go', c['wave two max time'], c['wave two max value'], 'ro', c['wave one max time'], c['wave one max value'], 'go',
                    c['wave two first fifty time'], c['wave two first fifty value'], 'ro', c['wave two second fifty time'], c['wave two second fifty value'], 'ro',
                    c['wave one second fifty time'], c['wave one second fifty value'], 'go')
            ax.legend(('Raw Data', 'Butterworth Filtered Data', 'Approximated Wave One Properties',
                       'Approximated Wave Two Properties'), loc='best')
        else:
            ax.plot(c['wave one first fifty time'], c['wave one first fifty value'], 'go', c['wave one max time'], c['wave one max value'], 'go',
                    c['wave one second fifty time'], c['wave one second fifty value'], 'go')
            ax.legend(('Raw Data',  'Butterworth Filtered Data', 'Approximated Wave One Properties'), loc='best')
        ax.set_xlabel('Time (microseconds)')
        ax.set_ylabel('Average Pixel Value')
        ax.set_title('Distance ' + str(i*pixel_len) + ' Raw: ' + str(i))

        plt.show()
        plt.savefig('/home/patrick/Documents/College/2020-Spring/geco/graphics/test_data_wave_properties_graphs/e5/dist_'+str(i*pixel_len)+'.png')
        plt.close()


def critical_points_plot(start, end, step, data):
    butterworth_filtered_data_rows = butterworth_filter_data_row_only(data)
    #one_wave_prop = get_wave_prop_one_wave_frame(data)
    two_wave_prop = get_wave_prop_two_wave_frame(data)

    for i in range(start, end, step):
        spline_4 = interpolate.UnivariateSpline(butterworth_filtered_data_rows.columns, butterworth_filtered_data_rows.iloc[i],k=4)
        cp = get_critical_points_dist(butterworth_filtered_data_rows.iloc[i])
        cp_x = [x for x,y,z in cp]
        cp_y = [y for x,y,z in cp]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(data.iloc[i], 'yo',butterworth_filtered_data_rows.columns, butterworth_filtered_data_rows.iloc[i], 'b',
                 butterworth_filtered_data_rows.columns, spline_4(butterworth_filtered_data_rows.columns), 'g',
                 cp_x, cp_y, 'ro')


def save_heatmap(data, vel_data, rowlabels, collabels, title, imgname):
    #plt.gray()
    keys = {'cmap':'Greys', 'origin':'lower', 'aspect':'auto'}
    #plt.matshow(data, cmap='Greys', origin='lower')
    plt.matshow(data, **keys)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, **keys)

    fig.colorbar(cax, boundaries=range(0, 250))
    #ax.set_xticks([int(float(x)) for x in list(data.columns)])
    #ax.set_xticks(list(map(str, range(0, int(float(data.columns[-1])), 50))))
    #ax.set_yticks(np.arange(data.shape[1]))
    #ax.set_xticks(np.arange(n_x_ticks))
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    #ax.set_ybound(0, 220)
    ax.set_xticklabels([0] + list(map(str, range(0, int(float(data.columns[-1])), int(float(data.columns[-1])/n_x_ticks)))))
    ax.set_yticklabels([0] + list(range(0, int(data.shape[1]*d_scale), int(float(data.shape[1]*d_scale/n_y_ticks)))))
    #ax.set_xticklabels(list(map(int, map(float, data.columns))))
    plt.xlabel(rowlabels)
    plt.ylabel(collabels)

    for edge, spine in ax.spines.items():
        spine.set_visible = False

    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.set_title(title, pad=15)

    for i in reversed(range(len(vel_data['wave one reached']))):
        if vel_data['wave one reached'][i]:
            plt.plot([i for x in range(data.shape[1])], 'b--')
            break

    if nwaves is 'two':
        for i in reversed(range(len(vel_data['wave two reached']))):
            if vel_data['wave two reached'][i]:
                plt.plot([i for x in range(data.shape[1])], 'g--')
                break

    plt.tight_layout()
    plt.savefig(imgname)


if len(sys.argv) < 4:
    print('Too few arguments. Include \'source\', \'destination\', and \'one\' or \'two\', in that order.')
    print('The optional argument(s) [pixel_len] should be appended to the arguments.')
else:
    src = sys.argv[1]
    dest = sys.argv[2]
    print(f'src: {src}\ndest: {dest}\n')
    nwaves = sys.argv[3]
    if len(sys.argv) > 4:
        pixel_len = sys.argv[4]
    data = pd.read_csv(src)
    data = pd.DataFrame(data, columns=list(map(float, data.columns)))

    print(f'\'{nwaves}\' == \'one\' {nwaves == "one"}')
    if nwaves == 'one':
        print('Computing properties for one wave. Analysis underway...')
        vel_data = get_wave_prop_one_wave_frame(data)
    else:
        print('Computing properties for two waves. Analysis underway...')
        vel_data = get_wave_prop_two_wave_frame(data)

    vel_data.to_csv(dest)
    print('Done!')
