'''
Created on Apr 12, 2013
@author: Stuti Agarwal
'''



import math

def init_matrix(data):
    for i in range(len(data)):
        data[i][0] = float('inf')
    for i in range(len(data[0])):
        data[0][i] = float('inf')
    data[0][0] = 0
    return data

def LpDist(time_pt_1, time_pt_2):
    # if (type(time_pt_1) == int and type(time_pt_2) == int):
    return abs(time_pt_1 - time_pt_2)
    # else:
    #     return sum(abs(time_pt_1 - time_pt_2))

def TWED(t1, t2, lam, nu):
    """"Requires: t1: multivariate time series in numpy matrix format. t2: multivariate time series in numpy matrix format. lam: penalty lambda parameter, nu: stiffness coefficient"""
    """Returns the TWED distance between the two time series. """
    t1_time, t1_data = t1
    t2_time, t2_data = t2
    
    result = [[0]*len(t2_data) for row in range(len(t1_data))]
    result = init_matrix(result)
    
    n = len(t1_data)
    m = len(t2_data)
    
    assert(len(t1_time) == n)
    assert(len(t2_time) == m)
    
    #t1_data[0] = 0 
    #t2_data[0] = 0
    #t1_time[0] = 0
    #t2_time[0] = 0

    for i in range(1, n):
        for j in range(1, m):
            cost = LpDist(t1_data[i], t2_data[j])

            insertion = (result[i-1][j] + LpDist(t1_data[i-1], t1_data[i]) +
                         nu*(t1_time[i] - t1_time[i-1] + lam))

            deletion = (result[i][j-1] + LpDist(t2_data[j-1], t2_data[j]) +
                        nu*(t2_time[j] - t2_time[j-1] + lam))

            #print i, j, n , m, t1_time[i], t2_time[j]
            match = (result[i-1][j-1] + LpDist(t1_data[i], t2_data[j]) +
                     nu*(abs(t1_time[i] - t2_time[j])) +
                     LpDist(t1_time[i-1], t2_time[j-1]) +
                     nu*(abs(t1_time[i-1] -t2_time[j-1])))

            result[i][j] = min(insertion, deletion, match)
    return result[n-1][m-1]

def TwedKernel(t1, t2, lam, nu, sigma):
    """ TWED kernel using time warp edit distance between multivariate time series A and B """
    """Returns the dot product in the gaussian space"""
    D = TWED(t1,t2, lam, nu)

    result = math.exp((-D*D)/2*sigma*sigma)

    return result