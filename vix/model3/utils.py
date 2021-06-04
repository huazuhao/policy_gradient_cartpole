import numpy as np
from copulas.multivariate import Multivariate

def day_counter_helper(vix_measure_list,threshold):

    counting_days = []
    current_episode = []
    current_counter = 0

    for time_index in range(len(vix_measure_list)-1,0,-1):

        current_measure = vix_measure_list[time_index]
        previous_measure = vix_measure_list[time_index-1]

        if previous_measure<threshold and current_measure >= threshold:
            current_counter += 1
            current_episode.append(current_counter)
            current_episode = current_episode[::-1]
            for entry in current_episode:
                counting_days.append(entry)
            current_counter = 0
            current_episode = []
        else:
            current_counter += 1
            current_episode.append(current_counter)

    current_episode = current_episode[::-1]
    for entry in current_episode:
        counting_days.append(entry)
    

    counting_days = counting_days[::-1]
    
    return counting_days



def construct_a_copula(cov_input,marginal_input):
    
    assert len(cov_input)<4
        
    if len(cov_input)==3:
        diag_1 = 1
        diag_2 = np.sqrt(1-cov_input[0]**2)
        if (1-cov_input[1]**2-cov_input[2]**2)<0:
            diag_3 = np.sqrt(np.abs(1-cov_input[1]**2-cov_input[2]**2))
        else:
            diag_3 = np.sqrt(1-cov_input[1]**2-cov_input[2]**2)

        lower_triangular_matrix = np.asarray([[diag_1,0,0],
                                [cov_input[0],diag_2,0],
                                [cov_input[1],cov_input[2],diag_3]])

        cov_matrix = lower_triangular_matrix@lower_triangular_matrix.transpose()

    
    if len(cov_input)==1:
        diag_1 = 1
        diag_2 = np.sqrt(1-cov_input[0]**2)
        
        lower_triangular_matrix = np.asarray([[diag_1,0],
                                [cov_input[0],diag_2]])
        
        cov_matrix = lower_triangular_matrix@lower_triangular_matrix.transpose()
    
    
    if len(marginal_input)== 6:
        univerates = [{'loc': 0,
        'scale': 1,
        'a': marginal_input[0],
        'b': marginal_input[1],
        'type': 'copulas.univariate.beta.BetaUnivariate'},
        {'loc': 0,
        'scale': 1,
        'a': marginal_input[2],
        'b': marginal_input[3],
        'type': 'copulas.univariate.beta.BetaUnivariate'},
        {'loc': 0,
        'scale': 1,
        'a': marginal_input[4],
        'b': marginal_input[5],
        'type': 'copulas.univariate.beta.BetaUnivariate'}]
        
    if len(marginal_input)==4:
        univerates = [{'loc': 0,
        'scale': 1,
        'a': marginal_input[0],
        'b': marginal_input[1],
        'type': 'copulas.univariate.beta.BetaUnivariate'},
        {'loc': 0,
        'scale': 1,
        'a': marginal_input[2],
        'b': marginal_input[3],
        'type': 'copulas.univariate.beta.BetaUnivariate'}]
    
    copula_parameters = {}
    copula_parameters['covariance'] = cov_matrix
    copula_parameters['univariates'] = univerates
    copula_parameters['type'] = 'copulas.multivariate.gaussian.GaussianMultivariate'
    if len(marginal_input)== 6:
        copula_parameters['columns'] = [0,1,2]
    if len(marginal_input)==4:
        copula_parameters['columns'] = [0,1]
        
    new_dist = Multivariate.from_dict(copula_parameters)
    
    return new_dist