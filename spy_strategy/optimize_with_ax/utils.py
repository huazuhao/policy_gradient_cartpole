import numpy as np


def generate_bounds(price_history,window_length,denoise_factor,lower_bound_factor,upper_bound_factor):

    price_history = price_history[-1*window_length:]
    price_history = np.asarray(price_history)
    price_history = np.reshape(price_history,(-1,1))

    mean = np.mean(price_history)
    std = np.std(price_history)

    while True:
        
        trimmed_history = price_history[price_history<mean+std*denoise_factor]
        
        if trimmed_history.shape[0]==0:
            denoise_factor = denoise_factor*1.01
        else:
            trimmed_history = trimmed_history[trimmed_history>mean-std*denoise_factor]

            if trimmed_history.shape[0]==0:
                denoise_factor = denoise_factor*1.01
            else:
                break

    trimmed_mean = np.mean(trimmed_history)
    trimmed_std = np.std(trimmed_history)

    lower_bound = trimmed_mean - trimmed_std*lower_bound_factor
    upper_bound = trimmed_mean + trimmed_std*upper_bound_factor

    return lower_bound,upper_bound
        

