import numpy as np

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


def compute_total_portfolio_value(vix_cash,vix_amount,vix_price,spy_cash,spy_positions,spy_price):
    total_value = 0
    
    total_value += vix_cash
    total_value += vix_amount*vix_price

    total_value += spy_cash
    for position in spy_positions:
        total_value += position['quantity']*spy_price
    
    return total_value

def total_value_in_vix(vix_cash,vix_amount,vix_price):
    value_in_vix = 0
    value_in_vix += vix_cash
    value_in_vix += vix_amount*vix_price

    return value_in_vix

def total_value_in_spy(spy_cash,spy_positions,spy_price):
    value_in_spy = 0
    value_in_spy += spy_cash
    for position in spy_positions:
        value_in_spy += position['quantity']*spy_price
    
    return value_in_spy

def returned_spy_position(spy_cash,spy_positions,spy_price):
    value_in_stock_form = 0
    for position in spy_positions:
        value_in_stock_form += position['quantity']*spy_price
    
    if (value_in_stock_form + spy_cash) > 0:
        return value_in_stock_form/(value_in_stock_form + spy_cash)
    else:
        return 0

def generate_spy_bounds(price_history,window_length,denoise_factor,lower_bound_factor,upper_bound_factor):

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

        if denoise_factor > 100:
            print('the denoise_factor is',denoise_factor)
            print('the mean is',mean)
            print('the std is',std)
            print('the price history is',price_history)
            print('the trimmed history is',trimmed_history)
            raise ValueError('the denoise factor is too large')

    trimmed_mean = np.mean(trimmed_history)
    trimmed_std = np.std(trimmed_history)

    lower_bound = trimmed_mean - trimmed_std*lower_bound_factor
    upper_bound = trimmed_mean + trimmed_std*upper_bound_factor

    return lower_bound,upper_bound