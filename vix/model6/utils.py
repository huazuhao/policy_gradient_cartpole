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
