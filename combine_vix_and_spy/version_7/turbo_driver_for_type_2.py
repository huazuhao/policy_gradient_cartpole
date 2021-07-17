from turbo import TurboM
import os
import turbo_objective_function_type_2
import numpy as np
import jsonpickle

if __name__ == '__main__':

    #first, initialize the objective function class
    f = turbo_objective_function_type_2.turbo_objective_function()


    #second, define the turbo class
    turbo_m = TurboM(
        f=f,  # Handle to objective function
        lb=f.lb,  # Numpy array specifying lower bounds
        ub=f.ub,  # Numpy array specifying upper bounds
        n_init=50,  # Number of initial trials from an Symmetric Latin hypercube design
        max_evals=1000,  # Maximum number of evaluations
        n_trust_regions=4,  # Number of trust regions
        batch_size=50,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos, from an exact method to an approximation method
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=2048,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )

    #begin optimization
    turbo_m.optimize()

    #get best parameters
    X = turbo_m.X  # Evaluated points
    fX = turbo_m.fX  # Observed values
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]

    print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))

    #store best parameters
    best_free_parameters = {}
    for index in range(f.dim):
        var_name = f.var_name_list[index]
        var_value = x_best[index]
        best_free_parameters[var_name] = var_value
    
    cwd = os.getcwd()
    #cwd = os.path.join(cwd, 'data_folder')
    parameter_file = 'best_free_parameters_from_turbo_type_2.json'
    cwd = os.path.join(cwd,parameter_file)
    with open(cwd, 'w') as statusFile:
        statusFile.write(jsonpickle.encode(best_free_parameters))


    #store the optimization process
    optimization_process = {}
    optimization_process['fx'] = fX

    cwd = os.getcwd()
    #cwd = os.path.join(cwd, 'data_folder')
    parameter_file = 'optimization_process_data_type_2.json'
    cwd = os.path.join(cwd,parameter_file)
    with open(cwd, 'w') as statusFile:
        statusFile.write(jsonpickle.encode(optimization_process))