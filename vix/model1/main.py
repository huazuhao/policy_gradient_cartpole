
import os
#from optimization import optimization
from optimization_trading_vix import optimization

if __name__ == '__main__':

    #now, we are going to learn the parameters
    train = optimization()
    train.run_optimization()