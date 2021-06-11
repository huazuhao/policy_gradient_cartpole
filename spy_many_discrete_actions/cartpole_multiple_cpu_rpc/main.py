from itertools import count
import torch.multiprocessing as mp
import agent_class
import config as C
import torch.distributed.rpc as rpc
import torch
import os
import numpy as np


def run_worker(rank,world_size):

    AGENT_NAME = "agent"
    OBSERVER_NAME = "observer{}"

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    if rank == 0:
        #rank 0 is the agent
        rpc.init_rpc(AGENT_NAME,rank = rank, world_size = world_size)

        agent = agent_class.agent_class(world_size)

        for epoch_count in range(0,C.NUM_EPOCHS):
            agent.run_episodes()
            total_rewards = agent.finish_episode()

            if epoch_count % C.LOG_INTERVAL == 0:
                pass

            if np.mean(total_rewards) > 200:
                    print('\nSolved!')
                    print("\nSaving the final neural network")

                    #save the neural network in the end
                    cwd = os.getcwd()
                    parameter_file = 'cartpole_nn_trained_model.pt'
                    cwd = os.path.join(cwd,parameter_file)
                    torch.save(agent.agent.state_dict(),cwd)

                    break

    else:
        #other ranks are observer
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
        # observers passively waiting for instructions from the agent


    # block until all rpcs finish, and shutdown the RPC instance
    rpc.shutdown()


if __name__ == "__main__":

    world_size = 2

    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)
