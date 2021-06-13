import trainer as trainer_class
import utils
import BatchUpdateParameterServer as bups
import config as C

import torch.distributed.rpc as rpc
import torch
import os
import torch.multiprocessing as mp

def run_trainer(ps_rref):
    trainer_instance = trainer_class.trainer(ps_rref)
    trainer_instance.train()


def run_ps(trainers):
    utils.timed_log("Start training")
    ps_rref = rpc.RRef(bups.BatchUpdateParameterServer())
    futs = []
    for trainer in trainers:
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,))
        )

    torch.futures.wait_all(futs)
    utils.timed_log("Finish training")


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=30,
        rpc_timeout=0  # infinite timeout
     )
    if rank != 0:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        # trainer passively waiting for ps to kick off training iterations
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_ps([f"trainer{r}" for r in range(1, world_size)])

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":

    assert C.UPDATE_SIZE%C.BATCH_SIZE_PER_THREAD == 0

    world_size = int(C.UPDATE_SIZE/C.BATCH_SIZE_PER_THREAD + 1)
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)