# mpiexec --allow-run-as-root -n 2 python broadcast.py
import torch
import torch.distributed as dist
def main(rank, world):
    if rank == 0:
        x = torch.tensor([1.]) # Tensor of interest
        print("current tensor rank: {} and value: {}".format(rank, x))
    else:
        x = torch.tensor([5.,]) # A holder for recieving the tensor
        print("current tensor rank: {} and value: {}".format(rank, x))

    #dist.all_reduce(x, op=dist.ReduceOp.SUM)
    #dist.all_gather(tensor_list, x)
    dist.broadcast(x, 0)
    print('Rank {} has {}'.format(rank, x))

if __name__ == '__main__':
    dist.init_process_group(backend='mpi')
    main(dist.get_rank(), dist.get_world_size())
