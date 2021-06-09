# Run Command:
# mpiexec --allow-run-as-root -n 2 python temp.py
import torch
import torch.distributed as dist
def main(rank, world):
    if rank == 0:
        x = torch.tensor([1.]) # Tensor of interest
    else:
        x = torch.tensor([2.]) # A holder for recieving the tensor

    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print('Rank {} has {}'.format(rank, x))

if __name__ == '__main__':
    dist.init_process_group(backend='mpi')
    main(dist.get_rank(), dist.get_world_size())
