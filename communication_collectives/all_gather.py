# mpiexec --allow-run-as-root -n 2 python all_gather.py
import torch
import torch.distributed as dist
def main(rank, world):
    tensor_list = [torch.zeros(1, ), torch.zeros(1,)]
    if rank == 0:
        x = torch.tensor([1.]) # Tensor of interest
    else:
        x = torch.tensor([1.,]) # A holder for recieving the tensor

    #dist.all_reduce(x, op=dist.ReduceOp.SUM)
    dist.all_gather(tensor_list, x)
    print('Rank {} has {}'.format(rank, tensor_list))

if __name__ == '__main__':
    dist.init_process_group(backend='mpi')
    main(dist.get_rank(), dist.get_world_size())
