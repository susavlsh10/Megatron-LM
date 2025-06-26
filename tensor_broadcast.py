import os
import time
import argparse
import torch
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile broadcast round trip with tensor parallel size"
    )
    parser.add_argument('--batch_size',    type=int,   default=128)
    parser.add_argument('--hidden_size',   type=int,   default=8192)
    parser.add_argument('--dtype',         type=str,   choices=['float16', 'float32', 'float64'], default='float16')
    parser.add_argument('--warmup_iters',  type=int,   default=10)
    parser.add_argument('--timing_iters',  type=int,   default=100)
    parser.add_argument('--tp_size',       type=int,   required=True, choices=[1, 2, 4, 8],
                        help="Tensor parallel size for configuring groups in a 16 GPU world")
    return parser.parse_args()


def main():
    args       = parse_args()
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    src = 0
    if rank == src:
        print("Running with args:", args)
        print(f"World size: {world_size}")
        print(f"TP size: {args.tp_size}")

    if world_size != 16:
        raise NotImplementedError(f'Only world_size 16 supported, got {world_size}')

    # configure send/receive ranks based on tp_size
    if args.tp_size == 1:
        forward_recv_ranks = [8]
        return_src         = 8
        return_recv_ranks  = [0]
    elif args.tp_size == 2:
        forward_recv_ranks = [8, 9]
        return_src         = 8
        return_recv_ranks  = [0, 1]
    elif args.tp_size == 4:
        forward_recv_ranks = list(range(8, 12))
        return_src         = 8
        return_recv_ranks  = list(range(0, 4))
    else:  # tp_size == 8
        forward_recv_ranks = list(range(8, 16))
        return_src         = 9
        return_recv_ranks  = list(range(0, 8))

    group1 = [src] + forward_recv_ranks
    group2 = [return_src] + return_recv_ranks
    pg1    = dist.new_group(ranks=group1)
    pg2    = dist.new_group(ranks=group2)

    dtype       = getattr(torch, args.dtype)
    shape       = (args.batch_size, args.hidden_size)
    elem_bytes  = torch.tensor([], dtype=dtype).element_size()
    tensor_bytes = args.batch_size * args.hidden_size * elem_bytes

    # allocate tensors
    forward_tensor = None
    return_tensor  = None
    if rank == src:
        forward_tensor = torch.randn(*shape, dtype=dtype, device='cuda')
    elif rank in forward_recv_ranks:
        forward_tensor = torch.empty(*shape, dtype=dtype, device='cuda')

    if rank == return_src:
        return_tensor = torch.empty(*shape, dtype=dtype, device='cuda')
    elif rank in return_recv_ranks:
        return_tensor = torch.empty(*shape, dtype=dtype, device='cuda')

    forward_ok = False
    return_ok  = False

    # warmup with correctness check on first iteration
    for i in range(args.warmup_iters):
        if rank in group1:
            dist.broadcast(forward_tensor, src=src, group=pg1)
        if rank in group2:
            dist.broadcast(return_tensor,  src=return_src, group=pg2)

        if i == 0:
            # forward correctness
            if rank in group1:
                # compute sum in float32 to avoid dtype mismatches
                root_sum = torch.tensor(0.0, dtype=torch.float32, device='cuda')
                if rank == src:
                    root_sum = forward_tensor.sum().float()
                dist.broadcast(root_sum, src=src, group=pg1)
                local_sum = forward_tensor.sum().float()

                ok_local = torch.tensor(
                    float(torch.allclose(local_sum, root_sum, atol=1e-2)),
                    dtype=torch.float32, device='cuda'
                )
                dist.all_reduce(ok_local, op=dist.ReduceOp.MIN, group=pg1)
                if rank == src:
                    forward_ok = bool(ok_local.item())

            # return correctness
            if rank in group2:
                root_sum2 = torch.tensor(0.0, dtype=torch.float32, device='cuda')
                if rank == return_src:
                    root_sum2 = return_tensor.sum().float()
                dist.broadcast(root_sum2, src=return_src, group=pg2)
                local_sum2 = return_tensor.sum().float()

                ok_local2 = torch.tensor(
                    float(torch.allclose(local_sum2, root_sum2, atol=1e-2)),
                    dtype=torch.float32, device='cuda'
                )
                dist.all_reduce(ok_local2, op=dist.ReduceOp.MIN, group=pg2)
                if rank == src:
                    return_ok = bool(ok_local2.item())

    # timing loop
    times = []
    for _ in range(args.timing_iters):
        if rank == src:
            torch.cuda.synchronize()
            t0 = time.time()

        if rank in group1:
            dist.broadcast(forward_tensor, src=src, group=pg1)
        if rank in group2:
            dist.broadcast(return_tensor,  src=return_src, group=pg2)

        if rank == src:
            torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1 - t0)

    if rank == src:
        avg_lat     = sum(times) / len(times)
        total_bytes = tensor_bytes * (len(forward_recv_ranks) + len(return_recv_ranks))
        throughput  = total_bytes / avg_lat / 1e9

        print(f'Forward correctness: {forward_ok}')
        print(f'Return correctness: {return_ok}')
        print(f'Round trip latency: {avg_lat * 1e3:.3f} ms')
        print(f'Throughput: {throughput:.2f} GB/s')

if __name__ == '__main__':
    main()
