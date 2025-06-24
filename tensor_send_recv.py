import os
import time
import argparse
import torch
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure GPU round trip latency throughput and bandwidth for a fp16 tensor of shape batch_size×hidden_dim"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of rows in the tensor"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=8192,
        help="Number of columns in the tensor"
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=10,
        help="Number of warm up round trips"
    )
    parser.add_argument(
        "--timing_iters",
        type=int,
        default=10,
        help="Number of timing iterations for averaging"
    )
    # verify data in warmup
    parser.add_argument(
        "--verify_data",
        action="store_true",
        help="Verify data in warmup"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    shape = (args.batch_size, args.hidden_dim)
    total_elems = args.batch_size * args.hidden_dim
    elem_size_bytes = torch.tensor([], dtype=torch.float16).element_size()
    tensor_size_bytes = total_elems * elem_size_bytes

    send_buffer = torch.arange(
        start=0,
        end=total_elems,
        dtype=torch.float16,
        device=device
    ).reshape(shape)
    recv_buffer = torch.empty(
        size=shape,
        dtype=torch.float16,
        device=device
    )

    torch.cuda.synchronize()

    for _ in range(args.warmup_iters):
        if rank == 0:
            dist.send(tensor=send_buffer, dst=1)
            dist.recv(tensor=recv_buffer, src=1)
            if args.verify_data:
                # Verify received data matches expected (original * 2.0)
                expected = send_buffer * 2.0
                if not torch.allclose(recv_buffer, expected, rtol=1e-5, atol=1e-5):
                    print(f"Rank {rank}: Data verification failed in warmup!")
                    print(f"Expected: {expected[:5]}")  # Show first 5 elements
                    print(f"Received: {recv_buffer[:5]}")
                else:
                    print(f"Rank {rank}: Data verification passed in warmup")
        else:
            dist.recv(tensor=recv_buffer, src=0)
            result = recv_buffer * 2.0
            dist.send(tensor=result, dst=0)

        torch.cuda.synchronize()

    torch.cuda.synchronize()
    
    # Timing measurements
    latencies = []
    
    for _ in range(args.timing_iters):
        start_time = time.time()
        
        if rank == 0:
            dist.send(tensor=send_buffer, dst=1)
            dist.recv(tensor=recv_buffer, src=1)
            torch.cuda.synchronize()
            
            elapsed_ms = (time.time() - start_time) * 1000
            latencies.append(elapsed_ms)
        else:
            dist.recv(tensor=recv_buffer, src=0)
            result = recv_buffer * 2.0
            dist.send(tensor=result, dst=0)

    if rank == 0:
        # Calculate average metrics
        avg_latency_ms = sum(latencies) / len(latencies)
        avg_latency_s = avg_latency_ms / 1000.0
        round_trips_per_s = 1.0 / avg_latency_s
        elems_per_s = total_elems * 2 / avg_latency_s
        total_bytes = tensor_size_bytes * 2
        bandwidth_gb_s = total_bytes / avg_latency_s / (1024 ** 3)

        print(f"tensor shape {shape}")
        print(f"tensor size {tensor_size_bytes / (1024 * 1024):.2f} MB")
        # print(f"timing iterations {args.timing_iters}")
        print(f"average round trip latency {avg_latency_ms:.3f} ms")
        # print(f"round trips per second {round_trips_per_s:.2f}")
        # print(f"elements per second {elems_per_s:.2e}")
        print(f"bandwidth {bandwidth_gb_s:.2f} GB/s")
        print(f"all latencies (ms): {latencies}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
