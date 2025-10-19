import torch, time, math, argparse

# Heavy batched matmul test to saturate GPU

def parse_args():
    parser = argparse.ArgumentParser(description='Run heavy GPU saturation test')
    parser.add_argument('--iterations', type=int, default=40,
                       help='Number of iterations to run')
    parser.add_argument('--compile', choices=['auto', 'force', 'disable'], default='auto',
                       help='Compilation mode: auto=try if Triton available, force=require compilation, disable=never compile')
    parser.add_argument('--target-util', type=float, default=0.8,
                       help='Target VRAM utilization (0.0-1.0)')
    return parser.parse_args()

def pick_sizes(total_memory_mb, target_fraction=0.6, dtype=torch.float16):
    bytes_per_elem = 2 if dtype==torch.float16 else 4
    allowed_bytes = total_memory_mb * 1024**2 * target_fraction
    allowed_for_matrices = allowed_bytes / 3.0
    B = 8
    N = int(math.sqrt(max(1, allowed_for_matrices / (B * bytes_per_elem))))
    # Cap to a safer maximum to avoid kernel/operator limitations on some devices
    N = max(256, min(N, 4096))
    return B, N


def heavy_batched_matmul_test(device='cuda', dtype=torch.float16, iterations=40, compile_mode='auto', target_util=0.8):
    assert torch.cuda.is_available(), "No CUDA"
    props = torch.cuda.get_device_properties(0)
    total_mb = props.total_memory / 1024**2
    B, N = pick_sizes(total_mb, target_fraction=target_util, dtype=dtype)
    print(f"Using batch={B}, N={N}, dtype={dtype}, VRAM={total_mb:.0f}MB")

    A = torch.randn(B, N, N, device=device, dtype=dtype)
    Bm = torch.randn(B, N, N, device=device, dtype=dtype)

    def step(a, b):
        return torch.bmm(a, b)

    # Handle compilation based on mode
    compiled = False
    if compile_mode != 'disable':
        try:
            import triton  # type: ignore
            try:
                step_compiled = torch.compile(step)
                compiled = True
                print("Successfully compiled with Triton acceleration")
            except Exception as e:
                if compile_mode == 'force':
                    raise RuntimeError("Compilation required but failed") from e
                print("Compilation attempted but failed, falling back to uncompiled")
                compiled = False
        except ImportError:
            if compile_mode == 'force':
                raise RuntimeError("Compilation required but Triton not available")
            if compile_mode == 'auto':
                print("Triton not available, running uncompiled")

    start = time.time()
    for i in range(iterations):
        # perform several matmuls per iteration to increase kernel time
        try:
            if compiled:
                out = step_compiled(A, Bm)
            else:
                out = step(A, Bm)
        except Exception as e:
            # If compiled path fails at runtime, fallback to uncompiled step
            print("Compiled step failed, falling back to uncompiled. Error:", e)
            out = step(A, Bm)
        out = torch.relu(out)
        if (i+1) % 5 == 0:
            torch.cuda.synchronize()
            print(f"Completed {i+1}/{iterations} iterations")
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print("Elapsed:", elapsed, "s")
    del A, Bm, out
    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args()
    heavy_batched_matmul_test(
        dtype=torch.float16,
        iterations=args.iterations,
        compile_mode=args.compile,
        target_util=args.target_util
    )
