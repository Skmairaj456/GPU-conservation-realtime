import argparse
import subprocess
import time
import csv


def collect(duration: int, out: str, interval: float = 1.0):
    cmd = [
        'nvidia-smi',
        '--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.graphics',
        '--format=csv'
    ]
    end = time.time() + duration
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp','index','name','util_gpu','util_mem','mem_used','mem_total','power_draw','gfx_clock'])
        while time.time() < end:
            try:
                p = subprocess.run(cmd, capture_output=True, text=True)
                lines = p.stdout.strip().splitlines()
                # skip header
                for line in lines[1:]:
                    parts = [x.strip() for x in line.split(',')]
                    writer.writerow(parts)
                f.flush()  # Ensure data is written immediately
            except Exception as e:
                print('nvidia-smi failed:', e)
            except Exception as e:
                print('nvidia-smi failed:', e)
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=30,
                      help='Duration in seconds to collect metrics')
    parser.add_argument('--output', type=str, default='perf_log.csv',
                      help='Output CSV file path')
    parser.add_argument('--interval', type=float, default=1.0,
                      help='Sampling interval in seconds')
    args = parser.parse_args()
    collect(args.duration, args.output, args.interval)
