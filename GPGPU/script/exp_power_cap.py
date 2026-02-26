import os
import subprocess
import time
import signal
import argparse
import csv
import re

num_gpu = 4

# Define paths and executables
home_dir = os.path.expanduser('~')
script_dir = os.path.dirname(os.path.abspath(__file__))
python_executable = os.path.join(home_dir, "env/ml/bin/python3")

# scripts for CPU, GPU power monitoring
read_cpu_power = os.path.join(script_dir, "power_util/read_cpu_power.py")
read_gpu_power = os.path.join(script_dir, "power_util/read_gpu_power.py")
read_gpu_metrics = os.path.join(script_dir, "power_util/read_gpu_metrics.py")
read_cpu_ips = os.path.join(script_dir, "power_util/read_cpu_ips.py")
read_mem = os.path.join(script_dir, "power_util/read_mem.py")
read_cpu_metrics = os.path.join(script_dir, "power_util/read_cpu_metrics.py")

# scritps for running various benchmarks
run_app = os.path.join(script_dir, "run_benchmark/run_app.py")

ecp_benchmarks = ['XSBench','miniGAN','CRADL','sw4lite','Laghos','bert_large','UNet','Resnet50','lammps','gromacs',"NAMD"]
hec_benchmarks = ["addBiasResidualLayerNorm", "aobench", "background-subtract", "chacha20", "convolution3D", "dropout", "extrema", "fft", "kalman", "knn", "softmax", "stencil3d", "zmddft", "zoom"]
altis_benchmarks_0 = ["maxflops"]
altis_benchmarks_1 = ['bfs','gemm','gups','pathfinder','sort']
altis_benchmarks_2 = ['cfd','cfd_double','fdtd2d','kmeans','lavamd',
                      'nw','particlefilter_float','particlefilter_naive','raytracing',
                      'srad','where']
ecp_benchmarks = ['XSBench','miniGAN','CRADL','sw4lite','Laghos','bert_large','UNet','Resnet50','lammps','gromacs',"NAMD"]


spec_benchmarks = ['lbm', 'cloverleaf', 'tealeaf', 'minisweep', 'pot3d', 'miniweather', 'hpgmg']

# ml_models = ["resnet50", "vgg16"]
ml_models = ["resnet50"]
cpu_caps = [700]
# GPU_ct = [1,2,3,4]
GPU_ct = [4]
gpu_caps = [400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800]
gpu_caps= [800]

ML_MIN_PER_GPU_CAP = 200
ML_MAX_PER_GPU_CAP = 700
ML_BATCH_SIZE = 2048
ML_EPOCHS = 3
ML_LR = 0.001


def _select_ml_python():
    return python_executable


def _extract_avg_train_throughput(stdout_text):
    # Preferred metric from dl.py summary
    m = re.search(
        r"Average train throughput \(excluding epoch 1\):\s*([0-9]+(?:\.[0-9]+)?)\s*images/sec",
        stdout_text,
    )
    if m:
        return float(m.group(1))

    # Fallback: use last epoch throughput line if summary is unavailable.
    fallback = re.findall(
        r"Epoch\s+\d+\s+Complete\s*-\s*Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*images/sec",
        stdout_text,
    )
    if fallback:
        return float(fallback[-1])
    return None



def run_ml_experiment(model_name=None):
    cpu_cap = 700
    ml_dir = os.path.join(home_dir, "overprovision", "ML")
    ml_script = os.path.join(ml_dir, "dl.py")
    ml_python = _select_ml_python()

    print("[ML] Using python: {}".format(ml_python))

    if model_name:
        if model_name not in ml_models:
            raise ValueError(f"Unknown ML model '{model_name}'. Available: {ml_models}")
        models = [model_name]
    else:
        models = ml_models

    output_root_dir = os.path.abspath(
        os.path.join(script_dir, "..", "data", "H100", "ml_power_motif")
    )
    os.makedirs(output_root_dir, exist_ok=True)
    throughput_csv_by_model = {}
    for model in models:
        model_output_dir = os.path.join(output_root_dir, model)
        os.makedirs(model_output_dir, exist_ok=True)
        output_csv = os.path.join(model_output_dir, "throughput.csv")
        throughput_csv_by_model[model] = output_csv
        if os.path.exists(output_csv):
            os.remove(output_csv)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "total_gpu_cap",
                "gpu_count",
                "per_gpu_cap",
                "model",
                "throughput_images_per_sec",
                "status",
            ])

    for model in models:
        for g_cnt in GPU_ct:
            for total_gpu_cap in gpu_caps:
                per_gpu_cap = total_gpu_cap / g_cnt
                if per_gpu_cap < ML_MIN_PER_GPU_CAP or per_gpu_cap > ML_MAX_PER_GPU_CAP:
                    # Skip invalid combinations that violate H100 per-GPU cap range.
                    continue

                per_gpu_cap_int = int(per_gpu_cap)
                subprocess.run(
                    [os.path.join(script_dir, "power_util/cap.sh"), str(cpu_cap), str(per_gpu_cap_int)],
                    check=True,
                )
                time.sleep(0.2)

                cmd = [
                    ml_python,
                    ml_script,
                    "--model", model,
                    "--num-gpus", str(g_cnt),
                    "--batch-size", str(ML_BATCH_SIZE),
                    "--epochs", str(ML_EPOCHS),
                    "--lr", str(ML_LR),
                ]

                print(
                    f"[ML] Running model={model} total_cap={total_gpu_cap} "
                    f"gpus={g_cnt} per_gpu_cap={per_gpu_cap_int}"
                )
                process = subprocess.Popen(
                    cmd,
                    cwd=ml_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                )
                model_output_dir = os.path.join(output_root_dir, model)
                output_gpu_metrics = os.path.join(
                    model_output_dir,
                    f"{total_gpu_cap}_{g_cnt}_gpu_metrics.csv",
                )

                monitor_command_gpu_metrics = f"{python_executable} {read_gpu_metrics}  --output_csv {output_gpu_metrics} --pid {process.pid} --num_gpu {num_gpu}"
                monitor_process0 = subprocess.Popen(monitor_command_gpu_metrics, shell=True, stdin=subprocess.PIPE,universal_newlines=True)
                
                output_lines = []
                for line in iter(process.stdout.readline, ""):
                    print(line, end="")
                    output_lines.append(line)
                process.stdout.close()
                return_code = process.wait()
                monitor_stdout = ""
                monitor_stderr = ""
                try:
                    if monitor_process0 is not None:
                        monitor_process0.terminate()
                        monitor_stdout, monitor_stderr = monitor_process0.communicate(timeout=3)
                except Exception:
                    pass

                run_output = "".join(output_lines)
                throughput = _extract_avg_train_throughput(run_output)
                status = "ok" if (return_code == 0 and throughput is not None) else "failed"

                with open(throughput_csv_by_model[model], "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        total_gpu_cap,
                        g_cnt,
                        per_gpu_cap_int,
                        model,
                        f"{throughput:.2f}" if throughput is not None else "",
                        status,
                    ])

                if status != "ok":
                    print(
                        f"[ML][WARN] model={model} total_cap={total_gpu_cap} gpus={g_cnt} "
                        f"per_gpu_cap={per_gpu_cap_int} failed. returncode={return_code}"
                    )
                    if run_output:
                        print(run_output[-1000:])
                if (not os.path.exists(output_gpu_metrics)) or os.path.getsize(output_gpu_metrics) == 0:
                    print(
                        f"[ML][WARN] missing/empty GPU metrics file for model={model} "
                        f"cap={total_gpu_cap} gpus={g_cnt}: {output_gpu_metrics}"
                    )
                    if monitor_stderr:
                        print("[ML][MONITOR STDERR]")
                        print(monitor_stderr[-1000:])

    subprocess.run([os.path.join(script_dir, "power_util/cap.sh"), str(700), str(700)], check=True)
    print("[ML] Throughput results saved to:")
    for model in models:
        print(f"  {throughput_csv_by_model[model]}")





def run_benchmark(benchmark_script_dir,benchmark, suite, test, size,cap_type):

    def cap_exp(g_cnt, cpu_cap, gpu_cap, output_cpu_power, output_gpu_power,output_ips, output_gpu_metrics,output_mem, output_cpu_metrics, output_runtime):

        # gpu_cap = min(gpu_cap / g_cnt, 700)

        subprocess.run([os.path.join(script_dir, "power_util/cap.sh"), str(cpu_cap), str(gpu_cap)], check=True)
        time.sleep(0.2)  # Wait for the power caps to take effect

        # Run the benchmark
        start = time.time()

        run_benchmark_command = f"bash {os.path.join(home_dir, benchmark_script_dir, f'{benchmark}.sh')} {g_cnt}"

        benchmark_process = subprocess.Popen(run_benchmark_command, shell=True)
        benchmark_pid = benchmark_process.pid


        # # monitor GPU metrics
        monitor_command_gpu_metrics = f"{python_executable} {read_gpu_metrics}  --output_csv {output_gpu_metrics} --pid {benchmark_pid} --num_gpu {num_gpu}"
        monitor_process4 = subprocess.Popen(monitor_command_gpu_metrics, shell=True, stdin=subprocess.PIPE,universal_newlines=True)

        benchmark_process.wait()  # Wait for the benchmark to complete

        end = time.time()
        elapsed_time = end - start

        # Write runtime to CSV (append mode)
        file_exists = os.path.exists(output_runtime)
        with open(output_runtime, 'a') as f:
            if not file_exists:
                f.write(f"power_cap,gpu_count,runtime_seconds\n")
            f.write(f"{gpu_cap},{g_cnt},{elapsed_time}\n")

        
################## end helper function ####################
    
    cpu_cap = 700

    # Create output directory and runtime file
    output_dir = f"../data/H100/{suite}_power_motif/{benchmark}"
    os.makedirs(output_dir, exist_ok=True)
    output_runtime = f"{output_dir}/runtime.csv"

    # Delete existing runtime file to start fresh
    if os.path.exists(output_runtime):
        os.remove(output_runtime)

    for g_cnt in GPU_ct:
        for gpu_cap in gpu_caps:
            output_cpu_power = f"{output_dir}/{gpu_cap}_cpu_power.csv"
            output_gpu_power = f"{output_dir}/{gpu_cap}_{g_cnt}_gpu_power.csv"
            output_ips = f"{output_dir}/{gpu_cap}_{g_cnt}_ips.csv"
            output_mem = f"{output_dir}/{gpu_cap}_{g_cnt}_mem.csv"
            output_gpu_metrics = f"{output_dir}/{gpu_cap}_{g_cnt}_gpu_metrics.csv"
            output_cpu_metrics = f"{output_dir}/{gpu_cap}_{g_cnt}_cpu_metrics.csv"
            cap_exp(g_cnt, cpu_cap, gpu_cap, output_cpu_power, output_gpu_power,output_ips,output_gpu_metrics,output_mem,output_cpu_metrics,output_runtime)



    subprocess.run([os.path.join(script_dir, "power_util/cap.sh"), str(700), str(700)], check=True)


if __name__ == "__main__":

   # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run benchmarks and monitor power consumption.')
    parser.add_argument('--benchmark', type=str, help='Optional name of the benchmark to run', default=None)
    parser.add_argument('--test', type=int, help='whether it is a test run', default=None)
    parser.add_argument('--suite', type=int, help='0 for ECP, 1 for ALTIS, 2 for ML', default=1)
    parser.add_argument('--benchmark_size', type=int, help='0 for big, 1 for small', default=0)
    parser.add_argument('--cap_type', type=int, help='0 for cpu, 1 for gpu, 2 for dual', default=2)
    parser.add_argument('--num_gpu', type=int, default=1)

    args = parser.parse_args()
    benchmark = args.benchmark
    test = args.test
    suite = args.suite
    benchmark_size = args.benchmark_size
    cap_type = args.cap_type
    # num_gpu = args.num_gpu


    if suite == 0 or suite ==5:
        benchmark_script_dir = f"power/GPGPU/script/run_benchmark/ecp_script"
        # single test
        if benchmark:
            run_benchmark(benchmark_script_dir, benchmark,"ecp",test,benchmark_size,cap_type)
        # run all ecp benchmarks
        else:
            for benchmark in ecp_benchmarks:
                run_benchmark(benchmark_script_dir, benchmark,"ecp",test,benchmark_size,cap_type)
    

    if suite == 1 or suite ==5:
        # Map of benchmarks to their paths
        benchmark_paths = {
            "level0": altis_benchmarks_0,
            "level1": altis_benchmarks_1,
            "level2": altis_benchmarks_2
        }
    
        if benchmark:
            # Find which level the input benchmark belongs to
            found = False
            for level, benchmarks in benchmark_paths.items():
                if benchmark in benchmarks:
                    benchmark_script_dir = f"power/GPGPU/script/run_benchmark/altis_script/{level}"
                    run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)
                    found = True
                    break
        else:
            for benchmark in altis_benchmarks_0:
                if benchmark_size==0:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level0"
                else:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level0"
                run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)
            
            
            for benchmark in altis_benchmarks_1:
                if benchmark_size==0:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level1"
                else:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level1"
                run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)
            
            
            for benchmark in altis_benchmarks_2:
                if benchmark_size==0:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level2"
                else:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level2"
                run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)


    if suite == 2 or suite == 5:
        run_ml_experiment(model_name=benchmark)


    if suite == 3 or suite == 5:
        benchmark_script_dir = f"power/GPGPU/script/run_benchmark/hec_script"
         # single test
        if benchmark:
            run_benchmark(benchmark_script_dir, benchmark,"hec",test,benchmark_size,cap_type)
        # run all ecp benchmarks
        else:
            for benchmark in hec_benchmarks:
                run_benchmark(benchmark_script_dir, benchmark,"hec",test,benchmark_size,cap_type)

    if suite == 4 or suite == 5:
        benchmark_script_dir = f"power/GPGPU/script/run_benchmark/spec_script"
        # single test
        if benchmark:
            run_benchmark(benchmark_script_dir, benchmark,"spec",test,benchmark_size,cap_type)
        # run all spec benchmarks
        else:
            for benchmark in spec_benchmarks:
                run_benchmark(benchmark_script_dir, benchmark,"spec",test,benchmark_size,cap_type)
