import subprocess
import os
import copy
import argparse

# Create outputs directory if it doesn't exist
os.makedirs("outputs/ISP_experiments/grad_compression", exist_ok=True)

parser = argparse.ArgumentParser(description="Run federated learning experiments.")
parser.add_argument(
    "--device_id",
    type=int,
    default=0,
    help="GPU device IDx to use (default: 0)",
)


args = parser.parse_args()

# Configuration parameters
BASE_PARAMS = [
    "dataset.alpha=1000",
    "federated_params.amount_of_clients=100",
    "training_params.batch_size=64",
    f"training_params.device_ids=[{args.device_id}]",
    "federated_params.print_client_metrics=False",
    "federated_params.communication_rounds=1500",
    "random_state=41",
]


def build_command(compression_metod, dynamic_clients):
    params = copy.deepcopy(BASE_PARAMS)
    compression_k_percent = 15
    if compression_metod == "topk":
        compression_k_percent = 5

    if dynamic_clients:
        params.append("federated_method=ISP")
        params.append(f"federated_method.base_method=compression")
        params.extend(
            [
                "+federated_method.args.num_clients_subset=20",
                f"+federated_method.args.compression_type={compression_metod}",
                f"+federated_method.args.compression_k_percent={compression_k_percent}",
                "federated_method.warmup_rounds=5",
                "federated_method.local_epoch_on_wp=1",
            ]
        )
    else:
        params.extend(
            [
                f"federated_method=compression",
                f"federated_method.compression_type={compression_metod}",
                f"federated_method.compression_k_percent={compression_k_percent}",
            ]
        )

    # Build output filename
    output_name = f"outputs/ISP_experiments/grad_compression/compression_{compression_metod}_cifar10.txt"
    if dynamic_clients:
        output_name = "isp_" + output_name

    return params, f"{output_name}"


for dynamic_clients in [True, False]:
    for compression_metod in ["topk", "randk"]:
        # Run experiments
        # Build command and output path
        params, output_path = build_command(compression_metod, dynamic_clients)

        # Create full command
        cmd = ["nohup", "python", "src/train.py"] + params

        # Convert to string with output redirection
        cmd_str = " ".join(cmd) + f" > {output_path}"

        print(
            f"Running compression strategy. Dynamic clients {args.dynamic_clients} ",
            flush=True,
        )
        subprocess.run(cmd_str, shell=True, check=True)
