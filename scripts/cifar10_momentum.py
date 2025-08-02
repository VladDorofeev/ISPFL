import subprocess
import os
import copy
import argparse

# Create outputs directory if it doesn't exist
os.makedirs("outputs/ISP_experiments/cifar10_momentum/", exist_ok=True)

parser = argparse.ArgumentParser(description="Run federated learning experiments.")
parser.add_argument(
    "--device_id",
    type=int,
    default=0,
    help="GPU device IDx to use (default: 0)",
)

args = parser.parse_args()

# Configuration parameters
FEDERATED_METHODS = ["fedavg"]

BASE_PARAMS = [
    "dataset.alpha=0.1",
    "federated_params.amount_of_clients=100",
    "training_params.batch_size=64",
    f"training_params.device_ids=[{args.device_id}]",
    "federated_params.print_client_metrics=False",
    "random_state=41",
    "manager.batch_size=25",
    "federated_params.communication_rounds=2000",
]


def build_command(federated_method, num_clients_momentum, dynamic_clients=False):
    params = copy.deepcopy(BASE_PARAMS)

    if dynamic_clients:
        params.append("federated_method=ISP")

        # For python class inherits compability only
        params.append("+federated_method.args.num_clients_subset=20")

        params.append(f"federated_method.base_method={federated_method}")
        params.append(f"federated_method.num_clients_momentum={num_clients_momentum}")
        output_name = (
            f"ISP_{federated_method}_num_clients_momentum_{num_clients_momentum}.txt"
        )
    else:
        params.append(f"federated_method={federated_method}")
        output_name = f"{federated_method}.txt"

    return params, f"outputs/ISP_experiments/cifar10_momentum/{output_name}"


# Run experiments
for method in FEDERATED_METHODS:
    for num_clients_momentum in [0.5, 1]:
        for dynamic_clients in [True]:
            # Build command and output path
            params, output_path = build_command(
                method, num_clients_momentum, dynamic_clients
            )

            # Create full command
            cmd = ["nohup", "python", "src/train.py"] + params

            # Convert to string with output redirection
            cmd_str = " ".join(cmd) + f" > {output_path}"

            print(
                f"Running {method} strategy. Momentum: {num_clients_momentum}.",
                flush=True,
            )
            print(f"Command is:\n{cmd_str}\n\n", flush=True)
            subprocess.run(cmd_str, shell=True, check=True)
