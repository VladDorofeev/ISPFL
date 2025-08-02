# Run commands for Adaptive Number of Participants

⚠️ Important: Follow the Prerequisites steps to set up enviroment.

⚠ Script argument: all scripts have a `--device_id`, which defines the GPU idx and `--dynamic_clients` that runs ISP technique.

## CIFAR-10 Experiments

```bash
python scripts/cifar10_script.py > cifar10_log_script.txt &
```

`--scheduler_clients` runs AdaFL strategy.

## Gradient Compression Experiments

```bash
python scripts/grad_compression_script.py > grad_compression_log_script.txt &
```

## ImageNet Experiments

```bash
python scripts/imagenet_script.py > imagenet_log_script.txt &
```

# Ablation

## AdaFL Experiments

```bash
python scripts/cifar10_adafl_isp.py > cifar10_adafl_isp_log_script.txt &
```

## Ema Usage Experiments

```bash
python scripts/cifar10_ema_usage.py > cifar10_ema_usage_log_script.txt &
```

## Corrupted Trust Experiments

```bash
python scripts/cifar10_corrupted_trust.py > cifar10_corrupted_trust_log_script.txt &
```

## Momentum Experiments

```bash
python scripts/cifar10_momentum.py > cifar10_momentum_log_script.txt &
```

## Not Full Communication Experiments

```bash
python scripts/cifar10_not_full_comm.py > cifar10_not_full_comm_log_script.txt &
```

## Num Sample Amount Experiments

```bash
python scripts/cifar10_sample_amount.py > cifar10_sample_amount_log_script.txt &
```
## Delta Experiments

```bash
python scripts/cifar10_step_find.py > cifar10_step_find_log_script.txt &
```