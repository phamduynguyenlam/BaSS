import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REWARD_PATTERN = re.compile(
    r"epoch\s+(\d+)\s+done\s+\|\s+mean reward/FE\s+=\s+([-+]?\d*\.?\d+)",
    re.IGNORECASE,
)
AGENT_PATTERN = re.compile(r"(?:^|\|)\s*agent\s*=\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)


def parse_training_log(log_path):
    epochs = []
    rewards = []
    agent_name = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if agent_name is None and "Training config" in line:
                agent_match = AGENT_PATTERN.search(line)
                if agent_match:
                    agent_name = agent_match.group(1).strip()

            reward_match = REWARD_PATTERN.search(line)
            if reward_match:
                epochs.append(int(reward_match.group(1)))
                rewards.append(float(reward_match.group(2)))

    return {
        "log_path": str(log_path),
        "label": Path(log_path).stem,
        "agent_name": agent_name,
        "epochs": epochs,
        "rewards": rewards,
    }


def limit_log_info_epochs(log_info, max_epochs):
    if max_epochs is None:
        return log_info

    limit = max(0, int(max_epochs))
    if limit == 0:
        return {
            **log_info,
            "epochs": [],
            "rewards": [],
        }

    return {
        **log_info,
        "epochs": list(log_info["epochs"][:limit]),
        "rewards": list(log_info["rewards"][:limit]),
    }


def moving_average(values, window):
    if len(values) < window:
        return np.asarray(values, dtype=np.float64)

    return np.convolve(
        values,
        np.ones(window, dtype=np.float64) / float(window),
        mode="valid",
    )


def build_series_label(log_info, suffix):
    agent_name = log_info.get("agent_name")
    stem = log_info.get("label", "log")
    if agent_name:
        return f"{agent_name} | {stem} | {suffix}"
    return f"{stem} | {suffix}"


def plot_single_log(log_info, window):
    epochs = log_info["epochs"]
    rewards = log_info["rewards"]

    print(f"Found {len(rewards)} reward entries in {log_info['log_path']}")

    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        rewards,
        label=build_series_label(log_info, "raw"),
        alpha=0.7,
        linewidth=1.8,
    )

    if len(rewards) >= window:
        smooth_rewards = moving_average(rewards, window)
        plt.plot(
            epochs[window - 1 :],
            smooth_rewards,
            linewidth=2,
            label=build_series_label(log_info, f"ma{window}"),
        )

    title = "Training Reward Curve"
    if log_info.get("agent_name"):
        title = f"{title} | agent={log_info['agent_name']}"
    plt.xlabel("Epoch")
    plt.ylabel("Mean reward/FE")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_compare_logs(log_info1, log_info2, window):
    print(f"Found {len(log_info1['rewards'])} reward entries in {log_info1['log_path']}")
    print(f"Found {len(log_info2['rewards'])} reward entries in {log_info2['log_path']}")

    plt.figure(figsize=(11, 5))

    plt.plot(
        log_info1["epochs"],
        log_info1["rewards"],
        label=build_series_label(log_info1, "raw"),
        alpha=0.6,
        linewidth=1.6,
    )
    plt.plot(
        log_info2["epochs"],
        log_info2["rewards"],
        label=build_series_label(log_info2, "raw"),
        alpha=0.6,
        linewidth=1.6,
    )

    if len(log_info1["rewards"]) >= window:
        smooth1 = moving_average(log_info1["rewards"], window)
        plt.plot(
            log_info1["epochs"][window - 1 :],
            smooth1,
            linewidth=2,
            label=build_series_label(log_info1, f"ma{window}"),
        )

    if len(log_info2["rewards"]) >= window:
        smooth2 = moving_average(log_info2["rewards"], window)
        plt.plot(
            log_info2["epochs"][window - 1 :],
            smooth2,
            linewidth=2,
            label=build_series_label(log_info2, f"ma{window}"),
        )

    agent1 = log_info1.get("agent_name") or "unknown"
    agent2 = log_info2.get("agent_name") or "unknown"
    plt.xlabel("Epoch")
    plt.ylabel("Mean reward/FE")
    plt.title(f"Training Reward Comparison | {agent1} vs {agent2}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Path to one training log file",
    )
    parser.add_argument(
        "--log_path1",
        type=str,
        default=None,
        help="Path to the first training log file for comparison",
    )
    parser.add_argument(
        "--log_path2",
        type=str,
        default=None,
        help="Path to the second training log file for comparison",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Moving average window",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Limit the number of epochs to plot from the start of the log",
    )

    args = parser.parse_args()

    using_single = bool(args.log_path)
    using_compare = bool(args.log_path1) or bool(args.log_path2)

    if using_single and using_compare:
        raise ValueError("Use either --log_path or (--log_path1 and --log_path2), not both.")
    if using_single:
        log_info = limit_log_info_epochs(parse_training_log(args.log_path), args.max_epochs)
        plot_single_log(log_info, int(args.window))
        return
    if bool(args.log_path1) and bool(args.log_path2):
        log_info1 = limit_log_info_epochs(parse_training_log(args.log_path1), args.max_epochs)
        log_info2 = limit_log_info_epochs(parse_training_log(args.log_path2), args.max_epochs)
        plot_compare_logs(log_info1, log_info2, int(args.window))
        return

    raise ValueError("Provide either --log_path or both --log_path1 and --log_path2.")


if __name__ == "__main__":
    main()
