"""Decide whether the new model should be released: only if it beats the last released one."""
import argparse
import json
import os


def should_promote(new_metrics, old_metrics):
    if old_metrics is None:
        return True, "No previous release found, so this becomes the first release."
    if new_metrics["r2"] > old_metrics["r2"]:
        return True, f"New R² {new_metrics['r2']:.4f} beats previous {old_metrics['r2']:.4f}."
    return False, (f"New R² {new_metrics['r2']:.4f} does not beat previous "
                   f"{old_metrics['r2']:.4f}, so no new release.")


def load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True, help="Metrics file of the model just trained")
    parser.add_argument("--old", default=None, help="Metrics file of the last released model")
    args = parser.parse_args()

    promote, reason = should_promote(load_json(args.new), load_json(args.old))
    print(reason)

    # Hand the decision to later workflow steps
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"promote={'true' if promote else 'false'}\n")
