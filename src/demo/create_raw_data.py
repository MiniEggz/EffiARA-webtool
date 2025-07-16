import argparse

import yaml
from effiara.data_generator import generate_samples

if __name__ == "__main__":
    # set up arguments
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    raw_config = config["raw_data"]

    df = generate_samples(
        raw_config["num_samples"], raw_config["num_classes"], config["seed"]
    )

    # save to file location
    df.to_csv(raw_config["save_path"], index=False)
