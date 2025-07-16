import argparse
import yaml
import pandas as pd

from effiara.preparation import SampleDistributor

if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # load data
    df = pd.read_csv(config["distribution"]["raw_data_path"])


    # create sample distributor
    sample_distributor = SampleDistributor(
        **config["distribution"]["distributor"]
    )

    # distribute samples
    sample_distributor.distribute_samples(df, config["distribution"]["save_path"])
