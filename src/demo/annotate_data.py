import pandas as pd
import yaml
from effiara.data_generator import generate_annotator_label

if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # loop through each annotator
    annotator_dict = config["annotate"]["accuracy"]
    for annotator in annotator_dict.keys():
        print(f"{annotator} is annotating...")
        # load their samples
        df = pd.read_csv(f"{config['annotate']['data_dir']}/{annotator}.csv")
        # annotate all samples with their accuracy
        df[f"{annotator}_label"] = df["true_label"].apply(
            generate_annotator_label,
            correctness_probability=annotator_dict[annotator],
            num_classes=config["raw_data"]["num_classes"],
        )
        # save annotated data
        df.to_csv(f"{config['annotate']['save_dir']}/{annotator}.csv", index=False)
        print(f"{annotator} finished.")
