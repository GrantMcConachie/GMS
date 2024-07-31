"""
Script to test the trained models and generate a test set submission form.
"""

from src.GMS.models import gms

import os
import json
import pandas as pd

import torch


def load_test_submission():
    """
    Loads in the test set submission form
    """
    return pd.read_csv(".data/raw/Test_set_Submission_form.csv")


def load_config(model_fp):
    """
    Loads in config file in the directory
    """
    return json.load(
        open(os.path.join(model_fp, "config.json"), "rb")
    )


def test_model(test_set, config, model_fp):
    """
    Runs model inference on all the test data
    """
    preds = []
    test_set_data = ".data/processed/test"
    cos_sim = torch.nn.CosineSimilarity()
    model_fp = os.path.join(model_fp, "model.pt")

    # Load model
    model = gms.GMS(config)
    model.load_state_dict(torch.load(model_fp, map_location="cpu"))
    model.eval()

    # Loop through test set
    for index, value in test_set.iterrows():
        # load data
        data_pt = f"test_{value.iloc[1]}_{value.iloc[2]}.pt"
        data_path = os.path.join(test_set_data, data_pt)
        mixture = torch.load(data_path)

        # get model output
        out = model(mixture)
        sim = cos_sim(out[None, 0], out[None, 1])
        preds.append(sim)

        # saving to dataframe
        test_set.at[index, "Preds"] = sim[0].detach().numpy()

    return test_set


def write_preds(test_set, model_fp):
    """
    saves a new csv that contains the model predictions
    """
    test_set.to_csv(os.path.join(model_fp, "preds.csv"))


def main(model_fp):
    """
    pipeline for model inference

    Args:
        model_fp (str) - filepath to the directory where the saved model is
    """
    # Load in test set submission form
    test_set = load_test_submission()

    # load config
    config = load_config(model_fp)

    # model inference
    test_set = test_model(test_set, config, model_fp)

    # Save submission form
    write_preds(test_set, model_fp)


if __name__ == '__main__':
    main(".experiments/pt1_drop")
