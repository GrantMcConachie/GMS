"""
Script to train the GMS model
"""

from src.GMS.models import gms
from src.GMS.datasets import data_handling

import os
import json
import pickle as pkl
from tqdm import tqdm

import torch
from torch_geometric.loader import DataLoader


def loss_fn(emb_1, emb_2, target, criterion):
    """
    Loss function that minimizes the mean squared error between the cosine
    similarity of two embeddings and a target value
    """
    cos_sim = torch.nn.CosineSimilarity()
    pred = cos_sim(emb_1, emb_2)
    return criterion(pred, target)


def train(config):
    """
    Runs the training loop for the model
    """
    # device init
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("device:", device)

    # model init
    model = gms.GMS(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    criterion = torch.nn.MSELoss()

    # dataset init
    data_dir = ".data/processed/train/"
    dataset = data_handling.GraphDataset(
        data_dir=data_dir
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    # training loop
    loss_list = []
    running_loss = []
    model.train()
    for epoch in range(config["training"]["num_epochs"]):
        for data in tqdm(dataloader, desc="training"):
            optimizer.zero_grad()
            out = model(data.to(device))
            idx = torch.Tensor(
                [1 if i % 2 == 1 else 0 for i in range(out.shape[0])]
            ).long()
            loss = loss_fn(out[idx == 0], out[idx == 1], data.y, criterion)
            loss.backward()
            optimizer.step()

            # keeping track of loss
            running_loss.append(loss)
            loss_list.append(loss)

        print(f"epoch {epoch} loss: {torch.mean(torch.Tensor(running_loss))}")
        running_loss = []

    return model, loss_list


if __name__ == '__main__':
    # Load in config file
    config = json.load(open("src/GMS/models/config.json", 'rb'))
    trained_model, loss_list = train(config)

    # save trained model
    save_path = ".experiments/long"
    torch.save(
        trained_model.state_dict(),
        open(os.path.join(save_path, "model.pt"), 'wb')
    )
    json.dump(
        config,
        open(os.path.join(save_path, "config.json"), 'w', encoding='utf8')
    )
    pkl.dump(
        loss_list,
        open(os.path.join(save_path, "loss.pkl"), 'wb')
    )
