"""
Preprocesses the data into graph format
"""


import os
import time
import requests
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm

from rdkit import Chem

import torch
from torch_geometric.data import Data


def get_inchi(raw_data, data_label):
    """
    Gets the inchis of all the mixtures from pubchem

    Args:
        raw_data (pandas.dataframe) - dataframes of the train test and 
          leaderboard sets
        data_label (str) - name of each of the dataset corresponding to
          `raw_data`

    Returns:
        inchi_data (dict) - inchis of all the datasets
    """
    # check if file alread exists
    temp_dir = "src/GMS/data_utils/tmp/"
    pkl_dir = os.path.join(temp_dir, f"{data_label}_inchi_data.pkl")
    if os.path.isfile(pkl_dir):
        return pkl.load(open(pkl_dir, 'rb'))

    # init
    api_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
    inchi_data = {}

    # loop through entries in each dataset
    for index, value in tqdm(raw_data.iterrows(), desc=f"getting inchi from {data_label}"):
        if data_label == "test":
            dataset_mix_label = "test" + "_" + str(value.iloc[0])
        else:
            dataset_mix_label = str(value.iloc[0]) + "_" + str(value.iloc[1])

        # loop through CIDs
        inchi_data[dataset_mix_label] = []
        for i, cid in enumerate(value[2:]):
            if cid == 0:
                continue

            pc_path = os.path.join(api_url, str(cid), "json")
            mol_info = requests.get(pc_path).json()
            lab = mol_info["PC_Compounds"][0]["props"][12]["urn"]["label"]
            inchi = mol_info["PC_Compounds"][0]["props"][12]["value"]["sval"]

            # make sure getting inchi
            try:
                assert lab == 'InChI', "Information requested not inchi, check pubchem API for changes."
            except AssertionError as msg:
                print(msg)

            # append dict
            inchi_data[dataset_mix_label].append(inchi)

            if not i % 5:
                time.sleep(1)  # can only do 5 requests per second

        # saving data
        pkl.dump(inchi_data, open(f'src/GMS/data_utils/tmp/{data_label}_inchi_data.pkl', 'wb'))

    return inchi_data


def build_graph(
    feature_vector_list,
    edges_list,
    senders_list,
    receivers_list,
):
    """
    Makes a pytorch graph out of a components of a graph
    """
    # concatenate everything
    node_features = np.concatenate(feature_vector_list)
    edge_features = np.concatenate(edges_list)
    senders = np.concatenate(senders_list)
    receivers = np.concatenate(receivers_list)
    edge_index = np.stack([senders, receivers])

    # make graph
    graph = Data(
        x=torch.Tensor(node_features),
        edge_index=torch.Tensor(edge_index),
        edge_attr=torch.Tensor(edge_features),
        n_node=torch.Tensor([len(node_features)]),
        n_edge=torch.tensor([len(edge_features)])
    )

    return graph


def convert_to_graphs(inchi_data, data_label, node_feat_len=125, add_self_edge=True):
    """
    Takes each mixture and makes it into a graph
    """
    # load in dicts
    one_hot_atom_dict = pkl.load(
        open("src/GMS/data_utils/graph_feats/one_hot_atom_dict_tot.pkl", 'rb')
    )
    one_hot_hybrid_dict = pkl.load(
        open("src/GMS/data_utils/graph_feats/one_hot_hybrid_dict_tot.pkl", 'rb')
    )
    one_hot_degree_dict = pkl.load(
        open("src/GMS/data_utils/graph_feats/one_hot_degree_dict_tot.pkl", 'rb')
    )
    one_hot_valence_dict = pkl.load(
        open("src/GMS/data_utils/graph_feats/one_hot_valence_dict_tot.pkl", 'rb')
    )
    one_hot_bond_dict = pkl.load(
        open("src/GMS/data_utils/graph_feats/one_hot_bond_dict_tot.pkl", 'rb')
    )

    # loop through data
    for key, value in inchi_data.items():
        feature_vector_list = []
        edges_list = []
        senders_list = []
        receivers_list = []

        for iter, inchi in enumerate(value):
            # Feature vector
            mol = Chem.MolFromInchi(inchi)
            mol = Chem.AddHs(mol)

            num_atoms = mol.GetNumAtoms()
            feat_matrix = np.zeros(shape=(num_atoms, node_feat_len))

            for i, atom in enumerate(mol.GetAtoms()):
                featurized_atom = one_hot_atom_dict[atom.GetSymbol()]
                featurized_atom = np.append(featurized_atom, atom.GetIsAromatic())
                featurized_atom = np.append(featurized_atom, one_hot_hybrid_dict[str(atom.GetHybridization())])
                featurized_atom = np.append(featurized_atom, one_hot_degree_dict[str(atom.GetDegree())])
                featurized_atom = np.append(featurized_atom, one_hot_valence_dict[str(atom.GetTotalValence())])
                feat_matrix[i, :] = featurized_atom

            # Adjacency matrix
            adj_matrix = np.zeros(shape=(num_atoms, num_atoms))
            for atom in mol.GetAtoms():
                i = atom.GetIdx()
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                for j in neighbors:
                    adj_matrix[i, j] = 1

            if add_self_edge:
                adj_matrix = adj_matrix + np.identity(mol.GetNumAtoms())

            # Adjacency Tensor
            adj_tensor = np.zeros(shape=(len(one_hot_bond_dict), num_atoms, num_atoms))
            for atom in mol.GetAtoms():
                i = atom.GetIdx()
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

                for j in neighbors:
                    bond_type = str(mol.GetBondBetweenAtoms(i, j).GetBondType())
                    adj_tensor[:, i, j] = one_hot_bond_dict[bond_type]

                # add self edge
                if add_self_edge:
                    adj_tensor[:, i, i] = one_hot_bond_dict["SELF"]

            # make edge attributes from adj matrix and tensor
            senders = np.where(adj_matrix == 1)[0]
            receivers = np.where(adj_matrix == 1)[1]
            edges = adj_tensor[:, receivers, senders].T

            # append graphs to lists
            feature_vector_list.append(feat_matrix)
            edges_list.append(edges)

            # adding num nodes to senders and receivers so multiple graphs can
            # live together in the 'same' graph
            if iter != 0:
                senders += max(senders_list[iter-1] + 1)
                receivers += max(senders_list[iter-1] + 1)

            senders_list.append(senders)
            receivers_list.append(receivers)

        # generate graph
        graph = build_graph(
            feature_vector_list,
            edges_list,
            senders_list,
            receivers_list
        )

        # save
        torch.save(graph, f"src/GMS/data_utils/single_graphs/{data_label}/{key}.pt")


def make_graph_mixtures(data_label):
    """
    Takes all the mixtures of graphs and puts them into a mixture of mixtures
    """
    if data_label == "train":
        mixture_dir = ".data/raw/TrainingData_mixturedist.csv"
        graph_dir = "src/GMS/data_utils/single_graphs/train"
    elif data_label == "test":
        mixture_dir = ".data/raw/Test_set_Submission_form.csv"
        graph_dir = "src/GMS/data_utils/single_graphs/test"

    mixtures = pd.read_csv(mixture_dir)

    # loop through data
    for idx, value in mixtures.iterrows():
        # find mixture pair
        if data_label == "train":
            dataset = mixtures.iloc[idx].iloc[0]
        elif data_label == "test":
            dataset = "test"

        mix_1_name = dataset + "_" + str(mixtures.iloc[idx].iloc[1])
        mix_2_name = dataset + "_" + str(mixtures.iloc[idx].iloc[2])

        # load associated graphs
        mix_1 = torch.load(os.path.join(graph_dir, f'{mix_1_name}.pt'))
        mix_2 = torch.load(os.path.join(graph_dir, f'{mix_2_name}.pt'))

        # load value
        y = mixtures.iloc[idx].iloc[-1]

        # make graph
        tot_x = torch.concatenate([mix_1.x, mix_2.x])
        tot_edge_index = torch.concatenate([mix_1.edge_index, mix_2.edge_index + mix_1.x.shape[0]], axis=1)
        tot_edge_attr = torch.concatenate([mix_1.edge_attr, mix_2.edge_attr])
        tot_n_node = torch.tensor([len(tot_x)])
        tot_n_edge = torch.tensor([len(tot_edge_attr)])
        mix_batch = torch.concatenate(
            [
                torch.zeros(size=(mix_1.x.shape[0],)),
                torch.ones(size=(mix_2.x.shape[0],))
            ]
        )

        mix_graph = Data(
            x=tot_x,
            edge_index=tot_edge_index.long(),
            edge_attr=tot_edge_attr,
            n_node=tot_n_node,
            n_edge=tot_n_edge,
            mix_batch=mix_batch.long(),
            y=torch.Tensor([y])
        )

        # save
        dat_name = f"{dataset}_{mixtures.iloc[idx].iloc[1]}_{mixtures.iloc[idx].iloc[2]}"
        torch.save(mix_graph, f".data/processed/{data_label}/{dat_name}.pt")


def main():
    """
    Main data preprocessing pipeline
    """
    # load in train, test, and leaderboard mixture definitions
    raw_train = pd.read_csv(".data/raw/Mixure_Definitions_Training_set.csv")
    raw_test = pd.read_csv(".data/raw/Mixure_Definitions_test_set.csv")

    # convert cid to inchi
    inchi_data_train = get_inchi(raw_train, "train")
    inchi_data_test = get_inchi(raw_test, "test")

    # making graphs
    convert_to_graphs(inchi_data_train, "train")
    convert_to_graphs(inchi_data_test, "test")

    # making mixtures
    make_graph_mixtures("train")
    make_graph_mixtures("test")


if __name__ == '__main__':
    main()
