import json
import numpy as np
import torch
from torch.utils.data import Dataset

#Dataset class to feed to VAE
class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data_list[idx], dtype=torch.float32)
        return sample

#Read json file and outputs a list of dicts corresponding to every transmitter-receiver pairing
def load_data(path):
    data_objects = [] #each element is a dict with the CSI of a transmitter-receiver pairing
    with open(path, 'r') as file:
        for line in file:
            data_objects.append(json.loads(line))
    return data_objects

#Returns the CSI dictionaries of a desired receiver with the specified transmitter nodes -> Singles out receiver node among all nodes
def getReceiverCSI(data_dicts, receiver_number, transmitterRange):
    return [dct for dct in data_dicts if dct["RX"] == receiver_number and dct["TX"] in range(transmitterRange)]

#Extract delay vectors from each CSI dict of one receiver and groups them into the MDP matrix
def getMDPsfromReceiverCSI(receiver_dicts, timestamps):
    # Number of MDPS <- one for each timestamp
    num_MDPs = timestamps
    MDPs = [[] for _ in range(num_MDPs)]
    # Extract nth vector after 'Delay' and save it to the nth array
    for d in receiver_dicts:
        delay_val = d["Delay"]
        for i in range(num_MDPs):
            MDPs[i].append(delay_val[i])
    return MDPs

def pad_MDP(input_list):
    #Length of longest MDP vector
    max_length = max(len(sublist) for inner_list in input_list for sublist in inner_list)
    while(max_length % 8 != 0):
        max_length = max_length + 1
    return  [ [sublist + [0] * (max_length - len(sublist)) for sublist in inner_list] for inner_list in input_list ]
