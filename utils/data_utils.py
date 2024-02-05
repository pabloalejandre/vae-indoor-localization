import json
import numpy as np
import torch
from torch.utils.data import Dataset

#######---------------------------------------------------------###########
####### Functions for loading EM MDPs from simulator output     ###########
#######---------------------------------------------------------###########

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

#This functions combines the previous functions for readibility of code in other scripts
def get_MDP_from_data(path, receiver_id, number_of_transmitters, timestamps):
    data = load_data(path)
    csi = getReceiverCSI(data, receiver_id, number_of_transmitters)
    mdps = getMDPsfromReceiverCSI(csi, timestamps)
    return mdps

#######--------------------------------------------------###########
####### Functions for preprocessing data for model input ###########
#######--------------------------------------------------###########

#Dataset class to turn data into pytorch dataset
class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data_list[idx], dtype=torch.float32)
        return sample

#Pads MDPs so that they can be turned to numpy arrays and sorts the MDPs in an ascending order
def pad_and_sort_MDPs(mdp):
    #Pad MDPs by matching the length of the longest sequence. Pad with infinities
    max_length = max(len(sublist) for inner_list in mdp for sublist in inner_list)
    print('Maximum length of a delay vector: ', max_length)
    padded_mdps = [[sublist+[np.inf]*(max_length-len(sublist)) for sublist in inner_list] for inner_list in mdp]
    padded_mdps = np.array(padded_mdps)
    print('Size of padded MDPS: ', padded_mdps.shape)
    #Sort the MDPs in an ascending order and scale them with the speed of light to turn delays to distances
    sorted_mdps = np.array([[np.sort(row) for row in matrix] for matrix in padded_mdps])
    sorted_mdps = sorted_mdps * 299792458 
    sorted_mdps = sorted_mdps * 0.5
    return sorted_mdps

    # Find the smallest and largest distance to normalize MDPs for VAE input
    finite_MDPS = sorted_mdps[np.isfinite(sorted_mdps)]
    min_val = np.min(sorted_mdps)
    print('Smallest distance: ', min_val)
    max_val = np.max(finite_MDPS)
    print('Largest distance: ', max_val)
    scale = max_val - min_val if max_val - min_val != 0 else 1
    normalized_mdps = sorted_mdps/scale
    return normalized_mdps
    
#Finds the shortest sequence length of non-padded-values
def find_shortest_sequence_length(mdps):
    shortest_length = np.inf
    for matrix in mdps:
        # Iterate over each row in the matrix
        for row in matrix:
            length = np.where(row == np.inf)[0][0] if np.any(row == np.inf) else len(row)
            shortest_length = min(shortest_length, length)
    return shortest_length

#Truncate batch two shortest sequence
def truncate_batch(batch, truncate_to):
    truncated_batch = batch[:, :, :truncate_to]
    return truncated_batch


#######--------------------------------------------------###########
#######    Functions for dealing with VAE outputs        ###########
#######--------------------------------------------------###########

#Function to generate latent variables from data MDPs (torch_dataloader)
def get_latent_representation(model, device, data_loader):
    latent_variables = []
    means = []
    logvars = []
    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            mean, logvar = model.encoder(batch)
            z = model.reparameterize(mean, logvar)
        latent_variables.append(z)
        means.append(mean)
        logvars.append(logvar)
    return torch.cat(means, dim=0)
    #return torch.cat(latent_variables, dim=0)