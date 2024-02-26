import csv
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

#######---------------------------------------------------------###########
#######    Functions for loading MDPs from simulator output     ###########
#######---------------------------------------------------------###########

def load_data(path):
    """
    Read json file and outputs a list of dicts corresponding to every transmitter-receiver pairing

    Parameters: 
    -path (str): path to json file containing output of QD Simulator

    Returns:
    - Python list, where each entry is a dictionary corresponding to every transmitter-receiver pairing
    """
    data_objects = [] #each element is a dict with the CSI of a transmitter-receiver pairing
    with open(path, 'r') as file:
        for line in file:
            data_objects.append(json.loads(line))
    return data_objects

def getReceiverCSI(data_dicts, receiver_id, number_of_transmitters):
    """
    Singles out desired receiver node to only work with its dictionaries/pairings

    Parameters:
    - data_dicts: List with all dictionaries with transmitter-receiver pairings
    - receiver_id: Id of node in the scenario which is considered the receiver
    - number_of_transmitters: Number of nodes in the scenario that are considered transmitters

    Returns:
    - List with only the pairings where our desired node is on the receiver side
    """
    return [dct for dct in data_dicts if dct["RX"] == receiver_id and dct["TX"] in range(number_of_transmitters)]

def getMDPsfromReceiverCSI(receiver_dicts, timestamps):
    """
    Extracts delay vectors from each CSI dict of one receiver and groups them into the MDP matrix 

    Parameters:
    - receiver_dicts: List of pairings of desired receiver
    - timestamps: Number of timestamps in the scenario which corresponds to the amount of receivers (diff. measurements)

    Returns:
    - 3D Python list: Every entry is a receiver MDP
    """
    # Number of MDPS <- one for each timestamp
    num_MDPs = timestamps
    MDPs = [[] for _ in range(num_MDPs)]
    #MDPs = []
    # Extract nth vector after 'Delay' and save it to the nth array
    for d in receiver_dicts:
        delay_val = d["Delay"]
        for i in range(num_MDPs):
            MDPs[i].append(delay_val[i])
            #MDPs.append(delay_val[i])a
    return MDPs

#Conbination of previous functions for readibility of code
def get_MDP_from_data(path, receiver_id, number_of_transmitters, timestamps):
    data = load_data(path)
    csi = getReceiverCSI(data, receiver_id, number_of_transmitters)
    mdps = getMDPsfromReceiverCSI(csi, timestamps)
    return mdps


#######----------------------------------------------------------------------###########
####### Class and function for preprocessing simulator data for model input  ###########
#######----------------------------------------------------------------------###########

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
def preprocess_mdps(mdp, feature_length):
    print('----------------Preprocessing step------------------------------------')
    #Get maximum and minimum lengths of MDPs
    max_length = max(len(sublist) for inner_list in mdp for sublist in inner_list)
    min_length = min(len(sublist) for inner_list in mdp for sublist in inner_list)
    
    #Set max_length
    if max_length > feature_length:
        print('Had to adjust maximum length because of MDP with extra paths in the set')
        max_length = feature_length
    #max_length = max(len(delay_vector) for delay_vector in mdp)
    #min_length = min(len(delay_vector) for delay_vector in mdp)
    print('Minimum length of a delay vector: ', min_length)
    print('Maximum length of a delay vector: ', max_length)
    
    #Pad with maximum value of MDPs or zeroes if not entries    
    padded_mdps = []
    for inner_list in mdp:
        padded_inner_list = []
        for id, sublist in enumerate(inner_list):
            if sublist:  # Non-empty sublist
                if len(sublist) > 14:
                    sublist = sublist[0:14]
                    print('Truncated sublist of point ' + str(id) + ' to first 14 paths')
                pad_length = max_length - len(sublist)
                half_pad_length = pad_length // 2
                # Decide how to distribute an odd number of padding elements
                max_pad = [max(sublist)] * (half_pad_length + pad_length % 2)  
                min_pad = [min(sublist)] * half_pad_length
                padded_sublist = sublist + max_pad + min_pad
            else:  # Empty sublist
                print('EMPTY MDP -> Point within Obstacle')
                print('Ignore Point ', id)
                padded_sublist = [0] * max_length
            padded_inner_list.append(padded_sublist)
        padded_mdps.append(padded_inner_list)


    #padded_mdps = [delay_vector+[0]*(max_length-len(delay_vector)) for delay_vector in mdp]
    padded_mdps = np.array(padded_mdps)
    print('Size of preprocessed (padded) MDPS: ', padded_mdps.shape)

    #Sort the MDPs in an ascending order and scale them with the speed of light to turn delays to distances
    sorted_mdps = np.array([[np.sort(row) for row in matrix] for matrix in padded_mdps])
    #sorted_mdps = np.array([np.sort(delay_vector) for delay_vector in padded_mdps])
    sorted_mdps = sorted_mdps * 299792458 
    sorted_mdps = sorted_mdps * 0.5 #Scaling for training reasons
    
    print('----------------------------------------------------------------------')
    return sorted_mdps

#######--------------------------------------------------------------------###########
#######         Functions for data augmentation for training model         ###########
#######--------------------------------------------------------------------###########

def random_delete_paths(batch, delete_probability, max_delete_count=4):
    """
    Randomly deletes elements in each batch sample, replacing them with random values from the same sample.
    This mimics obstacle blocking certain paths, used for data augmentation in training to make the model more flexible.

    Parameters:
    - batch: Input batch of samples.
    - delete_probability: Probability of deleting elements in a sample.
    - max_delete_count: Maximum number of elements to delete.

    Returns:
    - A new batch with elements randomly deleted and replaced with other random values from the sample.
    """
    deleted_batch = []
    for i in range(batch.size(0)):  # Iterate through each sample in the batch
        if np.random.rand() < delete_probability:
            # Decide how many elements to delete
            delete_count = np.random.randint(1, max_delete_count + 1)
            # Randomly choose indices to delete
            delete_indices = np.random.choice(range(batch[i].size(-1)), size=delete_count, replace=False)
            # Clone to avoid modifying the original batch
            deleted_sample = batch[i].clone()
            # For each index to delete, replace with a random value from the same sample
            for idx in delete_indices:
                # Choose a random index that is not being deleted
                replace_with_index = np.random.choice([i for i in range(batch[i].size(-1)) if i not in delete_indices])
                # Replace the value
                deleted_sample[..., idx] = deleted_sample[..., replace_with_index]
        else:
            # Leave sample as is
            deleted_sample = batch[i]
        # Add to new batch
        deleted_batch.append(deleted_sample)
    return torch.stack(deleted_batch)

def sort_batch(input_batch):
    """
    Sorts the elements of each feature vector in a batch [batch_size, channels, feature_length] in ascending order.
    """
    sorted_batch, _ = torch.sort(input_batch, dim=-1)
    return sorted_batch

def augment_data(batch, delete_probability=0.7):
    augmented_batch = random_delete_paths(batch, delete_probability)
    augmented_batch = sort_batch(augmented_batch)
    return augmented_batch

#######--------------------------------------------------###########
#######          Functions for evaluating VAE            ###########
#######--------------------------------------------------###########

#Function to evaluate VAE model and get latent representation and reconstruction
def evaluate_model(model, device, data_loader):
    latent_variables = []
    latent_means = []
    reconstructions = []
    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            mean, logvar = model.encoder(batch)
            z = model.reparameterize(mean, logvar)
            recon = model.decoder(z)
        latent_means.append(mean)
        latent_variables.append(z)
        reconstructions.append(recon)

    mean_tensors = torch.cat(latent_means, dim=0)
    z_tensors = torch.cat(latent_variables, dim=0)
    recon_tensors = torch.cat(reconstructions, dim=0)

    return mean_tensors.numpy(), recon_tensors.numpy()

#Function to evaluate VAE model and get latent representation and reconstruction on modified samples (augment data)
def evaluate_model_modified(model, device, data_loader):
    inputs = []
    latent_variables = []
    latent_means = []
    reconstructions = []
    for batch in data_loader:
        batch = batch.to(device)
        batch_augmented = augment_data(batch, delete_probability=1)
        with torch.no_grad():
            mean, logvar = model.encoder(batch_augmented)
            z = model.reparameterize(mean, logvar)
            recon = model.decoder(z)
        latent_means.append(mean)
        latent_variables.append(z)
        reconstructions.append(recon)
        inputs.append(batch_augmented)
    mean_tensors = torch.cat(latent_means, dim=0)
    recon_tensors = torch.cat(reconstructions, dim=0)
    input_tensors = torch.cat(inputs, dim=0)
    return mean_tensors.numpy(), recon_tensors.numpy(), input_tensors.numpy()


#######--------------------------------------------------###########
#######      Functions for systematic testing            ###########
#######--------------------------------------------------###########

def load_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            x, y, _ = row
            data.append((float(x), float(y)))
    return data

def find_closest_value(csv_path, target_x, target_y):
    csv_data = load_csv(csv_path)
    closest_distance = float('inf')
    closest_index = -1
    closest_values = None
    for index, (x, y) in enumerate(csv_data):
        distance = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
        if distance < closest_distance:
            closest_distance = distance
            closest_index = index
            closest_values = (x, y)
    return closest_index, closest_values

def get_MDP(csv_path, data_path, target_x, target_y):
    #find closest point in training set
    row, closest_values = find_closest_value(csv_path, target_x, target_y)
    #get training MPDs
    mdps = get_MDP_from_data(data_path, 1, 1, 1500)
    mdps_processed = preprocess_mdps(mdps)
    mdp = mdps_processed[row]
    return mdp, closest_values

def get_latent_variable(csv_path, data_path, target_x, target_y, model, device):
    #find closest point in training set
    row, closest_values = find_closest_value(csv_path, target_x, target_y)
    #get training MPDs
    mdps = get_MDP_from_data(data_path, 1, 1, 1500)
    mdps_processed = preprocess_mdps(mdps)
    mdp = mdps_processed[row]
    mdp = np.reshape(mdp, (1,14))
    #evaluate model on mdp
    mdp_set = MyDataset(mdp)
    mdp_dataloader = DataLoader(mdp_set, batch_size=1, shuffle=False)
    latent, recon = evaluate_model(model, device, mdp_dataloader)
    return mdp.squeeze(), latent.squeeze(), closest_values
