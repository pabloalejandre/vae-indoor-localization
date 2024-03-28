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
    - receiver_id: Id of node which is considered the receiver
    - number_of_transmitters: Number of transmitter nodes

    Returns:
    - List with only the pairings where our desired node is on the receiver side
    """
    return [dct for dct in data_dicts if dct["RX"] == receiver_id and dct["TX"] in range(number_of_transmitters)]

def getMDPsfromReceiverCSI(receiver_dicts):
    """
    Extracts delay vectors from each CSI dict of one receiver and groups them into the MDP matrix 

    Parameters:
    - receiver_dicts: List of pairings of desired receiver

    Returns:
    - 3D Python list: Every entry is a receiver MDP
    """
    speed_of_light = 3e8
    num_MDPs = len(receiver_dicts[0]["Delay"])
    #MDPs = [[] for _ in range(num_MDPs)]
    MDPs = []
    # Extract nth vector after 'Delay' and save it to the nth array, scaling it to obtain a distance
    for d in receiver_dicts:
        delay_mdp = d["Delay"]
        for i in range(num_MDPs):
            distance_mdp = [delay * speed_of_light for delay in delay_mdp[i]]
            if distance_mdp: 
                MDPs.append([distance_mdp]) 
            #MDPs[i].append(delay_mdp[i])
    print(f'Loaded {num_MDPs} MDPs from simulator output.')
    return MDPs

#Conbination of previous functions for readibility of code
def get_MDP_from_data(path, receiver_id=1, number_of_transmitters=1):
    data = load_data(path)
    csi = getReceiverCSI(data, receiver_id, number_of_transmitters)
    mdps = getMDPsfromReceiverCSI(csi)
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


def preprocess_mdps(mdp, feature_length):
    print('----------------Preprocessing step------------------------------------')
    # Get maximum and minimum lengths of MDPs
    max_length = max(len(sublist) for inner_list in mdp for sublist in inner_list)
    min_length = min(len(sublist) for inner_list in mdp for sublist in inner_list)
    
    print('Minimum length of a delay vector: ', min_length)
    print('Maximum length of a delay vector: ', max_length)
    # Set max_length
    if max_length != feature_length:
        max_length = feature_length
        print('Had to adjust maximum length for VAE input size. New maximum length: ', max_length)
        
    padded_mdps = []
    for inner_list in mdp:
        padded_inner_list = []
        for sublist in inner_list:
            if sublist:
                if len(sublist) > max_length:
                    sublist = sublist[:max_length]
                pad_length = max_length - len(sublist)
                if pad_length > 0:
                    print(f'Have to pad {pad_length} elements')
                # Sort the sublist for padding order
                sorted_sublist = sorted(sublist)
                # Prepare pad values
                pad_values = []
                left, right = 0, len(sorted_sublist) - 1
                available_indices = range(len(sublist))
                for _ in range(pad_length):
                    if _ % 2 == 0:  # Even index, choose from the right (max)
                        pad_values.append(sorted_sublist[right])
                        right -= 1
                        if right not in available_indices:
                            right = len(sorted_sublist) - 1
                    else:  # Odd index, choose from the left (min)
                        pad_values.append(sorted_sublist[left])
                        left += 1
                        if left not in available_indices:
                            left = 0
                padded_sublist = sublist + pad_values
            else:  # Empty sublist
                print('EMPTY MDP -> Point within Obstacle')
                padded_sublist = [0] * max_length
            padded_inner_list.append(padded_sublist)
        padded_mdps.append(padded_inner_list)

    padded_mdps = np.array(padded_mdps)
    print('Size of preprocessed (padded) MDPS: ', padded_mdps.shape)
    # Sort the MDPs in an ascending order and scale them with the speed of light to turn delays to distances
    sorted_mdps = np.array([[np.sort(row) for row in matrix] for matrix in padded_mdps])
    sorted_mdps = sorted_mdps * 0.5 # Scaling for training reasons
    print('----------------------------------------------------------------------')
    return sorted_mdps

#######--------------------------------------------------------------------###########
#######         Functions for data augmentation for training model         ###########
#######--------------------------------------------------------------------###########

def random_delete_paths(batch, delete_probability, max_groups, max_group_size):
    """
    Randomly deletes groups of elements in each batch sample, with each group having a separately determined size.
    This simulates obstacle blocking certain paths in groups, used for data augmentation in training to make the model more flexible.

    Parameters:
    - batch: Input batch of samples.
    - delete_probability: Probability of deleting groups of elements in a sample.
    - max_groups: Maximum number of groups to delete. The size of each group is determined independently.

    Returns:
    - A new batch with groups of elements randomly deleted and replaced with other random values from the sample.
    """
    deleted_batch = []
    for i in range(batch.size(0)):
        if np.random.rand() < delete_probability:
            deleted_sample = batch[i].clone()
            sample_size = batch[i].size(-1)
            group_sizes = np.random.randint(0, max_group_size+1, size=max_groups)
            for group_size in group_sizes:
                # Ensure the group can fit in the remaining elements
                if sample_size - group_size > 0:
                    start_index = np.random.randint(0, sample_size - group_size)
                    end_index = start_index + group_size
                    
                    # Generate a list of indices for replacement that are outside the current deletion group
                    possible_replacements = [i for i in range(sample_size) if i < start_index or i >= end_index]
                    for idx in range(start_index, end_index):
                        if possible_replacements:
                            replace_with_index = np.random.choice(possible_replacements)
                            # Replace the value
                            deleted_sample[..., idx] = deleted_sample[..., replace_with_index]
        else:
            deleted_sample = batch[i]
        
        deleted_batch.append(deleted_sample)
    return torch.stack(deleted_batch)

def random_delete_paths(batch, delete_probability, max_groups, max_group_size):
    """
    Randomly deletes groups of elements in each batch sample, with each group having a separately determined size.
    This simulates obstacle blocking certain paths in groups, used for data augmentation in training to make the model more flexible.

    Parameters:
    - batch: Input batch of samples.
    - delete_probability: Probability of deleting groups of elements in a sample.
    - max_groups: Maximum number of groups to delete. The size of each group is determined independently.

    Returns:
    - A new batch with groups of elements randomly deleted and replaced with other random values from the sample.
    """
    deleted_batch = []
    for i in range(batch.size(0)):
        if np.random.rand() < delete_probability:
            deleted_sample = batch[i].clone()
            sample_size = batch[i].size(-1)
            group_size = np.random.randint(0, max_group_size+1)
            
            start_index = np.random.randint(0, sample_size)
            upper_index = start_index + int(group_size/2)      
            lower_index = start_index - int(group_size/2)
            # Generate a list of indices for replacement that are outside the current deletion group
            possible_replacements = [i for i in range(sample_size) if (i < lower_index and i>=0) or (i >= upper_index and i<=sample_size)]
            for idx in range(lower_index, upper_index):
                if idx in range(0, sample_size):
                    replace_with_index = np.random.choice(possible_replacements)
                    # Replace the value
                    deleted_sample[..., idx] = deleted_sample[..., replace_with_index]   
        else:
            deleted_sample = batch[i]   
        deleted_batch.append(deleted_sample)
    return torch.stack(deleted_batch)


def sort_batch(input_batch):
    """
    Sorts the elements of each feature vector in a batch [batch_size, channels, feature_length] in ascending order.
    """
    sorted_batch, _ = torch.sort(input_batch, dim=-1)
    return sorted_batch

def augment_data(batch, delete_probability, max_groups=2, max_group_size=1):
    augmented_batch = random_delete_paths(batch, delete_probability, max_groups, max_group_size)
    augmented_batch = sort_batch(augmented_batch)
    return augmented_batch

#######--------------------------------------------------###########
#######          Functions for evaluating VAE            ###########
#######--------------------------------------------------###########

def evaluate_model(model, feature_length, data_loader):
    """
    Evaluates the model to obtain the latent space representations and the reconstructions of the input data.

    Parameters:
    - model: A VAE model instance.
    - feature_length: An integer specifying the size of the feature vector expected by the VAE.
    - data_loader: Pytorch dataloader with data to evaluate.

    Returns:
    - mean_tensors: A numpy array containing the latent means for all data points with shape (N, latent_dim).
    - recon_tensors: A numpy array containing the reconstructions of the input data with shape (N, feature_length).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent_variables = []
    latent_means = []
    reconstructions = []
    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            mean, logvar = model.encoder(batch)
            z = model.reparameterize(mean, logvar)
            recon = model.decoder(z, feature_length)
        latent_means.append(mean)
        latent_variables.append(z)
        reconstructions.append(recon)

    mean_tensors = torch.cat(latent_means, dim=0)
    z_tensors = torch.cat(latent_variables, dim=0)
    recon_tensors = torch.cat(reconstructions, dim=0)

    return mean_tensors.numpy(), recon_tensors.numpy()


#Function to evaluate VAE model and get latent representation and reconstruction on modified samples (augment data)
def evaluate_model_modified(model, feature_length, data_loader):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = []
    latent_variables = []
    latent_means = []
    reconstructions = []
    for batch in data_loader:
        batch = batch.to(device)
        batch_augmented = augment_data(batch, delete_probability=1, max_groups=1, max_group_size=4)
        with torch.no_grad():
            mean, logvar = model.encoder(batch_augmented)
            z = model.reparameterize(mean, logvar)
            recon = model.decoder(z, feature_length)
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
    mdps = get_MDP_from_data(data_path)
    mdps_processed = preprocess_mdps(mdps)
    mdp = mdps_processed[row]
    return mdp, closest_values

def get_latent_variable(csv_path, data_path, target_x, target_y, model):
    #find closest point in training set
    row, closest_values = find_closest_value(csv_path, target_x, target_y)
    #get training MPDs
    mdps = get_MDP_from_data(data_path)
    mdps_processed = preprocess_mdps(mdps)
    mdp = mdps_processed[row]
    mdp = np.reshape(mdp, (1,14))
    #evaluate model on mdp
    mdp_set = MyDataset(mdp)
    mdp_dataloader = DataLoader(mdp_set, batch_size=1, shuffle=False)
    latent, recon = evaluate_model(model, mdp_dataloader)
    return mdp.squeeze(), latent.squeeze(), closest_values
