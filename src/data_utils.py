import json
import numpy as np

#Read json file and outputs a list of dicts corresponding to every transmitter-receiver pairing
def load_data(path):
    data_objects = [] #each element is a dict with the CSI of a transmitter-receiver pairing
    with open(path, 'r') as file:
        for line in file:
            data_objects.append(json.loads(line))
    return data_objects

#Returns the CSI dictionaries of a desired receiver with the specified transmitter nodes -> Singles out receiver node among all nodes
def getReceiverCSI(data_dicts, receiver_number, transmitterRange):
    return [dct for dct in data_dicts if dct["RX"] == receiver_number and dct["TX"] in transmitterRange]

#Extract delay vectors from each CSI dict of one receiver and groups them into the MDP matrix
#Might have to change implementation if multiple timestamps -> Training data! If statement to check whether multiple timestamps or not
#Boolean for training (multiple timestamps) or validation (multiple receivers, one timestamp?)
def getMDPfromReceiverCSI(receiver_dicts, training):
    if(training == 1):
        pass
    else:
        return sum([d['Delay'] for d in receiver_dicts], [])

dicts = load_data("data/BoxLectureRoomOffline.json")
max_matrix_size = max(len(d['Delay']) for d in dicts)
data = []

for receiver_id in range(3,7):
    rcv_CSI = getReceiverCSI(dicts, receiver_id, range(3))
    rcv_MDP = getMDPfromReceiverCSI(rcv_CSI)
    data.append(rcv_MDP)

print(data)

#PROBLEMS:
#
# a) How to determine the number of reciever nodes in the json file to extract all the MDPS that will be inputs to VAE
#
# b) How to handle potential scenarios with multiple timestamps? /
#    For training: only one receiver with multiple timestamps: change implementation of getMDP()
#    getData() and getReceiverCSI() should be fine regardless
#
# c) Padding the MDPs to enforce same dimension
#