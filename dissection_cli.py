# Modified by Fahim
# Modified by Hassan

'''
DISSECTION AND VISUALIZATION SERVER
===================================

Run with given arguments. An example --models file is given in models-example.txt.

To run visualizations, go to `localhost:8080` (or `localhost:8080/index.html`).

To run modified translations, go to `localhost:8080/modify.html`.

'''
import torch
import numpy
import json
from tqdm import tqdm
from torch.utils.serialization import load_lua
from itertools import product as p
import codecs
import argparse
import os

parser = argparse.ArgumentParser(description='Visualization and dissection server')
parser.add_argument('--models', help='File with list of lines in format: `description_loc model_loc src_dict_loc targ_dict_loc`, space-separated')
parser.add_argument('--svcca', help='.pkl file output by svcca.py (optional)')
parser.add_argument('--source', help='tokenized source file for the description files')

args = parser.parse_args()

'''
LOAD NETWORKS
'''

# Get list of network filenames
with open(args.models) as f:
    network_fnames = [line.strip().split(' ') for line in f]

all_networks = {}
model_files = {}
src_dicts = {}
targ_dicts = {}

for fname, mname, sdict, tdict in tqdm(network_fnames):
    network_name = os.path.split(fname)[1]
    print ("Network Name:", network_name)

    # Record the location of the model file for this description file
    model_files[network_name] = os.path.abspath(mname)
    src_dicts[network_name] = os.path.abspath(sdict)
    targ_dicts[network_name] = os.path.abspath(tdict)

    # Load as 4000x(sentence_length)x500 matrix
    all_networks[network_name] = load_lua(fname)['encodings']

means = {}
variances = {}

# transforms
if args.svcca is not None:
    cca_transforms = torch.load(args.svcca)

# Get means and variances
for network in tqdm(all_networks, desc = 'norm, pca'):
    # large number x 500
    concatenated = torch.cat(all_networks[network], dim = 0).cuda()
    means[network] = concatenated.mean(0)
    variances[network] = concatenated.std(0)

    means[network] = means[network].cpu()
    variances[network] = variances[network].cpu()

import os
import subprocess
from subprocess import PIPE

current_loaded_subprocess = None
current_network = None

current_network = list(all_networks.keys())[0]

model_name = model_files[current_network]
src_dict = src_dicts[current_network]
targ_dict = targ_dicts[current_network]

if current_loaded_subprocess is not None:
    current_loaded_subprocess.kill()

current_loaded_subprocess = subprocess.Popen(
    [   '/usr/bin/env',
        'th', # Find user's torch
        os.path.abspath('seq2seq-attn/dissect.lua'), # Find our modified seq2seq
        '-model', model_name,
        '-src_file', os.path.abspath(args.source),
        '-src_dict', src_dict,
        '-targ_dict', targ_dict,
        '-replace_unk', '1',
        '-gpuid', '1'
    ],
    cwd = 'seq2seq-attn',
    stdin = PIPE,
    stdout = PIPE
)

# Read the two "loading" lines out
current_loaded_subprocess.stdout.readline()
current_loaded_subprocess.stdout.readline()

sentences = ["I went to school .", "UNICEF published videos depicting people being subjected to a range of abhorrent punishments .", "UNICEF disbursed emergency cash assistance to tens of thousands of displaced families in camps .", "We underscore the need to accelerate efforts at all levels ."]
neurons = [402, 369, 472, 146, 315, 440, 327,  62, 104,  83, 247,  52]

pos = 1
for sentence in sentences:
    for n in neurons:

        print ("Neuron:", n)
        modifications = [
        	{'position': [pos, n], 'value': 0},
        	{'position': [pos, n], 'value': -1},
            {'position': [pos, n], 'value': 1},        
        ]


        for modification in modifications:
            index, neuron = modification['position']

            modification['value'] = (
                means[current_network][neuron].item() + modification['value'] *
                variances[current_network][neuron].item()
            )

            modification['position'] = (index + 1, neuron + 1)

        
        	#(json.dumps(modifications) + '\n').encode('ascii')
        	# Put some things in
        	current_loaded_subprocess.stdin.write(
        	    (sentence + '\n').encode('ascii')
        	)
        	current_loaded_subprocess.stdin.write(
        	    (json.dumps(modifications) + '\n').encode('ascii')
        	)
        	current_loaded_subprocess.stdin.flush()

        	# Get response out
        	response = current_loaded_subprocess.stdout.readline().decode('utf-8')

        	print(response.strip())

current_loaded_subprocess.kill()
