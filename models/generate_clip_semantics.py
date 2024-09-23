## Running command -> python models/generate_clip_semantics.py

from clip import clip
import torch
from static_hico import ACT_IDX_TO_ACT_NAME, HICO_INTERACTIONS, ACT_TO_ING, HOI_IDX_TO_ACT_IDX, \
    HOI_IDX_TO_OBJ_IDX, MAP_AO_TO_HOI, UA_HOI_IDX, OBJ_IDX_TO_OBJ_NAME
import numpy as np

def prompt(d):
    a = d['action']
    o = d['object']
    if a == 'no_interaction':
        return "A photo of a person and {}".format(o.replace('_', ' '))
    else:
        return "A photo of a person {} {}".format(ACT_TO_ING[a], o.replace('_', ' '))
    
device = torch.device('cuda')

model_path = '../ckpt/RN50x16.pt'
clip_model, preprocess = clip.load(model_path, device=device)

print("Turning off gradients in both the image and the text encoder")
for name, param in clip_model.named_parameters():
    param.requires_grad_(False)

## Step1: Generate CLIP text features for each action
# text features shape -> 117x768
# for each action, see what all objects are possible to interact with
a_os_dict = {}
for interaction in HICO_INTERACTIONS:
    action_name = ACT_TO_ING[interaction['action']]
    object_name = interaction['object']
    if action_name not in a_os_dict:
        a_os_dict[action_name] = []
    a_os_dict[action_name].append(object_name)

text_features = torch.zeros((117, 768)).to(device)

for i in range(117):
    action_name = ACT_TO_ING[ACT_IDX_TO_ACT_NAME[i]]
    ao_pairs = [(action_name, object_name) for object_name in a_os_dict[action_name]]
    text_inputs = torch.cat([clip.tokenize("A photo of a person {} {}".format(a, o.replace('_', ' '))) for a, o in ao_pairs]).to(device)
    if ACT_IDX_TO_ACT_NAME[i] == 'no_interaction':
        text_inputs = torch.cat([clip.tokenize("A photo of a person and {}".format(o.replace('_', ' '))) for _, o in ao_pairs]).to(device)
    text_features_curr = clip_model.encode_text(text_inputs)
    text_features_curr = torch.mean(text_features_curr, dim=0)
    text_features[i] = text_features_curr

text_features_np = text_features.cpu().detach().numpy()
print(text_features_np.shape)
np.save('CLIP_action_semantics.npy', text_features_np)
print("CLIP text features for actions generated successfully .... ")


## Step 2: Generating CLIP text features for each interaction
# text_features.shape = 600x768
    
text_inputs = torch.cat([clip.tokenize(prompt(d)) for d in HICO_INTERACTIONS]).to(device)
text_features = clip_model.encode_text(text_inputs)
text_features_np = text_features.cpu().detach().numpy()
print(text_features_np.shape)
np.save('CLIP_interaction_semantics.npy', text_features_np)
print("CLIP text features for interactions generated successfully .... ")

## Step 3: Generating CLIP text features for each object
# text_features.shape = 80x768
objs = [OBJ_IDX_TO_OBJ_NAME[i] for i in range(80)] + ['background']
text_inputs = torch.cat(
    [clip.tokenize("A photo of {}".format(o.replace('_', ' '))) for o in objs]).to(device)
text_features = clip_model.encode_text(text_inputs)
text_features_np = text_features.cpu().detach().numpy()
print(text_features_np.shape)   
np.save('CLIP_object_semantics.npy', text_features_np)
print("CLIP text features for objects generated successfully .... ")