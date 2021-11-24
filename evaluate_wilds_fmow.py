import torch
import argparse
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms
from torchvision import models as torchmodels
from evaluate import evaluate
from wilds import get_dataset

def parse_bool(v):
    if v.lower()=='true':
        return True
    elif v.lower()=='false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-','').isnumeric():
                processed_val = int(value_str)
            elif value_str.replace('-','').replace('.','').isnumeric():
                processed_val = float(value_str)
            elif value_str in ['True', 'true']:
                processed_val = True
            elif value_str in ['False', 'false']:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x):
        return x

def initialize_torchvision_model(name, d_out, **kwargs):
    # get constructor and last layer names
    if name == 'wideresnet50':
        constructor_name = 'wide_resnet50_2'
        last_layer_name = 'fc'
    elif name == 'densenet121':
        constructor_name = name
        last_layer_name = 'classifier'
        print("HELLO")
    elif name in ('resnet50', 'resnet34', 'resnet18'):
        constructor_name = name
        last_layer_name = 'fc'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    # construct the default model, which has the default last layer
    constructor = getattr(torchmodels, constructor_name)
    print(constructor)
    model = constructor(pretrained=True)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    print(d_features)
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else: # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)
    print("in init ",model.d_out)
    return model

def initialize_model(config, d_out, is_featurizer=False):
    """
    Initializes models according to the config
        Args:
            - config (dictionary): config dictionary
            - d_out (int): the dimensionality of the model output
            - is_featurizer (bool): whether to return a model or a (featurizer, classifier) pair that constitutes a model.
        Output:
            If is_featurizer=True:
            - featurizer: a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality)
            - classifier: a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear layer.

            If is_featurizer=False:
            - model: a model that is equivalent to nn.Sequential(featurizer, classifier)
    """
    if config.model in ('resnet50', 'resnet34', 'resnet18', 'wideresnet50', 'densenet121'):
        if is_featurizer:
            featurizer = initialize_torchvision_model(
                name=config.model,
                d_out=None,
                **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            print("initialize ",config.model)
            model = initialize_torchvision_model(
                name=config.model,
                d_out=d_out,
                **config.model_kwargs)
        return model

if __name__=="__main__":
    '''
    python eval_wilds_fmow.py
    '''
    batch_size = 64
    num_classes = 62

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='densenet121')
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
        help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=True)
    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device={}".format(device))
    net = initialize_model(config,d_out=62)
    net.to(device)
    weight_path = '../pretrained/fmow_seed_0_epoch_best_model.pth'
    if device=='cpu':
        pre_dict = torch.load(weight_path,map_location=torch.device('cpu'))
    else:
        pre_dict = torch.load(weight_path)

    d2 = OrderedDict([(key[6:],val) for key,val in pre_dict['algorithm'].items()])
    net.load_state_dict(d2)

    dataset = get_dataset(dataset='fmow_mini', download=False)
    test_dataset = dataset.get_subset('test',transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    test_loss, test_acc, test_cerr = evaluate(net,device,test_dataset,batch_size)
    print("Test Loss={}, Test Acc={}, Test Calib Error={}".format(test_loss, test_acc, test_cerr))
    