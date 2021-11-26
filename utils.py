import os
import torch
from datetime import datetime

def print_and_log(log_file, message):
    print(message)
    log_file.write(message + '\n')


def get_log_files(model_path,checkpoint_dir):
    checkpoint_dir = os.path.join(checkpoint_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(checkpoint_dir)

    model_dir = os.path.join(model_path,datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(model_dir)
    
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return model_dir,logfile



def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for images, labels, metadata in data_loader:
            yield (images, labels, metadata)

def save_model(net, filename):
    """Save trained model."""
    torch.save(net.state_dict(),filename)
    print("save model to: {}".format(filename))
    
def load_model(net, filename):
    """Load trained model."""
    model_root = "model_weights"
    net.load_state_dict(torch.load(os.path.join(model_root, filename)))
    print("load pretrained model from: {}".format(os.path.join(model_root,
                                                            filename)))
