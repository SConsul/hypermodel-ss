import os
import torch

def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for images, labels in data_loader:
            yield (images, labels)

def save_model(net, filename):
    """Save trained model."""
    model_root = "model_weights"
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    torch.save(net.state_dict(),
               os.path.join(model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(model_root,
                                                             filename)))
def load_model(net, filename):
    """Load trained model."""
    model_root = "model_weights"
    net.load_state_dict(torch.load(os.path.join(model_root, filename)))
    print("load pretrained model from: {}".format(os.path.join(model_root,
                                                            filename)))
