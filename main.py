import torch
from train import pre_train, domain_adapt
from models.hydranet import HydraNet
from wilds import get_dataset
from torchvision import transforms
from evaluate import evaluate
from utils import load_model

def main():
    epoch_offset=0
    num_epochs = 50
    num_psudo_steps = 4
    num_adapt_epochs = 10
    num_target_init = 5000
    num_pseudo_heads = 0
    batch_size = 2
    num_classes = 62
    dataset = get_dataset(dataset='fmow_mini', download=False)
    train_dataset = dataset.get_subset('train', 
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    val_dataset = dataset.get_subset('val', 
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    net = HydraNet(num_heads=num_pseudo_heads, num_features=1024,
        num_classes=num_classes,pretrained=True)
    net = net.to(device)
    # load_model(net, "checkpoints/baseline/source_trained_4.pt")
    pre_train(net, device, train_dataset, val_dataset, batch_size,num_epochs,epoch_offset)
    
    if num_pseudo_heads>0:
        domain_adapt(net, device, train_dataset, val_dataset, num_psudo_steps, num_adapt_epochs)

    test_dataset = dataset.get_subset('test',transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    test_loss, test_acc, test_cerr = evaluate(net,device,test_dataset,batch_size)
    print("Test Loss={}, Test Acc={}, Test Calib Error={}".format(test_loss, test_acc, test_cerr))
if __name__=="__main__":
    main()