import torch
from train import source_train, domain_adapt
from models.hydranet import HydraNet
from wilds import get_dataset
from torchvision import transforms
from evaluate import evaluate
from utils import load_model, print_and_log, get_log_files

def main():
    epoch_offset=0
    num_epochs = 30
    num_pseudo_steps = 10
    num_adapt_epochs = 2
    num_pseudo_heads = 2
    batch_size = 64
    num_classes = 62
    orig_frac = 1 # fraction of data to be used while training
                  # useful to set to 5e-2 for local runs
    threshold = 0.9

    if num_pseudo_heads>0:
        log_loc = f"logs/ssl_{num_pseudo_heads}"
    else:
        log_loc = f"logs/baseline"

    log_file = get_log_files(log_loc)


    dataset = get_dataset(dataset='fmow_mini', download=False)
    train_dataset = dataset.get_subset('train', frac=orig_frac,
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    val_dataset = dataset.get_subset('val', frac=orig_frac,
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    net = HydraNet(num_heads=num_pseudo_heads, num_features=1024,
        num_classes=num_classes,pretrained=True)
    net = net.to(device)
    #net.load_state_dict(torch.load("pretrained/source_trained_2021-11-26-03-22-44_epoch_30.pt"))

    source_train(net, device, train_dataset, val_dataset, batch_size,num_epochs,log_file,epoch_offset)
    
    if num_pseudo_heads>0:
        for k in range(1,num_pseudo_steps+1):
            frac = min(1., k/num_pseudo_steps)
            target_dataset = dataset.get_subset('val', frac=orig_frac*frac,
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
            domain_adapt(net, device, train_dataset, val_dataset, 
                batch_size, k, num_adapt_epochs, threshold, log_file)

    test_dataset = dataset.get_subset('test',transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    test_loss, test_acc, test_cerr = evaluate(net,device,test_dataset,batch_size)
    print_and_log(message="Test Loss={}, Test Acc={}, Test Calib Error={}".format(
        test_loss, test_acc, test_cerr), log_file=log_file)

    log_file.close()

if __name__=="__main__":
    main()