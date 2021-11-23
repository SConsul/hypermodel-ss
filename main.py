import torch
from train import pre_train, pseudo_label, domain_adapt
from models.hydranet import HydraNet
from wilds import get_dataset
from torchvision import transforms
from evaluate import evaluate

def main():
    num_epochs = 10
    num_psudo_steps = 4
    num_tt_epochs = 10
    num_target_init = 5000
    num_pseudo_heads = 0
    batch_size = 16
    dataset = get_dataset(dataset='fmow_mini', download=False)
    train_dataset = dataset.get_subset('train',transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    val_dataset = dataset.get_subset('val',transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    net = HydraNet(num_heads=num_pseudo_heads, num_features=1024,num_classes=1000,pretrained=False)
    net = net.to(device)
    pre_train(net, device, train_dataset, val_dataset, batch_size, num_epochs)
    
    if num_pseudo_heads>0:
        excerpt, pseudo_labels = pseudo_label(net, device, val_dataset, num_target_init) 
        domain_adapt(net, device, train_dataset, val_dataset, excerpt, pseudo_labels)

    test_dataset = dataset.get_subset('test',transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    test_cerr = evaluate(net,device,test_dataset,batch_size)
    print("Test Calibration Error={:.5f}".format(test_cerr))
if __name__=="__main__":
    main()