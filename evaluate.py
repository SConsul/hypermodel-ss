import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from wilds import get_dataset
from models.hydranet import HydraNet
from utils import load_model
from tqdm import tqdm

def calib_err(confidence, correct, p='2', beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    # bins[-1] = [bins[-1][0], len(confidence)]
    bins.append([[beta*(len(confidence) // beta), len(confidence)]])

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)
    # print(cerr)
    return cerr

def evaluate(net,device,test_dataset,batch_size):
    net.eval()
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)
    confidences=[]
    correct = []
    vloss = []
    num_correct = 0.0
    num_total = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (img,lbl,_) in enumerate(tqdm(test_dataloader)):
            img = img.to(device)
            lbl = lbl.to(device)
            t_conf, _ = net(img.to(device))
            #t_conf = net(img.to(device))
            loss = criterion(t_conf,lbl)
            vloss.append(loss.cpu())
            conf, pred = t_conf.data.max(1)
            #print(f'loss {i}: {loss.cpu()}')
            num_correct += (pred==lbl).double().sum().item()
            num_total += pred.size(0)
            # print(f'conf shape: {conf.shape}')
            if conf.shape[0] > 1:
                confidences.extend(conf.data.cpu().numpy().squeeze().tolist())
                correct.extend(pred.eq(lbl).cpu().numpy().squeeze().tolist())
            else:
                confidences.append(conf.data.cpu().numpy().squeeze())
                correct.append(pred.eq(lbl).cpu().numpy().squeeze())

    val_loss = np.array(vloss).mean()            
    cerr = calib_err(np.array(confidences),np.array(correct)) 
    acc = num_correct/num_total  
    return val_loss, acc, cerr

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device={}".format(device))
    batch_size = 2
    num_classes = 62
    num_pseudo_heads = 0

    # net = HydraNet(num_heads=num_pseudo_heads, num_features=1024,
    #     num_classes=num_classes,pretrained=False)
    # net = net.to(device) 
    # load_model(net, "C:\\Users\\akagr\\model_weights\\checkpoints\\baseline\\source_trained_2021-11-23-22-44-52_10.pt")

    net = torchvision.models.densenet.densenet121()

    pretrained_dict = torch.load("pretrained/fmow_seed_0_epoch_best_model.pth")
    net.load_state_dict(pretrained_dict['algorithm'])
    net = net.to(device)
    
    dataset = get_dataset(dataset='fmow_mini', download=False)
    test_dataset = dataset.get_subset('val', 
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    test_loss, test_acc, test_cerr = evaluate(net,device,test_dataset,batch_size)
    print("Test Loss={}, Test Acc={}, Test Calib Error={}".format(test_loss, test_acc, test_cerr))
    
