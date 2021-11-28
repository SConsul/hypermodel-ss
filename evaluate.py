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
    device_loc = "cpu"

    test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)
    t_confidences = torch.Tensor().to(device_loc)
    t_correct = torch.Tensor().to(device_loc)
    p_confidences = torch.Tensor().to(device_loc)
    p_correct = torch.Tensor().to(device_loc)
    vloss = []
    num_total = 0.0
    t_num_correct = 0.0
    p_num_correct = torch.zeros(net.num_heads).to(device_loc)
    criterion = nn.CrossEntropyLoss()
    threshold=0.9 ###MAKE PARARM
    num_common_corrects = 0.0
    num_common_incorrects = 0.0
    num_common_incorrects_high_conf = 0.0
    num_common_corrects_high_conf = 0.0

    

    with torch.no_grad():
        for i, (img,lbl,_) in enumerate(tqdm(test_dataloader)):
            img = img.to(device)
            lbl = lbl.to(device)
            t_out, p_outs = net(img.to(device))

            loss = criterion(t_out,lbl)
            vloss.append(loss.to(device_loc))

            t_conf, t_pred = t_out.data.max(1)
            t_conf, t_pred = t_conf.to(device_loc), t_pred.to(device_loc)
            lbl = lbl.to(device_loc)

            t_num_correct += (t_pred==lbl).double().sum().item()
            num_total += t_pred.size(0)
            t_confidences = torch.cat([t_confidences, t_conf])
            t_correct = torch.cat([t_correct, t_pred.eq(lbl)])

            if net.num_heads>0:
                common_corr = torch.ones(t_conf.shape[0]).to(device_loc)
                common_inc = torch.ones(t_conf.shape[0]).to(device_loc)
                common_inc_high_conf = torch.zeros(t_conf.shape[0]).to(device_loc)
                common_corr_high_conf = torch.zeros(t_conf.shape[0]).to(device_loc)
                max_confs = torch.zeros(t_conf.shape[0])
                for i,p_out in enumerate(p_outs):
                    p_conf, p_pred = p_out.data.max(1)
                    p_conf, p_pred = p_conf.to(device_loc), p_pred.to(device_loc)
                    p_num_correct[i] += (p_pred==lbl).double().sum().item()

                    max_confs = torch.max(max_confs,p_conf)
                    p_corr = (p_pred==lbl)
                    p_inc = (p_pred!=lbl)
                    common_corr *= p_corr
                    common_inc *= p_inc #B,
                    
                    high_conf = p_conf>threshold
                    p_inc_high = high_conf*p_inc
                    p_corr_high = high_conf*p_corr
                    common_inc_high_conf = ((common_inc_high_conf+p_inc_high).clamp(0,1))*common_inc
                    common_corr_high_conf = ((common_corr_high_conf+p_corr_high).clamp(0,1))*common_corr

                    ens_p_conf = max_confs*((common_corr_high_conf+common_inc_high_conf).clamp(0,1))
                    ens_corr = common_corr_high_conf

                p_confidences = torch.cat([p_confidences, ens_p_conf])
                p_correct = torch.cat([p_correct, ens_corr])
                
                num_common_corrects += common_corr.double().sum().item()
                num_common_incorrects += common_inc.double().sum().item()
                num_common_incorrects_high_conf += common_inc_high_conf.double().sum().item()
                num_common_corrects_high_conf += common_corr_high_conf.double().sum().item()

    val_loss = np.array(vloss).mean()            
    t_cerr = calib_err(np.array(t_confidences),np.array(t_correct)) 
    p_cerr = calib_err(np.array(p_confidences),np.array(p_correct)) 
    acc = t_num_correct/num_total  
    acc_pheads = p_num_correct/num_total

    com_corr_high = num_common_corrects_high_conf/num_total
    com_corr = num_common_corrects/num_total
    com_inc = num_common_incorrects/num_total
    com_inc_high = num_common_incorrects_high_conf/num_total
    disag = 1.0 - com_corr - com_inc
    return val_loss, (acc, acc_pheads), t_cerr, (com_corr_high, com_corr, com_inc, com_inc_high, disag, p_cerr)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device={}".format(device))
    batch_size = 4
    num_classes = 62
    num_pseudo_heads = 0

    net = HydraNet(num_heads=num_pseudo_heads, num_features=1024,
        num_classes=num_classes,pretrained=False)
    net = net.to(device) 
    load_model(net, "C:\\Users\\akagr\\hypermodel-ss\\model_weights_vm\\baseline\\source_trained_10.pt")

    #net = torchvision.models.densenet.densenet121()

    # pretrained_dict = torch.load("pretrained/fmow_seed_0_epoch_best_model.pth")
    # net.load_state_dict(pretrained_dict['algorithm'])
    # net = net.to(device)
    
    dataset = get_dataset(dataset='fmow_mini', download=False)
    test_dataset = dataset.get_subset('val', frac=1,
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    test_loss, accs, test_cerr, val_pHead_stats = evaluate(net,device,test_dataset,batch_size)
    com_corr_high, com_corr, com_inc, com_inc_high, disag, p_cerr = val_pHead_stats
    print("Target Loss={}, Target Acc={}, Target Calib Error={}".format(test_loss, accs[0], test_cerr))
    print(f"Pseudo head accuracies: {accs[1]}")
    print("com_corr_high={:.7f}, com_corr={:.7f}, com_inc={:.7f}, com_inc_high={:.7f}, disag={:.7f}, P_Cal Error={:.7f}".format(
                com_corr_high, com_corr,com_inc,com_inc_high,disag,p_cerr))
            
    print("\a")
    
