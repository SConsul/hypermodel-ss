import torch 
import numpy as np
from torch.utils.data import DataLoader

def calib_err(confidence, correct, p='2', beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

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

    return cerr

def evaluate(net,test_dataset,batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)
    confidences=[]
    correct = []
    with torch.no_grad():
        for img,lbl,_ in test_dataloader:
            img = img.to(device)
            lbl = lbl.to(device)
            t_conf, _ = net(img.to(device))
        
            conf, pred = t_conf.data.max(1)
            confidences.extend(conf.data.cpu().numpy().squeeze().tolist())
            correct.extend(pred.eq(lbl).cpu().numpy().squeeze().tolist())

    return calib_err(confidences,correct)