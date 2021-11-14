import torch 
import torch.nn as nn
from hydranet import HydraNet
from itertools import chain



def pseudo_label(net,test_img, n_t):
    _,p_outs = net(test_img) #p_outs is a list of size num_heads of tensor of length num_classes


    return pseudo_img, pseudo_test_labels


def train(num_heads,num_epochs,num_psudo_steps,num_tt_epochs,n_t):
    net = HydraNet(num_heads=2, num_features=1024,num_classes=1000,pretrained=False)
    criterion = nn.CrossEntropy()
    optimizer_S = torch.optim.Adam(net.parameters())
    optimizer_pseudo = torch.optim.Adam(list(net.enc.parameters())+list(net.pheads.parameters()))
    optimizer_target = torch.optim.Adam(list(net.enc.parameters())+list(net.tHead.parameters()))
    # train with source samples


    for epoch in range(num_epochs):
        img, label = training_data
        optimizer_S.zero_grad()
        t_out, p_outs = net(img)
        loss_p = [criterion(p_out, label) for p_out in p_outs]
        loss = criterion(t_out, label) + sum(loss_p)
        loss.backward()
        optimizer_S.step()

    if net.num_heads >0:
        # get pseduolabels
        test_img, _ = test_data
        pseudo_img, pseudo_test_labels = pseudo_label(net,test_img,n_t)
        L_data = training_data
        for k in range(num_psudo_steps):
            for epoch in range(num_tt_epochs):
                img, label = L_data
                optimizer_pseudo.zero_grad()
                optimizer_target.zero_grad()
                t_out, p_outs = net(img)
                loss_p = [criterion(p_out, label) for p_out in p_outs]
                loss_p.backward()
                optimizer_pseudo.step()
                
                t_out, _ = net(pseudo_img)
                loss_t = criterion(t_out, pseudo_test_labels)
                loss_t.backward()
                optimizer_target.step()

            _,p_outs = net(test_img)
            n_t = k/20*n_t
            pseudo_img, pseudo_test_labels = pseudo_label(p_outs,test_img,n_t)
            L_data = chain(L_data,zip(pseudo_img, pseudo_test_labels)) #NEED TO FIGURE OUT
    # labeling function
    # evaluate the heads

    #VALIDDATION Compute CALIBRATION

if __name__=="__main__":
    num_epochs = 10
    num_psudo_steps = 4
    num_tt_epochs = 10
    n_t = 5000
    num_heads = 2
    train(num_heads,num_epochs,num_psudo_steps,num_tt_epochs,n_t)