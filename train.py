import torch 
import torch.nn as nn
from itertools import chain
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from wilds.common.data_loaders import get_train_loader,get_eval_loader
from models.hydranet import HydraNet
from datasets.dummy_datasets import get_dummy
from utils import get_inf_iterator, save_model, get_inf_iterator
from evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def pseudo_label(net, device, target_dataset, batch_size, threshold = 0.9):
    # need to make this online, GPU is giving errors.
    assert net.num_heads>1, 'Number of pseudo-heads must be 2 or more'
    net.eval()
    total_outs = None
    target_dataloader = DataLoader(target_dataset,batch_size=batch_size, shuffle=False)
    for img,_,_ in target_dataloader:
        _,p_outs = net(img.to(device)) #p_outs is of list of len num_heads of Tensors of size (BxC)
        print(f'length: {len(p_outs)}, shape: {p_outs[0].shape}')
        if total_outs is None:
            total_outs = p_outs
        else:
            for i in range(net.num_heads):
                total_outs[i] = torch.cat((total_outs[i],p_outs[i]),0) # total outs is a list of length num_heads of Tensors of size (N,C)

    if net.num_heads>1:
        _, pred_0 = torch.max(total_outs[0],1)
        mask_common = torch.ones(total_outs.shape)
        for i in range(1,net.num_heads):
            _, pred = torch.max(total_outs[i],1)
            mask_common = mask_common * torch.eq(pred_0,pred)
        equal_idx = torch.nonzero(mask_common)

        max_pred, _ = torch.max(total_outs[0],1)
        for i in range(1,net.num_heads):
            pred,_ = torch.max(total_outs[i],1)
            max_pred = torch.max(torch.stack([max_pred, pred], 1), 1)
        
        filtered_idx = torch.nonzero(max_pred > torch.log(threshold)).squeeze()
        _, pseudo_labels = torch.max(total_outs[0][filtered_idx, :], 1)
        excerpt = equal_idx[filtered_idx]
        print(excerpt)

    # target_dataset_labelled = get_dummy(target_dataset,excerpt,
    #     pseudo_labels,need_dataset=True)
    # return target_dataset_labelled

def pre_train(net,device, train_dataset,val_dataset,batch_size,num_epochs,epoch_offset=0):
    criterion = nn.CrossEntropyLoss()
    optimizer_enc = torch.optim.Adam(net.enc.parameters(), lr=1e-4)
    scheduler_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer_enc, gamma=0.96)
    optimizer_tHead = torch.optim.Adam(net.tHead.parameters(), lr=1e-3)
    scheduler_tHead = torch.optim.lr_scheduler.ExponentialLR(optimizer_tHead, gamma=0.96)
    if net.num_heads > 0:
        optimizer_pHeads = torch.optim.Adam(net.pHeads.parameters(), lr=1e-3)
        scheduler_pHeads = torch.optim.lr_scheduler.ExponentialLR(optimizer_pHeads, gamma=0.96)


    train_loader = get_train_loader('standard', train_dataset, batch_size=batch_size)
    writer = SummaryWriter()

    val_loss, val_acc, val_cerr = evaluate(net,device,val_dataset,batch_size)
    # print(val_cerr.shape)
    # train with source samples
    for epoch in range(epoch_offset,num_epochs):
        net.train()
        Loss_T = 0.0
        Loss_P = 0.0
        for img, label, _ in train_loader:
            img = Variable(img.to(device))
            label = Variable(label.to(device))
            
            optimizer_enc.zero_grad()
            optimizer_tHead.zero_grad()


            t_out, p_outs = net(img)
            loss_t = criterion(t_out, label)
            loss = loss_t 
            if net.num_heads>0:
                optimizer_pHeads.zero_grad()
                loss_p = [criterion(p_out, label) for p_out in p_outs]
                loss += sum(loss_p)
            loss.backward()

            optimizer_enc.step()
            optimizer_tHead.step()

            Loss_T +=loss_t
            if net.num_heads>0:
                optimizer_pHeads.step()
                Loss_P += sum(loss_p)/len(loss_p)
            # else:
            #     print("train_T_Loss={:.5f}".format(loss_t))
            #     print(type(Loss_T))
        scheduler_enc.step()
        scheduler_tHead.step()
        if net.num_heads>0:
            scheduler_pHeads.step()
            Loss_P = Loss_P/len(train_loader)

        Loss_T = Loss_T/len(train_loader)
        
        save_model(net,"checkpoints/ssl_2/source_trained_{}_{}.pt".format(
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), epoch+1))

        if net.num_heads>0:  
            print("Epoch {}/{}: train_T_Loss={:.5f}, train_P_Loss={:.5f}".format(epoch+1,num_epochs,Loss_T,Loss_P))
        else:
            print("Epoch {}/{}: train_T_Loss={:.5f}".format(epoch+1,num_epochs,Loss_T))
            writer.add_scalar("Loss/train", Loss_T.item(), epoch)
        if(epoch+1)%5 ==0:
            val_loss, val_acc, val_cerr = evaluate(net,device,val_dataset,batch_size)
            print("Epoch {}/{}: Val_Loss={:.5f}, Val_Acc={:.5f}, Val_Cal Error={:.5f}".format(epoch+1,num_epochs,val_loss, val_acc,val_cerr))

def domain_adapt(net, device, source_dataset, target_dataset, 
    batch_size, frac, num_adapt_epochs=1):
    if net.num_heads == 0:
        return

    # get pseudolabels
    target_dataset_labelled = pseudo_label(net, device, target_dataset, batch_size)
    merged_dataset = ConcatDataset([source_dataset, target_dataset_labelled])

    criterion = nn.CrossEntropyLoss()
    optimizer_pseudo = torch.optim.Adam(list(net.enc.parameters())+list(net.pHeads.parameters()),
        lr=1e-4)
    optimizer_target = torch.optim.Adam(list(net.enc.parameters())+list(net.tHead.parameters()),
        lr=1e-4)

    net.train()
    merged_dataloader = DataLoader(merged_dataset,batch_size=batch_size, shuffle=True)
    target_dataloader_labelled = get_inf_iterator(DataLoader(target_dataset_labelled,
        batch_size=batch_size, shuffle=True))

    for adapt_step in range(num_adapt_epochs):
        for img, label, _ in merged_dataloader:
            pseudo_img, pseudo_target_labels = next(target_dataloader_labelled)
            img = Variable(img.to(device))
            label = Variable(label.to(device))
            pseudo_img = Variable(pseudo_img)
            pseudo_target_labels = Variable(pseudo_target_labels)

            optimizer_pseudo.zero_grad()
            optimizer_target.zero_grad()
            # Train F, F_heads with merged dataset
            t_out, p_outs = net(img)
            loss_heads = []
            for p_out in p_outs:
                loss_p = criterion(p_out, label)
                loss_p.backward()
                loss_heads.append(loss_p)
            optimizer_pseudo.step()
            
            # Train F, F_t with pseudo-labelled target data
            t_out, _ = net(pseudo_img)
            loss_t = criterion(t_out, pseudo_target_labels)
            loss_t.backward()
            optimizer_target.step()
    
        save_model(net,"checkpoints/ssl_{}/ssl_trained_{}_frac_{}_epoch_{}.pt".format(
                net.num_heads, 
                datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), 
                int(frac*10), 
                adapt_step+1))

# def domain_adapt(net, device, source_dataset, target_dataset, batch_size, num_pseudo_steps, num_adapt_epochs, n_t):
#     # domain adaptation
#     if net.num_heads == 0:
#         return

#     # get pseudolabels
#     target_dataset_labelled = pseudo_label(net,target_dataset,n_t)
#     merged_dataset = ConcatDataset([source_dataset, target_dataset_labelled])

#     criterion = nn.CrossEntropyLoss()
#     optimizer_pseudo = torch.optim.Adam(list(net.enc.parameters())+list(net.pHeads.parameters()),
#         lr=1e-4, weight_decay=0.96)
#     optimizer_target = torch.optim.Adam(list(net.enc.parameters())+list(net.tHead.parameters()),
#         lr=1e-4, weight_decay=0.96)

#     for k in range(num_pseudo_steps):
#         net.train()
#         merged_dataloader = DataLoader(merged_dataset,batch_size=batch_size, shuffle=True)
#         target_dataloader_labelled = get_inf_iterator(DataLoader(target_dataset_labelled,batch_size=batch_size, shuffle=True))

#         for adapt_step in range(num_adapt_epochs):
#             for i_tt, (img, label, _) in enumerate(merged_dataloader):
#                 pseudo_img, pseudo_target_labels = next(target_dataloader_labelled)
#                 img = Variable(img.to(device))
#                 label = Variable(label.to(device))
#                 pseudo_img = Variable(pseudo_img)
#                 pseudo_target_labels = Variable(pseudo_target_labels)

#                 optimizer_pseudo.zero_grad()
#                 optimizer_target.zero_grad()
#                 # Train F, F_heads with merged dataset
#                 t_out, p_outs = net(img)
#                 loss_heads = []
#                 for p_out in p_outs:
#                     loss_p = criterion(p_out, label)
#                     loss_p.backward()
#                     loss_heads.append(loss_p)
#                 optimizer_pseudo.step()
                
#                 # Train F, F_t with pseudo-labelled target data
#                 t_out, _ = net(pseudo_img)
#                 loss_t = criterion(t_out, pseudo_target_labels)
#                 loss_t.backward()
#                 optimizer_target.step()

#             n_t = adapt_step*len(source_dataset)//20
#             target_dataset_labelled = pseudo_label(net,target_dataset,n_t)
#             merged_dataset = ConcatDataset([source_dataset, target_dataset_labelled])

# labeling function
# evaluate the heads

#VALIDDATION Compute CALIBRATION

