import os
import torch 
import torch.nn as nn
from itertools import chain
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from wilds.common.data_loaders import get_train_loader,get_eval_loader
from models.hydranet import HydraNet
from datasets.dummy_datasets import get_dummy
from utils import get_inf_iterator, save_model, get_inf_iterator, print_and_log
from evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def pseudo_label2(net, device, target_dataset, batch_size, threshold=0.9):
    assert net.num_heads>1, 'Number of pseudo-heads must be 2 or more'
    net.eval()
    target_dataloader = DataLoader(target_dataset,batch_size=batch_size, 
        shuffle=False)
    excerpt_final = torch.IntTensor()
    pseudo_labels_final = torch.IntTensor()
    for batch_counter, (img,_,_) in enumerate(target_dataloader):
        _, p_outs = net(img.to(device)) # num_heads x batch_size x num_classes
        p_outs = [p_out.data.cpu() for p_out in p_outs]
        _, pred_0 = torch.max(p_outs[0], 1)
        # print(pred_0.shape)
        mask_common = torch.ones(p_outs[0].shape[0]) # batch_size, 
        for i in range(1,net.num_heads):
            _, pred = torch.max(p_outs[i],1)
            # print(pred.shape)
            mask_common = mask_common * torch.eq(pred_0,pred)
        equal_idx = torch.nonzero(mask_common)

        max_conf, _ = torch.max(p_outs[0],1)
        for i in range(1,net.num_heads):
            conf,_ = torch.max(p_outs[i],1)
            max_conf,_ = torch.max(torch.stack([max_conf, conf], 1), 1)
        
        mask_conf = max_conf > threshold
        mask_final = max_conf*mask_common
        excerpt = torch.nonzero(mask_final).squeeze(-1)
        # print(excerpt)
        _, pseudo_labels = torch.max(p_outs[0][excerpt, :], 1)

        excerpt_final = torch.cat([excerpt_final, excerpt+batch_counter*batch_size])
        pseudo_labels_final = torch.cat([pseudo_labels_final, pseudo_labels])
    if excerpt_final.shape[0] == 0:
        is_empty = True
    else:
        is_empty = False
    target_dataset_labelled = get_dummy(target_dataset, excerpt_final,
        pseudo_labels_final, need_dataset=True)
    return target_dataset_labelled, is_empty


# def pseudo_label(net, device, target_dataset, batch_size, threshold = 0.9):
#     # need to make this online, GPU is giving errors.
#     assert net.num_heads>1, 'Number of pseudo-heads must be 2 or more'
#     net.eval()
#     total_outs = None
#     target_dataloader = DataLoader(target_dataset,batch_size=batch_size, shuffle=False)
#     for img,_,_ in target_dataloader:
#         _,p_outs = net(img.to(device)) #p_outs is of list of len num_heads of Tensors of size (BxC)
#         print(f'length: {len(p_outs)}, shape: {p_outs[0].shape}')
#         if total_outs is None:
#             total_outs = p_outs
#         else:
#             for i in range(net.num_heads):
#                 total_outs[i] = torch.cat((total_outs[i],p_outs[i]),0) # total outs is a list of length num_heads of Tensors of size (N,C)

#     if net.num_heads>1:
#         _, pred_0 = torch.max(total_outs[0],1)
#         mask_common = torch.ones(total_outs.shape)
#         for i in range(1,net.num_heads):
#             _, pred = torch.max(total_outs[i],1)
#             mask_common = mask_common * torch.eq(pred_0,pred)
#         equal_idx = torch.nonzero(mask_common)

#         max_pred, _ = torch.max(total_outs[0],1)
#         for i in range(1,net.num_heads):
#             pred,_ = torch.max(total_outs[i],1)
#             max_pred = torch.max(torch.stack([max_pred, pred], 1), 1)
        
#         filtered_idx = torch.nonzero(max_pred > torch.log(threshold)).squeeze()
#         _, pseudo_labels = torch.max(total_outs[0][filtered_idx, :], 1)
#         excerpt = equal_idx[filtered_idx]
#         print(excerpt)

#     target_dataset_labelled = get_dummy(target_dataset,excerpt,
#         pseudo_labels,need_dataset=True)
#     return target_dataset_labelled

def source_train(net,device, train_dataset,val_dataset,batch_size,num_epochs,
    model_dir,log_file,epoch_offset=0):
    criterion = nn.CrossEntropyLoss()
    optimizer_enc = torch.optim.Adam(net.enc.parameters(), lr=1e-4)
    scheduler_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer_enc, gamma=0.96)
    optimizer_tHead = torch.optim.Adam(net.tHead.parameters(), lr=1e-3)
    scheduler_tHead = torch.optim.lr_scheduler.ExponentialLR(optimizer_tHead, gamma=0.96)
    if net.num_heads > 0:
        optimizer_pHeads = torch.optim.Adam(net.pHeads.parameters(), lr=1e-3)
        scheduler_pHeads = torch.optim.lr_scheduler.ExponentialLR(optimizer_pHeads, gamma=0.96)


    train_loader = DataLoader(train_dataset,batch_size=batch_size, 
        shuffle=True, drop_last=True)

    # get_train_loader('standard', train_dataset, batch_size=batch_size)
    writer = SummaryWriter()

    # val_loss, val_acc, val_cerr = evaluate(net,device,val_dataset,batch_size)
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
            Loss_T +=loss_t
            # print(f"loss_t: {loss_t}")
            if net.num_heads>0:
                optimizer_pHeads.zero_grad()
                loss_p = [criterion(p_out, label) for p_out in p_outs]
                avg_loss_p = sum(loss_p)/len(loss_p)
                loss += avg_loss_p
                Loss_P += avg_loss_p

            loss.backward()

            optimizer_enc.step()
            optimizer_tHead.step()
            if net.num_heads>0:
                optimizer_pHeads.step()
                
            # else:
            #     print("train_T_Loss={:.7f}".format(loss_t))
            #     print(type(Loss_T))
        scheduler_enc.step()
        scheduler_tHead.step()
        if net.num_heads>0:
            scheduler_pHeads.step()
            Loss_P = Loss_P/len(train_loader)

        Loss_T = Loss_T/len(train_loader)

        save_model(net,os.path.join(model_dir,f"source_trained_epoch_{epoch+1}.pt"))

        print_and_log(message="Epoch {}/{}: train_T_Loss={:.7f}, train_P_Loss={:.7f}".format(
            epoch+1,num_epochs,Loss_T,Loss_P),
            log_file=log_file)
        writer.add_scalar("Loss/train", Loss_T.item(), epoch)
        if(epoch+1)%5 ==0:
            val_loss, val_acc, val_cerr, val_pHead_stats = evaluate(net,device,val_dataset,batch_size)
            com_corr_high, com_corr, com_inc, com_inc_high, disag, p_cerr = val_pHead_stats
            print_and_log(message="Val_Loss={:.7f}, Val_Acc={:.7f}, Val_Cal Error={:.7f}".format(
                val_loss, val_acc,val_cerr),log_file=log_file)
            print_and_log(message="com_corr_high={:.7f}, com_corr={:.7f}, com_inc={:.7f}, com_inc_high={:.7f}, disag={:.7f}, P_Cal Error={:.7f}".format(
                com_corr_high, com_corr,com_inc,com_inc_high,disag,p_cerr),log_file=log_file)
                
def domain_adapt(net, device, source_dataset, target_dataset, 
    batch_size, k_step, num_adapt_epochs, threshold, model_dir,log_file):
    if net.num_heads == 0:
        return

    # get pseudolabels
    target_dataset_labelled, is_empty = pseudo_label2(net, device, target_dataset, batch_size, threshold)
    if is_empty:
        print("No pseudo labelled points")
        return 
    merged_dataset = ConcatDataset([source_dataset, target_dataset_labelled])

    criterion = nn.CrossEntropyLoss()

    optimizer_enc = torch.optim.Adam(net.enc.parameters(), lr=1e-4*(0.96)**30)
    scheduler_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer_enc, gamma=0.96)
    optimizer_tHead = torch.optim.Adam(net.tHead.parameters(), lr=1e-3*(0.96)**30)
    scheduler_tHead = torch.optim.lr_scheduler.ExponentialLR(optimizer_tHead, gamma=0.96)
    optimizer_pHeads = torch.optim.Adam(net.pHeads.parameters(), lr=1e-3*(0.96)**30)
    scheduler_pHeads = torch.optim.lr_scheduler.ExponentialLR(optimizer_pHeads, gamma=0.96)


    # optimizer_pseudo = torch.optim.Adam(list(net.enc.parameters())+list(net.pHeads.parameters()),
    #     lr=1e-4)
    # optimizer_target = torch.optim.Adam(list(net.enc.parameters())+list(net.tHead.parameters()),
    #     lr=1e-4)

    net.train()
    merged_dataloader = DataLoader(merged_dataset,batch_size=batch_size, 
        shuffle=True, drop_last=True)
    target_dataloader_labelled = get_inf_iterator(DataLoader(target_dataset_labelled,
        batch_size=batch_size, shuffle=True, drop_last=True))

    for epoch in range(num_adapt_epochs):
        Loss_P = 0.0
        Loss_T = 0.0
        for img, label, _ in merged_dataloader:
            pseudo_img, pseudo_target_labels, _ = next(target_dataloader_labelled)
            img = Variable(img.to(device))
            label = Variable(label.to(device))
            pseudo_img = Variable(pseudo_img.to(device))
            pseudo_target_labels = Variable(pseudo_target_labels.to(device))

            optimizer_enc.zero_grad()
            optimizer_pHeads.zero_grad()
            # Train F, F_heads with merged dataset
            t_out, p_outs = net(img)
            loss_p = [criterion(p_out, label) for p_out in p_outs]
            loss = sum(loss_p)/len(loss_p)
            loss.backward()
            Loss_P +=loss
            optimizer_enc.step()
            optimizer_pHeads.step()

            # Train F, F_t with pseudo-labelled target data
            optimizer_enc.zero_grad()
            optimizer_tHead.zero_grad()
            t_out, _ = net(pseudo_img)
            loss_t = criterion(t_out, pseudo_target_labels)
            loss_t.backward()
            optimizer_enc.step()
            optimizer_tHead.step()
            Loss_T+=loss_t
        scheduler_enc.step()
        scheduler_tHead.step()
        scheduler_pHeads.step()

        Loss_P = Loss_P/len(merged_dataloader)
        Loss_T = Loss_T/len(merged_dataloader)

        save_model(net,os.path.join(model_dir,f"domain_adapt_step_{k_step}_epoch_{epoch+1}.pt"))
        print_and_log(message="Domain Adapt Step {} Epoch {}/{}: Target Loss={:.7f}, Pseudo Loss={:.7f}".format(
            k_step, epoch+1,num_adapt_epochs,Loss_T, Loss_P),
            log_file=log_file)

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

