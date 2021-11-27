import torch
from train import source_train, domain_adapt
from models.hydranet import HydraNet
from wilds import get_dataset
from torchvision import transforms
from evaluate import evaluate
from utils import load_model, print_and_log, get_log_files

def main():
    target_domain = 'test'
    epoch_offset=0
    num_epochs = 30
    num_pseudo_steps = 10
    num_adapt_epochs = 2
    num_pseudo_heads = 3
    batch_size = 64
    num_classes = 62
    orig_frac = 1 # fraction of data to be used while training
                  # useful to set to 5e-2 for local runs
    threshold = 0.9

    print("Num heads=",num_pseudo_heads)
    if num_pseudo_heads>0:
        log_loc = f"logs/ssl_{num_pseudo_heads}"
    else:
        log_loc = f"logs/baseline"

    model_path = f"model_weights/num_heads_{num_pseudo_heads}"
    model_dir, log_file = get_log_files(model_path,log_loc)


    dataset = get_dataset(dataset='fmow_mini', download=False)
    train_dataset = dataset.get_subset('train', frac=orig_frac,
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    target_dataset = dataset.get_subset(target_domain, frac=orig_frac,
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    net = HydraNet(num_heads=num_pseudo_heads, num_features=1024,
        num_classes=num_classes,pretrained=True)
    net = net.to(device)
    net.load_state_dict(torch.load("model_weights/num_heads_3/2021-11-27-01-34-29/source_trained_epoch_30.pt"))

    #source_train(net, device, train_dataset, target_dataset, batch_size,num_epochs,model_dir,log_file,epoch_offset)
    
    if num_pseudo_heads>0:
        for k in range(1,num_pseudo_steps+1):
            frac = min(1., k/num_pseudo_steps)
            target_dataset_frac = dataset.get_subset(target_domain, frac=orig_frac*frac,
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
            domain_adapt(net, device, train_dataset, target_dataset_frac, 
                batch_size, k, num_adapt_epochs, threshold, model_dir,log_file)

            if (k%5) ==0:
                tatget_loss, target_acc, target_cerr, target_pHead_stats = evaluate(net,device,target_dataset,batch_size)
                com_corr_high, com_corr, com_inc, com_inc_high, disag, p_cerr = target_pHead_stats
                print_and_log(message="Target_Loss={:.7f}, Target_Acc={:.7f}, Target_Cal Error={:.7f}".format(
                    tatget_loss, target_acc,target_cerr),log_file=log_file)
                print_and_log(message="com_corr_high={:.7f}, com_corr={:.7f}, com_inc={:.7f}, com_inc_high={:.7f}, disag={:.7f}, P_Cal Error={:.7f}".format(
                    com_corr_high, com_corr,com_inc,com_inc_high,disag,p_cerr),log_file=log_file)

    # test_dataset = dataset.get_subset('test',transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    # test_loss, test_acc, test_cerr, test_pHead_stats = evaluate(net,device,test_dataset,batch_size)
    # com_corr_high, com_corr, com_inc, com_inc_high, disag, p_cerr = test_pHead_stats
    # print_and_log(message="Test Loss={}, Test Acc={}, Test Calib Error={}".format(
    #     test_loss, test_acc, test_cerr), log_file=log_file)
    # print_and_log(message="com_corr_high={:.7f}, com_corr={:.7f}, com_inc={:.7f}, com_inc_high={:.7f}, disag={:.7f}, P_Cal Error={:.7f}".format(
    #             com_corr_high, com_corr,com_inc,com_inc_high,disag,p_cerr),log_file=log_file)
    log_file.close()

if __name__=="__main__":
    main()