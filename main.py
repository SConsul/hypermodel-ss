import torch
from train import source_train, source_train_bootstrap, domain_adapt
from models.hydranet import HydraNet
from wilds import get_dataset
from torchvision import transforms
from evaluate import evaluate
from utils import load_model, print_and_log, get_log_files
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_domain', required=True, default={'test'}, choices={'test','val'})
    parser.add_argument('--num_pseudo_heads', type=int, required=True, default=0)   
    parser.add_argument('--epoch_offset', type=int, required=False, default=0)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_pseudo_steps', type=int, default=10)
    parser.add_argument('--num_adapt_epochs', type=int, default=2)    
    parser.add_argument('--batch_size', type=int, default=64)    
    parser.add_argument('--num_classes', type=int, default=62)    
    parser.add_argument('--orig_frac', type=float, default=1.0, help="fraction of data to be used while training, useful to set to 5e-2 for local runs")
    parser.add_argument('--threshold', type=float, default=0.9)  
    parser.add_argument('--bootstrap', type=bool, default=False)  
    parser.add_argument('--saved_model_path', default=None)  
    parser.add_argument('--da_epoch_offset', type=int, required=False, default=0)

    args = parser.parse_args()
    target_domain = args.target_domain
    num_pseudo_heads = args.num_pseudo_heads
    epoch_offset = args.epoch_offset
    num_epochs = args.num_epochs
    num_pseudo_steps = args.num_pseudo_steps
    num_adapt_epochs = args.num_adapt_epochs
    batch_size = args.batch_size
    num_classes = args.num_classes
    orig_frac = args.orig_frac 
    threshold = args.threshold
    saved_model_path = args.saved_model_path
    da_epoch_offset = args.da_epoch_offset

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
    if saved_model_path is not None:
        net.load_state_dict(torch.load(saved_model_path))

    if args.bootstrap:
        print("SOURCE TRAINING WITH BOOTSTRAPPING")
        source_train_bootstrap(net, device, train_dataset, target_dataset, batch_size,num_epochs,model_dir,log_file,epoch_offset,threshold)
    else:
        print("STANDARD SOURCE TRAINING")
        source_train(net, device, train_dataset, target_dataset, batch_size,num_epochs,model_dir,log_file,epoch_offset,threshold)
    
    if num_pseudo_heads>0:
        if epoch_offset>=num_epochs:
            print("Skip Source training by loading model weights")
            target_loss, target_accs, target_cerr, target_pHead_stats = evaluate(net,device,target_dataset,batch_size,threshold)
            com_corr_high, com_corr, com_inc, com_inc_high, disag, p_cerr = target_pHead_stats
            print_and_log(message="Tgt_Loss={:.7f}, Tgt_Acc={:.7f}, Tgt_Cal Error={:.7f}".format(
                target_loss, target_accs[0], target_cerr),log_file=log_file)
            print_and_log(message=f"Accuracies of heads = {target_accs[1]}",log_file=log_file)
            print_and_log(message="com_corr_high={:.7f}, com_corr={:.7f}, com_inc={:.7f}, com_inc_high={:.7f}, disag={:.7f}, P_Cal Error={:.7f}".format(
                com_corr_high, com_corr,com_inc,com_inc_high,disag,p_cerr),log_file=log_file)

        for k in range(1+da_epoch_offset,num_pseudo_steps+1):
            frac = min(1., k/num_pseudo_steps)
            target_dataset_frac = dataset.get_subset(target_domain, frac=orig_frac*frac,
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
            domain_adapt(net, device, train_dataset, target_dataset_frac, 
                batch_size, k, num_adapt_epochs, threshold, model_dir,log_file)

            if (k%5) ==0:
                target_loss, target_accs, target_cerr, target_pHead_stats = evaluate(net,device,target_dataset,batch_size,threshold)
                com_corr_high, com_corr, com_inc, com_inc_high, disag, p_cerr = target_pHead_stats
                print_and_log(message="Tgt_Loss={:.7f}, Tgt_Acc={:.7f}, Tgt_Cal Error={:.7f}".format(
                    target_loss, target_accs[0], target_cerr),log_file=log_file)
                print_and_log(message=f"Accuracies of heads = {target_accs[1]}",log_file=log_file)
                print_and_log(message="com_corr_high={:.7f}, com_corr={:.7f}, com_inc={:.7f}, com_inc_high={:.7f}, disag={:.7f}, P_Cal Error={:.7f}".format(
                    com_corr_high, com_corr,com_inc,com_inc_high,disag,p_cerr),log_file=log_file)
                
    log_file.close()

if __name__=="__main__":
    main()