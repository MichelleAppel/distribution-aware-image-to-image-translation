import options
import dataset
import functions
import train
import network
import save_results

import os

if __name__ == '__main__':
    opt = options.Options().parser.parse_args()

    ratio_A = opt.ratio_A
    dataset_A = dataset.MNISTDataset(ratio=ratio_A)
    dataloader_A = dataset.DataLoader(dataset_A, batch_size=opt.batch_size_A, shuffle=True)

    ratio_B = opt.ratio_B
    dataset_B = dataset.MNISTDataset(ratio=ratio_B)
    dataloader_B = dataset.DataLoader(dataset_B, batch_size=opt.batch_size_B, shuffle=True)


    sampled_batch_size = opt.sampled_batch_size

    # Initialize the networks
    weight_network = network.WeightNet().cuda()

    train = train.Train(weight_network=weight_network,
                        dataset_A=dataset_A, 
                        dataloader_A=dataloader_A, 
                        dataset_B=dataset_B, 
                        dataloader_B=dataloader_B, 
                        opt=opt)
    train.train()
    

    destination = os.path.join(opt.results_dir, opt.experiment_name)
    os.makedirs(destination, exist_ok=True)
    save_results = save_results.SaveResults(destination, train, opt)

    save_results.plot_meansandvars()
    save_results.plot_w_loss()
    save_results.plot_L_loss()
    save_results.plot_ratios()
    save_results.plot_importances()
    save_results.plot_means()
    save_results.write_data()
