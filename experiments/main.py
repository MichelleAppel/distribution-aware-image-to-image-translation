import options
import dataset
import functions
import train_sample
import train_weight
import network
import save_results

import os

if __name__ == '__main__':
    opt = options.Options().parser.parse_args()

    ratio_A = opt.ratio_A
    dataset_A = dataset.Dataset(ratio=ratio_A, train=True, dataset=opt.dataset)
    dataloader_A = dataset.DataLoader(dataset_A, batch_size=opt.batch_size_A, shuffle=True)

    ratio_B = opt.ratio_B
    dataset_B = dataset.Dataset(ratio=ratio_B, train=True, dataset=opt.dataset)
    dataloader_B = dataset.DataLoader(dataset_B, batch_size=opt.batch_size_B, shuffle=True)

    sampled_batch_size = opt.sampled_batch_size

    # Initialize the networks
    weight_network = network.WeightNet().cuda()

    if opt.importance_sampling == 0:
      train = train_weight.Train(weight_network=weight_network,
                          dataset_A=dataset_A, 
                          dataloader_A=dataloader_A, 
                          dataset_B=dataset_B, 
                          dataloader_B=dataloader_B, 
                          opt=opt,
                          is_train=True)
    else:
      train = train_sample.Train(weight_network=weight_network,
                          dataset_A=dataset_A, 
                          dataloader_A=dataloader_A, 
                          dataset_B=dataset_B, 
                          dataloader_B=dataloader_B, 
                          opt=opt,
                          is_train=True)
    train.train()

    # Test

    testset_A = dataset.MNISTDataset(ratio=ratio_A, train=False)
    testloader_A = dataset.DataLoader(testset_A, batch_size=opt.batch_size_A, shuffle=True)

    testset_B = dataset.MNISTDataset(ratio=ratio_B, train=False)
    testloader_B = dataset.DataLoader(testset_B, batch_size=opt.batch_size_B, shuffle=True)

    if opt.importance_sampling == 0:
      test = train_weight.Train(weight_network=weight_network,
                          dataset_A=testset_A, 
                          dataloader_A=testloader_A, 
                          dataset_B=testset_B, 
                          dataloader_B=testloader_B, 
                          opt=opt,
                          is_train=False)
    else:
      test = train_sample.Train(weight_network=weight_network,
                          dataset_A=testset_A, 
                          dataloader_A=testloader_A, 
                          dataset_B=testset_B, 
                          dataloader_B=testloader_B, 
                          opt=opt,
                          is_train=False)
    test.train()



    destination = os.path.join(opt.results_dir, opt.experiment_name)
    os.makedirs(destination, exist_ok=True)
    save_results = save_results.SaveResults(destination, train, test, opt)

    save_results.plot_meansandvars()
    save_results.plot_w_loss()
    save_results.plot_L_loss()
    save_results.plot_ratios()
    save_results.plot_importances()
    save_results.plot_means()
    save_results.write_data()
