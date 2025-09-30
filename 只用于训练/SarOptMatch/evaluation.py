import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import SarOptMatch


def get_predictions(data, heatmaps, batch_size = 4):
    """
    Retrieves the predicted offset coordinates on the validation set.
    """
    preds = []
    iter_val = iter(data)
    for i in range(len(heatmaps)):
        v = iter_val.get_next()[1]
        for j in range(batch_size):
            preds.append((np.unravel_index(v[j].numpy().argmax(), v[j].numpy().shape),np.unravel_index(heatmaps[i][j].argmax(), heatmaps[i][j].shape)[:2]))
    return preds


def print_perf_table(euc_dist):
    print("Matching Accuracy")
    print("<= 1 pixel: " +str(np.where(euc_dist <= 1, 1,0).sum()/len(euc_dist)) )
    print("<= 2 pixel: " +str(np.where(euc_dist <= 2, 1,0).sum()/len(euc_dist)) )
    print("<= 3 pixel: " +str(np.where(euc_dist <= 3, 1,0).sum()/len(euc_dist)) )

    print("<= 5 pixel: " +str(np.where(euc_dist <= 5, 1,0).sum()/len(euc_dist)) )
    # print("<= 10 pixel: " +str(np.where(euc_dist <= 10, 1,0).sum()/len(euc_dist)) )
    # print("<= 20 pixel: " +str(np.where(euc_dist <= 20, 1,0).sum()/len(euc_dist)) )
    # print("<= 25 pixel: " +str(np.where(euc_dist <= 25, 1,0).sum()/len(euc_dist)) )
    # print("<= 30 pixel: " + str(np.where(euc_dist <= 30, 1, 0).sum() / len(euc_dist)))
    # print("<= 50 pixel: " +str(np.where(euc_dist <= 50, 1,0).sum()/len(euc_dist)) )
    print(f"Average Euclidean Distance: {euc_dist.mean()}")
    
def print_results(validation_data, heatmaps):
    bs = 4
    heatmaps_reshaped = heatmaps.reshape([heatmaps.shape[0]//bs,bs,65,65,1])
    result = get_predictions(validation_data,heatmaps_reshaped,bs)
    true, predicted = zip(*result)
    euc_dists = np.linalg.norm(np.array(true)-np.array(predicted),axis=1)
    mean_dist = euc_dists.mean()
    print_perf_table(euc_dists)
    # rmse = np.sqrt(np.mean(np.square(euc_dists)))
    #
    # print(f"Root Mean Squared Error (RMSE): {rmse}")


    return euc_dists


