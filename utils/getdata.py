from lyft_dataset_sdk.lyftdataset import LyftDataset
import os
import cv2
import matplotlib.pyplot as plt


DATASET_VERSION = 'v1.02-train'
DATASET_ROOT = '/home/ezhang/lyft/v1.02-train/v1.02-train'
ARTIFACTS_FOLDER = "/home/ezhang/lyft/artifacts-2"

level5data = LyftDataset(json_path=DATASET_ROOT + "/v1.02-train", data_path=DATASET_ROOT, verbose=True)

sample_token = "0fe60c49a24a1915ecca43543de3941cecd5d6b60bc3a6d7da211bfc7bc776cc"

for i in range(20):

    fname = "./bev_validation_data/{}_target.png".format(sample_token)
    new = "./test2/{0:03}.png".format(i)

    os.system("cp {} {}".format(fname, new))

    #im = cv2.imread(fname)[:, :, 0]
    #plt.imshow(im)
    #plt.show()

    sample = level5data.get("sample", sample_token)
    sample_token = sample["next"]
