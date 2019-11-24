import os
import matplotlib.pyplot as plt
from src import memory
import cv2
import numpy as np
from tqdm import tqdm

def get_blue_histograms(path: str, mask):
    histograms = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0], mask, [180], [0, 180])  # HUE
        # hist = cv2.calcHist([img], [0], mask, [180], [0, 180]) #BLUE
        # hist = cv2.calcHist([cv2.cvtColor(img,cv2.COLOR_BGR2HSV)], [0, 1], mask, [180, 256], [0, 180, 0, 256]) #HUE AND SATURATION

        histograms.append(hist)

    return histograms

@memory.cache()
def compare_histograms(path: str, blue_hist_list, mask, threshold=2100):
    histograms = []
    images = []
    for file in os.listdir(path):
        blue = False
        img = cv2.imread(os.path.join(path , file))
        hist = cv2.calcHist([cv2.cvtColor(img,cv2.COLOR_BGR2HSV)], [0], mask, [180], [0, 180]) #HUE
        for blue_hist in blue_hist_list:
          shirt_flag = cv2.compareHist(blue_hist, hist, cv2.HISTCMP_INTERSECT)
          if shirt_flag > threshold:
            blue = True
        if blue:
          histograms.append(1)
          images.append(file)
        else:
          histograms.append(0)
    return np.array(histograms), images


def ploting_blue_shirts(path: str, list_blue_shirts):

    fig = plt.figure(figsize=(15, 15))

    columns = 10
    rows = 10
    counter = 0

    for path_img in tqdm(list_blue_shirts):
        counter += 1
        img = cv2.imread(os.path.join(path, path_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(rows, columns, counter)
        #ax.title.set_text(path_img)
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.imshow(img)
        plt.axis('off')
        if counter == (columns*rows):
            counter = 0
            plt.show()
            plt.close()
            fig = plt.figure(figsize=(15, 15))


def main():
    im_path_train = '../dataset/shirts/train'

    im_path_blue = '../dataset/shirts/blue'

    #Declaring a region of interest where most of the region of the shirt should be
    mask = np.zeros((224, 224), np.uint8)
    mask[75:150, 75:150] = 255

    #Get list of blue shirts histograms
    blue_hist_list = get_blue_histograms(im_path_blue, mask)

    #Comparing each histogram of  train with each element of the blue shirt histogram list
    labels, images = compare_histograms(im_path_train, blue_hist_list, mask)
    print(np.unique(labels, return_counts=True))
    print(images)

    ploting_blue_shirts(im_path_train, images)

if __name__ == '__main__':
    main()