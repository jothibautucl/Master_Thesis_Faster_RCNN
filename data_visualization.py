import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import os
import pandas as pd
import ast


def display_histogram(hist, title, type):
    hist, bin_edges = np.histogram(hist)
    plt.figure(figsize=[10, 8])

    if type == 'train':
        col = '#0504aa'
    elif type == 'test':
        col = '#d533df'
    else:
        col = '#5cb226'

    plt.bar(bin_edges[:-1], hist, width=0.5, color=col, alpha=None)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value', fontsize=22)
    plt.ylabel('Frequency', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.title(title, fontsize=22)
    plt.savefig(os.path.join('./other/histograms', title))


def get_box_image_ratio(input_path, set):
    global imgs_temp
    ratio_surface = {}
    ratio_width_height = {}
    rgb_intensity = {}
    gray_intensity = {}

    i = 1

    imgs_record_df = pd.read_csv(input_path)
    last_row = imgs_record_df.tail(1)
    if (set == 'train'):
        imgs_temp = ast.literal_eval(last_row['test'].tolist()[0])
    elif (set == 'test'):
        imgs_temp = ast.literal_eval(last_row['train'].tolist()[0])

    for img_dict in imgs_temp:
        sys.stdout.write('\r' + 'idx=' + str(i))
        i += 1

        thepath = img_dict['filepath']
        bboxes = img_dict['bboxes'][0]
        x1, y1, x2, y2, class_name = bboxes['x1'], bboxes['y1'], bboxes['x2'], bboxes['y2'], bboxes['class']
        hbox = np.abs(int(y1) - int(y2))
        wbox = np.abs(int(x1) - int(x2))
        box_surface = hbox * wbox

        if class_name not in ratio_surface.keys():
            ratio_surface[class_name] = []
            ratio_width_height[class_name] = []
            rgb_intensity[class_name] = []
            gray_intensity[class_name] = []

        # thepath = os.path.join(data_path, filename)
        im = cv2.imread(thepath)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mean_rgb_intensity = np.mean(im)
        mean_gray_intensity = np.mean(im_gray)
        h, w, c = im.shape
        surface = h * w
        ratio_surface[class_name].append(box_surface / surface)
        ratio_width_height[class_name].append(wbox / hbox)
        rgb_intensity[class_name].append(mean_rgb_intensity)
        gray_intensity[class_name].append(mean_gray_intensity)

    return ratio_surface, ratio_width_height, rgb_intensity, gray_intensity


if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Use Faster R-CNN to detect insects.')
    parser.add_argument('--split_img_file', required=False,
                        metavar="path_to_split_image_file", default='./config/model10classes - imgs.csv',
                        help='Path to the .csv file that contains the train/test/validation split of images')
    parser.add_argument('--ratio_area', required=False, default='True',
                        metavar="True/False",
                        help="True if you want to plot the histogram of the ratio of the box area on the picture area")
    parser.add_argument('--ratio_width_height', required=False, default='True',
                        metavar="True/False",
                        help="True if you want to plot the histogram of the ratio of the box width on the box height")
    parser.add_argument('--rgb_intensity', required=False, default='True',
                        metavar="True/False",
                        help="True if you want to plot the histogram of the rgb intensity of pictures")
    parser.add_argument('--gray_intensity', required=False, default='True',
                        metavar="True/False",
                        help="True if you want to plot the histogram of the gray intensity of pictures")

    args = parser.parse_args()

    csv_file = args.split_img_file
    plot_ratio_surface = eval(args.ratio_area)
    plot_ratio_width_height = eval(args.ratio_width_height)
    plot_rgb_intensity = eval(args.rgb_intensity)
    plot_gray_intensity = eval(args.gray_intensity)


    ratio_surface, ratio_width_height, rgb_intensity, gray_intensity = get_box_image_ratio(csv_file, 'train')
    ratio_surface_test, ratio_width_height_test, rgb_intensity_test, gray_intensity_test = get_box_image_ratio(csv_file, 'test')

    if plot_ratio_surface:
        for class_name in ratio_surface.keys():
            display_histogram(ratio_surface[class_name], '[Train] Ratio box on image - ' + class_name, type='train')

        for class_name in ratio_surface_test.keys():
            display_histogram(ratio_surface_test[class_name], '[Test] Ratio box on image - ' + class_name, type='test')

    if plot_ratio_width_height:
        for class_name in ratio_width_height.keys():
            display_histogram(ratio_width_height[class_name], '[Train] Ratio width on height - ' + class_name, type='train')

        for class_name in ratio_width_height_test.keys():
            display_histogram(ratio_width_height_test[class_name], '[Test] Ratio width on height - ' + class_name,
                              type='test')

    if plot_rgb_intensity:
        for class_name in rgb_intensity.keys():
            display_histogram(rgb_intensity[class_name], '[Train] RGB Intensity - ' + class_name, type='train')

        for class_name in rgb_intensity_test.keys():
            display_histogram(rgb_intensity_test[class_name], '[Test] RGB Intensity - ' + class_name, type='test')

    if plot_gray_intensity:
        for class_name in gray_intensity.keys():
            display_histogram(gray_intensity[class_name], '[Train] Gray Intensity - ' + class_name, type='train')

        for class_name in gray_intensity_test.keys():
            display_histogram(gray_intensity_test[class_name], '[Test] Gray Intensity - ' + class_name, type='test')
