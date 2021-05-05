import sklearn.metrics
import os

from libraries import *

def preprocess_img(img, noise_reduction, histogram_equalization, gamma_correction):
    if noise_reduction is None:
        pass
    elif noise_reduction == "box_filter":
        img = cv2.boxFilter(img, -1, C.noise_reduction_shape, normalize=True)
    elif noise_reduction == "gaussian":
        img = cv2.GaussianBlur(img,(7,7), 0)

    if histogram_equalization:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    if gamma_correction:
        img = adjust_gamma(img, gamma=C.gamma_value)

    return img

def init_models():
    num_features = 512

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = rpn_layer(shared_layers, num_anchors)

    classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    class_mapping = C.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)

    return model_rpn, class_mapping, model_classifier_only


def draw_box_on_images(noise_reduction, histogram_equalization, gamma_correction):
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    if from_csv:
        imgs_record_df = pd.read_csv(imgs_record_path)
        last_row = imgs_record_df.tail(1)
        test_imgs_temp = ast.literal_eval(last_row['test'].tolist()[0])
        test_imgs = []
        for img_dict in test_imgs_temp:
            test_imgs.append(img_dict['filepath'])
    else:
        test_imgs_temp = os.listdir(data_test_path)
        test_imgs = []
        for img_name in test_imgs_temp:
            test_imgs.append(os.path.join(data_test_path, img_name))

    # imgs_path = []
    # for i in range(10):
    #     idx = np.random.randint(len(test_imgs))
    #     imgs_path.append(test_imgs[idx])
    #
    # all_imgs = []
    #
    # classes = {}

    #bbox_threshold = 0.822
    bbox_threshold = 0.6

    # for idx, img_name in enumerate(imgs_path):
    length = len(test_imgs)
    print(length)
    for idx, filepath in enumerate(test_imgs):
        print("Progression : " + str(idx + 1) + "/" + str(length))
        if not filepath.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(filepath)
        st = time.time()

        img = cv2.imread(filepath)

        initial_img = img.copy()

        img = preprocess_img(img, noise_reduction, histogram_equalization, gamma_correction)

        X, ratio = format_img(img, C)

        X = np.transpose(X, (0, 2, 3, 1))

        # get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        [Y1, Y2, F] = model_rpn.predict(X)

        # Get bboxes by applying NMS
        # R.shape = (300, 4)
        R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[1]):
                # Ignore 'bg' class
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue


                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
            #Has to be < overlap_threshold used in the return value of rpn
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                # Calculate real coordinates on original image
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(initial_img, (real_x1, real_y1), (real_x2, real_y2),
                              (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),
                              4)

                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                all_dets.append((key, 100 * new_probs[jk]))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)

                cv2.rectangle(initial_img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 1)
                cv2.rectangle(initial_img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(initial_img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        print('Elapsed time = {}'.format(time.time() - st))
        print(all_dets)
        plt.figure(figsize=(10, 10))
        plt.grid()
        plt.imshow(cv2.cvtColor(initial_img, cv2.COLOR_BGR2RGB))
        filename = filepath.split(".")[0].split("/")[-1].split("\\")[-1]
        plt.savefig('other/box_figure/{}/{}.jpg'.format(processed_directory, filename))

        print(class_mapping)


def plot_precision_recall(precision, recall, thresholds, class_name):
    plt.figure()
    plt.plot(recall, precision, lw=2)
    plt.title('[' + class_name + ']' + ' Precision - Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.savefig("./graphs/{}_prec-rec_{}".format(str(args.model_name), str(class_name)))

    plt.figure()
    plt.xlabel('Thresholds')
    plt.ylabel('AP metrics')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot(thresholds, precision[:-1], label='precision')
    plt.plot(thresholds, recall[:-1], label='recall')
    plt.legend()
    plt.title('[' + class_name + ']' + ' AP thresholds')
    plt.savefig("./graphs/{}_AP_thresh_{}".format(str(args.model_name), str(class_name)))

def plot_roc(fpr, tpr, class_name, thresholds):
    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.title('[' + class_name + ']' + 'ROC')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.savefig("./graphs/{}_roc_{}".format(args.model_name, str(class_name)))

    # plt.figure()
    # plt.xlabel('Thresholds')
    # plt.ylabel('AP metrics')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.plot(thresholds, fpr)
    # plt.plot(thresholds, tpr)
    # plt.legend('fpr', 'tpr')
    # plt.title('[' + class_name + ']' + ' ROC thresholds')
    # plt.savefig("./graphs/test_{}_ROC_thresh_{}".format(str(args.model_name), str(class_name)))

def maximise_F_score(precision, recall, threshold):
    best_F_score = 0
    best_threshold = -1
    best_precision = -1
    best_recall = -1
    for i in range(len(threshold)):
        F_score = 2 * precision[i]*recall[i]/(precision[i] + recall[i])
        if F_score > best_F_score:
            best_threshold = threshold[i]
            best_precision = precision[i]
            best_recall = recall[i]
            best_F_score = F_score

    return round(best_threshold, 3), round(best_F_score, 3), round(best_precision, 3), round(best_recall, 3)

def accuracy(noise_reduction, histogram_equalization, gamma_correction):
    if from_csv:
        imgs_record_df = pd.read_csv(imgs_record_path)
        last_row = imgs_record_df.tail(1)
        test_imgs = ast.literal_eval(last_row['test'].tolist()[0])
    else:
        test_imgs, _, _ = get_data(test_path, data_test_path)

    T = {}
    P = {}
    T_all = np.empty((0,nbr_classes), int)
    P_all = np.empty((0,nbr_classes), float)
    # T_all = []
    # P_all = []
    mAPs = []
    mROC_AUCs = []
    for idx, img_data in enumerate(test_imgs):
        print('{}/{}'.format(idx, len(test_imgs)))
        st = time.time()
        filepath = img_data['filepath']

        print(filepath)
        img = cv2.imread(filepath)

        img = preprocess_img(img, noise_reduction, histogram_equalization, gamma_correction)

        X, fx, fy = format_img_map(img, C)

        # Change X (img) shape from (1, channel, height, width) to (1, height, width, channel)
        X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        all_probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            # Calculate all classes' bboxes coordinates on resized image (300, 400)
            # Drop 'bg' classes bboxes
            for ii in range(P_cls.shape[1]):

                # If class name is 'bg', continue
                if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                # Get class name
                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                    all_probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
                all_probs[cls_name].append(list(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])
            all_prob = np.array(all_probs[key])

            # Apply non-max-suppression on final bboxes to get the output bounding boxes
            new_boxes, new_probs, new_all_prob = non_max_suppression_fast_with_all_probs(bbox, np.array(probs[key]), all_prob, overlap_thresh=0.2)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk], 'all_probs': new_all_prob[jk]}
                all_dets.append(det)

        print('Elapsed time = {}'.format(time.time() - st))
        t, p = get_map(all_dets, img_data['bboxes'], (fx, fy)) #p contient les proba de prédiction des classes
        T_all_for_image, P_all_for_image = get_map_all(all_dets, img_data['bboxes'], (fx, fy), class_mapping)
        for T_all_box in T_all_for_image:
            T_all = np.append(T_all, np.array([T_all_box]), axis=0)
        for P_all_box in P_all_for_image:
            P_all = np.append(P_all, np.array([P_all_box]), axis=0)
        for key in t.keys():
            if key not in T:
                T[key] = []
                P[key] = []
            T[key].extend(t[key]) #C'est cumulatif, on garde les prédictions des images précédemment considérées dans le test
            P[key].extend(p[key])
        all_aps = []
        all_roc_aucs = []
        for key in T.keys():
            ap = average_precision_score(T[key], P[key])
            # roc_auc = roc_auc_score(T[key], P[key])
            print('{} AP: {}'.format(key, ap))
            # print('{} ROC AUC: {}'.format(key, roc_auc))
            all_aps.append(ap)
            # all_roc_aucs.append(roc_auc)
        print('mAP = {}'.format(np.mean(np.array(all_aps)))) #Mean on all classes
        # print('mROC_AUC = {}'.format(np.mean(np.array(all_roc_aucs)))) #Mean on all classes
        mAPs.append(np.mean(np.array(all_aps)))
        # mROC_AUCs.append(np.mean(np.array(all_roc_aucs)))

    # T_all = np.concatenate(T_all, axis=0)
    # P_all = np.concatenate(P_all, axis=0)
    T_all = T_all.ravel()
    P_all = P_all.ravel()

    print()
    print('mean average precision:', np.nanmean(np.array(mAPs)))
    # print('mean Area Under the Receiver Operating Characteristic Curve:', np.nanmean(np.array(mROC_AUCs)))

    mAP = [mAP for mAP in mAPs if str(mAP) != 'nan']
    # mROC_AUC = [mROC_AUC for mROC_AUC in mROC_AUCs if str(mROC_AUC) != 'nan']
    mean_average_prec = np.nanmean(np.array(mAP))
    # mean_roc_auc = np.nanmean(np.array(mROC_AUC))
    print('The mean average precision is %0.3f' % (mean_average_prec))
    # print('The mean Area Under the Receiver Operating Characteristic Curve is %0.3f' % (mean_roc_auc))

    optimal_data = dict()
    threshold_list = list()
    F_score_list = list()
    precision_list = list()
    recall_list = list()
    for key in T.keys():
        print(key)
        precision, recall, thresholds = precision_recall_curve(T[key], P[key])
        best_threshold, best_F_score, best_precision, best_recall = maximise_F_score(precision, recall, thresholds)
        plot_precision_recall(precision, recall, thresholds, key)
        fpr, tpr, thresholds = roc_curve(T[key], P[key])
        roc_auc = auc(fpr, tpr)
        print("ROC AUC for {} = {}".format(str(key), str(roc_auc)))
        plot_roc(fpr, tpr, key, thresholds)
        optimal_data[key] = best_threshold, best_F_score, best_precision, best_recall
        threshold_list.append(best_threshold)
        F_score_list.append(best_F_score)
        precision_list.append(best_precision)
        recall_list.append(best_recall)

    precision, recall, thresholds = precision_recall_curve(T_all, P_all)
    best_threshold, best_F_score, best_precision, best_recall = maximise_F_score(precision, recall, thresholds)
    plot_precision_recall(precision, recall, thresholds, 'Micro Average')
    fpr, tpr, thresholds = roc_curve(T_all, P_all)
    roc_auc = auc(fpr, tpr)
    print("ROC AUC for {} = {}".format('micro average', str(roc_auc)))
    plot_roc(fpr, tpr, 'Micro Average', thresholds)
    optimal_data['Micro Average'] = best_threshold, best_F_score, best_precision, best_recall

    print("Best data summary with order : (Threshold, F-score, precision, recall)")
    print(optimal_data)

    P_all[P_all >= best_threshold] = 1.0
    P_all[P_all < best_threshold] = 0.0
    acc = accuracy_score(T_all, P_all)

    print("Accuracy of the model : " + str(acc))

    # record_df.loc[len(record_df)-1, 'mAP'] = mean_average_prec
    # record_df.to_csv(C.record_path, index=0)
    # print('Save mAP to {}'.format(C.record_path))


if __name__ == "__main__":

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Use Faster R-CNN to detect insects.')
    parser.add_argument('--model_name', required=False,
                        metavar="name_of_your_model", default='model',
                        help='Name of the model being tested')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/insects/test/dataset/", default='./data',
                        help='Directory of the Insects test dataset')
    parser.add_argument('--from_csv', required=False, default='True',
                        metavar="True/False",
                        help='True if loading test images from csv file')
    parser.add_argument('--annotations', required=False, default='./data/data_annotations.txt',
                        metavar="/path/to/insects/dataset/annotations/file.txt",
                        help='Annotation file for the provided dataset')
    parser.add_argument('--save_images', required=False, default='False',
                        metavar="True/False",
                        help="True if you want to save the images after detection, False otherwise")
    parser.add_argument('--compute_accuracy', required=False, default='True',
                        metavar="True/False",
                        help="True if you want to compute the different 'accuracy' metrics, False otherwise")
    parser.add_argument('--use_gpu', required=False, default="True",
                        metavar="True/False",
                        help="True if you want to run the training on a gpu, False otherwise")
    parser.add_argument('--noise_reduction', required=False, default='None',
                        metavar="None/box_filter/gaussian",
                        help="Noise reduction technique to use as preprocessing")
    parser.add_argument('--histogram_equalization', required=False, default="False",
                        metavar="True/False",
                        help="True if you want to apply histogram equlization as preprocessing, False otherwise")
    parser.add_argument('--gamma_correction', required=False, default="False",
                        metavar="True/False",
                        help="True if you want to apply gamma correction as preprocessing, False otherwise")
    parser.add_argument('--processed_directory', type=str, required=False, default="new_directory",
                        metavar="directory_name",
                        help="Name of the directory in which the processed images will be saved when show_images is "
                             "True") 


    args = parser.parse_args()

    use_gpu = eval(args.use_gpu)
    from_csv = eval(args.from_csv)
    save_images = eval(args.save_images)
    compute_accuracy = eval(args.compute_accuracy)
    noise_reduction = eval(args.noise_reduction)
    histogram_equalization = eval(args.histogram_equalization)
    gamma_correction = eval(args.gamma_correction)
    processed_directory = args.processed_directory

    try:
        os.mkdir('other/box_figure/{}'.format(processed_directory))
    except OSError:
        print('Creation of the directory {} failed'.format(processed_directory))
    else:
        print('Successfully created the directory {}'.format(processed_directory))

    if use_gpu is True:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session = tf.compat.v1.InteractiveSession(config=config)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        session = tf.compat.v1.InteractiveSession(config=config)

    test_path = args.annotations  # Test data (annotation file)
    data_test_path = args.dataset

    imgs_record_path = "./logs/{} - imgs.csv".format(args.model_name)

    # output_results_filename = "./results/{}".format(args.model_name)
    # if not os.path.exists(output_results_filename):
    #     os.mkdir(output_results_filename)
    # TODO: output results of accuracy in file
    # TODO: parametrer le programme pour afficher les images ou calculer les metrics

    config_output_filename = "./config/{}.pickle".format(args.model_name)

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    C.model_path = "./model/{}.hdf5".format(args.model_name)  # UPDATE WEIGHTS PATH HERE !!!!!!!!

    # record_df = plot_some_graphs(C)
    model_rpn, class_mapping, model_classifier_only = init_models()
    nbr_classes = len(class_mapping.keys()) - 1
    print(nbr_classes)

    if save_images:
        draw_box_on_images(noise_reduction, histogram_equalization, gamma_correction)
    if compute_accuracy:
        accuracy(noise_reduction, histogram_equalization, gamma_correction)
