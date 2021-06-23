import os
import numpy as np
import matplotlib.pyplot as plt

def get_average_process_time(filename):
    with open(filename, 'r', encoding="utf8") as fr:
        fr.readline()
        elapsed_time = []
        for line in fr:
            elements = line.split(" ")
            if elements[0] != "Elapsed":
                continue

            elapsed_time.append(float(elements[3]))

        return len(elapsed_time), np.mean(elapsed_time)

def get_average_precision(filename):
    with open(filename, 'r', encoding="utf8") as fr:
        fr.readline()
        for line in fr:
            elements = line.split(" ")
            if elements[0] == "mean" and elements[1] == "average" and elements[2] == "precision:":
                ap = elements[3]
        return float(ap)

def get_average_roc(filename):
    with open(filename, 'r', encoding="utf8") as fr:
        fr.readline()
        roc = []
        for line in fr:
            elements = line.split(" ")
            if elements[0] == "ROC":
                roc.append(float(elements[5]))

        return np.mean(roc)

def size_mAP_graph(time, mAP):
    plt.figure()
    plt.xlabel('image size [px]', fontsize=14)
    plt.ylabel('mAP', fontsize=14)
    plt.plot(time, mAP)
    plt.title('Variation of mAP according to image size', fontsize=14)
    plt.savefig('./other/graphs/mAP-time')


def size_roc_graph(time, roc):
    plt.figure()
    plt.xlabel('image size [px]', fontsize=14)
    plt.ylabel('mean roc auc', fontsize=14)
    plt.plot(time, roc)
    plt.title('Variation of mean ROC AUC according to image size', fontsize=14)
    plt.savefig('./other/graphs/rocauc-time')

def errorbars_plot(x, ymean, ymin, ymax, title):
    yerr = np.array([ymean-ymin, ymax-ymean])
    plt.figure()
    plt.xlabel('Parameters set number')
    plt.ylabel('Loss value')
    plt.errorbar(x, ymean, yerr=yerr, fmt='o', label='Simulations')
    plt.title(title)
    plt.savefig("./other/graphs/{}".format(title))

def model800_validated_errorbars():
    x = np.linspace(1, 24, 24)
    losses = [[1.51926252825744, 1.66853861213862, 1.45826058800838, 1.47642351591153, 1.5998043735957, 1.56645346965844, 1.46353520653816, 1.51243538985389, 1.50971336742702, 1.56846334136927, 1.39489259470674, 1.51818823894659, 1.31143558608332, 1.3059632644277, 1.41368113469937, 1.52907256648486, 1.51697500688232, 1.5076247528201, 1.58084173579007, 1.50852613539146, 1.39371157777398, 1.55986712122155, 1.43283133997476, 1.48253216779134],
              [1.50048879063642, 1.56877873788748, 1.4626782253462, 1.49729563685323, 1.65667210475073, 1.66522977050467, 1.47180397624649, 1.75733709887429, 1.53668522549892, 1.61764072199194, 1.53495123768125, 1.562101909043, 1.28345518296078, 1.35816820380058, 1.40582106456544, 1.5184432818596, 1.51198253607407, 1.66066691482243, 1.43230374512626, 1.53549377950613, 1.54672491749029, 1.60643677643201, 1.45561674542076, 1.36877871573994],
              [1.45626764667145, 1.59321313468249, 1.43732953786955, 1.35479328285842, 1.6075923709721, 1.58742617519135, 1.59363668711094, 1.5746187025893, 1.48958683341946, 1.41085702953133, 1.55644923219051, 1.58437782953392, 1.42171404346225, 1.35891807584029, 1.34583736087195, 1.30783053535425, 1.34812426120192, 1.51757786766249, 1.61030448299616, 1.61590782265686, 1.43492812127462, 1.38266109533211, 1.39493724802103, 1.32470909805524],
              [1.41490639604257, 1.41526352881915, 1.49721135968022, 1.5446495084799, 1.52709185446875, 1.5327937464671, 1.68420480405549, 1.56807459151254, 1.47161904648676, 1.49836916259534, 1.43024072999476, 1.30483072372281, 1.3139832931516, 1.39232750500077, 1.3555113423455, 1.35639885572937, 1.40092117486683, 1.4851465684809, 1.52887192633679, 1.53735080520218, 1.27231421465246, 1.39403830286818, 1.45910157719283, 1.45829081028252]]
    losses_min = np.min(losses, axis=0)
    losses_max = np.max(losses, axis=0)
    losses_mean = np.mean(losses, axis=0)
    print(losses_mean)
    errorbars_plot(x, losses_mean, losses_min, losses_max, 'Validation of model800')

def model6000_validated_errorbars():
    x = np.linspace(13, 16, 4)
    # losses = [[0.847500530434425, 0.8160113169647802, 0.8157910783364376, 0.8594888867015318],
    #           [0.8881682209536967, 0.8268874593089663, 0.862405197471224, 0.8200220964624049]]
    losses = [[0.8594888867015318, 0.8160113169647802, 0.8157910783364376, 0.847500530434425],
              [0.8200220964624049, 0.8268874593089663, 0.862405197471224, 0.8881682209536967]]
    losses_min = np.min(losses, axis=0)
    losses_max = np.max(losses, axis=0)
    losses_mean = np.mean(losses, axis=0)
    print(losses_mean)
    errorbars_plot(x, losses_mean, losses_min, losses_max, 'Validation of model6000')


if __name__ == '__main__':
    model_names = ["size100", "size200", "size300"]
    time = []
    mAP = []
    roc = []
    img_sizes = [100, 200, 300]
    for name in model_names:
        RPI_logs = "./other/output files/logs_" + name + ".txt"
        metrics = "./other/output files/metrics_" + name + ".txt"
        _, process_time = get_average_process_time(RPI_logs)
        time.append(round(process_time, 3))
        roc.append(round(get_average_roc(metrics), 3))
        mAP.append(round(get_average_precision(metrics), 3))

    size_roc_graph(img_sizes, roc)
    size_mAP_graph(img_sizes, mAP)
    # model6000_validated_errorbars()
