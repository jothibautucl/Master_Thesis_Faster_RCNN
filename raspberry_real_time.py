# from detection_libraries import *
import os
import pickle
from real_time_libraries import *
from multiprocessing import Process, Queue

if __name__ == "__main__":
    import argparse
    import csv
    from ast import literal_eval

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Use Faster R-CNN to detect insects.')
    parser.add_argument('--model_name', required=False,
                        metavar="name_of_your_model", default='model',
                        help='Name of the model being tested')
    parser.add_argument('--bbox_threshold', required=False, default=0.822,
                        metavar="Value from 0 to 1",
                        help="Model probability threshold to detect an object")
    parser.add_argument('--demo', required=False, default="False",
                        metavar="True/False",
                        help="True if you want to run the demo, False otherwise")
    parser.add_argument('--show_images', required=False, default="False",
                        metavar="True/False",
                        help="True if you want to see the images read by the camera")
    parser.add_argument('--use_gpu', required=False, default="True",
                        metavar="True/False",
                        help="True if you want to run the training on a gpu, False otherwise")
    parser.add_argument('--use_motor', required=False, default="False",
                        metavar="True/False",
                        help="True if you want to use a motor")
    parser.add_argument('--resolution', required=False, default="(1920, 1088)",
                        metavar="(xres,yres)",
                        help="Resolution of the pictures taken by the camera")
    parser.add_argument('--fps', required=False, default=1,
                        metavar="integer",
                        help="Framerate of the camera")
    args = parser.parse_args()

    use_gpu = eval(args.use_gpu)
    demo = eval(args.demo)
    use_motor = eval(args.use_motor)
    resolution = literal_eval(args.resolution)
    fps = int(args.fps)

    config_output_filename = "./config/{}.pickle".format(args.model_name)

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    C.show_images = eval(args.show_images)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    C.model_path = "./model/{}.hdf5".format(args.model_name)  # UPDATE WEIGHTS PATH HERE !!!!!!!!


    bbox_threshold = float(args.bbox_threshold)
    output_results_filename = "./results/{}".format(args.model_name)
    if not os.path.exists(output_results_filename):
        os.mkdir(output_results_filename)

    if demo:
        run_demo(C, bbox_threshold)
    else:
        frame_queue = Queue()
        flag_queue = Queue()
        images_output_dir = os.path.join(output_results_filename, "images_{}".format(get_timestamp()))
        record_path = os.path.join(output_results_filename, "raw_detections_{}.csv".format(get_timestamp()))
        if not os.path.exists(images_output_dir):
            os.mkdir(images_output_dir)
        p_detection = Process(target=run_detection, args=(10, resolution, 0.3, 1000, 100, use_motor, images_output_dir, C, frame_queue, flag_queue))
        p_processing = Process(target=run_processing, args=(bbox_threshold, C, record_path, images_output_dir, use_gpu, frame_queue, flag_queue))
        p_detection.start()
        p_processing.start()
        try:
            p_detection.join()
            p_processing.join()
        except KeyboardInterrupt:
            print()
            print("[INFO] Stopping subprocesses")
            p_detection.terminate()
            p_processing.terminate()
            print("[INFO] Exiting")


