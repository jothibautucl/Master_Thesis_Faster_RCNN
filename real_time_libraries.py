def close_trap(kit):
    import time
    import board
    from adafruit_motorkit import MotorKit
    from adafruit_motor import stepper

    for i in range(50):
        kit.stepper1.onestep(direction=stepper.FORWARD)
        time.sleep(0.01)

def open_trap(kit):
    import time
    import board
    from adafruit_motorkit import MotorKit
    from adafruit_motor import stepper

    for i in range(50):
        kit.stepper1.onestep(direction=stepper.BACKWARD)
        time.sleep(0.1)


def trap_insect(kit, trap_duration, free_duration):
    import time

    close_trap(kit)
    time.sleep(trap_duration)
    open_trap(kit)
    time.sleep(free_duration)


def init_session(use_gpu):
    from detection_libraries import tf
    from detection_libraries import os

    if use_gpu is True:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session = tf.compat.v1.InteractiveSession(config=config)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        session = tf.compat.v1.InteractiveSession(config=config)


def run_detection(fps, alpha, min_area, C, frame_queue, flag_queue):
    import imutils
    from imutils.video import VideoStream
    import cv2
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    import time
    import board
    from adafruit_motorkit import MotorKit
    from adafruit_motor import stepper
    from multiprocessing import Process

    kit = MotorKit(i2c=board.I2C())
    p_trap = None

    flag = flag_queue.get(block=True, timeout=None)
    if flag == "ready":
        if C.verbose:
            print("[INFO] detection_proc - models ready: detection can start")
    else:
        print("[ERROR] detection_proc - error loading models: aborted")
        return

    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    # initialize the first frame in the video stream
    avg = None
    # loop over the frames of the video
    while True:
        time.sleep(round(1 / fps))
        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = vs.read()

        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
            break
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # if the first frame is None, initialize it
        if avg is None:
            if C.verbose:
                print("[INFO] detection_proc - starting background model...")
            avg = gray.copy().astype("float")
            continue
        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, avg, alpha)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue
            if C.verbose:
                print("[INFO] detection_proc - *** Movement detected ***")
            if p_trap  is None or not p_trap.is_alive():
                p_trap = Process(target=trap_insect, args=(kit, 60, 30))
                p_trap.start()
            frame_queue.put((get_timestamp(), frame))
            break

        if C.show_images:
            cv2.imshow("image", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break

    if C.show_images:
        cv2.destroyAllWindows()


def run_demo(C, bbox_threshold):
    import numpy as np
    from detection_libraries import init_models
    from detection_libraries import detect
    import imutils
    import time
    from imutils.video import VideoStream
    from imutils.video import FPS
    import cv2
    # from picamera.array import PiRGBArray
    # from picamera import PiCamera

    print("[INFO] loading model...")
    model_rpn, class_mapping, model_classifier_only = init_models(C)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    # initialize the video stream, allow the camera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    fps = FPS().start()
    while True:
        img = vs.read()
        imutils.resize(img, width=400)

        detect(img, model_rpn, model_classifier_only, C, class_mapping, bbox_threshold, class_to_color)
        cv2.imshow("Frame", img)

        key = cv2.waitKey(1000) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    print("[INFO] ending video stream...")
    vs.stop()


def run_processing(bbox_threshold, C, output_results_filename, use_gpu, frame_queue, flag_queue):

    print("[INFO] processing_proc - initializing session...")
    import numpy as np
    from detection_libraries import init_models
    from detection_libraries import detect
    import csv
    import time
    import cv2
    import os

    record_path = os.path.join(output_results_filename, "raw_detections_{}.csv".format(get_timestamp()))
    images_output_dir = os.path.join(output_results_filename, "images_{}".format(get_timestamp()))
    if not os.path.exists(images_output_dir):
        os.mkdir(images_output_dir)

    init_session(use_gpu)
    if C.verbose:
        print("[INFO] processing_proc - loading model...")
    model_rpn, class_mapping, model_classifier_only = init_models(C)
    if C.verbose:
        print("[INFO] processing_proc - done loading model")
    flag_queue.put("ready")
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    fieldnames = ['date',
                  'class', 'probability',
                  'x1', 'y1', 'x2', 'y2',
                  'temperature', 'humidity', 'pressure', 'wind', 'sun_exposure', 'rain', 'weather description',
                  'lat', 'lon']

    while True:
        time.sleep(1)
        timestamp, img = frame_queue.get(block=True, timeout=None)
        if C.verbose:
            print("[INFO] processing_proc - starting detection on a new image")
        all_dets = detect(img, model_rpn, model_classifier_only, C, class_mapping, bbox_threshold, class_to_color)
        if not len(all_dets) == 0:
            for detected_class, probability, ((x1, y1), (x2, y2)) in all_dets:
                with open(record_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(
                        {'date': timestamp, 'class': detected_class, 'probability': round(probability, 3),
                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'temperature': 'empty', 'humidity': 'empty',
                         'pressure': 'empty', 'wind': 'empty', 'sun_exposure': 'empty', 'rain': 'empty',
                         'weather description': 'empty', 'lat': 'empty', 'lon': 'empty'})

        cv2.imwrite(os.path.join(images_output_dir, "{}.jpg".format(timestamp)), img)
        if C.show_images:
            cv2.imshow("detection", img)
            key = cv2.waitKey(1000) & 0xFF
            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
               break

    if C.show_images:
        cv2.destroyAllWindows()


def get_timestamp():
    from datetime import datetime
    
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
