import tensorflow as tf
import numpy as np
import os
import cv2
import math
from moviepy.editor import VideoFileClip
# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#if tf.__version__ < '1.4.0':
    #raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

class ShipDetection:
    frame = 0
    max_frame = 0
    progress = 0
    jlog = {}
    def __init__(self, ship_detection_graph_path,
                 ship_detection_graph_label_path,
                 num_classes):
        self.__ship_detection_graph_path = ship_detection_graph_path
        self.__ship_detection_graph = ShipDetection.load_graph(self.__ship_detection_graph_path)
        self.__num_classes = num_classes
        self.__label_map = label_map_util.load_labelmap(ship_detection_graph_label_path)
        self.__categories = label_map_util.convert_label_map_to_categories(self.__label_map,
                                                                           max_num_classes=self.__num_classes,
                                                                           use_display_name=True)
        self.__category_index = label_map_util.create_category_index(self.__categories)


    @staticmethod
    def load_graph(graph_path):
        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path,'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def calculate_distance(x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        delta = 0.0033 * (dist ** 2) - 1.9918 * dist + 372.28

        return delta

    def pipeline(self,image):

        ShipDetection.frame += 1
        ShipDetection.progress = int((ShipDetection.frame / ShipDetection.max_frame) * 100)
        print(ShipDetection.progress)
        detection_graph= self.__ship_detection_graph
        category_index= self.__category_index
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:

                image_np = np.asarray(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')

                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=.9,
                    line_thickness=2)

                ################### Data analysis ###################
                print("")
                final_score = np.squeeze(scores)  # scores
                r_count = 0  # counting
                r_score = []  # temp score, <class 'numpy.ndarray'>
                final_category = np.array([category_index.get(i) for i in classes[0]])  # category
                r_category = np.array([])  # temp category

                for i in range(100):
                    if scores is None or final_score[i] > 0.7:
                        r_count = r_count + 1
                        r_score = np.append(r_score, final_score[i])
                        r_category = np.append(r_category, final_category[i])

                if r_count > 0:
                    print("Number of bounding boxes: ", r_count)
                    print("")
                else:
                    print("Not Detect")
                    print("")
                for i in range(len(r_score)):  # socre array`s length
                    print(
                        "Object Num: {} , Category: {} , Score: {}%".format(i + 1, r_category[i]['name'],
                                                                            100 * r_score[i]))
                    print("")
                    final_boxes = np.squeeze(boxes)[i]  # ymin, xmin, ymax, xmax
                    xmin = final_boxes[1]
                    ymin = final_boxes[0]
                    xmax = final_boxes[3]
                    ymax = final_boxes[2]
                    location_x = (xmax + xmin) / 2
                    location_y = (ymax + ymin) / 2
                    min_location = location_y * 100

                    # print("final_boxes [ymin xmin ymax xmax]")
                    # print("final_boxes", final_boxes)
                    # if (min_location > 35):
                    #     cv2.putText(image_np, 'FAR', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), thickness=2)
                    # else:
                    #     cv2.putText(image_np, 'NEAR', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0),
                    #                 thickness=2)
                    print("Location x: {}, y: {}".format(location_x, location_y))
                    print("")
                print("+ " * 30)

        return image_np


def main(INPUT_FILE,subclip_start,subclip_end):
    ShipDetection.max_frame = 0
    ShipDetection.jlog = {}
    ShipDetection.frame = 0
    ShipDetection.progress = 0
    INPUT_DIRECTORY = 'input_video'
    OUTPUT_DIRECTORY = 'output_video'
    INPUT_FILE = 'test.mp4'
    OUTPUT_FILE = 'result4.mp4'
    vid_output = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE)
    vid_input = os.path.join(INPUT_DIRECTORY, INPUT_FILE)
    clip = VideoFileClip(vid_input)



    # Model preparation
    MODEL_NAME = './models/ssd_mobilenet_v1_ship_15000'

    ship_detection_graph_path = MODEL_NAME + '/frozen_inference_graph.pb'

    ship_detection_graph_label_path = os.path.join('data', 'ship_label_map.pbtxt')

    NUM_CLASSES = 1
    object_detect = ShipDetection(ship_detection_graph_path,
                                  ship_detection_graph_label_path,
                                     num_classes=NUM_CLASSES)
    ShipDetection.max_frame=clip.duration*clip.fps
    # Write processed images to video
    vid = clip.fl_image(object_detect.pipeline)
    vid.write_videofile(vid_output, audio=False)


if __name__ == '__main__':
    main()