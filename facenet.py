# coding=utf-8
"""Face Detection and Recognition"""
import os
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
from .src import facenet, align
from .src.align import detect_face


gpu_memory_fraction = 0.3
facenet_model_checkpoint = os.path.dirname(__file__) + "/model_checkpoints/20170512-110547"
debug = False


def compare_faces(embs, emb, tolerance=1.1, mode='min_5'):
    """compare a list of face embeddings to one embedding
    modes:
    -vote
        returns True when distance is not above threshold
    -min_x
        returns True when distance is the minimal x of all
         and not above threshold
    :param embs: list;
    :param emb: np.array;
    :param tolerance: float
    :return: matches: list; list of True/False indicating match/no match
    """
    matches = []
    dists = []
    for i in range(len(embs)):
        dist = np.sqrt(np.sum(np.square(np.subtract(embs[i], emb))))
        dists.append(dist)
        matches.append(dist <= tolerance)
    if mode == 'vote':
        return matches
    elif 'min_' in mode:
        x = int(mode.split('_')[-1])
        arg_sort_dists = np.argsort(dists)[:x]
        matches = [False] * len(dists)
        for idx in arg_sort_dists:
            matches[idx] = True
        return matches
    else:
        raise ValueError("mode is either vote or min_x")


class FaceDetector(object):
    """
    face detector for initializing the MTCNN model
    """
    minsize = 20  # minimum size of face
    minratio = 1 / 20 # ratio of minimum size of face to minimum side of image
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        """
        find faces in image. return face bbox info.
        facebbox: [t,l,b,r]; values are distance to topleft corner
        :param image: np.ndarray; image data
        :return: bboxes: list; list of face bbox lists.
        """
        bboxes = []
        img_size = np.asarray(image.shape)[0:2]

        bounding_boxes, _ = detect_face.detect_face(image, self.minsize,
                                                    self.pnet, self.rnet, self.onet,
                                                    self.threshold, self.factor)
        if bounding_boxes is None:
            return bboxes
        for bb in bounding_boxes:
            if bb is None:
                continue
            # points coordinates has topleft as origin; [width/to/origin， height/to/origin]
            tl = [np.maximum(bb[0] - self.face_crop_margin / 2, 0),
                  np.maximum(bb[1] - self.face_crop_margin / 2, 0)]
            br = [np.minimum(bb[2] + self.face_crop_margin / 2, img_size[0]),
                  np.minimum(bb[3] + self.face_crop_margin / 2, img_size[1])]
            bbox = [tl[0], tl[1], br[0], br[1]]
            # case when predicted face can't be drawn
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            bboxes.append(bbox)

        return bboxes

    def batch_find_faces(self, images):
        bounding_boxes = detect_face.bulk_detect_face(images, self.minratio, self.pnet,
                                              self.rnet, self.onet, self.threshold,
                                              self.factor)
        bboxes = []
        for image, bounding_boxes_1 in zip(images, bounding_boxes):
            # no face detected
            if bounding_boxes_1 is None:
                continue
            img_size = np.asarray(image.shape)[0:2]
            bboxes_1 = []
            for bb in bounding_boxes_1[0]:
                if bb is None:
                    continue
                # points coordinates has topleft as origin; [width/to/origin， height/to/origin]
                tl = [np.maximum(bb[0] - self.face_crop_margin / 2, 0),
                      np.maximum(bb[1] - self.face_crop_margin / 2, 0)]
                br = [np.minimum(bb[2] + self.face_crop_margin / 2, img_size[0]),
                      np.minimum(bb[3] + self.face_crop_margin / 2, img_size[1])]
                bbox = [tl[0], tl[1], br[0], br[1]]
                # case when predicted face can't be drawn
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                bboxes_1.append(bbox)
            bboxes.append(bboxes_1)
        return bboxes, bounding_boxes


class FaceEncoder(object):
    def __init__(self, face_crop_size=160):
        self.face_crop_size = face_crop_size
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        """
        resize face and produce face embeddings
        :param face:
        :return:
        """
        # Get input and output tensors
        face = misc.imresize(face, (self.face_crop_size, self.face_crop_size), interp='bilinear')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def batch_generate_embeddings(self, faces):
        faces = np.array([facenet.prewhiten(misc.imresize(face, (self.face_crop_size, self.face_crop_size),
                                                          interp='bilinear')) for face in faces])
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)
