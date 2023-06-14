from detectron2.engine import DefaultPredictor
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from PIL import Image 
import PIL 
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

class Detector:

    def __init__(self, model_type = "instanceSegmentation"):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = "/Users/obinwosu/otomatic/DrumAI_06102023.pth"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu" # cpu or cuda

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(imagePath):
        im = cv2.imread(imagePath)
        outputs = predictor(im)
        masks=outputs["instances"].pred_masks.to("cpu").numpy().astype('uint8')
        areas = np.zeros(masks.shape[0])
        i=0
        blank = np.zeros(im.shape[0:2])

        for mask in masks:
          contours = []
          contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
          contours.append(contour[0])
          cnt=contour[0]
          area = cv2.contourArea(cnt)
          areas[i]= area
          i = i + 1
          if i==2:
            ratio=np.min(areas)/np.max(areas)
            ratio_value=ratio.item()
            percentage="{:.1%}".format(ratio_value)
            img1 = cv2.drawContours( blank.copy(), contours[0], 0, 1)
            img2 = cv2.drawContours( blank.copy(), contours[0], 1, 1)
            intersection = np.logical_and( img1, img2 )
            marginal = (intersection.any())
            if marginal == False:
              x = "central perforation"
            else:
              x = "marginal perforation"
            result = "The model identified a {}".format(x) + " which is {} of the total tympanic membrane".format(percentage)
            print(result)

        v = Visualizer(im[:, :, ::-1],
                    metadata=perf_seg_metadata,
                    scale=0.8,
                    instance_mode=ColorMode.SEGMENTATION
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        filename = 'result.jpg'
        plt.figure(figsize = (22, 22))
        plt.subplot(121);plt.imshow(im[:,:,::-1]);plt.title("Input");plt.axis('off')
        plt.subplot(122);plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB));plt.title("Predicted Segmentation");plt.axis('off');
        plt.show()
