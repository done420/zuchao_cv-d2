import cv2,os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



##coding:utf-8

import cv2,os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import numpy
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
# from google.colab.patches import cv2_imshow
from PIL import Image


def prepare_pridctor():

    class_list = ['diaodeng','pishafa'] ### 需要自己添加
    train_metadata =  'zc_data' ### 需要自己添加

    modelFile =  "/home/user/qunosen/2_project/0_dvep/1_detectron2/ImageDetectionAPI/d2_object_detection/models/model20200728.pth"

    cfgFile = "/home/user/qunosen/2_project/0_dvep/1_detectron2/ImageDetectionAPI/d2_object_detection/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    num_class =len(class_list)
    MetadataCatalog.get(train_metadata).thing_classes = class_list
    train_metadata = MetadataCatalog.get(train_metadata)

    # create config
    cfg = get_cfg()
    # below path applies to current installation location of Detectron2
    cfg.merge_from_file(cfgFile)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class
    cfg.MODEL.WEIGHTS = modelFile
    cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy

    classes = train_metadata.get("thing_classes", None)
    predictor = DefaultPredictor(cfg)
    print("Predictor has been initialized.")

    return (predictor, classes)


def image_predict(bgr_img):

    predictor,classes = prepare_pridctor()

    outputs = predictor(bgr_img)

    pred_scores = outputs["instances"].scores.tolist()
    pred_classes = outputs["instances"].pred_classes.numpy().tolist() if outputs["instances"].get_fields()["pred_boxes"] else None
    pred_bboxes = outputs["instances"].pred_boxes.tensor.tolist() if outputs["instances"].get_fields()["pred_boxes"] else None


    results = {
        "scores": pred_scores,### 预测分值列表，[0.2, ...]
        "pred_classes": pred_classes,### 预测物体编码列表，[0,1，...]
        "pred_boxes" : pred_bboxes,### 预测物体rois [[x1,y1,x2,y2],[....]]
        "classes": classes ## 物体名称， ['diaodeng','pishafa', ....]
            }
    # print(results)
    return results



def test():

    # imgfile = "../demo.jpg"

    current_path = os.path.abspath(__file__) ## 当前目录
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    imgfile = '/home/user/qunosen/2_project/0_dvep/1_detectron2/ImageDetectionAPI/demo.jpg'

    cfgFile = os.path.join(os.getcwd(),"configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    modelFile = os.path.join(os.getcwd(),"models/model20200728.pth")

    if not os.path.exists(cfgFile):
        print("NotFound: {}".format(cfgFile))

    if not os.path.exists(modelFile):
        print("NotFound: {}".format(modelFile))


    # img = cv2.imread(imgfile)

    ### use PIL open image
    image = Image.open(imgfile)
    img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)

    results = image_predict(img)

    class_names = results.get('classes')
    for i in range(len(results['pred_boxes'])):
        roi = results['pred_boxes'][i]
        pred_class = class_names[results.get('pred_classes')[i]]
        pred_score = results.get('scores')[i]
        x1,y1,x2,y2 = roi
        x1,x2 = int(x1),int(x2)
        y1,y2 = int(y1),int(y2)
        x,y = int((x2 + x1)/2) , int((y2 + y1)/2)

        text = '{},{:.2f}'.format(pred_class,pred_score)
        crop = img[y1:y2,x1:x2]

        save_file = "../test_out/demo_object-detection.png"

        cv2.putText(img,text,(x,y), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.imwrite(save_file,img)

    print("{}:\n {}".format(imgfile,results))

#### https://juejin.im/post/6844904201135357960


# if __name__ == "__main__":
#     imgfile = '/home/user/qunosen/2_project/0_dvep/1_detectron2/ImageDetectionAPI/demo.jpg'
#     image = Image.open(imgfile)
#     img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
#
#     results = image_predict(img)
#
#     class_names = results.get('classes')
#
#     print("{}:\n {}".format(imgfile,results))


