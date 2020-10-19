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


def prepare_predictor():

    class_list = ['diaodeng','pishafa'] ### 需要自己添加
    train_metadata =  'zc_data' ### 需要自己添加

    # current_path = os.getcwd()
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    # print("father_path:",father_path)

    modelFile = os.path.join(father_path,"models/model20200728.pth")
    cfgFile = os.path.join(father_path,"configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

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
    cfg.MODEL.DEVICE = "cuda" # we use a CPU Detectron copy

    classes = train_metadata.get("thing_classes", None)
    predictor = DefaultPredictor(cfg)
    print("Predictor has been initialized.")

    return (predictor, classes)


PREDICTOR, CLASSES = prepare_predictor()

def image_predict(bgr_img):

    # regist_train_name = 'zc_data' ### 需要自己添加
    # class_list = ['diaodeng','pishafa'] ### 需要自己添加
    # num_class = len(class_list)
    # MetadataCatalog.get(regist_train_name).thing_classes = class_list
    # train_metadata = MetadataCatalog.get(regist_train_name)

    import time
    start_time = time.time()
    print('预测开始')
    outputs = PREDICTOR(bgr_img)
    print('预测完成', time.time() - start_time)

    pred_scores = outputs["instances"].scores.tolist()
    pred_classes = outputs["instances"].pred_classes.data.cpu().numpy().tolist() if outputs["instances"].get_fields()["pred_boxes"] else None
    pred_bboxes = outputs["instances"].pred_boxes.tensor.tolist() if outputs["instances"].get_fields()["pred_boxes"] else None


    # v = Visualizer(bgr_img[:, :, ::-1],metadata=train_metadata,scale=1,
    #                        instance_mode=ColorMode.IMAGE_BW)   # remove the colors of unsegmented pixels)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #
    # cv2.imwrite("./1.png",v.get_image()[:, :, ::-1])


    results = {
        "scores": pred_scores,### 预测分值列表，[0.2, ...]
        "pred_classes": pred_classes,### 预测物体编码列表，[0,1，...]
        "pred_boxes" : pred_bboxes,### 预测物体rois [[x1,y1,x2,y2],[....]]
        "classes": CLASSES ## 物体名称， ['diaodeng','pishafa', ....]
    }
    # print(results)
    return results



def test():

    # imgfile = "../demo.jpg"

    current_path = os.path.abspath(__file__) ## 当前目录
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    imgfile = os.path.join(father_path,'../demo.jpg')

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


# if __name__ == "__main__":
#     test()



