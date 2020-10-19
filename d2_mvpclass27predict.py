##coding:utf-8

import cv2,os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import numpy,time,datetime
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
# from google.colab.patches import cv2_imshow
from PIL import Image
from detectron2.utils.visualizer import ColorMode


classname_to_id = {'shuangrenchuang': 1, 'danrenchuang': 2, 'chaji': 3, 'danyi': 4, 'diaodeng': 5, 'chugui': 6, 'chuwugui': 7,
                       'ertongchuang': 8, 'dianshigui': 9, 'buyishafa': 10, 'yigui': 11, 'chuangtougui': 12, 'jiugui': 13, 'huwaideng': 14,
                       'xuanguangui': 15, 'bianji': 16, 'shuijingdeng': 17, 'shuzhuangtai': 18, 'weiyugui': 19, 'zhuangshigui': 20,
                       'chazhuo': 21, 'xiegui': 22, 'shafazuhe': 23, 'canzhuo': 24, 'pizhishafa': 25, 'shuzhuo': 26, 'batai': 27}

class_list = [ele for ele in classname_to_id]
regist_train_name = "zc1_data"


def prepare_predictor():
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

    # modelFile = os.path.join(father_path,"models/model20200728.pth")
    #modelFile = os.path.join(father_path,"models/model_6class20200918_final.pth") ### class-6
    modelFile = os.path.join(father_path,"models/model_mvp27class70k_20201015.pth") ### class-27;30:model_mvp27class20201014.pth,

    cfgFile = os.path.join(father_path,"configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    num_class =len(class_list)
    MetadataCatalog.get(regist_train_name).thing_classes = class_list
    train_metadata = MetadataCatalog.get(regist_train_name)

    # create config
    cfg = get_cfg()
    # below path applies to current installation location of Detectron2
    cfg.merge_from_file(cfgFile)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class
    cfg.MODEL.WEIGHTS = modelFile
    cfg.MODEL.DEVICE = "cuda" # we use a GPU
    # cfg.MODEL.DEVICE = "cpu" # we use a GPU

    classes = train_metadata.get("thing_classes", None)
    predictor = DefaultPredictor(cfg)
    print("Predictor has been initialized.")

    return (predictor, classes)


PREDICTOR, CLASSES = prepare_predictor()

def image_predict(bgr_img):

    outputs = PREDICTOR(bgr_img)

    pred_scores = outputs["instances"].scores.tolist()
    pred_classes = outputs["instances"].pred_classes.data.cpu().numpy().tolist() if outputs["instances"].get_fields()["pred_boxes"] else None
    pred_bboxes = outputs["instances"].pred_boxes.tensor.tolist() if outputs["instances"].get_fields()["pred_boxes"] else None
    print(dict([i, class_list[i]] for i in pred_classes))
    results = {
        "scores": pred_scores,### 预测分值列表，[0.2, ...]
        "pred_classes": pred_classes,### 预测物体编码列表，[0,1，...]
        "pred_boxes" : pred_bboxes,### 预测物体rois [[x1,y1,x2,y2],[....]]
        "classes": CLASSES ## 物体名称， ['diaodeng','pishafa', ....]
    }
    # print(results)
    return results



def view_pred_img(bgr_img,savefile):
    outputs = PREDICTOR(bgr_img)
    pred_scores = outputs["instances"].scores.tolist()

    train_metadata = MetadataCatalog.get(regist_train_name)
    v = Visualizer(bgr_img[:, :, ::-1], metadata=train_metadata, scale=0.9, instance_mode=ColorMode.IMAGE_BW)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    predict_file = savefile

    cv2.imwrite(predict_file, v.get_image()[:, :, ::-1])




def test():

    # imgfile = "../demo.jpg"
    current_path = os.path.abspath(__file__) ## 当前目录
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    imgfile = os.path.join(father_path,'../demo.jpg')
    # imgfile = '/home/user/zs/ImagePredictor/xiandai1.jpg'

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

    # class_names = results.get('classes')
    # for i in range(len(results['pred_boxes'])):
    #     roi = results['pred_boxes'][i]
    #     pred_class = class_names[results.get('pred_classes')[i]]
    #     pred_score = results.get('scores')[i]
    #     x1,y1,x2,y2 = roi
    #     x1,x2 = int(x1),int(x2)
    #     y1,y2 = int(y1),int(y2)
    #     x,y = int((x2 + x1)/2) , int((y2 + y1)/2)
    #
    #     text = '{},{:.2f}'.format(pred_class,pred_score)
    #     crop = img[y1:y2,x1:x2]
    #
    #     #save_file = "./test_out/demo_object-detection.png"
    #     save_file = os.path.join('test_out', os.path.splitext(os.path.basename(imgfile))[0] + "_predict.png")
    #     cv2.putText(img,text,(x,y), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0))
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
    #     cv2.imwrite(save_file,img)

    savefile = os.path.join("/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/test_out3/", os.path.splitext(os.path.basename(imgfile))[0] + "_pred.png")
    view_pred_img(img,savefile)

    print("{}:\n {}".format(imgfile,results))


# if __name__ == "__main__":
#     test()



