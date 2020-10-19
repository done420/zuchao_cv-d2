import random,time,datetime
import cv2,os,sys,glob
from detectron2.data.datasets import register_coco_instances
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results,inference_on_dataset
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import build_detection_test_loader
from collections import OrderedDict
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import ColorMode
import copy,torch,logging
import numpy as np
import json

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader




def custom_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # transform_list = [T.Resize(800,600),
    #                   T.RandomFlip(prob=0.5, horizontal=True, vertical=True),
    #                   T.RandomContrast(0.8, 3),
    #                   T.RandomBrightness(0.8, 1.6),
    #                   ]

    transform_list = [#T.Resize((800, 800)),
                      T.RandomContrast(0.8, 3),
                      T.RandomBrightness(0.8, 1.6),
                      T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False)] ### 数据增强方式

    image, transforms = T.apply_transform_gens(transform_list, image) ## # 数组增强
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32")) ##转成Tensor

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2]) # 将标注转成Instance（Tensor）
    dataset_dict["instances"] = utils.filter_empty_instances(instances) ## 去除空的
    return dataset_dict



class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)



if __name__=="__main__":

    # model_lr = 0.0006
    model_lr = 0.0025
    num_workers = 2
    bach_size_per_img = 512 ### 大一点的512 视乎更加好一点。
    max_train_iter = 30000
    ims_per_batch = 6
    num_labels = 27

    classname_to_id = {'shuangrenchuang': 1, 'danrenchuang': 2, 'chaji': 3, 'danyi': 4, 'diaodeng': 5, 'chugui': 6, 'chuwugui': 7,
                       'ertongchuang': 8, 'dianshigui': 9, 'buyishafa': 10, 'yigui': 11, 'chuangtougui': 12, 'jiugui': 13, 'huwaideng': 14,
                       'xuanguangui': 15, 'bianji': 16, 'shuijingdeng': 17, 'shuzhuangtai': 18, 'weiyugui': 19, 'zhuangshigui': 20,
                       'chazhuo': 21, 'xiegui': 22, 'shafazuhe': 23, 'canzhuo': 24, 'pizhishafa': 25, 'shuzhuo': 26, 'batai': 27}


    work_root = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/2_train/6_mvp_2020_10_13"
    model_name = "mask_rcnn_R_50_FPN_3x.yaml"
    log_file = os.path.join(work_root, "log_dat.txt")
    regist_train_name = "zc_train_data"
    regist_val_name = 'zc_val_1data'

    train_json_path = "/data/1_qunosen/4_train_set/1_objetct_detection/mvp_class27_set/train.json"
    val_json_path = "/data/1_qunosen/4_train_set/1_objetct_detection/mvp_class27_set/val.json"
    train_images_dir = "/data/1_qunosen/4_train_set/1_objetct_detection/mvp_class27_set/images/train"
    val_images_dir = "/data/1_qunosen/4_train_set/1_objetct_detection/mvp_class27_set/images/val"


    register_coco_instances(regist_train_name, {}, train_json_path, train_images_dir)
    register_coco_instances(regist_val_name, {}, val_json_path, val_images_dir)

    train_metadata = MetadataCatalog.get(regist_train_name)
    val_metadata = MetadataCatalog.get(regist_val_name)
    dataset_dict = DatasetCatalog.get(regist_train_name)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (regist_train_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = model_lr
    cfg.SOLVER.MAX_ITER = max_train_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = bach_size_per_img
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_labels
    cfg.OUTPUT_DIR = "{}/maskrcnn_{}_{}_{}_{}_{}".format(work_root, num_labels, max_train_iter, bach_size_per_img, ims_per_batch, model_lr)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    ####### 训练

    # trainer = CustomTrainer(cfg)
    trainer = DefaultTrainer(cfg)
    train_data_loader = trainer.build_train_loader(cfg)

    # ######## 可视化
    # check_dir = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/2_train/5_train_2020_10_12/class31_set/check_images"
    # data_iter = iter(train_data_loader)
    # batch = next(data_iter)
    # rows, cols = 2, 2
    # plt.figure(figsize=(20,20))
    #
    # n = 0
    # for i, per_image in enumerate(batch[:4]):
    #     n+=1
    #     plt.subplot(rows, cols, i+1)
    #
    #     # Pytorch tensor is in (C, H, W) format
    #     img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
    #     img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
    #
    #     visualizer = Visualizer(img, metadata=train_metadata, scale=0.5)
    #
    #     target_fields = per_image["instances"].get_fields()
    #     labels = None
    #     vis = visualizer.overlay_instances(
    #         labels=labels,
    #         boxes=target_fields.get("gt_boxes", None),
    #         masks=target_fields.get("gt_masks", None),
    #         keypoints=target_fields.get("gt_keypoints", None),
    #     )
    #     plt.imshow(vis.get_image()[:, :, ::-1])
    #     save_file = os.path.join(check_dir,str(n)+"_check.png")
    #     # cv2.imwrite(save_file, vis.get_image())
    #     plt.savefig(save_file)


    trainer.resume_or_load(resume=False)
    trainer.train()

    output_dir = cfg['OUTPUT_DIR']
    save_dir = os.path.join(output_dir,'result')
    if not os.path.exists(save_dir):os.makedirs(save_dir)



    ##### 模型测试
    cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2 # set the testing threshold for this model
    cfg.DATASETS.TEST = (regist_val_name,)
    predictor = DefaultPredictor(cfg)


    evaluator = COCOEvaluator(regist_val_name, cfg, False, output_dir=save_dir)
    val_loader = build_detection_test_loader(cfg, regist_val_name)
    my_eval = inference_on_dataset(trainer.model, val_loader, evaluator)

    log_file = os.path.join(save_dir,"log.txt")
    print(my_eval, file=open(log_file, "a"))
    print("评估结果：\n {}".format(my_eval))


    test_dir = val_images_dir
    imgs_list = [os.path.join(test_dir, file_name) for file_name in os.listdir(test_dir) if
                     file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".bmp") or file_name.endswith(".jpeg")]

    for d in imgs_list:
        im = cv2.imread(d)
        outputs = predictor(im)
        a = outputs["instances"].pred_classes.data.cpu().numpy().tolist() if outputs["instances"].get_fields()["pred_boxes"] else None
        print(a)

        v = Visualizer(im[:, :, ::-1], metadata=train_metadata, scale=0.9, instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        predict_file = os.path.join(save_dir, os.path.splitext(os.path.basename(d))[0] + "_predict.png")

        cv2.imwrite(predict_file, v.get_image()[:, :, ::-1])

        if os.path.exists(predict_file): print("Done: %s" % predict_file)





