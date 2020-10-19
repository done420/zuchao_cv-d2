#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

import cv2
from detectron2.utils.visualizer import Visualizer

#### https://blog.csdn.net/weixin_39916966/article/details/103299051?utm_medium=distribute.pc_relevant.none-task-blog-title-3&spm=1001.2101.3001.4242


#引入以下注释
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import pycocotools
#声明类别，尽量保持
# CLASS_NAMES =["__background__",'chaji', 'shuzhuangtai', 'dianshigui', 'shuzhuo', 'diaodeng', 'batai']
CLASS_NAMES =['chaji', 'shuzhuangtai', 'dianshigui', 'shuzhuo', 'diaodeng', 'batai']


# 数据集路径
DATASET_ROOT = '/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_6/'
# ANN_ROOT = os.path.join(DATASET_ROOT, 'COCOformat')
ANN_ROOT = DATASET_ROOT
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'val')

TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
#VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
VAL_JSON = os.path.join(ANN_ROOT, 'val.json')

# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "coco_my_train": (TRAIN_PATH, TRAIN_JSON),
    "coco_my_val": (VAL_PATH, VAL_JSON),
}

#注册数据集（这一步就是将自定义数据集注册进Detectron2）
def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)


#注册数据集实例，加载数据集中的对象实例
def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")


# 注册数据集和元数据
def plain_register_dataset():
    #训练集
    DatasetCatalog.register("coco_my_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_my_train").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                    evaluator_type='coco', # 指定评估方式
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)

    #DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
    #验证/测试集
    DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_my_val").set(thing_classes=CLASS_NAMES, # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                evaluator_type='coco', # 指定评估方式
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)
# 查看数据集标注，可视化检查数据集标注是否正确，
#这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
#可选择使用此方法
def checkout_dataset_annotation(name="coco_my_train"):
    #dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH)
    # print(len(dataset_dicts))
    for i, d in enumerate(dataset_dicts,0):
        img_path = d["file_name"]
        print("img_path: {}".format(img_path))
        img = cv2.imread(img_path)


        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)

        # v = Visualizer(im[:, :, ::-1], metadata=train_metadata, scale=0.9, instance_mode=ColorMode.IMAGE_BW)
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        #cv2.imshow('show', vis.get_image()[:, :, ::-1])
        cv2.imwrite('./out/'+str(os.path.basename(img_path)),vis.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
        # if i == 200:
        #     break






class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() # 拷贝default config副本
    args.config_file = "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("coco_my_train",)
    cfg.DATASETS.TEST = ("coco_my_val",)
    cfg.DATALOADER.NUM_WORKERS = 4  # 单线程
    # cfg.INPUT.MAX_SIZE_TRAIN = 400
    # cfg.INPUT.MAX_SIZE_TEST = 400
    # cfg.INPUT.MIN_SIZE_TRAIN = (160,)
    # cfg.INPUT.MIN_SIZE_TEST = 160
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # 类别数
    cfg.MODEL.WEIGHTS = '/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/d2_object_detection/pre_trained_model/R-50.pkl'
    cfg.SOLVER.IMS_PER_BATCH = 2  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size
    ITERS_IN_ONE_EPOCH = int(300 / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1 # 12 epochs
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (70,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # print("cfg:",cfg)

    # 注册数据集
    plain_register_dataset()

    # # 检测数据集注释是否正确
    # checkout_dataset_annotation()

    # 如果只是进行评估
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )




class_list = CLASS_NAMES
regist_train_name = "coco_my_train"

def test_prepare_predictor():
    from detectron2.engine.defaults import DefaultPredictor
    # current_path = os.getcwd()
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    # print("father_path:",father_path)

    modelFile = os.path.join(father_path,'output/model_final.pth')
    cfgFile = os.path.join(father_path,"configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    num_class =len(class_list)
    print(MetadataCatalog.get(regist_train_name))

    MetadataCatalog.get(regist_train_name).thing_classes = class_list

    train_metadata = MetadataCatalog.get(regist_train_name)


    # create config
    cfg = get_cfg()
    # below path applies to current installation location of Detectron2
    cfg.merge_from_file(cfgFile)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class
    cfg.MODEL.WEIGHTS = modelFile
    # cfg.MODEL.DEVICE = "cuda" # we use a GPU
    cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy


    classes = train_metadata.get("thing_classes", None)
    predictor = DefaultPredictor(cfg)
    print("Predictor has been initialized.")
    #
    return (predictor, classes)

PREDICTOR, CLASSES = test_prepare_predictor()

def image_predict(imgfile):
    from PIL import Image
    import numpy
    from detectron2.utils.visualizer import ColorMode
    import time

    image = Image.open(imgfile)
    bgr_img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)


    num_class = len(class_list)
    # MetadataCatalog.get(regist_train_name).thing_classes = class_list
    MetadataCatalog.get(regist_train_name)
    train_metadata = MetadataCatalog.get(regist_train_name)

    start_time = time.time()
    print('预测开始')
    outputs = PREDICTOR(bgr_img)
    print('预测完成', time.time() - start_time)

    pred_scores = outputs["instances"].scores.tolist()
    pred_classes = outputs["instances"].pred_classes.data.cpu().numpy().tolist() if outputs["instances"].get_fields()["pred_boxes"] else None
    pred_bboxes = outputs["instances"].pred_boxes.tensor.tolist() if outputs["instances"].get_fields()["pred_boxes"] else None

    v = Visualizer(bgr_img[:, :, ::-1],metadata=train_metadata,scale=1,
                           instance_mode=ColorMode.IMAGE_BW)   # remove the colors of unsegmented pixels)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    print(pred_classes)
    savr_dir = "./test_out2/"
    save_file = os.path.join(savr_dir,os.path.basename(imgfile) + "_pre.png")
    cv2.imwrite(save_file,v.get_image()[:, :, ::-1])


import glob
imgs_list = glob.glob(VAL_PATH + "/*.jpg")

for img_file in imgs_list:
    print(img_file)
    image_predict(img_file)



