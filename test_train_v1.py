import random,time,datetime
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import cv2,os,sys,glob
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


d2_start = time.time()




def set_up():

    cfgFile = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    mode_config = cfgFile

    log = "model_to_train: %s ..." % mode_config
    print(log)
    print(log, file=open(log_file, "a"))

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(mode_config))

    cfg.DATASETS.TRAIN = (regist_train_name,)
    cfg.DATASETS.TEST = (regist_val_name,)  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(mode_config)  ## out_model : ./output/model_final.pth
    # cfg.MODEL.WEIGHTS = '/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/d2_object_detection/pre_trained_model/model_final_a54504.pkl'

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = model_lr

    cfg.SOLVER.MAX_ITER = (max_train_iter)  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (bach_size_per_img)  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_labels  # len(select_cats)  # 5 classes ['chair', 'table', 'swivelchair', 'sofa', 'bed']

    cfg.OUTPUT_DIR = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/2_train/1_train_2020_9_17/test6/maskrcnn_r50_{}_{}_{}_{}_{}".\
        format(model_lr,bach_size_per_img,max_train_iter,ims_per_batch,num_labels)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def register_dataset():

    if regist_train_name in DatasetCatalog._REGISTERED:
        log = 'regist_data exists before: %s , and try to del.... ' % regist_train_name
        print(log)
        print(log, file=open(log_file, "a"))
        DatasetCatalog._REGISTERED.pop(regist_train_name)
        DatasetCatalog._REGISTERED.pop(regist_val_name)

    else:
        log = 'regist_data : %s .... ' % regist_train_name
        print(log)
        print(log, file=open(log_file, "a"))
        register_coco_instances(regist_train_name, {}, train_json_path, train_images_dir)
        register_coco_instances(regist_val_name, {}, val_json_path, val_images_dir)

    train_metadata = MetadataCatalog.get(regist_train_name)
    val_metadata = MetadataCatalog.get(regist_val_name)
    trainset_dicts = DatasetCatalog.get(regist_train_name)

    return train_metadata, val_metadata


def d2_run():

    train_metadata, val_metadata = register_dataset()
    cfg = set_up()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    output_dir = cfg['OUTPUT_DIR']
    save_dir = os.path.join(output_dir,'result')
    if not os.path.exists(save_dir):os.makedirs(save_dir)

    cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set the testing threshold for this model
    cfg.DATASETS.TEST = (regist_val_name,)

    predictor = DefaultPredictor(cfg)

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


    ### evaluate
    evaluator = COCOEvaluator(regist_val_name, cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, regist_val_name)
    my_eval = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(my_eval)
    log = ("%s evaluate: \n" % (model_name), my_eval)
    print(log, file=open(log_file, "a"))


if __name__=="__main__":

    model_lr = 0.0025
    num_workers = 4
    bach_size_per_img = 512
    max_train_iter = 15000
    ims_per_batch = 6
    num_labels = 13

    work_root = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/2_train/1_train_2020_9_17/test6/class_13"
    model_name = "mask_rcnn_R_50_FPN_3x.yaml"
    log_file = os.path.join(work_root, "log_dat.txt")
    regist_train_name = "zc_train_data"
    regist_val_name = 'zc_val_1data'

    train_json_path = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_13/train.json"
    val_json_path = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_13/val.json"
    train_images_dir = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_13/train"
    val_images_dir = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_13/val"

    d2_run()