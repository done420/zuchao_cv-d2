import random,time,datetime
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import cv2,os,sys
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


def d2_train_model(regist_train_name, regist_val_name, train_json_path,train_images_dir,val_json_path,val_images_dir,ims_per_batch, model_lr, bach_size_per_img, max_train_iter,num_workers,num_labels):
    ## 1. models:
    model_name = "mask_rcnn_R_50_FPN_3x.yaml"
    cfgFile = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    # work_root = os.getcwd()
    work_root = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/2_train/1_train_2020_9_17/test6"

    log_file = os.path.join(work_root, "log_dat.txt")

    d2_start = time.time()
    datetime_now = datetime.datetime.now()
    log = ("###" * 100 + "\n") * 5 + " %s\n" % (str(datetime_now)) + "model_name: %s  ..." % model_name
    print(log)
    print(log, file=open(log_file, "a"))

    log = "parameter setting:\n model_to_try:%s\n num_labels: %d\n ims_per_batch:%d\n num_workers:%d\n model_lr:%s\n max_train_iter:%d\n bach_size_per_img:%d\n" % \
          (model_name, num_labels, ims_per_batch, num_workers, str(model_lr), max_train_iter, bach_size_per_img)

    print(log)
    print(log, file=open(log_file, "a"))

    new_root = os.path.join(work_root, str(model_name) + "_%s_%s_%s_%s" % (
    str(model_lr), str(bach_size_per_img), str(max_train_iter), str(ims_per_batch)))

    if not os.path.exists(new_root): os.makedirs(new_root)
    os.chdir(new_root)

    # register_coco_instances(regist_train_name, {}, train_json_path, train_images_dir)
    # register_coco_instances(regist_val_name, {}, val_json_path, val_images_dir)



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



    ################
    # #### trainning:
    cfg = get_cfg()
    mode_config = cfgFile
    log = "model_to_train: %s ..." % mode_config
    print(log)
    print(log, file=open(log_file, "a"))

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
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    model_path = os.path.join(new_root, 'output/model_final.pth')

    if os.path.exists(model_path):
        log = "model_save: %s" % model_path
        print(log)
        print(log, file=open(log_file, "a"))

        #### predict
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set the testing threshold for this model
        cfg.DATASETS.TEST = (regist_val_name,)
        predictor = DefaultPredictor(cfg)


        out_model_dir = os.path.join(new_root, "output")
        out_dir = os.path.join(out_model_dir, 'result_' + str(model_name))
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        test_dir = val_images_dir  # os.path.join(work_dir,"./val_images")
        # test_dir = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_6/val"
        imgs_list = [os.path.join(test_dir, file_name) for file_name in os.listdir(test_dir) if
                     file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".bmp")
                     or file_name.endswith(".jpeg")]

        for d in imgs_list:
            im = cv2.imread(d)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1], metadata=train_metadata, scale=0.9, instance_mode=ColorMode.IMAGE_BW)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            predict_file = os.path.join(out_dir, os.path.splitext(os.path.basename(d))[0] + "_predict.png")

            cv2.imwrite(predict_file, v.get_image()[:, :, ::-1])

            if os.path.exists(predict_file):
                print("Done: %s" % predict_file)

        #### evaluate
        evaluator = COCOEvaluator(regist_val_name, cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(cfg, regist_val_name)
        my_eval = inference_on_dataset(trainer.model, val_loader, evaluator)
        print(my_eval)
        log = ("%s evaluate: \n" % (model_name), my_eval)
        print(log, file=open(log_file, "a"))


        ###############
        DatasetCatalog._REGISTERED.pop(regist_train_name)
        DatasetCatalog._REGISTERED.pop(regist_val_name)

        log = "clean regist_data: %s and %s" % (regist_train_name, regist_val_name)
        print(log)
        print(log, file=open(log_file, "a"))

        d2_end = time.clock()
        log = "model %s : it takes %s ." % (model_name, str(d2_end - d2_start))
        print(log)
        print(log, file=open(log_file, "a"))

        os.chdir(work_root)

    else:
        print("NotFound: {}".format(model_path))




if __name__ == "__main__":
    model_lr = 0.025
    num_workers = 4
    bach_size_per_img = 128
    max_train_iter = 10000
    ims_per_batch = 6
    num_labels = 13

    regist_train_name = 'zc_1data'
    regist_val_name = 'zc_val_1data'


    # train_json_path = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_6/train.json"
    # val_json_path = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_6/val.json"
    # train_images_dir = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_6/train"
    # val_images_dir = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_6/val"


    # train_json_path = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_6/new_train.json"
    # val_json_path = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_6/new_val.json"
    # train_images_dir = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_6/train"
    # val_images_dir = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_6/val"


    train_json_path = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_13/train.json"
    val_json_path = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_13/val.json"
    train_images_dir = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_13/train"
    val_images_dir = "/home/user/qunosen/2_project/4_train/2_zhuchao/6_d2_final_train/1_data/1_data_2020_9_17/new_set_13/val"

    d2_train_model(regist_train_name, regist_val_name,
                   train_json_path, train_images_dir,
                   val_json_path, val_images_dir,
                   ims_per_batch, model_lr, bach_size_per_img,
                   max_train_iter, num_workers, num_labels)

d2_end = time.time()
print("It takes %s ." % str(d2_end - d2_start))


