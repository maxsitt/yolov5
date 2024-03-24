# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 classification model on a classification dataset.

Usage:
    $ bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
    $ python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate ImageNet

Usage - formats:
    $ python classify/val.py --weights yolov5s-cls.pt                 # PyTorch
                                       yolov5s-cls.torchscript        # TorchScript
                                       yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                       yolov5s-cls_openvino_model     # OpenVINO
                                       yolov5s-cls.engine             # TensorRT
                                       yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                       yolov5s-cls_saved_model        # TensorFlow SavedModel
                                       yolov5s-cls.pb                 # TensorFlow GraphDef
                                       yolov5s-cls.tflite             # TensorFlow Lite
                                       yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                       yolov5s-cls_paddle_model       # PaddlePaddle

---

Source:       https://github.com/maxsitt/yolov5
License:      GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Modified by:  Maximilian Sittinger (https://github.com/maxsitt)
Docs:         https://maxsitt.github.io/insect-detect-docs/

Modifications:
- add additional option (argparse argument):
  '--task' use '--task val' (default) to validate on the dataset validation split
           and '--task test' to validate on the dataset test split
- save validation results to '{save_dir}/valid_results_{task}.csv' (top1 + top5 accuracy)
- save validation metrics to '{save_dir}/valid_metrics_{task}.csv' (precision, recall, f1-score)
- plot validation results as confusion matrix and save to '{save_dir}/confusion_matrix_{task}.png'
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import create_classification_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_img_size,
    check_requirements,
    colorstr,
    increment_path,
    print_args,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    data=ROOT / "../datasets/mnist",  # dataset dir
    weights=ROOT / "yolov5s-cls.pt",  # model.pt path(s)
    batch_size=128,  # batch size
    imgsz=224,  # inference size (pixels)
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    verbose=False,  # verbose output
    project=ROOT / "runs/val-cls",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    criterion=None,
    pbar=None,
    task="val",  # val or test dataset split for validation
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Dataloader
        data = Path(data)
        task = task if task in ("val", "test") else "val"
        if task == "val":
            test_dir = data / "val" if (data / "val").exists() else data / "test"  # data/val or data/test
        if task == "test":
            test_dir = data / "test" if (data / "test").exists() else data / "val"  # data/test or data/val
        dataloader = create_classification_dataloader(
            path=test_dir, imgsz=imgsz, batch_size=batch_size, augment=False, rank=-1, workers=workers
        )

    model.eval()
    pred, targets, loss, dt = [], [], 0, (Profile(device=device), Profile(device=device), Profile(device=device))
    n = len(dataloader)  # number of batches
    action = "validating" if dataloader.dataset.root.stem == "val" else "testing"
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"
    bar = tqdm(dataloader, desc, n, not training, bar_format=TQDM_BAR_FORMAT, position=0)
    with torch.cuda.amp.autocast(enabled=device.type != "cpu"):
        for images, labels in bar:
            with dt[0]:
                images, labels = images.to(device, non_blocking=True), labels.to(device)

            with dt[1]:
                y = model(images)

            with dt[2]:
                pred.append(y.argsort(1, descending=True)[:, :5])
                targets.append(labels)
                if criterion:
                    loss += criterion(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()

    if pbar:
        pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"
    if verbose:  # all classes
        LOGGER.info(f"{'Class':>24}{'Images':>12}{'top1_acc':>12}{'top5_acc':>12}")
        LOGGER.info(f"{'all':>24}{targets.shape[0]:>12}{top1:>12.3g}{top5:>12.3g}")
        for i, c in model.names.items():
            acc_i = acc[targets == i]
            top1i, top5i = acc_i.mean(0).tolist()
            LOGGER.info(f"{c:>24}{acc_i.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}")

        # Print results
        t = tuple(x.t / len(dataloader.dataset.samples) * 1e3 for x in dt)  # speeds per image
        shape = (1, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}" % t)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    if not training:
        # Write number of images and top1 + top5 acc per class to list
        images_list = [targets.shape[0]]
        top1_acc_list = [round(top1, 3)]
        top5_acc_list = [round(top5, 3)]

        class_names = list(model.names.values())
        for i, c in enumerate(class_names):
            acc_i = acc[targets == i]
            top1i, top5i = acc_i.mean(0).tolist()
            images_list.append(acc_i.shape[0])
            top1_acc_list.append(round(top1i, 3))
            top5_acc_list.append(round(top5i, 3))

        # Write results to .csv
        df_results = pd.DataFrame(
            {"class": ["all"] + class_names,
             "images": images_list,
             "top1_acc": top1_acc_list,
             "top5_acc": top5_acc_list
            })
        df_results.to_csv(save_dir / f"valid_results_{task}.csv", index=False)

        # Write true classes and predicted classes to list
        classes_true = targets.tolist()
        classes_pred = [label[0] for label in pred.tolist()]

        # Write metrics to .csv
        report = classification_report(classes_true, classes_pred, target_names=class_names, output_dict=True)
        df_metrics = (pd.DataFrame(report).transpose()
                                          .drop(["accuracy"])
                                          .rename({"macro avg": "all", "weighted avg": "all_weighted"})
                                          .rename(columns={"support": "images"})
                                          .astype({"images": int})
                                          .round(3))
        df_metrics_all = df_metrics.loc[["all", "all_weighted"]]
        df_metrics = (pd.concat([df_metrics_all, df_metrics.drop(["all", "all_weighted"])])
                        .reset_index(names="class"))
        df_metrics = df_metrics[["class", "images", "precision", "recall", "f1-score"]]
        df_metrics.to_csv(save_dir / f"valid_metrics_{task}.csv", index=False)

        # Plot results as confusion matrix
        number_classes = len(class_names)
        if number_classes >= 25:
            font_size, font_size_values = 8, 3
        elif 20 <= number_classes < 25:
            font_size, font_size_values = 10, 4
        elif 15 <= number_classes < 20:
            font_size, font_size_values = 10, 6
        elif 10 <= number_classes < 15:
            font_size, font_size_values = 10, 8
        elif 5 <= number_classes < 10:
            font_size, font_size_values = 12, 10
        else:
            font_size, font_size_values = 12, 12
        cf_matrix = np.around(confusion_matrix(classes_true, classes_pred, normalize="true"), 3)
        cf_matrix_plot = ConfusionMatrixDisplay(cf_matrix, display_labels=class_names)
        plt.rcParams.update({"font.size": font_size})
        cf_matrix_plot.plot(cmap="Blues", xticks_rotation="vertical", values_format=".2g")
        for values in cf_matrix_plot.text_.ravel():
            values.set_fontsize(font_size_values)
        cf_matrix_plot.ax_.set_title("Normalized confusion matrix")
        plt.savefig(save_dir / f"confusion_matrix_{task}.png", dpi=600, bbox_inches="tight")
        plt.close()

    return top1, top5, loss


def parse_opt():
    """Parses and returns command line arguments for YOLOv5 model evaluation and inference settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "../datasets/mnist", help="dataset path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-cls.pt", help="model.pt path(s)")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="inference size (pixels)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--verbose", nargs="?", const=True, default=True, help="verbose output")
    parser.add_argument("--project", default=ROOT / "runs/val-cls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--task", default="val", help="val or test dataset split for validation")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes the YOLOv5 model prediction workflow, handling argument parsing and requirement checks."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
