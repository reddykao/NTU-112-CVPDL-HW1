import torchvision
import os
import numpy as np
import os
from PIL import Image, ImageDraw
import json
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection, DeformableDetrForObjectDetection
from transformers import AutoModelForObjectDetection
from transformers import AutoFeatureExtractor
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from ptflops import get_model_complexity_info
import lightning as L
import torch
import glob
from pytorch_lightning import Trainer
from transformers import DetrFeatureExtractor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import CSVLogger
import argparse
import sys
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", default='custom path',required=True,	help="output path  to the model")
ap.add_argument('-a', '--arch', default='detr', choices=['detr', 'def-detr'], help='Choose different transformer based object detection architecture')
ap.add_argument("-e", "--epochs", type=int, help="No of Epochs for training")
ap.add_argument("-r", '--profile', default=False, type=bool, help='Profiling different model')

args = vars(ap.parse_args())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "train.json" if train else "val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

class CocoDetection2(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "train.json" if train else "test.json")
        super(CocoDetection2, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection2, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img,  return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        #target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


feature_extractor = None

if args["arch"] == 'detr':
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
elif args["arch"] == 'def-detr':
    feature_extractor = AutoFeatureExtractor.from_pretrained("SenseTime/deformable-detr")
    
train_dataset = CocoDetection(img_folder='../hw1_dataset/train', feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder='../hw1_dataset/valmini', feature_extractor=feature_extractor, train=False)
test_dataset = CocoDetection2(img_folder='../hw1_dataset/test', feature_extractor=feature_extractor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))
print("Number of test examples:", len(test_dataset))


image_ids = train_dataset.coco.getImgIds()
# let's pick a random image
image_id = image_ids[np.random.randint(0, len(image_ids))]
print('Image n°{}'.format(image_id))
image = train_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join('../hw1_dataset/train', image['file_name']))

annotations = train_dataset.coco.imgToAnns[image_id]
draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

for annotation in annotations:
  box = annotation['bbox']
  class_idx = annotation['category_id']
  x,y,w,h = tuple(box)
  draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
  draw.text((x, y), id2label[class_idx], fill='white')

image.show()


def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

if args["arch"] == 'def-detr' or args["arch"] == 'detr':
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=1)


batch = next(iter(train_dataloader))
print(batch.keys())

pixel_values, target = train_dataset[0]

print(pixel_values.shape)
print(target)
class MyLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.training_step_outputs = []

    def training_step(self):
        loss = ...
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        # do something with all training_step outputs, for example:
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_mean", epoch_mean)
        # free up the memory
        self.training_step_outputs.clear()

class ObjectDetector(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay, architecture):
         super().__init__()
         # replace COCO classification head with custom head
         if architecture == 'detr':
             #self.model = MyLightningModule.load_from_checkpoint("/home/jpkao/CVPDL/object_detection/ckpt/lightning_logs/version_5/checkpoints/epoch=99-step=3800.ckpt")
             self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                                 num_labels=len(id2label),
                                                                 ignore_mismatched_sizes=True)
         elif architecture == 'cond-detr':
             self.model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50",
                                                                      id2label={0: "creatures",1:"fish",2:"jellyfish",3:"penguin",4:"puffin"
                                                                           ,5:"shark",6:"starfish",7:"stingray"},
                                                                 label2id={"creatures": 0, "fish":1,"jellyfish":2, "penguin":3,"puffin":4,
                                                                           "shark":5, "starfish":6, "stingray":7 },
                                                                      ignore_mismatched_sizes=True)
         elif architecture == 'def-detr':
             self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr",
                                                                 id2label={0: "creatures",1:"fish",2:"jellyfish",3:"penguin",4:"puffin"
                                                                           ,5:"shark",6:"starfish",7:"stingray"},
                                                                 label2id={"creatures": 0, "fish":1,"jellyfish":2, "penguin":3,"puffin":4,
                                                                           "shark":5, "starfish":6, "stingray":7 },
                                                                 ignore_mismatched_sizes=True)
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict
       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss
        print("validation_loss", loss)

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)
        
        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader
     def save_model(self, path):
        self.model.save_pretrained(path)


arch = args["arch"]
model = None
output = None

if arch == 'detr' or arch == 'cond-detr' or arch == 'def-detr':
    model = ObjectDetector(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, architecture=arch) 
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])


class MyLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.training_step_outputs = []

    def training_step(self):
        loss = ...
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        # do something with all training_step outputs, for example:
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_mean", epoch_mean)
        # free up the memory
        self.training_step_outputs.clear()



trainer = Trainer(accelerator='gpu', devices=1, gradient_clip_val=0.1,max_epochs=50)
trainer.fit(model, ckpt_path="custom path")

model_path = "{}".format(args["arch"])
outdir = args["path"]

path = os.path.join(outdir, model_path)
print("path {}".format(path))

if not os.path.exists(path):
    os.makedirs(os.path.join(outdir, model_path))

model.save_model(path)
feature_extractor.save_pretrained(path)
#from detr.datasets import get_coco_api_from_dataset

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
base_ds = get_coco_api_from_dataset(val_dataset) # this is actually just calling the coco attribute

from coco_eval import CocoEvaluator
from tqdm import tqdm

iou_types = ['bbox']
coco_evaluator = CocoEvaluator(base_ds, iou_types) # initialize evaluator with ground truths

model.to(device)
model.eval()

#======= code below is modified from my train.py for generating test.json and visualization output
#======= Warning: sould prepare test.son including image name, id and the pixel size of image
#------- The format of json file can modify directly from val.json
print("=======================Running Testing=================================")

#=======================================
class val_loader(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder)
        super(val_loader, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(val_loader, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


#--------------------------------------------
#============================================================================================

#==================================================================

#===================================================

#Custom script for generating subimission valid.json

#===================================================

#==================================================================

#============================================================================================

#-----------------------------------
model.to(device)
model.eval()

#========choose the figure you like to add detection boxes by id in test.json
pixel_values, target = test_dataset[28]

pixel_values = pixel_values.unsqueeze(0).to(device)
print("pixel_values.shape: {}".format(pixel_values.shape))

if args["profile"]:
    _, _, width, height = pixel_values.shape
    print("Profiling: Input width = {}, height = {}".format(width, height))
    input = (3, width, height)
    print("=====START Profile With PTFLOPS========")
    macs, params = get_model_complexity_info(model.model, input, as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("=====END Profile With PTFLOPS========")


    # forward pass to get class logits and bounding boxes
np.set_printoptions(threshold=sys.maxsize)

outputs = model.model(pixel_values=pixel_values, pixel_mask=None)


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def plot_output(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


#====== draw the prediction boxes with the image
def visualize_predictions(image, outputs, threshold=0.9):
  # keep only predictions with confidence >= threshold
  probas = outputs.logits.softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold
  
  # convert predicted boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

  # plot results
  plot_results(pil_img=image, prob=probas[keep], boxes=bboxes_scaled)

#===== open the image same as the id set line 385
image = Image.open("custom path")

if args['arch'] == 'detr' or args['arch'] == 'yolos' or args['arch']=='def-detr':
    visualize_predictions(image, outputs)

