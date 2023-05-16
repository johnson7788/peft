#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/3/28 15:57
# @File  : owl_train.py
# @Author: 
# @Desc  :

import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    DetrForObjectDetection,
    DetrConfig,
    default_data_collator
)

# Prepare the dataset
train_data_path = "/media/wac/backup/john/johnson/.cache/coco/train2017"
train_annotations_path = "/media/wac/backup/john/johnson/.cache/coco/annotations/instances_train2017.json"
val_data_path = "/media/wac/backup/john/johnson/.cache/coco/val2017"
val_annotations_path = "/media/wac/backup/john/johnson/.cache/coco/annotations/instances_val2017.json"

train_transforms = ToTensor()
val_transforms = ToTensor()

train_dataset = CocoDetection(train_data_path, train_annotations_path, transform=train_transforms)
val_dataset = CocoDetection(val_data_path, val_annotations_path, transform=val_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=default_data_collator)
val_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=default_data_collator)

# Prepare the model and processor
# config = DetrConfig.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
# model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", config=config)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for i, (images, targets) in enumerate(train_dataloader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        outputs = model(images, targets=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item()}")

    # Validation
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_dataloader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images, targets=targets)
            loss = outputs.loss

            if (i + 1) % 10 == 0:
                print(f"Validation Step [{i+1}], Loss: {loss.item()}")

    scheduler.step()

# Save the model
torch.save(model.state_dict(), "owlvit_coco.pth")