from difflib import restore
import torch
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training import models


dataset_params = {
    'data_dir':'../dataset',
    'train_images_dir':'train/images',
    'train_labels_dir':'train/labels',
    'val_images_dir':'valid/images',
    'val_labels_dir':'valid/labels',
    # 'test_images_dir':'test/images',
    # 'test_labels_dir':'test/labels',
    'classes': ['person','small-vehicle','medium-vehicle','big-vehicle','dummy']
}

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':12,
        'num_workers':4
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':12,
        'num_workers':4
    }
)
# test_data = coco_detection_yolo_format_val(
#     dataset_params={
#         'data_dir': dataset_params['data_dir'],
#         'images_dir': dataset_params['test_images_dir'],
#         'labels_dir': dataset_params['test_labels_dir'],
#         'classes': dataset_params['classes']
#     },
#     dataloader_params={
#         'batch_size':10,
#         'num_workers':2
#     }
# )
print(type(train_data.dataset.plot()))
print(type(val_data.dataset.plot()))


CHECKPOINT_DIR = 'checkpoints'
trainer = Trainer(experiment_name='drone-model', ckpt_root_dir=CHECKPOINT_DIR)


model = models.get('yolo_nas_l', 
                  num_classes=len(dataset_params['classes']), 
                  pretrained_weights="coco"
                  )


train_params = {
    # ENABLING SILENT MODE
    'silent_mode': False,
    "batch_accumulate":4,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
    "max_epochs": 100,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.3,
            top_k_predictions=100,
            # NOTE: num_classes needs to be defined here
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}

# Move the model to the GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


trainer.train(model=model, 
             training_params=train_params, 
             train_loader=train_data, 
             valid_loader=val_data,
             )

# best_model = models.get('yolo_nas_l',
#                         num_classes=len(dataset_params['classes']),
#                         checkpoint_path="checkpoints/bdd-model/ckpt_best.pth")

# best_model.to(device)


# trainer.train(model=best_model, 
#                training_params=train_params, 
#                train_loader=train_data, 
#                valid_loader=val_data,
#                )

best_model = models.get('yolo_nas_l',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="checkpoints/drone-model/ckpt_best.pth")

result =trainer.test(model=best_model,
            test_loader=val_data,
            test_metrics_list=DetectionMetrics_050(score_thres=0.5, 
                                                   top_k_predictions=300, 
                                                   num_cls=len(dataset_params['classes']), 
                                                   normalize_targets=True, 
                                                   post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                          nms_top_k=1000, 
                                                                                                          max_predictions=300,                                                                              
                                                                                                          nms_threshold=0.7)
                                                                                                          ))

print("-------------------------------------------------------")
print(result)
print("-------------------------------------------------------")
