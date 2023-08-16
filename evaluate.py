from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import  coco_detection_yolo_format_val
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training import models
import matplotlib.pyplot as plt
import numpy as np


dataset_params = {
    'data_dir':'../dataset',
    'test_images_dir':'test/images',
    'test_labels_dir':'test/labels',
    # 'classes' : ['Pedestrian','Cyclist','Car','Van','Truck','Tram','Person_sitting','Misc']
    'classes': ['person','small-vehicle','medium-vehicle','large-vehicle','dummy']
}

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':8,
        'num_workers':2
    }
)

CHECKPOINT_DIR = 'checkpoints'
trainer = Trainer(experiment_name='model-eval', ckpt_root_dir=CHECKPOINT_DIR)


best_model = models.get('yolo_nas_l',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="checkpoints/bdd-model/ckpt_best.pth")



def eval_model(confidence_threshold):
    result =trainer.test(model=best_model,
                test_loader=test_data,
                test_metrics_list=DetectionMetrics_050(score_thres=confidence_threshold, 
                                                    top_k_predictions=300, 
                                                    num_cls=len(dataset_params['classes']), 
                                                    normalize_targets=True, 
                                                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                            nms_top_k=1000, 
                                                                                                            max_predictions=300,                                                                              
                                                                                                            nms_threshold=0.7)
                                                                                                            ))
    return result



confidence_thresholds = np.arange(0.1, 1.0, 0.05)
f1_scores = []
map_scores = []
recall_scores = []
precision_scores=[]
for confidence_threshold in confidence_thresholds:
    result = eval_model(confidence_threshold)
    print(result)
    f1_scores.append(result['F1@0.50'].item())
    map_scores.append(result['mAP@0.50'].item())
    recall_scores.append(result['Recall@0.50'].item())
    precision_scores.append(result['Precision@0.50'].item())

#F1 Score
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.plot(confidence_thresholds,f1_scores)
ax1.set_xlabel('Confidence Threshold')
ax1.set_ylabel('F1 Score')
ax1.set_title('F1 Curve')
plt.savefig('F1_Score.jpg')
# mAP Score
fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.plot(confidence_thresholds,map_scores)
ax2.set_xlabel('Confidence Threshold')
ax2.set_ylabel('mAP Score')
ax2.set_title('mAP Curve')
plt.savefig('mAP_Score.jpg')

fig3 = plt.figure()
ax3 = fig3.add_subplot()
ax3.plot(confidence_thresholds,recall_scores)
ax3.set_xlabel('Confidence Threshold')
ax3.set_ylabel('Recall Score')
ax3.set_title('Recall Curve')
plt.savefig('Recall_Score.jpg')

fig4 = plt.figure()
ax4= fig4.add_subplot()
ax4.plot(confidence_thresholds,precision_scores)
ax4.set_xlabel('Confidence Threshold')
ax4.set_ylabel('Precision Score')
ax4.set_title('Precision Curve')
plt.savefig('Precision_Score.jpg')

fig5 = plt.figure()
ax5= fig5.add_subplot()
ax5.plot(precision_scores,recall_scores)
ax5.set_xlabel('Recall Score')
ax5.set_ylabel('Precision Score')
ax5.set_title('Precision-Recall Curve')
plt.savefig('Precision_Recall.jpg')

plt.show()
