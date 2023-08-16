# Drone-object-detection

## Model Training (Yolo-nas):

### Output

<p align="center">
  <img src="Output/drone-sample.gif" alt="output">
</p>

# Steps :

	  1.Data gathering
	
	  2.Data Preprocessing
	
	  3.Model Training (yolo-nas)
 
    4.Model evaluation
  
	  5.Run the infrence on video
	
	
3. Model training (yolo-nas)

	* run the command : python train.py
	
	Note: You might need to change the path of the dataset as per your need.
	
	* After training the model file will be stored in checkpoint directory.

4. Model Evaluation:
    
  * run the command : python evaluate.py

  * After execution of this command you'll get charts of models performance
	
5. Run the infrence:

	* run the command : python infrence.py
	
	Note: you might need to change the path to the model and video.
	

# Model Evaluation:

<img src="https://github.com/jayvaghasiya/Drone-object-detection/blob/main/Evaluation-curves/F1_Score.jpg" alt="output" height=350 width=350>
<img src="https://github.com/jayvaghasiya/Drone-object-detection/blob/main/Evaluation-curves/Precision_Score.jpg" alt="output" height=350 width=350>
<img src="https://github.com/jayvaghasiya/Drone-object-detection/blob/main/Evaluation-curves/Recall_Score.jpg" alt="output" height=350 width=350>
<img src="https://github.com/jayvaghasiya/Drone-object-detection/blob/main/Evaluation-curves/Precision_Recall.jpg" alt="output" height=350 width=350>
<img src="https://github.com/jayvaghasiya/Drone-object-detection/blob/main/Evaluation-curves/mAP_Score" alt="output" height=350 width=350>

