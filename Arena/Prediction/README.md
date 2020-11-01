# Realtime Detection and Movement Prediction Module
Final BA
This part of the system uses a YOLOv4 object detector to find the bounding box of the pogona's head in the arena. The detections trajectories are then used to predict the future trajectories, and alert the system when the lizard is going to attack the screen.

## Jupyter notebooks
Several jupyter notebooks with usage examples reside in the notebooks folder.

- notebooks/yolo_eval.ipynb - YOLOv4 evaluation notebook
- notebooks/predictor_train.ipynb - Training code for trajectory predictor models
- notebooks/data_collection.ipynb - Analyze new experiment data
- notebooks/predictor_evaluate.ipynb - Various trajectory predictors evaluation and visualization methods

## Project report
a PDF report detailing the project process, various methodological and technical considerations, and related literature is available in the file project-report.pdf.