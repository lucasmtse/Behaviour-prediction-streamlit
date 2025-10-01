# Predict Mouse Behavior — Streamlit App

## 📌 Description
This Streamlit application predicts mouse behavior from videos .mp4. 
It applies a pre-trained model to generate a CSV of body part coordinates (with DLC), then loads the extracted features, applies a pre-trained XGBoost model, and finally displays a video overlay with the predicted behaviors.

The app is designed for **single animal tracking** and outputs a CSV file with per-frame behavior predictions.

---

## Steps

1. [📹 Upload a video (.mp4 format)](#step-1--upload-a-video)  
2. [📊 Upload the corresponding behavior CSV](#step-2--upload-the-behavior-csv)  
3. [🐭 Run the DeepLabCut SuperAnimal model on the uploaded video](#step-3--run-the-deeplabcut-superanimal-model)  
4. [⚙️ Extract features from the pose estimation](#step-4--extract-features-from-the-pose-estimation)  
5. [🤖 Apply the pre-trained XGBoost model](#step-5--apply-the-pre-trained-xgboost-model)  
5. [💾 Export predictions to a CSV file](#step-5-bis--export-predictions-to-csv)  

---

### Step 1 — Upload a video
Provide an input video of the mouse in `.mp4` format.

### Step 2 — Upload the behavior CSV
You can upload two types of CSV files:  

- **Ground truth CSV (optional)**: a manually labeled file where each frame is associated with a behavior.  
- **Prediction CSV**: the output of the model, following the same structure as the ground truth file.  
> ⚠️ **Note:** This prediction file is obtained after completing all the following steps.

Both files are aligned with the video frames, and below the video you can see both annotations side by side, making it possible to visually compare manual labels with model predictions.  


### Step 3 — Run the DeepLabCut SuperAnimal model
The pose estimation is performed using the **SuperAnimal Top-View Mouse** model:  
`resnet_50` and detector `fasterrcnn_resnet50_fpn_v2`.  
The app uses the DeepLabCut SuperAnimal pre-train model to perform pose estimation on the uploaded video. It automatically detects and tracks 27 keys body parts of the mouse across frames.  


### Step 4 — Extract features from the pose estimation
From the estimated positions, the app computes higher-level features such as distances between body parts, angular changes, and movement speeds. These features are essential for behavior classification.  

- **FPS (optional)**: can be set manually to adjust time-based features; if left empty, a default value is used.  
- **Max gap fill**: controls how missing values are treat.  
  - If the gap of missing frames is smaller than the chosen threshold, missing values are filled by linear interpolation between the last valid and the next valid point.  
  - If the gap is larger, values remain missing.  
- **Bones selection**: lets you define which body part pairs (bones) to track in order to capture how their distance/angle move across frames.  
- **Angle computation**: you can choose between signed angles (which preserve orientation and direction) or unsigned angles (always positive).  


### Step 5 — Apply the pre-trained XGBoost model
A pre-trained XGBoost classifier train using `video 376`, is applied to the extracted features. It predicts the probability of each behavior (e.g., stretch posture, flight) for every frame.

### Step 5 bis — Export predictions to CSV
Once the predictions are generated, press the **Download predicted behavior** button to export them as a CSV file.  
The file contains `frame`, `time`, and binary columns for each behavior, and can be directly used for further analysis or visualization.  

This CSV can also be uploaded back in **Step 2**, allowing you to visually compare model predictions with manual scoring or directly with the video.   

> ⚠️ **Note:** When using DeepLabCut to reconstruct videos, the frames may not be perfectly aligned with the original video.  
> For example, the frame corresponding to second `1:05` in the original video may not match the same timestamp in the reconstructed video.  
> Always double-check your labels and scoring, or correct this misalignment before analysis.

## Ideas for future improvements

- **Feature engineering**  
  Currently, all distances and angles are extracted automatically. This can be improved by manually selecting the most meaningful distances and angles instead of taking every possible combination (“brute force”). A more curated set of features would likely improve model performance and reduce noise.  

- **Model generalization**  
  Unfortunately the current model performs well just on video *376* but is not well adapted to other videos. Improving generalization is crucial.  

  Possible directions:  
  - Incorporate **temporal context**: use information from past frames (e.g., last 5–10 frames) to predict the current frame.  
  - Experiment with **XGBoost using temporal features** (sliding-window features).  
  - Explore **sequential models** such as RNNs, LSTMs, or Temporal CNNs to better capture the dynamics of behavior across time.  

> ⚠️ **Note:** Always use the same camera setup across all videos to ensure consistency and reliable predictions.


## 📦 Requirements
- Recommended: **Python 3.11**  
- Conda env: **DEEPLABCUT**
- Python packages:
    - streamlit
    - xgboost



