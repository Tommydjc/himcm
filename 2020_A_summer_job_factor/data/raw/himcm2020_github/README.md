# Multiclass classifiers

## Requirements
* numpy
* matplotlib
* sklearn

## Results
1. K-means (L2)
  <div align="center">
    <img src="k_means.png" alt="K-means" width="400px" />
  </div>
2. Softmax linear classifier
  <div align="center">
    <img src="softmax.png" alt="Softmax linear classifier" width="400px" />
  </div>
3. 2 layers neuronal network (ReLU)
  <div align="center">
    <img src="net.png" alt="2 layers neuronal network" width="400px" />
  </div>





Document description：

DATA:

OriginalData.txt:  Processed data from questionnaire survey

W_matrix.txt:   Weight matrix trained by linear model

work_choice.txt： Each of the 50 students chose the job category, which is the label of training

entropy_vector.txt: Trained entropy vector

outfile_i : Visualization of parameter matrix corresponding to the i-1 th job



CODE:

entropy.py:  Calculate the entropy matrix of the data

draw_res.py:  Visualization of parameter matrix

Models.py:   LogisticRegression Model 







