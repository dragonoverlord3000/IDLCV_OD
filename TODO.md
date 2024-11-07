## Split data into training, testing and validation 
# Thoughts - just do it
# 70% (Train), 15% (Val), 15%(Test) split
DONE!!!!!!!!!!!!
_________________________

## 2.1 Build  a  convolutional  neural  network  to  classify  object  proposals  (N+1classes)
# Thoughts: 2 classes

## 2.2 Build a dataloader for the object detection task.  Think about the classimbalance issue of the background proposal
# Thoughts: Dataloader to split between background and foreground
# 64 batch size
# 48 background (75%)
# 16 foreground (25%)

## 2.3 Finetune the network on the training set
# Just do it

## 2.4 Evaluate the classification accuracy of the network on the validation set.
## Note that this is different from the evaluation of the object detection task
# Thoughts: Use background and foreground as labels and see if they are correctly classified
# F1, Precision, Recall, Accuracy
# 
____________________

## 3.1 Apply the CNN that you trained on the test image

## 3.2  Implement and apply NMS to discard overlapping boxes

## 3.3  Evaluate  the  object  detection  output  using  the  Average  Precision  (AP)metric
# 
