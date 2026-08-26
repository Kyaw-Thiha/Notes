# Transfer Learning
[[Transfer Learning|Transfer learning]] is when a [[Neural Network|model]] trained on one task is reused as starting point for a new task.

![400](https://analyticsstepsfiles.s3.ap-south-1.amazonaws.com/backend/media/thumbnail/1967565/9315476_1592890541_transfer.jpg)

---
## Feature Transfer
In [[Convolutional Neural Network (CNN)|CNN]], we note that [[Convolution Layer|convolution layers]] can learn feature representations of the data, while the [[Neural Network|fully connected layer]] learn to classify them.

So, we could freeze these learned embeddings, and retrain the classifier head on the new tasks instead.

---
## Fine-tuning
Alternatively, we could train some or all of the model's weights on new task using a [[Learning Rate|lower learning rate]].

![300](https://i.redd.it/4p3j8qej1nn91.png)

---
