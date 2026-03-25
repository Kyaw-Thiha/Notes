# Convolutional Neural Network
#cv/cnn 

## Motivation
In early attempts of training an image with a [[Neural Network|fully connected neural network]], researches would face $2$ problems:
- A $100 \times 100$ image in the first layer would yield $10,000$ neurons
- Vanishing gradient problems

---
### Depth Increases Receptive Fields
Nodes further from the input have larger receptive fields.

Earlier layers can be thought of as low-level features such as edges of an image.
Later layers are more semantic like exact objects.

### Pooling Provides Shift Invariance
Pooling makes the representation invariant to small translations.
This is useful since presence of a field is more relevant than the precise location.

[[Pooling|Read More]]