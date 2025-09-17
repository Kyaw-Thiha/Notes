# Pooling
#cv/cnn/pooling

Pooling layers shrink the spatial size of the feature map without adding new weights.

![Pooling Layer](https://media.springernature.com/m685/springer-static/image/art%3A10.1038%2Fs41598-024-51258-6/MediaObjects/41598_2024_51258_Fig1_HTML.png)

## Different Pooling Strategies
Different pooling strategies can be used to downsample the image.

![[Pooling Strategies.png]]

## Why Pooling is useful
- **Downsampling**: Used early in the `CNN` model to reduce computation.
- **Translation Invariance**: less sensitive to small shifts (if edge move by 1 pixel, it is still max)
- **Highlight Key Features**: Keep strongest activations (which are usually edges, corners & textures)
