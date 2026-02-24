# Interpolation Methods
`Interpolation` can be used to upsample an image from smaller resolution to higher resolution.

`Nearest-Neighbour Interpolation`
Set the $n \times n$ pixels around the input pixel to have the same value as the input pixel.
Used in [[Feature Pyramid Network (FPN)]]

![Nearest Neighbour Interpolation|500](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgP0HwgbIG9YlU_NhxQqT1rFZUyHfFVeA4vGzmQ_EtVxeCNcfgvy6bFKHGCTG-XA7Ip3I5JXsvac5PL_sryLgTCoefJ23bCZyaUJ9ojAfw727NlHeZivoV9BBHfMRdhtzJMirikTxwe3Yk/s1600/Nearest1.png)

`Billinear Interpolation`
Set the pixel value to be the average of its input neighbouring pixels.

![Billinear Interpolation|500](https://i.ytimg.com/vi/NnksKpJZEkA/maxresdefault.jpg)

`Bicubic Interpolation`
Use the bicubic formula to upsample the image.

![Bicubic Interpolation](https://i0.wp.com/theailearner.com/wp-content/uploads/2018/10/Bicubic_interpolation.png?w=423&ssl=1)

`Transposed Convolution`
Apply a filter, and slide it across the image.
Then, sum all the values across the filter outputs.

![Transposed Convolution|500](https://miro.medium.com/v2/resize:fit:1400/1*YwVviBiy2qAp0CwS5CDwmA.gif)

