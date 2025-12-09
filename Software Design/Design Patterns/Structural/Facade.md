# Facade

> [!summary] Main Idea
> Client talks to 1 easy class, while the complexity of the module is hidden behind it.

---
`Example`

```java
class VideoFile {}
class Codec {}
class CodecFactory {
    static Codec extract(VideoFile file) { return new Codec(); }
}
class BitrateReader {
    static VideoFile read(VideoFile file, Codec codec) { return file; }
    static VideoFile convert(VideoFile buffer, Codec codec) { return buffer; }
}
class AudioMixer {
    VideoFile fix(VideoFile result) { return result; }
}
```

```java
class VideoConversionFacade {

    public VideoFile convertVideo(String filename, String format) {
        System.out.println("Converting video...");

        VideoFile file = new VideoFile();
        Codec sourceCodec = CodecFactory.extract(file);
        Codec destinationCodec = new Codec();

        VideoFile buffer = BitrateReader.read(file, sourceCodec);
        VideoFile intermediate = BitrateReader.convert(buffer, destinationCodec);

        AudioMixer mixer = new AudioMixer();
        VideoFile result = mixer.fix(intermediate);

        System.out.println("Conversion complete!");
        return result;
    }
}
```

---
`ML Example`

```c
#include <opencv2/opencv.hpp>

class Preprocessor {
public:
    cv::Mat preprocess(const cv::Mat& img) {
        cv::Mat resized, floatImg;
        cv::resize(img, resized, cv::Size(224, 224));
        resized.convertTo(floatImg, CV_32F, 1.0f/255);
        return floatImg;
    }
};

```

```c
#include <opencv2/dnn.hpp>

class Model {
private:
    cv::dnn::Net net;

public:
    Model() {
        net = cv::dnn::readNetFromONNX("classifier.onnx");
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }

    cv::Mat infer(const cv::Mat& input) {
        cv::Mat blob = cv::dnn::blobFromImage(input);
        net.setInput(blob);
        return net.forward();   // logits
    }
};
```

```c
#include <vector>
#include <algorithm>

class Postprocessor {
public:
    int getTopPrediction(const cv::Mat& logits) {
        cv::Point classIdPoint;
        double maxVal;
        cv::minMaxLoc(logits, 0, &maxVal, 0, &classIdPoint);
        return classIdPoint.x; // index of best class
    }
};
```

Facade Class
```c
class VisionInferenceFacade {
private:
    Preprocessor pre;
    Model model;
    Postprocessor post;

public:
    int predict(const cv::Mat& img) {
        cv::Mat input = pre.preprocess(img);
        cv::Mat logits = model.infer(input);
        return post.getTopPrediction(logits);
    }
};
```

---
## See Also
- [[Important Design Patterns]]