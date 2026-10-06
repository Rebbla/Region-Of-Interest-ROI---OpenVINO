#include "yolo_ov.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

YoloOV::YoloOV(const std::string& modelPath, const std::string& device, int imgsz, float conf, float iou,
               const std::vector<std::string>& names)
    : names_(names), imgsz_(imgsz), conf_(conf), iou_(iou)
{
    const std::shared_ptr<ov::Model> model = core_.read_model(modelPath);

    const ov::PartialShape inShape = model->input().get_partial_shape();
    const bool isStatic = inShape.rank().is_static() && inShape.rank().get_length() == 4 &&
                          inShape[2].is_static() && inShape[3].is_static();
    if (isStatic)
    {
        imgsz_ = static_cast<int>(inShape[2].get_length());
    }
    else
    {
        model->reshape(ov::PartialShape{1, 3, imgsz_, imgsz_});
    }

    compiled_ = core_.compile_model(model, device);
    request_ = compiled_.create_infer_request();
}

cv::Mat YoloOV::letterbox(const cv::Mat& src, float& scale, int& padX, int& padY) const
{
    const int w = src.cols;
    const int h = src.rows;

    scale = std::min(static_cast<float>(imgsz_) / static_cast<float>(w),
                     static_cast<float>(imgsz_) / static_cast<float>(h));

    const int newW = static_cast<int>(std::round(w * scale));
    const int newH = static_cast<int>(std::round(h * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH));

    padX = (imgsz_ - newW) / 2;
    padY = (imgsz_ - newH) / 2;

    cv::Mat out(imgsz_, imgsz_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(padX, padY, newW, newH)));
    return out;
}

std::vector<Detection> YoloOV::detect(const cv::Mat& frame)
{
    float scale = 1.0f;
    int padX = 0;
    int padY = 0;
    cv::Mat letterboxed = letterbox(frame, scale, padX, padY);

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

    cv::Mat blob;
    rgb.convertTo(blob, CV_32F, 1.0 / 255.0);

    ov::Tensor input = request_.get_input_tensor();
    float* data = input.data<float>();

    const int channels = 3;
    for (int c = 0; c < channels; ++c)
    {
        for (int y = 0; y < imgsz_; ++y)
        {
            for (int x = 0; x < imgsz_; ++x)
            {
                data[c * imgsz_ * imgsz_ + y * imgsz_ + x] = blob.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    request_.infer();

    ov::Tensor output = request_.get_output_tensor();
    const ov::Shape shape = output.get_shape();
    if (shape.size() != 3)
    {
        throw std::runtime_error("Unexpected YOLO output rank");
    }

    const bool transposed = shape[1] > shape[2];
    const int channelsOut = transposed ? shape[2] : shape[1];
    const int numAnchors = transposed ? shape[1] : shape[2];
    const int numClasses = channelsOut - 4;
    if (numClasses <= 0)
    {
        throw std::runtime_error("YOLO output has no class channels");
    }

    const float* out = output.data<float>();
    auto valueAt = [&](int channel, int anchor) -> float
    {
        if (transposed)
        {
            return out[anchor * channelsOut + channel];
        }
        return out[channel * numAnchors + anchor];
    };

    const float frameW = static_cast<float>(frame.cols);
    const float frameH = static_cast<float>(frame.rows);

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    for (int i = 0; i < numAnchors; ++i)
    {
        float best = 0.0f;
        int bestId = -1;
        for (int c = 0; c < numClasses; ++c)
        {
            const float s = valueAt(4 + c, i);
            if (s > best)
            {
                best = s;
                bestId = c;
            }
        }

        if (best < conf_)
        {
            continue;
        }

        const float cx = valueAt(0, i);
        const float cy = valueAt(1, i);
        const float bw = valueAt(2, i);
        const float bh = valueAt(3, i);

        float x = (cx - bw / 2.0f - padX) / scale;
        float y = (cy - bh / 2.0f - padY) / scale;
        float w = bw / scale;
        float h = bh / scale;

        x = std::max(0.0f, std::min(x, frameW - 1.0f));
        y = std::max(0.0f, std::min(y, frameH - 1.0f));
        w = std::min(w, frameW - x);
        h = std::min(h, frameH - y);

        boxes.emplace_back(cv::Rect(cv::Point(static_cast<int>(x), static_cast<int>(y)),
                                    cv::Point(static_cast<int>(x + w), static_cast<int>(y + h))));
        confs.push_back(best);
        classIds.push_back(bestId);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confs, conf_, iou_, keep, 1.0f, 300);

    std::vector<Detection> detections;
    detections.reserve(keep.size());
    for (const int idx : keep)
    {
        detections.push_back({boxes[idx], classIds[idx], confs[idx]});
    }
    return detections;
}
