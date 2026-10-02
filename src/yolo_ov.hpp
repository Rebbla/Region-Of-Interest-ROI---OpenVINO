#pragma once

#include <openvino/openvino.hpp>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct Detection
{
    cv::Rect box;
    int classId = -1;
    float conf = 0.0f;
};

class YoloOV
{
public:
    YoloOV(const std::string& modelPath, const std::string& device, int imgsz, float conf, float iou,
           const std::vector<std::string>& names);

    std::vector<Detection> detect(const cv::Mat& frame);

    const std::vector<std::string>& names() const { return names_; }

private:
    cv::Mat letterbox(const cv::Mat& src, float& scale, int& padX, int& padY) const;

    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest request_;
    std::vector<std::string> names_;
    int imgsz_ = 640;
    float conf_ = 0.5f;
    float iou_ = 0.45f;
};
