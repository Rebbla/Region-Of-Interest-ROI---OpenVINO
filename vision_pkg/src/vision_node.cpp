#include "roi.hpp"
#include "yolo_ov.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<std::string> parseNames(const YAML::Node& node)
{
    std::vector<std::string> names;
    if (!node)
    {
        return names;
    }
    if (node.IsMap())
    {
        int maxKey = -1;
        for (const auto& kv : node)
        {
            maxKey = std::max(maxKey, kv.first.as<int>());
        }
        names.resize(maxKey + 1);
        for (const auto& kv : node)
        {
            names[kv.first.as<int>()] = kv.second.as<std::string>();
        }
    }
    else if (node.IsSequence())
    {
        for (const auto& v : node)
        {
            names.push_back(v.as<std::string>());
        }
    }
    return names;
}

static std::vector<std::string> loadNames(const std::string& modelPath)
{
    const fs::path metadata = fs::path(modelPath).parent_path() / "metadata.yaml";
    if (!fs::exists(metadata))
    {
        return {};
    }
    try
    {
        const YAML::Node meta = YAML::LoadFile(metadata.string());
        return parseNames(meta["names"]);
    }
    catch (const std::exception&)
    {
        return {};
    }
}

static std::vector<cv::Point2f> roiFromFlat(const std::vector<double>& flat)
{
    std::vector<cv::Point2f> pts;
    for (std::size_t i = 0; i + 1 < flat.size(); i += 2)
    {
        pts.emplace_back(static_cast<float>(flat[i]), static_cast<float>(flat[i + 1]));
    }
    return pts;
}

class VisionNode : public rclcpp::Node
{
public:
    VisionNode() : Node("vision_node")
    {
        // --- Load config file (default: share/vision_pkg/config/vision.yaml) ---
        // Nilai dari config dipakai HANYA jika parameter tidak di-override via CLI/launch.
        const std::string configFile = this->declare_parameter<std::string>("config_file", "");
        if (!configFile.empty())
        {
            configFile_ = configFile;
        }
        else
        {
            try
            {
                configFile_ =
                    ament_index_cpp::get_package_share_directory("vision_pkg") + "/config/vision.yaml";
            }
            catch (const std::exception&)
            {
                configFile_.clear();
            }
        }

        YAML::Node cfg;
        if (!configFile_.empty() && fs::exists(configFile_))
        {
            try
            {
                cfg = YAML::LoadFile(configFile_);
                if (cfg["vision_node"] && cfg["vision_node"]["ros__parameters"])
                {
                    cfg = cfg["vision_node"]["ros__parameters"];
                }
            }
            catch (const std::exception& e)
            {
                RCLCPP_WARN(this->get_logger(), "Gagal baca config '%s': %s", configFile_.c_str(),
                            e.what());
                cfg = YAML::Node();
            }
        }

        // Declare semua parameter supaya get_parameter() valid.
        this->declare_parameter<std::string>("image_topic", "/zed/zed_node/rgb/image_rect_color");
        this->declare_parameter<std::string>("model_path", "");
        this->declare_parameter<std::string>("device", "CPU");
        this->declare_parameter<int>("imgsz", 640);
        this->declare_parameter<double>("conf", 0.5);
        this->declare_parameter<double>("iou", 0.45);
        this->declare_parameter<bool>("publish_debug", true);
        this->declare_parameter<std::string>("debug_qos", "reliable");
        this->declare_parameter<std::vector<double>>("roi_active", std::vector<double>{});
        this->declare_parameter<std::vector<double>>("roi_guard", std::vector<double>{});

        const auto& overrides = this->get_node_parameters_interface()->get_parameter_overrides();
        auto isOverridden = [&](const std::string& name) { return overrides.count(name) > 0; };

        auto strVal = [&](const std::string& name, const std::string& def) -> std::string {
            if (isOverridden(name))
            {
                return this->get_parameter(name).get_value<std::string>();
            }
            if (cfg && cfg[name])
            {
                return cfg[name].as<std::string>();
            }
            return def;
        };
        auto intVal = [&](const std::string& name, int def) -> int {
            if (isOverridden(name))
            {
                return this->get_parameter(name).get_value<int>();
            }
            if (cfg && cfg[name])
            {
                return cfg[name].as<int>();
            }
            return def;
        };
        auto dblVal = [&](const std::string& name, double def) -> double {
            if (isOverridden(name))
            {
                return this->get_parameter(name).get_value<double>();
            }
            if (cfg && cfg[name])
            {
                return cfg[name].as<double>();
            }
            return def;
        };
        auto boolVal = [&](const std::string& name, bool def) -> bool {
            if (isOverridden(name))
            {
                return this->get_parameter(name).get_value<bool>();
            }
            if (cfg && cfg[name])
            {
                return cfg[name].as<bool>();
            }
            return def;
        };
        auto roiVal = [&](const std::string& name) -> std::vector<double> {
            if (isOverridden(name))
            {
                return this->get_parameter(name).get_value<std::vector<double>>();
            }
            if (cfg && cfg[name])
            {
                return cfg[name].as<std::vector<double>>();
            }
            return {};
        };

        const std::string imageTopic = strVal("image_topic", "/zed/zed_node/rgb/image_rect_color");
        const std::string device = strVal("device", "CPU");
        const int imgsz = intVal("imgsz", 640);
        const double conf = dblVal("conf", 0.5);
        const double iou = dblVal("iou", 0.45);
        const std::string debugQos = strVal("debug_qos", "reliable");
        publishDebug_ = boolVal("publish_debug", true);
        modelPath_ = strVal("model_path", "");

        if (modelPath_.empty())
        {
            modelPath_ =
                ament_index_cpp::get_package_share_directory("vision_pkg") + "/models/best.xml";
        }

        names_ = loadNames(modelPath_);
        roiActive_ = roiFromFlat(roiVal("roi_active"));
        roiGuard_ = roiFromFlat(roiVal("roi_guard"));

        // Kalau ROI kosong, jangan buang semua deteksi: anggap seluruh frame sebagai guard.
        if (roiActive_.size() < 3 && roiGuard_.size() < 3)
        {
            roiGuard_ = {cv::Point2f(0.0f, 0.0f), cv::Point2f(1.0f, 0.0f),
                         cv::Point2f(1.0f, 1.0f), cv::Point2f(0.0f, 1.0f)};
            RCLCPP_WARN(this->get_logger(),
                        "ROI kosong -> pakai seluruh frame sebagai guard (deteksi tidak difilter)");
        }
        filterByRoi_ = roiActive_.size() >= 3 || roiGuard_.size() >= 3;

        RCLCPP_INFO(this->get_logger(), "Config : %s", configFile_.c_str());
        RCLCPP_INFO(this->get_logger(), "Model  : %s", modelPath_.c_str());
        RCLCPP_INFO(this->get_logger(), "Device : %s", device.c_str());
        RCLCPP_INFO(this->get_logger(), "Image  : %s", imageTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Debug QoS : %s (publish_debug=%d)", debugQos.c_str(),
                    publishDebug_);

        model_ = std::make_unique<YoloOV>(modelPath_, device, imgsz, static_cast<float>(conf),
                                          static_cast<float>(iou), names_);

        detectPub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/vision/detections", 10);

        rclcpp::QoS debugQosProfile(rclcpp::KeepLast(10));
        if (debugQos == "best_effort")
        {
            debugQosProfile.best_effort();
        }
        debugPub_ =
            this->create_publisher<sensor_msgs::msg::Image>("/vision/debug_image", debugQosProfile);

        imageSub_ = this->create_subscription<sensor_msgs::msg::Image>(
            imageTopic, rclcpp::SensorDataQoS(),
            std::bind(&VisionNode::imageCallback, this, std::placeholders::_1));

        fpsStart_ = std::chrono::steady_clock::now();
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImageConstPtr cvPtr;
        try
        {
            cvPtr = cv_bridge::toCvShare(msg, "bgr8");
        }
        catch (const cv_bridge::Exception& e)
        {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000, "cv_bridge error: %s", e.what());
            return;
        }

        const cv::Mat& frame = cvPtr->image;
        const int W = frame.cols;
        const int H = frame.rows;
        const std::vector<cv::Point> active = toPixels(roiActive_, W, H);
        const std::vector<cv::Point> guard = toPixels(roiGuard_, W, H);

        const std::vector<Detection> detections = model_->detect(frame);

        std_msgs::msg::Float32MultiArray out;
        out.data.reserve(detections.size() * 8);

        cv::Mat dbg;
        if (publishDebug_)
        {
            dbg = frame.clone();
        }

        for (const auto& d : detections)
        {
            const int cx = d.box.x + d.box.width / 2;
            const int cy = d.box.y + d.box.height / 2;

            const bool inActive = pointInPoly(cv::Point(cx, cy), active);
            const bool inGuard = pointInPoly(cv::Point(cx, cy), guard);
            if (filterByRoi_ && !inActive && !inGuard)
            {
                continue;
            }

            out.data.push_back(static_cast<float>(d.classId));
            out.data.push_back(d.conf);
            out.data.push_back(static_cast<float>(cx));
            out.data.push_back(static_cast<float>(cy));
            out.data.push_back(static_cast<float>(d.box.width));
            out.data.push_back(static_cast<float>(d.box.height));
            out.data.push_back(inActive ? 1.0f : 0.0f);
            out.data.push_back(inGuard ? 1.0f : 0.0f);

            if (publishDebug_)
            {
                const cv::Scalar color = inActive ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
                cv::rectangle(dbg, d.box, color, 2);
                cv::circle(dbg, cv::Point(cx, cy), 4, color, -1);

                const std::string label =
                    (d.classId >= 0 && d.classId < static_cast<int>(names_.size()))
                        ? names_[d.classId]
                        : std::to_string(d.classId);
                char buf[128];
                std::snprintf(buf, sizeof(buf), "%s %.2f %s", label.c_str(), d.conf,
                              inActive ? "ACTIVE" : "GUARD");
                cv::putText(dbg, buf, cv::Point(d.box.x, d.box.y - 8), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2);
            }
        }

        detectPub_->publish(out);

        if (publishDebug_)
        {
            if (guard.size() >= 3)
            {
                cv::polylines(dbg, std::vector<std::vector<cv::Point>>{guard}, true,
                              cv::Scalar(0, 255, 255), 2);
            }
            if (active.size() >= 3)
            {
                cv::polylines(dbg, std::vector<std::vector<cv::Point>>{active}, true,
                              cv::Scalar(255, 0, 0), 2);
            }
            cv_bridge::CvImage dbgMsg(msg->header, "bgr8", dbg);
            debugPub_->publish(*dbgMsg.toImageMsg());
        }

        ++frameCount_;
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - fpsStart_).count() >= 1)
        {
            RCLCPP_INFO(this->get_logger(), "FPS: %d, deteksi: %zu", frameCount_, out.data.size() / 8);
            frameCount_ = 0;
            fpsStart_ = now;
        }
    }

    std::string configFile_;
    std::string modelPath_;
    std::vector<std::string> names_;
    std::vector<cv::Point2f> roiActive_;
    std::vector<cv::Point2f> roiGuard_;
    bool filterByRoi_{true};
    bool publishDebug_{true};

    std::unique_ptr<YoloOV> model_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr detectPub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debugPub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub_;

    int frameCount_{0};
    std::chrono::steady_clock::time_point fpsStart_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisionNode>());
    rclcpp::shutdown();
    return 0;
}
