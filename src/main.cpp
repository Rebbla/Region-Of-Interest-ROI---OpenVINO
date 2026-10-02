#include "roi.hpp"
#include "yolo_ov.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<cv::Point2f> parsePoints(const YAML::Node& node)
{
    std::vector<cv::Point2f> pts;
    if (!node || !node.IsSequence())
    {
        return pts;
    }
    for (const auto& p : node)
    {
        pts.emplace_back(p[0].as<float>(), p[1].as<float>());
    }
    return pts;
}

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
    const fs::path modelDir = fs::path(modelPath).parent_path();
    const fs::path metadata = modelDir / "metadata.yaml";
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

int main(int argc, char** argv)
{
    std::string configPath = "config.yaml";
    bool headless = false;
    std::string modelOverride;
    std::string deviceOverride;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)
        {
            configPath = argv[++i];
        }
        else if (arg == "--model" && i + 1 < argc)
        {
            modelOverride = argv[++i];
        }
        else if (arg == "--device" && i + 1 < argc)
        {
            deviceOverride = argv[++i];
        }
        else if (arg == "--headless")
        {
            headless = true;
        }
    }

    YAML::Node cfg;
    try
    {
        cfg = YAML::LoadFile(configPath);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Gagal membaca config: " << e.what() << "\n";
        return 1;
    }

    const std::string modelPath = modelOverride.empty() ? cfg["model"].as<std::string>() : modelOverride;
    const std::string device = deviceOverride.empty()
                                   ? (cfg["device"] ? cfg["device"].as<std::string>() : "CPU")
                                   : deviceOverride;
    const int cameraId = cfg["camera"] ? cfg["camera"].as<int>() : 0;
    const float conf = cfg["conf"] ? cfg["conf"].as<float>() : 0.5f;
    const float iou = cfg["iou"] ? cfg["iou"].as<float>() : 0.45f;
    const bool roiShow = cfg["roi_show"] ? cfg["roi_show"].as<bool>() : true;

    std::vector<std::string> names = parseNames(cfg["names"]);
    if (names.empty())
    {
        names = loadNames(modelPath);
    }

    const std::vector<cv::Point2f> roiActiveNorm = parsePoints(cfg["roi_active"]);
    const std::vector<cv::Point2f> roiGuardNorm = parsePoints(cfg["roi_guard"]);

    std::cout << "Model  : " << modelPath << "\n";
    std::cout << "Device : " << device << "\n";

    YoloOV model(modelPath, device, cfg["imgsz"] ? cfg["imgsz"].as<int>() : 640, conf, iou, names);

    cv::VideoCapture cap(cameraId);
    if (!cap.isOpened())
    {
        std::cerr << "Tidak dapat menemukan Webcam\n";
        return 1;
    }

    int frameCount = 0;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();

    while (true)
    {
        cv::Mat frame;
        if (!cap.read(frame))
        {
            std::cerr << "Tidak dapat menemukan Webcam\n";
            break;
        }

        const int W = frame.cols;
        const int H = frame.rows;
        const std::vector<cv::Point> active = toPixels(roiActiveNorm, W, H);
        const std::vector<cv::Point> guard = toPixels(roiGuardNorm, W, H);

        if (roiShow)
        {
            if (guard.size() >= 3)
            {
                cv::polylines(frame, std::vector<std::vector<cv::Point>>{guard}, true, cv::Scalar(0, 255, 255), 2);
            }
            if (active.size() >= 3)
            {
                cv::polylines(frame, std::vector<std::vector<cv::Point>>{active}, true, cv::Scalar(255, 0, 0), 2);
            }
            if (!guard.empty())
            {
                cv::putText(frame, "GUARD ZONE :tidak bisa", cv::Point(guard[0].x, guard[0].y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
            if (!active.empty())
            {
                cv::putText(frame, "ACTIVE ZONE : Berhasil", cv::Point(active[0].x, active[0].y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
        }

        const std::vector<Detection> detections = model.detect(frame);

        for (const auto& d : detections)
        {
            const int cx = (d.box.x + d.box.x + d.box.width) / 2;
            const int cy = (d.box.y + d.box.y + d.box.height) / 2;

            const bool inActive = pointInPoly(cv::Point(cx, cy), active);
            const bool inGuard = pointInPoly(cv::Point(cx, cy), guard);

            if (!inActive && !inGuard)
            {
                continue;
            }

            cv::rectangle(frame, d.box, cv::Scalar(0, 255, 255), 2);

            if (inActive)
            {
                cv::circle(frame, cv::Point(cx, cy), 4, cv::Scalar(0, 255, 0), -1);
                const std::string label = (d.classId >= 0 && d.classId < static_cast<int>(names.size()))
                                              ? names[d.classId]
                                              : std::to_string(d.classId);
                char buf[128];
                std::snprintf(buf, sizeof(buf), "%s %.2f", label.c_str(), d.conf);
                cv::putText(frame, buf, cv::Point(d.box.x, d.box.y - 8), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 255, 0), 2);
            }
            else
            {
                cv::putText(frame, "Geser Ke tengah", cv::Point(d.box.x, d.box.y - 8), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 255, 0), 2);
            }
        }

        ++frameCount;
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last).count() >= 1)
        {
            std::cout << "FPS: " << frameCount << "\n";
            frameCount = 0;
            last = now;
        }

        if (headless)
        {
            continue;
        }

        cv::imshow("demoroi- q: quit s:save", frame);
        const int k = cv::waitKey(1) & 0xFF;
        if (k == 'q')
        {
            break;
        }
        if (k == 's')
        {
            const std::string out = "snapshot_" + std::to_string(std::time(nullptr)) + ".jpg";
            cv::imwrite(out, frame);
            std::cout << "Saved: " << out << "\n";
        }
    }

    cap.release();
    if (!headless)
    {
        cv::destroyAllWindows();
    }
    return 0;
}
