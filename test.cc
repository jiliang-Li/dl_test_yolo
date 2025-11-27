#include <cassert>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "dlutil/dl_log.h"
#include "dlcuda/dl_device.h"
#include "dlcuda/dl_memory_copy.h"
#include "dlvid/dl_vpu_device.h"
#include "dlnne/dlnne_device.h"
#include "dlnne_algo_unit_yolov5.h"

using namespace dl;

// 简单的工具函数：列出 dir 下所有 .jpg/.jpeg 文件（不递归）
static std::vector<std::string> list_jpg_files(const std::string &dir)
{
    std::vector<std::string> files;

    DIR *dp = opendir(dir.c_str());
    if (!dp)
    {
        std::cerr << "[ERROR] cannot open dir: " << dir << std::endl;
        return files;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)) != nullptr)
    {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        if (name.size() >= 4)
        {
            std::string lower = name;
            for (auto &c : lower) c = std::tolower(static_cast<unsigned char>(c));
            if (lower.size() >= 4 &&
                (lower.compare(lower.size() - 4, 4, ".jpg") == 0 ||
                 lower.compare(lower.size() - 5, 5, ".jpeg") == 0))
            {
                files.push_back(dir + "/" + name);
            }
        }
    }
    closedir(dp);
    return files;
}

static void showUsage()
{
    DlLogI << "Usage: ./test_yolov5 [options]";
    DlLogI << "  -n <name>   : 算法单元名称，默认 DlAlgorithmUnit_Yolov5";
    DlLogI << "  -d <dir>    : 图片目录，默认 ../../../resource/images";
    DlLogI << "  -l <level>  : 日志级别 0-4 (debug/info/warn/error/fatal)，默认 1";
    DlLogI << "  -h          : 显示帮助";
    exit(0);
}

static void getCustomOpt(int argc, char *argv[],
                         std::string &network_unit_name,
                         std::string &image_dir,
                         DlLoggerSeverity &log_level)
{
    int opt = 0;
    const char *opt_string = "n:d:l:h";
    while (-1 != (opt = getopt(argc, argv, opt_string)))
    {
        switch (opt)
        {
        case 'n':
            network_unit_name = optarg;
            break;
        case 'd':
            image_dir = optarg;
            break;
        case 'l':
            log_level = static_cast<DlLoggerSeverity>(atoi(optarg));
            break;
        default:
            showUsage();
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    DlLogI << "Test yolov5 start!";

    // 参数默认值
    std::string network_unit_name = "DlAlgorithmUnit_Yolov5";   // DlAlgorithmUnitYolov5
    std::string image_dir = "../../../resource/images";
    DlLoggerSeverity log_level = DlLoggerSeverity_INFO;

    getCustomOpt(argc, argv, network_unit_name, image_dir, log_level);
    setDlLoggerSeverity(log_level);

    // 枚举图片
    auto image_files = list_jpg_files(image_dir);
    if (image_files.empty())
    {
        std::cerr << "[ERROR] no jpg/jpeg found in " << image_dir << std::endl;
        return -1;
    }
    std::cout << "[INFO] found " << image_files.size()
              << " images in " << image_dir << std::endl;

    // 获取设备 & 算法单元
    int device_id = 0;
    auto dl_device    = DlcudaDeviceManager::getInstance().getDlcudaDevice(device_id);
    auto dlnne_device = DlnneDeviceManager::getInstance().getDlnneDevice(device_id);
    auto dlnne_network_unit = dlnne_device->requireDlAlgorithmUnit(network_unit_name);
    dlnne_network_unit->waitEngineSetupDone();

    auto h2d_copy_helper =
        dl_device->getMemCopyHelper(cudaMemcpyHostToDevice);

    int frame_index = 0;

    for (const auto &img_path : image_files)
    {
        std::cout << "================================================\n";
        std::cout << "[INFO] load image: " << img_path << std::endl;

        // 1) 读 BGR 图
        cv::Mat bgr = cv::imread(img_path, cv::IMREAD_COLOR);
        if (bgr.empty())
        {
            std::cerr << "[WARN] imread failed: " << img_path << std::endl;
            continue;
        }

        int width  = bgr.cols;
        int height = bgr.rows;

        // 2) 宽高若不是 2 的倍数，裁掉最后一行/列（YUV420 要求偶数）
        if (width % 2 != 0 || height % 2 != 0)
        {
            int even_w = width  & ~1;
            int even_h = height & ~1;
            cv::Rect roi(0, 0, even_w, even_h);
            bgr = bgr(roi).clone();
            width  = even_w;
            height = even_h;
            std::cout << "[INFO] adjust to even size: "
                      << width << "x" << height << std::endl;
        }

        // 3) BGR -> YUV420P (I420 / YU12)
        cv::Mat yuv;
        try
        {
            cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV_I420);
        }
        catch (const cv::Exception &e)
        {
            std::cerr << "[WARN] cvtColor failed for "
                      << img_path << ": " << e.what() << std::endl;
            continue;
        }

        if (yuv.cols != width || yuv.rows != height * 3 / 2)
        {
            std::cerr << "[WARN] yuv shape mismatch for "
                      << img_path << ", got "
                      << yuv.cols << "x" << yuv.rows
                      << ", expect " << width << "x" << height * 3 / 2
                      << std::endl;
            continue;
        }

        // 4) 拷贝到 host DlFrame (YUV420P)
        auto host_frame = std::make_shared<DlFrame>(DlMemoryType_Host);
        host_frame->resize(width, height, DlPixelFormat_YUV420P);
        host_frame->index = frame_index++;
        if (host_frame->size != static_cast<uint64_t>(yuv.total()))
        {
            std::cerr << "[WARN] frame_size(" << host_frame->size
                      << ") != yuv.total(" << yuv.total()
                      << ") for " << img_path << std::endl;
        }
        std::memcpy(host_frame->data, yuv.data,
                    std::min<uint64_t>(host_frame->size, yuv.total()));

        // 5) 拷贝到 device DlFrame
        auto dev_frame = std::make_shared<DlFrame>(DlMemoryType_Device);
        dev_frame->resize(width, height, DlPixelFormat_YUV420P);
        dev_frame->index = host_frame->index;

        h2d_copy_helper
            ->addMemcpyTask(dev_frame->data,
                            host_frame->data,
                            host_frame->size)
            .get();

        // 6) 构造输入 & 推理
        auto dlnne_input  = dlnne_network_unit->createTestInput(dev_frame);
        auto base_output  = dlnne_network_unit
                                ->addDlnneInferTask(dlnne_input)
                                .get();

        auto yolov5_output =
            std::dynamic_pointer_cast<DlnneYolov5Output>(base_output);

        if (!yolov5_output)
        {
            std::cerr << "[ERROR] output cast to DlnneYolov5Output failed."
                      << std::endl;
            continue;
        }

        // 7) 打印结果
        std::cout << "[RESULT] detections for " << img_path << ":\n";
        for (const auto &rect : yolov5_output->rect_vector)
        {
            std::cout << "  cls=" << rect.classification
                      << " score=" << rect.precision
                      << " box=(" << rect.left << "," << rect.top
                      << "," << rect.right << "," << rect.bottom << ")"
                      << std::endl;
        }

        // 可选：调用已有的 checkNetworkOutput，在图片上画框并输出 output.jpg
        dlnne_network_unit->checkNetworkOutput(base_output);
    }

    dlnne_device->releaseDlAlgorithmUnit(network_unit_name);
    DlLogI << "Test yolov5 done!";
    return 0;
}
