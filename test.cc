#include <cassert>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <thread>

#include <opencv2/opencv.hpp>

#include "dlutil/dl_log.h"
#include "dlcuda/dl_device.h"
#include "dlcuda/dl_memory_copy.h"
#include "dlvid/dl_vpu_device.h"
#include "dlnne/dlnne_device.h"
#include "dlnne_algo_unit_yolov5.h"

using namespace dl;

// 列出 dir 下所有 .jpg/.jpeg 文件（不递归）
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
            for (auto &c : lower)
                c = std::tolower(static_cast<unsigned char>(c));

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
    DlLogI << "Usage: ./test_yolov5_mt [options]";
    DlLogI << "  -n <name>   : 算法单元名称，默认 DlAlgorithmUnit_Yolov5";
    DlLogI << "  -d <dir>    : 图片目录，默认 ../../../resource/images";
    DlLogI << "  -l <level>  : 日志级别 0-4 (debug/info/warn/error/fatal)，默认 1";
    DlLogI << "  -t <num>    : 线程数，默认 4";
    DlLogI << "  -h          : 显示帮助";
    exit(0);
}

static void getCustomOpt(int argc, char *argv[],
                         std::string &network_unit_name,
                         std::string &image_dir,
                         DlLoggerSeverity &log_level,
                         int &num_threads)
{
    int opt = 0;
    const char *opt_string = "n:d:l:t:h";
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
        case 't':
            num_threads = std::max(1, atoi(optarg));
            break;
        default:
            showUsage();
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    DlLogI << "Test yolov5 multi-thread start!";

    // 默认参数
    std::string network_unit_name = "DlAlgorithmUnit_Yolov5";
    std::string image_dir = "../../../resource/images";
    DlLoggerSeverity log_level = DlLoggerSeverity_INFO;
    int num_threads = 16;

    getCustomOpt(argc, argv, network_unit_name, image_dir, log_level, num_threads);
    setDlLoggerSeverity(log_level);

    // 枚举所有图片
    auto image_files = list_jpg_files(image_dir);
    if (image_files.empty())
    {
        std::cerr << "[ERROR] no jpg/jpeg found in " << image_dir << std::endl;
        return -1;
    }
    std::cout << "[INFO] found " << image_files.size()
              << " images in " << image_dir << std::endl;

    // 获取设备 & 算法单元（所有线程共享一个）
    int device_id = 0;
    auto dl_device    = DlcudaDeviceManager::getInstance().getDlcudaDevice(device_id);
    auto dlnne_device = DlnneDeviceManager::getInstance().getDlnneDevice(device_id);
    auto dlnne_network_unit =
        dlnne_device->requireDlAlgorithmUnit(network_unit_name);
    dlnne_network_unit->waitEngineSetupDone();

    auto start_all = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([t,
                              &image_files,
                              dl_device,
                              dlnne_network_unit]()
        {
            DlLogI << "[Thread " << t << "] start.";

            // 每个线程自己的 memcpy helper
            auto h2d_copy_helper =
                dl_device->getMemCopyHelper(cudaMemcpyHostToDevice);

            std::vector<std::future<std::shared_ptr<DlnneUserOutputBase>>> futures;
            futures.reserve(image_files.size());

            // ☆☆☆ 关键修改：每个线程完整遍历 image_files ☆☆☆
            for (size_t idx = 0; idx < image_files.size(); ++idx)
            {
                const std::string &img_path = image_files[idx];
                // std::cout << "[Thread " << t << "] load image: "                          << img_path << std::endl;

                // 1) 读 BGR 图
                cv::Mat bgr = cv::imread(img_path, cv::IMREAD_COLOR);
                if (bgr.empty())
                {
                    std::cerr << "[Thread " << t
                              << "] WARN imread failed: " << img_path << std::endl;
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
                    std::cout << "[Thread " << t << "] adjust to even size: "
                              << width << "x" << height << std::endl;
                }

                // 3) BGR -> YUV420P (I420)
                cv::Mat yuv;
                try
                {
                    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV_I420);
                }
                catch (const cv::Exception &e)
                {
                    std::cerr << "[Thread " << t
                              << "] cv::cvtColor failed for " << img_path
                              << ", err = " << e.what() << std::endl;
                    continue;   // 这一张跳过
                }

                if (yuv.cols != width || yuv.rows != height * 3 / 2)
                {
                    std::cerr << "[Thread " << t
                              << "] WARN yuv shape mismatch for " << img_path
                              << ", got " << yuv.cols << "x" << yuv.rows
                              << ", expect " << width << "x" << height * 3 / 2
                              << std::endl;
                    continue;
                }

                // 4) 拷贝到 host DlFrame (YUV420P)
                auto host_frame = std::make_shared<DlFrame>(DlMemoryType_Host);
                host_frame->resize(width, height, DlPixelFormat_YUV420P);
                // 帧 index 可以带线程信息，方便调试
                host_frame->index = static_cast<int>(t * 1000000 + idx);

                if (host_frame->size != static_cast<uint64_t>(yuv.total()))
                {
                    std::cerr << "[Thread " << t
                              << "] WARN frame_size(" << host_frame->size
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

                // 6) 构造输入 & 入队推理（不立即 get，让引擎自己组 batch）
                auto dlnne_input = dlnne_network_unit->createTestInput(dev_frame);
                futures.emplace_back(
                    dlnne_network_unit->addDlnneInferTask(dlnne_input)
                );
            }

            // 7) 线程收尾：等待所有 future 完成
            for (auto &f : futures)
            {
                try
                {
                    auto base_output = f.get();
                    (void)base_output;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "[Thread " << t
                              << "] ERROR in future.get(): " << e.what()
                              << std::endl;
                }
            }

            DlLogI << "[Thread " << t << "] done.";
        });
    }

    for (auto &th : workers)
    {
        if (th.joinable()) th.join();
    }

    auto end_all = std::chrono::steady_clock::now();
    double total_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_all - start_all).count() / 1000.0;

    DlLogI << "All tasks done. total cost = " << total_ms << " ms";

    dlnne_device->releaseDlAlgorithmUnit(network_unit_name);
    DlLogI << "Test yolov5 multi-thread done!";
    return 0;
}
