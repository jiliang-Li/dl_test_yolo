#include <cassert>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <cstdlib>
#include <atomic>   // 新增

#include <httplib.h>

#include "dlutil/dl_log.h"
#include "dlcuda/dl_device.h"
#include "dlcuda/dl_memory_copy.h"
#include "dlnne/dlnne_device.h"
#include "helper/dlnne_impl/dlnne_algo_unit_yolov5.h"
#include "dlvid/dl_vpu_device.h"
#include "dlvid/dl_codec_base.h"

#include <getopt.h>

using namespace dl;

// 简单的 ping handler
void register_ping(httplib::Server &svr) {
    svr.Get("/ping", [](const httplib::Request &, httplib::Response &res) {
        DlLogI << "收到 GET /ping";
        res.set_content("pong\n", "text/plain");
    });
}

int main(int argc, char *argv[])
{
    DlLogI << "HTTP Yolov5 server (decodeJpeg + multi-units) start!";

    // ----------------- 解析命令行参数：端口 -----------------
    int port = 6650;  // 默认端口

    int opt = 0;
    const char *opt_string = "p:h";
    while ((opt = getopt(argc, argv, opt_string)) != -1) {
        switch (opt) {
        case 'p': {
            int p = std::atoi(optarg);
            if (p > 0 && p < 65536) {
                port = p;
            } else {
                std::cerr << "[ERROR] invalid port: " << optarg << std::endl;
                return -1;
            }
            break;
        }
        case 'h':
        default:
            std::cout << "Usage: " << argv[0] << " [-p port]" << std::endl;
            return 0;
        }
    }

    // ----------------- 初始化登临设备 & Yolov5 算法单元 -----------------
    int device_id = 1;
    std::string network_unit_name = "DlAlgorithmUnit_Yolov5";

    // auto dl_device    = DlcudaDeviceManager::getInstance().getDlcudaDevice(device_id);
    auto dlnne_device = DlnneDeviceManager::getInstance().getDlnneDevice(device_id);

    // VPU 设备，用于 JPEG 硬解码
    auto dlvid_device = DlvidDeviceManager::getInstance().getVpuDevice(device_id);

    // --- 多套 Yolov5 算法单元：每个有自己的 executor 线程 ---
    const int NUM_UNITS = 1;

    auto first_unit = dlnne_device->requireDlAlgorithmUnit(network_unit_name);
    first_unit->waitEngineSetupDone();

    using NetworkUnitPtr = decltype(first_unit);
    std::vector<NetworkUnitPtr> yolo_units;
    yolo_units.reserve(NUM_UNITS);
    yolo_units.push_back(first_unit);

    for (int i = 1; i < NUM_UNITS; ++i) {
        auto u = dlnne_device->requireDlAlgorithmUnit(network_unit_name);
        u->waitEngineSetupDone();
        yolo_units.push_back(u);
    }

    // 轮询索引：用于在多个算法单元之间分发请求
    std::atomic<int> rr_index{0};

    // ----------------- 配置 HTTP 服务器 -----------------
    httplib::Server svr;

    // 自定义任务队列：固定 n 个 worker 线程（按需要启用）
    // svr.new_task_queue = [] { return new httplib::ThreadPool(16); };   // 根据机器 CPU 核心数调

    svr.set_payload_max_length(10 * 1024 * 1024);   // 请求体大小限制<n>MB, <n> * 1024 * 1024

    // 访问日志（如需调试可打开）
    svr.set_logger([](const httplib::Request &req, const httplib::Response &res) {
        // DlLogI << "[HTTP] " << req.method << " " << req.path
        //        << " status=" << res.status
        //        << " body_len=" << req.body.size();
    });

    // 错误日志（包含 413 之类的）
    svr.set_error_handler([](const httplib::Request &req, httplib::Response &res) {
        DlLogE << "[HTTP ERROR] status=" << res.status
               << " method=" << req.method
               << " path=" << req.path
               << " body_len=" << req.body.size();
    });

    // 健康检查
    register_ping(svr);

    // ----------------- 推理接口：POST /predictions/yolo5 -----------------
    svr.Post(
        "/predictions/yolo5",
        [dlvid_device, &yolo_units, &rr_index, device_id](const httplib::Request &req,
                                                          httplib::Response &res) {
        // DlLogI << "收到 POST /predictions/yolo5";

        try {
            // 1) URL 参数（目前不做裁图，只是解析出来，保持原有逻辑）
            std::string size_param;
            if (req.has_param("size")) {
                size_param = req.get_param_value("size");
            }
            std::string codec = "cv2bytes";
            if (req.has_param("codec")) {
                codec = req.get_param_value("codec");
            }

            // 2) 请求体即为 JPEG bytes
            const std::string &body = req.body;
            if (body.empty()) {
                res.status = 400;
                res.set_content(R"({"code":1,"msg":"empty body"})", "application/json");
                return;
            }

            std::vector<unsigned char> jpeg_data(body.begin(), body.end());

            // 每个线程只初始化一次 CUDA device
            thread_local bool s_cuda_inited = false;
            if (!s_cuda_inited) {
                DlLogI << "cuda_init";
                assert(cudaSetDevice(device_id) == 0);
                s_cuda_inited = true;
            }

            // ---- 2.5) 轮询选择一个 Yolov5 算法单元 ----
            int idx = rr_index.fetch_add(1, std::memory_order_relaxed);
            idx = idx % static_cast<int>(yolo_units.size());
            auto dlnne_network = yolo_units[idx];
            // 如需调试可打开：
            // DlLogI << "[HTTP] dispatch to Yolov5 unit #" << idx;

            // 3) 构造 DlBuffer（Host），调用框架内 JPEG 硬解码
            DlBuffer jpeg_buffer(DlMemoryType_Host);
            jpeg_buffer.resize(jpeg_data.size());
            std::memcpy(jpeg_buffer.data, jpeg_data.data(), jpeg_data.size());

            // 每次请求各自申请 / 释放一个 jpeg_decoder，避免多线程抢占
            auto jpeg_decoder = dlvid_device->requireDlJpegDecoder();
            auto frame = jpeg_decoder->decodeJpeg(jpeg_buffer);
            dlvid_device->releaseDlJpegDecoder(jpeg_decoder);

            if (!frame || frame->width <= 0 || frame->height <= 0) {
                res.status = 500;
                res.set_content(R"({"code":3,"msg":"decodeJpeg failed"})", "application/json");
                return;
            }

            // DlLogI << "[HTTP] decodeJpeg OK, frame size "
            //        << frame->width << "x" << frame->height;

            // 4) 送入 Yolov5 算法单元做推理
            auto input  = dlnne_network->createTestInput(frame);
            auto future = dlnne_network->addDlnneInferTask(input);
            auto base_output = future.get();

            auto yolo_out = std::dynamic_pointer_cast<DlnneYolov5Output>(base_output);
            if (!yolo_out) {
                res.status = 500;
                res.set_content(R"({"code":4,"msg":"bad output type"})", "application/json");
                return;
            }

            // 5) 汇总检测结果，返回 list: [[x1,y1,x2,y2,score,cls], ...]
            std::ostringstream oss;
            oss << "[";

            bool first = true;
            for (const auto &rect : yolo_out->rect_vector) {
                int x1 = rect.left;
                int y1 = rect.top;
                int x2 = rect.right;
                int y2 = rect.bottom;
                float score = rect.precision;
                int cls = rect.classification;
                if (!first) {
                    oss << ",";
                }
                first = false;
                oss << "[" << x1 << "," << y1 << "," << x2 << "," << y2 << "," << score << "," << cls << "]";
            }
            oss << "]";
            res.status = 200;
            res.set_content(oss.str(), "application/json");
        }
        catch (const std::exception &e) {
            DlLogE << "[HTTP] exception: " << e.what();
            res.status = 500;
            std::string msg = std::string(R"({"code":500,"msg":")") + e.what() + "\"}";
            res.set_content(msg, "application/json");
        }
        catch (...) {
            DlLogE << "[HTTP] unknown exception";
            res.status = 500;
            res.set_content(R"({"code":500,"msg":"unknown error"})", "application/json");
        }
    });
    // ----------------- 解码测试接口：POST /predictions/decode -----------------
    svr.Post(
        "/predictions/decode",
        [dlvid_device, &yolo_units, &rr_index, device_id](const httplib::Request &req,
                                                          httplib::Response &res) {
        // DlLogI << "收到 POST /predictions/yolo5";

        try {
            // 1) URL 参数（目前不做裁图，只是解析出来，保持原有逻辑）
            std::string size_param;
            if (req.has_param("size")) {
                size_param = req.get_param_value("size");
            }
            std::string codec = "cv2bytes";
            if (req.has_param("codec")) {
                codec = req.get_param_value("codec");
            }

            // 2) 请求体即为 JPEG bytes
            const std::string &body = req.body;
            if (body.empty()) {
                res.status = 400;
                res.set_content(R"({"code":1,"msg":"empty body"})", "application/json");
                return;
            }

            std::vector<unsigned char> jpeg_data(body.begin(), body.end());

            // 每个线程只初始化一次 CUDA device
            thread_local bool s_cuda_inited = false;
            if (!s_cuda_inited) {
                DlLogI << "cuda_init";
                assert(cudaSetDevice(device_id) == 0);
                s_cuda_inited = true;
            }

            // ---- 2.5) 轮询选择一个 Yolov5 算法单元 ----
            int idx = rr_index.fetch_add(1, std::memory_order_relaxed);
            idx = idx % static_cast<int>(yolo_units.size());
            auto dlnne_network = yolo_units[idx];
            // 如需调试可打开：
            // DlLogI << "[HTTP] dispatch to Yolov5 unit #" << idx;

            // 3) 构造 DlBuffer（Host），调用框架内 JPEG 硬解码
            DlBuffer jpeg_buffer(DlMemoryType_Host);
            jpeg_buffer.resize(jpeg_data.size());
            std::memcpy(jpeg_buffer.data, jpeg_data.data(), jpeg_data.size());

            // 每次请求各自申请 / 释放一个 jpeg_decoder，避免多线程抢占
            auto jpeg_decoder = dlvid_device->requireDlJpegDecoder();
            auto frame = jpeg_decoder->decodeJpeg(jpeg_buffer);
            dlvid_device->releaseDlJpegDecoder(jpeg_decoder);

            if (!frame || frame->width <= 0 || frame->height <= 0) {
                res.status = 500;
                res.set_content(R"({"code":3,"msg":"decodeJpeg failed"})", "application/json");
                return;
            }

            // DlLogI << "[HTTP] decodeJpeg OK, frame size "
            //        << frame->width << "x" << frame->height;


            // 5) 汇总检测结果，返回 list: [[x1,y1,x2,y2,score,cls], ...]
            std::ostringstream oss;
            oss << "[";
            oss << "[50, 50, 100, 100, 0.5, 0]";
            oss << "]";
            res.status = 200;
            res.set_content(oss.str(), "application/json");
        }
        catch (const std::exception &e) {
            DlLogE << "[HTTP] exception: " << e.what();
            res.status = 500;
            std::string msg = std::string(R"({"code":500,"msg":")") + e.what() + "\"}";
            res.set_content(msg, "application/json");
        }
        catch (...) {
            DlLogE << "[HTTP] unknown exception";
            res.status = 500;
            res.set_content(R"({"code":500,"msg":"unknown error"})", "application/json");
        }
    });
    // ----------------- 推理测试接口：POST /predictions/infer -----------------
    // ----------------- 纯推理性能测试：POST /predictions/infer -----------------
    svr.Post(
        "/predictions/infer",
        [dlvid_device, &yolo_units, &rr_index, device_id](const httplib::Request &req,
                                                          httplib::Response &res) {
        try {
            // 1) URL 参数（保持和 /predictions/yolo5 一致，方便复用压测脚本）
            std::string size_param;
            if (req.has_param("size")) {
                size_param = req.get_param_value("size");
            }
            std::string codec = "cv2bytes";
            if (req.has_param("codec")) {
                codec = req.get_param_value("codec");
            }

            // 2) 请求体为 JPEG bytes
            //    这里为了简单，依然要求每次请求都带一张图，
            //    但只会在第一次调用时真正解码，后续都会复用同一个 frame。
            const std::string &body = req.body;
            if (body.empty()) {
                res.status = 400;
                res.set_content(R"({"code":1,"msg":"empty body"})", "application/json");
                return;
            }

            std::vector<unsigned char> jpeg_data(body.begin(), body.end());

            // 每个线程只初始化一次 CUDA device
            thread_local bool s_cuda_inited = false;
            if (!s_cuda_inited) {
                DlLogI << "cuda_init";
                assert(cudaSetDevice(device_id) == 0);
                s_cuda_inited = true;
            }

            // 3) 轮询选择一个 Yolov5 算法单元
            int idx = rr_index.fetch_add(1, std::memory_order_relaxed);
            idx = idx % static_cast<int>(yolo_units.size());
            auto dlnne_network = yolo_units[idx];

            // 4) 构造 DlBuffer（Host），仅在第一次调用 infer 时做 JPEG 硬解码并缓存 frame
            DlBuffer jpeg_buffer(DlMemoryType_Host);
            jpeg_buffer.resize(jpeg_data.size());
            std::memcpy(jpeg_buffer.data, jpeg_data.data(), jpeg_data.size());

            // static 缓存解码后的 frame：
            //  - 第一次调用 /predictions/infer 时执行解码
            //  - 之后所有请求都直接复用 cached_frame，不再解码
            static auto cached_frame = [&]() {
                auto jpeg_decoder = dlvid_device->requireDlJpegDecoder();
                auto frame        = jpeg_decoder->decodeJpeg(jpeg_buffer);
                dlvid_device->releaseDlJpegDecoder(jpeg_decoder);
                DlLogI << "decode frame";

                if (!frame || frame->width <= 0 || frame->height <= 0) {
                    DlLogE << "decodeJpeg failed in /predictions/infer";
                    throw "decodeJpeg failed in /predictions/infer";
                }
                // DlLogI << "[HTTP] infer: cached frame " << frame->width << "x" << frame->height;
                return frame;
            }();

            auto frame = cached_frame;

            // 5) 送入 Yolov5 算法单元做推理（只做推理，不再解码）
            auto input      = dlnne_network->createTestInput(frame);
            auto future     = dlnne_network->addDlnneInferTask(input);
            auto base_output = future.get();

            auto yolo_out = std::dynamic_pointer_cast<DlnneYolov5Output>(base_output);
            if (!yolo_out) {
                res.status = 500;
                res.set_content(R"({"code":4,"msg":"bad output type"})", "application/json");
                return;
            }

            // 6) 和 /predictions/yolo5 一样的输出格式
            std::ostringstream oss;
            oss << "[";

            bool first = true;
            for (const auto &rect : yolo_out->rect_vector) {
                int   x1    = rect.left;
                int   y1    = rect.top;
                int   x2    = rect.right;
                int   y2    = rect.bottom;
                float score = rect.precision;
                int   cls   = rect.classification;
                if (!first) {
                    oss << ",";
                }
                first = false;
                oss << "[" << x1 << "," << y1 << "," << x2 << "," << y2
                    << "," << score << "," << cls << "]";
            }

            oss << "]";
            res.status = 200;
            res.set_content(oss.str(), "application/json");
        }
        catch (const std::exception &e) {
            DlLogE << "[HTTP] exception in /predictions/infer: " << e.what();
            res.status = 500;
            std::string msg = std::string(R"({"code":500,"msg":")") + e.what() + "\"}";
            res.set_content(msg, "application/json");
        }
        catch (...) {
            DlLogE << "[HTTP] unknown exception in /predictions/infer";
            res.status = 500;
            res.set_content(R"({"code":500,"msg":"unknown error"})", "application/json");
        }
    });

    svr.Post(
        "/predictions/post",
        [dlvid_device, &yolo_units, &rr_index, device_id](const httplib::Request &req,
                                                          httplib::Response &res) {
        try {
            // 1) URL 参数（保持和 /predictions/yolo5 一致，方便复用压测脚本）
            std::string size_param;
            if (req.has_param("size")) {
                size_param = req.get_param_value("size");
            }
            std::string codec = "cv2bytes";
            if (req.has_param("codec")) {
                codec = req.get_param_value("codec");
            }

            // 2) 请求体为 JPEG bytes
            //    这里为了简单，依然要求每次请求都带一张图，
            //    但只会在第一次调用时真正解码，后续都会复用同一个 frame。
            const std::string &body = req.body;
            if (body.empty()) {
                res.status = 400;
                res.set_content(R"({"code":1,"msg":"empty body"})", "application/json");
                return;
            }

            std::vector<unsigned char> jpeg_data(body.begin(), body.end());

            // 每个线程只初始化一次 CUDA device
            thread_local bool s_cuda_inited = false;
            if (!s_cuda_inited) {
                DlLogI << "cuda_init";
                assert(cudaSetDevice(device_id) == 0);
                s_cuda_inited = true;
            }

            // 3) 轮询选择一个 Yolov5 算法单元
            int idx = rr_index.fetch_add(1, std::memory_order_relaxed);
            idx = idx % static_cast<int>(yolo_units.size());
            auto dlnne_network = yolo_units[idx];

            // 4) 构造 DlBuffer（Host），仅在第一次调用 infer 时做 JPEG 硬解码并缓存 frame
            DlBuffer jpeg_buffer(DlMemoryType_Host);
            jpeg_buffer.resize(jpeg_data.size());
            std::memcpy(jpeg_buffer.data, jpeg_data.data(), jpeg_data.size());

            // static 缓存解码后的 frame：
            //  - 第一次调用 /predictions/infer 时执行解码
            //  - 之后所有请求都直接复用 cached_frame，不再解码
            static auto cached_frame = [&]() {
                auto jpeg_decoder = dlvid_device->requireDlJpegDecoder();
                auto frame        = jpeg_decoder->decodeJpeg(jpeg_buffer);
                dlvid_device->releaseDlJpegDecoder(jpeg_decoder);
                DlLogI << "decode frame";

                if (!frame || frame->width <= 0 || frame->height <= 0) {
                    DlLogE << "decodeJpeg failed in /predictions/infer";
                    throw "decodeJpeg failed in /predictions/infer";
                }
                // DlLogI << "[HTTP] infer: cached frame " << frame->width << "x" << frame->height;
                return frame;
            }();

            auto frame = cached_frame;

            // 5) 送入 Yolov5 算法单元做推理（只做推理，不再解码）
            std::ostringstream oss;
            oss << "[[50, 50, 100, 100, 0.5, 0]]";
            res.status = 200;
            res.set_content(oss.str(), "application/json");
        }
        catch (const std::exception &e) {
            DlLogE << "[HTTP] exception in /predictions/infer: " << e.what();
            res.status = 500;
            std::string msg = std::string(R"({"code":500,"msg":")") + e.what() + "\"}";
            res.set_content(msg, "application/json");
        }
        catch (...) {
            DlLogE << "[HTTP] unknown exception in /predictions/infer";
            res.status = 500;
            res.set_content(R"({"code":500,"msg":"unknown error"})", "application/json");
        }
    });


    // ----------------- 启动监听 -----------------
    DlLogI << "Listening on 0.0.0.0:" << port;
    svr.listen("0.0.0.0", port);

    // ----------------- 退出清理 -----------------
    // 理论上每 require 一次就应该 release 一次，这里对应 NUM_UNITS 次
    for (int i = 0; i < NUM_UNITS; ++i) {
        dlnne_device->releaseDlAlgorithmUnit(network_unit_name);
    }

    DlLogI << "HTTP Yolov5 server stop.";
    return 0;
}
