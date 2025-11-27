#include <cassert>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH (50 * 1024 * 1024)
#include <httplib.h>
#include <opencv2/opencv.hpp>

#include "dlutil/dl_log.h"
#include "dlcuda/dl_device.h"
#include "dlcuda/dl_memory_copy.h"
#include "dlnne/dlnne_device.h"
#include "helper/dlnne_impl/dlnne_algo_unit_yolov5.h"


using namespace dl;

// 解析 size 字符串，支持 "[640, 640]"、"640,640"、"640x640"、"640 640"
static bool parse_size(const std::string& s, int& w, int& h)
{
    std::string tmp;
    tmp.reserve(s.size());
    for (char c : s)
    {
        if (c == ' ' || c == '\t') continue;
        tmp.push_back(c);
    }

    // [w,h]
    if (sscanf(tmp.c_str(), "[%d,%d]", &w, &h) == 2) return true;
    // w,h
    if (sscanf(tmp.c_str(), "%d,%d", &w, &h) == 2) return true;
    // w x h
    if (sscanf(tmp.c_str(), "%dx%d", &w, &h) == 2) return true;
    // w h
    if (sscanf(tmp.c_str(), "%d%d", &w, &h) == 2) return true;
    return false;
}

// 把一张 BGR 图按 patch_w * patch_h 切成网格
struct PatchInfo
{
    cv::Mat bgr;
    int offset_x;
    int offset_y;
};

static std::vector<PatchInfo> split_image_to_patches(const cv::Mat& bgr, int patch_w, int patch_h)
{
    std::vector<PatchInfo> patches;

    if (patch_w <= 0 || patch_h <= 0)
    {
        // 不指定 size，整张图作为一个 patch
        patches.push_back({bgr, 0, 0});
        return patches;
    }

    // 确保 patch 宽高为偶数（YUV420 要求）
    patch_w = (patch_w / 2) * 2;
    patch_h = (patch_h / 2) * 2;

    if (patch_w <= 0 || patch_h <= 0)
    {
        patches.push_back({bgr, 0, 0});
        return patches;
    }

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    if (img_w < patch_w || img_h < patch_h)
    {
        // 太小了，就不切了
        patches.push_back({bgr, 0, 0});
        return patches;
    }

    int num_cols = img_w / patch_w;
    int num_rows = img_h / patch_h;

    for (int r = 0; r < num_rows; ++r)
    {
        for (int c = 0; c < num_cols; ++c)
        {
            int x0 = c * patch_w;
            int y0 = r * patch_h;

            if (x0 + patch_w > img_w || y0 + patch_h > img_h)
            {
                continue;
            }

            cv::Rect roi(x0, y0, patch_w, patch_h);
            PatchInfo p;
            p.bgr = bgr(roi).clone(); // clone 防止后面覆盖
            p.offset_x = x0;
            p.offset_y = y0;
            patches.push_back(std::move(p));
        }
    }

    if (patches.empty())
    {
        patches.push_back({bgr, 0, 0});
    }

    return patches;
}

int main(int argc, char* argv[])
{
    DlLogI << "HTTP Yolov5 server start!";

    int device_id = 0;
    DlPixelFormat pixel_format = DlPixelFormat_YUV420P;
    std::string network_unit_name = "DlAlgorithmUnit_Yolov5";

    // 初始化设备和网络单元（全局共用一份）
    auto dl_device = DlcudaDeviceManager::getInstance().getDlcudaDevice(device_id);
    auto dlnne_device = DlnneDeviceManager::getInstance().getDlnneDevice(device_id);
    auto dlnne_network_unit = dlnne_device->requireDlAlgorithmUnit(network_unit_name);
    dlnne_network_unit->waitEngineSetupDone();

    httplib::Server svr;
    svr.set_payload_max_length(10 * 1024 * 1024);   // 允许最多 <n>MB 的 body: n * 1024 * 1024

    // 简单 access log
    svr.set_logger([](const httplib::Request &req, const httplib::Response &res) {
        DlLogI << "[HTTP] " << req.method << " " << req.path << " status=" << res.status << " body_len=" << req.body.size();
    });

    // 错误处理（4xx/5xx 时会走这里）
    svr.set_error_handler([](const httplib::Request &req, httplib::Response &res) {
        DlLogE << "[HTTP ERROR] " << res.status << " on " << req.method << " " << req.path << ", body_len=" << req.body.size();
    });

    // 路径兼容 TorchServe 风格: /predictions/<model_name>
    // svr.Post(R"(/predictions/(.+))",
    svr.Post("/predictions/yolo5", [dl_device, dlnne_network_unit, pixel_format]
        (const httplib::Request& req, httplib::Response& res)
             {
                DlLogI << "收到Post请求";
                 try
                 {
                    // 1) URL 上的参数：?size=[640,640]&codec=cv2bytes
                    std::string size_param;
                    if (req.has_param("size")) {
                        size_param = req.get_param_value("size");
                    }
                    std::string codec = "cv2bytes";
                    if (req.has_param("codec")) {
                        codec = req.get_param_value("codec");
                    }


                    // 2) body 就是 JPEG bytes
                    const std::string &body = req.body;
                    if (body.empty()) {
                        res.status = 400;
                        res.set_content(R"({"code":1,"msg":"empty body"})", "application/json");
                        return;
                    }

                    std::vector<unsigned char> jpeg_data(body.begin(), body.end());
                    cv::Mat bgr = cv::imdecode(jpeg_data, cv::IMREAD_COLOR);
                    if (bgr.empty()) {
                        res.status = 400;
                        res.set_content(R"({"code":2,"msg":"imdecode failed"})", "application/json");
                        return;
                    }

                     int img_w = bgr.cols;
                     int img_h = bgr.rows;
                     DlLogI << "[HTTP] recv image, codec=" << codec << ", size=" << img_w << "x" << img_h;

                     // 3) 解析 size，决定裁剪 patch 大小
                     int patch_w = 0, patch_h = 0;
                     if (!size_param.empty())
                     {
                         if (!parse_size(size_param, patch_w, patch_h))
                         {
                             DlLogW << "[HTTP] parse size failed: " << size_param << ", use whole image.";
                             patch_w = 0;
                             patch_h = 0;
                         }
                     }

                     auto patches = split_image_to_patches(bgr, patch_w, patch_h);
                     DlLogI << "[HTTP] split into " << patches.size() << " patches.";

                     // 4) 为每个 patch 构造 DlFrame、异步提交推理任务
                     auto h2d_copy_helper = dl_device->getMemCopyHelper(cudaMemcpyHostToDevice);

                     struct TaskCtx
                     {
                         std::future<std::shared_ptr<DlnneUserOutputBase>> fut;
                         int offset_x;
                         int offset_y;
                     };
                     std::vector<TaskCtx> tasks;
                     tasks.reserve(patches.size());

                     for (size_t i = 0; i < patches.size(); ++i)
                     {
                         const auto& p = patches[i];

                         // 确保 patch 宽高为偶数
                         int even_w = (p.bgr.cols / 2) * 2;
                         int even_h = (p.bgr.rows / 2) * 2;
                         if (even_w <= 0 || even_h <= 0) continue;

                         cv::Mat patch_even = p.bgr(cv::Rect(0, 0, even_w, even_h));

                         // BGR -> YUV420P (I420)
                         cv::Mat yuv;
                         cv::cvtColor(patch_even, yuv, cv::COLOR_BGR2YUV_I420);

                         auto host_frame = std::make_shared<DlFrame>(DlMemoryType_Host);
                         host_frame->resize(even_w, even_h, pixel_format);

                         if (host_frame->size != static_cast<uint64_t>(yuv.total()))
                         {
                             DlLogW << "[HTTP] WARN patch " << i << " frame_size=" << host_frame->size <<
                                 " != yuv.total=" << yuv.total();
                         }

                         std::memcpy(host_frame->data, yuv.data, std::min<uint64_t>(host_frame->size, yuv.total()));

                         auto dev_frame = std::make_shared<DlFrame>(DlMemoryType_Device);
                         dev_frame->index = static_cast<int>(i); // 仅调试用
                         dev_frame->resize(even_w, even_h, pixel_format);

                         h2d_copy_helper->addMemcpyTask(dev_frame->data, host_frame->data, host_frame->size).get();

                         auto input = dlnne_network_unit->createTestInput(dev_frame);
                         auto fut = dlnne_network_unit->addDlnneInferTask(input);

                         tasks.push_back(TaskCtx{std::move(fut), p.offset_x, p.offset_y});
                     }

                     // 5) 等待所有 patch 的结果，并汇总
                     //    这里简单返回 JSON 数组，每个元素是一个检测框
                     std::ostringstream oss;
                     oss << R"({"code":0,"msg":"ok","detections":[)";

                     bool first_det = true;

                     for (auto& tk : tasks)
                     {
                         auto base_output = tk.fut.get();
                         auto yolo_out = std::dynamic_pointer_cast<DlnneYolov5Output>(base_output);
                         if (!yolo_out) continue;

                         for (auto& rect : yolo_out->rect_vector)
                         {
                             // 如果你想返回“原图全局坐标”，在这里加偏移即可：
                             int gx1 = rect.left + tk.offset_x;
                             int gy1 = rect.top + tk.offset_y;
                             int gx2 = rect.right + tk.offset_x;
                             int gy2 = rect.bottom + tk.offset_y;

                             if (!first_det)
                             {
                                 oss << ",";
                             }
                             first_det = false;

                             oss << "{"
                                 << R"("cls":)" << rect.classification << ","
                                 << R"("score":)" << rect.precision << ","
                                 << R"("x1":)" << gx1 << ","
                                 << R"("y1":)" << gy1 << ","
                                 << R"("x2":)" << gx2 << ","
                                 << R"("y2":)" << gy2
                                 << "}";
                         }
                     }

                     oss << "]}";

                     res.status = 200;
                     res.set_content(oss.str(), "application/json");
                 }
                 catch (const std::exception& e)
                 {
                     DlLogE << "[HTTP] exception: " << e.what();
                     res.status = 500;
                     std::string msg = std::string(R"({"code":500,"msg":")") + e.what() + "\"}";
                     res.set_content(msg, "application/json");
                 } catch (...)
                 {
                     DlLogE << "[HTTP] unknown exception";
                     res.status = 500;
                     res.set_content(R"({"code":500,"msg":"unknown error"})", "application/json");
                 }
             });




    // 简单的健康检查
    svr.Get("/ping", [](const httplib::Request&, httplib::Response& res)
    {
        DlLogI << "收到Get Ping请求";
        res.set_content("pong\n", "text/plain");
    });

    int port = 6650;
    DlLogI << "Listening on 0.0.0.0:" << port;
    svr.listen("0.0.0.0", port);

    dlnne_device->releaseDlAlgorithmUnit(network_unit_name);
    DlLogI << "HTTP Yolov5 server stop.";
    return 0;
}
