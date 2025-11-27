#include "dlutil/dl_log.h"
#include "dlnne_algo_unit_yolov5.h"
#include "dlnne/dl_network_runner.h"
#include "dlcuda/dl_device.h"
#include "dlcuda/dl_kernel_helper.h"
#include "dlpreprocess/include/image_proc.h"
#include "dlutil/dl_math.h"
#include "dlcuda/dl_memory_copy.h"
#include "dlcuda/dl_packet_frame.h"
#include "dlvid/dl_vpu_device.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>
#include <algorithm>

using namespace std;
using namespace dl;


inline void sigmoid(float* out, int length)
{
    for (int i = 0; i < length; i++)
    {
        out[i] = 1.0 / (1 + expf(-out[i]));
    }
}

typedef enum
{
    IoU_UNION = 0,
    IoU_MIN,
    IoU_MAX
} DL_IOU_E;

float box_IoUI(DlnneRectI _stRect1, DlnneRectI _stRect2, DL_IOU_E _eIouType)
{
    float f32OverlapRatio = 0.0f;
    float f32Eps = 1e-15f;

    float f32InterArea = 0.0f;
    float f32Rect1Area = 0.0f;
    float f32Rect2Area = 0.0f;

    float f32Left = static_cast<float>(max(_stRect1.left, _stRect2.left));
    float f32Top = static_cast<float>(max(_stRect1.top, _stRect2.top));
    float f32Right = static_cast<float>(min(_stRect1.right, _stRect2.right));
    float f32Bottom = static_cast<float>(min(_stRect1.bottom, _stRect2.bottom));

    f32InterArea = max(0.0f, f32Bottom - f32Top + 1) * max(0.0f, f32Right - f32Left + 1);
    if (fabs(f32InterArea) < (float)1e-6)
    {
        goto EXIT;
    }

    f32Rect1Area = static_cast<float>((_stRect1.bottom - _stRect1.top + 1) * (_stRect1.right - _stRect1.left + 1));
    f32Rect2Area = static_cast<float>((_stRect2.bottom - _stRect2.top + 1) * (_stRect2.right - _stRect2.left + 1));

    switch (_eIouType)
    {
    case IoU_UNION:
        f32OverlapRatio = f32InterArea / (f32Rect1Area + f32Rect2Area - f32InterArea + f32Eps);
        break;
    case IoU_MAX:
        f32OverlapRatio = f32InterArea / (max(f32Rect1Area, f32Rect2Area) + f32Eps);
        break;
    case IoU_MIN:
        f32OverlapRatio = f32InterArea / (min(f32Rect1Area, f32Rect2Area) + f32Eps);
        break;
    default:
        f32OverlapRatio = 0.0f;
        break;
    }

EXIT:
    return f32OverlapRatio;
}

void dlNMSBoxes(vector<DlnneRectI>& boxs, float iou_thresh)
{
    sort(boxs.begin(), boxs.end(), [](const DlnneRectI& box1, const DlnneRectI& box2)
    {
        return box1.precision > box2.precision;
    });

    for (auto it = boxs.begin(); it != boxs.end(); it++)
    {
        for (auto tm = it + 1; tm != boxs.end(); tm++)
        {
            if (tm->classification != it->classification)
            {
                continue;
            }
            if (box_IoUI(*it, *tm, IoU_UNION) > iou_thresh)
            {
                tm->classification = -1;
            }
        }
    }

    for (int i = boxs.size() - 1; i >= 0; i--)
    {
        if (-1 == boxs[i].classification)
            boxs.erase(boxs.begin() + i);
    }
}

namespace
{
    // 把 box 坐标裁剪到 [0, img_w-1] × [0, img_h-1]
    inline void clip_box_to_image(DlnneRectI &r, int img_w, int img_h)
    {
        if (r.left   < 0)        r.left   = 0;
        if (r.top    < 0)        r.top    = 0;
        if (r.right  >= img_w)   r.right  = img_w - 1;
        if (r.bottom >= img_h)   r.bottom = img_h - 1;
    }
}


namespace dl
{
    DlAlgorithmUnitYolov5::DlAlgorithmUnitYolov5(DlcudaDevice* device, const std::string& name)
        : DlnneNetworkUnit(device, name)
    {
        m_max_batch = 28;
        m_model_path = "/dl/DlDeepToolkit-master/c++/resource/yolov5s.onnx";
        //m_model_path = "../../../resource/yolov5s.slz";
        m_classesFile = "/dl/DlDeepToolkit-master/c++/resource/coco.names";

        m_model_path = "/dl/DlDeepToolkit-master/c++/resource/onion59_28x3x640x640.slz";
        m_classesFile = "/dl/DlDeepToolkit-master/c++/resource/onion_yolo.names";

        // 序列化模型默认以.slz文件后缀名
        m_is_serialized_engine = (m_model_path.find(".slz") != std::string::npos);

        m_input_width = 640;
        m_input_height = 640;

        ifstream ifs(m_classesFile.c_str());
        string line;
        while (getline(ifs, line))
            classes.push_back(line);

        startExecutor();
    }

    DlAlgorithmUnitYolov5::~DlAlgorithmUnitYolov5()
    {
        stopExecutor();
    }

    void DlAlgorithmUnitYolov5::preProcess(std::shared_ptr<DlnneUserInputBase> input, int batch_index, DlnneNetworkInout* device_inout, cudaStream_t stream)
    {
        auto yolov5_input = std::dynamic_pointer_cast<DlnneYolov5Input>(input);
        auto frame = yolov5_input->frame;

        auto binding = device_inout->network_binding_list.at(0);
        auto dest = (char*)(device_inout->binding_data_array[0]) + batch_index * binding->getBindingSize() / binding->batch_count;

        /* 预处理核函数 (pixel_value - mean) * scale *,  BRG -> RGB顺序 */
        YU12ToRGBBilinearResizeNormPlane((uint8_t*)frame->data, (float*)(dest), frame->width, frame->height, m_input_width, m_input_height,
                                         m_input_width, m_input_height, 0, 0, m_mean1, m_mean2, m_mean3, m_scale, m_scale, m_scale, 1.0, 0.0f, 0.0f, 0.0f, stream);
    }

    std::shared_ptr<DlnneUserOutputBase> DlAlgorithmUnitYolov5::postProcess(std::shared_ptr<DlnneUserInputBase> input, int batch_index, DlnneNetworkInout* host_inout)
    {
        // ---- 批次统计 & 计时（executor 内部是单线程调用 postProcess，用 static 就够了）----
        int batch_size = host_inout->execute_batch;
        static int s_batch_id = 0;
        static std::chrono::steady_clock::time_point s_batch_start;

        if (batch_index == 0)
        {
            // 新的一个 batch 开始
            ++s_batch_id;
            s_batch_start = std::chrono::steady_clock::now();
            // DlLogI << "[Yolov5 postProcess] batch " << s_batch_id << " start, batch_size = " << batch_size;
        }

        /* 分析当前batch的算法结果，保存到yolonano_output中 */
        auto yolov5_output = std::make_shared<DlnneYolov5Output>();
        auto yolov5_input = std::dynamic_pointer_cast<DlnneYolov5Input>(input);

        /* 获取所有输出bindings */
        std::vector<DlnneNetworkBinding*> outBindings;
        for (auto binding : host_inout->network_binding_list)
        {
            /* 忽略输入binding */
            if (binding->is_input)
                continue;
            outBindings.push_back(binding);
        }

        // 原图尺寸与网络输入的比例（用于把坐标缩回原图）
        float ratioh = (float)yolov5_input->frame->height / m_input_height;
        float ratiow = (float)yolov5_input->frame->width / m_input_width;
        int class_num = classes.size();

        int q = 0, i = 0, j = 0, nout = class_num + 5, c = 0;    // 每个 anchor：cx,cy,w,h,obj + num_classes

        int max_bindings = std::min(outBindings.size(), (size_t)3);

        /* 多尺度检测结果输出 */
        for (int n = 0; n < max_bindings; ++n)
        {
            DlnneNetworkBinding* binding = outBindings[n];

            // 当前 batch 的输出首地址
            auto data_c = (char*)(binding->data) + batch_index * binding->getBindingSize() / binding->batch_count;
            float* data_f = (float*)data_c;

            int num_grid_x = (int)(m_input_width / m_stride[n]);
            int num_grid_y = (int)(m_input_height / m_stride[n]);
            int area = num_grid_x * num_grid_y;

            // 3 个 anchor
            for (q = 0; q < 3; ++q)
            {
                const float& anchor_w = m_anchors[n][q * 2];
                const float& anchor_h = m_anchors[n][q * 2 + 1];

                // 这一组 anchor 的起始指针
                // HWC 布局: [num_grid_y, num_grid_x, nout]
                float* pdata = data_f + q * nout * area;

                for (i = 0; i < num_grid_y; ++i)
                {
                    for (j = 0; j < num_grid_x; ++j)
                    {
                        // ---- 1) 取 obj score ----
                        int base_idx = i * num_grid_x * nout + j * nout; // (i,j) 这一格的起点
                        float& box_score = pdata[base_idx + 4];
                        sigmoid(&box_score, 1);

                        if (box_score <= m_objThreshold)
                            continue;

                        // ---- 2) 取分类得分，找最大类 ----
                        float max_class_score = 0.f;
                        int max_class_id = 0;

                        for (c = 0; c < class_num; ++c)
                        {
                            float& cls_score = pdata[base_idx + 5 + c];
                            sigmoid(&cls_score, 1);
                            if (cls_score > max_class_score)
                            {
                                max_class_score = cls_score;
                                max_class_id = c;
                            }
                        }

                        if (max_class_score <= m_confThreshold)
                            continue;

                        // ---- 3) 取回归框参数 cx,cy,w,h ----
                        float cx = pdata[base_idx + 0];
                        float cy = pdata[base_idx + 1];
                        float w = pdata[base_idx + 2];
                        float h = pdata[base_idx + 3];

                        sigmoid(&cx, 1);
                        sigmoid(&cy, 1);
                        sigmoid(&w, 1);
                        sigmoid(&h, 1);

                        // 映射到网络输入尺度（YOLOv5 的 decode 公式）
                        cx = (cx * 2.f - 0.5f + j) * m_stride[n];
                        cy = (cy * 2.f - 0.5f + i) * m_stride[n];
                        w = powf(w * 2.f, 2.f) * anchor_w;
                        h = powf(h * 2.f, 2.f) * anchor_h;

                        // ---- 4) 映射回原图，并裁剪到图像内部 ----
                        DlnneRectI obj;
                        obj.left = static_cast<int>((cx - 0.5f * w) * ratiow);
                        obj.top = static_cast<int>((cy - 0.5f * h) * ratioh);
                        obj.right = static_cast<int>(obj.left + w * ratiow);
                        obj.bottom = static_cast<int>(obj.top + h * ratioh);
                        obj.classification = max_class_id;
                        obj.precision = max_class_score;

                        // 裁剪到原图范围
                        clip_box_to_image(obj, yolov5_input->frame->width, yolov5_input->frame->height);

                        // 无效框丢弃
                        if (obj.right <= obj.left || obj.bottom <= obj.top)
                            continue;

                        yolov5_output->rect_vector.push_back(obj);
                    }
                }
            }
        }

        // 3. NMS
        dlNMSBoxes(yolov5_output->rect_vector, m_nmsThreshold);

        // 记录 batch 结束耗时（默认 batch 中最后一张的 batch_index = s_batch_size-1）
        if (batch_index == (batch_size - 1))
        {
            auto end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - s_batch_start).count() / 1000.0;
            DlLogI << "[Yolov5 postProcess] batch " << s_batch_id << " done, cost = " << ms << " ms, batch_size = " << batch_size;;
        }

        yolov5_output->input = input;
        return yolov5_output;
    }

    std::shared_ptr<DlnneUserInputBase> DlAlgorithmUnitYolov5::createTestInput(std::shared_ptr<DlFrame> frame)
    {
        return std::make_shared<DlnneYolov5Input>(frame, 0.7);
    }

    bool DlAlgorithmUnitYolov5::checkNetworkOutput(std::shared_ptr<DlnneUserOutputBase> output)
    {
        /* 推理结果画框 */
        auto yolov5_output = std::dynamic_pointer_cast<DlnneYolov5Output>(output);
        auto dev_frame = std::dynamic_pointer_cast<DlnneYolov5Input>(output->input)->frame;
        auto dl_device = DlcudaDeviceManager::getInstance().getDlcudaDevice(dev_frame->getDeviceId());
        auto host_frame = std::make_shared<DlFrame>(DlMemoryType_Host);
        host_frame->resize(dev_frame->width, dev_frame->height, dev_frame->format);
        auto d2h_copy_helper = dl_device->getMemCopyHelper(cudaMemcpyDeviceToHost);
        d2h_copy_helper->addMemcpyTask(host_frame->data, dev_frame->data, dev_frame->size).get();

        DlLogI << "frame " << host_frame->index << "： ";
        for (auto rect : yolov5_output->rect_vector)
        {
            // DlLogI << "id=" << rect.classification << ", "
            //     << (!classes.empty() ? classes[rect.classification] : " ") << ", "
            //     << rect.precision << ", ("
            //     << rect.left << ","
            //     << rect.top << ","
            //     << rect.right << ","
            //     << rect.bottom << ")";

            rect.left = rect.left / 2 * 2;
            rect.right = rect.right / 2 * 2;
            rect.top = rect.top / 2 * 2;
            rect.bottom = rect.bottom / 2 * 2;
            auto y_data = (char*)host_frame->data;
            auto uv_data = y_data + host_frame->stride * host_frame->height;
            /* 画上下两条横边 */
            for (int i = 0; i < rect.right - rect.left; i++)
            {
                y_data[rect.top * host_frame->stride + rect.left + i] = 0x4c;
                y_data[(rect.top + 1) * host_frame->stride + rect.left + i] = 0x4c;
                y_data[rect.bottom * host_frame->stride + rect.left + i] = 0x4c;
                y_data[(rect.bottom + 1) * host_frame->stride + rect.left + i] = 0x4c;
                if (DlPixelFormat_YUV420P == host_frame->format)
                {
                    uv_data[rect.top / 2 * host_frame->stride / 2 + (rect.left + i) / 2] = 0x55;
                    uv_data[rect.bottom / 2 * host_frame->stride / 2 + (rect.left + i) / 2] = 0x55;
                    uv_data[(rect.top / 2 + host_frame->height / 2) * host_frame->stride / 2 + (rect.left + i) / 2] = 0xff;
                    uv_data[(rect.bottom / 2 + host_frame->height / 2) * host_frame->stride / 2 + (rect.left + i) / 2] = 0xff;
                }
            }

            /* 画左右两条竖边 */
            for (int i = 0; i < rect.bottom - rect.top; i++)
            {
                y_data[(rect.top + i) * host_frame->stride + rect.left] = 0x4c;
                y_data[(rect.top + i) * host_frame->stride + rect.left + 1] = 0x4c;
                y_data[(rect.top + i) * host_frame->stride + rect.right] = 0x4c;
                y_data[(rect.top + i) * host_frame->stride + rect.right + 1] = 0x4c;
                if (DlPixelFormat_YUV420P == host_frame->format)
                {
                    uv_data[(rect.top + i) / 2 * host_frame->stride / 2 + rect.left / 2] = 0x55;
                    uv_data[(rect.top + i) / 2 * host_frame->stride / 2 + rect.right / 2] = 0x55;
                    uv_data[((rect.top + i) / 2 + host_frame->height / 2) * host_frame->stride / 2 + rect.left / 2] = 0xff;
                    uv_data[((rect.top + i) / 2 + host_frame->height / 2) * host_frame->stride / 2 + rect.right / 2] = 0xff;
                }
            }
        }

        /* 保存修改后的图片 */
        auto dlvid_device = DlvidDeviceManager::getInstance().getVpuDevice(host_frame->getDeviceId());
        auto jpeg_encoder = dlvid_device->requireDlJpegEncoder();
        auto encodedBuffer = jpeg_encoder->encodeJpeg(host_frame);
        if (encodedBuffer != nullptr)
        {
            int random_num = rand() % 90000000 + 10000000;  // 公式：rand() % (max - min + 1) + min
            std::string filename = "./output" + std::to_string(random_num) + ".jpg";
            std::ofstream slz(filename);
            slz.write(static_cast<char*>(encodedBuffer->data), static_cast<int64_t>(encodedBuffer->size));
            slz.close();
        }
        dlvid_device->releaseDlJpegEncoder(jpeg_encoder);
        return true;
    }

    static DlnneNetworkUnit* createDlAlgorithmUnit(DlcudaDevice* device)
    {
        return new DlAlgorithmUnitYolov5(device);
    }

    static void releaseDlAlgorithmUnit(DlnneNetworkUnit* dl_algo_unit)
    {
        auto unit = dynamic_cast<DlAlgorithmUnitYolov5*>(dl_algo_unit);
        delete unit;
    }

    REGISTER_DL_ALGORITHM_UNIT_DES_CONS(DlAlgorithmUnit_Yolov5, releaseDlAlgorithmUnit, createDlAlgorithmUnit);
}
