#ifndef DLNNE_ALGO_UNIT_YOLOV5_H
#define DLNNE_ALGO_UNIT_YOLOV5_H

#include "dlnne/dlnne_unit.h"

namespace dl {

    class DlnneYolov5Input : public DlnneUserInputBase
    {
    public:
        explicit DlnneYolov5Input(const std::shared_ptr<DlFrame>& frame, float threshold)
            : frame(frame), threshold(threshold) {}

        std::shared_ptr<DlFrame> frame;
        float threshold = 0.5;
    };

    class DlnneYolov5Output : public DlnneUserOutputBase
    {
    public:
        explicit DlnneYolov5Output() {}

        std::vector<DlnneRectI> rect_vector;
    };


    class DlAlgorithmUnitYolov5 : public DlnneNetworkUnit
    {
    public:
        DlAlgorithmUnitYolov5(DlcudaDevice* device, const std::string& name = "Yolov5");
        ~DlAlgorithmUnitYolov5();

        std::shared_ptr<DlnneUserInputBase> createTestInput(std::shared_ptr<DlFrame> frame) override;
        bool checkNetworkOutput(std::shared_ptr<DlnneUserOutputBase> output) override;

    protected:
        void preProcess(std::shared_ptr<DlnneUserInputBase>, int batch_index, DlnneNetworkInout* device_inout, cudaStream_t stream) override;
        std::shared_ptr<DlnneUserOutputBase> postProcess(std::shared_ptr<DlnneUserInputBase> input, int batch_index, DlnneNetworkInout* host_inout) override;

    private:
        int m_input_width = 0;
        int m_input_height = 0;

        const float m_anchors[3][6] = { {10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},{116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };
        const float m_stride[3] = { 8.0, 16.0, 32.0 };
        std::string m_classesFile;

        float m_mean1 = 0.0f;
        float m_mean2 = 0.0f;
        float m_mean3 = 0.0f;
        float m_scale = 1.0f / 255;

        float m_confThreshold = 0.5;
        float m_nmsThreshold = 0.4;
        float m_objThreshold = 0.5;

        std::vector<std::string> classes;
    };


}

#endif // DLNNE_ALGO_UNIT_YOLOV5_H
