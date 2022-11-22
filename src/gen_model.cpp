#include <mm_parser.h>
#include <mm_builder.h>
#include <mm_runtime.h>
#include <mm_remote.h>
#include <cnrt.h>
#include <sys/stat.h>
#include <memory>


#include <CLI11.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "calibrate.hpp"
#include "pre_process.hpp"
#include "utils.hpp"


using namespace magicmind;
using namespace std;
using namespace cv;

struct Args
{
    string model = "resnet18-v1-7.onnx";
    string output_model_file = "resnet18_model";
    string build_config_file = "build_config.json";
    string val_txt_file = "data/val.txt";
    string image_file_path = "data/images";
};


int main(int argc, char **argv){
    Args args;
    CLI::App app{"yolov3 demo"};
    app.add_option("-o", args.output_model_file, "output model path. ");
    CLI11_PARSE(app,argc, argv);
    
    // 1. parse netowrk
    auto parser = magicmind::CreateIParser<magicmind::ModelKind::kOnnx, string>();

    auto network = magicmind::CreateINetwork();
    MM_CHECK_OK(parser->Parse(network, args.model));

    // 2. set input shape
    network->GetInput(0)->SetDimension(magicmind::Dims({1, 3, 224, 224}));

    // 4. set build_config
    auto build_config = magicmind::CreateIBuilderConfig();
    MM_CHECK_OK(build_config->ParseFromFile(args.build_config_file));

    // 5. calibrate
    vector<Mat> imgs;
    vector<int> labels;
    load_imagenet(imgs, labels,args.val_txt_file, args.image_file_path, 10);

    vector<Mat> calib_datas;
    for (auto img : imgs)
    {
        auto blob = process_img(img, true, true);
        calib_datas.push_back(blob);
    }

    CVImageCalibData calib_data(calib_datas);
    magicmind::ICalibrator *calibrator = magicmind::CreateICalibrator(&calib_data);


    MM_CHECK_OK(calibrator->Calibrate(network, build_config));

    // 6. build model
    auto builder = magicmind::CreateIBuilder();
    auto model = builder->BuildModel("model", network, build_config);

    // 7. save model
    model->SerializeToFile(args.output_model_file.c_str());

    // destroy resouce
    calibrator->Destroy();
    build_config->Destroy();
    builder->Destroy();
    parser->Destroy();
    network->Destroy();
    model->Destroy();
    return 0;
}