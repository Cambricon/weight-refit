#include <mm_runtime.h>
#include <cnrt.h>
#include <CLI11.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "pre_process.hpp"
#include "utils.hpp"

using namespace magicmind;
using namespace std;
using namespace cv;

struct Args
{
    string model_file = "resnet18_model";
    string image_file = "data/images/ILSVRC2012_val_00000001.JPEG";
    string name_file = "data/names.txt";
};


int main(int argc, char **argv)
{
    Args args;
    CLI::App app{"yolov3 caffe demo"};
    app.add_option("-m", args.model_file, "output model path")->check(CLI::ExistingFile);
    app.add_option("-i", args.image_file, "predict image file")->check(CLI::ExistingFile);
    app.add_option("-c", args.name_file, "name file")->check(CLI::ExistingFile);
    CLI11_PARSE(app, argc, argv);

    // 1. cnrt init
    cnrtQueue_t queue;
    cnrtQueueCreate(&queue);

    IModel *model = CreateIModel();
    model->DeserializeFromFile(args.model_file.c_str());

    // 2.crete engine
    auto engine = model->CreateIEngine();
    PTR_CHECK(engine);

    // 3.create context
    auto context = engine->CreateIContext();
    PTR_CHECK(context);

    // 4.crete input tensor and output tensor and memory alloc
    vector<magicmind::IRTTensor *> input_tensors, output_tensors;
    MM_CHECK_OK(context->CreateInputTensors(&input_tensors));
    MM_CHECK_OK(context->CreateOutputTensors(&output_tensors));
    MM_CHECK_OK(context->InferOutputShape(input_tensors, output_tensors));

    // input tensor memory alloc
    for (auto tensor : input_tensors)
    {
        void *mlu_addr_ptr;
        CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()));
        MM_CHECK_OK(tensor->SetData(mlu_addr_ptr));
    }

    // output tensor memory alloc
    for (auto tensor : output_tensors)
    {
        void *mlu_addr_ptr;
        CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()));
        MM_CHECK_OK(tensor->SetData(mlu_addr_ptr));
    }

    Mat img = imread(args.image_file);
    if (img.empty())
    {
        std::cout << "Failed to open image file " + args.image_file;
        exit(1);
    }
    img = process_img(img, false, false);
    if (!check_file_exist(args.image_file))
    {
        std::cout << "image file " + args.image_file + " not found.\n";
        exit(1);
    }

    // 5. copy data from cpu to mlu
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img.data, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 6. compute
    MM_CHECK_OK(context->Enqueue(input_tensors, output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 7. copy result from mlu to cpu
    void *output_cpu_ptrs = (void *)malloc(output_tensors[0]->GetSize());

    CNRT_CHECK(cnrtMemcpy(output_cpu_ptrs, output_tensors[0]->GetMutableData(), output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
    std::vector<float> output_data((float *)output_cpu_ptrs, (float *)output_cpu_ptrs + (output_tensors[0]->GetSize() / sizeof(float)));
    std::vector<int> sorted_index = sort_indexes(output_data, true);

    map<int, string> name_map = load_names(args.name_file);
    
    // print result 
    Record record("infer_result.txt");
    for (int i = 0; i < 5; ++i)
    {
        std::string res = "top" + to_string(i) + ", idx:" + to_string(sorted_index[i]) + ", " + name_map[sorted_index[i]];
        record.write(res, true);
    }

    // create weight refit 
    auto fit = magicmind::CreateIRefitter(engine);

    auto weight = magicmind::CreateIRTTensor(magicmind::DataType::FLOAT32, "test", Layout::NONE, TensorLocation::kHost);
    weight->SetDimensions(Dims({1000}));
    
    // generate random data
    void *cpu_ptr = malloc(weight->GetSize());
    for(int i = 0 ; i < weight->GetDimensions().GetElementCount(); ++i){
        *((float *)(cpu_ptr) + i) = rand()/1000000;
    }
    weight->SetData(cpu_ptr);


    // apply weight refit
    std::cout << "---------------------------- replace fc layer weight  ------------------------------------" <<"\n";
    auto status = fit->SetNamedWeights("resnetv15_dense0_bias", weight);
    if(!status.ok()){
        std::cout<< "err: "  << status.error_message()<<"\n";
    }
    status = fit->RefitEngine();
    if(!status.ok()){
        std::cout<< "err: "  << status.error_message()<<"\n";
    }
       
    // infer
    MM_CHECK_OK(context->Enqueue(input_tensors, output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // copy result from mlu to cpu
    CNRT_CHECK(cnrtMemcpy(output_cpu_ptrs, output_tensors[0]->GetMutableData(), output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
    std::vector<float> output_data2((float *)output_cpu_ptrs, (float *)output_cpu_ptrs + (output_tensors[0]->GetSize() / sizeof(float)));
    std::vector<int> sorted_index2 = sort_indexes(output_data2, true);
    
    // print result, because modify weight to randome data, result is wrong.
    Record record2("infer_result2.txt");
    for (int i = 0; i < 5; ++i)
    {
        std::string res = "top" + to_string(i) + ", idx:" + to_string(sorted_index2[i]) + ", " + name_map[sorted_index2[i]];
        record2.write(res, true);
    }

    free(output_cpu_ptrs);

    // 8. destroy resource
    for (auto tensor : input_tensors)
    {
        cnrtFree(tensor->GetMutableData());
        tensor->Destroy();
    }
    for (auto tensor : output_tensors)
    {
        cnrtFree(tensor->GetMutableData());
        tensor->Destroy();
    }
    context->Destroy();
    engine->Destroy();
    model->Destroy();
    return 0;
}