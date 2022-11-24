#include <cnrt.h>
#include <mm_parser.h>
#include <mm_runtime.h>
#include <utils.hpp>

using namespace std;
using namespace magicmind;

void print_data_info(float input[9], float kernel[9], float output){
    printf("input: %g %g %g  conv kernel: %g %g %g  result: %g\n", input[0],input[1],input[2],kernel[0], kernel[1], kernel[2], output);
    printf("       %g %g %g               %g %g %g \n", input[3],input[4],input[5],kernel[3], kernel[4], kernel[5]);
    printf("       %g %g %g               %g %g %g \n", input[6],input[7],input[8],kernel[6], kernel[7], kernel[8]);
}


int main()
{
    // 1. cnrt init
    cnrtQueue_t queue;
    cnrtQueueCreate(&queue);

    IModel *model = CreateIModel();
    model->DeserializeFromFile("add.mm");

    // 2.crete engine
    auto engine = model->CreateIEngine();
    assert(engine != nullptr);

    // 3.create context
    auto context = engine->CreateIContext();
    assert(context != nullptr);

    // 4.crete input and output tensor
    vector<IRTTensor *> input_tensors, output_tensors;
    MM_CHECK_OK(context->CreateInputTensors(&input_tensors));
    MM_CHECK_OK(context->CreateOutputTensors(&output_tensors));
    
    // set input dim and infer output shape
    MM_CHECK_OK(input_tensors[0]->SetDimensions(Dims({1, 1, 3 ,3})));
    MM_CHECK_OK(context->InferOutputShape(input_tensors, output_tensors));

    // mlu input memery alloc
    void *mlu_input_ptr;
    CNRT_CHECK(cnrtMalloc(&mlu_input_ptr, input_tensors[0]->GetSize()));
    MM_CHECK_OK(input_tensors[0]->SetData(mlu_input_ptr));

    // mlu output memory alloc
    void *mlu_output_ptr;
    CNRT_CHECK(cnrtMalloc(&mlu_output_ptr, output_tensors[0]->GetSize()));
    MM_CHECK_OK(output_tensors[0]->SetData(mlu_output_ptr));

    // cpu output memory alloc
    void *output_cpu_ptr = malloc(output_tensors[0]->GetSize());

    // data copyin  
    float input_data[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(),&input_data, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

    // infer
    MM_CHECK_OK(context->Enqueue(input_tensors, output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // data copyout
    CNRT_CHECK(cnrtMemcpy(output_cpu_ptr,
                          output_tensors[0]->GetMutableData(),
                          output_tensors[0]->GetSize(),
                          CNRT_MEM_TRANS_DIR_DEV2HOST));

    // print result
    std::vector<float> output_data((float *)output_cpu_ptr,
                                 (float *)output_cpu_ptr +
                                     (output_tensors[0]->GetDimensions().GetElementCount()));
    float origin_weight[] = {0,0,0,0,0,0,0,0,0,0};
    print_data_info(input_data, origin_weight, output_data[0]);

    // weight refit start
    std::cout << "--- weight refit ---" <<"\n";
    float new_weight[9] = {1,1,1,1,1,1,1,1,1};
    auto fit = magicmind::CreateIRefitter(engine);

    // vector<string> names;
    // fit->GetAllNames(&names);
    // for(auto name: names){
    //     std::cout<< name <<"\n";
    // }

    // create new weight tensor
    auto weight = magicmind::CreateIRTTensor(magicmind::DataType::FLOAT32,
                                             "const_data",
                                             Layout::NONE,
                                             TensorLocation::kHost);
    MM_CHECK_OK(weight->SetDimensions(Dims({1,1,3,3})));
    MM_CHECK_OK(weight->SetData(&new_weight));

    // weight refit
    MM_CHECK_OK(fit->SetNamedWeights("main/mm.const/mm.refit_const:0", weight));
    MM_CHECK_OK(fit->RefitEngine());

    // data copyin 
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(),
                          &input_data, input_tensors[0]->GetSize(),
                          CNRT_MEM_TRANS_DIR_HOST2DEV));

    // infer
    MM_CHECK_OK(context->Enqueue(input_tensors, output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // data copyout
    CNRT_CHECK(cnrtMemcpy(output_cpu_ptr,
                          output_tensors[0]->GetMutableData(),
                          output_tensors[0]->GetSize(),
                          CNRT_MEM_TRANS_DIR_DEV2HOST));

    // print result
    std::vector<float> output_data2((float *)output_cpu_ptr,
                                 (float *)output_cpu_ptr +
                                     (output_tensors[0]->GetDimensions().GetElementCount()));

    print_data_info(input_data, new_weight, output_data2[0]);

    free(output_cpu_ptr);

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