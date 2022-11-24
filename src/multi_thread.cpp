#include <cnrt.h>
#include <mm_parser.h>
#include <mm_runtime.h>
#include <utils.hpp>
#include <thread>
#include <unistd.h>

using namespace std;
using namespace magicmind;

void print_data_info(float input[9], float output){
    printf("-------------------------\n");
    printf("input: %g %g %g  result: %g\n", input[0],input[1],input[2], output);
    printf("       %g %g %g  \n", input[3],input[4],input[5]);
    printf("       %g %g %g  \n", input[6],input[7],input[8]);
}

void infer_fn(cnrtQueue_t queue,IContext *context, bool *run_flag){
    // 4.crete input and output tensor
    vector<IRTTensor *> input_tensors, output_tensors;
    MM_CHECK_OK(context->CreateInputTensors(&input_tensors));
    MM_CHECK_OK(context->CreateOutputTensors(&output_tensors));

    // set input dim and infer output shape
    MM_CHECK_OK(input_tensors[0]->SetDimensions(Dims({1, 1, 3, 3})));
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

    float input_data[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    while(*run_flag){
        // data copyin
        CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), &input_data, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

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

        print_data_info(input_data, output_data[0]);
    
    }
    
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

    bool run_flag = true;
    std::thread t(infer_fn, queue, context, &run_flag);

    auto fit = magicmind::CreateIRefitter(engine);

    // create new weight tensor
    auto weight = magicmind::CreateIRTTensor(magicmind::DataType::FLOAT32,
                                             "const_data",
                                             Layout::NONE,
                                             TensorLocation::kHost);
    weight->SetDimensions(Dims({1, 1 ,3 ,3}));

    for(int i = 0 ;i < 10; ++i){
        float i_f = float(i);
        float new_weight[9] = {i_f, i_f, i_f, i_f, i_f, i_f, i_f, i_f, i_f};
        weight->SetData(&new_weight);
        MM_CHECK_OK(fit->SetNamedWeights("main/mm.const/mm.refit_const:0", weight));
        MM_CHECK_OK(fit->RefitEngine());
        sleep(1);
    }

    run_flag = false;
    
    t.join();

    context->Destroy();
    engine->Destroy();
    model->Destroy();
    return 0;
}