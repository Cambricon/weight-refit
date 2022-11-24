#include <mm_parser.h>
#include <mm_builder.h>
#include <mm_runtime.h>
#include <mm_remote.h>
#include <cnrt.h>
#include <sys/stat.h>
#include <memory>
#include <CLI11.hpp>

#include "utils.hpp"

using namespace magicmind;
using namespace std;


struct Args
{
    string model = "conv.pt";
    string output_model_file = "conv.mm";
};


int main(int argc, char **argv){
    Args args;
    CLI::App app{"weight refit demo"};
    app.add_option("-o", args.output_model_file, "output model path. ");
    CLI11_PARSE(app,argc, argv);
    
    // 1. parse netowrk
    auto parser = magicmind::CreateIParser<magicmind::ModelKind::kPytorch, string>();
    parser->SetModelParam("pytorch-input-dtypes", {magicmind::DataType::FLOAT32});
    auto network = magicmind::CreateINetwork();
    MM_CHECK_OK(parser->Parse(network, args.model));

    // build model
    auto builder = magicmind::CreateIBuilder();
    auto build_config = CreateIBuilderConfig();
    build_config->ParseFromString(R"({"enable_refit": true})");
    // build_config->ParseFromString(R"({"debug_config": {"print_ir": {"print_level": 1}}})");
    auto model = builder->BuildModel("model", network, build_config);

    // save model
    model->SerializeToFile(args.output_model_file.c_str());

    // destroy resouce
    build_config->Destroy();
    builder->Destroy();
    parser->Destroy();
    network->Destroy();
    model->Destroy();
    return 0;
}