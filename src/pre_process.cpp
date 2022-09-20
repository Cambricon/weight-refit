#include "pre_process.hpp"
#include "utils.hpp"

bool load_imagenet(std::vector<cv::Mat> &imgs, std::vector<int> &labels,std::string val_txt, std::string image_file_path, int count)
{
    std::map<cv::Mat, int> dataset;
    std::ifstream in(val_txt);
    std::vector<std::string> lines;
    if (!in)
    {
        std::cout << val_txt << ", file not exists\n";
        return false;
    }
    std::string line;
    int image_count = 0;
    while (getline(in, line))
    {
        std::vector<std::string> strs = split(line, " ");
        std::string image_path = image_file_path + "/" + strs[0];
        auto img = cv::imread(image_path);
        if (img.empty())
        {
            return false;
        }
        auto label = std::stoi(strs[1]);
        imgs.push_back(img.clone());
        labels.push_back(label);
        image_count += 1;
        if (image_count >= count)
            break;
    }
    return true;
}

cv::Mat process_img(cv::Mat img, bool transpose, bool normlize){
    size_t h = img.rows;
    size_t w = img.cols;
    float scale = h < w ? scale = 256. / h:256. / w;
    size_t new_h = h * scale;
    size_t new_w = w * scale;

    cv::resize(img, img, cv::Size(new_w, new_h));
    size_t left_x = int((new_w - 224) / 2);
    size_t top_y = int((new_h - 224) / 2);
    img = img(cv::Rect(left_x, top_y, 224, 224)).clone();
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    if(normlize){
        img.convertTo(img, CV_32F);
        cv::Scalar mean(0.485, 0.456, 0.406);
        cv::Scalar std(1.0/0.229, 1.0/0.224, 1.0/0.225);
        img /=255.0;
        img -= mean;
        cv::multiply(img, std, img);
    }

    cv::Mat blob;
    if (transpose)
    {
        int c = img.channels();
        int h = img.rows;
        int w = img.cols;
        int sz[] = {1, c, h, w};
        blob.create(4, sz, img.depth());
        cv::Mat ch[3];
        for (int j = 0; j < c; j++)
        {
            ch[j] = cv::Mat(img.rows, img.cols, img.depth(), blob.ptr(0, j));
        }
        cv::split(img, ch);
    }
    else
    {
        blob = img;
    }

    return blob;
}
