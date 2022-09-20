#include "calibrate.hpp"

CVImageCalibData::CVImageCalibData(std::vector<cv::Mat> calib_imgs)
{
    if(calib_imgs.size() == 0){
        std::cout<<"Calibrate data is empty.\n";
        exit(0);
    }
    if(calib_imgs[0].depth()!=CV_32F){
        std::cout<<"Calibrate data type must be float type(CV_32F).\n";
        exit(0);
    }
    current_index_ = 0;
    max_count_ = calib_imgs.size();
    imgs_ = calib_imgs;
    std::vector<int64_t> shape;
    for (int i = 0; i < calib_imgs[0].dims; ++i)
    {
        shape.push_back(calib_imgs[0].size[i]);
    }
    shape_ = magicmind::Dims(shape);
    buffer_.resize(shape_.GetElementCount());
    float *data_ptr = (float *)imgs_[current_index_].data;
    std::vector<float> buffer(data_ptr, data_ptr + shape_.GetElementCount());
    buffer_ = buffer;
}

magicmind::Dims CVImageCalibData::GetShape() const
{
    return shape_;
}

void *CVImageCalibData::GetSample()
{
    return (void *)buffer_.data();
}

magicmind::Status CVImageCalibData::Next()
{
    if (current_index_ >= max_count_)
    {
        std::string msg = "sample number is bigger than max sample number!\n";
        magicmind::Status status_(magicmind::error::Code::OUT_OF_RANGE, msg);
        return status_;
    }
    float *data_ptr = (float *)imgs_[current_index_].data;
    std::vector<float> buffer(data_ptr, data_ptr + (shape_.GetElementCount()));
    buffer_ = buffer;
    current_index_ += 1;
    return magicmind::Status::OK();
}

magicmind::Status CVImageCalibData::Reset()
{
    current_index_ = 0;
    return magicmind::Status::OK();
}

magicmind::DataType CVImageCalibData::GetDataType() const
{
    return magicmind::DataType::FLOAT32;
}
