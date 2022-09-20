#ifndef _SAMPLE_CALIBRATE_HPP
#define _SAMPLE_CALIBRATE_HPP

#include <mm_calibrator.h>
#include <opencv2/imgproc.hpp>
#include <iostream>

class CVImageCalibData : public magicmind::CalibDataInterface{
public:
    CVImageCalibData(std::vector<cv::Mat> calib_imgs);
    void *GetSample() override;
    magicmind::Dims GetShape() const;
    magicmind::Status Next();
    magicmind::Status Reset();
    magicmind::DataType GetDataType() const;

private:
    std::vector<cv::Mat> imgs_;
    magicmind::Dims shape_;
    int max_count_;
    int current_index_;
    std::vector<float> buffer_;
};

#endif