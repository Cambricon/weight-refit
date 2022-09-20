#ifndef _SAMPLE_PRE_PROCESS_HPP
#define _SAMPLE_PRE_PROCESS_HPP

#include <map>
#include <regex>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


bool load_imagenet(std::vector<cv::Mat> &imgs, std::vector<int> &labels,std::string val_txt, std::string image_file_path, int count = -1);

cv::Mat process_img(cv::Mat img, bool transpose, bool normlize);

#endif