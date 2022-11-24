#ifndef _SAMPLE_UTILS111_HPP
#define _SAMPLE_UTILS111_HPP

#include <fstream>
#include <iostream>
#include <regex>
#include "sys/stat.h"

#define MM_CHECK_OK(status)                                                                           \
    do                                                                                                \
    {                                                                                                 \
        auto ret = (status);                                                                          \
        if (ret != magicmind::Status::OK())                                                           \
        {                                                                                             \
            std::cout << "[" << __FILE__ << ":" << __LINE__ << "]  mm failure: " << ret << std::endl; \
            abort();                                                                                  \
        }                                                                                             \
    } while (0)

#endif