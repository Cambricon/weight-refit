#ifndef _SAMPLE_UTILS111_HPP
#define _SAMPLE_UTILS111_HPP

#include <fstream>
#include <iostream>
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

#define PTR_CHECK(ptr)                               \
    do                                               \
    {                                                \
        if (ptr == nullptr)                          \
        {                                            \
            std::cout << "mm failure " << std::endl; \
            abort();                                 \
        }                                            \
    } while (0)

class Record
{
public:
    Record(std::string filename){
        outfile.open(("output/" + filename).c_str(), std::ios::trunc | std::ios::out);
    }

    ~Record(){
        if(outfile.is_open())
            outfile.close();
    }

    void write(std::string line, bool print = false){
        outfile << line << std::endl;
        if (print)
        {
            std::cout << line << std::endl;
        }
    }

private:
    std::ofstream outfile;
};

inline std::vector<std::string> split(const std::string &in, const std::string &delim)
{
    std::regex re{delim};
    return std::vector<std::string>{
        std::sregex_token_iterator(in.begin(), in.end(), re, -1), std::sregex_token_iterator()};
}

inline bool check_file_exist(std::string path){
    struct stat buffer;
    if (stat(path.c_str(), &buffer) == 0)
    {
        if ((buffer.st_mode & S_IFDIR) == 0)
            return true;
        return false;
    }
    return false;
}

inline bool check_folder_exist(std::string path){
    struct stat buffer;
    if (stat(path.c_str(), &buffer) == 0)
    {
        if ((buffer.st_mode & S_IFDIR) == 0)
            return false;
        return true;
    }
    return false;
}

inline std::vector<int> sort_indexes(const std::vector<float> &v, bool reverse = false)
{
    std::vector<int> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i)
        idx[i] = i;
    if (reverse)
    {
        std::sort(idx.begin(), idx.end(),
                  [&v](int i1, int i2)
                  { return v[i1] > v[i2]; });
    }
    else
    {
        std::sort(idx.begin(), idx.end(),
                  [&v](int i1, int i2)
                  { return v[i1] < v[i2]; });
    }

    return idx;
}

inline std::map<int, std::string> load_names(std::string names_file)
{
    if (!check_file_exist(names_file))
    {
        std::cout << "imagenet name file: " + names_file + " does not exist.\n";
        exit(0);
    }
    std::map<int, std::string> name_map;
    std::ifstream in(names_file);
    if (!in)
    {
        std::cout << "failed to load imagenet name file: " + names_file + ".\n";
        exit(0);
    }
    std::string line;
    int index = 0;
    while (getline(in, line))
    {
        std::vector<std::string> idx_and_name = split(line, " ");
        int index = std::stoi(idx_and_name[0]);
        std::string name = "";
        for (int i = 1; i < idx_and_name.size(); ++i)
        {
            name += " "+idx_and_name[i];
        }
        name_map[index] = name;
    }
    return name_map;
}

#endif