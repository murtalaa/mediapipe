#include "client.h"
#include <opencv2/core/core.hpp>

using namespace cv;
using boost::asio::ip::udp;

void display_msg(const mediapipe::Coordinates &coord);
int main(int argc, char *argv[])
{
    char ip[20];
    std::printf("Enter the Host IP: ");
    std::scanf("%19s", ip);
    mediapipe::Coordinates features;
    try
    {
        boost::asio::io_service io_service;
        client cl(io_service, ip);
        std::cout << "Connecting..........\n";
        cl.send_msg();
        for(;;){
            
            cl.recv_msg(&features);
            display_msg(features);
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}

void print_matrix(const mediapipe::Matrix &mat)
{
    int count = 0;
    for (int i = 0; i < mat.row(); i++)
    {
        for (int j = 0; j < mat.col(); j++)
        {
            std::cout << mat.value(count++) << "\t";
        }
        std::cout << std::endl;
    }
}

cv::Mat decode_matrix(const mediapipe::Matrix &mat)
{
    int count = 0;
    cv::Mat ret = cv::Mat::zeros(mat.row(), mat.col(), mat.type());
    for (int i = 0; i < mat.row(); i++)
    {
        for (int j = 0; j < mat.col(); j++)
        {
            ret.at<double>(i, j) = mat.value(count++);
        }
    }
    return ret;
}

void display_msg(const mediapipe::Coordinates &coord)
{
    for (int i = 0; i < coord.transforms_size(); i++)
    {
        const mediapipe::Matrix &trans = coord.transforms(i);
        cv::Mat ret = decode_matrix(trans);
        std::cout << "Matrix " << i + 1 << "\n"
                  << ret << "\n"
                  << std::endl;
    }
}