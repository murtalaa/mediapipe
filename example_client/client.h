//client.h
#ifndef CLIENT_H
#define CLIENT_H
#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include "coordinate.pb.h"
#define UDP_PORT "7777"
using boost::asio::ip::udp;

class client
{
public:
    client(boost::asio::io_service &io, char *host);
    ~client();
    bool send_msg();
    void recv_msg(mediapipe::Coordinates * features);

private:
    udp::socket socket;
    udp::resolver resolver;
    udp::resolver::query query;
    udp::endpoint receiver_endpoint;
};

#endif