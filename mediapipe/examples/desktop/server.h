//server.h
#ifndef SERVER_H
#define SERVER_H

#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <opencv2/core/core.hpp>
//#include "coordinate.pb.h"
#include "mediapipe/framework/formats/coordinate.pb.h"
#include <thread>
#include <future>
#define UDP_PORT 7777


using boost::asio::ip::udp;

class server
{
public:
    server(boost::asio::io_service &io, udp::endpoint &endpoint);
    ~server();
    void listen();
    void handler(const boost::system::error_code &error, std::size_t bytes_transferred);
    bool async_recv_msg();
    bool recv_msg();
    bool send_msg(mediapipe::Coordinates *features);

private:
    bool connected = false;
    udp::socket socket;
    boost::array<char, 1> recv_buf;
    udp::endpoint remote_endpoint;
};

#endif
