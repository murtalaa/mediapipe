//server.cpp
#include "server.h"
using namespace cv;

server::server(boost::asio::io_service &io, udp::endpoint &endpoint)
    : socket(io, endpoint)
{
    std::cout << "Server with IP: " << endpoint << " created \n";
}

server::~server()
{
    std::cout << "Server Closed "
              << "\n";
}

void server::listen()
{
    std::cout << "Waiting for Connection\n";
    do
    {
        std::cout << "Remote Endpoint (Before): " << server::remote_endpoint.port() << "\n";
        recv_msg();
        std::cout << "Remote Endpoint (After): " << server::remote_endpoint.port() << "\n";

    } while (server::remote_endpoint.port() == 0);

    server::connected = true;
}

void server::handler(const boost::system::error_code &error, std::size_t bytes_transferred)
{
    std::cout << "ulala" << std::endl;
    std::cout << "Received: '" << std::string(server::recv_buf.begin(), server::recv_buf.begin() + bytes_transferred) << "'\n";

    if (!error || error == boost::asio::error::message_size)
        recv_msg();
}

bool server::async_recv_msg()
{
    try
    {
        boost::system::error_code error;
        socket.async_receive_from(boost::asio::buffer(recv_buf), remote_endpoint,
                                  boost::bind(&server::handler, this, error,
                                              boost::asio::placeholders::bytes_transferred));
        if (error && error != boost::asio::error::message_size)
        {
            throw boost::system::system_error(error);
            server::connected = false;
            return false;
        }
        return true;
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }
}

bool server::recv_msg()
{
    try
    {
        boost::system::error_code error;

        socket.receive_from(boost::asio::buffer(server::recv_buf),
                            server::remote_endpoint, 0, error);
        if (error && error != boost::asio::error::message_size)
        {
            throw boost::system::system_error(error);
            server::connected = false;
            return false;
        }
        return true;
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }
}

bool server::send_msg(mediapipe::Coordinates *features)
{
    try
    {
        if (server::connected)
        {
            boost::system::error_code ignored_error;
            boost::asio::streambuf b;
            std::ostream os(&b);
            features->SerializeToOstream(&os);
            size_t len = socket.send_to(b.data(),
                                        server::remote_endpoint, 0, ignored_error);
            std::cout << "Sending Matrix of length (" << len << ")" << std::endl;
            return true;
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        server::connected = false;
        return false;
    }
    std::cout << "No client connected\n";
    return false;
}


