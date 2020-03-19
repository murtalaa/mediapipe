//client.cpp
#include "client.h"

client::client(boost::asio::io_service &io, char *host)
    : socket(io),
      resolver(io),
      query(udp::v4(), host, UDP_PORT)
{
    std::cout << "Host: " << host << std::endl;
    socket.open(udp::v4());
    receiver_endpoint = *resolver.resolve(query);
    std::cout << "Reciever Endpoint: " << receiver_endpoint << std::endl;
}

client::~client()
{
    std::cout << "Client Closed" << std::endl;
}

bool client::send_msg()
{
    try
    {
        boost::array<char, 1> send_buf = {{0}};
        socket.send_to(boost::asio::buffer(send_buf), receiver_endpoint);
        return true;
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}

void client::recv_msg(mediapipe::Coordinates *features)
{
    try
    {
        udp::endpoint sender_endpoint;
        boost::asio::streambuf b;
        auto bufs = b.prepare(4096 * 4);
        size_t len = socket.receive_from(
            bufs, sender_endpoint);
        b.commit(len);
        std::istream os(&b);
        std::cout << "Recieve Data of Matrix (" << len << ")" << std::endl;
        if (!features->ParseFromIstream(&os))
        {
            std::cerr << "Failed to Server Data." << std::endl;
            return;
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}
