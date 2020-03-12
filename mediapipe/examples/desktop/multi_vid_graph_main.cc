#include <cstdlib>
#include <pthread.h>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/coordinate.pb.h"//Coordinate
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include <thread>
#include <iostream>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#define UDP_PORT 7777

using boost::asio::ip::udp;

void print(const boost::system::error_code& /*e*/,
    boost::asio::deadline_timer* t, int* count)
{
      if (*count < 10)
  {
    std::cout << *count << "\n";
    ++(*count);
        t->expires_at(t->expires_at() + boost::posix_time::seconds(5));
    t->async_wait(boost::bind(print,
          boost::asio::placeholders::error, t, count));
  }
}

void test()
{
  boost::asio::io_service io;
  int count = 0;
  boost::asio::deadline_timer t(io, boost::posix_time::seconds(1));

  t.async_wait(boost::bind(print,
        boost::asio::placeholders::error, &t, &count));

  io.run();
  std::cout << "Final count is " << count << "\n";
}

using namespace std;
using namespace cv;

const char kInputStream[] = "input_video";
const char kOutputStream[] = "output_video";
const char kOutputStream_[] = "output_video";
const char kWindowName[] = "MediaPipe";

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");


/*---------------------------Server-------------------------------*/
class server
{
public:
    server(boost::asio::io_service &io, udp::endpoint &endpoint)
        : socket(io, endpoint)
    {
        std::cout << "Server with IP: " << endpoint << " created \n";
    }

    ~server()
    {
        std::cout << "Server Closed "
                  << "\n";
    }

    void listen(){
        std::cout << "Waiting for Connection\n";
        do{
            std::cout << "Remote Endpoint (Before): " << remote_endpoint.port() << "\n";
            recv_msg();
            std::cout << "Remote Endpoint (After): " << remote_endpoint.port() << "\n";
            
        }
        while(remote_endpoint.port() == 0);

        connected = true;
    }

    void handler(const boost::system::error_code &error, std::size_t bytes_transferred)
    {
        std::cout << "ulala" << std::endl;
        std::cout << "Received: '" << std::string(recv_buf.begin(), recv_buf.begin() + bytes_transferred) << "'\n";

        if (!error || error == boost::asio::error::message_size)
            recv_msg();
    }

    bool recv_msg()
    {
        try
        {
            boost::system::error_code error;
            
            socket.receive_from(boost::asio::buffer(recv_buf),
                                remote_endpoint, 0, error);
            //socket.async_receive_from(boost::asio::buffer(recv_buf), remote_endpoint,
            //                          boost::bind(&server::handler, this, error,
            //                                      boost::asio::placeholders::bytes_transferred));
            if (error && error != boost::asio::error::message_size)
            {
                throw boost::system::system_error(error);
                connected = false;
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

    bool send_msg(mediapipe::Coordinates *features)
    {
        try
        {
            if (connected){
                boost::system::error_code ignored_error;
                boost::asio::streambuf b;
                std::ostream os(&b);
                features->SerializeToOstream(&os);
                size_t len = socket.send_to(b.data(),
                                            remote_endpoint, 0, ignored_error);
                std::cout << "Sending Matrix of length (" << len << ")" << std::endl;
                return true;
            }
            
        }
        catch (std::exception &e)
        {
            std::cerr << e.what() << std::endl;
            connected = false;
            return false;
        }
        std::cout << "No client connected\n";
        return false;
    }

private:
    bool connected = false;
    udp::socket socket;
    boost::array<char, 1> recv_buf;
    udp::endpoint remote_endpoint;
};
/*-------------------------Coordinate-----------------------------*/
void encode_matrix(mediapipe::Matrix *matrices, cv::Mat *mat)
{
    for (int i = 0; i < mat->rows; i++)
    {
        for (int j = 0; j < mat->cols; j++)
        {
            matrices->add_value(mat->at<double>(i, j));
        }
    }
    matrices->set_valid(true);
    matrices->set_row(mat->rows);
    matrices->set_col(mat->cols);
}

void write_coordinate(mediapipe::Coordinates *coord, cv::Mat *mat)
{
    mediapipe::Matrix *matrices = coord->add_transforms();
    {
        encode_matrix(matrices, mat);
    }
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
            ret.at<double>(i,j) = mat.value(count++);
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
        std::cout << "Matrix "<<  i + 1 << "\n" << ret << "\n" << std::endl;
    }
}

void populate(mediapipe::Coordinates *coord)
{
    for (int i = 0; i < 10; i++)
    {
        cv::Mat mat = cv::Mat::zeros(3, 1, CV_64F);
        {
            for (int j = 0; j < mat.rows; j++)
                for (int k = 0; k < mat.cols; k++)
                    mat.at<double>(j, k) = rand() % 20 + 10.204;
        }
        write_coordinate(coord, &mat);
    }
}
/*-----------------------------------------------------------------*/

void print_a()
{
    boost::asio::io_service io;
  boost::asio::deadline_timer t(io, boost::posix_time::seconds(5));
  t.wait();
  std::cout << "Hello, world!" << std::endl;
}

::mediapipe::Status run_multiple(int cam, server * srv){
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  LOG(INFO) << "BOOST TEST:";
  //test();
  mediapipe::Coordinates features;
  populate(&features);
  LOG(INFO) << "Initialize the camera or load the video.";
  
  cv::VideoCapture capture;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video) {
    capture.open(FLAGS_input_video_path);
  } else {
    capture.open(cam);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !FLAGS_output_video_path.empty();
  if (!save_video) {
    //cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    srv->send_msg(&features);
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) break;  // End of video.
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Prepare and add graph input packet.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph,
                                   &gpu_helper]() -> ::mediapipe::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us))));
          return ::mediapipe::OkStatus();
        }));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;
    std::unique_ptr<mediapipe::ImageFrame> output_frame;

	std::unique_ptr<cv::Mat> val;
    // Convert GpuBuffer to ImageFrame.
    MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
        [&packet, &output_frame, &gpu_helper]() -> ::mediapipe::Status {
          auto& gpu_frame = packet.Get<mediapipe::GpuBuffer>();
          auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
          output_frame = absl::make_unique<mediapipe::ImageFrame>(
              mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
              gpu_frame.width(), gpu_frame.height(),
              mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
          gpu_helper.BindFramebuffer(texture);
          const auto info =
              mediapipe::GlTextureInfoForGpuBufferFormat(gpu_frame.format(), 0);
          glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                       info.gl_type, output_frame->MutablePixelData());
          glFlush();
          texture.Release();
          return ::mediapipe::OkStatus();
        }));

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
	string window_name = cam+") My Camera Feed ";
	string thresh_window_name = cam+") Thresh Window ";
	cv::namedWindow(window_name);
    if (save_video) {
      if (!writer.isOpened()) {
        LOG(INFO) << "Prepare video writer.";
        writer.open(FLAGS_output_video_path,
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
	  cv::imshow(window_name, output_frame_mat);
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

void listen_task(server * srv){
    srv->listen();
}

int main(int argc, char** argv) {
  try{

  boost::asio::io_service io_service;
  udp::endpoint endpoint = udp::endpoint(udp::v4(), UDP_PORT);
  server srv(io_service, endpoint);

  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::thread fun_2(listen_task,&srv);
  std::thread fun_1(run_multiple,0, &srv);
  //thread fun_2(run_multiple,2);
  fun_1.join();
  fun_2.join();
  /*::mediapipe::Status run_status = run_multiple(1,"MediaPipe");
  ::mediapipe::Status run_status_1 = run_multiple(0, "MediaPipe 2");//, "MediaPipe_1", "output_video_1");
  //RunMPPGraph();
  if (!run_status.ok() || !run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }*/
  }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

  return EXIT_SUCCESS;
}
