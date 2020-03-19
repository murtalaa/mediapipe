// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.
#include <cstdlib>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/coordinate.pb.h" //Coordinate
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
#include <future>
#include <iostream>
#include <chrono>
#include <boost/asio.hpp>
#include <boost/array.hpp>

#define UDP_PORT 7777

using boost::asio::ip::udp;
std::vector<std::vector<cv::Point2f>> landMarks[2];

//Take stream from /mediapipe/graphs/hand_tracking/hand_detection_desktop_live.pbtxt
// RendererSubgraph - LANDMARKS:hand_landmarks
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

// input and output streams to be used/retrieved by calculators
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "multi_hand_landmarks";
constexpr char kWindowName[] = "MediaPipe";

DEFINE_string(
		calculator_graph_config_file, "",
		"Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
		"Full path of video to load. "
		"If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
		"Full path of where to save result (.mp4 only). "
		"If not provided, show result in a window.");

/*-------------------------------Server-----------------------------------*/
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

    void listen()
    {
        std::cout << "Waiting for Connection\n";
        do
        {
            std::cout << "Remote Endpoint (Before): " << remote_endpoint.port() << "\n";
            recv_msg();
            std::cout << "Remote Endpoint (After): " << remote_endpoint.port() << "\n";
        } while (remote_endpoint.port() == 0);
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
            if (connected)
            {
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

void listen_task(server *srv)
{
    srv->listen();
}
/*----------------------------------------------------------------------*/

/*------------------------------Coordinate--------------------------------*/
void encode_matrix(mediapipe::Matrix * matrices, cv::Mat mat)
{
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            matrices->add_value(mat.at<double>(i, j));
        }
    }
    matrices->set_valid(true);
    matrices->set_row(mat.rows);
    matrices->set_col(mat.cols);
}

void write_coordinate(mediapipe::Coordinates * coord, cv::Mat mat)
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
        write_coordinate(coord, mat);
    }
}

void populate_v2(mediapipe::Coordinates *coord, 
                 ::std::vector<cv::Mat> system)
{
    for (int i = 0; i < system.size(); i++)
    {
        write_coordinate(coord, system[i]);
    }
}
/*-----------------------------------------------------------------*/

/*-----------------------------Generate Transform-------------------------*/
void genCoordinate_v2(int ref, int start, std::vector<cv::Mat> * system, ::std::vector < cv::Point3d > handCoord3d)
{
    std::ostringstream stringStream;
    cv::Point3d pt_ref, pt_start, pt_next;

    pt_ref = handCoord3d[ref];
    pt_start = handCoord3d[start];
    pt_next = handCoord3d[start+1];

    cv::Point3d y = pt_start - pt_ref;
    cv::Point3d y_ = pt_next - pt_start;

    y = y / norm(y);
    y_ = y_ / norm(y_);
    cv::Point3d z = y.cross(y_);
    z = z / norm(z);
    cv::Point3d x = y.cross(z);
    x = x / norm(x);
    cv::Point3d x_ = y_.cross(z);
    x_ = x_ / norm(x_);

    cv::Mat coord  = cv::Mat::zeros(4, 4, CV_64F);
    cv::Mat coord_ = cv::Mat::zeros(4, 4, CV_64F);
    {
        coord.at<double>(0, 0) = x.x; coord_.at<double>(0, 0) = x_.x;
        coord.at<double>(1, 0) = x.y; coord_.at<double>(1, 0) = x_.y;
        coord.at<double>(2, 0) = x.z; coord_.at<double>(2, 0) = x_.z;

        coord.at<double>(0, 1) = y.x; coord_.at<double>(0, 1) = y_.x;
        coord.at<double>(1, 1) = y.y; coord_.at<double>(1, 1) = y_.y;
        coord.at<double>(2, 1) = y.z; coord_.at<double>(2, 1) = y_.z;

        coord.at<double>(0, 2) = z.x; coord_.at<double>(0, 2) = z.x;
        coord.at<double>(1, 2) = z.y; coord_.at<double>(1, 2) = z.y;
        coord.at<double>(2, 2) = z.z; coord_.at<double>(2, 2) = z.z;
    }

    cv::Point3d trans_ref = handCoord3d[ref] - handCoord3d[0];
    cv::Point3d trans_start = handCoord3d[start] - handCoord3d[0];
    {
        coord.at<double>(0, 3) = trans_ref.x; coord_.at<double>(0, 3) = trans_start.x;
        coord.at<double>(1, 3) = trans_ref.y; coord_.at<double>(1, 3) = trans_start.y;
        coord.at<double>(2, 3) = trans_ref.z; coord_.at<double>(2, 3) = trans_start.z;
        coord.at<double>(3, 3) = 1.0; coord_.at<double>(3, 3) = 1.0;
    }

    system->push_back(coord);
    system->push_back(coord_);

    if ((start + 1) % 4 == 0)
    {
        cv::Mat coord_end = (coord_.clone());
        cv::Point3d trans_next = handCoord3d[start + 1] - handCoord3d[0];
        {
            coord_end.at<double>(0, 3) = trans_next.x;
            coord_end.at<double>(1, 3) = trans_next.y;
            coord_end.at<double>(2, 3) = trans_next.z;
        }
        system->push_back(coord_end); // Tip of the finger
    }
}


void genCoordSys_v2(::std::vector<cv::Point3d> handCoord3d, 
                    ::std::vector<cv::Mat> *system)
{
    /*
Landmark Points range from 0 to 20
*/
    int ref = 0;
    int start = 1;
    while (start < 20)
    {
        genCoordinate_v2(ref, start, system, handCoord3d);
        if ((start + 1) % 4 == 0)
        {
            ref = 0;
        }
        else
        {
            ref += start + 1;
        }
        start = start + 2;
    }
    system->erase(system->begin() + 00);
    system->erase(system->begin() + 04);
    system->erase(system->begin() + 13);
    system->erase(system->begin() + 17);
}
/*------------------------------------------------------------------------*/

/*--------------------------------Triangulate-----------------------------*/
/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Vec3d LinearLSTriangulation(cv::Point3d u,  //homogenous image point (u,v,1)
                                cv::Mat P,      //camera 1 projection matrix
                                cv::Point3d u1, //homogenous image point in 2nd camera
                                cv::Mat P1      //camera 2 projection matrix
)
{
    double data1[12] = {
        u.x * P.at<double>(2, 0) - P.at<double>(0, 0),
        u.x * P.at<double>(2, 1) - P.at<double>(0, 1),
        u.x * P.at<double>(2, 2) - P.at<double>(0, 2),
        u.y * P.at<double>(2, 0) - P.at<double>(1, 0),
        u.y * P.at<double>(2, 1) - P.at<double>(1, 1),
        u.y * P.at<double>(2, 2) - P.at<double>(1, 2),
        u1.x * P1.at<double>(2, 0) - P1.at<double>(0, 0),
        u1.x * P1.at<double>(2, 1) - P1.at<double>(0, 1),
        u1.x * P1.at<double>(2, 2) - P1.at<double>(0, 2),
        u1.y * P1.at<double>(2, 0) - P1.at<double>(1, 0),
        u1.y * P1.at<double>(2, 1) - P1.at<double>(1, 1),
        u1.y * P1.at<double>(2, 2) - P1.at<double>(1, 2)};
    cv::Mat A = cv::Mat(4, 3, CV_64FC1, &data1);
    double data2[4] = {
        -(u.x * P.at<double>(2, 3) - P.at<double>(0, 3)),
        -(u.y * P.at<double>(2, 3) - P.at<double>(1, 3)),
        -(u1.x * P1.at<double>(2, 3) - P1.at<double>(0, 3)),
        -(u1.y * P1.at<double>(2, 3) - P1.at<double>(1, 3))};
    cv::Mat B = cv::Mat(4, 1, CV_64FC1, &data2);
    cv::Vec3d X;
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}

void triangulate(server *srv, 
                 ::std::string calib_file, 
                 ::std::vector<cv::Point3d> v1_pts,
                 ::std::vector<cv::Point3d> v2_pts)
{
    if (v1_pts.size() != v2_pts.size() || v1_pts.size() == 0)
        return;
    cv::FileStorage fs(calib_file, cv::FileStorage::READ);
    cv::Mat P1, P2;
    std::vector<cv::Mat> system;
    fs["P1"] >> P1;
    fs["P2"] >> P2;
    ::std::vector<cv::Point3d> handCoord3d;
    for (int j = 0; j < v1_pts.size(); j++)
    {
        cv::Vec3d Coords3D = LinearLSTriangulation(v1_pts[j], P1, v2_pts[j],
                                                   P2);
        handCoord3d.push_back(Coords3D);
        std::cout << "LM " << j << ": " << Coords3D << std::endl;
    }
    mediapipe::Coordinates features;
    genCoordSys_v2(handCoord3d, &system);
    cv::Mat ref = cv::Mat(system[8]);
    system.erase(system.begin()+8);
    system.insert(system.begin(), ref);
    populate_v2(&features, system);
    srv->send_msg(&features);
}
/*------------------------------------------------------------------------*/

::mediapipe::Status RunMPPGraph(server *srv, 
								std::promise<::mediapipe::Status> && pr, 
								std::promise<::mediapipe::Status> && pl) 
{
    std::this_thread::sleep_for(std::chrono::seconds(1));
    char filename[20];
	printf("Enter stereo calibration filename: ");
	scanf("%19s", filename);

	std::string calculator_graph_config_contents;
	MP_RETURN_IF_ERROR(
			mediapipe::file::GetContents(FLAGS_calculator_graph_config_file,
					&calculator_graph_config_contents));
	LOG(INFO) << "Get calculator graph config contents: "
			<< calculator_graph_config_contents;
	mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie
			< mediapipe::CalculatorGraphConfig
			> (calculator_graph_config_contents);

	LOG(INFO) << "Initialize the calculator graph.";
	mediapipe::CalculatorGraph rgraph;
	MP_RETURN_IF_ERROR(rgraph.Initialize(config));

	mediapipe::CalculatorGraph lgraph;
	MP_RETURN_IF_ERROR(lgraph.Initialize(config));

	LOG(INFO) << "Initialize the GPU.";
	ASSIGN_OR_RETURN(auto rgpu_resources, mediapipe::GpuResources::Create());
	MP_RETURN_IF_ERROR(rgraph.SetGpuResources(std::move(rgpu_resources)));
	mediapipe::GlCalculatorHelper rgpu_helper;
	rgpu_helper.InitializeForTest(rgraph.GetGpuResources().get());

	ASSIGN_OR_RETURN(auto lgpu_resources, mediapipe::GpuResources::Create());
	MP_RETURN_IF_ERROR(lgraph.SetGpuResources(std::move(lgpu_resources)));
	mediapipe::GlCalculatorHelper lgpu_helper;
	lgpu_helper.InitializeForTest(lgraph.GetGpuResources().get());

	LOG(INFO) << "Initialize the camera or load the video.";
	cv::VideoCapture capture;
	const bool load_video = !FLAGS_input_video_path.empty();
	if (load_video) {
		capture.open(FLAGS_input_video_path);
	} else {
		capture.open(0);
	}
	RET_CHECK(capture.isOpened());

	cv::VideoWriter writer;
	std::ostringstream lstringStream, rstringStream;

	lstringStream << kWindowName << " left";
	rstringStream << kWindowName << " right";

	cv::String left = lstringStream.str();
	cv::String right = rstringStream.str();

	const bool save_video = !FLAGS_output_video_path.empty();
	if (!save_video) {
		cv::namedWindow(left, /*flags=WINDOW_AUTOSIZE*/1);
		cv::namedWindow(right, /*flags=WINDOW_AUTOSIZE*/1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
	}

	LOG(INFO) << "Start running the calculator graph.";
	ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller rpoller,
			rgraph.AddOutputStreamPoller(kOutputStream));

	ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller lpoller,
			lgraph.AddOutputStreamPoller(kOutputStream));

	// hand landmarks stream
	ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller lpoller_landmark,
			rgraph.AddOutputStreamPoller(kLandmarksStream));

	ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller rpoller_landmark,
			lgraph.AddOutputStreamPoller(kLandmarksStream));

	MP_RETURN_IF_ERROR(rgraph.StartRun( { }));
	MP_RETURN_IF_ERROR(lgraph.StartRun( { }));

	LOG(INFO) << "Start grabbing and processing frames.";
	bool grab_frames = true;
	while (grab_frames) {
		// Capture opencv camera or video frame.
		cv::Mat camera_frame_raw;
		capture >> camera_frame_raw;
		if (camera_frame_raw.empty())
			break;  // End of video.
		cv::Mat camera_frame, rcamera_frame, lcamera_frame;
		cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

		rcamera_frame = camera_frame(cv::Rect(0, 0, 640, 480));
		lcamera_frame = camera_frame(cv::Rect(640, 0, 640, 480));

		if (!load_video) {
			cv::flip(rcamera_frame, rcamera_frame, /*flipcode=HORIZONTAL*/1);
			cv::flip(lcamera_frame, lcamera_frame, /*flipcode=HORIZONTAL*/1);
		}

		//std::cout << "Image Width: "  << camera_frame.cols  << std::endl;
		//std::cout << "Image Height: " << camera_frame.rows << std::endl;

		// Wrap Mat into an ImageFrame.
		auto rinput_frame =
				absl::make_unique < mediapipe::ImageFrame
						> (mediapipe::ImageFormat::SRGB, rcamera_frame.cols, rcamera_frame.rows, mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
		cv::Mat rinput_frame_mat = mediapipe::formats::MatView(
				rinput_frame.get());
		rcamera_frame.copyTo(rinput_frame_mat);

		auto linput_frame =
				absl::make_unique < mediapipe::ImageFrame
						> (mediapipe::ImageFormat::SRGB, lcamera_frame.cols, lcamera_frame.rows, mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
		cv::Mat linput_frame_mat = mediapipe::formats::MatView(
				linput_frame.get());
		lcamera_frame.copyTo(linput_frame_mat);

		// Prepare and add graph input packet.
		size_t frame_timestamp_us = (double) cv::getTickCount()
				/ (double) cv::getTickFrequency() * 1e6;
		MP_RETURN_IF_ERROR(
				rgpu_helper.RunInGlContext(
						[&rinput_frame, &frame_timestamp_us, &rgraph,
								&rgpu_helper]() -> ::mediapipe::Status {
							// Convert ImageFrame to GpuBuffer.
							auto rtexture = rgpu_helper.CreateSourceTexture(
									*rinput_frame.get());
							auto rgpu_frame = rtexture.GetFrame<
									mediapipe::GpuBuffer>();
							glFlush();
							rtexture.Release();
							// Send GPU image packet into the graph.
							MP_RETURN_IF_ERROR(
									rgraph.AddPacketToInputStream(kInputStream,
											mediapipe::Adopt(
													rgpu_frame.release()).At(
													mediapipe::Timestamp(
															frame_timestamp_us))));
							return ::mediapipe::OkStatus();
						}));

		frame_timestamp_us = (double) cv::getTickCount()
				/ (double) cv::getTickFrequency() * 1e6;
		MP_RETURN_IF_ERROR(
				lgpu_helper.RunInGlContext(
						[&linput_frame, &frame_timestamp_us, &lgraph,
								&lgpu_helper]() -> ::mediapipe::Status {
							// Convert ImageFrame to GpuBuffer.
							auto ltexture = lgpu_helper.CreateSourceTexture(
									*linput_frame.get());
							auto lgpu_frame = ltexture.GetFrame<
									mediapipe::GpuBuffer>();
							glFlush();
							ltexture.Release();
							// Send GPU image packet into the graph.
							MP_RETURN_IF_ERROR(
									lgraph.AddPacketToInputStream(kInputStream,
											mediapipe::Adopt(
													lgpu_frame.release()).At(
													mediapipe::Timestamp(
															frame_timestamp_us))));
							return ::mediapipe::OkStatus();
						}));

		// Get the graph result packet, or stop if that fails.
		mediapipe::Packet rpacket, lpacket;
		mediapipe::Packet rlandmark_packet, llandmark_packet;

		if (!rpoller.Next(&rpacket))
			break;
		if (!rpoller_landmark.Next(&rlandmark_packet))
			break;

		if (!lpoller.Next(&lpacket))
			break;
		if (!lpoller_landmark.Next(&llandmark_packet))
			break;

		std::unique_ptr<mediapipe::ImageFrame> routput_frame, loutput_frame;

		auto &nroutput_landmarks = rlandmark_packet.Get<
				std::vector<mediapipe::NormalizedLandmarkList>>();
		auto &nloutput_landmarks = llandmark_packet.Get<
				std::vector<mediapipe::NormalizedLandmarkList>>();

		// Convert GpuBuffer to ImageFrame.
		MP_RETURN_IF_ERROR(
				rgpu_helper.RunInGlContext(
						[&rpacket, &routput_frame, &rgpu_helper]() -> ::mediapipe::Status {
							auto &rgpu_frame =
									rpacket.Get<mediapipe::GpuBuffer>();
							auto rtexture = rgpu_helper.CreateSourceTexture(
									rgpu_frame);
							routput_frame =
									absl::make_unique < mediapipe::ImageFrame
											> (mediapipe::ImageFormatForGpuBufferFormat(
													rgpu_frame.format()), rgpu_frame.width(), rgpu_frame.height(), mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
							rgpu_helper.BindFramebuffer(rtexture);
							const auto info =
									mediapipe::GlTextureInfoForGpuBufferFormat(
											rgpu_frame.format(), 0);
							glReadPixels(0, 0, rtexture.width(),
									rtexture.height(), info.gl_format,
									info.gl_type,
									routput_frame->MutablePixelData());
							glFlush();
							rtexture.Release();
							return ::mediapipe::OkStatus();
						}));

		// Convert GpuBuffer to ImageFrame.
		MP_RETURN_IF_ERROR(
				lgpu_helper.RunInGlContext(
						[&lpacket, &loutput_frame, &lgpu_helper]() -> ::mediapipe::Status {
							auto &lgpu_frame =
									lpacket.Get<mediapipe::GpuBuffer>();
							auto ltexture = lgpu_helper.CreateSourceTexture(
									lgpu_frame);
							loutput_frame =
									absl::make_unique < mediapipe::ImageFrame
											> (mediapipe::ImageFormatForGpuBufferFormat(
													lgpu_frame.format()), lgpu_frame.width(), lgpu_frame.height(), mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
							lgpu_helper.BindFramebuffer(ltexture);
							const auto info =
									mediapipe::GlTextureInfoForGpuBufferFormat(
											lgpu_frame.format(), 0);
							glReadPixels(0, 0, ltexture.width(),
									ltexture.height(), info.gl_format,
									info.gl_type,
									loutput_frame->MutablePixelData());
							glFlush();
							ltexture.Release();
							return ::mediapipe::OkStatus();
						}));

		// Convert back to opencv for display or saving.
		cv::Mat routput_frame_mat = mediapipe::formats::MatView(
				routput_frame.get());
		cv::cvtColor(routput_frame_mat, routput_frame_mat, cv::COLOR_RGB2BGR);

		cv::Mat loutput_frame_mat = mediapipe::formats::MatView(
				loutput_frame.get());
		cv::cvtColor(loutput_frame_mat, loutput_frame_mat, cv::COLOR_RGB2BGR);

		if (save_video) {
			if (!writer.isOpened()) {
				LOG(INFO) << "Prepare video writer.";
				writer.open(FLAGS_output_video_path,
						mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
						capture.get(cv::CAP_PROP_FPS),
						routput_frame_mat.size());
				RET_CHECK(writer.isOpened());
			}
			writer.write(routput_frame_mat);
		} else {

			//cv::imshow(right, routput_frame_mat);
			//cv::imshow(left, loutput_frame_mat);

			// Press any key to exit.
			const int pressed_key = cv::waitKey(5);
			if (pressed_key >= 0 && pressed_key != 255)
				grab_frames = false;

			int j = 0;
			::std::vector < std::vector < cv::Point2f >> all_pts(2);
			::std::vector<cv::Point3d> v1_pts, v2_pts;
			//RIGHT
			for (const auto &loutput_landmarks : nloutput_landmarks) {
				j++;
				LOG(INFO) << "Camera 1: " << "Palm: " << j;
				for (int i = 0; i < loutput_landmarks.landmark_size(); ++i) {
					const mediapipe::NormalizedLandmark &landmark =
							loutput_landmarks.landmark(i);
					cv::Point3d pt(landmark.x() * camera_frame.cols / 2,
							landmark.y() * camera_frame.rows, 1);
					v1_pts.push_back(pt);
					//LOG(INFO) << "x: " << landmark.x() * camera_frame.cols
					//		<< " y: " << landmark.y() * camera_frame.rows
					//		<< " z: " << landmark.z();
					//int x = landmark.x() * camera_frame.cols / 2;
					//int y = landmark.y() * camera_frame.rows;
					//cv::circle(routput_frame_mat, cv::Point(x, y), 5,
					//		cv::Scalar(0, 255, 255), 3);
				}
			}

			//LEFT
			j = 0;
			for (const auto &routput_landmarks : nroutput_landmarks) {
				j++;
				LOG(INFO) << "Camera 2: " << "Palm: " << j;
				for (int i = 0; i < routput_landmarks.landmark_size(); ++i) {
					const mediapipe::NormalizedLandmark &landmark =
							routput_landmarks.landmark(i);
					cv::Point3d pt(landmark.x() * camera_frame.cols / 2,
							landmark.y() * camera_frame.rows, 1);
					v2_pts.push_back(pt);
					//LOG(INFO) << "Right: " << i << ": " << "x: "
					//		<< landmark.x() * camera_frame.cols << " y: "
					//		<< landmark.y() * camera_frame.rows << " z: "
					//		<< landmark.z();
					//int x = landmark.x() * camera_frame.cols / 2;
					//int y = landmark.y() * camera_frame.rows;
					//cv::circle(loutput_frame_mat, cv::Point(x, y), 5,
					//		cv::Scalar(0, 255, 255), 3);
				}
			}

			cv::imshow(right, routput_frame_mat);
			cv::imshow(left, loutput_frame_mat);

			triangulate(srv, filename, v1_pts, v2_pts);
		}
	}

    LOG(INFO) << "Shutting down.";
    if (writer.isOpened())
        writer.release();
    MP_RETURN_IF_ERROR(rgraph.CloseInputStream(kInputStream));
    MP_RETURN_IF_ERROR(lgraph.CloseInputStream(kInputStream));

    ::mediapipe::Status r = rgraph.WaitUntilDone();
    ::mediapipe::Status l = lgraph.WaitUntilDone();
	pr.set_value(r);
	pl.set_value(l);
	return lgraph.WaitUntilDone();
}

int main(int argc, char **argv)
{
    try
    {
        boost::asio::io_service io_service;
        udp::endpoint endpoint = udp::endpoint(udp::v4(), UDP_PORT);
        server srv(io_service, endpoint);
        google::InitGoogleLogging(argv[0]);
        gflags::ParseCommandLineFlags(&argc, &argv, true);
		std::promise<::mediapipe::Status> l, r;
		auto f = r.get_future();
		auto g = l.get_future();
        std::thread fun_2(listen_task, &srv);
        std::thread fun_1(RunMPPGraph, &srv, std::move(r), std::move(l));
        fun_2.join();       
		std::cout << "Listen Task Over \n";      		
		fun_1.join();
		std::cout << "Graph Task Over \n";
        ::mediapipe::Status rrun_status = f.get();
        ::mediapipe::Status lrun_status = g.get();

		std::cout << "Thread Output Recieved\n";
        bool s = rrun_status.ok() && lrun_status.ok();
        if (!s)
        {
		std::cout << "EXITING\n";
            LOG(ERROR) << "Failed to run the graph: \n";
            return EXIT_FAILURE;
        }
        else
        {
					std::cout << "SUCCESS\n";
            LOG(INFO) << "Success!";
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

    return EXIT_SUCCESS;
}

/*
int main(int argc, char **argv)
{
    try
    {
        boost::asio::io_service io_service;
        udp::endpoint endpoint = udp::endpoint(udp::v4(), UDP_PORT);
        server srv(io_service, endpoint);
        google::InitGoogleLogging(argv[0]);
        gflags::ParseCommandLineFlags(&argc, &argv, true);
        auto t1 = std::async(listen_task, &srv);
        auto t2 = std::async(RunMPPGraph, &srv);
        ::mediapipe::Status run_status = t2.get();
        bool s = run_status.ok();
        if (!s)
        {
            LOG(ERROR) << "Failed to run the graph: " << run_status.message();
            return EXIT_FAILURE;
        }
        else
        {
            LOG(INFO) << "Success!";
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

    return EXIT_SUCCESS;
}


int main(int argc, char **argv) {
	google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	::mediapipe::Status run_status = RunMPPGraph(640, 0);
	bool s = run_status.ok();
	if (!s) {
		LOG(ERROR) << "Failed to run the graph: " << run_status.message();
		return EXIT_FAILURE;
	} else {
		LOG(INFO) << "Success!";
	}
	return EXIT_SUCCESS;
}
*/
