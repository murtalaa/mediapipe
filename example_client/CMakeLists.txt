# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lboost_system -pthread")

set( Boost_USE_STATIC_LIBS        ON )
set( Boost_USE_MULTITHREADED      ON )

# Define project name
project(client_output)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

find_package(Boost
             1.55.0
             COMPONENTS thread
                        system
                        log
                        log_setup
                        program_options
             REQUIRED)

if(Boost_FOUND)
    message(STATUS "Success!")
endif()
include_directories( ${Boost_INCLUDE_DIR} )
include_directories(${CMAKE_CURRENT_BINARY_DIR})
LINK_DIRECTORIES( ${Boost_LIBRARY_DIR} )
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS coordinate.proto)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them

message(STATUS "OpenCV library status:")
message(STATUS "        config: ${OpenCV_DIR}")
message(STATUS "        version: ${OpenCV_VERSION}")
message(STATUS "        include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "        libraries: ${OpenCV_LIBS}")

message(STATUS "Boost library status:")
message(STATUS "        config: ${Boost_DIR}")
message(STATUS "        version: ${Boost_VERSION}")
message(STATUS "        include path: ${Boost_INCLUDE_DIR}")
message(STATUS "        libraries: ${Boost_LIB}")

# Declare the executable target built from your sources
add_executable(client_output main.cpp client.cpp ${PROTO_SRCS} ${PROTO_HDRS})

# Link your application with OpenCV libraries
target_link_libraries(client_output PRIVATE ${OpenCV_LIBS})
target_link_libraries(client_output PRIVATE ${Boost_LIBRARIES})
target_link_libraries(client_output PRIVATE ${PROTOBUF_LIBRARIES})
