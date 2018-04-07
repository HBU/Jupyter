cmake_minimum_required(VERSION 2.8.12)

PROJECT(imageClassification)

#MAC Users Uncomment the line below
#SET(OpenCV_DIR /usr/local/Cellar/opencv3/3.2.0/share/OpenCV/)

#Linux users uncomment the line below
#SET(OpenCV_DIR /usr/local/share/OpenCV/)

find_package( OpenCV REQUIRED )

# ###  Windows users uncomment the 6 lines below  ###

#find_package( dlib REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )
#MACRO(add_example name)
#  ADD_EXECUTABLE(${name} ${name}.cpp)
#  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})
#ENDMACRO()



# ###  Linux and MacOS uncomment the 5 lines below ###
MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})
ENDMACRO()


add_example(train_digits)