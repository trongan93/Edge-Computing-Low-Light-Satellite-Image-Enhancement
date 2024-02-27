//
// Created by Andrew Bui on 4/17/23.
//

#include <string>
#include "test_1.h"
#include <opencv2/opencv.hpp>

using namespace std;
int test_1() {
    string file_path = "/home/trongan93/Desktop/test_sentinel_2_rgb.png";
    cv::Mat image;
    image = imread( file_path, cv::IMREAD_COLOR );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    cv::waitKey(0);
    return 0;
}
