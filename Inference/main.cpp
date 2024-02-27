#include <iostream>
#include <opencv2/opencv.hpp>
#include "test_1.h"
#include "ll_enhancement_algorithm.h"

using namespace cv;


int main() {
//    std::cout << "Test show image with OPENCV LIB" << std::endl;
//    return test_1(); # The test for showing the image with OPENCV in UBUNTU

    std::cout << "============= Inference in CPU with C =============";

    // Silent cv
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    //initialMem();
    //cvReadImg();
    //return 0;
    std::string test_img_path = "/home/trongan93/Projects/LowLightEnhanceOnBoard/QuantizedZeroDCE_CPP/paper_case2/_mnt_d_Seaport_satellite_images_lng_4.240000_lat_51.520000_sentinel_2_rgb.png";
    std::string write_img_test_path = "/home/trongan93/Projects/LowLightEnhanceOnBoard/QuantizedZeroDCE_CPP/paper_case2/_mnt_d_Seaport_satellite_images_lng_4.240000_lat_51.520000_sentinel_2_rgb_enhanced_onboard.png";

    clock_t t;
    double elapse;

    printf("Start initialMem\n");
    t = clock();
    initialMem();
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Start cvReadImg\n");
    t = clock();
    cvReadImg(test_img_path);
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Start qLoadParam\n");
    t = clock();
    qLoadParam();
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Start qNorm_256\n");
    t = clock();
    qNorm_256();
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Start qDownSample\n");
    t = clock();
    qDownSample();
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Start qConv1st\n");
    t = clock();
    qConv1st();
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Start qConv2nd\n");
    t = clock();
    qConv2nd();
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Start qConv3rdV2\n");
    t = clock();
    qConv3rdV2();
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Start qUpSample\n");
    t = clock();
    qUpSample();
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Start qEnhance_256\n");
    t = clock();
    qEnhance_256();
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Start cvOutputImg\n");
    t = clock();
    cvOutputImg();
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Write to file writeImageData\n");
    t = clock();
    writeImageData(write_img_test_path);
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    printf("Start clenaMem\n");
    t = clock();
    cleanMem();
    t = clock() - t;
    elapse = ((double)t) / CLOCKS_PER_SEC;
    printf("%f s\n", elapse);

    return 0;
}