//
// Created by trongan93 on 2/27/24.
//

#include "ll_enhancement_algorithm.h"
#include "memAlloc.h"
#include "q_Train_Model_Weight.h"

cmosRawData_t*      ISP_RAWDATA = NULL;
cmosTMData_t*       ISP_TMDATA = NULL;
cmosRGBData_t*      ISP_DBDATA = NULL;
cmosRGBData_t*      ISP_WBDATA = NULL;
cmosRGBData_t*      ISP_AIISPDATA = NULL;
qNormImg_t*         AIIP_NORM = NULL;
qNetIO_t*           AIIP_NETIO = NULL;
qNetFeature_t*      AIIP_FEATURE1 = NULL;
qNetFeature_t*      AIIP_FEATURE2 = NULL;
qEnhanceParam_t*    AIIP_PARAM = NULL;
qEnhanceParam_t*    AIIP_USBUFFER = NULL;
qWConv1st_t*        AIIP_CONVW01 = NULL;
qBConv1st_t*        AIIP_CONVB01 = NULL;
qWConv2nd_t*        AIIP_CONVW02 = NULL;
qBConv2nd_t*        AIIP_CONVB02 = NULL;
qWConv3rd_t*        AIIP_CONVW03 = NULL;
qBConv3rd_t*        AIIP_CONVB03 = NULL;

int quantized_conv = 14;
int quantized_norm = 6; // quantized convolution - 8
void cvOutputImg() {
    // TODO Output image

    cv::Mat image(IMGHEIGHT, IMGWIDTH, CV_8UC3);
    printf("Output image rows : %10u\timage cols : %10u\timage channels : %10u\n", image.rows, image.cols, image.channels());

    for (int c = 0; c < CMOS_IMGC; ++c)
        for (int y = 0; y < CMOS_IMGH; ++y)
            for (int x = 0; x < CMOS_IMGW; ++x) {
                image.at<cv::Vec3b>(y, x)[c] = ISP_AIISPDATA->data[y][x][c];
            }

    printf("Showed image rows : %10u\timage cols : %10u\timage channels : %10u\n", image.rows, image.cols, image.channels());
    cv::namedWindow("Enhanced_image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Enhanced_image", image);

    cv::waitKey(0);
}

//void cvOutputImg_ChenKai()
//{
//    // TODO Output image
//
//    cv::Mat image(1200, 1920, CV_8UC3);
//    printf("Output image rows : %10u\timage cols : %10u\timage channels : %10u\n", image.rows, image.cols, image.channels());
//
//    for (int c = 0; c < CMOS_IMGC; ++c)
//        for (int y = 0; y < CMOS_IMGH; ++y)
//            for (int x = 0; x < CMOS_IMGW; ++x) {
//                image.at<cv::Vec3b>(y, x)[c] = ISP_AIISPDATA->data[y][x][c];
//            }
//
//    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
//
//    cv::Mat org_image = cv::imread(test_img_path, cv::IMREAD_COLOR);
//    cv::resize(org_image, org_image, cv::Size(960, 600), cv::InterpolationFlags::INTER_AREA);
//    cv::imshow("lowLight_cpp_test1920x1200.jpg", org_image);
//
//    cv::Mat python_image = cv::imread("cpp_test1920x1200_python.png", cv::IMREAD_COLOR);
//    cv::resize(python_image, python_image, cv::Size(960, 600), cv::InterpolationFlags::INTER_AREA);
//    cv::imshow("Enhanced_cpp_test1920x1200_python.png", python_image);
//    //cv::resize(python_image, python_image, cv::Size(1920, 1200), cv::InterpolationFlags::INTER_AREA);
//
//
//    cv::resize(image, image, cv::Size(960, 600), cv::InterpolationFlags::INTER_AREA);
//
//    cv::Mat diffMask(600, 960, CV_8UC3);
//    //L1 Distance
//    int dist = 0;
//    for (int c = 0; c < image.channels(); ++c)
//        for (int y = 0; y < image.rows; ++y)
//            for (int x = 0; x < image.cols; ++x) {
//                int diff = std::abs(image.at<cv::Vec3b>(y, x)[c] - python_image.at<cv::Vec3b>(y, x)[c]);
//                if (diff > 4) {
//                    //printf("image: %d, python image : %d, diff : %d\n", image.at<cv::Vec3b>(y, x)[c], python_image.at<cv::Vec3b>(y, x)[c], diff);
//                    diffMask.at<cv::Vec3b>(y, x)[c] = 255;
//                }
//                else {
//                    diffMask.at<cv::Vec3b>(y, x)[c] = 0;
//                }
//                dist += diff;
//            }
//
//    cv::namedWindow("precisionDiffMask", cv::WINDOW_AUTOSIZE);
//    cv::imshow("precisionDiffMask", diffMask);
//
//    printf("Showed image rows : %10u\timage cols : %10u\timage channels : %10u\n", image.rows, image.cols, image.channels());
//    cv::namedWindow("Enhanced_CPP_output", cv::WINDOW_AUTOSIZE);
//    cv::imshow("Enhanced_CPP_output", image);
//
//    printf("L1 distance(absolute distance) : %d with image size [%d, %d, %d]\n", dist, python_image.channels(), python_image.rows, python_image.cols);
//
//
//    printf("This distance has two factors.\n1. The quantization precision loss.\n2. The resize difference(Upscaling)\n");
//
//    // TODO Show image
//    cv::waitKey(0);
//}


void initialMem() {

    ISP_RAWDATA = (cmosRawData_t*)malloc(sizeof(cmosRawData_t));
    ISP_TMDATA = (cmosTMData_t*)malloc(sizeof(cmosTMData_t));
    ISP_DBDATA = (cmosRGBData_t*)malloc(sizeof(cmosRGBData_t));
    ISP_WBDATA = (cmosRGBData_t*)malloc(sizeof(cmosRGBData_t));
    ISP_AIISPDATA = (cmosRGBData_t*)malloc(sizeof(cmosRGBData_t));
    AIIP_NORM = (qNormImg_t*)malloc(sizeof(qNormImg_t));
    AIIP_NETIO = (qNetIO_t*)malloc(sizeof(qNetIO_t));
    AIIP_FEATURE1 = (qNetFeature_t*)malloc(sizeof(qNetFeature_t));
    AIIP_FEATURE2 = (qNetFeature_t*)malloc(sizeof(qNetFeature_t));
    AIIP_PARAM = (qEnhanceParam_t*)malloc(sizeof(qEnhanceParam_t));
    AIIP_USBUFFER = (qEnhanceParam_t*)malloc(sizeof(qEnhanceParam_t));
    AIIP_CONVW01 = (qWConv1st_t*)malloc(sizeof(qWConv1st_t));
    AIIP_CONVB01 = (qBConv1st_t*)malloc(sizeof(qBConv1st_t));
    AIIP_CONVW02 = (qWConv2nd_t*)malloc(sizeof(qWConv2nd_t));
    AIIP_CONVB02 = (qBConv2nd_t*)malloc(sizeof(qBConv2nd_t));
    AIIP_CONVW03 = (qWConv3rd_t*)malloc(sizeof(qWConv3rd_t));
    AIIP_CONVB03 = (qBConv3rd_t*)malloc(sizeof(qBConv3rd_t));
}

void cleanMem() {

    free((void*)ISP_RAWDATA);
    free((void*)ISP_TMDATA);
    free((void*)ISP_DBDATA);
    free((void*)ISP_WBDATA);
    free((void*)ISP_AIISPDATA);
    free((void*)AIIP_NORM);
    free((void*)AIIP_NETIO);
    free((void*)AIIP_FEATURE1);
    free((void*)AIIP_FEATURE2);
    free((void*)AIIP_PARAM);
    free((void*)AIIP_USBUFFER);
    free((void*)AIIP_CONVW01);
    free((void*)AIIP_CONVB01);
    free((void*)AIIP_CONVW02);
    free((void*)AIIP_CONVB02);
    free((void*)AIIP_CONVW03);
    free((void*)AIIP_CONVB03);
}

/*
* Contain some error need to estimated
* Consider comparing the output result from pytorch
*/
int sigmoidMapping(int x)
{
    if(x >= -QX && x <= QX)
        return (x*7571)>>18;
    else if((x >= -2*QX && x < -QX) || (x <= 2*QX && x > QX))
        return x>0?((x*4890+44166158)>>18):((x*4890-44166158)>>18);
    else if((x >= -3*QX && x < -2*QX) || (x <= 3*QX && x > 2*QX))
        return x>0?((x*2340+127926272)>>18):((x*2340-127926272)>>18);
    else if((x >= -4*QX && x < -3*QX) || (x <= 4*QX && x > 3*QX))
        return x>0?((x*960+195870720)>>18):((x*960-195870720)>>18);
    else if((x >= -5*QX && x < -4*QX) || (x <= 5*QX && x > 4*QX))
        return x>0?((x*368+234700800)>>18):((x*368-234700800)>>18);
    else
        return (x>0?QA:-QA);
}

/*
* New version of the sigmoidMapping with polyfit from numpy
*/
int sigmoidMappingV2(int x)
{
    if (x >= -QX && x <= QX)
        return (x * 7810 + 195) >> 18;
    else if ((x >= -2 * QX && x < -QX) || (x <= 2 * QX && x > QX))
        return x > 0 ? ((x * 4899 + 47996260) >> 18) : ((x * 4899 - 47996260) >> 18);
    else if ((x >= -3 * QX && x < -2 * QX) || (x <= 3 * QX && x > 2 * QX))
        return x > 0 ? ((x * 2330 + 130915972) >> 18) : ((x * 2330 - 130915972) >> 18);
    else if ((x >= -4 * QX && x < -3 * QX) || (x <= 4 * QX && x > 3 * QX))
        return x > 0 ? ((x * 952 + 197514809) >> 18) : ((x * 952 - 197514809) >> 18);
    else if ((x >= -5 * QX && x < -4 * QX) || (x <= 5 * QX && x > 4 * QX))
        return x > 0 ? ((x * 364 + 235417895) >> 18) : ((x * 364 - 235417895) >> 18);
    else
        return (x > 0 ? QA : -QA);
}

int sigmoidMappingV2_32(int x)
{
    if (x >= -QX && x <= QX)
        return (x * 511836160 + 12779520) >> 2;
    else if ((x >= -2 * QX && x < -QX) || (x <= 2 * QX && x > QX))
        return x > 0 ? ((x * 321060864 + (int)(3.145482895 * pow(10,12))) >> 2) : ((x * 321060864 - (int)(3.145482895 * pow(10,12))) >> 2);
    else if ((x >= -3 * QX && x < -2 * QX) || (x <= 3 * QX && x > 2 * QX))
        return x > 0 ? ((x * 152698880 + (int)(8.579709141* pow(10,12)))  >> 2) : ((x * 2330 - (int)(8.579709141* pow(10,12))) >> 2);
//    else if ((x >= -4 * QX && x < -3 * QX) || (x <= 4 * QX && x > 3 * QX))
//        return x > 0 ? ((x * 952 + 197514809) >> 2) : ((x * 952 - 197514809) >> 2);
//    else if ((x >= -5 * QX && x < -4 * QX) || (x <= 5 * QX && x > 4 * QX))
//        return x > 0 ? ((x * 364 + 235417895) >> 2) : ((x * 364 - 235417895) >> 2);
    else
        return (x > 0 ? QA : -QA);
}
/*
 * Sigmoid full calculation
 */
#define EULER_NUMBER_F 2.71828182846
//1073741824 = 2^30
int sigmoidfull(float n) {
    return (int)(((1 / (1 + powf(EULER_NUMBER_F, -(n))))*2-1)*1073741824);
}
void qNorm()
{
    uint8_t *s_src = (uint8_t*)&ISP_DBDATA->data[0][0][0];
    uint8_t *d_src = s_src + IMGHEIGHT * IMGWIDTH * IMGCHANNEL;
    short *s_dst = (short*)&AIIP_NORM->data[0][0][0];

    while(s_src<d_src)
    {
        *s_dst = (short)(((int)(*s_src) * Qtune)>>2); // Qtune = (4 * 2^14) / 255 = 257.00...
        s_src++;
        s_dst++;
    }
}

/*
* Now model has been trained with / 256 as normalization.
*/
void qNorm_256()
{
    uint8_t* s_src = (uint8_t*)&ISP_DBDATA->data[0][0][0];
    uint8_t* d_src = s_src + IMGHEIGHT * IMGWIDTH * IMGCHANNEL;
    short* s_dst = (short*)&AIIP_NORM->data[0][0][0];

    while (s_src < d_src)
    {
        *s_dst = (short)((*s_src) << quantized_norm); // ((4 * 2^14) / 256) / 4 = 2^14 / 2^8 = 2^6
        s_src++;
        s_dst++;
    }
}


void qDownSample()
{
    for(int h = 0; h < DCENET_HEIGHT; ++h)
    {
        for(int w = 0; w < DCENET_WIDTH; ++w)
        {
            AIIP_NETIO->data[h][w][0] = AIIP_NORM->data[h*DCENET_DSRATE][w*DCENET_DSRATE][0];
            AIIP_NETIO->data[h][w][1] = AIIP_NORM->data[h*DCENET_DSRATE][w*DCENET_DSRATE][1];
            AIIP_NETIO->data[h][w][2] = AIIP_NORM->data[h*DCENET_DSRATE][w*DCENET_DSRATE][2];
        }
    }
}


void qConv1st()
{
    for(int h = 0; h < DCENET_HEIGHT; ++h)
    {
        for(int w = 0; w < DCENET_WIDTH; ++w)
        {
            for(int cout = 0; cout < DCENET_CHANNEL; ++cout)
            {
                int sum = 0;
                for(int cin = 0; cin < IMGCHANNEL; ++cin)
                {
                    for(int kh = -1; kh <= 1; ++kh)
                    {
                        for(int kw = -1; kw <= 1; ++kw)
                        {
                            if(((h+kh) >=0) && ((w+kw) >=0) && ((h+kh) < DCENET_HEIGHT) && ((w+kw) < DCENET_WIDTH))
                                sum += AIIP_NETIO->data[h+kh][w+kw][cin] * AIIP_CONVW01->data[cout][cin][kh+1][kw+1];
                        }
                    }
                }
                sum += AIIP_CONVB01->data[cout]; // Add bias
                sum = (sum>0)?sum:0; // ReLU
                AIIP_FEATURE1->data[h][w][cout] = sum >> quantized_conv; // Divide Qw and store to the layer output (feature map 1)
            }
        }
    }
}

void qConv2nd()
{
    for(int h = 0; h < DCENET_HEIGHT; ++h)
    {
        for(int w = 0; w < DCENET_WIDTH; ++w)
        {
            for(int cout = 0; cout < DCENET_CHANNEL; ++cout)
            {
                int sum = 0;
                for(int cin = 0; cin < DCENET_CHANNEL; ++cin)
                {
                    for(int kh = -1; kh <= 1; ++kh)
                    {
                        for(int kw = -1; kw <= 1; ++kw)
                        {
                            if(((h+kh) >=0) && ((w+kw) >=0) && ((h+kh) < DCENET_HEIGHT) && ((w+kw) < DCENET_WIDTH))
                                sum += AIIP_FEATURE1->data[h+kh][w+kw][cin] * AIIP_CONVW02->data[cout][cin][kh+1][kw+1];
                        }
                    }
                }
                sum += AIIP_CONVB02->data[cout];
                sum = (sum>0)?sum:0;
                AIIP_FEATURE2->data[h][w][cout] = sum >> quantized_conv;
            }
        }
    }
}

void qConv3rd()
{
    for(int h = 0; h < DCENET_HEIGHT; ++h)
    {
        for(int w = 0; w < DCENET_WIDTH; ++w)
        {
            for(int cout = 0; cout < IMGCHANNEL; ++cout)
            {
                int sum = 0;
                for(int cin = 0; cin < DCENET_CHANNEL; ++cin)
                {
                    for(int kh = -1; kh <= 1; ++kh)
                    {
                        for(int kw = -1; kw <= 1; ++kw)
                        {
                            if(((h+kh) >=0) && ((w+kw) >=0) && ((h+kh) < DCENET_HEIGHT) && ((w+kw) < DCENET_WIDTH))
                                sum += (AIIP_FEATURE2->data[h+kh][w+kw][cin] + AIIP_FEATURE1->data[h+kh][w+kw][cin]) * AIIP_CONVW03->data[cout][cin][kh+1][kw+1];
                        }
                    }
                }
                sum += AIIP_CONVB03->data[cout];
                AIIP_NETIO->data[h][w][cout] = (short)sigmoidMapping(sum >> 14);
            }
        }
    }
}


void qConv3rdV2()
{
    for (int h = 0; h < DCENET_HEIGHT; ++h)
    {
        for (int w = 0; w < DCENET_WIDTH; ++w)
        {
            for (int cout = 0; cout < IMGCHANNEL; ++cout)
            {
                int sum = 0;
                for (int cin = 0; cin < DCENET_CHANNEL; ++cin)
                {
                    for (int kh = -1; kh <= 1; ++kh)
                    {
                        for (int kw = -1; kw <= 1; ++kw)
                        {
                            if (((h + kh) >= 0) && ((w + kw) >= 0) && ((h + kh) < DCENET_HEIGHT) && ((w + kw) < DCENET_WIDTH))
                                sum += (AIIP_FEATURE2->data[h + kh][w + kw][cin] + AIIP_FEATURE1->data[h + kh][w + kw][cin]) * AIIP_CONVW03->data[cout][cin][kh + 1][kw + 1];
                        }
                    }
                }
                sum += AIIP_CONVB03->data[cout];
                AIIP_NETIO->data[h][w][cout] = (long)sigmoidfull(sum >> quantized_conv);
//                AIIP_NETIO->data[h][w][cout] = (long)sigmoidMappingV2_32(sum >> quantized_conv);

            }
        }
    }
}



void qUpSample()
{
#if(DCENET_USOPTION == 0)
    for(int h = 0; h < IMGHEIGHT; ++h)
    {
        for(int w = 0; w < IMGWIDTH; ++w)
        {
            AIIP_PARAM->data[h][w][0] = AIIP_NETIO->data[(int)(h/DCENET_DSRATE)][(int)(w/DCENET_DSRATE)][0];
            AIIP_PARAM->data[h][w][1] = AIIP_NETIO->data[(int)(h/DCENET_DSRATE)][(int)(w/DCENET_DSRATE)][1];
            AIIP_PARAM->data[h][w][2] = AIIP_NETIO->data[(int)(h/DCENET_DSRATE)][(int)(w/DCENET_DSRATE)][2];
        }
    }

#elif(DCENET_USOPTION == 1)
    for(int h = 0; h < DCENET_HEIGHT; ++h)
    {
        for(int w = 0; w < IMGWIDTH; ++w)
        {
            int sW = (Qus*w + Qusc) / DCENET_DSRATE - Qusc;
            if(sW >= 0 && sW < (Qus* (DCENET_WIDTH-1)))
            {
                int i = sW / Qus;
                int j = i + 1;
                AIIP_USBUFFER->data[h][w][0] = ((sW - i*Qus) * (AIIP_NETIO->data[h][j][0] - AIIP_NETIO->data[h][i][0]) >> 10) + AIIP_NETIO->data[h][i][0];
                AIIP_USBUFFER->data[h][w][1] = ((sW - i*Qus) * (AIIP_NETIO->data[h][j][1] - AIIP_NETIO->data[h][i][1]) >> 10) + AIIP_NETIO->data[h][i][1];
                AIIP_USBUFFER->data[h][w][2] = ((sW - i*Qus) * (AIIP_NETIO->data[h][j][2] - AIIP_NETIO->data[h][i][2]) >> 10) + AIIP_NETIO->data[h][i][2];
            }
            else
            {
            	AIIP_USBUFFER->data[h][w][0] = AIIP_NETIO->data[h][(int)(w/DCENET_DSRATE)][0];
            	AIIP_USBUFFER->data[h][w][1] = AIIP_NETIO->data[h][(int)(w/DCENET_DSRATE)][0];
            	AIIP_USBUFFER->data[h][w][2] = AIIP_NETIO->data[h][(int)(w/DCENET_DSRATE)][0];
            }
        }
    }

    for(int h = 0; h < IMGHEIGHT; ++h)
    {
        int sH = (Qus*h + Qusc) / DCENET_DSRATE - Qusc;
        if(sH >= 0 && sH < (Qus* (DCENET_HEIGHT-1)))
        {
            int i = sH / Qus;
            int j = i + 1;
            for(int w = 0; w < IMGWIDTH; ++w)
            {
            	AIIP_PARAM->data[h][w][0] = ((sH - i*Qus) * (AIIP_USBUFFER->data[j][w][0] - AIIP_USBUFFER->data[i][w][0]) >> 10) + AIIP_USBUFFER->data[i][w][0];
            	AIIP_PARAM->data[h][w][1] = ((sH - i*Qus) * (AIIP_USBUFFER->data[j][w][1] - AIIP_USBUFFER->data[i][w][1]) >> 10) + AIIP_USBUFFER->data[i][w][1];
            	AIIP_PARAM->data[h][w][2] = ((sH - i*Qus) * (AIIP_USBUFFER->data[j][w][2] - AIIP_USBUFFER->data[i][w][2]) >> 10) + AIIP_USBUFFER->data[i][w][2];
            }
        }
        else
        {
            for(int w = 0; w < IMGWIDTH; ++w)
            {
            	AIIP_PARAM->data[h][w][0] = AIIP_USBUFFER->data[(int)(h/DCENET_DSRATE)][w][0];
            	AIIP_PARAM->data[h][w][1] = AIIP_USBUFFER->data[(int)(h/DCENET_DSRATE)][w][1];
            	AIIP_PARAM->data[h][w][2] = AIIP_USBUFFER->data[(int)(h/DCENET_DSRATE)][w][2];
            }
        }
    }

#else
    int coef[12] = {42, 128, 213, 298, 384, 469, 554, 640, 725, 810, 896, 981};
    for(int h = 0; h < DCENET_HEIGHT; ++h)
    {
        int wi = 0;
        for(int d = 0; d < DCENET_DSRATE/2; ++d, ++wi)
        {
            AIIP_USBUFFER->data[h][wi][0] = AIIP_NETIO->data[h][0][0];
            AIIP_USBUFFER->data[h][wi][1] = AIIP_NETIO->data[h][0][1];
            AIIP_USBUFFER->data[h][wi][2] = AIIP_NETIO->data[h][0][2];
        }
        for(int w = 1; w < DCENET_WIDTH; ++w)
        {
            for(int d = 0; d < DCENET_DSRATE; ++d, ++wi)
            {
                AIIP_USBUFFER->data[h][wi][0] = (coef[d] * (AIIP_NETIO->data[h][w][0] - AIIP_NETIO->data[h][w-1][0]) >> 10) + AIIP_NETIO->data[h][w-1][0];
                AIIP_USBUFFER->data[h][wi][1] = (coef[d] * (AIIP_NETIO->data[h][w][1] - AIIP_NETIO->data[h][w-1][1]) >> 10) + AIIP_NETIO->data[h][w-1][1];
                AIIP_USBUFFER->data[h][wi][2] = (coef[d] * (AIIP_NETIO->data[h][w][2] - AIIP_NETIO->data[h][w-1][2]) >> 10) + AIIP_NETIO->data[h][w-1][2];
            }
        }
        for(int d = 0; d < DCENET_DSRATE/2; ++d, ++wi)
        {
            AIIP_USBUFFER->data[h][wi][0] = AIIP_NETIO->data[h][DCENET_WIDTH-1][0];
            AIIP_USBUFFER->data[h][wi][1] = AIIP_NETIO->data[h][DCENET_WIDTH-1][1];
            AIIP_USBUFFER->data[h][wi][2] = AIIP_NETIO->data[h][DCENET_WIDTH-1][2];
        }
    }

    int hi = 0;
    for(int d = 0; d < DCENET_DSRATE/2; ++d, ++hi)
    {
        for(int w = 0; w < IMGWIDTH; ++w)
        {
            AIIP_PARAM->data[hi][w][0] = AIIP_USBUFFER->data[0][w][0];
            AIIP_PARAM->data[hi][w][1] = AIIP_USBUFFER->data[0][w][1];
            AIIP_PARAM->data[hi][w][2] = AIIP_USBUFFER->data[0][w][2];
        }
    }
    for(int h = 1; h < DCENET_HEIGHT; ++h)
    {
        for(int d = 0; d < DCENET_DSRATE; ++d, ++hi)
        {
            for(int w = 0; w < IMGWIDTH; ++w)
            {
                AIIP_PARAM->data[hi][w][0] = (coef[d] * (AIIP_USBUFFER->data[h][w][0] - AIIP_USBUFFER->data[h-1][w][0]) >> 10) + AIIP_USBUFFER->data[h-1][w][0];
                AIIP_PARAM->data[hi][w][1] = (coef[d] * (AIIP_USBUFFER->data[h][w][1] - AIIP_USBUFFER->data[h-1][w][1]) >> 10)  + AIIP_USBUFFER->data[h-1][w][1];
                AIIP_PARAM->data[hi][w][2] = (coef[d] * (AIIP_USBUFFER->data[h][w][2] - AIIP_USBUFFER->data[h-1][w][2]) >> 10)  + AIIP_USBUFFER->data[h-1][w][2];
            }

        }
    }
    for(int d = 0; d < DCENET_DSRATE/2; ++d, ++hi)
    {
        for(int w = 0; w < IMGWIDTH; ++w)
        {
            AIIP_PARAM->data[hi][w][0] = AIIP_USBUFFER->data[DCENET_HEIGHT-1][w][0];
            AIIP_PARAM->data[hi][w][1] = AIIP_USBUFFER->data[DCENET_HEIGHT-1][w][1];
            AIIP_PARAM->data[hi][w][2] = AIIP_USBUFFER->data[DCENET_HEIGHT-1][w][2];
        }
    }
#endif
}

void qEnhance()
{
    short *s = (short*)&AIIP_NORM->data[0][0][0];
    short *sa = (short*)&AIIP_PARAM->data[0][0][0];
    short *d = s + IMGCHANNEL * IMGHEIGHT * IMGWIDTH;
    uint8_t *yd = (uint8_t*)&ISP_AIISPDATA->data[0][0][0];

    int output;
    while(s<d)
    {
        int x_q = (*s) >> 4;
        int a_q = *sa;
        for(int i = 0; i < 8; ++i)
        {
            int x_q2 = x_q * QI;
            int x_q3 = x_q2 * QA;
            x_q3 = x_q3 + a_q * (x_q * x_q - x_q2);
            x_q = (int)(x_q3 / (QI * QA));
        }
        output = x_q/(QI/255); // (QI/256) In pytorch the output is [0 ~ 1] need to * 255 (in our case 256)
        output = output>255?255:output;
        *yd = (uint8_t)(output);
        s++;
        sa++;
        yd++;
    }
}

void qEnhance_256() // TODO : Long Long int
{
    short* s = (short*)&AIIP_NORM->data[0][0][0]; // with Qx : 2^14
    short* sa = (short*)&AIIP_PARAM->data[0][0][0];
    short* d = s + IMGCHANNEL * IMGHEIGHT * IMGWIDTH;
    uint8_t* yd = (uint8_t*)&ISP_AIISPDATA->data[0][0][0];

    long long int output;
    while (s < d)
    {
        long long int x_q = (*s) >> 4; // with Qx / 2^4 => 2^10 => QI
        long long int a_q = *sa;       // already with 2^10 ( /= 2^4 happened in sigmoidmapping function)
        for (int i = 0; i < 8; ++i)
        {
            long long int x_q2 = x_q << 10; // * QI
            long long int x_q3 = x_q2 << 10; // * QA
            x_q3 = x_q3 + a_q * (x_q * x_q - x_q2);
            x_q = (long long int)(x_q3 >> 20); // (/ (Qi * QA))
            if (x_q > static_cast<long long int>(1023))
                x_q = static_cast<long long int>(1023);
            if (x_q < static_cast<long long int>(0))
                x_q = static_cast<long long int>(0);
        }
        output = x_q >> 2; // (/ (QI/256)) In pytorch the output is [0 ~ 1] need to * 255 (in our case 256)
        output = output > 255 ? 255 : output;
        *yd = (uint8_t)(output);
        s++;
        sa++;
        yd++;
    }
}


void qEnhance_256_q8()
{
    short* s = (short*)&AIIP_NORM->data[0][0][0]; // with Qx : 2^14
    short* sa = (short*)&AIIP_PARAM->data[0][0][0];
    short* d = s + IMGCHANNEL * IMGHEIGHT * IMGWIDTH;
    uint8_t* yd = (uint8_t*)&ISP_AIISPDATA->data[0][0][0];

    int output;
    while (s < d)
    {
        long long int x_q = (*s) >> 6; // with Qx / 2^6 => 2^8 => QI
        long long int a_q = *sa >> 2;       // already with 2^8 ( /= 2^4 happened in sigmoidmapping function)
        for (int i = 0; i < 8; ++i)
        {
            long long int x_q2 = x_q << 8; // * QI / 2^2
            long long int x_q3 = x_q2 << 8; // * QA / 2^2
            x_q3 = x_q3 + a_q * (x_q * x_q - x_q2);
            x_q = (long long int)(x_q3 >> 16); // (/ (Qi * QA / 2^4))
            if (x_q > static_cast<long long int>(1023))
                x_q = static_cast<long long int>(1023);
            if (x_q < static_cast<long long int>(0))
                x_q = static_cast<long long int>(0);
        }
        output = x_q; // (/ (QI/256)) In pytorch the output is [0 ~ 1] need to * 255 (in our case 256)
        output = output > 255 ? 255 : output;
        *yd = (uint8_t)(output);
        s++;
        sa++;
        yd++;
    }
}

void qEnhance_256_q4()
{
    short* s = (short*)&AIIP_NORM->data[0][0][0]; // with Qx : 2^14
    short* sa = (short*)&AIIP_PARAM->data[0][0][0];
    short* d = s + IMGCHANNEL * IMGHEIGHT * IMGWIDTH;
    uint8_t* yd = (uint8_t*)&ISP_AIISPDATA->data[0][0][0];

    int output;
    while (s < d)
    {
        int x_q = (*s) >> 10; // with Qx / 2^10 => 2^4 => QI
        int a_q = *sa >> 6;       // already with 2^10 ( /= 2^4 happened in sigmoidmapping function)
        for (int i = 0; i < 8; ++i)
        {
            int x_q2 = x_q << 4; // * QI
            int x_q3 = x_q2 << 4; // * QA
            x_q3 = x_q3 + a_q * (x_q * x_q - x_q2);
            x_q = (int)(x_q3 >> 8); // (/ (Qi * QA))
        }
        output = x_q << 4; // (/ (QI/256)) In pytorch the output is [0 ~ 1] need to * 255 (in our case 256)
        output = output > 255 ? 255 : output;
        *yd = (uint8_t)(output);
        s++;
        sa++;
        yd++;
    }
}


//void qDCENet()
//{
//	qNorm();
//	qDownSample();
//	qConv1st();
//	qConv2nd();
//	qConv3rd();
//	qUpSample();
//	qEnhance();
//}


void qLoadParam()
{
    memcpy((void*)AIIP_CONVW01, conv1_w, 2 * 864);
    memcpy((void*)AIIP_CONVB01, conv1_b, 4 * 32);
    memcpy((void*)AIIP_CONVW02, conv2_w, 2 * 9216);
    memcpy((void*)AIIP_CONVB02, conv2_b, 4 * 32);
    memcpy((void*)AIIP_CONVW03, conv3_w, 2 * 864);
    memcpy((void*)AIIP_CONVB03, conv3_b, 4 * 3);
}

void writeImageData(std::string fileName) {
    cv::Mat image(IMGHEIGHT, IMGWIDTH, CV_8UC3);
    printf("Output image rows : %10u\timage cols : %10u\timage channels : %10u\n", image.rows, image.cols, image.channels());

    for (int c = 0; c < CMOS_IMGC; ++c)
        for (int y = 0; y < CMOS_IMGH; ++y)
            for (int x = 0; x < CMOS_IMGW; ++x) {
                image.at<cv::Vec3b>(y, x)[c] = ISP_AIISPDATA->data[y][x][c];
            }

    cv::imwrite(fileName,image);
    printf("Showed image rows : %10u\timage cols : %10u\timage channels : %10u\n", image.rows, image.cols, image.channels());
}

void cvReadImg(std::string fileName) {
    cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR);
    printf("image rows : %u\timage cols : %u\timage channels : %u\n", image.rows, image.cols, image.channels());

    //cv::resize(image, image, cv::Size(1920, 1200), cv::InterpolationFlags::INTER_AREA);
    //cv::imshow("[1920 * 1200] 04.jpg", image);
//    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    //printf("image rows : %u\timage cols : %u\timage channels : %u\n", image.rows, image.cols, image.channels());

    for (int c = 0; c < image.channels(); ++c)
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x) {
                ISP_DBDATA->data[y][x][c] = image.at<cv::Vec3b>(y, x)[c];
                /*if (c == 0 && y == 0 && x == 0)
                    printf("ISP_DBDATA->data[%d][%d][%d] = %u\n", y, x, c, image.at<cv::Vec3b>(y, x)[c]);*/
            }

    printf("Showed image rows : %10u\timage cols : %10u\timage channels : %10u\n", image.rows, image.cols, image.channels());
    cv::namedWindow("Input_image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input_image", image);
    cv::waitKey(0);
}

