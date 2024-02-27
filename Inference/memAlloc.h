//
// Created by Andrew Bui on 4/17/23.
//

#ifndef QUANTIZEDZERODCE_CPP_ZERODCE_MEMALLOC_H
#define QUANTIZEDZERODCE_CPP_ZERODCE_MEMALLOC_H

#include <cstdint>


#define DCENET_USOPTION   2

//#define CMOS_IMGH  1200
//#define CMOS_IMGW  1920
#define CMOS_IMGH  1018
#define CMOS_IMGW  1021
#define CMOS_IMGC  3
#define CMOS_DATABUFFER_AMOUNT  10

#define DCENET_DSRATE    1
#define IMGHEIGHT        (CMOS_IMGH) // 1200
#define IMGWIDTH         (CMOS_IMGW) // 1920
#define IMGCHANNEL       (CMOS_IMGC) // 3
#define DCENET_HEIGHT    (IMGHEIGHT/DCENET_DSRATE) // 1200 / 12 => 100
#define DCENET_WIDTH     (IMGWIDTH/DCENET_DSRATE)  // 1920 / 12 => 160
#define DCENET_CHANNEL   32


#define QX     16384    // 2^14
#define QW     16384    // 2^14
#define QB     (QX*QW)  // 2^28
#define QI     1024     // 2^10
#define QA     1024     // 2^10
#define Qtune  257      // Don't need in new version
//#define Qus    1024     // No use
//#define Qusc   (Qus/2)  // No use

//typedef short QuanType; // No use
//typedef int DQuanType;  // No use



typedef volatile struct cmosRawData     /* No use */
{
    uint8_t data[CMOS_DATABUFFER_AMOUNT][CMOS_IMGH][CMOS_IMGW*10/8];
} cmosRawData_t;

typedef volatile struct cmosTMData      /* TM: Tone Mapping */
{
    uint8_t data[CMOS_IMGH][CMOS_IMGW];
} cmosTMData_t;

typedef volatile struct cmosRGBData     /* Model Input data */
{
    uint8_t data[CMOS_IMGH][CMOS_IMGW][CMOS_IMGC];
} cmosRGBData_t;


typedef volatile struct qNormImg        /* Normalized */
{
    short data[IMGHEIGHT][IMGWIDTH][IMGCHANNEL];
}qNormImg_t;

typedef volatile struct qNetIO          /* Neural Network Input/Output */
{
    short data[DCENET_HEIGHT][DCENET_WIDTH][IMGCHANNEL];
}qNetIO_t;

typedef volatile struct qNetFeature     /* Neural Network Feature Map */
{
    short data[DCENET_HEIGHT][DCENET_WIDTH][DCENET_CHANNEL];
}qNetFeature_t;

typedef volatile struct qEnhanceParam   /* Estimated Enhancing Parameters */
{
    short data[IMGHEIGHT][IMGWIDTH][IMGCHANNEL];
}qEnhanceParam_t;

typedef volatile struct usBuffer        /* Up Sampling Buffer*/
{
    short data[DCENET_HEIGHT][IMGWIDTH][IMGCHANNEL];
}usBuffer_t;



typedef volatile struct qWConv1st
{
    short data[DCENET_CHANNEL][IMGCHANNEL][3][3];
}qWConv1st_t;
typedef volatile struct qBConv1st
{
    int data[DCENET_CHANNEL];
}qBConv1st_t;


typedef volatile struct qWConv2nd
{
    short data[DCENET_CHANNEL][DCENET_CHANNEL][3][3];
}qWConv2nd_t;
typedef volatile struct qBConv2nd
{
    int data[DCENET_CHANNEL];
}qBConv2nd_t;


typedef volatile struct qWConv3rd
{
    short data[IMGCHANNEL][DCENET_CHANNEL][3][3];
}qWConv3rd_t;
typedef volatile struct qBConv3rd
{
    int data[IMGCHANNEL];
}qBConv3rd_t;


extern cmosRawData_t* ISP_RAWDATA;
extern cmosTMData_t* ISP_TMDATA;
extern cmosRGBData_t* ISP_DBDATA;
extern cmosRGBData_t* ISP_WBDATA;
extern cmosRGBData_t* ISP_AIISPDATA;
extern qNormImg_t* AIIP_NORM;
extern qNetIO_t* AIIP_NETIO;
extern qNetFeature_t* AIIP_FEATURE1;
extern qNetFeature_t* AIIP_FEATURE2;
extern qEnhanceParam_t* AIIP_PARAM;
extern qEnhanceParam_t* AIIP_USBUFFER;
extern qWConv1st_t* AIIP_CONVW01;
extern qBConv1st_t* AIIP_CONVB01;
extern qWConv2nd_t* AIIP_CONVW02;
extern qBConv2nd_t* AIIP_CONVB02;
extern qWConv3rd_t* AIIP_CONVW03;
extern qBConv3rd_t* AIIP_CONVB03;

//#define ISP_RAWDATA       ((cmosRawData_t*)(0xA0000000))
//#define ISP_TMDATA        ((cmosTMData_t*)(0xA1B77400))
//#define ISP_DBDATA        ((cmosRGBData_t*)(0xA1DA9C00))
//#define ISP_WBDATA        ((cmosRGBData_t*)(0xA2441400))
//#define ISP_AIISPDATA     ((cmosRGBData_t*)(0xA2AD8C00))
//#define AIIP_NORM         ((qNormImg_t*)(0xA3170400))
//#define AIIP_NETIO        ((qNetIO_t*)(0xA3E9F400))
//#define AIIP_FEATURE1     ((qNetFeature_t*)(0xA3EB6B00))
//#define AIIP_FEATURE2     ((qNetFeature_t*)(0xA3FB0B00))
//#define AIIP_PARAM        ((qEnhanceParam_t*)(0xA40AAB00))
//#define AIIP_USBUFFER     ((qEnhanceParam_t*)(0xA40AAB00))
//#define AIIP_CONVW01      ((qWConv1st_t*)(0xA4EF2F00))
//#define AIIP_CONVB01      ((qBConv1st_t*)(0xA4EF35C0))
//#define AIIP_CONVW02      ((qWConv2nd_t*)(0xA4EF3640))
//#define AIIP_CONVB02      ((qBConv2nd_t*)(0xA4EF7E40))
//#define AIIP_CONVW03      ((qWConv3rd_t*)(0xA4EF7EC0))
//#define AIIP_CONVB03      ((qBConv3rd_t*)(0xA4EF8580))

#endif //QUANTIZEDZERODCE_CPP_ZERODCE_MEMALLOC_H
