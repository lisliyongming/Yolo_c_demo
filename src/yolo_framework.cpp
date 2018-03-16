//============================================================================
// Name        : yolo_framework.cpp
// Author      : Steven li
// Version     :
// Copyright   : v3
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;
#include "cv.h"
#include "highgui.h"
#include "./include/detection.h"
#include "./include/network.h"
#define IMAGE_RANGE 675840//640*352*3
#define MUTIPLE_VALUE11 3840//1280*3
//#define DETECTION_TEST
CvSize image_size;
CvSize image_resize_picture;
CvScalar color_value_r;
CvScalar color_value_g;
CvScalar color_value_b;
CvScalar color_value_y;
CvFont font;
const char *label_display[3] = {"ped", "car", "cyc"};//pedestrian car cyclist
int main() {
	cout << "!!!Hello v3best yolo pc device test!!!" << endl; // prints !!!Hello v3best!!!
	//================= variable declaration =================//
	uint32_t k = 0;
	uint32_t ii = 0;
	uint32_t a = 0;
	uint32_t b = 0;
	network net;
	int i = 0;
	uint32_t j = 0;
	uint32_t valid_box_num = 0;
	IplImage *pIplImage_data_pret;
	int* bboxs = new int[7700];//The variable must be initialed if transport to function
	float* result_from_conv9;
	char *input_image_data;
	char data_in[IMAGE_RANGE];
	float* image_data_float = new float[IMAGE_RANGE];

	//================= variable initialization =================//
	image_size.height = 720;
	image_size.width = 1280;
	image_resize_picture.height = 352;//1280/720->640/352
	image_resize_picture.width = 640;
	color_value_r = CV_RGB(255,0,0);
	color_value_g = CV_RGB(0,255,0);
	color_value_b = CV_RGB(0,0,255);
	color_value_y = CV_RGB(244,208,0);
	cvInitFont(&font, CV_FONT_HERSHEY_TRIPLEX, 1, 1, 0, 1, 8);
	pIplImage_data_pret = cvCreateImage(image_resize_picture,IPL_DEPTH_8U,3);

	//================= pre read image by opencv =================//
	IplImage* pIplImage_data_input = cvLoadImage( "/home/hans/workspace_eclipse/yolo_framework/src/test1.jpg",CV_LOAD_IMAGE_COLOR );
	if(!pIplImage_data_input){
		cout << "Reading picture failed" << endl;
		exit(0);
	}
	//resize to suitable size for yolo input pattern 	BGR -> RGB
	input_image_data = (pIplImage_data_input->imageData) + 1280*8*3;
	a = 0;
	for(i = 0; i < 352; ++i) {
		for(j = 0; j < 640; ++j) {
			data_in[a] = *(input_image_data+i*MUTIPLE_VALUE11*2+j*6+2);
			data_in[a+1] = *(input_image_data+i*MUTIPLE_VALUE11*2+j*6+1);
			data_in[a+2] = *(input_image_data+i*MUTIPLE_VALUE11*2+j*6+0);
			a = a + 3;
		}
	}
	pIplImage_data_pret->imageData = data_in;
	cvShowImage( "Resize", pIplImage_data_pret );
	//================= pretreat image data =================//
	//input data need normalization to 0~1 range /255.0, must transform to unsigned char type
#ifdef DEBUG_TEST
	unsigned char *data = (unsigned char *)pIplImage_data_pret->imageData;
	int h = pIplImage_data_pret->height;
	int w = pIplImage_data_pret->width;
	int c = pIplImage_data_pret->nChannels;
	int step = pIplImage_data_pret->widthStep;
	for(i = 0; i < h; ++i){
		for(k= 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				*(image_data_float+k*w*h + i*w + j) = data[i*step + j*c + k]/255.0;
			}
		}
	}
#endif
	i = 0;
	a = 640*352;
	b = a*2;
	//format RGB...RGB to RRR...GGG...BBB
	for(k = 0; k < a; k++){
		//		*(image_data_float+k) = (*(data+i)) / 255.0;
		//		*(image_data_float+k+a) = (*(data+i+1)) / 255.0;
		//		*(image_data_float+k+b) = (*(data+i+2)) / 255.0;
		//		*(image_data_float+k) = *((pIplImage_data_pret->imageData)+i) / 255.0;
		//		*(image_data_float+k+a) = *((pIplImage_data_pret->imageData)+i+1) / 255.0;
		//		*(image_data_float+k+b) = *((pIplImage_data_pret->imageData)+i+2) / 255.0;
		*(image_data_float+k) = ((unsigned char)data_in[i]) / 255.0;
		*(image_data_float+k+a) = ((unsigned char)data_in[i+1]) / 255.0;
		*(image_data_float+k+b) = ((unsigned char)data_in[i+2]) / 255.0;
		i += 3;
	}

	cout << "Pretreatment is OK!" << endl;
#ifdef DEBUG_TEST
	FILE *pfile_pic = fopen("/home/hans/workspace_eclipse/yolo_framework/src/picture.bin","rb");
	if(pfile_pic == NULL){
		cout << "Open file failed!" << endl;
		fclose(pfile_pic);
		exit(0);
	}
	ii = fread(image_data_float,sizeof(float),640*352*3,pfile_pic);
	cout << ii << endl;
	fclose(pfile_pic);
#endif
	//================= forward propagation =================//
	//network architecture display
	// layer num  --- layer name  ---  filter quantity  ---  filter size/stride  ---    input   ---   output
	//	   1            conv 1              16                   3x3/1                640x352x3       640x352x16       
	//     2          max pool 1                                 2x2/2                640x352x16      320x176x16
	//	   3            conv 2              32                   3x3/1                320x176x16      320x176x32       
	//     4          max pool 2                                 2x2/2                320x176x32      160x88x32
	//	   5            conv 3              64                   3x3/1                160x88x32       160x88x64     
	//     6          max pool 3                                 2x2/2                160x88x64       80x44x64
	//	   7            conv 4              128                  3x3/1                80x44x64        80x44x128    
	//     8          max pool 4                                 2x2/2                80x44x128       40x22x128
	//	   9            conv 5              256                  3x3/1                40x22x128       40x22x256    
	//     10         max pool 5                                 2x2/2                40x22x256       20x11x256
	//	   11           conv 6              512                  3x3/1                20x11x256       20x11x512    
	//	   12           conv 7              512                  3x3/1                20x11x512       20x11x512    
	//	   13           conv 8              512                  3x3/1                20x11x512       20x11x512    
	//	   14           conv 9              40                   1x1/1                20x11x512       20x11x40    
	//     15           detection

	//convolution 1 BatchNormalization Relu
	//max pool 1
	//convolution 2 BatchNormalization Relu
	//max pool 2
	//convolution 3 BatchNormalization Relu
	//max pool 3
	//convolution 4 BatchNormalization Relu
	//max pool 4
	//convolution 5 BatchNormalization Relu
	//max pool 5
	//convolution 6 BatchNormalization Relu
	//convolution 7 BatchNormalization Relu
	//convolution 8 BatchNormalization Relu
	//convolution 9
	net = cfg_network(14);
	cout << "Network Initialization is done!" <<endl;
	net.workspace_input = image_data_float;
	for(i=0;i<net.n;i++){
		layer l = net.layers[i];
		l.forward(l,&net);
		net.workspace_input = l.output;
		cout << "The "<< i << " layer is done" << endl;
#ifdef DEBUG_TEST
		if(i==0){
			FILE *pfile_conv1 = fopen("/home/hans/workspace_eclipse/yolo_framework/src/conv1_output.bin","wb");
			if(pfile_conv1 == NULL){
				cout << "Open file failed!" << endl;
				fclose(pfile_conv1);
				exit(0);
			}
			ii = fwrite(net.workspace_input,sizeof(float),640*352*16,pfile_conv1);
			cout << ii << endl;
			fclose(pfile_conv1);
		}
#endif
	}
	delete []image_data_float;

#ifdef DEBUG_TEST
	char *m=new char[8800*4];
	FILE *pfile_1 = fopen("/home/hans/workspace_eclipse/yolo_framework/src/conv9_output.bin","rb");
	if(pfile_1 == NULL){
		cout << "Open file failed!" << endl;
		fclose(pfile_1);
		exit(0);
	}
	fread(m,8800*4,1,pfile_1);
	result_from_conv9_bin = (float *)m;
	for(ii = 0; ii < 10; ii++){
		cout << "The deepred " << ii << "is "<< *(result_from_conv9_bin+ii) << endl;
	}
	fclose(pfile_1);
#endif
	//================= detection process=================//
#ifdef DETECTION_TEST
	char *m=new char[8800*4];
	FILE *pfile_1 = fopen("/home/hans/workspace_eclipse/yolo_framework/src/conv9_output.bin","rb");
	if(pfile_1 == NULL){
		cout << "Open file failed!" << endl;
		fclose(pfile_1);
		exit(0);
	}
	fread(m,8800*4,1,pfile_1);
	result_from_conv9 = (float *)m;
	fclose(pfile_1);
	char *m_parameter=new char[4*4]();
	float *parameter_display;
	FILE *pfile_2 = fopen("/home/hans/workspace_eclipse/yolo_framework/src/parameter.bin","rb");
	if(pfile_2 == NULL){
		cout << "Open file failed!" << endl;
		fclose(pfile_2);
		exit(0);
	}
	uint32_t a = fread(m_parameter,1,16,pfile_2);
	cout << "read quantity is " << a << endl;
	//uint32_t b = sizeof(m_parameter);//only the size of point,64bit system => 8 byte
	parameter_display = (float *)m_parameter;
	for(k=0;k<4;k++){
		cout << *(parameter_display+k) << endl;
	}
	fclose(pfile_2);
	delete []m_parameter;
#endif
	result_from_conv9 = net.workspace_input;
#ifdef DEBUG_TEST
	for(ii = 0; ii < 10; ii++){
		cout << "The " << ii << " is "<< *(result_from_conv9+ii) << endl;
	}
#endif
	valid_box_num = detection(bboxs,result_from_conv9);
	cout << "valid_box_num=" << valid_box_num << endl;
	//================= OpenCV display process=================//
	//rectangle and put text process
	for (ii = 0; ii< valid_box_num; ++ii)
	{
		if (bboxs[ii*7+6] == 0)
		{
			//put text and rectangle on valid place of image
			cvRectangle( pIplImage_data_input,
					cvPoint(bboxs[ii*7+1], bboxs[ii*7+2]),
					cvPoint(bboxs[ii*7+3], bboxs[ii*7+4]),
					color_value_y, 3, 4, 0 );
#ifdef DETECTION_TEST
			cout << bboxs[ii*7+1] << " "<< bboxs[ii*7+2] << endl;
			cout << bboxs[ii*7+3] << " "<< bboxs[ii*7+4] << endl;
			cout << label_display[bboxs[ii*7+0]] << endl;
#endif
			cvPutText(pIplImage_data_input, label_display[bboxs[ii*7+0]], cvPoint(bboxs[ii*7+1], bboxs[ii*7+2]), &font, color_value_r);
		}
	}

#ifdef DETECTION_TEST
	delete []m;
#endif
	delete []bboxs;
	cvShowImage( "Rectangle Result", pIplImage_data_input );
	cvWaitKey( 0 );
	cvReleaseImage( &pIplImage_data_input );
	cvReleaseImage( &pIplImage_data_pret );
	return 0;
}
