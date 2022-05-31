#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

inline int modul( int a, int b)
{
    return ( (a % b)+b)%b;
}

Mat bgr2hsv(const Mat &img);

Mat get_subwindow(
    const Mat &image,
    const Point2f center,
    const int w,
    const int h,
    Rect * valid_pixels
);

Mat get_hann_win(Size sz);

Mat divide_complex_matrices(const Mat &A, const Mat &B);
void computeHOG32D(const Mat &imageM, Mat &featM, const int sbin, const int pad_x, const int pad_y);


#endif

