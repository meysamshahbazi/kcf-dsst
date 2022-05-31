
#include "utils.hpp"


Mat bgr2hsv(const Mat &img)
{
    Mat hsv;
    cvtColor(img,hsv,COLOR_BGR2HSV);
    vector<Mat> hsv_channels;
    split(hsv,hsv_channels);
    hsv_channels.at(0).convertTo(hsv_channels.at(0),CV_8UC1, 255.0 / 180.0 );
    merge(hsv_channels,hsv);
    return hsv;
    
}


Mat get_subwindow(
    const Mat &image,
    const Point2f center,
    const int w,
    const int h,
    Rect * valid_pixels)
{
    int start_x = cvFloor(center.x - w/2 + 1);
    int start_y = cvFloor(center.y - h/2 + 1);

    Rect roi(start_x,start_y,w,h);

    int padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0;

    if(roi.x <0)
    {
        padding_left = -roi.x;
        roi.x = 0;
    }
    if(roi.y <0)
    {
        padding_top = -roi.y;
        roi.y = 0;
    }

    roi.width -= padding_left;
    roi.height -= padding_top;

    if (roi.x + roi.width >= image.cols)
    {
        padding_right = roi.x + roi.width - image.cols;
        roi.width = image.cols -  roi.x;
    }
    if (roi.y + roi.height >= image.rows)
    {
        padding_bottom = roi.y + roi.height - image.rows;
        roi.height = image.rows - roi.y;
    }

    Mat subwin = image(roi).clone();

    copyMakeBorder(subwin,subwin,padding_top,padding_bottom,padding_left,padding_right,
                    BORDER_REPLICATE);

    if (valid_pixels != NULL)
    {
        *valid_pixels = Rect(padding_left, padding_top, roi.width, roi.height);
    }

    return subwin;
}


Mat get_hann_win(Size sz)
{
    Mat hann_rows = Mat::ones(sz.height, 1, CV_32F);
    Mat hann_cols = Mat::ones(1, sz.width, CV_32F);
    int NN = sz.height - 1;
    if(NN != 0) {
        for (int i = 0; i < hann_rows.rows; ++i) {
            hann_rows.at<float>(i,0) = (float)(1.0/2.0 * (1.0 - cos(2*CV_PI*i/NN)));
        }
    }
    NN = sz.width - 1;
    if(NN != 0) {
        for (int i = 0; i < hann_cols.cols; ++i) {
            hann_cols.at<float>(0,i) = (float)(1.0/2.0 * (1.0 - cos(2*CV_PI*i/NN)));
        }
    }
    return hann_rows * hann_cols;
}


Mat divide_complex_matrices(const Mat &A,const Mat &B)
{
    vector<Mat> va,vb;
    split(A,va);
    split(B,vb);

    Mat a_r = va.at(0); // _r stand for real part
    Mat a_i = va.at(1); // _i stand for imaginary part 

    Mat b_r = vb.at(0); // _r stand for real part
    Mat b_i = vb.at(1); // _i stand for imaginary part 

    Mat div = b_r.mul(b_r) + b_i.mul(b_i);

    Mat real_part = a_r.mul(b_r) + a_i.mul(b_i);

    Mat im_part = a_i.mul(b_r) - a_r.mul(b_i);

    divide(real_part,div,real_part);
    divide(im_part,div,im_part);

    vector<Mat> tmp(2);


    tmp[0] = real_part;
    tmp[1] = im_part;

    Mat res;

    merge(tmp,res);

    return res;
     
}

void computeHOG32D(const Mat &imageM, Mat &featM, const int sbin, const int pad_x, const int pad_y)
{
    // cout<<"featM.size "<<featM.size()<<endl;
    const int dimHOG = 32; //#MEYSHA: this can be changed but at  get_features_hog the 32 number must be changed 
    CV_Assert(pad_x >= 0);
    CV_Assert(pad_y >= 0);
    CV_Assert(imageM.channels() == 3);
    CV_Assert(imageM.depth() == CV_64F);

    // epsilon to avoid division by zero
    const double eps = 0.0001;
    // number of orientations
    const int numOrient = 18;
    // unit vectors to compute gradient orientation
    const double uu[9] = {1.000, 0.9397, 0.7660, 0.5000, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397};
    const double vv[9] = {0.000, 0.3420, 0.6428, 0.8660, 0.9848,  0.9848,  0.8660,  0.6428,  0.3420};

    // image size
    const Size imageSize = imageM.size();
    // block size
    // int bW = cvRound((double)imageSize.width/(double)sbin);
    // int bH = cvRound((double)imageSize.height/(double)sbin);
    int bW = cvFloor((double)imageSize.width/(double)sbin);
    int bH = cvFloor((double)imageSize.height/(double)sbin);
    const Size blockSize(bW, bH);
    // size of HOG features
    int oW = max(blockSize.width-2, 0) + 2*pad_x;
    int oH = max(blockSize.height-2, 0) + 2*pad_y;
    Size outSize = Size(oW, oH);
    // size of visible
    const Size visible = blockSize*sbin;

    // initialize historgram, norm, output feature matrices
    Mat histM = Mat::zeros(Size(blockSize.width*numOrient, blockSize.height), CV_64F);
    Mat normM = Mat::zeros(Size(blockSize.width, blockSize.height), CV_64F);
    featM = Mat::zeros(Size(outSize.width*dimHOG, outSize.height), CV_64F);

    // get the stride of each matrix
    const size_t imStride = imageM.step1();
    const size_t histStride = histM.step1();
    const size_t normStride = normM.step1();
    const size_t featStride = featM.step1();

    // calculate the zero offset
    const double* im = imageM.ptr<double>(0);
    double* const hist = histM.ptr<double>(0);
    double* const norm = normM.ptr<double>(0);
    double* const feat = featM.ptr<double>(0);

    for (int y = 1; y < visible.height - 1; y++)
    {
        for (int x = 1; x < visible.width - 1; x++)
        {
            // OpenCV uses an interleaved format: BGR-BGR-BGR
            const double* s = im + 3*min(x, imageM.cols-2) + min(y, imageM.rows-2)*imStride;

            // blue image channel
            double dyb = *(s+imStride) - *(s-imStride);
            double dxb = *(s+3) - *(s-3);
            double vb = dxb*dxb + dyb*dyb;

            // green image channel
            s += 1;
            double dyg = *(s+imStride) - *(s-imStride);
            double dxg = *(s+3) - *(s-3);
            double vg = dxg*dxg + dyg*dyg;

            // red image channel
            s += 1;
            double dy = *(s+imStride) - *(s-imStride);
            double dx = *(s+3) - *(s-3);
            double v = dx*dx + dy*dy;

            // pick the channel with the strongest gradient
            if (vg > v) { v = vg; dx = dxg; dy = dyg; }
            if (vb > v) { v = vb; dx = dxb; dy = dyb; }

            // snap to one of the 18 orientations
            double best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < (int)numOrient/2; o++)
            {
                double dot =  uu[o]*dx + vv[o]*dy;
                if (dot > best_dot)
                {
                    best_dot = dot;
                    best_o = o;
                }
                else if (-dot > best_dot)
                {
                    best_dot = -dot;
                    best_o = o + (int)(numOrient/2);
                }
            }

            // add to 4 historgrams around pixel using bilinear interpolation
            double yp =  ((double)y+0.5)/(double)sbin - 0.5;
            double xp =  ((double)x+0.5)/(double)sbin - 0.5;
            int iyp = (int)cvFloor(yp);
            int ixp = (int)cvFloor(xp);
            double vy0 = yp - iyp;
            double vx0 = xp - ixp;
            double vy1 = 1.0 - vy0;
            double vx1 = 1.0 - vx0;
            v = sqrt(v);

            // fill the value into the 4 neighborhood cells
            if (iyp >= 0 && ixp >= 0)
                *(hist + iyp*histStride + ixp*numOrient + best_o) += vy1*vx1*v;

            if (iyp >= 0 && ixp+1 < blockSize.width)
                *(hist + iyp*histStride + (ixp+1)*numOrient + best_o) += vx0*vy1*v;

            if (iyp+1 < blockSize.height && ixp >= 0)
                *(hist + (iyp+1)*histStride + ixp*numOrient + best_o) += vy0*vx1*v;

            if (iyp+1 < blockSize.height && ixp+1 < blockSize.width)
                *(hist + (iyp+1)*histStride + (ixp+1)*numOrient + best_o) += vy0*vx0*v;

        } // for y
    } // for x

    // compute the energy in each block by summing over orientation
    for (int y = 0; y < blockSize.height; y++)
    {
        const double* src = hist + y*histStride;
        double* dst = norm + y*normStride;
        double const* const dst_end = dst + blockSize.width;
        // for each cell
        while (dst < dst_end)
        {
            *dst = 0;
            for (int o = 0; o < (int)(numOrient/2); o++)
            {
                *dst += (*src + *(src + numOrient/2))*
                    (*src + *(src + numOrient/2));
                src++;
            }
            dst++;
            src += numOrient/2;
        }
    }

    // compute the features
    for (int y = pad_y; y < outSize.height - pad_y; y++)
    {
        for (int x = pad_x; x < outSize.width - pad_x; x++)
        {
            double* dst = feat + y*featStride + x*dimHOG;
            double* p, n1, n2, n3, n4;
            const double* src;

            p = norm + (y - pad_y + 1)*normStride + (x - pad_x + 1);
            n1 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y - pad_y)*normStride + (x - pad_x + 1);
            n2 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y- pad_y + 1)*normStride + x - pad_x;
            n3 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y - pad_y)*normStride + x - pad_x;
            n4 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);

            double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;

            // contrast-sesitive features
            src = hist + (y - pad_y + 1)*histStride + (x - pad_x + 1)*numOrient;
            for (int o = 0; o < numOrient; o++)
            {
                double val = *src;
                double h1 = min(val*n1, 0.2);
                double h2 = min(val*n2, 0.2);
                double h3 = min(val*n3, 0.2);
                double h4 = min(val*n4, 0.2);
                *(dst++) = 0.5 * (h1 + h2 + h3 + h4);

                src++;
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
            }

            // contrast-insensitive features
            src =  hist + (y - pad_y + 1)*histStride + (x - pad_x + 1)*numOrient;
            for (int o = 0; o < numOrient/2; o++)
            {
                double sum = *src + *(src + numOrient/2);
                double h1 = min(sum * n1, 0.2);
                double h2 = min(sum * n2, 0.2);
                double h3 = min(sum * n3, 0.2);
                double h4 = min(sum * n4, 0.2);
                *(dst++) = 0.5 * (h1 + h2 + h3 + h4);
                src++;
            }

            // texture features
            *(dst++) = 0.2357 * t1;
            *(dst++) = 0.2357 * t2;
            *(dst++) = 0.2357 * t3;
            *(dst++) = 0.2357 * t4;
            // truncation feature
            *dst = 0;
        }// for x
    }// for y
    // Truncation features
    for (int m = 0; m < featM.rows; m++)
    {
        for (int n = 0; n < featM.cols; n += dimHOG)
        {
            if (m > pad_y - 1 && m < featM.rows - pad_y && n > pad_x*dimHOG - 1 && n < featM.cols - pad_x*dimHOG)
                continue;

            featM.at<double>(m, n + dimHOG - 1) = 1;
        } // for x
    }// for y

        // cout<<"featM.size at end "<<featM.size()<<endl;

}


