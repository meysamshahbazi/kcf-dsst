// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// #include "precomp.hpp"

#include "dsst.hpp"
#include "utils.hpp"

// #include "CSRTUtils.hpp"

//Discriminative Scale Space Tracking
namespace cv
{

    static void computeHOG32D(const Mat &imageM, Mat &featM, const int sbin, const int pad_x, const int pad_y)
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

template <typename _Tp>
void OLBP_(const Mat& src, Mat& dst) {
	dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
	for(int i=1;i<src.rows-1;i++) {
		for(int j=1;j<src.cols-1;j++) {
			_Tp center = src.at<_Tp>(i,j);
			unsigned char code = 0;
			code |= (src.at<_Tp>(i-1,j-1) > center) << 7;
			code |= (src.at<_Tp>(i-1,j) > center) << 6;
			code |= (src.at<_Tp>(i-1,j+1) > center) << 5;
			code |= (src.at<_Tp>(i,j+1) > center) << 4;
			code |= (src.at<_Tp>(i+1,j+1) > center) << 3;
			code |= (src.at<_Tp>(i+1,j) > center) << 2;
			code |= (src.at<_Tp>(i+1,j-1) > center) << 1;
			code |= (src.at<_Tp>(i,j-1) > center) << 0;
			dst.at<unsigned char>(i,j) = code;
		}
	}
}




std::vector<Mat> get_features_hog_dsst(const Mat &im, const int bin_size)
{
    std::vector<Mat> features;
    
    // im.convertTo(img_patch, CV_32FC3);
    Mat hsv = bgr2hsv(im);
    // std::cout<<im.rows<<"|"<<im.cols<<" - "<<im.channels() <<std::endl;
    // bgr2hsv(im,hsv);
    // cvtColor(im,im_hsv,COLOR_BGR2GRAY);




    Mat hogmatrix;
    Mat im_;// =  bgr2hsv(im);
    im.convertTo(im_, CV_64FC3, 1.0/255.0);
    computeHOG32D(im_,hogmatrix,bin_size,1,1);
    // std::cout<<hogmatrix.rows<<"|"<<hogmatrix.cols<<" - "<<hogmatrix.channels() <<std::endl;
    hogmatrix.convertTo(hogmatrix, CV_32F);
    Size hog_size = im.size();
    // cout<<"hog_size"<<hog_size<<endl;
    hog_size.width /= bin_size;
    hog_size.height /= bin_size;
    Mat hogc(hog_size, CV_32FC(32), hogmatrix.data); //#MEYSHA: change this constant number 
    //  std::cout<<"hogc: "<<hogc.rows<<"|"<<hogc.cols<<" - "<<hogc.channels() <<std::endl;
    // std::vector<Mat> features;
    std::vector<Mat> hog_vec;
    split(hogc, hog_vec);

    features.insert(features.end(), hog_vec.begin(),
        hog_vec.begin()+8);
    
    // std::cout<<features.at(0).rows<<"|"<<features.at(0).cols<<"-"<<features.at(0).channels() <<std::endl;

    
    Mat gray_m;
    cvtColor(im, gray_m, COLOR_BGR2GRAY);
    resize(gray_m, gray_m, hog_size, 0, 0, INTER_CUBIC);
    gray_m.convertTo(gray_m, CV_32FC1, 1.0/255.0, -0.5);
    features.push_back(gray_m);




    std::vector<Mat> hsv_ch;
    split(hsv,hsv_ch);

    for(auto & c:hsv_ch)
    {
        Mat lbp;
        OLBP_<unsigned char>(c,lbp);
        lbp.convertTo(lbp, CV_32F, 1.0/255.0, -0.5);
        resize(lbp,lbp,hog_size);
        features.push_back(lbp);
        // std::cout<<lbp.rows<<"|"<<lbp.cols<<" - "<<lbp.channels() <<std::endl;
        // add hsv channels!
        // Mat cc;
        // c.convertTo(cc, CV_32FC1, 1.0/255.0, -0.5);
        // resize(cc,cc,hog_size);
        // features.push_back(cc);
    }

   // features.push_back(im_g);

    // std::cout<<features.size()<<std::endl;


    //----------------------------------------------------
    //------------add lbp and hsv fetures 
    // Mat lbp;
    // OLBP_<unsigned char>(im_g,lbp);

    // lbp.convertTo(lbp, CV_32FC1, 1.0/255.0, -0.5);
    // features.push_back(lbp);

    // im_g.convertTo(im_g, CV_32FC1, 1.0/255.0,-0.5);
    // features.push_back(im_g);    

    // std::cout<< "im.size()"<<im.size()<<std::endl;
    // imshow("im",im);
    // waitKey(1);
    // Mat hsv = bgr2hsv(im_.clone());
    //     // Mat hsv = patch.clone();
    // // resize(hsv, hsv, feature_size, 0, 0, INTER_CUBIC);
    // std::vector<Mat> hsv_ch;
    // split(hsv,hsv_ch);

    // for(auto & c:hsv_ch)
    // {
    //     Mat lbp;
    //     OLBP_<unsigned char>(c,lbp);
    //     lbp.convertTo(lbp, CV_32FC1, 1.0/255.0, -0.5);
    //     features.push_back(lbp);
    //     // add hsv channels!
    //     // Mat cc;
    //     // c.convertTo(cc, CV_32FC1, 1.0/255.0, -0.5);
    //     // features.push_back(cc);
    // }


    // cout<<"features size: "<<features.size()<<endl;
    return features;
}



class ParallelGetScaleFeatures : public ParallelLoopBody
{
public:
    ParallelGetScaleFeatures(
        Mat img,
        Point2f pos,
        Size2f base_target_sz,
        float current_scale,
        std::vector<float> &scale_factors,
        Mat scale_window,
        Size scale_model_sz,
        int col_len,
        Mat &result)
    {
        this->img = img;
        this->pos = pos;
        this->base_target_sz = base_target_sz;
        this->current_scale = current_scale;
        this->scale_factors = scale_factors;
        this->scale_window = scale_window;
        this->scale_model_sz = scale_model_sz;
        this->col_len = col_len;
        this->result = result;
    }
    virtual void operator ()(const Range& range) const CV_OVERRIDE
    {
        for (int s = range.start; s < range.end; s++) {
            Size patch_sz = Size(static_cast<int>(current_scale * scale_factors[s] * base_target_sz.width),
                    static_cast<int>(current_scale * scale_factors[s] * base_target_sz.height));
            Mat img_patch = get_subwindow(img, pos, patch_sz.width, patch_sz.height,NULL);
            // imshow("patch2",img_patch);
            // waitKey(200);
            // img_patch.convertTo(img_patch, CV_32FC3); // #MEYSHA: we comment this line becuse aother convert is laready called inside get_features_hog_dsst()
            resize(img_patch, img_patch, Size(scale_model_sz.width, scale_model_sz.height),0,0,INTER_LINEAR);
            // imshow("patch2",img_patch);
            // waitKey(200);
            std::vector<Mat> hog;
            hog = get_features_hog_dsst(img_patch, 2); //MEYSHA: change this to 4 
            // std::cout<<"hog size: "<<hog.size()<<std::endl;
            
            for (int i = 0; i < static_cast<int>(hog.size()); ++i) {
                hog[i] = hog[i].t();
                hog[i] = scale_window.at<float>(0,s) * hog[i].reshape(0, col_len);
                hog[i].copyTo(result(Rect(Point(s, i*col_len), hog[i].size())));
            }
        }
    }

    ParallelGetScaleFeatures& operator=(const ParallelGetScaleFeatures &) {
        return *this;
    }

private:
    Mat img;
    Point2f pos;
    Size2f base_target_sz;
    float current_scale;
    std::vector<float> scale_factors;
    Mat scale_window;
    Size scale_model_sz;
    int col_len;
    Mat result;
};


DSST::DSST(const Mat &image,
        Rect2f bounding_box,
        Size2f template_size,
        int numberOfScales,
        float scaleStep,
        float maxModelArea,
        float sigmaFactor,
        float scaleLearnRate):
    scales_count(numberOfScales), scale_step(scaleStep), max_model_area(maxModelArea),
    sigma_factor(sigmaFactor), learn_rate(scaleLearnRate)
{
    original_targ_sz = bounding_box.size();
    Point2f object_center = Point2f(bounding_box.x + original_targ_sz.width / 2,
            bounding_box.y + original_targ_sz.height / 2);

    current_scale_factor = 1.0;
    if(scales_count % 2 == 0)
        scales_count++;

    scale_sigma = static_cast<float>(sqrt(scales_count) * sigma_factor);

    min_scale_factor = static_cast<float>(pow(scale_step,
            cvCeil(log(max(5.0 / template_size.width, 5.0 / template_size.height)) / log(scale_step))));
    max_scale_factor = static_cast<float>(pow(scale_step,
            cvFloor(log(min((float)image.rows / (float)bounding_box.width,
            (float)image.cols / (float)bounding_box.height)) / log(scale_step))));
    ys = Mat(1, scales_count, CV_32FC1);
    float ss, sf;
    for(int i = 0; i < ys.cols; ++i) {
        ss = (float)(i+1) - cvCeil((float)scales_count / 2.0f);
        ys.at<float>(0,i) = static_cast<float>(exp(-0.5 * pow(ss,2) / pow(scale_sigma,2)));
        sf = static_cast<float>(i + 1);
        scale_factors.push_back(pow(scale_step, cvCeil((float)scales_count / 2.0f) - sf));
    }

    scale_window = get_hann_win(Size(scales_count, 1));

    float scale_model_factor = 1.0;
    if(template_size.width * template_size.height * pow(scale_model_factor, 2) > max_model_area)
    {
        scale_model_factor = sqrt(max_model_area /
                (template_size.width * template_size.height));
    }
    scale_model_sz = Size(cvFloor(template_size.width * scale_model_factor),
            cvFloor(template_size.height * scale_model_factor));

    Mat scale_resp = get_scale_features(image, object_center, original_targ_sz,
            current_scale_factor, scale_factors, scale_window, scale_model_sz);

    Mat ysf_row = Mat(ys.size(), CV_32FC2);
    dft(ys, ysf_row, DFT_ROWS | DFT_COMPLEX_OUTPUT, 0);
    ysf = repeat(ysf_row, scale_resp.rows, 1);
    Mat Fscale_resp;
    dft(scale_resp, Fscale_resp, DFT_ROWS | DFT_COMPLEX_OUTPUT);
    mulSpectrums(ysf, Fscale_resp, sf_num, 0 , true);
    Mat sf_den_all;
    mulSpectrums(Fscale_resp, Fscale_resp, sf_den_all, 0, true);
    reduce(sf_den_all, sf_den, 0, REDUCE_SUM, -1);
}

DSST::~DSST()
{
}

Mat DSST::get_scale_features(
        Mat img,
        Point2f pos,
        Size2f base_target_sz,
        float current_scale,
        std::vector<float> &scale_factors,
        Mat scale_window,
        Size scale_model_sz)
{
    Mat result;
    int col_len = 0;
    Size patch_sz = Size(cvFloor(current_scale * scale_factors[0] * base_target_sz.width),
            cvFloor(current_scale * scale_factors[0] * base_target_sz.height));
    Mat img_patch = get_subwindow(img, pos, patch_sz.width, patch_sz.height,NULL);
    // img_patch.convertTo(img_patch, CV_32FC3);
    resize(img_patch, img_patch, Size(scale_model_sz.width, scale_model_sz.height),0,0,INTER_LINEAR);
    std::vector<Mat> hog;
    hog = get_features_hog_dsst(img_patch, 2); //MEYSHA: change this to 4 
    // std::cout<<"hog size: "<<hog.size()<<std::endl;

    result = Mat(Size((int)scale_factors.size(), hog[0].cols * hog[0].rows * (int)hog.size()), CV_32F);
    col_len = hog[0].cols * hog[0].rows;
    for (int i = 0; i < static_cast<int>(hog.size()); ++i) {
        hog[i] = hog[i].t();
        hog[i] = scale_window.at<float>(0,0) * hog[i].reshape(0, col_len);
        hog[i].copyTo(result(Rect(Point(0, i*col_len), hog[i].size())));
    }

    ParallelGetScaleFeatures parallelGetScaleFeatures(img, pos, base_target_sz,
            current_scale, scale_factors, scale_window, scale_model_sz, col_len, result);
    parallel_for_(Range(1, static_cast<int>(scale_factors.size())), parallelGetScaleFeatures);
    return result;
}
  
void DSST::update(const Mat &image, const Point2f object_center)
{
    Mat scale_features = get_scale_features(image, object_center, original_targ_sz,
            current_scale_factor, scale_factors, scale_window, scale_model_sz);
    Mat Fscale_features;
    dft(scale_features, Fscale_features, DFT_ROWS | DFT_COMPLEX_OUTPUT);
    Mat new_sf_num;
    Mat new_sf_den;
    Mat new_sf_den_all;
    mulSpectrums(ysf, Fscale_features, new_sf_num, DFT_ROWS, true);
    Mat sf_den_all;
    mulSpectrums(Fscale_features, Fscale_features, new_sf_den_all, DFT_ROWS, true);
    reduce(new_sf_den_all, new_sf_den, 0, REDUCE_SUM, -1);

    sf_num = (1 - learn_rate) * sf_num + learn_rate * new_sf_num;
    sf_den = (1 - learn_rate) * sf_den + learn_rate * new_sf_den;
}

float DSST::getScale(const Mat &image, const Point2f object_center)
{
    Mat scale_features = get_scale_features(image, object_center, original_targ_sz,
            current_scale_factor, scale_factors, scale_window, scale_model_sz);

    Mat Fscale_features;
    dft(scale_features, Fscale_features, DFT_ROWS | DFT_COMPLEX_OUTPUT);

    mulSpectrums(Fscale_features, sf_num, Fscale_features, 0, false);
    Mat scale_resp;
    reduce(Fscale_features, scale_resp, 0, REDUCE_SUM, -1);
    scale_resp = divide_complex_matrices(scale_resp, sf_den + 0.01f);
    idft(scale_resp, scale_resp, DFT_REAL_OUTPUT|DFT_SCALE);
    Point max_loc;
    minMaxLoc(scale_resp, NULL, NULL, NULL, &max_loc);

    current_scale_factor *= scale_factors[max_loc.x];
    if(current_scale_factor < min_scale_factor)
        current_scale_factor = min_scale_factor;
    else if(current_scale_factor > max_scale_factor)
        current_scale_factor = max_scale_factor;

    return current_scale_factor;
}
} /* namespace cv */
