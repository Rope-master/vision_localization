#ifndef VISION_LOCALIZATION_IPM
#define VISION_LOCALIZATION_IPM

#include <opencv2/imgproc.hpp>

#include "basic_type.hpp"


/**
 * IpmTF: Inverse Perspective Mapping Transform
 */
class IpmTF
{
public:
    /**
     * This function computes
     *
     *  dst = tf * homogeneous(pt_src);
     *
     *  pt_dst = inhomogeneous(dst);
     *
     * @param pt_src [in] Input point.
     * @param tf     [in] A 3x3 matrix.
     * @return inhomogeneous(tf*homogeneous(pt_src))
     */
    cv::Point2f ptTF(
            const cv::Point2f& pt_src,
            const cv::Mat& tf
    ) const;

    /**
     *
     * @param camera_info [in] Camera parameters.
     * @param roi_in
     * @param roi_out
     * @return
     *
     * @warning **Left-hand** rule instead of right-hand rule is used here.
     *
     * camera_info.pitch defines the rotation angle around x-axis and the rotation matrix is defined as
     * \f[
     *  \begin{cases}
     *   c_1 = \cos (\mathrm{camera\_info.pitch})\\
     *   s_1 = \sin (\mathrm{camera\_info.pitch})\\
     * \mathrm{pitch\_m}=
     * \begin{bmatrix}
     *   1 & 0 & 0\\
     *   0 & c_1 & s_1\\
     *   0 & -s_1 & c_1\\
     * \end{bmatrix}\\
     *  \end{cases}
     * \f]
     *
     * camera_info.yaw defines the rotation angle around z-axis and the rotation matrix is defined as
     * \f[
     *  \begin{cases}
     *   c_2 = \cos (\mathrm{camera\_info.yaw})\\
     *   s_2 = \sin (\mathrm{camera\_info.yaw})\\
     * \mathrm{yaw\_m}=
     * \begin{bmatrix}
     *   c_2 & s_2 & 0\\
     *   -s_2 & c_2 & 0\\
     *   0 & 0 & 1\\
     * \end{bmatrix}\\
     *  \end{cases}
     * \f]
     *
     * camera_info.roll defines the rotation angle around y-axis and the rotation matrix is defined as
     * \f[
     *  \begin{cases}
     *   c_3 = \cos (\mathrm{camera\_info.roll})\\
     *   s_3 = \sin (\mathrm{camera\_info.roll})\\
     * \mathrm{roll\_m}=
     * \begin{bmatrix}
     *   c_3 & 0 & -s_3\\
     *   0 & 1 & 0\\
     *   s_3 & 0 & c_3\\
     * \end{bmatrix}\\
     *  \end{cases}
     * \f]
     *
     * The matrix `image_to_camera` is defined as
     * \f[
     *   \mathrm{image\_to\_camera}=
     *   \begin{bmatrix}
     *    \frac{1}{f_x} & 0 & -\frac{c_x}{f_x} \\
     *    0 & \frac{1}{f_y} & -\frac{c_y}{f_y} \\
     *    0 & 0 & 1\\
     *   \end{bmatrix}
     * \f]
     * where \f$f_x\f$ ,\f$f_y\f$, \f$c_x\f$ and \f$c_y\f$ are provided by `camera_info`.
     *
     * The `camera_to_world` matrix is defined as
     * \f[
     *  \mathrm{camera\_to\_world}=
     *  \begin{bmatrix}
     *   1 & 0 & 0\\
     *   0 & 0 & -1\\
     *   0 & 1 & 0\\
     *  \end{bmatrix}
     * \f]
     *
     * The `image_to_world` matrix is defined as
     * \f[
     *  \mathrm{image\_to\_world}=
     *  \mathrm{roll\_m} * \mathrm{yaw\_m} * \mathrm{pitch\_m} * \mathrm{camera\_to\_world} * \mathrm{image\_to\_camera}
     * \f]
     *
     */
    int doIpm(
            const CameraInfo& camera_info,
            const cv::Rect& roi_in,
            const cv::Rect& roi_out
    );

    cv::Mat getImgToIpm() const
    {
        return img_to_ipm_;
    }

    cv::Mat getIpmToImg() const
    {
        return ipm_to_img_;
    }

private:
    cv::Mat img_to_ipm_; //!< Transformation matrix from pixel coordinate to world coordinate
    cv::Mat ipm_to_img_; //!< Transformation matrix from world coordinate to pixel coordinate
    cv::Point2f pts_src_[4];
    cv::Point2f pts_dst_[4];

    double w_versus_h_; //!< Aspect ration: width/height
};

// TODO: (fangjun) use float instead of double
cv::Point2f IpmTF::ptTF(
        const cv::Point2f &pt_src,
        const cv::Mat &tf) const
{
    CV_Assert(tf.type() == CV_32FC1);
    cv::Point2f pt_dst;

    cv::Mat pt_m = (cv::Mat_<float>(3, 1)
            << pt_src.x, pt_src.y, 1);
    cv::Mat pt_m_new = tf * pt_m;
    pt_dst.x = pt_m_new.at<float>(0, 0) / pt_m_new.at<float>(2, 0);
    pt_dst.y = pt_m_new.at<float>(1, 0) / pt_m_new.at<float>(2, 0);

    return pt_dst;
}

int IpmTF::doIpm(
        const CameraInfo &camera_info,
        const cv::Rect &roi_in,
        const cv::Rect &roi_out)
{

    float c1 = cos(camera_info.pitch * CV_PI / 180);
    float s1 = sin(camera_info.pitch * CV_PI / 180);
    cv::Mat pitch_m = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, c1, s1, 0, -s1, c1);

    float c2 = cos(camera_info.yaw * CV_PI / 180);
    float s2 = sin(camera_info.yaw * CV_PI / 180);
    cv::Mat yaw_m = (cv::Mat_<float>(3, 3) << c2, s2, 0, -s2, c2, 0, 0, 0, 1);

    float c3 = cos(camera_info.roll * CV_PI / 180);
    float s3 = sin(camera_info.roll * CV_PI / 180);
    cv::Mat roll_m = (cv::Mat_<float>(3, 3) << c3, 0, -s3, 0, 1, 0, s3, 0, c3);

    cv::Mat image_to_camera = (cv::Mat_<float>(3, 3) <<
            1 / camera_info.fx, 0, -camera_info.cx / camera_info.fx,
            0, 1 / camera_info.fy, -camera_info.cy / camera_info.fy,
            0, 0, 1);
    cv::Mat camera_to_world = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 0, -1, 0, 1, 0);
    cv::Mat image_to_world = roll_m * yaw_m * pitch_m * camera_to_world * image_to_camera;

    pts_src_[0] = cv::Point2f(roi_in.x, roi_in.y);
    pts_src_[1] = cv::Point2f(roi_in.width, roi_in.y); // TODO: (fangjun) fix it. See LaneLineDetection::LaneLineDetection
    pts_src_[2] = cv::Point2f(roi_in.width, roi_in.height);
    pts_src_[3] = cv::Point2f(roi_in.x, roi_in.height);

    float min_x = INT_MAX;
    float max_x = INT_MIN;
    float min_y = INT_MAX;
    float max_y = INT_MIN;
    for (int i = 0; i < 4; ++i)
    {
        pts_dst_[i] = ptTF(pts_src_[i], image_to_world);

        min_x = std::min(min_x, pts_dst_[i].x);
        max_x = std::max(max_x, pts_dst_[i].x);
        min_y = std::min(min_y, pts_dst_[i].y);
        max_y = std::max(max_y, pts_dst_[i].y);
    }


    float scale_x = roi_out.width / (max_x - min_x);
    float scale_y = roi_out.height / (max_y - min_y);
    cv::Mat scale_m = (cv::Mat_<float>(3, 3) << scale_x, 0, -min_x * scale_x,
            0, scale_y, -min_y * scale_y,
            0, 0, 1);
#if 1
    for (int i = 0; i < 4; ++i)
    {
        pts_dst_[i].x = (pts_dst_[i].x - min_x) / (max_x - min_x) * roi_out.width;
        pts_dst_[i].y = (pts_dst_[i].y - min_y) / (max_y - min_y) * roi_out.height;
    }
#else
    for (int i = 0; i < 4; ++i)
    {
        pts_dst_[i] = ptTF(pts_dst_[i], scale_m);
    }
#endif
    img_to_ipm_ = cv::getPerspectiveTransform(pts_src_, pts_dst_);

    img_to_ipm_.convertTo(img_to_ipm_, CV_32F);

    cv::invert(img_to_ipm_, ipm_to_img_);

    w_versus_h_ = scale_x / scale_y;

    return 0;
}

#endif
