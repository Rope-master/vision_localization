#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "ipm.hpp"
#include "lane_line_detection.hpp"

std::string g_camera_parameter_file = "/home/fangjun/task1/vision_localization/test/data/pudong.yaml";
struct TestIpmTF: public ::testing::Test
{
    TestIpmTF()
            : lane_detection_(g_camera_parameter_file)
    {}
    virtual void SetUp()
    {
    }

    IpmTF tf_;
    LaneLineDetection lane_detection_;
};

#if 0
TEST_F(TestIpmTF, ptTF)
{
    cv::Point2f src;
    cv::Point2f dst;
    cv::Point2f expected;

    src.x = (float)cv::theRNG();
    src.y = (float)cv::theRNG();

    // identity transform
    cv::Mat tf = cv::Mat::eye(3, 3, CV_32FC1);
    expected = src;

    dst = tf_.ptTF(src, tf);
    EXPECT_NEAR(0, cv::norm(dst-expected), 1e-10);

    // scaling
    tf = (cv::Mat_<float>(3,3) <<
            1, 0, 0,
            0, 1, 0,
            0, 0, 10
    );
    expected = src/10;
    dst = tf_.ptTF(src, tf);
    EXPECT_NEAR(0, cv::norm(dst-expected), 1e-10);

    tf = (cv::Mat_<float>(3,3) <<
            2, 0, 0,
            0, 0.5, 0,
            0, 0, 100
    );
    expected.x = 2*src.x/100;
    expected.y = 0.5*src.y/100;
    dst = tf_.ptTF(src, tf);
    EXPECT_NEAR(0, cv::norm(dst-expected), 1e-10);

    tf = (cv::Mat_<float>(3,3) <<
            2, 3, 4,
            5, 6, 7,
            0, 0, 100
    );
    expected.x = (2*src.x + 3*src.y + 4)/100;
    expected.y = (5*src.x + 6*src.y + 7)/100;
    dst = tf_.ptTF(src, tf);
    EXPECT_NEAR(0, cv::norm(dst-expected), 1e-10);
}
#endif

TEST_F(TestIpmTF, doIpm)
{
    std::cout << lane_detection_.getIpmTF().getImgToIpm() << std::endl;
    std::cout << lane_detection_.paramAsString() << std::endl;

    std::vector<cv::String> image_lists;
    image_lists.push_back("1515666448632636.jpg");
    image_lists.push_back("1515666449809240.jpg");
    image_lists.push_back("1515666451108024.jpg");
    image_lists.push_back("1515666452420853.jpg");
    image_lists.push_back("1515666453868917.jpg");
    image_lists.push_back("1515666455462546.jpg");
    image_lists.push_back("1515666456712187.jpg");
    image_lists.push_back("1515666458407627.jpg");
    image_lists.push_back("1515666459809910.jpg");
    image_lists.push_back("1515666461196603.jpg");
    image_lists.push_back("1515666463058874.jpg");
    image_lists.push_back("1515666464277079.jpg");
    image_lists.push_back("1515666465360121.jpg");
    image_lists.push_back("1515666466794292.jpg");
    image_lists.push_back("1515666468143589.jpg");
    for (size_t i = 0; i < image_lists.size(); i++)
    {
        //if (i != 2) continue;
        cv::String image_filename = cv::format("%s/%s",
                                               "/home/fangjun/task1/vision_localization/test/data",
                                               image_lists[i].c_str()
        );
        std::cout << image_filename << "\n";
        cv::Mat image = cv::imread(image_filename, cv::IMREAD_COLOR);

        cv::Mat ipm_image;
        cv::warpPerspective(image, ipm_image, lane_detection_.getIpmTF().getImgToIpm(),
                            cv::Size(lane_detection_.getRoiOut().width, lane_detection_.getRoiOut().height)
        );
        cv::imshow("image", image);
        cv::imshow("ipm image", ipm_image);
        cv::waitKey(0);
    }
}
