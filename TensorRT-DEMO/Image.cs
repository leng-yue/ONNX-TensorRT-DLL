using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;

namespace TensortDEMO
{
    public class Image
    {
        private static readonly float[] kMean = { 0.5f, 0.5f, 0.5f };
        private static readonly float[] kStdDev = { 0.5f, 0.5f, 0.5f };

        public static float[] Normalize(Mat img)
        {
            float[] data = new float[img.Rows * img.Cols * 3];

            for (int c = 0; c < 3; ++c)
            {
                for (int i = 0; i < img.Rows; ++i)
                {
                    for (int j = 0; j < img.Cols; ++j)
                    {
                        data[c * img.Cols * img.Rows + i * img.Cols + j] = (img.At<Vec3b>(i, j)[c] / 255.0f - kMean[c]) / kStdDev[c];
                    }
                }
            }
            return data;
        }
    }
}
