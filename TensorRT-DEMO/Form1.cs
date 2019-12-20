using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Diagnostics;
using System.IO;
using OpenCvSharp;

namespace TensortDEMO
{
    public partial class Form1 : Form
    {
        private IntPtr ptr;
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
        }

        [DllImport("TensorRT.dll")]
        public static extern int ONNX2TRT(string onnxFileName, string trtFileName, int batchSize);

        [DllImport("TensorRT.dll")]
        public static extern IntPtr LoadNet(string trtFileName);

        [DllImport("TensorRT.dll")]
        public static extern void ReleaseNet(IntPtr ptr);

        [DllImport("TensorRT.dll")]
        public static extern void DoInference(IntPtr ptr, string input_name, string output_name, float[] input, float[] output, int input_size, int output_size);

        private void button1_Click(object sender, EventArgs e)
        {
            Debug.WriteLine(ONNX2TRT("./data/dense.onnx", "./data/dense.trt", 1));
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Mat source = new Mat(@"./data/test.jpg", ImreadModes.Color);
            source = source.Resize(new OpenCvSharp.Size(128, 128));
            Debug.WriteLine(Image.Normalize(source)[0]);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            ptr = LoadNet("./data/dense.trt");
            Debug.WriteLine(ptr);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            ReleaseNet(ptr);
        }

        private void button5_Click(object sender, EventArgs e)
        {
            Mat source = new Mat(@"./data/test.jpg", ImreadModes.Color);
            source = source.Resize(new OpenCvSharp.Size(128, 128));
            float[] input = Image.Normalize(source);
            float[] output = new float[1];
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 0; i < 1000; i++)
            {
                DoInference(ptr, "features", "classifier", input, output, 3 * 128 * 128, 1);
            }
            sw.Stop();
            Debug.WriteLine("1000次 耗时{0}ms.", sw.Elapsed.TotalMilliseconds);
            Debug.WriteLine(output[0]);
        }
    }
}
