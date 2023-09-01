using CommandLine;
using CSharpApi;
using OpenCvSharp;
using CommandLine;
using System.Data;
using static System.Net.Mime.MediaTypeNames;

namespace ApiTest
{
    internal class Program
    {
        static void HelpMessage()
        {
            string message = $"usage: {AppDomain.CurrentDomain.FriendlyName}.exe " +
              "[--mode detection | landmarks| recognition] \r\n" +
              " [--input_image <path to image>] \r\n" +
              " [--input_image2 <path to image>] \r\n" +
              " [--sdk_path ..] \r\n" +
              " [--window <yes/no>] \r\n" +
              " [--output <yes/no>] \r\n";
            Console.WriteLine(message);
        }
        static string mode = "", image1path = "", image2path = "", sdkpath = "", window = "", output = "";
        static void ParseArguments(string[] args)
        {
            Parser.Default.ParseArguments<ConsoleOptions>(args).WithParsed<ConsoleOptions>(parsed =>
            {
                mode = parsed.Mode;
                image1path = parsed.InputImagePath;
                image2path = parsed.InputImagePath2;
                sdkpath = parsed.SdkPath;
                window = parsed.Window;
                output = parsed.Output;
            }).WithNotParsed<ConsoleOptions>(notparsed =>
            {
                HelpMessage();
                Environment.Exit(0);
            });
        }
        static void CheckFileExist(string path)
        {
            if (File.Exists(path))
                return;
            else
                throw new FileNotFoundException($"File {path} not found");
        }
        static Mat GetCrop(Context obj, Mat image) 
        {
            int img_w = image.Width;
            int img_h = image.Height;

            Context rectCtx = obj["bbox"];
            int x = (int)(rectCtx[0].GetDouble() * img_w);
            int y = (int)(rectCtx[1].GetDouble() * img_h);
            int width = (int)(rectCtx[2].GetDouble() * img_w) - x;
            int height = (int)(rectCtx[3].GetDouble() * img_h) - y;

            Rect rect = new Rect(Math.Max(0, x - (int)(width * 0.25)), Math.Max(0, y - (int)(height * 0.25)),
                                 Math.Min(img_w, (int)(width * 1.5)), Math.Min(img_h, (int)(height * 1.5)));
            return image[rect];
        }
        static Context GetObjectWithMaxConfidence(Context data)
        {
            double max_confidence = 0;
            int index_max_confidence = 0;
            for (int i = 0; i < (int)data["objects"].GetLength(); i++)
            {
                if (data["objects"][i]["class"].GetStr().Equals("face"))
                    continue;
                double confidence = data["objects"][i]["confidence"].GetDouble();
                if (confidence > max_confidence)
                {
                    index_max_confidence = i;
                }
            }
            return data["objects"][index_max_confidence];
        }
        static void DrawBBox(Context obj, Mat img, string output, Scalar color) 
        {
            Context rectCtx = obj["bbox"];
            Point topLeft = new Point((int)(rectCtx[0].GetDouble() * img.Width), (int)(rectCtx[1].GetDouble() * img.Height));
            Point bottomRight = new Point((int)(rectCtx[2].GetDouble() * img.Width), (int)(rectCtx[3].GetDouble() * img.Height));
            int thickness = 2;
            Cv2.Rectangle(img, topLeft, bottomRight, color, thickness);
            if (output == "yes")
            {
                Console.WriteLine("BBox Coordinates: "
                + (int)(rectCtx[0].GetDouble() * img.Width) + ", " + (int)(rectCtx[1].GetDouble() * img.Height) +
           ", " + (int)(rectCtx[2].GetDouble() * img.Width) + ", " + (int)(rectCtx[3].GetDouble() * img.Height));
            }
        }
        static void DrawPoints(Context obj, Mat img, string output)
        {
            Context rectCtx = obj["bbox"];
            double width = rectCtx[2].GetDouble() * img.Cols - rectCtx[0].GetDouble() * img.Cols;
            double height = rectCtx[3].GetDouble() * img.Rows - rectCtx[1].GetDouble() * img.Rows;

            int point_size = width * height > 320 * 480 ? 3 : 1;

            var fitter = obj["fitter"];


            for (ulong i = 0; i < fitter["keypoints"].GetLength(); i++)
            {
                Cv2.Circle(img, new Point(fitter["keypoints"][(int)i][0].GetDouble() * img.Cols, fitter["keypoints"][(int)i][1].GetDouble() * img.Rows), 1, new Scalar(0, 255, 0), point_size);
            }
            
            Cv2.Circle(img, new Point(fitter["left_eye"][0].GetDouble() * img.Cols, fitter["left_eye"][1].GetDouble() * img.Rows), 1, new Scalar( 0, 0, 255), point_size);
            Cv2.Circle(img, new Point(fitter["right_eye"][0].GetDouble() * img.Cols, fitter["right_eye"][1].GetDouble() * img.Rows), 1, new Scalar( 0, 0, 255), point_size);
            Cv2.Circle(img, new Point(fitter["mouth"][0].GetDouble() * img.Cols, fitter["mouth"][1].GetDouble() * img.Rows), 1, new Scalar( 0, 0, 255), point_size);
            if (output == "yes")
            {
                Console.WriteLine($"left eye({fitter["left_eye"][0].GetDouble() * img.Cols:0.000}, {fitter["left_eye"][1].GetDouble() * img.Rows:0.000}), " +
                                $"right eye({fitter["right_eye"][0].GetDouble() * img.Cols:0.000}, {fitter["right_eye"][1].GetDouble() * img.Rows:0.000}), " +
                                    $"mouth({fitter["mouth"][0].GetDouble() * img.Cols:0.000}, {fitter["mouth"][1].GetDouble() * img.Rows:0.000})\r\n");
            }
        }
        public static void MatToBsm(ref Dictionary<object, object> bsmCtx, Mat img, bool copy = false)
        {
            var input_img = img.IsContinuous() ? img : img.Clone();
            long copySize = (copy || !img.IsContinuous()) ? input_img.Total() * input_img.ElemSize() : 0;

            bsmCtx["format"] = "NDARRAY";
            long size = input_img.Total() * input_img.ElemSize();

            byte[] arr = new byte[size]; 
            using (Mat temp = new Mat(input_img.Rows, input_img.Cols, input_img.Type(), arr))
            {
                input_img.CopyTo(temp);
            }
            bsmCtx["blob"] = arr; 
            List<object> sizes = new List<object>();
            for (int i = 0; i < input_img.Dims; ++i)
            {
                sizes.Add(input_img.Size(i));
            }
            sizes.Add(input_img.Channels());
            bsmCtx["shape"] = sizes;
            bsmCtx["dtype"] = CvTypeToStr[input_img.Depth()];
        }
        static void DetectionDemo()
        {
            Service service = Service.CreateService(sdkpath);
            CheckFileExist(image1path);
            ProcessingBlock face_detector = service.CreateProcessingBlock(new Dictionary<object, object> { { "unit_type", "FACE_DETECTOR" } });

            Mat image = Cv2.ImRead(image1path, ImreadModes.Color);
            Mat input_image = new Mat();    
            Cv2.CvtColor(image, input_image, ColorConversionCodes.BGR2RGB);
            Dictionary<object, object> imgCtx = new Dictionary<object, object>();
            MatToBsm(ref imgCtx, input_image);
            Context iodata = service.CreateContext(new Dictionary<object, object> { { "image", imgCtx } });
            face_detector.Invoke(iodata);
            
            for (ulong i = 0; i < iodata["objects"].GetLength(); i++)
            {
                DrawBBox(iodata["objects"][(int)i], image, output, new Scalar(0, 255, 0));
            }
            if (window == "yes")
            {
                Cv2.ImShow("result", image);
                Cv2.WaitKey(0);
                Cv2.DestroyAllWindows();
            }

        }
        static void LandmarksDemo()
        {
            Service service = Service.CreateService(sdkpath);
            CheckFileExist(image1path);
            ProcessingBlock face_detector = service.CreateProcessingBlock(new Dictionary<object, object> { { "unit_type", "FACE_DETECTOR" } });
            ProcessingBlock mesh_fitter = service.CreateProcessingBlock(new Dictionary<object, object> { { "unit_type", "MESH_FITTER" } });

            Mat image = Cv2.ImRead(image1path, ImreadModes.Color);
            Mat input_image = new Mat();
            Cv2.CvtColor(image, input_image, ColorConversionCodes.BGR2RGB);
            Dictionary<object, object> imgCtx = new Dictionary<object, object>();
            MatToBsm(ref imgCtx, input_image);

            Context iodata = service.CreateContext(new Dictionary<object, object> { { "image", imgCtx } });
            face_detector.Invoke(iodata);

            for (ulong i = 0; i < iodata["objects"].GetLength(); i++)
            {
                mesh_fitter.Invoke(iodata["objects"][(int)i]);
            }
            for (ulong i = 0; i < iodata["objects"].GetLength(); i++)
            {
                DrawBBox(iodata["objects"][(int)i], image, output, new Scalar(0, 255, 0));
                DrawPoints(iodata["objects"][(int)i], image, output);
            }
            if (window == "yes")
            {
                Cv2.ImShow("result", image);
                Cv2.WaitKey(0);
                Cv2.DestroyAllWindows();
            }
        }
        static void RecognitionDemo()
        {
            Service service = Service.CreateService(sdkpath);
            CheckFileExist(image1path);
            CheckFileExist(image2path);
            ProcessingBlock face_detector = service.CreateProcessingBlock(new Dictionary<object, object> { { "unit_type", "FACE_DETECTOR" } });
            ProcessingBlock mesh_fitter = service.CreateProcessingBlock(new Dictionary<object, object> { { "unit_type", "MESH_FITTER" } });
            ProcessingBlock recognizer = service.CreateProcessingBlock(new Dictionary<object, object> { { "unit_type", "FACE_RECOGNIZER" } });
            ProcessingBlock matcher = service.CreateProcessingBlock(new Dictionary<object, object> { { "unit_type", "MATCHER_MODULE" } });

            Mat image_1 = Cv2.ImRead(image1path, ImreadModes.Color);
            Mat input_image_1 = new Mat();
            Cv2.CvtColor(image_1, input_image_1, ColorConversionCodes.BGR2RGB);
            Dictionary<object, object> imgCtx_1 = new Dictionary<object, object>();
            MatToBsm(ref imgCtx_1, input_image_1);

            Context iodata_1 = service.CreateContext(new Dictionary<object, object> { { "image", imgCtx_1 } });
            face_detector.Invoke(iodata_1);
            if (iodata_1["objects"].GetLength() == 0)
            {
                throw new Exception($"no face detected on {image1path}");
            }
            Context obj1 = GetObjectWithMaxConfidence(iodata_1);

            Mat image_2 = Cv2.ImRead(image2path, ImreadModes.Color);
            Mat input_image_2 = new Mat();
            Cv2.CvtColor(image_2, input_image_2, ColorConversionCodes.BGR2RGB);
            Dictionary<object, object> imgCtx_2 = new Dictionary<object, object>();
            MatToBsm(ref imgCtx_2, input_image_2);

            Context iodata_2 = service.CreateContext(new Dictionary<object, object> { { "image", imgCtx_2 } });
            face_detector.Invoke(iodata_2);
            if (iodata_2["objects"].GetLength() == 0)
            {
                throw new Exception($"no face detected on {image2path}");
            }
            Context obj2 = GetObjectWithMaxConfidence(iodata_2);
            mesh_fitter.Invoke(obj1);
            recognizer.Invoke(obj1);

            mesh_fitter.Invoke(obj2);
            recognizer.Invoke(obj2);

            Context matcherData = service.CreateContext(new Dictionary<object, object>
            {
                {
                    "verification" , new Dictionary<object, object> 
                    {
                        {
                            "objects" , new List<object>() 
                        }
                    }
                }
            });
            matcherData["verification"]["objects"].PushBack(obj1);
            matcherData["verification"]["objects"].PushBack(obj2);

            matcher.Invoke(matcherData);
            bool verdict = matcherData["verification"]["result"]["verdict"].GetBool();
            double distance = matcherData["verification"]["result"]["distance"].GetDouble();

            Scalar color = verdict ? new Scalar(0, 255, 0) : new Scalar(0, 0, 255); 
            DrawBBox(obj1, image_1, output, color);
            DrawBBox(obj2, image_2, output, color);

            Mat crop1 = GetCrop(obj1, image_1);
            Mat crop2 = GetCrop(obj2, image_2);

            Cv2.Resize(crop1, crop1, new Size(320, 480));
            Cv2.Resize(crop2, crop2, new Size(320, 480));

            Mat result = new Mat(new Size(640, 480), MatType.CV_8UC3, new Scalar(0, 0, 0));

            crop1.CopyTo(result[new Rect(0, 0, crop1.Cols, crop1.Rows)]);
            crop2.CopyTo(result[new Rect(crop1.Cols, 0, crop2.Cols, crop2.Rows)]);
            string verd = verdict ? "True" : "False";
            Console.WriteLine($"distance = {distance}\r\n");
            Console.WriteLine($"verdict = {verd}\r\n");
            if (window == "yes")
            {
                Cv2.ImShow("result", result);
                Cv2.WaitKey(0);
                Cv2.DestroyAllWindows();
            }


        }
        unsafe static void Main(string[] args)
        {

            HelpMessage();
            ParseArguments(args);

            if (mode == "detection")
            {
               DetectionDemo();
            }
            else if (mode == "landmarks")
            {
               LandmarksDemo();
            }
            else if (mode == "recognition")
            {
               RecognitionDemo();
            }
        }
        public static Dictionary<int, string> CvTypeToStr = new Dictionary<int, string>
        { 
                {MatType.CV_8U,"uint8_t"}, {MatType.CV_8S, "int8_t"}, 
                {MatType.CV_16U, "uint16_t"}, {MatType.CV_16S, "int16_t"} ,
                {MatType.CV_32S, "int32_t"}, {MatType.CV_32F, "float"}, {MatType.CV_64F, "double"} 
        };
    }
}