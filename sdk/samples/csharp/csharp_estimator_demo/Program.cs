using CommandLine;
using CSharpApi;
using OpenCvSharp;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Data;
using static System.Net.Mime.MediaTypeNames;

namespace ApiTest
{
    internal class Program
    {
        static HersheyFonts fontFace = HersheyFonts.HersheySimplex;
        static int thickness = 1;
        static int precision = 5;
        static double fontScale = 0.3;
        static Scalar color = new Scalar(0, 255, 0);
        static List<string> allModes = new List<string>{ "all", "age", "gender", "emotion", "liveness", "mask", "glasses", "eye_openness" };
        
        static string GetAllModes(char delimiter)
        {
            string result = "";
            for (int i = 0; i < allModes.Count; i++)
            {
                result += allModes[i];

                if (i + 1 != allModes.Count)
                {
                    result += $" {delimiter} ";
                }
            }
            return result;
        }
        static void HelpMessage()
        {
            string message = $"usage: {AppDomain.CurrentDomain.FriendlyName}.exe " +
             $"[--mode {GetAllModes('|')}] \r\n" +
              " [--input_image <path to image>] \r\n" +
              " [--sdk_path ..] \r\n" +
              " [--window <yes/no>] \r\n" +
              " [--output <yes/no>] \r\n";
            Console.WriteLine(message);
        }
        static string mode = "", inputImagePath = "", image2path = "", sdkpath = "", window = "", output = "";
        static void ParseArguments(string[] args)
        {
            Parser.Default.ParseArguments<ConsoleOptions>(args).WithParsed<ConsoleOptions>(parsed =>
            {
                mode = parsed.Mode;
                inputImagePath = parsed.InputImagePath;
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
        static Mat GetCrop(Context obj, Mat image) //нужно тестирование
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
        static KeyValuePair<Point,Point> DrawBBox(Context obj, Mat img) //нужно тестирование
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
            return new KeyValuePair<Point,Point>(topLeft, bottomRight);
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

            byte[] arr = new byte[size]; //start of fix
            using (Mat temp = new Mat(input_img.Rows, input_img.Cols, input_img.Type(), arr))
            {
                input_img.CopyTo(temp);
            }
            bsmCtx["blob"] = arr; //end of fix
            List<object> sizes = new List<object>();
            for (int i = 0; i < input_img.Dims; ++i)
            {
                sizes.Add(input_img.Size(i));
            }
            sizes.Add(input_img.Channels());
            bsmCtx["shape"] = sizes;
            bsmCtx["dtype"] = CvTypeToStr[input_img.Depth()];
        }
        static string ApplyPrecision(double value)
        {
            string result = value.ToString("F" + precision);

            return result;
        }
        static string BoolToString(bool value)
        {
            return value ? "true" : "false";
        }
        static void GenderEvaluateParse(Context data, ref List<string> estimationText)
        {
            estimationText.Add("Gender: " + data["gender"].GetStr());
        }
        static void AgeEvaluateParse(Context data, ref List<string> estimationText)
        {
            estimationText.Add("Age: " + data["age"].GetLong());
        }
        static void EmotionEvaluateParse(Context data, ref List<string> estimationText)
        {
            Context emotions = data["emotions"];
            for (ulong i = 0; i < emotions.GetLength(); i++)
            {
                estimationText.Add(emotions[(int)i]["emotion"].GetStr() + ": " + ApplyPrecision(emotions[(int)i]["confidence"].GetDouble()));
            }
        }
        static void MaskEvaluateParse(Context data, ref List<string> estimationText)
        {
            estimationText.Add("Has Mask: " + BoolToString(data["has_medical_mask"]["value"].GetBool()));
        }
        static void GlassesEvaluateParse(Context data, ref List<string> estimationText)
        {
	        estimationText.Add("Has glasses: " + BoolToString(data["has_glasses"].GetBool()));

            estimationText.Add("Glasses confidence: " + ApplyPrecision(data["glasses_confidence"].GetDouble()));
        }
        static void EyeOpennessEvaluateParse(Context data, ref List<string> estimationText)
        {
	        Context is_left_eye_open = data["is_left_eye_open"];
	        Context is_right_eye_open = data["is_right_eye_open"];

	        estimationText.Add("Is left eye open: " + BoolToString(is_left_eye_open["value"].GetBool()));

	        estimationText.Add("Is right eye open: " + BoolToString(is_right_eye_open["value"].GetBool()));

	        estimationText.Add("Left eye openness: " + ApplyPrecision(is_left_eye_open["confidence"].GetDouble()));

	        estimationText.Add("Right eye openness: " + ApplyPrecision(is_right_eye_open["confidence"].GetDouble()));
        }
        static void LivenessEvaluateParse(Context data, ref List<string> estimationText)
        {
            Context liveness = data["liveness"];
            estimationText.Add("Liveness confidence: " + ApplyPrecision(liveness["confidence"].GetDouble()));
            estimationText.Add("Liveness value: " + liveness["value"].GetStr());
        }
        static void ParseAll(Context data, ref List<string> estimationText)
        {
            AgeEvaluateParse(data, ref estimationText);
            GenderEvaluateParse(data, ref estimationText);
            EmotionEvaluateParse(data, ref estimationText);
            MaskEvaluateParse(data, ref estimationText);
            GlassesEvaluateParse(data, ref estimationText);
            EyeOpennessEvaluateParse(data, ref estimationText);
            LivenessEvaluateParse(data, ref estimationText);
        }
        static Context ImageToSDKForm(Mat image, Service service)   
        {
            Dictionary<object, object> data = new Dictionary<object, object>();

            MatToBsm(ref data, image);
            return service.CreateContext(new Dictionary<object, object> { { "image" , data } });

        }
        static void EstimatorRealisation(string unit_type, Context outputContext, Service service)
        {
                service.CreateProcessingBlock(new Dictionary<object, object> { { "unit_type", unit_type } }).Invoke(outputContext);            
        }
        static void Estimate(Service service, string mode, Context data)
        {
            mode = mode.ToUpper() + "_ESTIMATOR";
            EstimatorRealisation(mode, data, service);
        }
        static void ModeParser (string key, Context data, ref List<string> estimationText)
        {
            switch (key)
            {
                case "all": ParseAll(data, ref estimationText); break;
                case "age": AgeEvaluateParse(data, ref estimationText); break;
                case "gender": GenderEvaluateParse(data, ref estimationText); break;
                case "emotion": EmotionEvaluateParse(data, ref estimationText); break;
                case "liveness": LivenessEvaluateParse(data, ref estimationText); break;
                case "mask": MaskEvaluateParse(data, ref estimationText); break;
                case "glasses": GlassesEvaluateParse(data, ref estimationText); break;
                case "eye_openness": EyeOpennessEvaluateParse(data, ref estimationText); break;
                default: throw new Exception($"wrong key {key}");
            }
        }
        static void DetectorFitterSample()
        {
            
            Service service = Service.CreateService(sdkpath);
            Mat image = Cv2.ImRead(inputImagePath);
            Mat inputImage = new Mat();

            Cv2.CvtColor(image, inputImage, ColorConversionCodes.BGR2RGB);
            Context data = ImageToSDKForm(inputImage, service);

            EstimatorRealisation("FACE_DETECTOR", data, service);
            if (mode == "eye_openness" || mode == "all")
            {
                EstimatorRealisation("MESH_FITTER", data, service);
                if (mode == "all")
                {
                    foreach (var value in allModes)
                    {
                        if (value == "all")
                        {
                            continue;
                        }

                        Estimate(service, value, data);
                    }
                }
            }

            if (mode != "all")
            {
                Estimate(service, mode, data);
            }

            List<List<string>> estimationText = new List<List<string>>();
            List<KeyValuePair<Point, Point>> bboxes = new List<KeyValuePair<Point, Point>>();
            int maxWidth = 0;
            int maxHeight = 0;

            for (ulong i = 0; i < data["objects"].GetLength(); i++)
            {
                if (data["objects"][(int)i]["class"].GetStr() != "face")
                {
                    continue;   
                }

                List<string> tem = new List<string>();
                int height = 0;
                
                bboxes.Add(DrawBBox(data["objects"][(int)i], image));

                ModeParser(mode,data["objects"][(int)i], ref tem);

                foreach (var text in tem)
                {
                    int baseline = 0;
                    Size textSize = Cv2.GetTextSize(text, fontFace,fontScale, thickness, out baseline);

                    height += textSize.Height;

                    maxWidth = Math.Max(maxWidth, textSize.Width); 
                }
                maxHeight = Math.Max(maxHeight, height);

                estimationText.Add(tem);
                
            }
            Cv2.CopyMakeBorder(image, image, maxHeight, maxHeight, maxWidth, maxWidth, BorderTypes.Constant);

            for (int i = 0; i < estimationText.Count; i++)
            {
                int previousHeight = 0;
                Point first = bboxes[i].Key;
                Point second = bboxes[i].Value;

                Console.WriteLine($"BBox coordinates: ({first.X} , {first.Y}), ({second.X}, {second.Y})");

                foreach (var text in estimationText[i])
                {
                    Console.WriteLine(text);

                    if (window == "yes")
                    {
                        const int offset = 2;

                        Cv2.PutText(
                            image,
                            text,
                            new Point(second.X + maxWidth, first.Y + maxHeight + previousHeight),
                            fontFace,
                            fontScale,
                            color,
                            thickness
                            );
                        int baseline = 0;
                        previousHeight += offset + Cv2.GetTextSize(text, fontFace, fontScale, thickness, out baseline).Height;
                    }
                }
            }
            if (window == "yes")
            {
                Cv2.ImShow("result", image);
                Cv2.WaitKey();
                Cv2.DestroyWindow("result");
            }
        }
        static void CheckMode(string mode)
        {
            if (!allModes.Exists(p => p == mode))
            {
                throw new Exception($"--mode should be {GetAllModes('|')}");
            }
        }
     
        unsafe static void Main(string[] args)
        {
            HelpMessage();
            ParseArguments(args);
            CheckFileExist(inputImagePath);
            CheckMode(mode);
            DetectorFitterSample();
        }
        public static Dictionary<int, string> CvTypeToStr = new Dictionary<int, string>
        { 
                {MatType.CV_8U,"uint8_t"}, {MatType.CV_8S, "int8_t"}, 
                {MatType.CV_16U, "uint16_t"}, {MatType.CV_16S, "int16_t"} ,
                {MatType.CV_32S, "int32_t"}, {MatType.CV_32F, "float"}, {MatType.CV_64F, "double"} 
        };
    }
}
