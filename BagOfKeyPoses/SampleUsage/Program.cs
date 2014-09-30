/*
   Copyright (C) 2014 Alexandros Andre Chaaraoui

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Util;
using BagOfKeyPoses;
using TrainDataType = Util.AssociativeArray<string, System.Collections.Generic.List<System.Collections.Generic.List<double[]>>>;

namespace SampleUsage
{
    class Program
    {
        static void Main(string[] args)
        {
            // Run usage samples.
            sequenceBasedSimpleSample();
            sequenceBasedAdvancedSample();
            continuousRecognitionSample();
            Console.ReadKey();
        }

        /// <summary>
        /// In this simple usage sample of the Bag of Key Poses learning method, a set of training and a set of testing sequences are used to apply 
        /// template-based matching based on the previously learned model and Dynamic-time-warping sequence alignment.
        /// </summary>
        private static void sequenceBasedSimpleSample()
        {
            // Prepare learning parameters.
            LearningParams learning_params = new LearningParams();
            List<string> class_labels = new List<string>();
            class_labels.Add("red");
            class_labels.Add("green");
            class_labels.Add("blue");
            learning_params.ClassLabels = class_labels;
            learning_params.Clustering = LearningParams.ClusteringType.Kmeans;
            // K, the number of key poses per class, depends on the number of samples and desired generality, normally between 10 and 100 
            // for thousands to tens of thousands of samples.
            learning_params.InitialK = 5;
            learning_params.FeatureSize = 3;

            // Prepare train data.
            double[] red_feature = new double[] { 1, 0, 0 };
            double[] green_feature = new double[] { 0, 1, 0 };
            double[] blue_feature = new double[] { 0, 0, 1 };
            List<double[]> red_sequence = new List<double[]>();
            List<double[]> green_sequence = new List<double[]>();
            List<double[]> blue_sequence = new List<double[]>();
            for (int i = 0; i < 100; ++i) 
            {
                red_sequence.Add(red_feature);
                green_sequence.Add(green_feature);
                blue_sequence.Add(blue_feature);
            }

            TrainDataType train_data = new TrainDataType();
            train_data["red"].Add(red_sequence);
            train_data["green"].Add(green_sequence);
            train_data["blue"].Add(blue_sequence);

            // Train.
            BoKP bokp = new BoKP(learning_params);
            bokp.Train(train_data.Dictionary);
            
            // Prepare test sequence.
            double[] orange_feature = new double[] { 0.9, 0.3, 0.1 };
            List<double[]> orange_sequence = new List<double[]>();
            for (int i = 0; i < 100; ++i) orange_sequence.Add(orange_feature);

            // Test.
            string recognition = bokp.EvaluateSequence(orange_sequence);

            // Print result.
            Console.WriteLine("The test sequence has been recognized as " + recognition);
        }

        /// <summary>
        /// This sample adds feature fusion and source weights to the previous one.
        /// </summary>
        private static void sequenceBasedAdvancedSample()
        {
            // Prepare learning parameters.
            LearningParams learning_params = new LearningParams();
            List<string> class_labels = new List<string>();
            class_labels.Add("red");
            class_labels.Add("green");
            class_labels.Add("blue");
            learning_params.ClassLabels = class_labels;
            learning_params.Clustering = LearningParams.ClusteringType.Kmeans;
            // K depends on the number of samples and desired generality, normally between 10 and 100 for thousands to tens of thousands of samples.
            learning_params.InitialK = 5;
            List<string> sources = new List<string>();
            sources.Add("source1");
            sources.Add("source2");
            learning_params.Sources = sources;
            AssociativeArray<string, int> feature_sizes = new AssociativeArray<string, int>();
            feature_sizes["source1"] = 3;
            feature_sizes["source2"] = 3;
            learning_params.FeatureSizes = feature_sizes;

            // Weights can be learned in different ways, an easy option is to apply a test for each of the sources and use the success rate of each
            // class normalized across data sources. For instance, if camera 1 obtains 80% for class 1 and 70% for class 2, and camera 2 obtains 60% 
            // for class 1 and 70% for class 2, we would assign respectively 80/140 and 70/140 to camera 1 and 60/140 and 70/140 to camera 2.
            //
            // For greater detail, see weighted feature fusion scheme at Chaaraoui, A. A., Padilla-López, J. R., Ferrández-Pastor, F. J., 
            // Nieto-Hidalgo, M., & Flórez-Revuelta, F. (2014). A Vision-Based System for Intelligent Monitoring: Human Behaviour Analysis and 
            // Privacy by Context. Sensors, 14(5), 8895-8925.
            var source_weights = new AssociativeMatrix<string, string, double>();
            source_weights["source1", "red"] = 0.8;
            source_weights["source1", "green"] = 0.5;
            source_weights["source1", "blue"] = 0;
            source_weights["source2", "red"] = 0.2;
            source_weights["source2", "green"] = 0.5;
            source_weights["source2", "blue"] = 1;
            learning_params.SourceWeights = source_weights;
            learning_params.UseSourceWeights = true;

            // Prepare train data (features are obtained from two sources and concatenated).
            double[] red_feature = new double[] { 1, 0, 0, 1, 0, 0 };
            double[] green_feature = new double[] { 0, 1, 0, 0, 1, 0 };
            double[] blue_feature = new double[] { 0, 0, 1, 0, 0, 1 };
            List<double[]> red_sequence = new List<double[]>();
            List<double[]> green_sequence = new List<double[]>();
            List<double[]> blue_sequence = new List<double[]>();
            for (int i = 0; i < 100; ++i)
            {
                red_sequence.Add(red_feature);
                green_sequence.Add(green_feature);
                blue_sequence.Add(blue_feature);
            }

            TrainDataType train_data = new TrainDataType();
            train_data["red"].Add(red_sequence);
            train_data["green"].Add(green_sequence);
            train_data["blue"].Add(blue_sequence);

            // Train.
            BoKP bokp = new BoKP(learning_params);
            bokp.Train(train_data.Dictionary);

            // Prepare test sequence.
            double[] orange_feature = new double[] { 0.9, 0.3, 0.9, 0.3, 0.3, 0.1 };
            List<double[]> orange_sequence = new List<double[]>();
            for (int i = 0; i < 100; ++i) orange_sequence.Add(orange_feature);

            // Test.
            string recognition = bokp.EvaluateSequence(orange_sequence);

            // Print result.
            Console.WriteLine("The test sequence has been recognized as " + recognition);
        }

        /// <summary>
        /// This sample shows how continuous recognition is performed. Note that this type of recognition is based
        /// on segment analysis (instead of frame-by-frame or sequence-based recognition) and a sliding window technique.
        /// (Learning doesn't change, but additional parameters are set.)
        /// </summary>
        private static void continuousRecognitionSample()
        {
            // Prepare learning parameters.
            LearningParams learning_params = new LearningParams();
            List<string> class_labels = new List<string>();
            class_labels.Add("red");
            class_labels.Add("green");
            class_labels.Add("blue");
            learning_params.ClassLabels = class_labels;
            learning_params.Clustering = LearningParams.ClusteringType.Kmeans;
            // K depends on the number of samples and desired generality, normally between 10 and 100 for thousands to tens of thousands of samples.
            learning_params.InitialK = 5;
            learning_params.FeatureSize = 3;
            // Minimum distance threshold that has to be reached by sequence alimgnet to consider the segment to be the matched class.
            // Typically in the order of hundreds (Note that scale 10^4 is employed).
            learning_params.SetDistThresholdsSelection(400);

            // Minimum class evidence value that has to be reached to obtain a class zone. Typically in the order of hundreds (Note that
            // a scale of 10^3 is employed).
            learning_params.SetEvidThresholdsSelection(300);
            learning_params.UseZones = true;

            // Sliding window
            learning_params.MinFrames = 5;
            learning_params.MaxFrames = 35;

            // Prepare train data.
            double[] red_feature = new double[] { 1, 0, 0 };
            double[] green_feature = new double[] { 0, 1, 0 };
            double[] blue_feature = new double[] { 0, 0, 1 };
            List<double[]> red_sequence = new List<double[]>();
            List<double[]> green_sequence = new List<double[]>();
            List<double[]> blue_sequence = new List<double[]>();
            for (int i = 0; i < 100; ++i)
            {
                red_sequence.Add(red_feature);
                green_sequence.Add(green_feature);
                blue_sequence.Add(blue_feature);
            }

            TrainDataType train_data = new TrainDataType();
            train_data["red"].Add(red_sequence);
            train_data["green"].Add(green_sequence);
            train_data["blue"].Add(blue_sequence);

            // Train.
            BoKP bokp = new BoKP(learning_params);
            bokp.Train(train_data.Dictionary);

            // Prepare test sequence.
            double[] orange_feature = new double[] { 0.9, 0.3, 0.1 };
            List<double[]> orange_sequence = new List<double[]>();
            for (int i = 0; i < 100; ++i) orange_sequence.Add(orange_feature);

            // Test.
            List<string> recognition = bokp.EvaluateCHARSequence(orange_sequence);

            // Print result.
            Console.WriteLine("The test sequence has been recognized as: ");
            foreach (string frame in recognition) Console.WriteLine(frame);
        }
    }
}
