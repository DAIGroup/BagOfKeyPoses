/*
   Copyright (C) 2014 Alexandros Andre Chaaraoui and Francisco Flórez-Revuelta

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
using System.Xml;
using System.Collections.ObjectModel;


namespace BagOfKeyPoses
{
    /// <summary>
    /// This class saves the parameters of the learning stage of the BoKP method.
    /// </summary>
    public class LearningParams
    {
        // Const
        public const int Delta = 5;                                     // Minimum number of frames to accumulate and retest
        public const int Gamma = 10;                                    // Step size to move the window
        public const int HistoryCount = 22;                             // Number of frames to be considered for Gaussian smoothing
        public const string UNKNOWN = "unknown";                        // The unknown or null class

        // Static
        public enum ClusteringType { Kmeans, Random };                  // Possible clustering methods for key pose generation (Kmeans recommended)
        public static int[] GaussianWeights = { 20, 20, 20, 20, 20, 19, 19, 19, 18, 18, 17, 17, 16, 16, 15, 14, 14, 13, 
                                                 13, 12, 11, 11 };      // GaussianWeights[0] corresponds to the current frame t, GaussianWeights[1] to the frame t-1, etc.
                                                                        // σ = 10.486 frames (22 relevant values)
        public const double HMAX = 5000;                                // Highest evidence value possible  (Estimated HMAX = 3468.68)


        // Public
        public int InitialK = 10;                                       // Number of clusters and therefore maximum number of key poses per class
        public AssociativeArray<string, int> K;                         // Specific K for each action class
        public int FeatureSize;                                         // Feature size
        public List<string> Sources;                                    // Fused feature sources (e.g. views)
        public List<string> ClassLabels;                                // Classes used (e.g. actions)
        public bool OneClassLearning = false;                           // Use one class learning for instance for anomaly detection.
        public bool CalcKPWeightsAndSeqs = true;                        // Whether or not key pose weights and key pose sequences should be obtained
        public bool UseSummarization = false;                           // Whether or not key pose sequences should be summarized
        public bool UseSourceWeights = false;                           // Whether or not source (views) weights are employed
        public bool UseZones = false;                                   // Whether or not class zones should be used
        public AssociativeMatrix<string, string, double> SourceWeights; // Weight of each source (view) and class (action)
        public AssociativeArray<string, int> FeatureSizes;             // Feature sizes hashed by source
        public int TotalFeatureLength;                                  // The sum of features sizes
        public AssociativeArray<string, double> HThreshold;             // Per-class threshold for detection of class zones
        public int MinFrames;                                           // Minimum number of frames of a class zone
        public int MaxFrames;                                           // Maximum number of frames of a class zone
        public ClusteringType Clustering = ClusteringType.Kmeans;       // Clustering method to be employed
        public AssociativeArray<string, double> DThreshold;             // Per-class Threshold for CHAR DTW comparison (per frame) (CHAR)

        /// <summary>
        /// Returns source feature length
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public int GetFeatureLength(string source)
        {
            int length;

            if (FeatureSizes == null || !FeatureSizes.TryGetValue(source, out length))
                length = FeatureSize;

            return length;
        }

        /// <summary>
        /// Sets class specific K's always in the same order
        /// </summary>
        /// <param name="parametersValues"></param>
        public void SetK(int[] parametersValues)
        {
            int i = 0;
            K = new AssociativeArray<string, int>();

            foreach (var action in ClassLabels)
            {
                K[action] = parametersValues[i++];
            }
        }

        /// <summary>
        /// Sets the same evidence threshold for all action classes
        /// </summary>
        /// <param name="singleValue">Value expressed in 10^3</param>
        public void SetEvidThresholdsSelection(int singleValue)
        {
            HThreshold = new AssociativeArray<string, double>();

            foreach (var action in ClassLabels)
            {
                HThreshold[action] = singleValue / 1000.0;
            }
        }

        /// <summary>
        /// Sets class specific evidence thresholds's always in the same order
        /// </summary>
        /// <param name="selection">Value expressed in 10^3</param>
        public void SetEvidThresholdsSelection(int[] selection)
        {
            int i = 0;
            HThreshold = new AssociativeArray<string, double>();

            foreach (var action in ClassLabels)
            {
                HThreshold[action] = selection[i++] / 1000.0;
            }
        }

        /// <summary>
        /// Sets the same distance threshold for all action classes.
        /// </summary>
        /// <param name="singleValue">Value expressed in 10^4</param>
        public void SetDistThresholdsSelection(int singleValue)
        {
            DThreshold = new AssociativeArray<string, double>();

            foreach (var action in ClassLabels)
            {
                DThreshold[action] = singleValue / 10000.0;
            }
        }

        /// <summary>
        /// Sets class specific distance thresholds's always in the same order.
        /// </summary>
        /// <param name="selection">Value expressed in 10^4</param>
        public void SetDistThresholdsSelection(int[] selection)
        {
            int i = 0;
            DThreshold = new AssociativeArray<string, double>();

            foreach (var action in ClassLabels)
            {
                DThreshold[action] = selection[i++] / 10000.0;
            }
        }

    }

    /// <summary>
    /// This class encapsulates data relative to a specific instance of training: key poses, key pose sequences (or action zones), etc.
    /// </summary>
    public class TrainConfig 
    {  
        // Public
        public LearningParams Params;                                                   // Learning parameters
        public Dictionary<string, List<KeyPose>> KeyPoses;                              // Learned Key Poses
        public AssociativeArray<string, double> SSE;                                    // Sum of squared errors of clustering for each action class
        public Dictionary<string, List<KeyPoseSequence>> MatchedKeyPoseSequences;       // The sequences of key poses that have been matched during training (DTW)
        public AssociativeMatrix<KeyPose, KeyPose, double> KeyPoseDistanceCache;        // Cache memory of distances between key poses
        public AssociativeArray<string, List<List<double[]>>> TrainData;                // The data this learning memory has been trained with

        /// <summary>
        /// Parameterless constructor
        /// </summary>
        public TrainConfig() { }

        /// <summary>
        /// Constructor requires the initial parameters
        /// </summary>
        /// <param name="parameters"></param>
        public TrainConfig(LearningParams parameters)
        {
            MatchedKeyPoseSequences = new Dictionary<string, List<KeyPoseSequence>>();
            Params = parameters;
        }     

        /// <summary>
        /// Returns a XML document witht the matched key pose sequences and their action class information
        /// </summary>
        /// <returns></returns>
        public XmlDocument MatchedKPSeqsToXML()
        {
            XmlDocument xml = new XmlDocument();

            XmlNode root = xml.CreateElement("key-pose-sequences");
            xml.AppendChild(root);

            int sequenceCount = 0;
            foreach (var action in MatchedKeyPoseSequences.Keys)
            {
                foreach (var sequence in MatchedKeyPoseSequences[action])
                {
                    XmlDocument sequenceXML = sequence.ToXML(sequenceCount);
                    root.AppendChild(xml.ImportNode(sequenceXML.DocumentElement, true));
                    sequenceCount++;
                }
            }

            return xml; 
        }
    }
}
