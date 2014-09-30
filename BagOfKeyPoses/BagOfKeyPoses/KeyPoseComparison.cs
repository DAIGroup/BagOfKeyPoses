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

namespace BagOfKeyPoses
{
    /// <summary>
    /// Implements the comparison of key poses for DTW recognition purposes.
    /// </summary>
    public class KeyPoseComparison : DTWComparison<KeyPose>
    {
        // Static
        private static readonly object LockCache = new object();        // Thread-safe cache access

        // Private
        private string sequenceClassLabel;                                  // The label of the sequence that is currently being compared (ground-truth)
        private TrainConfig config;                                     // Used configuration

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="config"></param>
        /// <param name="sequenceAction"></param>
        public KeyPoseComparison(TrainConfig config, string sequenceAction)
        {
            this.config = config;
            this.sequenceClassLabel = sequenceAction;
        }

        /// <summary>
        /// Returns the distance between two keyposes based on its feature 'distances' and on the relative importance of the key poses
        /// To be used at the recognition stage (with learned weights)
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public double Distance(KeyPose a, KeyPose b)
        {
            double distance = 0.0;
            bool loaded = false;

            if (config.KeyPoseDistanceCache != null && (config.KeyPoseDistanceCache.TryGetValue(a, b, out distance) || config.KeyPoseDistanceCache.TryGetValue(b, a, out distance)))
                loaded = true;
            
            if (!loaded)
            {
                if (a != b)
                {
                    // Distance between key pose features
                    distance = FeatureDistance(a.Distances, b.Distances);
                }

                // Save cache
                if (config.KeyPoseDistanceCache != null)
                    lock(LockCache) // Only writing is protected, we do not care about a few extra calculations or overridings
                        config.KeyPoseDistanceCache[a, b] = distance;
            }

            return distance;
        }

        /// <summary>
        /// Returns the correlation between two key poses based on its feature 'distances' and on the relative importance of the key poses
        /// To be used at the recognition stage (with learned weights)
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public double Correlation(KeyPose a, KeyPose b)
        {
            double corr = 0.0;
            bool loaded = false;

            if (config.KeyPoseDistanceCache != null && (config.KeyPoseDistanceCache.TryGetValue(a, b, out corr) || config.KeyPoseDistanceCache.TryGetValue(b, a, out corr)))
                loaded = true;

            if (!loaded)
            {
                if (a != b) // Distance between key pose features
                    corr = FeatureCorrelation(a.Distances, b.Distances);

                // Save cache
                if (config.KeyPoseDistanceCache != null)
                    lock(LockCache)
                        config.KeyPoseDistanceCache[a, b] = corr;
            }

            return corr;
        }

        /// <summary>
        /// Returns the distance between two keypose distances
        /// To be used at the recognition stage (with learned source weights for feature fusion)
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        private double FeatureDistance(double[] a, double[] b)
        {
            double featureDistance = 0.0;

            // Distance between key pose features
            if (config.Params.UseSourceWeights)
            {
                // Apply source weights
                int pos = 0;

                foreach (string source in config.Params.Sources) // For each source
                {
                    // Obtain local distance
                    int L = config.Params.GetFeatureLength(source);
                    double dist = 0;

                    for (int i = 0; i < L; ++i) // For each feature part
                    {
                        dist += Math.Abs(a[pos] - b[pos]);
                        pos++;
                    }

                    // Normalize and apply weight (we suppose that this is a match in order to apply the right weight if it actually were)
                    featureDistance += dist / L * config.Params.SourceWeights[source, sequenceClassLabel];
                }
            }
            else
                featureDistance = Functions.ManhattanDistanceNormalized(a, b);

            return featureDistance;
        }

        /// <summary>
        /// Returns the correlation between two key pose distances
        /// To be used at the recognition stage (with learned source weights for feature fusion)
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        private double FeatureCorrelation(double[] a, double[] b)
        {
            double featureCorrelation = 0.0;

            // Distance between key pose features
            if (config.Params.UseSourceWeights)
            {
                // Apply camera weights
                int pos = 0;

                foreach (string cam in config.Params.Sources) // For each source
                {
                    // Obtain local distance
                    int L = config.Params.GetFeatureLength(cam);
                    double corr = 0;
                    double[] sub_a = new double[L];
                    double[] sub_b = new double[L];

                    Array.Copy(a, pos, sub_a, 0, L);
                    Array.Copy(b, pos, sub_b, 0, L);
                    corr = Functions.Correlation(sub_a, sub_b);
                    
                    pos += L;

                    // Apply weight (we suppose that this is a match in order to apply the right weight if it actually were)
                    featureCorrelation += corr * config.Params.SourceWeights[cam, sequenceClassLabel];
                }
            }
            else
                featureCorrelation = Functions.Correlation(a, b);

            return featureCorrelation;
        }
    }
}
