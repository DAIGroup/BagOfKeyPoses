/*
   Copyright (C) 2014 Alexandros Andre Chaaraoui, Francisco Flórez-Revuelta and Pau Climent-Pérez

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
using System.Drawing;
using System.ComponentModel;
using System.IO;
using System.Globalization;
using Util;
using TrainDataType = System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<System.Collections.Generic.List<double[]>>>;

namespace BagOfKeyPoses
{
    /// <summary>
    /// This class provides an algorithm, which, by means of k-means, computes the most discriminative Key Poses for each action class.
    /// </summary>
    public static class KeyPoseWeighter
    {
        public const int MIN_FRAMES = 5;                // Minimum frames for action zones
        public const int KMeansIterations = 6;          // Number of times Kmeans is executed (best run is chosen)

        /// <summary>
        /// Learning algorithm that computes K cluster means (or less) per action and obtains their weight, 
        /// as more discriminative for that class, as higher is the weight.
        /// Executes k-means 6 times and keeps the result with the best compactness
        /// </summary>
        /// <param name="sequenceGroups">Sequences of features hashed per class</param>
        /// <returns></returns>
        public static Dictionary<string, List<KeyPose>> DoLearn(TrainDataType sequenceGroups, TrainConfig config,
            BackgroundWorker bgw = null, bool calcWeights = true, AssociativeArray<string, List<double[]>> keyPosesMemory = null)
        {
            var keyPoses = new Dictionary<string, List<KeyPose>>();

            //
            // Obtain clusters for each action
            //
            int count = 0, total = sequenceGroups.Count;
            config.SSE = new AssociativeArray<string, double>();

            foreach (KeyValuePair<string, List<List<double[]>>> sequences in sequenceGroups) // For each action
            {
                string action = sequences.Key;

                if (bgw != null)
                    bgw.ReportProgress((int)((double)count / (double)total * 59 + 10), "Obtaining key poses for " + action + " (" + count++ + "/" + total + ")");
                
                if (keyPosesMemory != null && keyPosesMemory.ContainsKey(action)) // This action has already been learned
                {
                    // Recover the key poses
                    keyPoses.Add(action, new List<KeyPose>());

                    foreach (var distances in keyPosesMemory[action])
                        keyPoses[action].Add(new KeyPose(action, distances));
                }
                else // New action to be learned
                {
                    // Cluster the samples in K clusters
                    List<double[]> group = new List<double[]>();

                    foreach (List<double[]> sequence in sequences.Value)
                        group.AddRange(sequence);
                    
                    List<double[]> bestCenters = null;

                    if (config.Params.Clustering == LearningParams.ClusteringType.Kmeans)
                    {
                        // K-Means Clustering
                        KMeans kmeans = new KMeans();
                        bestCenters = kmeans.Cluster(group, config.Params.K == null ? config.Params.InitialK : config.Params.K[action], KMeansIterations);
                        config.SSE[action] = kmeans.SSE;
                    }
                    else if (config.Params.Clustering == LearningParams.ClusteringType.Random)
                    {
                        // Random initialisation of Kmeans
                        KMeans kmeans = new KMeans();
                        var centers = kmeans.PickRandomCenters(config.Params.K == null ? config.Params.InitialK : config.Params.K[action], group);
                        bestCenters = centers.Values.ToList();

                        // Establish final SSE
                        var assignments = kmeans.ParallelGetClusterAssignments(group, centers);
                        config.SSE[action] = Functions.SumOfSquaredErrors(centers, assignments);
                    }

                    // Take cluster centers as key poses
                    for (int i = 0; i < bestCenters.Count; ++i)
                    {
                        // Create new key pose
                        KeyPose kp = new KeyPose(action, bestCenters[i]);

                        if (keyPoses.ContainsKey(action))
                            keyPoses[action].Add(kp);
                        else
                        {
                            keyPoses.Add(action, new List<KeyPose>());
                            keyPoses[action].Add(kp);
                        }

                        // Save it also in the memory
                        if (keyPosesMemory != null)
                            keyPosesMemory[action].Add(kp.Distances);
                    }
                }
            }

            //
            // Assign weights
            //
            if (calcWeights)
            {
                WithinClassWeighting(sequenceGroups, keyPoses, config, bgw);
            }

            return keyPoses;
        }

        /// <summary>
        /// Uses the training data in order to obtain the weights of the key poses
        /// As more discriminative a key pose is, a higher weight it receives.
        /// Weight = within-class assignments rate
        /// MatchedSequences (for DTW) are obtained too.
        /// </summary>
        /// <param name="sequenceGroups"></param>
        /// <param name="keyPoses"></param>
        /// <param name="config"></param>
        public static void WithinClassWeighting(TrainDataType sequenceGroups, Dictionary<string, List<KeyPose>> keyPoses, TrainConfig config, BackgroundWorker bgw = null)
        {
            //
            // Assign weights
            //
            int count = 0, total = sequenceGroups.Count;

            foreach (KeyValuePair<string, List<List<double[]>>> sequences in sequenceGroups) // For each action
            {
                string action = sequences.Key;

                if (bgw != null)
                    bgw.ReportProgress((int)((double)count / (double)total * 30 + 69), "Weighting key poses of " + action + " (" + count++ + "/" + total + ")");

                if (action != "unknown" && !config.MatchedKeyPoseSequences.ContainsKey(action))
                    config.MatchedKeyPoseSequences.Add(action, new List<KeyPoseSequence>());

                foreach (List<double[]> sequence in sequences.Value) // For each sequence
                {
                    KeyPoseSequence matchedSequence = new KeyPoseSequence();
                    KeyPose lastKP = null;

                    foreach (double[] feature in sequence) // For each frame
                    {
                        KeyPose closestKP = KeyPose.ClosestAmongAll(feature, keyPoses, config, true);

                        if (closestKP.ClassLabel.Equals(sequences.Key))
                            closestKP.WithinClass++;
                        else
                            closestKP.OutOfClass++;

                        if (config.Params.UseSummarization) // Summarize the sequences avoiding repeated key poses
                        {
                            if (lastKP != closestKP) // Pointer comparison is valid
                                matchedSequence.Items.Add(closestKP);

                            lastKP = closestKP;
                        }
                        else matchedSequence.Items.Add(closestKP);
                    }

                    // Save the matched sequence
                    if (action != "unknown")
                    {
                        matchedSequence.ClassLabel = action;
                        config.MatchedKeyPoseSequences[action].Add(matchedSequence);
                    }
                }
            }

            //
            // Calc weights 
            //
            if (bgw != null)
                bgw.ReportProgress(99, "\nNormalizing weights...");

            foreach (KeyValuePair<string, List<KeyPose>> clusters in keyPoses) // For each action
            {
                foreach (KeyPose kp in clusters.Value) // For each cluster
                {
                    int sum = kp.WithinClass + kp.OutOfClass;

                    if (sum != 0)
                        kp.Weight = (double)kp.WithinClass / (double)sum;
                }
            }
        }

        /// <summary>
        /// Learns the parts of the sequences which present a high class evidence value ("action zones")
        /// </summary>
        /// <typeparam name="S">Training data single or multi-view</typeparam>
        /// <param name="trainData">Training data</param>
        /// <param name="parameters">Parameters</param>
        /// <param name="groundTruthActions">Ground truth labels</param>
        public static bool LearnZones(TrainDataType trainData, TrainConfig config)
        {
            bool error = true;
            var zones = new AssociativeArray<string, List<KeyPoseSequence>>();

            /*
            TextWriter output = new StreamWriter("action_zones.txt");
            output.Write("Action Class; Action Zone");
            foreach (var key in Config.Params.Actions)
                output.Write("; " + key + " (r)" + "; " + key);
            output.Write("\n");
            */

            foreach (var classData in trainData)
            {
                string classLabel = classData.Key;
                int selfAdded = 0, total = 0;

                foreach (var trainSequence in classData.Value)
                {
                    bool zoneFound = false;
                    var rHistory = new Dictionary<string, FixedSizedQueue<double>>();
                    var zone = new KeyPoseSequence();
                    var fullSequence = new KeyPoseSequence();

                    // Process the sequence
                    foreach (var feature in trainSequence)
                    {
                        // Find nearest neighbour key poses
                        var currentKeyPoses = KeyPose.ClosestAmong(feature, config.KeyPoses, config);
                        var currentKeyPose = currentKeyPoses.Dictionary.Aggregate((l, r) => l.Value.MatchedDistance < r.Value.MatchedDistance ? l : r).Value;
                        fullSequence.Items.Add(currentKeyPose);
                        fullSequence.ClassLabel = classLabel;

                        // Normal evidence processing
                        var hSmoothed = ProcessClassEvidence(currentKeyPoses, rHistory, config.Params);

                        //
                        // Compare the class r with the median value in order to detect the zone
                        // (doesn't have to be the highest, there could be multiple high ones)
                        //
                        double hClass = hSmoothed[classLabel];
                        var values = new List<double>(hSmoothed.Dictionary.Values);
                        values.Remove(hClass);     // We exclude the (expected to be) high value of the class
                        double hMedian = Functions.Median(values);

                        //if(sample == 0)
                        //    output.Write(action + "; ");

                        if (hClass > hMedian + config.Params.HThreshold[classLabel])   // Zone found
                        {
                            zone.Items.Add(currentKeyPose);

                            if (zone.Items.Count >= config.Params.MaxFrames) // The biggest it can get
                            {
                                zoneFound = true;
                                zone.ClassLabel = classLabel;
                                zones[classLabel].Add(zone);
                                zone = new KeyPoseSequence();
                            }

                            //if (sample == 0)
                            //    output.Write(rAction);
                        }
                        else if(zone.Items.Count > 0) // No action zone here, but collect existing one
                        {
                            if (zone.Items.Count > config.Params.MinFrames) // ActionZone ended, sequence to collect
                            {
                                zoneFound = true;
                                zone.ClassLabel = classLabel;
                                zones[classLabel].Add(zone);
                                zone = new KeyPoseSequence();
                            }
                            else // To small, ignore it
                                zone = new KeyPoseSequence();

                            //if (sample == 0)
                            //    output.Write("0");
                        }

                        /*
                        if (sample == 0)
                        {                            
                            foreach (var key in Config.Params.Actions)
                            {
                                if (rHistory.ContainsKey(key))
                                    output.Write("; " + rHistory[key].ElementAt(rHistory[key].Count - 1) + "; " + rSmoothed[key]);
                            }

                            output.Write("\n");
                        }
                        */
                    }

                    if (zone.Items.Count > config.Params.MinFrames) // ActionZone ended, sequence to collect
                    {
                        zone.ClassLabel = classLabel;
                        zones[classLabel].Add(zone);
                    }
                    else if (!zoneFound)
                    {
                        // Keep 1/3 of the fullSequence using the center of it
                        //int oneThird = fullSequence.Items.Count / 3;
                        //if(fullSequence.Items.Count > MIN_FRAMES) // At least 5 frames
                        //    fullSequence.Items = fullSequence.Items.Skip(oneThird).Take(oneThird).ToList();
                        //zones[classLabel].Add(fullSequence);

                        // Keep the whole key pose sequence (this is better for gestures since these are short)
                        zones[classLabel].Add(fullSequence);

                        selfAdded++;
                    }

                    total++;
                }

                //if (selfAdded > 0) // Activate to check if there are really class-evidence-based action zones
                //    Console.WriteLine("Self added zones: " + selfAdded + "/" + total + "(RT=" + config.Params.HThreshold[classLabel] + ")");
            }

            if (zones.Total > 0)
            {
                // Set action zones as matchedKeyPoseSequences (no difference from now on...)
                config.MatchedKeyPoseSequences = zones.Dictionary;
                error = false;
            }

            //output.Close();

            return error;
        }

        /// <summary>
        /// Updates the current weighting scheme by adding the processed r value considering Gaussian smoothing, normalization
        /// and exponentialization.
        /// </summary>
        /// <param name="nearestNeighbours">The class-wise best matches of the current key pose</param>
        /// <param name="hHistory">The history of r values (for smoothing)</param>
        /// <param name="weightingScheme">The updated weights will be returned</param>
        /// <returns>Current evidence values</returns>
        public static AssociativeArray<string, double> ProcessClassEvidence(AssociativeArray<string, KeyPose> nearestNeighbours,
            Dictionary<string, FixedSizedQueue<double>> hHistory, LearningParams parameters)
        {
            //
            // Track the H evidence values of all classes
            //
            var hSmoothed = new AssociativeArray<string, double>();

            foreach (var key in parameters.ClassLabels)
            {
                if (!hHistory.ContainsKey(key))
                    hHistory.Add(key, new FixedSizedQueue<double>(LearningParams.HistoryCount));

                // Update the history with the current r
                if (nearestNeighbours[key].MatchedDistance != 0)
                {
                    // W/D normalized
                    double h = nearestNeighbours[key].Weight / nearestNeighbours[key].MatchedDistance;
                    hHistory[key].Enqueue(h / LearningParams.HMAX);
                }
                else // Happens sparingly (key pose == pose)
                {
                    if (hHistory[key].Count > 0)    // Reply the last value if possible
                        hHistory[key].Enqueue(hHistory[key].ElementAt(hHistory[key].Count - 1));
                    else
                        hHistory[key].Enqueue(0.1); // Simulated "pretty good" match
                }

                //
                // Smooth the h values (with Gaussian)
                //
                int totalWeights = 0;
                double smoothed = 0.0;
                for (int i = 0; i < hHistory[key].Count; ++i)
                {
                    smoothed += LearningParams.GaussianWeights[i] * hHistory[key].ElementAt(hHistory[key].Count - i - 1);
                    totalWeights += LearningParams.GaussianWeights[i];
                }

                // Normalize and save
                hSmoothed[key] = smoothed / totalWeights;

                //
                // Attenuate the h values (In order to capture the peak of the exp)
                //
                hSmoothed[key] = Math.Exp(10 * hSmoothed[key]);
            }

            return hSmoothed;
        }

        /// <summary>
        /// Converts a Matrix.Data row into an int array
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        private static double[] MatrixRowToIntArray(int rowIndex, float[,] data, int S)
        {
            double[] array = new double[S];
            
            for (int i = 0; i < array.Length; ++i)
                array[i] = (double)data[rowIndex, i];

            return array;
        }

        /// <summary>
        /// Converts a Matrix.Data row into an int array
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        private static double[] MatrixRowToIntArray(int rowIndex, double[,] data, int S)
        {
            double[] array = new double[S];

            for (int i = 0; i < array.Length; ++i)
                array[i] = (double)data[rowIndex, i];

            return array;
        }
    }
}
