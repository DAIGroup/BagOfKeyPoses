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
using System.Drawing;
using System.Diagnostics;
using System.Threading.Tasks;

namespace Util
{
    /// <summary>
    /// This class implements a typical K-Means clustering with Random initialization (MarsenneTwister) and using newCenters ~ oldCenters as
    /// termination criteria. Supports missing dimensions (indicated by 0 value) and uses Euclidean distance.
    /// </summary>
    public class KMeans
    {
        // Static
        private const double EPSILON = 1e-6; //1e-36 for Squared distances      // Cluster centers with this distance or less are considered equal (taken from OpenCV)
        private const int ITERATIONS = 1000;                                    // Maximum number of iterations
        private const int INIT_ITER = 100;                                      // Maximum number of initialization iterations

        // Public
        public double SSE;                                                      // The sum of squared errors of the best clustering

        // Private
        private MersenneTwister random;                                         // Random numbers generator
        private AssociativeArray<int, List<double[]>> clusterAssignments;       // Current cluster assignments (shared index)
        private Dictionary<int, double[]> centers;                              // Current cluster centers (shared index)
        

        /// <summary>
        /// Creates a new object of this class
        /// </summary>
        public KMeans()
        {
            random = new MersenneTwister();
        }

        /// <summary>
        /// Will cluster the given data into K groups until no new centers are found.
        /// Note: It could happen that less than K groups are returned if a cluster center doesn't have any samples. 
        /// (Typically that cluster is dismissed as no cluster center can be computed without samples)
        /// </summary>
        /// <param name="samples"></param>
        /// <param name="K"></param>
        /// <param name="attempts">Number of times K-Means is executed. (Should be 50-1000 for small Ks, less for high Ks)</param>
        public List<double[]> Cluster(List<double[]> samples, int K, int attempts)
        {
            if (K >= samples.Count) // It doesn't make sense
                return samples;

            List<double[]> bestCenters = null;
            double error = double.MaxValue;

            for (int i = 0; i < attempts; ++i)
            {
                Dictionary<int, double[]> randomCenters = PickRandomCenters(K, samples);
                processGroups(samples, randomCenters);

                bool better;
                double local = clusteringCompactnessError(centers, clusterAssignments, error, out better);

                if (better)
                {
                    bestCenters = centers.Values.ToList();
                    error = local;
                    SSE = Functions.SumOfSquaredErrors(centers, clusterAssignments);
                }
            }

            return bestCenters;
        }

        /// <summary>
        /// Returns K randomly chosen cluster centers of the given samples
        /// </summary>
        /// <param name="K"></param>
        /// <param name="samples"></param>
        /// <returns></returns>
        public Dictionary<int, double[]> PickRandomCenters(int K, List<double[]> samples)
        {
            var randomCenters = new Dictionary<int, double[]>();
            int pickedCenters = 0, it = 0;

            while (pickedCenters < K && it++ < INIT_ITER)
            {
                int rand = random.Next(0, samples.Count - 1);
                double[] sample = samples[rand];

                if (!randomCenters.Any(c => tooSimilarCenters(c.Value, sample)))
                {
                    randomCenters.Add(pickedCenters, sample);
                    pickedCenters++;
                }
            }

            return randomCenters;
        }

        /// <summary>
        /// Runs the Kmeans process: computes center assignments and generates new centers until no new centers are found
        /// </summary>
        /// <param name="samples"></param>
        /// <param name="randomCenters"></param>
        private void processGroups(List<double[]> samples, Dictionary<int, double[]> randomCenters)
        {
            Dictionary<int, double[]> oldCenters = null, newCenters = randomCenters;
            int it = 0;

            do
            {
                clusterAssignments = ParallelGetClusterAssignments(samples, newCenters); 

                oldCenters = newCenters;

                newCenters = ParallelGetNewCenters(clusterAssignments);
            }
            while (!centersEqual(newCenters.Values.ToList(), oldCenters.Values.ToList()) && ++it < ITERATIONS);

            centers = newCenters;
        }

        /// <summary>
        /// Returns for each given center the samples which are nearest to it
        /// </summary>
        /// <param name="samples"></param>
        /// <param name="centers">Uses the index of the centers as hash</param>
        /// <returns></returns>
        public AssociativeArray<int, List<double[]>> ParallelGetClusterAssignments(List<double[]> samples, Dictionary<int, double[]> centers)
        {
            var centerAssignments = new AssociativeArray<int, List<double[]>>();

            samples.AsParallel().ForAll(sample =>   // For each sample in parallel
            {
                int closestCenter = -1;
                double closestCenterDistance = double.MaxValue;

                foreach (int key in centers.Keys)
                {
                    double d = distance(centers[key], sample);

                    if (d < closestCenterDistance)
                    {
                        closestCenterDistance = d;
                        closestCenter = key;
                    }
                }

                lock (centerAssignments)
                {
                    if(closestCenter >= 0)
                        centerAssignments[closestCenter].Add(sample);
                }
            });

            return centerAssignments;
        }

        /// <summary>
        /// Returns the new centers by averaging the samples which have been assigned to the same old center.
        /// Takes into account missing elements/dimensions computing the average only over the existing ones.
        /// </summary>
        /// <param name="clusterAssignments"></param>
        /// <returns></returns>
        public static Dictionary<int, double[]> ParallelGetNewCenters(AssociativeArray<int, List<double[]>> clusterAssignments)
        {
            Dictionary<int, double[]> newCenters = new Dictionary<int, double[]>();

            Parallel.ForEach(clusterAssignments.Keys, (key, state) =>
            {
                double[] total = new double[clusterAssignments[key][0].Length];
                int[] count = new int[total.Length];

                foreach (double[] sample in clusterAssignments[key])
                    Functions.SumSamples(total, sample, count);

                double[] avg = Functions.Normalize(total, count);

                lock (newCenters)
                {
                    newCenters.Add(key, avg);
                }
            });

            return newCenters;
        }

        /// <summary>
        /// Returns wether or not two center groups are equivalent, this means each center already existed with a 
        /// distance below Epsilon
        /// </summary>
        /// <param name="newCenters"></param>
        /// <param name="oldCenters"></param>
        /// <returns></returns>
        private bool centersEqual(List<double[]> newCenters, List<double[]> oldCenters)
        {
            if (newCenters == null || oldCenters == null)
            {
                return false;
            }

            foreach (double[] newCenter in newCenters)
            {
                // Using Euclidean distance because we want 'equal' cluster centers, therefore empty dimensions are considered
                if (!oldCenters.Any(c => tooSimilarCenters(c, newCenter)))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Returns true if the Euclidean Distance among the centers is below Epsilon
        /// (Using normal distance because we want 'equal' cluster centers, therefore empty dimensions are considered.)
        /// </summary>
        /// <param name="oldCenter"></param>
        /// <param name="newCenter"></param>
        /// <returns></returns>
        private static bool tooSimilarCenters(double[] oldCenter, double[] newCenter)
        {
            bool better;
            
            Functions.ManhattanDistance(oldCenter, newCenter, EPSILON, out better);
            
            return better;
        }

        /// <summary>
        /// Returns the clustering compactness error: the sum of distances of each sample to its cluster.
        /// Ignore the returned value if better = false
        /// </summary>
        /// <param name="centers"></param>
        /// <param name="clusterAssignments"></param>
        /// <param name="minError">Best error at the moment</param>
        /// <param name="better">True if no pruning happened</param>
        /// <returns></returns>
        private static double clusteringCompactnessError(Dictionary<int, double[]> centers, AssociativeArray<int, List<double[]>> clusterAssignments, double minError, out bool better)
        {
            double error = 0.0;
            better = true;

            if (centers != null && centers.Count > 1)
            {
                foreach (int key in clusterAssignments.Keys)
                {
                    foreach (double[] sample in clusterAssignments[key])
                    {
                        error += distance(sample, centers[key]);

                        if (error >= minError)
                        {
                            better = false;
                            break;
                        }
                    }

                    if (!better)
                        break;
                }
            }

            return error;
        }

        /// <summary>
        /// Returns the Euclidean Distance between a sample and a cluster center taking into account missing dimensions
        /// (Distance between a sample and a center)
        /// </summary>
        /// <param name="sample"></param>
        /// <param name="center"></param>
        /// <returns></returns>
        private static double distance(double[] sample, double[] center)
        {
            return Functions.ManhattanDistanceNormalized(sample, center);
        }
    }
}