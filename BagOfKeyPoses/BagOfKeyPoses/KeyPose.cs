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
using System.Drawing;
using System.Linq;
using System.Text;
using System.Xml;
using Util;

namespace BagOfKeyPoses
{
    /// <summary>
    /// This object handles the data of a characteristic pose (mean of a cluster) of an action.
    /// K key poses are available for each action.
    /// </summary>
    public class KeyPose
    {
        // Static
        private static int NextID = 0;                              // Next ID to be used
        private static readonly object lockID = new object();       // Corresponding concurrency lock

        // Public
        public double[] Distances;                          // S distances or feature elements
        public string ClassLabel;                           // Class label
        public double Weight;                               // Learned weight (percentual relation of within class assignments)
        public string ID;                                   // Database ID (for viewing purposes)
        public int WithinClass;                             // Number of within class assignments
        public int OutOfClass;                              // Number of out of class assignments
        public double MatchedDistance;                      // Distance with which this key pose has been matched

        /// <summary>
        /// Constructor
        /// </summary>
        public KeyPose()
        {
            Weight = 0;
            WithinClass = 0;
            OutOfClass = 0;
            lock(lockID)
                ID = (NextID++).ToString();
        }

        /// <summary>
        /// Creates a new Keypose of a given actions and parses the distances data (1 row and N distances)
        /// </summary>
        /// <param name="action"></param>
        /// <param name="data"></param>
        public KeyPose(string action, float[,] data, int S)
        {
            this.Weight = 0;
            this.ClassLabel = action;
            this.Distances = new double[S];
            for (int i = 0; i < Distances.Length; ++i)
                Distances[i] = (double) data[0, i];
            this.WithinClass = 0;
            this.OutOfClass = 0;
            lock (lockID)
                this.ID = (NextID++).ToString();
        }

        /// <summary>
        /// Creates a new Keypose with the given data
        /// </summary>
        /// <param name="action"></param>
        /// <param name="distances"></param>
        public KeyPose(string action, double[] distances)
        {
            this.Weight = 0;
            this.ClassLabel = action;
            this.Distances = distances;
            this.WithinClass = 0;
            this.OutOfClass = 0;
            lock (lockID)
                this.ID = (NextID++).ToString();
        }

        /// <summary>
        /// Returns a XML document with the details of the key pose
        /// </summary>
        /// <param name="count"></param>
        /// <param name="printFeature"></param>
        /// <returns></returns>
        public XmlDocument ToXML(int count, bool printFeature = true)
        {
            XmlDocument xml = new XmlDocument();
            XmlNode root = xml.CreateElement("key-pose");
            xml.AppendChild(root);

            XmlAttribute countAtt = xml.CreateAttribute("count");
            countAtt.Value = count.ToString();
            root.Attributes.Append(countAtt);

            XmlNode idNode = xml.CreateElement("ID");
            idNode.InnerText = ID;
            root.AppendChild(idNode);

            XmlNode actionNode = xml.CreateElement("action-class");
            actionNode.InnerText = ClassLabel;
            root.AppendChild(actionNode);

            if (printFeature)
            {
                XmlNode featureNode = xml.CreateElement("feature-vector");
                string feature = "";
                foreach (var f in Distances)
                    feature += f + " ";

                featureNode.InnerText = feature.Substring(0, feature.Length - 1);
                root.AppendChild(featureNode);
            }

            return xml;
        }

        /// <summary>
        /// Compares the given feature to those of the keyposes and returns the closest key pose/nearest neighbor
        /// </summary>
        /// <param name="distances">Feature to compare</param>
        /// <param name="keyPoses">Keyposes to search among</param>
        /// <param name="config">Used TrainConfig</param>
        /// <param name="pruning">Wether or not NN search should be pruned. NOTE that this will not consider all feature elements.</param>
        /// <returns></returns>
        public static KeyPose ClosestAmongAll(double[] distances, Dictionary<string, List<KeyPose>> keyPoses, TrainConfig config, bool pruning = false)
        {
            KeyPose closestKP = null;
            double minDistance = double.MaxValue;

            foreach (KeyValuePair<string, List<KeyPose>> clusters in keyPoses) // For each action
            {
                foreach (KeyPose kp in clusters.Value) // For each key pose
                {
                    bool better = true;
                    double d = 0.0;
                                
                    if (pruning)
                        d = Functions.ManhattanDistance(distances, kp.Distances, minDistance, out better);
                    else
                        d = Functions.ManhattanDistanceNormalized(distances, kp.Distances);

                    if (closestKP == null || (better && d < minDistance))
                    {
                        minDistance = d;
                        closestKP = kp;
                    }
                }
            }

            if(closestKP != null) // Due to pruning we could not have any
                closestKP.MatchedDistance = minDistance;

            return closestKP;
        }

        /// <summary>
        /// Compares the given feature to those of the keyposes and returns the closest key pose of each action class
        /// </summary>
        /// <param name="distances">Feature to compare</param>
        /// <param name="keyPoses">Keyposes to search among</param>
        /// <param name="config">Used TrainConfig</param>
        /// <param name="pruning">Wether or not NN search should be pruned</param>
        /// <param name="sumDist">Returns sum of distances</param>
        /// <param name="countDist">Returns number of summed distances</param>
        /// <returns></returns>
        public static AssociativeArray<string, KeyPose> ClosestAmong(double[] distances, Dictionary<string, List<KeyPose>> keyPoses, TrainConfig config)
        {
            AssociativeArray<string, KeyPose> closestKPs = new AssociativeArray<string, KeyPose>();
            
            foreach (KeyValuePair<string, List<KeyPose>> clusters in keyPoses) // For each action
            {
                string action = clusters.Key;
                double minDistance = double.MaxValue;

                foreach (KeyPose kp in clusters.Value) // For each key pose
                {
                    double d = Functions.ManhattanDistanceNormalized(distances, kp.Distances);

                    if (!closestKPs.ContainsKey(action) || d < minDistance)
                    {
                        minDistance = d;
                        closestKPs[action] = kp;
                    }
                }

                closestKPs[action].MatchedDistance = minDistance;
            }            

            return closestKPs;
        }

        /// <summary>
        /// Merges two collections of poses
        /// </summary>
        /// <param name="keyPoses"></param>
        /// <param name="kp"></param>
        internal static Dictionary<string, List<KeyPose>> MergePoses(Dictionary<string, List<KeyPose>> a, Dictionary<string, List<KeyPose>> b)
        {
            Dictionary<string, List<KeyPose>> keyPoses = new Dictionary<string, List<KeyPose>>();
            HashSet<string> keys = new HashSet<string>();
            foreach (string key in a.Keys.ToArray()) keys.Add(key);
            foreach (string key in b.Keys.ToArray()) keys.Add(key);

            foreach (string key in keys)
            {
                keyPoses.Add(key, new List<KeyPose>());
                if (a.ContainsKey(key)) keyPoses[key].AddRange(a[key]);
                if (b.ContainsKey(key)) keyPoses[key].AddRange(b[key]);
            }

            return keyPoses;
        }
    }
}
