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
using System.Drawing;
using System.ComponentModel;
using System.Threading.Tasks;
using TrainDataType = System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<System.Collections.Generic.List<double[]>>>;

namespace BagOfKeyPoses
{
    /// <summary>
    /// This is the entry point of this library. It handles the learning and recognition of any kind of temporally related sequences of features by means
    /// of the bag of key poses method.
    /// NOTE: This class is not thread-safe.
    /// 
    /// In order to use this code in research work the following citation is required: 
    ///     Chaaraoui, A. A., Climent-Pérez, P., & Flórez-Revuelta, F. (2013). 
    ///     Silhouette-based human action recognition using sequences of key poses. 
    ///     Pattern Recognition Letters, 34(15), 1799-1807.
    ///     
    /// Code is distributed under the free software license Apache 2.0 License (http://www.apache.org/licenses/LICENSE-2.0.html) 
    /// and requires preservation of the copyright notice and disclaimer.
    /// </summary>
    public class BoKP
    {
        // Static
        public static readonly object LockFolds = new object();                         // The employed lock for concurrent folds
        //public static int FoldsInParallel = -1;                                         // By default folds are executed in parallel (-1 no limit)

        // Public
        public TrainConfig Config;                                                      // Current train configuration (constitutes train memory)
        public KeyPoseSequence MatchedTestSequence;                                     // Matched sequence of test data and key poses (DTW)
        
        // Private
        private List<double[]> testSequence;                                            // Current test sequence
        private int testIndex;                                                          // Next contour to test
        private KeyPose CurrentKeyPose;                                                 // The currently matched key pose

        /// <summary>
        /// New BoKP with previously learned Key Poses (test can be used right on)
        /// <param name="tc">Previously learning training configuration</param>
        /// </summary>
        public BoKP(TrainConfig tc)
        {
            this.Config = tc;
        }

        /// <summary>
        /// Sets the training parameters
        /// </summary>
        public BoKP(LearningParams parameters)
        {
            this.Config = new TrainConfig(parameters);
        }

        /// <summary>
        /// Train function uses previously setup configuration in order to process the data and learn the key poses
        /// <param name="trainData">Sequences of features hashed per class</param>
        /// <param name="keyPoses">Already learnt Key poses for learning of new action classes</param>
        /// </summary>
        public void Train(TrainDataType trainData, BackgroundWorker bgw = null, AssociativeArray<string, List<double[]>> keyPoses = null)
        {
            try
            {            
                // Learn Key Poses, key pose weights and key pose sequences
                Config.KeyPoses = KeyPoseWeighter.DoLearn(trainData, Config, bgw, Config.Params.CalcKPWeightsAndSeqs, keyPoses);
                Config.KeyPoseDistanceCache = new AssociativeMatrix<KeyPose, KeyPose, double>();

                if(Config.Params.UseZones)
                    KeyPoseWeighter.LearnZones(trainData, Config);               

                if (bgw != null)
                    bgw.ReportProgress(100, "\nDone.");
            }
            catch (Exception e)
            {
                throw new Exception("(BoKP::Train) Error occurred during training: " + e.Message + " [" + e.InnerException + "]");
            }
        }

        /// <summary>
        /// Recognizes a given feature and returns the closest key pose
        /// </summary>
        /// <param name="contour"></param>
        /// <returns></returns>
        private KeyPose TestOne(double[] feature)
        {
            if (!Config.Params.OneClassLearning)
            {
                // Substitute with nearest neighbour key pose
                CurrentKeyPose = KeyPose.ClosestAmongAll(feature, Config.KeyPoses, Config, true);
            }
            else
            {
                // Test feature could belong to an unknown class
                CurrentKeyPose = new KeyPose("test", feature);
            }

            return CurrentKeyPose;
        }

        /// <summary>
        /// Evaluates the next sequence and returns current weights (TestSequenceInit should be called at the beginning)
        /// </summary>
        /// <returns>True if there are more sequences</returns>
        public bool TestSequenceNext()
        {
            bool framesLeft = false;

            // Recognize
            KeyPose kp = TestOne(testSequence[testIndex]);

            if (testIndex < testSequence.Count - 1)
                framesLeft = true;

            // Save as matched keypose (considering summarization)
            if (!Config.Params.UseSummarization || MatchedTestSequence.Items.Count < 1 || MatchedTestSequence.Items.Last() != kp)
                MatchedTestSequence.Items.Add(kp);
           
            testIndex++;

            return framesLeft;
        }    

        /// <summary>
        /// Initializes a new test sequence (should be called before TestSequenceNext)
        /// </summary>
        /// <param name="sequence"></param>
        public void TestSequenceInit(List<double[]> sequence)
        {
            testIndex = 0;
            MatchedTestSequence = new KeyPoseSequence();
            testSequence = sequence;
        }

        /// <summary>
        /// Uses the current matchedTestSequence to match it to the closest of the saved config.MatchedKeyPoses by using
        /// Dynamic Time Warping
        /// </summary>
        /// <returns></returns>
        public string TestSequenceDTW(out double minDistance)
        {
            string action = null;
            minDistance = double.MaxValue;
            
            foreach (KeyValuePair<string, List<KeyPoseSequence>> trainSequences in Config.MatchedKeyPoseSequences) // For each action
            {
                foreach (KeyPoseSequence trainSequence in trainSequences.Value) // For each sequence
                {
                    if (trainSequence.Items.Count > 0)
                    {
                        // Simple DTW
                        //SimpleDTW<KeyPose> dtw = new SimpleDTW<KeyPose>(matchedTestSequence.Items, trainSequence.Items, new KeyPoseComparison(Config, keyPoseDistanceCache, trainSequence.Action));
                        //dtw.ComputeDTW();

                        // Early Abandon DTW
                        SimpleEarlyAbandonDTW<KeyPose> dtw = new SimpleEarlyAbandonDTW<KeyPose>(MatchedTestSequence.Items, trainSequence.Items);
                        dtw.ComputeDTW(new KeyPoseComparison(Config, trainSequence.ClassLabel), minDistance);

                        double d = dtw.GetSum();

                        if (d < minDistance)
                        {
                            minDistance = d;
                            action = trainSequences.Key;
                        }
                    }
                }
            }

            return action;
        }

        /// <summary>
        /// Parameter-less overload of TestSequenceDTW
        /// </summary>
        /// <returns></returns>
        private string TestSequenceDTW()
        {
            double d;
            return TestSequenceDTW(out d);
        }

        /// <summary>
        /// Evaluates a given sequence with DTW and returns the class label of the nearest neighbor sequence
        /// </summary>
        /// <param name="sequence"></param>
        /// <returns></returns>
        public string EvaluateSequence(List<double[]> sequence)
        {
            double dist;
            return EvaluateSequence(sequence, out dist);
        }

        /// <summary>
        /// Evaluates a given sequence with DTW and returns the class label of the nearest neighbor sequence.
        /// </summary>
        /// <param name="sequence"></param>
        /// <returns></returns>
        public string EvaluateSequence(List<double[]> sequence, out double distance)
        {
            // Init sequence
            TestSequenceInit(sequence);

            // Evaluate sequence
            while (TestSequenceNext());

            // Obtain result            
            return TestSequenceDTW(out distance);
        }

        /// <summary>
        /// Evaluates a given sequence based only on its features and returns the class labels and distances of the 
        /// nearest neighbour key poses.
        /// </summary>
        /// <param name="sequence">Input sequence of features</param>
        /// <param name="distances">Distances of the nearest neighbour key poses</param>
        public void EvaluatePoses(List<double[]> sequence, out List<double> distances)
        {
            distances = new List<double>();
            
            // Evaluate sequence.
            foreach (var feature in sequence)
            {
                KeyPose nnkp = KeyPose.ClosestAmongAll(feature, Config.KeyPoses, Config, true);
                distances.Add(nnkp.MatchedDistance);
            }
        }

        /// <summary>
        /// Applys continuous evaluation to the given sequence based on sliding window technique and segment evaluation.
        /// Use EvaluateSequence to obtain a single result for the whole sequence.
        /// </summary>
        /// <param name="sequence"></param>
        /// <returns></returns>
        public List<string> EvaluateCHARSequence(List<double[]> sequence)
        {
            int frame = 0, segmentFrame = 0;
            bool evaluateDTW = false, framesLeft = true, split;
            string recognizedAction;
            List<string> recognitions = new List<string>();

            // Init sequence
            TestSequenceInit(sequence);

            // Evaluate sequence
            while (framesLeft)
            {
                framesLeft = TestSequenceNext();

                // Update counters
                frame++;
                segmentFrame++;

                //
                // Sliding & Growing Window
                //

                // Enough frames to try and delta frames have been accumulated                            
                if (segmentFrame >= Config.Params.MinFrames && segmentFrame % LearningParams.Delta == 0)
                    evaluateDTW = true;

                // Shift the window if the whole window didn't return an acceptable recognition that caused a splitting
                if (segmentFrame > Config.Params.MaxFrames)
                {
                    // We decide to ignore Gamma frames
                    MatchedTestSequence.Items.RemoveRange(0, LearningParams.Gamma);
                    segmentFrame -= LearningParams.Gamma;

                    // Therefore, these frames have not been recognised
                    recognitions.AddRange(Enumerable.Repeat(LearningParams.UNKNOWN, LearningParams.Gamma));
                }

                recognizedAction = "PROCESSING";
                split = false;

                //
                // Obtain current result
                //
                if (evaluateDTW)
                {
                    // Evaluate the current sequence
                    double d;
                    string bestMatch = TestSequenceDTW(out d);

                    if (d <= Config.Params.DThreshold[bestMatch] * segmentFrame) // Frame-wise threshold
                    {
                        recognizedAction = bestMatch;
                        split = true; // This part has been recognized

                        // The whole segment has been recognised
                        recognitions.AddRange(Enumerable.Repeat(recognizedAction, segmentFrame));
                    }

                    evaluateDTW = false; // Do not try again until we reach MinFrames
                }
                // else we cannot recognize yet

                // A new sliding window should be started
                if (split)
                {
                    MatchedTestSequence = new KeyPoseSequence();
                    segmentFrame = 0;
                }
            }

            return recognitions;
        }
    }
}
