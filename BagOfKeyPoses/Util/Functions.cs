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
using System.IO;

namespace Util
{
    /// <summary>
    /// Utilities and non-object-specific functions.
    /// </summary>
    public static class Functions
    {
        // Static
        public static Random rand = new Random();           // Random generator for all key poses

        /// <summary>
        /// Returns Manhattan distance.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double ManhattanDistance(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new Exception("(Functions::ManhattanDistance) In order to compare vectors they should have the same size.");

            double distance = 0;

            for (int i = 0; i < a.Length; ++i)
            {
                distance += Math.Abs(a[i] - b[i]);
            }

            return distance;
        }

        /// <summary>
        /// Returns Correlation (similarity value)
        /// (Sample Correlation Coefficient)
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double Correlation(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new Exception("(Functions::Correlation) In order to compare vectors they should have the same size.");

            double first = 0, second = 0, third = 0;
            double a_avg = a.Average(), b_avg = b.Average();

            for (int i = 0; i < a.Length; ++i)
            {
                double a_des = a[i] - a_avg;
                double b_des = b[i] - b_avg;

                first += a_des * b_des;
                second += a_des * a_des;
                third += b_des * b_des;
            }

            return first / Math.Sqrt(second * third);
        }

        /// <summary>
        /// Returns Manhattan distance.
        /// Optimized to support Branch & Bound (when better = false, the returned value should be ignored)
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="minDistance">Best distance at the moment</param>
        /// <param name="better">True if no pruning happened</param>
        /// <returns></returns>
        public static double ManhattanDistance(double[] a, double[] b, double minDistance, out bool better)
        {
            if (a.Length != b.Length)
                throw new Exception("(Functions::ManhattanDistance) In order to compare vectors they should have the same size.");

            double distance = 0;
            better = true;
            
            for (int i = 0; i < a.Length; ++i)
            {
                distance += Math.Abs(a[i] - b[i]);

                if (distance >= minDistance)
                {
                    better = false;
                    break;
                }
            }

            return distance;
        }

        /// <summary>
        /// Obtains Manhattan distance considering missing dimensions (zero-valued),
        /// and returns a value normalized to the number of coincident elements.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double ManhattanDistanceNormalized(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new Exception("(Function::ManhattanDistanceNormalized) In order to compare vectors they should have the same size.");

            double d = 0.0;
            int matching = 0;

            for (int i = 0; i < a.Length; ++i)
            {
                if (a[i] != 0 && b[i] != 0) // Null values are missing or irrelevant dimensions
                {
                    d += Math.Abs(a[i] - b[i]);
                    matching++;
                }
            }

            if (matching > 0) // Normalization needed because number of coincident elements may change
                d = d / matching;
            else // No matches means that these samples are not similar
                d = 100000;

            return d;
        }

        /// <summary>
        /// Returns Euclidean distance
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double EuclideanDistance(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new Exception("(Function::EuclideanDistance) In order to compare vectors they should have the same size.");

            double distance = 0;

            for (int i = 0; i < a.Length; ++i)
            {
                distance += Math.Pow(a[i] - b[i], 2);
            }

            return Math.Sqrt(distance);
        }

        /// <summary>
        /// Returns Euclidean distance
        /// Optimized to support lower bound (when better = false, the returned value should be ignored)
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double EuclideanDistance(double[] a, double[] b, double minDistance, out bool better)
        {
            if (a.Length != b.Length)
                throw new Exception("(Function::EuclideanDistance) In order to compare vectors they should have the same size.");

            double distance = 0;
            better = true;

            for (int i = 0; i < a.Length; ++i)
            {
                distance += Math.Pow(a[i] - b[i], 2);

                if (distance >= minDistance)
                {
                    better = false;
                    break;
                }
            }

            return Math.Sqrt(distance);
        }

        /// <summary>
        /// Obtains Euclidean distance considering missing dimensions (zero-valued),
        /// and returns a value normalized to the number of coincident elements.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double EuclideanDistanceNormalized(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new Exception("(Function::EuclideanDistanceNormalized) In order to compare vectors they should have the same size.");

            double d = 0.0;
            int matching = 0;

            for (int i = 0; i < a.Length; ++i)
            {
                if (a[i] != 0 && b[i] != 0) // Null values are missing or irrelevant dimensions
                {
                    d += Math.Pow(a[i] - b[i], 2);
                    matching++;
                }
            }

            if (matching > 0) // Normalization needed because number of coincident elements may change
                d = d / matching;
            else // No matches means that these samples are not similar
                d = double.MaxValue - 1;

            return Math.Sqrt(d);
        }

        /// <summary>
        /// Returns the Chi-Square distance.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double ChiSquareDistance(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new Exception("(Function::ChiSquareDistance) In order to compare vectors they should have the same size.");

            int length = a.Length;
            double dist = 0.0;

            for (int i = 0; i < length; ++i)
            {
                if (a[i] == 0 && b[i] == 0) continue;

                dist += (a[i] - b[i]) * (a[i] - b[i]) / a[i] + b[i];
            }

            dist /= 2;

            return dist;
        }

        /// <summary>
        /// Returns the Bhattacharyya coefficient (amount of overlap).
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double BhattacharyyaCoefficient(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new Exception("(Function::BhattacharyyaDistance) In order to compare vectors they should have the same size.");

            int length = a.Length;
            double coeff = 0.0;

            for (int i = 0; i < length; ++i)
            {
                coeff += Math.Sqrt(a[i] * b[i]);
            }

            return coeff;
        }

        /// <summary>
        /// Returns the Kullback-Leibler divergence (from a to b, not symmetric).
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double KullbackLeiblerDivergence(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new Exception("(Function::KullbackLeiblerDivergence) In order to compare vectors they should have the same size.");

            int length = a.Length;
            double klDiv = 0.0;

            for (int i = 0; i < length; ++i)
            {
                if (a[i] == 0) continue;
                if (b[i] == 0) continue;

                klDiv += a[i] * Math.Log(a[i] / b[i]);
            }

            return klDiv / Math.Log(2);
        }

        /// <summary>
        /// Checks wether or not two arrays are equal
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool Equal(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new Exception("(Function::Equal) In order to compare vectors they should have the same size.");

            bool result = true;

            for (int i = 0; i < a.Length; ++i)
            {
                if (a[i] != b[i])
                {
                    result = false;
                    break;
                }
            }

            return result;
        }

        /// <summary>
        /// Normalizes the given vector wrt p (divides each element by p)
        /// </summary>
        /// <param name="vec"></param>
        /// <param name="p"></param>
        /// <returns></returns>
        public static double[] Normalize(double[] vec, double p)
        {
            double[] norm = new double[vec.Length];

            if (p != 0)
            {
                for (int i = 0; i < norm.Length; ++i)
                    norm[i] = vec[i] / p;
            }

            return norm;
        }

        /// <summary>
        /// Normalizes the given array by dividing each element by the corresponding position of count
        /// </summary>
        /// <param name="array"></param>
        /// <param name="count"></param>
        public static double[] Normalize(double[] array, double[] count)
        {
            double[] norm = new double[array.Length];

            for (int i = 0; i < array.Length; ++i)
                if (count[i] > 0)
                    norm[i] = array[i] / count[i];

            return norm;
        }

        /// <summary>
        /// Normalizes the given array by dividing each element by the corresponding position of count
        /// </summary>
        /// <param name="array"></param>
        /// <param name="count"></param>
        public static double[] Normalize(double[] array, int[] count)
        {
            double[] norm = new double[array.Length];

            for (int i = 0; i < array.Length; ++i)
                if (count[i] > 0)
                    norm[i] = array[i] / count[i];

            return norm;
        }

        /// <summary>
        /// Returns an array with the sums of the i-th values of the given vectors
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double[] SumArrays(double[] a, double[] b)
        {
            double[] sum = new double[a.Length];

            for (int i = 0; i < sum.Length; ++i)
                sum[i] = a[i] + b[i];

            return sum;
        }

        /// <summary>
        /// Returns the array elevated to the given power
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double[] PowArray(double[] a, double pow)
        {
            double[] result = new double[a.Length];

            for (int i = 0; i < result.Length; ++i)
                result[i] = Math.Pow(a[i], pow);

            return result;
        }

        /// <summary>
        /// Returns the array with its squared root values
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double[] SqrtArray(double[] a)
        {
            double[] result = new double[a.Length];

            for (int i = 0; i < result.Length; ++i)
                result[i] = Math.Sqrt(a[i]);

            return result;
        }

        /// <summary>
        /// Vector substraction
        /// </summary>
        /// <param name="vector1"></param>
        /// <param name="vector2"></param>
        /// <returns></returns>
        public static double[] Subtract(double[] vector1, double[] vector2)
        {
            double[] sub = new double[vector1.Length];

            for (int i = 0; i < vector1.Length; ++i)
                sub[i] = vector1[i] - vector2[i];

            return sub;
        }

        /// <summary>
        /// Returns a random vector of doubles
        /// </summary>
        /// <param name="length"></param>
        /// <returns></returns>
        public static double[] Random(int length)
        {
            double[] random = new double[length];

            for (int i = 0; i < length; ++i)
                random[i] = rand.NextDouble(); // it can't be 1, but 0.99 should do it

            return random;
        }

        /// <summary>
        /// Adapts the first vector to the second by a certain coeff
        /// </summary>
        /// <param name="from"></param>
        /// <param name="to"></param>
        /// <returns>adaptation</returns>
        public static double[] AdaptTo(double[] from, double[] to, double coeff)
        {
            double[] result = new double[from.Length];

            for (int i = 0; i < result.Length; ++i)
                result[i] = from[i] + coeff * (to[i] - from[i]);

            return result;
        }

        /// <summary>
        /// Returns the average vector of the given ones
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double[] Average(double[] a, double[] b)
        {
            double[] result = new double[a.Length];

            for (int i = 0; i < result.Length; ++i)
                result[i] = 0.5 * (a[i] + b[i]);

            return result;
        }

        /// <summary>
        /// Adds to the given total array the sample array and updates the count array with the new amount of non-zero values
        /// </summary>
        /// <param name="total"></param>
        /// <param name="sample"></param>
        /// <param name="count"></param>
        public static void SumSamples(double[] total, double[] sample, int[] count)
        {
            for (int i = 0; i < sample.Length; ++i)
            {
                if (sample[i] != 0)
                {
                    total[i] += sample[i];
                    count[i]++;
                }
            }
        }

        /// <summary>
        /// Returns the median value (average of medians if there are an even number of values)
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double Median(IEnumerable<double> values)
        {
            double median;
            List<double> rSorted = new List<double>(values);
            rSorted.Sort();
            
            if (rSorted.Count % 2 == 1)
                median = rSorted[rSorted.Count / 2];
            else
                median = (rSorted[rSorted.Count / 2] + rSorted[(rSorted.Count / 2) - 1]) / 2;

            return median;
        }

        /// <summary>
        /// Returns the indicated percentil value. Call with 0.25 for 1st quartil, 0.5 for median and 0.75 for 3rd quartil
        /// </summary>
        /// <param name="sequence"></param>
        /// <param name="excelPercentile">0 to 1</param>
        /// <returns></returns>
        public static double Percentile(List<double> values, double excelPercentile)
        {
            List<double> sequence = new List<double>(values);
            sequence.Sort();
            int N = sequence.Count;
            double n = (N - 1) * excelPercentile + 1;
            // Another method: double n = (N + 1) * excelPercentile;
            if (n == 1d) return sequence[0];
            else if (n == N) return sequence[N - 1];
            else
            {
                int k = (int)n;
                double d = n - k;
                return sequence[k - 1] + d * (sequence[k] - sequence[k - 1]);
            }
        }

        /// <summary>
        /// Obtains the corrected sample standard deviation and returns also the average
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double CalculateStdDev(IEnumerable<double> values)
        {
            double mean;
            return CalculateStdDev(values, out mean);
        }

        /// <summary>
        /// Obtains the corrected sample standard deviation and returns also the average
        /// </summary>
        /// <param name="values"></param>
        /// <param name="mean"></param>
        /// <returns></returns>
        public static double CalculateStdDev(IEnumerable<double> values, out double mean)
        {
            double ret = 0;
            mean = 0;

            if (values.Count() > 0)
            {
                //Compute the Average      
                double avg = values.Average();

                //Perform the Sum of (value-avg)^2      
                double sum = values.Sum(d => Math.Pow(d - avg, 2));

                //Put it all together      
                ret = Math.Sqrt((sum) / (values.Count() - 1));
                mean = avg;
            }
                        
            return ret;
        }

        public static Bitmap RotateBitmap(Bitmap b, float angle)
        {
            // Take width/height change into account
            int R1 = 0, R2 = 0;
           
            if (b.Width > b.Height)
                R2 = b.Width - b.Height;
            else
                R1 = b.Height - b.Width;

            //create a new empty bitmap to hold rotated image
            int newWidth = b.Width + R1 + 40;
            int newHeight = b.Height + R2 + 40;
            Bitmap returnBitmap = new Bitmap(newWidth, newHeight);
            returnBitmap.SetResolution(b.HorizontalResolution, b.VerticalResolution);

            //make a graphics object from the empty bitmap
            Graphics g = Graphics.FromImage(returnBitmap);

            //move rotation point to center of image
            g.TranslateTransform((float)newWidth / 2, (float)newHeight / 2);

            //rotate
            g.RotateTransform(angle);

            //move image back
            g.TranslateTransform(-(float)newWidth / 2, -(float)newHeight / 2);

            //draw passed in image onto graphics object
            g.DrawImage(b, new PointF(R1 / 2 + 20, R2 / 2 + 20));

            return returnBitmap;
        }

        /// <summary>
        /// Resizes given bitmap to the given size
        /// </summary>
        /// <param name="sourceBMP"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <returns></returns>
        public static Bitmap ResizeBitmap(Bitmap sourceBMP, int width, int height)
        {
            Bitmap result = new Bitmap(width, height);
            using (Graphics g = Graphics.FromImage(result))
                g.DrawImage(sourceBMP, 0, 0, width, height);
            return result;
        }

        /// <summary>
        /// Returns the two highest values and the key of the heighest
        /// </summary>
        /// <param name="weightingScheme"></param>
        /// <param name="rSecond"></param>
        /// <param name="currentHighest"></param>
        /// <returns></returns>
        public static double FirstAndSecondMax(AssociativeArray<string, double> weightingScheme, out double rSecond, out string currentHighest)
        {
            double rMax = double.MinValue; 
            rSecond = double.MinValue;
            currentHighest = "";

            foreach (var key in weightingScheme.Keys) // Compute the max and the second max
            {
                if (weightingScheme[key] > rMax)
                {
                    rSecond = rMax;
                    rMax = weightingScheme[key];
                    currentHighest = key;
                }
                else if (weightingScheme[key] > rSecond)
                    rSecond = weightingScheme[key];
            }

            return rMax;
        }

        /// <summary>
        /// Returns the sum of squared errors: the sum of squared distances of each sample to its cluster.
        /// </summary>
        /// <param name="centers"></param>
        /// <param name="clusterAssignments"></param>
        /// <returns></returns>
        public static double SumOfSquaredErrors(Dictionary<int, double[]> centers, AssociativeArray<int, List<double[]>> clusterAssignments)
        {
            double sse = 0.0;

            if (centers != null && centers.Count > 1)
            {
                foreach (int key in clusterAssignments.Keys)
                {
                    foreach (double[] sample in clusterAssignments[key])
                    {
                        for (int i = 0; i < sample.Length; ++i)
                            sse += Math.Pow(sample[i] - centers[key][i], 2);
                    }
                }
            }

            return sse;
        }

        /// <summary>
        /// Prints the Q1, median and Q3 values in seperate files for vox plots.
        /// studyStats[k, l, success_rate]
        /// </summary>
        /// <param name="studyStats"></param>
        public static void Print3DBoxPlotData(AssociativeMatrix<int, int, List<double>> studyStats)
        {
            // Print arrays for Median, Q1 and Q3
            TextWriter stats = new StreamWriter("Medians.csv");

            foreach(var k in studyStats.RowKeys.OrderBy(k => k))
                stats.Write(", " + k);
            stats.WriteLine();

            foreach(var l in studyStats.ColumnKeys(studyStats.RowKeys[0]).OrderBy(k => k))
            {
                stats.Write(l);
                foreach(var k in studyStats.RowKeys.OrderBy(k => k))
                {
                    var rates = studyStats[k, l];
                    double median = Functions.Median(rates);                        
                    stats.Write("; " + median);
                }

                stats.WriteLine();
            }

            stats.Close();

            stats = new StreamWriter("Q1s.csv");

            foreach (var k in studyStats.RowKeys.OrderBy(k => k))
                stats.Write(", " + k);
            stats.WriteLine();

            foreach (var l in studyStats.ColumnKeys(studyStats.RowKeys[0]).OrderBy(k => k))
            {
                stats.Write(l);
                foreach (var k in studyStats.RowKeys.OrderBy(k => k))
                {
                    var rates = studyStats[k, l];
                    double q1 = Functions.Percentile(rates, 0.25);
                    stats.Write("; " + q1);
                }

                stats.WriteLine();
            }

            stats.Close();

            stats = new StreamWriter("Q3s.csv");

            foreach (var k in studyStats.RowKeys.OrderBy(k => k))
                stats.Write(", " + k);
            stats.WriteLine();

            foreach (var l in studyStats.ColumnKeys(studyStats.RowKeys[0]).OrderBy(k => k))
            {
                stats.Write(l);
                foreach (var k in studyStats.RowKeys.OrderBy(k => k))
                {
                    var rates = studyStats[k, l];
                    double q3 = Functions.Percentile(rates, 0.75);

                    stats.Write("; " + q3);
                }

                stats.WriteLine();
            }

            stats.Close();
        }

        /// <summary>
        /// Returns true if all members are zero
        /// </summary>
        /// <param name="feature"></param>
        /// <returns></returns>
        public static bool IsZero(double[] feature)
        {
            bool result = true;

            foreach (var f in feature)
            {
                if (f != 0.0)
                {
                    result = false;
                    break;
                }
            }

            return result;
        }

        /// <summary>
        /// 2D Convolution for 1D arrays.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="dimX"></param>
        /// <param name="dimY"></param>
        /// <param name="kernel"></param>
        /// <returns></returns>
        public static double[] Convolution2D(double[] source, int dimX, int dimY, double[,] kernel)
        {
            double kSize = Math.Sqrt(kernel.Length);

            if (kernel.Length < 9 || kSize % 1 != 0 || kSize % 2 == 0)
                throw new Exception("(Function::Convolution2D) Invalid kernel.");

            double[] result = new double[source.Length];            
            int halfKSize = (int) kSize / 2;

            for (int i = 0; i < dimX; ++i)
            {
                for (int j = 0; j < dimY; ++j)
                {
                    double value = 0.0;
                    int pos = i * dimY + j;

                    for (int ki = -halfKSize; ki <= halfKSize; ++ki)
                    {
                        for (int kj = -halfKSize; kj <= halfKSize; ++kj)
                        {
                            if (i + ki < 0 || i + ki >= dimX)
                                break;
                            if (j + kj < 0 || j + kj >= dimY)
                                break;

                            value += kernel[halfKSize + ki, halfKSize + kj] * source[(i + ki) * dimY + (j + kj)];
                        }
                    }

                    result[pos] = value;
                }
            }

            return result;
        }

        /// <summary>
        /// Returns an IEnumerable of integer values that increase by stepSize in the specified range.
        /// </summary>
        /// <param name="startIndex"></param>
        /// <param name="endIndex"></param>
        /// <param name="stepSize"></param>
        /// <returns></returns>
        public static IEnumerable<int> SteppedIterator(int startIndex, int endIndex, int stepSize)
        {
            for (int i = startIndex; i < endIndex; i = i + stepSize)
            {
                yield return i;
            }
        }

        /// <summary>
        /// Returns an IEnumerable of double values that increase by stepSize in the specified range.
        /// </summary>
        /// <param name="startIndex"></param>
        /// <param name="endIndex"></param>
        /// <param name="stepSize"></param>
        /// <returns></returns>
        public static IEnumerable<double> SteppedIterator(double startIndex, double endIndex, double stepSize)
        {
            for (double i = startIndex; i < endIndex; i = i + stepSize)
            {
                yield return i;
            }
        }
    }
}
