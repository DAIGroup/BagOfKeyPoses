/*
   Copyright (C) 2014 Francisco Flórez-Revuelta and Alexandros Andre Chaaraoui

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
using System.Collections;

namespace Util
{
    /// <summary>
    /// Template-based Dynamic-Time-Warping implementation with upper bound.
    /// BASED ON: Junkui, L., & Yuanzhen, W. (2009). Early abandon to accelerate exact dynamic time warping. 
    /// Int. Arab J. Inf. Technol., 6(2), 144-152.www.ccis2k.org/iajit/PDF/vol.6,no.2/6EAAEDTW144.pdf
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class SimpleEarlyAbandonDTW<T>
    {
        List<T> x;                                          // Should be the query/test sequence (same size or shorter)
        List<T> y;                                          // Should be the reference/train sequence (same size or longer)
        double[,] distance;
        double[,] f;
        ArrayList pathX;
        ArrayList pathY;
        ArrayList distanceList;
        double sum;

        public SimpleEarlyAbandonDTW(List<T> x_query_test, List<T> _y_ref_train)
        {
            x = x_query_test;
            y = _y_ref_train;
            distance = new double[x.Count, y.Count];
            f = new double[x.Count + 1, y.Count + 1];

            for (int i = 0; i < x.Count; ++i)
            {
                for (int j = 0; j < y.Count; ++j)
                {
                    distance[i, j] = -1.0;
                }
            }

            sum = 0.0;

            pathX = new ArrayList();
            pathY = new ArrayList();
            distanceList = new ArrayList();
        }

        public ArrayList GetPathX()
        {
            return pathX;
        }

        public ArrayList GetPathY()
        {
            return pathY;
        }

        public double GetSum()
        {
            return sum;
        }

        public double[,] GetFMatrix()
        {
            return f;
        }

        public ArrayList GetDistanceList()
        {
            return distanceList;
        }
                
        public void ComputeDTW(DTWComparison<T> comparison, double minDistance)
        {
            distance[0, 0] = comparison.Distance(x[0], y[0]);

            if (distance[0, 0] > minDistance)
            {
                sum = Double.MaxValue;
                return;
            }
            else f[0, 0] = distance[0, 0];

            // Calculate first row
            for (int i = 1; i < x.Count; ++i)
            {
                if (f[i - 1, 0] > minDistance)
                    f[i, 0] = Double.MaxValue;
                else
                {
                    if (distance[i, 0] == -1.0)
                        distance[i, 0] = comparison.Distance(x[i], y[0]);

                    f[i, 0] = f[i - 1, 0] + distance[i, 0];
                } 
            }

            // Calculate first column
            for (int i = 1; i < y.Count; ++i)
            {
                if (f[0, i-1] > minDistance)
                    f[0, i] = Double.MaxValue;
                else
                {
                    if (distance[0, i] == -1.0)
                        distance[0, i] = comparison.Distance(x[0], y[i]);

                    f[0, i] = f[0, i - 1] + distance[0, i];
                }
            }

            bool overflow = false;

            for (int i = 1; i < x.Count; ++i)
            {
                overflow = true;

                for (int j = 1; j < y.Count; ++j)
                {
                    double v = f[i - 1, j - 1];

                    if (f[i - 1, j] < v)
                        v = f[i - 1, j];

                    if (f[i, j - 1] < v)
                        v = f[i, j - 1];

                    if (v > minDistance)
                        f[i, j] = double.MaxValue;
                    else
                    {
                        if (distance[i, j] == -1.0)
                            distance[i, j] = comparison.Distance(x[i], y[j]);

                        f[i, j] = v + distance[i, j];

                        if (f[i, j] > minDistance)
                            f[i, j] = double.MaxValue;

                        else overflow = false; // If none of the row cells overflows, the process continues.
                    }
                }

                if (overflow)
                {
                    break;
                }
            }

            if (overflow)
                sum = double.MaxValue;

            else sum = f[x.Count - 1, y.Count - 1];
        }

        /// <summary>
        /// This version of DTW considers overlapping subsequences. The beginning of the reference sequence Y may be ignored, and the end of 
        /// both sequences X and Y may be ignored in the alignment. The distance of the best alignment is returned considering these differences.
        /// </summary>
        /// <param name="comparison"></param>
        /// <param name="minDistance"></param>
        public void ComputeOverlapDTW(DTWComparison<T> comparison = null, double minDistance = double.MaxValue)
        {
            // Set first column to max => Which means that the beginning of the X sequence will be aligned.
            for (int i = 1; i <= x.Count; ++i)
                f[i, 0] = Double.MaxValue;

            // Set first row to 0 => Which means that the beginning of the Y sequence doesn't need to be aligned (anything could match with its last frame)
            for (int i = 1; i <= y.Count; ++i)
                f[0, i] = 0;

            // Fill the matrix
            bool overflow = false;

            for (int i = 1; i <= x.Count; ++i)
            {
                overflow = true;

                for (int j = 1; j <= y.Count; ++j)
                {
                    double v = f[i - 1, j - 1];

                    if (f[i - 1, j] < v)
                        v = f[i - 1, j];

                    if (f[i, j - 1] < v)
                        v = f[i, j - 1];

                    if (v > minDistance) // If it already overflows, it shouldn't be considered in the future.
                        f[i, j] = double.MaxValue;
                    else
                    {
                        if (distance[i - 1, j - 1] == -1.0)
                            distance[i - 1, j - 1] = comparison != null ? comparison.Distance(x[i - 1], y[j - 1]) : 
                                Math.Abs((double)Convert.ChangeType(x[i - 1], typeof(double)) - (double)Convert.ChangeType(y[j - 1], typeof(double)));

                        f[i, j] = v + distance[i - 1, j - 1];

                        if (f[i, j] > minDistance) // If it already overflows, it shouldn't be considered in the future.
                            f[i, j] = double.MaxValue;

                        else overflow = false; // If none of the row cells overflows, the process continues.
                    }
                }

                if (overflow) // A whole row overflows, which means that this is not a better match, the alignment cost can only increase.
                {
                    break;
                }
            }

            if (overflow)
                sum = double.MaxValue;

            else
            {
                // Look for the best alignment, without considering that the last frames need to be aligned,
                // therefore both the last row and column need to be checked.
                double best = double.MaxValue;
                int row = 0, column = 0;

                // Last row => The end of y could be ignored
                for (int i = 1; i < y.Count; ++i)
                {
                    if (f[x.Count, i] < best)
                    {
                        best = f[x.Count, i];
                        row = x.Count;
                        column = i;
                    }
                }

                /* Last column => The end of x could be ignored
                for (int i = 1; i < x.Count; ++i)
                {
                    if (f[i, y.Count] < best)
                    {
                        best = f[i, y.Count];
                        row = i;
                        column = y.Count;
                    }
                }*/

                sum = best;
            }
        }

        public void PrintF()
        {
            for (int i = 0; i <= x.Count; ++i)
            {
                for (int j = 0; j <= y.Count; ++j)
                    Console.Write(f[i, j] + ",");
                Console.WriteLine();
            }
        }
    }
}
