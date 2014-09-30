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
using System.Collections;

namespace Util
{
    /// <summary>
    /// Interface which simplifies the distance function between DTW elements.
    /// This way unnecessary traspassing of parameters can be avoided (as TrainConfig)
    /// </summary>
    /// <typeparam name="U"></typeparam>
    public interface DTWComparison<U>
    {
        double Distance(U x, U y);
        double Correlation(U x, U y);
    }

    /// <summary>
    /// Simple template-based Dynamic-Time-Warping implementation.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class SimpleDTW<T>
    {
        List<T> x;
        List<T> y;
        double[,] distance;
        double[,] f;
        ArrayList pathX;
        ArrayList pathY;
        ArrayList distanceList;
        double sum;

        public SimpleDTW(List<T> _x, List<T> _y)
        {
            x = _x;
            y = _y;
            distance = new double[x.Count, y.Count];
            f = new double[x.Count + 1, y.Count + 1];
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

        public void ComputeDTW(DTWComparison<T> comparison)
        {
            for (int i = 0; i < x.Count; ++i)
            {
                for (int j = 0; j < y.Count; ++j)
                {
                    distance[i, j] = comparison.Distance(x[i], y[j]);
                }
            }

            for (int i = 0; i <= x.Count; ++i)
            {
                for (int j = 0; j <= y.Count; ++j)
                {
                    f[i, j] = -1.0;
                }
            }

            f[0, 0] = 0.0;
            sum = 0.0;

            // Hack, so that the function can always consider i-1, j-1
            for (int i = 1; i <= x.Count; ++i)
            {
                f[i, 0] = double.PositiveInfinity;
            }

            for (int j = 1; j <= y.Count; ++j)
            {
                f[0, j] = double.PositiveInfinity;
            }

            sum = ComputeForward();
        }

        private double ComputeForward()
        {
            for (int i = 1; i <= x.Count; ++i)
            {
                for (int j = 1; j <= y.Count; ++j)
                {
                    if (f[i - 1, j] <= f[i - 1, j - 1] && f[i - 1, j] <= f[i, j - 1])
                    {
                        f[i, j] = distance[i - 1, j - 1] + f[i - 1, j];
                    }
                    else if (f[i, j - 1] <= f[i - 1, j - 1] && f[i, j - 1] <= f[i - 1, j])
                    {
                        f[i, j] = distance[i - 1, j - 1] + f[i, j - 1];
                    }
                    else if (f[i - 1, j - 1] <= f[i, j - 1] && f[i - 1, j - 1] <= f[i - 1, j])
                    {
                        f[i, j] = distance[i - 1, j - 1] + f[i - 1, j - 1];
                    }
                }
            }

            return f[x.Count, y.Count];
        }

        /// <summary>
        /// This version of DTW considers overlapping subsequences. The beginning of the reference sequence Y may be ignored, and the end of 
        /// both sequences X and Y may be ignored in the alignment. 
        /// The similarity of the best alignment is returned considering these differences.
        /// BASED ON: Michelet, S., Karp, K., Delaherche, E., Achard, C., & Chetouani, M. (2012). 
        /// Automatic imitation assessment in interaction. In Human Behavior Understanding (pp. 161-173). 
        /// Springer Berlin Heidelberg.
        /// </summary>
        /// <param name="comparison"></param>
        /// <param name="minDistance"></param>
        public void ComputeOverlapDTW(DTWComparison<T> comparison = null, double minDistance = double.MaxValue)
        {
            // Set first column to 0 => Which means that the beginning of the X sequence can be aligned at any point
            for (int i = 1; i <= x.Count; ++i)
                f[i, 0] = 0;

            // Set first row to 0 => Which means that the beginning of the Y sequence can be aligned at any point
            for (int i = 1; i <= y.Count; ++i)
                f[0, i] = 0;

            // Needleman Similarity DTW
            for (int i = 1; i <= x.Count; ++i)
            {
                for (int j = 1; j <= y.Count; ++j)
                {
                    double first = 0, second = 0, third = 0;

                    if (i >= 2)
                        first = f[i - 1, j] - (1 - comparison.Correlation(x[i - 2], x[i - 1]));

                    if (j >= 2)
                        second = f[i, j - 1] - (1 - comparison.Correlation(y[j - 2], y[j - 1]));

                    third = f[i - 1, j - 1] + comparison.Correlation(x[i - 1], y[j - 1]);

                    f[i, j] = Math.Max(first, Math.Max(second, third));
                }
            }

            // Look for the best alignment, without considering that the last frames need to be aligned
            // therefore both the last row and column need to be checked
            double best = double.MinValue;

            // Last row => The end of y could be ignored
            for (int i = 1; i < y.Count; ++i)
                if (f[x.Count, i] > best)
                    best = f[x.Count, i];

            // Last column => The end of x could be ignored
            for (int i = 1; i < x.Count; ++i)
                if (f[i, y.Count] > best)
                    best = f[i, y.Count];

            //best = f[x.Count, y.Count];
            sum = best;
        }
    }
}
