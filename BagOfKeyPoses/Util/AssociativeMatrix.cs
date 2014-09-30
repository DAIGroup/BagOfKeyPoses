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

namespace Util
{
    /// <summary>
    /// Implements a bidimensional associative matrix, behaves similar to Dictionary[V, Dictionary[X, T]] but offers transparency
    /// and a fail-resistant [] operator (as a PHP associative matrix).
    /// </summary>
    public class AssociativeMatrix<V, X, T> : MarshalByRefObject where T : new() 
    {
        // Private
        private Dictionary<V, AssociativeArray<X, T>> matrix;       // Matrix (hash of rows)
        private int total;                                          // Total number of elements
        private int rows;                                           // Total number of rows

        /// <summary>
        /// Used row keys
        /// </summary>
        public List<V> RowKeys
        {
            get { return matrix.Keys.ToList(); }
        }

        /// <summary>
        /// Number of elements
        /// </summary>
        public int Total
        {
          get { return total; }
        }

        /// <summary>
        /// Number of rows
        /// </summary>
        public int Rows
        {
            get { return rows; }
        }

        /// <summary>
        /// Returns the number of elements of a specific row
        /// </summary>
        /// <param name="rowKey"></param>
        /// <returns></returns>
        public int TotalRow(V rowKey)
        {
            if (matrix.ContainsKey(rowKey))
                return matrix[rowKey].Total;
            else
                return 0;
        }

        /// <summary>
        /// Creates a new bidimensional associative matrix
        /// </summary>
        public AssociativeMatrix()
        {
            matrix = new Dictionary<V, AssociativeArray<X, T>>();
        }

        /// <summary>
        /// Hashes the value with the keys, will not fail if key exists already. (No need to call ContainsKey and Add as at Dictionary)
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        /// <returns>Returns true if an element was lost by overwriting</returns>
        public bool Set(V key1, X key2, T value)
        {
            bool result = false;

            if (matrix.ContainsKey(key1))
            {
                if (!matrix[key1].Set(key2, value)) // False == new element
                    total++;
                else
                    result = true;
            }
            else
            {
                matrix.Add(key1, new AssociativeArray<X, T>());
                matrix[key1].Set(key2, value);
                total++;
                rows++;
            }

            return result;
        }

        /// <summary>
        /// Returns true if row exists already
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        public bool ContainsRow(V key)
        {
            return matrix.ContainsKey(key);
        }

        /// <summary>
        /// Returns true if keys exist already
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        public bool ContainsKeys(V key1, X key2)
        {
            if (matrix.ContainsKey(key1))
                return matrix[key1].ContainsKey(key2);
            else
                return false;
        }

        /// <summary>
        /// Tries to get the value and returns wether or not it existed.
        /// This is cheaper than ContainsKey + GetValue
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public bool TryGetValue(V key1, X key2, out T value)
        {
            AssociativeArray<X, T> array;

            if (matrix.TryGetValue(key1, out array))
            {
                if (array.TryGetValue(key2, out value))
                    return true;
            }
            else
                value = default(T);

            return false;
        }

        /// <summary>
        /// [,] Operator, get will return value if available, otherwise an empty object will be created, added and returned, 
        /// set will set value, even if key doesn't exist yet (no need to call ContainsKey first).
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value">Returns true if an element was lost by overwriting</param>
        /// <returns></returns>
        public T this[V key1, X key2]
        {
            get
            {
                if (matrix.ContainsKey(key1))
                    return matrix[key1][key2];
                else
                {
                    T newObject = new T();
                    Set(key1, key2, newObject);
                    return newObject;
                }
            }

            set
            {
                Set(key1, key2, value);
            }
        }

        /// <summary>
        /// [] Operator, get will return AssociativeArray if available, otherwise an empty object will be created, added and returned, 
        /// set will set value, even if key doesn't exist yet (no need to call ContainsKey first).
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value">Returns true if an element was lost by overwriting</param>
        /// <returns></returns>
        public AssociativeArray<X, T> this[V key1]
        {
            get
            {
                if (matrix.ContainsKey(key1))
                    return matrix[key1];
                else
                {
                    AssociativeArray<X, T> newArray = new AssociativeArray<X, T>();
                    matrix.Add(key1, newArray);
                    rows++;
                    return newArray;
                }
            }

            set
            {
                matrix.Add(key1, value);
                rows++;
            }
        }

        /// <summary>
        /// Returns the keys of a specific row
        /// </summary>
        /// <param name="rowKey"></param>
        /// <returns></returns>
        public List<X> ColumnKeys(V rowKey)
        {
            if (matrix.ContainsKey(rowKey))
                return matrix[rowKey].Keys;
            else
                return null;
        }
    }

    /// <summary>
    /// Handles the confusion matrix data with an AssociativeMatrix[string, string]
    /// and the related average success rates.
    /// </summary>
    public class TestResults : AssociativeMatrix<string, string, int>
    {
        // Public
        public double SuccessRate;      // Overall success rate of the test
        public double F1Score;          // F1 score of the test

        /// <summary>
        /// Writes the AssociativeMatrix as a CSV file, specially designed for the results matrix which has string keys, and symmetric rows and columns.
        /// </summary>
        /// <param name="filename"></param>
        public StringBuilder CreateCSV()
        {
            // Create csv
            List<string> classes = RowKeys;

            foreach (var row in RowKeys)
                classes = classes.Union(ColumnKeys(row)).ToList();

            StringBuilder sb = new StringBuilder();
            string[] headers = new string[classes.Count];
            Dictionary<string, int> columns = new Dictionary<string, int>();
            int r = 0;
            foreach (string row in classes)
            {
                columns.Add(row, columns.Count);
                headers[r++] = row;
            }
            sb.AppendLine(StringArrayToString(headers));

            // Add rows looping through the matrix
            r = 0;
            foreach (string rowKey in RowKeys)
            {
                List<string> columnKeys = ColumnKeys(rowKey);

                // Get total number of recognitions
                int sum = 0;
                for (int i = 0; i < columnKeys.Count; ++i)
                    sum += this[rowKey, columnKeys[i]];

                // Add cells
                string[] cells = new string[columns.Count + 1];
                cells[0] = rowKey;
                for (int i = 0; i < columnKeys.Count; ++i)
                {
                    if (columns.ContainsKey(columnKeys[i]))
                    {
                        int count = this[rowKey, columnKeys[i]];
                        int cell = columns[columnKeys[i]] + 1; // Each row has different columns as it's a associative matrix

                        if (count != 0)
                            cells[cell] = count + "/" + sum;
                        else
                            cells[cell] = "0";
                    }
                }

                sb.AppendLine(StringArrayToString(cells));
                r++;
            }

            return sb;
        }

        public const string Delimiter = ", ";

        /// <summary>
        /// Returns a string with the elements of the array adding a delimiter
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static string StringArrayToString(string[] a)
        {
            string s = "";
            for (int i = 0; i < a.Length; ++i)
            {
                s += a[i];

                if (i < a.Length - 1)
                    s += Delimiter;
            }

            return s;
        }
    }

    #region AssociativeArray
    /// <summary>
    /// Implements an associative array of one dimension.
    /// </summary>
    public class AssociativeArray<W, U> : MarshalByRefObject where U:new() 
    {
        // Private
        private Dictionary<W, U> array;                             // Row
        private int total;                                          // Total number of elements
        
        public Dictionary<W, U> Dictionary
        {
            get { return array; }
        }

        /// <summary>
        /// Used Keys
        /// </summary>
        public List<W> Keys
        {
            get { return array.Keys.ToList(); }
        }

        /// <summary>
        /// Number of elements
        /// </summary>
        public int Total
        {
            get { return total; }
        }

        /// <summary>
        /// Creates a new one-dimensional associative array (just like a Dictionary)
        /// </summary>
        public AssociativeArray()
        {
            array = new Dictionary<W, U>();
            total = 0;
        }

        /// <summary>
        /// Hashes the value with the key, will not fail if key exists already. (No need to call ContainsKey and Add as at Dictionary)
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        /// <returns>Returns true if an element was lost by overwriting</returns>
        public bool Set(W key, U value)
        {
            bool result = false;

            if (array.ContainsKey(key))
            {
                array[key] = value;
                result = true;
            }
            else
            {
                array.Add(key, value);
                total++;
            }

            return result;
        }

        /// <summary>
        /// Returns true if key exists already
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        public bool ContainsKey(W key)
        {
            return array.ContainsKey(key);
        }


        /// <summary>
        /// Tries to get the value and returns wether or not it existed.
        /// This is cheaper than ContainsKey + GetValue
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public bool TryGetValue(W key, out U value)
        {
            if (array.TryGetValue(key, out value))
                return true;
            else
                return false;
        }

        /// <summary>
        /// [] Operator, get will return object if available, otherwise an empty object will be created, added and returned, 
        /// set will set value, even if key doesn't exist yet (no need to call ContainsKey first).
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value">Returns true if an element was lost by overwriting</param>
        /// <returns></returns>
        public U this[W key]
        {
            get
            {
                if (array.ContainsKey(key))
                    return array[key];
                else
                {
                    U newObject = new U();                   
                    Set(key, newObject);
                    return newObject;
                }
            }

            set
            {
                Set(key, value);
            }
        }
    }
    #endregion
}
