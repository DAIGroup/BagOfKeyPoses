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
    /// Implements a C# 4.0 2-Tuple but with an empty constructor
    /// </summary>
    public class MyTuple<T1, T2> : Tuple<T1, T2> where T1 : new() where T2 : new()
    {
        /// <summary>
        /// Creates a new 2-tuple with the default objects
        /// </summary>
        public MyTuple() : base(new T1(), new T2()) { }
    }


    /// <summary>
    /// Implements a queue with a fixed size, so it automatically dequeues.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class FixedSizedQueue<T> : Queue<T>
    {
        public int Size { get; private set; }

        public FixedSizedQueue(int size)
        {
            Size = size;
        }

        public new void Enqueue(T obj) 
        {
            lock (this)
            {
                base.Enqueue(obj);
                while (Count > Size)
                {
                    Dequeue();
                }
            }
        }
    }
}
