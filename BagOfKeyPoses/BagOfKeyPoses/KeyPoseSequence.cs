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
using System.Xml;

namespace BagOfKeyPoses
{
    /// <summary>
    /// This object handles a sequence of key poses.
    /// </summary>
    public class KeyPoseSequence
    {
        public List<KeyPose> Items;                             // Sequence of Key Poses
        public string ClassLabel;                               // Class label

        /// <summary>
        /// Constructor
        /// </summary>
        public KeyPoseSequence()
        {
            Items = new List<KeyPose>();
        }

        /// <summary>
        /// Returns an XML with the details of the key pose sequence
        /// </summary>
        /// <returns></returns>
        public XmlDocument ToXML(int sequenceCount)
        {
            XmlDocument xml = new XmlDocument();
            XmlNode sequenceNode = xml.CreateElement("key-pose-sequence");
            XmlAttribute sequenceNodeCount = xml.CreateAttribute("count");
            sequenceNodeCount.Value = sequenceCount.ToString();
            sequenceNode.Attributes.Append(sequenceNodeCount);
            XmlAttribute sequenceNodeAction = xml.CreateAttribute("action-class");
            sequenceNodeAction.Value = ClassLabel;
            sequenceNode.Attributes.Append(sequenceNodeAction);
            xml.AppendChild(sequenceNode);

            int frame = 0;
            foreach (var kp in Items)
            {                
                sequenceNode.AppendChild(xml.ImportNode(kp.ToXML(frame, false).DocumentElement, true));
                frame++;
            }            

            return xml;
        }
    }
}
