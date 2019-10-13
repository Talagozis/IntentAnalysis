using System;
using Microsoft.ML.Data;

namespace IntentAnalysis.DataStructures
{
    public class IntentData
    {
        [LoadColumn(0)]
        public string text;

        [LoadColumn(1)]
        public string label;
    }
}
