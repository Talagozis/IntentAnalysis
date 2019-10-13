using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace IntentAnalysis.DataStructures
{
    public class IntentPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel;

        public float[] Score;
    }
}