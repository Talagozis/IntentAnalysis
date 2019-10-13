using System;
using System.Linq;

namespace IntentAnalysis
{
    public class FullPrediction
    {
        public readonly string PredictedLabel;
        public readonly float Score;
        public readonly int OriginalSchemaIndex;

        public FullPrediction(string predictedLabel, float score, int originalSchemaIndex)
        {
            this.PredictedLabel = predictedLabel;
            this.Score = score;
            this.OriginalSchemaIndex = originalSchemaIndex;
        }

    }
}