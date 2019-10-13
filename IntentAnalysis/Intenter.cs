using System;
using System.Collections.Generic;
using System.Linq;
using IntentAnalysis.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace IntentAnalysis
{
    public class Intenter
    {
        private readonly PredictionEngine<IntentData, IntentPrediction> _predEngine;

        public Intenter(string modelPath, MLContext mlContext)
        {
            // Load model from file.
            ITransformer trainedModel = mlContext.Model.Load(modelPath, out DataViewSchema inputSchema);

            // Create prediction engine related to the loaded trained model.
            this._predEngine = mlContext.Model.CreatePredictionEngine<IntentData, IntentPrediction>(trainedModel);
        }

        public void testPredictionForSingleIssue(IntentData intentData)
        {
            // Predict labels and scores for single hard-coded issue.
            var prediction = this._predEngine.Predict(intentData);

            var fullPredictions = this.getBestThreePredictions(prediction);

            Console.WriteLine($"==== Displaying prediction of Issue with Title = {intentData.text} ====");

            Console.WriteLine("1st Label: " + fullPredictions[0].PredictedLabel + " with score: " + fullPredictions[0].Score);
            Console.WriteLine("2nd Label: " + fullPredictions[1].PredictedLabel + " with score: " + fullPredictions[1].Score);
            Console.WriteLine("3rd Label: " + fullPredictions[2].PredictedLabel + " with score: " + fullPredictions[2].Score);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.PredictedLabel} ===============");
        }

        private FullPrediction[] getBestThreePredictions(IntentPrediction prediction)
        {
            float[] scores = prediction.Score;
            int size = scores.Length;

            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            this._predEngine.OutputSchema[nameof(IntentPrediction.Score)].GetSlotNames(ref slotNames);

            this.getIndexesOfTopThreeScores(scores, size, out var index0, out var index1, out var index2);

            return new[]
            {
                new FullPrediction(slotNames.GetItemOrDefault(index0).ToString(), scores[index0], index0),
                new FullPrediction(slotNames.GetItemOrDefault(index1).ToString(), scores[index1], index1),
                new FullPrediction(slotNames.GetItemOrDefault(index2).ToString(), scores[index2], index2)
            };
        }

        private void getIndexesOfTopThreeScores(float[] scores, int n, out int index0, out int index1, out int index2)
        {
            int i;
            float first, second;
            index0 = index1 = index2 = 0;
            if (n < 3)
            {
                Console.WriteLine("Invalid Input");
                return;
            }
            var third = first = second = 000;
            for (i = 0; i < n; i++)
            {
                // If current element is  
                // smaller than first 
                if (scores[i] > first)
                {
                    third = second;
                    second = first;
                    first = scores[i];
                }
                // If arr[i] is in between first 
                // and second then update second 
                else if (scores[i] > second)
                {
                    third = second;
                    second = scores[i];
                }

                else if (scores[i] > third)
                    third = scores[i];
            }
            var scoresList = scores.ToList();
            index0 = scoresList.IndexOf(first);
            index1 = scoresList.IndexOf(second);
            index2 = scoresList.IndexOf(third);
        }

    }
}
