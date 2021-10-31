using System;
using System.ComponentModel;
using System.Linq;
using IntentAnalysis.Common;
using IntentAnalysis.DataStructures;
using Microsoft.ML;

namespace IntentAnalysis
{
    public class Program
    {
        private const string DATA_PATH = "Data/intents.tsv";
        private const string TEST_PATH = "Data/testData.tsv";

        private const string BASE_MODELS_RELATIVE_PATH = @"../../../MLModels/IntenterModel.zip";
        private static readonly string modelPath = PathHelper.getAbsolutePath(BASE_MODELS_RELATIVE_PATH);

        static void Main(string[] args)
        {
            //1. ChainedBuilderExtensions and Train the model
            buildAndTrainModel(DATA_PATH, TEST_PATH, modelPath, MyTrainerStrategy.SdcaMultiClassTrainer);

            //2. Try/test to predict a label for a single hard-coded Issue
            testSingleLabelPrediction();

            ConsoleHelper.consolePressAnyKey();
        }

        private static void buildAndTrainModel(string dataSetLocation, string testSetLocation, string modelPath, MyTrainerStrategy selectedStrategy)
        {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<IntentData>(dataSetLocation, hasHeader: true, separatorChar: '\t', allowSparse: false);
            var testingDataView = mlContext.Data.LoadFromTextFile<IntentData>(testSetLocation, hasHeader: true, separatorChar: '\t', allowSparse: false);

            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "label", inputColumnName: nameof(IntentData.label))
                .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "text", inputColumnName: nameof(IntentData.text)))
                .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "text"))
                .AppendCacheCheckpoint(mlContext)
                ;
                            // Use in-memory cache for small/medium datasets to lower training time. 
                            // Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.

            // (OPTIONAL) Peek data (such as 2 records) in training DataView after applying the ProcessPipeline's transformations into "Features" 
            ConsoleHelper.peekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 2);

            // STEP 3: Create the selected training algorithm/trainer
            IEstimator<ITransformer> trainer;
            switch (selectedStrategy)
            {
                case MyTrainerStrategy.SdcaMultiClassTrainer:
                    trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("label", "Features");
                    break;
                case MyTrainerStrategy.OvaAveragedPerceptronTrainer:
                    {
                        // Create a binary classification trainer.
                        var averagedPerceptronBinaryTrainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron("label", "Features", numberOfIterations: 10);
                        // Compose an OVA (One-Versus-All) trainer with the BinaryTrainer.
                        // In this strategy, a binary classification algorithm is used to train one classifier for each class, "
                        // which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers, "
                        // and choosing the prediction with the highest confidence score.
                        trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptronBinaryTrainer, labelColumnName: "label");
                        break;
                    }
                default:
                    throw new InvalidEnumArgumentException($"The value of '{nameof(selectedStrategy)}' is not an enum of type '{nameof(MyTrainerStrategy)}'.");
            }

            //Set the trainer/algorithm and map label to value (original readable state)
            var trainingPipeline = dataProcessPipeline.Append(trainer).Append(mlContext.Transforms.Conversion
                .MapKeyToValue("PredictedLabel"))
                ;

            // STEP 4: Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics

            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(data: testingDataView, estimator: trainingPipeline, numberOfFolds: 6, labelColumnName: "label");

            ConsoleHelper.printMulticlassClassificationFoldsAverageMetrics(trainer.ToString() ?? string.Empty, crossValidationResults);

            // STEP 5: Train the model fitting to the DataSet
            Console.WriteLine("=============== Training the model ===============");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // (OPTIONAL) Try/test a single prediction with the "just-trained model" (Before saving the model)
            var issue = new IntentData
            {
                text = "food"
            };

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<IntentData, IntentPrediction>(trainedModel);
            //Score
            var prediction = predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.PredictedLabel} ===============");
            //

            // STEP 6: Save/persist the trained model to a .ZIP file
            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);

            ConsoleHelper.consoleWriteHeader("Training process finalized");
        }


        private static void testSingleLabelPrediction()
        {
            var input = Console.ReadLine();
            while (input != "exit")
            {
                var intenter = new Intenter(modelPath: modelPath, new MLContext());

                var intent = new IntentData {text = input, label = string.Empty};

                intenter.testPredictionForSingleIssue(intent);

                input = Console.ReadLine();
            }
        }


    }
}