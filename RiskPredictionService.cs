using Microsoft.ML;
using DiseaseRiskPredictor.Models;

namespace DiseaseRiskPredictor.Services
{
    public class RiskPredictionService
    {
        private readonly MLContext _ml;
        private readonly PredictionEngine<PatientData, RiskPrediction> _engine;

        public double Accuracy { get; private set; }
        public double F1Score { get; private set; }
        public double AucRoc { get; private set; }

        public RiskPredictionService(IWebHostEnvironment env)
        {
            _ml = new MLContext(seed: 42);

            // ── 1. Load CSV ──────────────────────────────────────────────
            var dataPath = Path.Combine(env.ContentRootPath, "Data", "diabetes.csv");
            var data = _ml.Data.LoadFromTextFile<PatientData>(
                            dataPath, hasHeader: true, separatorChar: ',');

            // ── 2. Train / Test split ────────────────────────────────────
            var split = _ml.Data.TrainTestSplit(data, testFraction: 0.2);

            // ── 3. Pipeline: combine features → normalise → FastTree ─────
            var pipeline =
                _ml.Transforms
                   .Concatenate("Features",
                       nameof(PatientData.Pregnancies),
                       nameof(PatientData.Glucose),
                       nameof(PatientData.BloodPressure),
                       nameof(PatientData.SkinThickness),
                       nameof(PatientData.Insulin),
                       nameof(PatientData.BMI),
                       nameof(PatientData.DiabetesPedigreeFunction),
                       nameof(PatientData.Age))
                   .Append(_ml.Transforms.NormalizeMinMax("Features"))
                   .Append(_ml.BinaryClassification.Trainers.FastTree(
                               numberOfLeaves: 50,
                               numberOfTrees: 100,
                               minimumExampleCountPerLeaf: 10));

            // ── 4. Train ─────────────────────────────────────────────────
            var model = pipeline.Fit(split.TrainSet);

            // ── 5. Evaluate ──────────────────────────────────────────────
            var metrics = _ml.BinaryClassification
                             .Evaluate(model.Transform(split.TestSet));
            Accuracy = metrics.Accuracy;
            F1Score = metrics.F1Score;
            AucRoc = metrics.AreaUnderRocCurve;

            // ── 6. Create prediction engine (thread-safe singleton) ──────
            _engine = _ml.Model.CreatePredictionEngine<PatientData, RiskPrediction>(model);
        }

        // Call this from the controller
        public (float probability, string riskLevel) Predict(PatientData input)
        {
            var result = _engine.Predict(input);

            string level = result.Probability switch
            {
                < 0.40f => "Low",
                < 0.70f => "Medium",
                _ => "High"
            };

            return (result.Probability, level);
        }
    }
}