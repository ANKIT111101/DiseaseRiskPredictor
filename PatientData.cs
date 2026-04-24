using Microsoft.ML.Data;

namespace DiseaseRiskPredictor.Models
{
    // This maps to each row in diabetes.csv
    public class PatientData
    {
        [LoadColumn(0)] public float Pregnancies { get; set; }
        [LoadColumn(1)] public float Glucose { get; set; }
        [LoadColumn(2)] public float BloodPressure { get; set; }
        [LoadColumn(3)] public float SkinThickness { get; set; }
        [LoadColumn(4)] public float Insulin { get; set; }
        [LoadColumn(5)] public float BMI { get; set; }
        [LoadColumn(6)] public float DiabetesPedigreeFunction { get; set; }
        [LoadColumn(7)] public float Age { get; set; }

        [LoadColumn(8), ColumnName("Label")]
        public bool Outcome { get; set; }
    }

    // This holds the ML model's output after prediction
    public class RiskPrediction
    {
        [ColumnName("PredictedLabel")] public bool IsHighRisk { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }

    // This is what the user fills in on the form
    public class PredictViewModel
    {
        public float? Age { get; set; }
        public float? BMI { get; set; }
        public float? Glucose { get; set; }
        public float? BloodPressure { get; set; }
        public float? Insulin { get; set; }
        public float? Pregnancies { get; set; }

        // These are filled after prediction
        public float? Probability { get; set; }
        public string? RiskLevel { get; set; }
        public double ModelAccuracy { get; set; }
        public double ModelF1 { get; set; }
        public double ModelAuc { get; set; }
    }
}