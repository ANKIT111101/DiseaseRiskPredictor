using Microsoft.AspNetCore.Mvc;
using DiseaseRiskPredictor.Models;
using DiseaseRiskPredictor.Services;

namespace DiseaseRiskPredictor.Controllers
{
    public class HomeController : Controller
    {
        private readonly RiskPredictionService _svc;

        // ASP.NET automatically injects RiskPredictionService here
        public HomeController(RiskPredictionService svc)
        {
            _svc = svc;
        }

        // GET /  → show empty form + model metrics
        public IActionResult Index()
        {
            var vm = new PredictViewModel
            {
                ModelAccuracy = _svc.Accuracy,
                ModelF1 = _svc.F1Score,
                ModelAuc = _svc.AucRoc
            };
            return View(vm);
        }

        // POST /  → run prediction, show result
        [HttpPost]
        public IActionResult Index(PredictViewModel vm)
        {
            // Always carry metrics back to the view
            vm.ModelAccuracy = _svc.Accuracy;
            vm.ModelF1 = _svc.F1Score;
            vm.ModelAuc = _svc.AucRoc;

            if (!ModelState.IsValid)
                return View(vm);

            // Map the form values to ML input
            var input = new PatientData
            {
                Age = vm.Age != null ? (float)vm.Age : 0,
                BMI = vm.BMI != null ? (float)vm.BMI : 0,
                Glucose = vm.Glucose != null ? (float)vm.Glucose : 0,
                BloodPressure = vm.BloodPressure != null ? (float)vm.BloodPressure : 0,
                Insulin = vm.Insulin != null ? (float)vm.Insulin : 0,
                Pregnancies = vm.Pregnancies != null ? (float)vm.Pregnancies : 0,
                SkinThickness = 20,                         // default – not asked in form
                DiabetesPedigreeFunction = 0.5f             // default – not asked in form
            };

            (vm.Probability, vm.RiskLevel) = _svc.Predict(input);

            return View(vm);
        }
    }
}