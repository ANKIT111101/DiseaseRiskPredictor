using DiseaseRiskPredictor.Services;

var builder = WebApplication.CreateBuilder(args);

// Add MVC
builder.Services.AddControllersWithViews();

// Register ML service as Singleton (trains once at startup)
builder.Services.AddSingleton<RiskPredictionService>();

var app = builder.Build();

// Warm up the model at startup so the first request is fast
app.Services.GetRequiredService<RiskPredictionService>();

app.UseStaticFiles();
app.UseRouting();

// Default route: /Home/Index  (or just /)
app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();