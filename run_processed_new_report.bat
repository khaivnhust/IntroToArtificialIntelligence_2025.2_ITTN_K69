@echo off
setlocal

cd /d "%~dp0"

echo ============================================================
echo Running full report pipeline on data\processed_new
echo ============================================================

python scripts\run_report_pipeline.py --profile full --stages check-data,train-hybrid,train-compare,test --data-dir data\processed_new --visual-features data\processed_new\visual_features_full.npz --checkpoint-dir checkpoints\processed_new --output-root reports\report_pipeline_processed_new --models popularity,mf,ncf,hybrid
if errorlevel 1 (
    echo.
    echo Pipeline failed. Check logs under reports\report_pipeline_processed_new\full\logs
    exit /b 1
)

echo.
echo ============================================================
echo Generating report diagnostics and plots
echo ============================================================

python scripts\generate_report_diagnostics.py --data-dir data\processed_new --visual-features data\processed_new\visual_features_full.npz --checkpoint-dir checkpoints\processed_new --pipeline-output-dir reports\report_pipeline_processed_new\full --output-dir reports\report_pipeline_processed_new\full\diagnostics --max-eval-users all --negative-candidates 1000
if errorlevel 1 (
    echo.
    echo Diagnostics failed. Check reports\report_pipeline_processed_new\full\diagnostics
    exit /b 1
)

echo.
echo Done.
echo Main outputs: reports\report_pipeline_processed_new\full
echo Diagnostics:  reports\report_pipeline_processed_new\full\diagnostics

endlocal
