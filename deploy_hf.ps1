# ==============================================================================
# Deployment Helper Script for Hugging Face Spaces (Docker SDK)
# ==============================================================================
# Prerequisites:
# 1. Hugging Face account and User Access Token (Write permissions)
#    Get token at: https://huggingface.co/settings/tokens
# 2. Python 3.12 installed in environment
#
# Usage Example:
#   .\deploy_hf.ps1 -SpaceName "your-username/japanese-ocr-app" [-GeminiApiKey "your-api-key"]
# ==============================================================================

param (
    [Parameter(Mandatory=$true, HelpMessage="Hugging Face Space repository name (e.g., username/japanese-ocr-app)")]
    [string]$SpaceName,

    [Parameter(Mandatory=$false, HelpMessage="GEMINI_API_KEY secret to set on the Space")]
    [string]$GeminiApiKey
)

$ErrorActionPreference = "Stop"

Write-Host "=== Hugging Face Spaces Deployment ===" -ForegroundColor Cyan

# Step 1: Install / update huggingface_hub CLI
Write-Host "`n[Step 1/5] Installing/Updating huggingface_hub CLI..." -ForegroundColor Yellow
python -m pip install --quiet --upgrade "huggingface_hub[cli]"

# Step 2: Login to Hugging Face
Write-Host "`n[Step 2/5] Logging into Hugging Face..." -ForegroundColor Yellow
Write-Host "Please paste your Hugging Face User Access Token (Write permission required) when prompted." -ForegroundColor Gray
huggingface-cli login

# Step 3: Create Private Space with Docker SDK
Write-Host "`n[Step 3/5] Creating Private Docker Space: $SpaceName..." -ForegroundColor Yellow
huggingface-cli repo create $SpaceName --type space --space_sdk docker --private

# Step 4: Push code to Space
Write-Host "`n[Step 4/5] Uploading project files to Space..." -ForegroundColor Yellow
huggingface-cli upload $SpaceName . . --repo-type space

# Step 5: Set secret variables
if ($GeminiApiKey) {
    Write-Host "`n[Step 5/5] Setting GEMINI_API_KEY secret on Space..." -ForegroundColor Yellow
    python -c "from huggingface_hub import HfApi; api = HfApi(); api.add_space_secret('$SpaceName', 'GEMINI_API_KEY', '$GeminiApiKey')"
    Write-Host "Secret GEMINI_API_KEY successfully configured." -ForegroundColor Green
} else {
    Write-Host "`n[Step 5/5] GEMINI_API_KEY parameter was omitted." -ForegroundColor Yellow
    Write-Host "Remember to set GEMINI_API_KEY manually in Settings -> Variables and secrets on Hugging Face." -ForegroundColor Gray
}

Write-Host "`n=== Deployment Completed Successfully ===" -ForegroundColor Green
Write-Host "View your application at: https://huggingface.co/spaces/$SpaceName" -ForegroundColor Cyan
