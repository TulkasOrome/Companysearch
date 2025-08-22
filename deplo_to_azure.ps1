# deploy_to_azure.ps1
# Azure Web App Deployment Script for Company Search Platform

# Variables
$RESOURCE_GROUP = "company-search-rg"
$APP_NAME = "company-search-app-267"
$LOCATION = "Australia East"

Write-Host "=====================================" -ForegroundColor Green
Write-Host "Azure Web App Deployment Script" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Step 1: Configure App Service for Python
Write-Host "`nStep 1: Configuring Python runtime..." -ForegroundColor Yellow
az webapp config set `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME `
    --linux-fx-version "PYTHON|3.11"

# Step 2: Set startup command
Write-Host "`nStep 2: Setting startup command..." -ForegroundColor Yellow
az webapp config set `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME `
    --startup-file "python -m streamlit run ui/streamlit_app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true"

# Step 3: Configure app settings (environment variables)
Write-Host "`nStep 3: Setting environment variables..." -ForegroundColor Yellow

# Set Azure OpenAI credentials
az webapp config appsettings set `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME `
    --settings `
    AZURE_OPENAI_KEY="CUxPxhxqutsvRVHmGQcmH59oMim6mu55PjHTjSpM6y9UwIxwVZIuJQQJ99BFACL93NaXJ3w3AAABACOG3kI1" `
    AZURE_OPENAI_ENDPOINT="https://amex-openai-2025.openai.azure.com/" `
    AZURE_API_VERSION="2024-02-01" `
    SERPER_API_KEY="99c44b79892f5f7499accf2d7c26d93313880937" `
    SCM_DO_BUILD_DURING_DEPLOYMENT="true" `
    WEBSITES_PORT="8000" `
    WEBSITE_HTTPLOGGING_RETENTION_DAYS="7"

# Step 4: Configure deployment source (if using local git)
Write-Host "`nStep 4: Configuring deployment method..." -ForegroundColor Yellow
$deploymentChoice = Read-Host "Choose deployment method: [1] ZIP deploy, [2] Local Git (enter 1 or 2)"

if ($deploymentChoice -eq "2") {
    # Configure local git
    az webapp deployment source config-local-git `
        --resource-group $RESOURCE_GROUP `
        --name $APP_NAME

    # Get deployment credentials
    $creds = az webapp deployment list-publishing-credentials `
        --resource-group $RESOURCE_GROUP `
        --name $APP_NAME `
        --query "{username:publishingUserName, password:publishingPassword}" `
        --output json | ConvertFrom-Json

    Write-Host "`nGit remote URL:" -ForegroundColor Green
    Write-Host "https://$($creds.username)@$APP_NAME.scm.azurewebsites.net/$APP_NAME.git"
}

# Step 5: Create ZIP file for deployment (if ZIP deploy)
if ($deploymentChoice -eq "1") {
    Write-Host "`nStep 5: Creating deployment package..." -ForegroundColor Yellow

    # Files and folders to include
    $includeItems = @(
        "agents",
        "config",
        "core",
        "ui",
        "sessions",
        "requirements.txt",
        ".env",
        "startup.txt",
        "runtime.txt"
    )

    # Check if files exist
    $missingItems = @()
    foreach ($item in $includeItems) {
        if (-not (Test-Path $item)) {
            $missingItems += $item
        }
    }

    if ($missingItems.Count -gt 0) {
        Write-Host "Warning: The following items are missing:" -ForegroundColor Red
        $missingItems | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
        $continue = Read-Host "Continue anyway? (y/n)"
        if ($continue -ne "y") {
            exit
        }
    }

    # Create deployment package
    $zipFile = "deploy.zip"
    if (Test-Path $zipFile) {
        Remove-Item $zipFile
    }

    Write-Host "Creating ZIP file..." -ForegroundColor Yellow

    # Use PowerShell compression
    $compress = @{
        Path = $includeItems | Where-Object { Test-Path $_ }
        CompressionLevel = "Fastest"
        DestinationPath = $zipFile
    }
    Compress-Archive @compress

    # Deploy the ZIP file
    Write-Host "`nDeploying to Azure..." -ForegroundColor Yellow
    az webapp deploy `
        --resource-group $RESOURCE_GROUP `
        --name $APP_NAME `
        --src-path $zipFile `
        --type zip
}

# Step 6: Enable logging
Write-Host "`nStep 6: Enabling application logging..." -ForegroundColor Yellow
az webapp log config `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME `
    --application-logging filesystem `
    --detailed-error-messages true `
    --failed-request-tracing true `
    --level information

# Step 7: Restart the app
Write-Host "`nStep 7: Restarting application..." -ForegroundColor Yellow
az webapp restart `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME

# Step 8: Check deployment status
Write-Host "`nStep 8: Checking deployment status..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

$appUrl = "https://$APP_NAME.azurewebsites.net"
Write-Host "`n=====================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "App URL: $appUrl" -ForegroundColor Cyan
Write-Host "`nOpening browser..." -ForegroundColor Yellow
Start-Process $appUrl

# View logs
Write-Host "`nTo view logs, run:" -ForegroundColor Yellow
Write-Host "az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME" -ForegroundColor Cyan

# Troubleshooting info
Write-Host "`n=====================================" -ForegroundColor Yellow
Write-Host "Troubleshooting Commands:" -ForegroundColor Yellow
Write-Host "=====================================" -ForegroundColor Yellow
Write-Host "View logs:           az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME"
Write-Host "SSH into container:  az webapp ssh --resource-group $RESOURCE_GROUP --name $APP_NAME"
Write-Host "View configuration:  az webapp config show --resource-group $RESOURCE_GROUP --name $APP_NAME"
Write-Host "Restart app:         az webapp restart --resource-group $RESOURCE_GROUP --name $APP_NAME"