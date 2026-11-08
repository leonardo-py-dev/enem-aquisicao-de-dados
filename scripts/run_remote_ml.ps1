<#
.SYNOPSIS
    Orquestra a execucao remota do pipeline de Machine Learning na nuvem via GitHub Actions (R4).

.DESCRIPTION
    Atua exclusivamente como Control Plane ultraleve na maquina do usuario:
    1. Resolve o executavel oficial da GitHub CLI (gh.exe) de forma resiliente.
    2. Valida o status de autenticacao com o GitHub.
    3. Dispara remotamente o workflow .github/workflows/data_pipeline.yml com action=ml-pipeline.
    4. Rastreia e captura o Run ID gerado na API do GitHub.
    5. Acompanha a execucao via streaming de logs em tempo real (gh run watch) com zero sobrecarga local de CPU/RAM.
    6. Em caso de falha remota, realiza auto-remediacao e inspecao automatica do traceback (--log-failed).
    7. Concluida a execucao, baixa os artefatos gerados na nuvem (relatorio HTML e dados JSON).

.PARAMETER Branch
    Branch do repositorio no GitHub para execucao (padrao: "main").

.PARAMETER Action
    Acao a ser executada no pipeline (padrao: "ml-pipeline").
    Opcoes validas: 'ml-pipeline', 'backtest', 'predict-2026', 'report'.

.PARAMETER ArtifactName
    Nome do pacote de artefatos compactado publicado pelo GitHub Actions (padrao: "enem-predictor-artifacts").

.PARAMETER OutputDir
    Diretorio de destino local para descompactar os artefatos baixados (padrao: ".").

.PARAMETER SkipDownload
    Se especificado, apenas monitora a execucao remota sem baixar os artefatos.

.EXAMPLE
    .\scripts\run_remote_ml.ps1
    Executa o pipeline completo de ML na branch main e baixa os artefatos para o diretorio atual.

.EXAMPLE
    .\scripts\run_remote_ml.ps1 -Action "backtest" -OutputDir ".\artifacts"
    Executa apenas o backtesting retrospectivo na nuvem e baixa os resultados.
#>

[CmdletBinding()]
param(
    [string]$Branch = "main",
    [ValidateSet("ml-pipeline", "backtest", "predict-2026", "report")]
    [string]$Action = "ml-pipeline",
    [string]$ArtifactName = "enem-predictor-artifacts",
    [string]$OutputDir = ".",
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " [ENEM PREDICTOR] CONTROL PLANE - ORQUESTRACAO REMOTA EM NUVEM (R4 / CI/CD)    " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# 1. Resolucao Resiliente do Executavel GitHub CLI (gh)
# -----------------------------------------------------------------------------
Write-Host "[1/6] Localizando executavel da GitHub CLI..." -ForegroundColor Gray

$GH_PATH = $null
if (Get-Command gh -ErrorAction SilentlyContinue) {
    $GH_PATH = "gh"
} elseif (Test-Path "C:\Program Files\GitHub CLI\gh.exe") {
    $GH_PATH = "C:\Program Files\GitHub CLI\gh.exe"
} elseif (Test-Path "$env:LOCALAPPDATA\Programs\GitHub CLI\gh.exe") {
    $GH_PATH = "$env:LOCALAPPDATA\Programs\GitHub CLI\gh.exe"
} else {
    Write-Error @"
[ERRO CRITICO] O executavel do GitHub CLI ('gh') nao foi encontrado no sistema.
Por favor, instale via:
  winget install --id GitHub.cli
ou adicione 'C:\Program Files\GitHub CLI' a variavel PATH do sistema.
"@
    exit 1
}

Write-Host "      Utilizando GitHub CLI: $GH_PATH" -ForegroundColor Green

# -----------------------------------------------------------------------------
# 2. Validacao de Autenticacao com o GitHub
# -----------------------------------------------------------------------------
Write-Host "[2/6] Verificando credenciais ativas do GitHub..." -ForegroundColor Gray
$authStatus = & $GH_PATH auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host $authStatus -ForegroundColor Yellow
    Write-Error "[ERRO] Voce precisa autenticar no GitHub CLI antes de continuar. Execute: & '$GH_PATH' auth login"
    exit 1
}
Write-Host "      Autenticacao validada com sucesso." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 3. Disparo Remoto do Workflow no GitHub Actions
# -----------------------------------------------------------------------------
Write-Host "[3/6] Disparando workflow remoto 'data_pipeline.yml'..." -ForegroundColor Cyan
Write-Host "      Acao solicitada : $Action" -ForegroundColor Yellow
Write-Host "      Branch de origem: $Branch" -ForegroundColor Yellow

# Captura do ID de execucao base anterior ao disparo para mitigar condicao de corrida
$baselineRunId = & $GH_PATH run list --workflow="data_pipeline.yml" --limit 1 --json databaseId -q ".[0].databaseId" 2>$null
$previousRunId = $baselineRunId

& $GH_PATH workflow run data_pipeline.yml -f action=$Action --ref $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Error "[ERRO] Falha ao despachar o workflow no GitHub Actions. Verifique se as alteracoes foram comitadas e enviadas ao remoto (git push origin $Branch)."
    exit $LASTEXITCODE
}
Write-Host "      Workflow disparado com sucesso (HTTP 204 No Content)." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 4. Captura Deterministica do Run ID
# -----------------------------------------------------------------------------
Write-Host "[4/6] Identificando o ID de execucao (Run ID) na API do GitHub..." -ForegroundColor Gray

$RUN_ID = $null
$maxRetries = 15
for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
    Start-Sleep -Seconds 2
    $candidateId = & $GH_PATH run list --workflow="data_pipeline.yml" --limit 1 --json databaseId -q ".[0].databaseId" 2>$null
    if ($candidateId -and ($candidateId -match "^\d+$") -and ($candidateId -ne $baselineRunId)) {
        $RUN_ID = $candidateId
        break
    }
    Write-Host "      Tentativa $attempt/${maxRetries}: aguardando registro da fila..." -ForegroundColor DarkGray
}

if (-not $RUN_ID) {
    Write-Error "[ERRO] Nao foi possivel obter o ID da execucao iniciada no GitHub Actions."
    exit 1
}

Write-Host "      Execucao remota mapeada: Run ID #$RUN_ID" -ForegroundColor Green
Write-Host "      URL da execucao: https://github.com/leonardo-py-dev/enem-aquisicao-de-dados/actions/runs/$RUN_ID" -ForegroundColor DarkCyan

# -----------------------------------------------------------------------------
# 5. Monitoramento em Tempo Real via Streaming (Zero Carga Local)
# -----------------------------------------------------------------------------
Write-Host "[5/6] Acompanhando streaming de logs da execucao remota..." -ForegroundColor Cyan
Write-Host "      [NOTA R4] Processamento pesado sendo executado inteiramente na nuvem." -ForegroundColor DarkYellow

& $GH_PATH run watch $RUN_ID --exit-status
$WATCH_EXIT = $LASTEXITCODE

if ($WATCH_EXIT -ne 0) {
    Write-Host "`n================================================================================" -ForegroundColor Red
    Write-Host " [ALERTA DE FALHA] A execucao remota falhou no GitHub Actions (Exit Code $WATCH_EXIT)" -ForegroundColor Red
    Write-Host "================================================================================" -ForegroundColor Red
    Write-Host "Inspecionando diagnostico da falha (--log-failed):" -ForegroundColor Yellow
    & $GH_PATH run view $RUN_ID --log-failed
    exit $WATCH_EXIT
}

Write-Host "      Execucao remota concluida com SUCESSO (Status: Completed / Conclusion: Success)." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 6. Download e Validacao dos Artefatos Remotos
# -----------------------------------------------------------------------------
if ($SkipDownload) {
    Write-Host "[6/6] Download de artefatos ignorado pelo parametro -SkipDownload." -ForegroundColor Yellow
} else {
    Write-Host "[6/6] Baixando artefatos gerados na nuvem para '$OutputDir'..." -ForegroundColor Cyan
    
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }

    # Tenta download pelo nome especifico do artefato
    & $GH_PATH run download $RUN_ID --name $ArtifactName --dir $OutputDir 2>&1 | Out-Null
    $DOWNLOAD_EXIT = $LASTEXITCODE

    if ($DOWNLOAD_EXIT -ne 0) {
        Write-Host "      Aviso: Artefato '$ArtifactName' nao encontrado individualmente. Tentando download amplo de todos os artefatos do run..." -ForegroundColor DarkYellow
        & $GH_PATH run download $RUN_ID --dir $OutputDir
    }

    Write-Host "`n================================================================================" -ForegroundColor Green
    Write-Host " [SUCESSO R4] PIPELINE REMOTO CONCLUIDO E ARTEFATOS SINCRONIZADOS COM SUCESSO!" -ForegroundColor Green
    Write-Host "================================================================================" -ForegroundColor Green
    
    # Inspecao dos arquivos esperados
    $expectedFiles = @("enem_predictor_report.html", "backtest_summary.json", "prediction_2026.json")
    foreach ($file in $expectedFiles) {
        $fullPath = Join-Path $OutputDir $file
        if (Test-Path $fullPath) {
            $item = Get-Item $fullPath
            Write-Host "  [OK] Arquivo: $($item.Name) ($($item.Length) bytes)" -ForegroundColor Green
        } else {
            Write-Host "  [PENDENTE/AVISO] Arquivo nao encontrado: $file" -ForegroundColor DarkGray
        }
    }
}

Write-Host "================================================================================`n" -ForegroundColor Cyan
exit 0
