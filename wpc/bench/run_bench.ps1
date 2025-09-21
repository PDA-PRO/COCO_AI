# run_bench.ps1
param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("before","after")]
  [string]$Tag,

  [string]$Image = "bench-common:cuda12.4",
  [string]$Inputs = "test_code_fairs.json",
  [int]$Warmup = 200,

  # 선택 파라미터
  [string]$Cpus = "8",
  [string]$Memory = "16g",
  [int]$GpuIndex = -1,             # 특정 GPU만 보이게 하려면 0 등의 값. -1이면 전체
  [int]$DmonIntervalSec = 1,       # nvidia-smi dmon 샘플링 주기(초)
  [int]$StatsIntervalSec = 1       # docker stats 샘플링 주기(초)
)

# 경로 및 파일명
$WorkDir = (Get-Location).Path
$LogsDir = Join-Path $WorkDir "bench"
$LogsDir = Join-Path $LogsDir "logs"
$BenchName = "bench-$Tag"
$DmonName  = "gpu-dmon-$Tag"

$GpuCsv    = Join-Path $LogsDir "gpu_$Tag.csv"
$StatsCsv  = Join-Path $LogsDir "docker_stats_$Tag.csv"

$GpuCsvIn   = "./logs/gpu_$Tag.csv"
$BenchCsvIn = "./logs/${Tag}_single.csv"

# 사전 준비
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

# 기존 동일 이름 컨테이너 정리(있다면)
docker rm -f $DmonName 2>$null | Out-Null
docker rm -f $BenchName 2>$null | Out-Null

# GPU 로거 환경변수(특정 GPU만 노출하고 싶을 때)
$CudaEnv = @()
if ($GpuIndex -ge 0) { $CudaEnv = @("-e","CUDA_VISIBLE_DEVICES=$GpuIndex") }

# docker stats 백그라운드 잡 함수(컨테이너 종료 시 자동 종료됨)
$statsScript = {
  param($containerName, $outPath, $intervalSec)
  while ($true) {
    $line = docker stats $containerName --no-stream --format "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}}"
    if (-not $line) { break }
    $line | Out-File -Append $outPath
    Start-Sleep -Seconds $intervalSec
  }
}

# 실행
$statsJob = $null
try {
  # 1) GPU 로거 컨테이너 (nvidia-smi dmon)
  $dmonCmd = @(
    "run","-d","--name",$DmonName,"--gpus","all"
  ) + $CudaEnv + @(
    "-v","${WorkDir}:/work","-w","/work",
    "nvidia/cuda:12.4.1-base-ubuntu22.04",
    "bash","-lc","nvidia-smi dmon -s pucmt -d $DmonIntervalSec -o DT -f `"$GpuCsvIn`""
  )
  docker @dmonCmd | Out-Null

  # 2) docker stats 로깅(호스트 PowerShell 잡)
  $statsJob = Start-Job -ScriptBlock $statsScript -ArgumentList $BenchName,$StatsCsv,$StatsIntervalSec

  # 3) 벤치마크 컨테이너
  $benchCmd = @(
    "run","--rm","--gpus","all","--cpus",$Cpus,"--memory",$Memory,"--name",$BenchName
  ) + $CudaEnv + @(
    "-v","${WorkDir}:/work","-w","/work",$Image,
    "bash","-lc","python3 -m bench.bench_single_gpu.py --inputs `"$Inputs`" --warmup $Warmup"
  )
  docker @benchCmd

} finally {
  if ($statsJob) {
    # 먼저 정중히 중지
    Stop-Job -Job $statsJob -ErrorAction SilentlyContinue | Out-Null
    # 잠깐 대기
    Wait-Job -Job $statsJob -Timeout 3 | Out-Null
    # 그래도 남아있으면 강제 제거
    Remove-Job -Job $statsJob -Force -ErrorAction SilentlyContinue | Out-Null
  }
  docker stop $DmonName 2>$null | Out-Null
  docker rm   $DmonName 2>$null | Out-Null
  Write-Host "Done. Logs at: $LogsDir"
  Write-Host " - GPU:     $GpuCsv"
  Write-Host " - CPU/Mem: $StatsCsv"
}
