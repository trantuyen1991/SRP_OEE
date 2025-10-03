<# 
release.ps1
Automate: add -> commit -> push branch -> tag -> push tag (optional: GitHub Release)

Usage:
  .\release.ps1 -Version v1.1.0 -Message "feat: add OEE daily metrics"
  .\release.ps1 -Version v1.1.1 -Message "fix: handle zero planned_sec" -Branch main -Paths "." 
  .\release.ps1 -Version v1.2.0 -Message "feat: ..." -CreateRelease -ReleaseTitle "OEE API v1.2.0" -ReleaseNotes "New endpoints"
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Version,                      # e.g. v1.1.0

    [Parameter(Mandatory = $true)]
    [string]$Message,                      # commit message

    [string]$Branch = "main",              # branch to push
    [string[]]$Paths = @("."),             # what to add; default add all
    [switch]$AllowEmptyCommit,             # allow commit even if no changes (creates tag only)
    [switch]$CreateRelease,                # requires GitHub CLI `gh`
    [string]$ReleaseTitle,                 # optional release title
    [string]$ReleaseNotes                  # optional release notes
)

$ErrorActionPreference = "Stop"

function G {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    git @Args
    if ($LASTEXITCODE -ne 0) { throw "git $($Args -join ' ') failed." }
}

# --- 0) Basic validations -----------------------------------------------------
if ($Version -notmatch '^v?\d+\.\d+\.\d+$') {
    throw "Version must look like vX.Y.Z (e.g. v1.2.3). Got: '$Version'"
}

# Ensure we are inside a git repo
G status -s | Out-Null

# Check remote 'origin' exists
$remotes = (git remote) -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
if ("origin" -notin $remotes) { throw "No 'origin' remote found. Run: git remote add origin <URL>" }

# --- 1) Add files -------------------------------------------------------------
Write-Host ">> Adding paths: $($Paths -join ', ')" -ForegroundColor Cyan
G add @Paths

# --- 2) Commit ---------------------------------------------------------------
$needCommit = (git diff --cached --name-only) -ne $null -and (git diff --cached --name-only).Length -gt 0
if ($needCommit) {
    Write-Host ">> Committing changes..." -ForegroundColor Cyan
    G commit -m $Message
} elseif ($AllowEmptyCommit) {
    Write-Host ">> No staged changes. Creating an empty commit (allowed)..." -ForegroundColor Yellow
    G commit --allow-empty -m $Message
} else {
    Write-Host ">> No staged changes. Skipping commit (will still tag current HEAD)..." -ForegroundColor Yellow
}

# --- 3) Push branch first -----------------------------------------------------
Write-Host ">> Pushing branch '$Branch'..." -ForegroundColor Cyan
G push origin $Branch

# --- 4) Safety: avoid duplicate tag locally/remote ---------------------------
$localTagExists = (git tag -l $Version) -ne $null -and (git tag -l $Version).Length -gt 0
if ($localTagExists) {
    throw "Tag '$Version' already exists locally. Delete it first: git tag -d $Version"
}
$remoteTagExists = (git ls-remote --tags origin "refs/tags/$Version") 
if ($remoteTagExists) {
    throw "Tag '$Version' already exists on remote. Delete it first: git push --delete origin $Version"
}

# --- 5) Create annotated tag --------------------------------------------------
Write-Host ">> Creating tag '$Version'..." -ForegroundColor Cyan
G tag -a $Version -m "Release $Version"

# --- 6) Push tag --------------------------------------------------------------
Write-Host ">> Pushing tag '$Version'..." -ForegroundColor Cyan
G push origin $Version

# --- 7) Optional: create GitHub Release via `gh` ------------------------------
if ($CreateRelease) {
    if (Get-Command gh -ErrorAction SilentlyContinue) {
        $title = $(if ($ReleaseTitle) { $ReleaseTitle } else { $Version })
        $notes = $(if ($ReleaseNotes) { $ReleaseNotes } else { $Message })
        Write-Host ">> Creating GitHub Release '$title'..." -ForegroundColor Cyan
        gh release create $Version --title $title --notes $notes
    } else {
        Write-Host ">> 'gh' CLI not found. Skipping GitHub Release." -ForegroundColor Yellow
    }
}

Write-Host "`n✅ Done. Branch and tag have been pushed successfully." -ForegroundColor Green
