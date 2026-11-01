param(
    [Parameter(Mandatory=$true)][string]$DocumentDirectory,
    [Parameter(Mandatory=$true)][string]$RenderDirectory
)
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $RenderDirectory) { throw 'Use a new render directory to retain earlier page reviews.' }
New-Item -ItemType Directory -Path $RenderDirectory | Out-Null
$word=$null
$receipts=@()
try {
    $word=New-Object -ComObject Word.Application
    $word.Visible=$false
    $word.DisplayAlerts=0
    $word.AutomationSecurity=3
    foreach ($name in @('Main_manuscript','Supplementary_information')) {
        $source=Join-Path $DocumentDirectory ($name+'.docx')
        $before=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
        $pdf=Join-Path $RenderDirectory ($name+'.pdf')
        $document=$null
        try {
            $document=$word.Documents.Open($source,$false,$true,$false)
            $document.Repaginate()
            $pages=$document.ComputeStatistics(2)
            $document.ExportAsFixedFormat($pdf,17)
            $receipts+=@{document=$name;renderer='Microsoft Word';version=$word.Version;pages=$pages;source_sha256=$before;pdf=$pdf}
        } finally {
            if ($null -ne $document) { $document.Close(0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
        }
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $before) { throw 'Read-only source changed during rendering.' }
    }
} finally {
    if ($null -ne $word) { $word.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
}
$receipts | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $RenderDirectory 'WORD_RENDER_AUDIT.json') -Encoding utf8
$receipts | ConvertTo-Json -Depth 5
