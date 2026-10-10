# Open the generated report in Word, resolve every field (TOC + PAGEREF +
# page numbers), repaginate and save in place.  Also reports the resulting
# page count and a few sanity figures.
$ErrorActionPreference = 'Stop'
$path = Join-Path $PSScriptRoot 'PROJECT PHASE_I REPORT_RGCET_2025.docx'
$pdf  = [System.IO.Path]::ChangeExtension($path, '.pdf')

if (-not (Test-Path -LiteralPath $path)) { throw "missing: $path" }

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0          # wdAlertsNone

try {
    $doc = $word.Documents.Open($path, $false, $false)

    # build the table of contents, then every other field, twice so that page
    # numbers settle after pagination changes
    foreach ($i in 1..2) {
        foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
        $doc.Fields.Update() | Out-Null
        foreach ($s in $doc.Sections) {
            foreach ($hf in $s.Footers) { $hf.Range.Fields.Update() | Out-Null }
        }
        $doc.Repaginate()
    }

    $pages      = $doc.ComputeStatistics(2)   # wdStatisticPages
    $words      = $doc.ComputeStatistics(0)   # wdStatisticWords
    $tables     = $doc.Tables.Count
    $inlineShp  = $doc.InlineShapes.Count
    $fields     = $doc.Fields.Count
    $tocCount   = $doc.TablesOfContents.Count
    $tocLen     = 0
    if ($tocCount -gt 0) {
        $tocLen = ($doc.TablesOfContents.Item(1).Range.Text -split "`r").Count
    }

    $doc.Save()
    $doc.SaveAs([ref]$pdf, [ref]17)   # wdFormatPDF - layout proof

    Write-Output "pages        : $pages"
    Write-Output "words        : $words"
    Write-Output "tables       : $tables"
    Write-Output "inlineShapes : $inlineShp"
    Write-Output "fields       : $fields"
    Write-Output "TOCs         : $tocCount ($tocLen lines)"

    $doc.Close(0)
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
